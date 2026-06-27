"""
run_planner.py — Command-line interface for the PRM + A* path planner.

Usage
-----
    python scripts/run_planner.py --map examples/maze_1.png \\
        --nodes 500 --neighbors 20 --start "5,235" --goal "350,450" \\
        --seed 42 --clearance 3 --smooth --save outputs/result.png

Run with --help for the full argument list.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.map_loader import load_map, apply_clearance
from src.prm import sample_free_space, build_roadmap
from src.astar import astar, path_length
from src.utils import set_seed, PlanningStats, validate_planning_point
from src.visualizer import visualize
from src.smoother import shortcut_smooth, smooth_path_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PRM + A* path planner for 2D binary maze images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--map", type=str, default="examples/maze_1.png",
        help="Path to the input maze/map image (PNG or JPG).",
    )
    parser.add_argument(
        "--nodes", type=int, default=300,
        help="Number of random nodes to sample in free space.",
    )
    parser.add_argument(
        "--neighbors", type=int, default=15,
        help="Maximum k-nearest neighbors to attempt connecting per node.",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help='Start coordinate as "row,col" (e.g. "5,235"). '
             'If omitted, a safe default is chosen from the map.',
    )
    parser.add_argument(
        "--goal", type=str, default=None,
        help='Goal coordinate as "row,col" (e.g. "350,450"). '
             'If omitted, a safe default is chosen from the map.',
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible node sampling.",
    )
    parser.add_argument(
        "--clearance", type=int, default=5,
        help="Obstacle inflation radius in pixels (models robot footprint).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help='Save the output figure to this path (e.g. "outputs/result.png").',
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Suppress the interactive display window.",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="Apply greedy shortcut smoothing to reduce path waypoints.",
    )
    return parser.parse_args()


def parse_coordinate(coord_str: str) -> np.ndarray:
    """Parse a 'row,col' string into a numpy array."""
    parts = coord_str.strip().split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Coordinate must be 'row,col', got: '{coord_str}'"
        )
    return np.array([int(parts[0]), int(parts[1])])


def default_start_goal(obstacle_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute default start and goal positions from the map.

    Scans the left and right columns for the last free-space pixel
    to produce a cross-map challenge.
    """
    h, w = obstacle_map.shape
    margin = 20

    left_col = margin
    for r in range(h - 1, -1, -1):
        if obstacle_map[r, left_col] == 255:
            start = np.array([r, left_col])
            break
    else:
        start = np.array([h // 2, margin])

    right_col = w - margin - 1
    for r in range(h - 1, -1, -1):
        if obstacle_map[r, right_col] == 255:
            goal = np.array([r, right_col])
            break
    else:
        goal = np.array([h // 2, w - margin - 1])

    return start, goal


def main() -> None:
    args = parse_args()

    print("=" * 56)
    print("  PRM + A* Path Planner")
    print("=" * 56)

    # ── Load map ───────────────────────────────────────────────────────────────
    map_path = Path(args.map)
    print(f"\n[1/5] Loading map: {map_path}")
    obstacle_map = load_map(map_path)
    h, w = obstacle_map.shape
    print(f"      Map size: {w} x {h} px")

    # ── Apply clearance ────────────────────────────────────────────────────────
    print(f"[2/5] Applying obstacle clearance: {args.clearance} px")
    safe_map = apply_clearance(obstacle_map, args.clearance)
    set_seed(args.seed)

    # ── Resolve start / goal ───────────────────────────────────────────────────
    if args.start:
        start_coord = parse_coordinate(args.start)
    else:
        start_coord, _ = default_start_goal(obstacle_map)

    if args.goal:
        goal_coord = parse_coordinate(args.goal)
    else:
        _, goal_coord = default_start_goal(obstacle_map)

    try:
        start_coord = validate_planning_point(
            start_coord,
            obstacle_map,
            safe_map,
            clearance=args.clearance,
            name="start",
        )
        goal_coord = validate_planning_point(
            goal_coord,
            obstacle_map,
            safe_map,
            clearance=args.clearance,
            name="goal",
)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print(
            "Choose coordinates on white free space and away from obstacles, "
            "or reduce --clearance."
        )
        return

    print(f"      Start: {start_coord.tolist()}")
    print(f"      Goal : {goal_coord.tolist()}")

    # ── Build roadmap ──────────────────────────────────────────────────────────
    print(f"\n[3/5] Building PRM roadmap (nodes={args.nodes}, k={args.neighbors}) ...")
    t0 = time.perf_counter()
    nodes = sample_free_space(safe_map, args.nodes, seed=args.seed)

    start_in_graph = len(nodes)
    goal_in_graph  = len(nodes) + 1
    nodes = np.vstack([nodes, start_coord, goal_coord])

    edges, total_edges = build_roadmap(
        nodes, safe_map, args.neighbors,
        anchor_indices=[start_in_graph, goal_in_graph],
    )
    build_time = time.perf_counter() - t0
    print(f"      Roadmap built: {len(nodes)} nodes, {total_edges} edges ({build_time * 1000:.1f} ms)")

    # ── A* search ─────────────────────────────────────────────────────────────
    print("\n[4/5] Running A* search ...")
    start_idx = start_in_graph
    goal_idx  = goal_in_graph

    t1 = time.perf_counter()
    path, explored = astar(edges, nodes, start_idx, goal_idx)
    search_time = time.perf_counter() - t1

    px_length = path_length(path, nodes)

    # ── Path smoothing ─────────────────────────────────────────────────────────
    smoothed_path = path
    smooth_length = px_length
    if path and args.smooth:
        smoothed_path = shortcut_smooth(path, nodes, safe_map)
        smooth_length = smooth_path_length(nodes[smoothed_path])

    stats = PlanningStats(
        map_name=map_path.name,
        num_nodes=len(nodes),
        num_edges=total_edges,
        astar_explored=explored,
        path_found=len(path) > 0,
        path_length_px=smooth_length if args.smooth else px_length,
        build_time_s=build_time,
        search_time_s=search_time,
    )

    print()
    print("-" * 40)
    print(stats.report())
    if path and args.smooth:
        reduction = (1 - len(smoothed_path) / len(path)) * 100
        print(f"Path smoothing    : {len(path)} nodes -> {len(smoothed_path)} nodes "
              f"({reduction:.0f}% reduction)")
        print(f"Smoothed length   : {smooth_length:.1f} px  (raw: {px_length:.1f} px)")
    print("-" * 40)

    if not path:
        print(
            "\n[WARNING] No path found between start and goal.\n"
            "  Try increasing --nodes or --neighbors, "
            "decreasing --clearance, or verifying start/goal coordinates."
        )

    # ── Visualize ──────────────────────────────────────────────────────────────
    print("\n[5/5] Generating visualization ...")
    visualize(
        obstacle_map=obstacle_map,
        nodes=nodes,
        edges=edges,
        path=smoothed_path,
        start_idx=start_idx,
        goal_idx=goal_idx,
        save_path=args.save,
        show=not args.no_show,
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
