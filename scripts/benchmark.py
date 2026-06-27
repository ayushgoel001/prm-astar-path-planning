"""
benchmark.py - Evaluate the PRM + A* planner across fixed test scenarios.

The script runs the full planning pipeline on each configured map, records
planning statistics, prints a compact summary table, saves a CSV file, and
exports result visualizations.

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.astar import astar, path_length
from src.map_loader import apply_clearance, load_map
from src.prm import build_roadmap, sample_free_space
from src.utils import PlanningStats, set_seed, validate_planning_point
from src.visualizer import visualize


SEED = 42
NUM_NODES = 500
K_NEIGHBORS = 20
CLEARANCE = 3

TEST_CASES: list[dict] = [
    {
        "name": "Maze 1 Easy",
        "map": "examples/maze_1.png",
        "start": [60, 40],
        "goal": [280, 40],
    },
    {
        "name": "Maze 1 Hard",
        "map": "examples/maze_1.png",
        "start": [5, 235],
        "goal": [350, 450],
    },
    {
        "name": "Maze 2 Grid",
        "map": "examples/maze_2.png",
        "start": [30, 30],
        "goal": [460, 460],
    },
    {
        "name": "Maze 3 Obstacles",
        "map": "examples/maze_3.png",
        "start": [30, 30],
        "goal": [460, 440],
    },
]


def sanitize_filename(name: str) -> str:
    """Convert a scenario name into a safe output filename."""
    safe_name = (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "")
        .strip("_")
    )

    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    return safe_name


def run_case(
    case: dict,
    root: Path,
) -> tuple[
    PlanningStats,
    list[int],
    np.ndarray,
    dict[int, list[int]],
    np.ndarray,
    int,
    int,
]:
    """Run the complete PRM + A* pipeline for one benchmark case."""
    map_path = root / case["map"]

    obstacle_map = load_map(map_path)
    safe_map = apply_clearance(obstacle_map, CLEARANCE)

    start_coord = np.array(case["start"], dtype=int)
    goal_coord = np.array(case["goal"], dtype=int)

    start_coord = validate_planning_point(
        start_coord,
        obstacle_map,
        safe_map,
        clearance=CLEARANCE,
        name="start",
    )

    goal_coord = validate_planning_point(
        goal_coord,
        obstacle_map,
        safe_map,
        clearance=CLEARANCE,
        name="goal",
    )

    set_seed(SEED)

    build_start = time.perf_counter()

    nodes = sample_free_space(safe_map, NUM_NODES, seed=SEED)
    start_idx = len(nodes)
    goal_idx = len(nodes) + 1
    nodes = np.vstack([nodes, start_coord, goal_coord])

    edges, total_edges = build_roadmap(
        nodes,
        safe_map,
        K_NEIGHBORS,
        anchor_indices=[start_idx, goal_idx],
    )

    build_time = time.perf_counter() - build_start

    search_start = time.perf_counter()
    path, explored = astar(edges, nodes, start_idx, goal_idx)
    search_time = time.perf_counter() - search_start

    stats = PlanningStats(
        map_name=case["name"],
        num_nodes=len(nodes),
        num_edges=total_edges,
        astar_explored=explored,
        path_found=len(path) > 0,
        path_length_px=path_length(path, nodes) if path else 0.0,
        build_time_s=build_time,
        search_time_s=search_time,
    )

    return stats, path, nodes, edges, obstacle_map, start_idx, goal_idx


def print_results_table(all_stats: list[PlanningStats]) -> None:
    """Print benchmark results in a compact terminal-friendly format."""
    print()
    print(
        f"{'Scenario':<22}"
        f"{'Nodes':>7}"
        f"{'Edges':>8}"
        f"{'Explored':>10}"
        f"{'Found':>8}"
        f"{'Length':>10}"
        f"{'Build':>10}"
        f"{'Search':>10}"
        f"{'Total':>10}"
    )
    print("-" * 95)

    for stats in all_stats:
        print(
            f"{stats.map_name[:22]:<22}"
            f"{stats.num_nodes:>7}"
            f"{stats.num_edges:>8}"
            f"{stats.astar_explored:>10}"
            f"{'Yes' if stats.path_found else 'No':>8}"
            f"{stats.path_length_px:>10.1f}"
            f"{stats.build_time_s * 1000:>10.1f}"
            f"{stats.search_time_s * 1000:>10.1f}"
            f"{stats.total_time_s * 1000:>10.1f}"
        )


def save_csv(all_stats: list[PlanningStats], output_path: Path) -> None:
    """Save benchmark statistics as a CSV file."""
    if not all_stats:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_stats[0].to_dict().keys())

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for stats in all_stats:
            writer.writerow(stats.to_dict())

    print(f"\nBenchmark CSV saved: {output_path}")


def main() -> None:
    """Run all configured benchmark cases."""
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("PRM + A* Benchmark")
    print(f"nodes={NUM_NODES}  k={K_NEIGHBORS}  clearance={CLEARANCE}  seed={SEED}")
    print("=" * 60)

    all_stats: list[PlanningStats] = []
    missing_maps: list[str] = []

    for index, case in enumerate(TEST_CASES, start=1):
        map_path = root / case["map"]

        if not map_path.exists():
            print(f"\n[{index}/{len(TEST_CASES)}] {case['name']}")
            print(f"SKIP: Map not found: {map_path}")
            missing_maps.append(case["map"])
            continue

        print(f"\n[{index}/{len(TEST_CASES)}] {case['name']}")

        try:
            stats, path, nodes, edges, obstacle_map, start_idx, goal_idx = run_case(
                case,
                root,
            )

            all_stats.append(stats)

            status = "Path found" if stats.path_found else "No path"
            print(
                f"{status}: edges={stats.num_edges}, "
                f"explored={stats.astar_explored}, "
                f"length={stats.path_length_px:.0f}px, "
                f"time={stats.total_time_s * 1000:.0f}ms"
            )

            output_path = output_dir / f"{sanitize_filename(case['name'])}.png"
            visualize(
                obstacle_map,
                nodes,
                edges,
                path,
                start_idx,
                goal_idx,
                save_path=output_path,
                show=False,
                title_prefix=f"{case['name']} | ",
            )

        except ValueError as error:
            print(f"SKIP: {error}")

        except Exception as error:
            print(f"ERROR: {error}")

    if missing_maps:
        print(
            "\nSome maps were not found. Regenerate them with:\n"
            "python scripts/generate_maps.py"
        )

    if all_stats:
        print("\n" + "-" * 60)
        print("Benchmark Results")
        print("-" * 60)
        print_results_table(all_stats)
        save_csv(all_stats, output_dir / "benchmark_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()