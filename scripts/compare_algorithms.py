"""
compare_algorithms.py - Compare PRM + A* and RRT on fixed map scenarios.

The script runs fresh PRM and RRT attempts with matched seeds on the same
start-goal queries, reports aggregate metrics, saves a CSV file, and exports
comparison figures. Runtime statistics include every attempt; path-length
statistics include successful runs only.

Usage:
    python scripts/compare_algorithms.py
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.astar import astar, path_length
from src.collision_checker import collision
from src.map_loader import apply_clearance, load_map
from src.prm import build_roadmap, sample_free_space
from src.rrt import RRTPlanner, rrt_path_length
from src.utils import set_seed, validate_planning_point


SEED = 42
CLEARANCE = 3
NUM_RUNS = 10
SEEDS = [SEED + run_index for run_index in range(NUM_RUNS)]

PRM_SAMPLES = 500
PRM_NEIGHBORS = 20

RRT_MAX_ITER = 8000
RRT_STEP_SIZE = 25.0
RRT_GOAL_BIAS = 0.15
RRT_GOAL_RADIUS = 20.0

TEST_CASES: list[dict] = [
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


@dataclass
class PlannerRun:
    """Single-run output for a planner."""

    seed: int
    found: bool
    length_px: float
    time_s: float
    path: list
    nodes: np.ndarray | None
    prm_samples: int | None = None
    graph_vertices: int | None = None
    graph_edges: int | None = None
    astar_expanded_nodes: int | None = None
    tree_nodes_at_termination: int | None = None


@dataclass
class SummaryResult:
    """Aggregated result for one planner on one scenario."""

    scenario: str
    algorithm: str
    attempted_runs: int
    successful_runs: int
    success_rate_percent: float
    mean_runtime_ms: float
    runtime_std_ms: float
    mean_path_length_px: float | None
    path_length_std_px: float | None
    prm_samples_per_run: int | None
    mean_graph_vertices: float | None
    mean_graph_edges: float | None
    mean_tree_nodes_at_termination: float | None


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


def validate_case(case: dict, obstacle_map: np.ndarray, safe_map: np.ndarray) -> dict:
    """Validate start and goal coordinates for a benchmark case."""
    start = validate_planning_point(
        np.array(case["start"], dtype=int),
        obstacle_map,
        safe_map,
        clearance=CLEARANCE,
        name="start",
    )

    goal = validate_planning_point(
        np.array(case["goal"], dtype=int),
        obstacle_map,
        safe_map,
        clearance=CLEARANCE,
        name="goal",
    )

    return {
        **case,
        "start": start.tolist(),
        "goal": goal.tolist(),
    }


def validate_run_path(
    run: PlannerRun,
    path_coordinates: np.ndarray,
    safe_map: np.ndarray,
) -> None:
    """Reject invalid run metrics before they enter aggregate results."""
    if run.found != bool(run.path):
        raise RuntimeError("Planner success flag does not match its returned path")
    if not np.isfinite(run.time_s) or run.time_s < 0:
        raise RuntimeError("Planner run has an invalid runtime")

    if not run.found:
        if run.length_px != 0.0:
            raise RuntimeError("Failed run must not report a path length")
        return

    if not np.isfinite(run.length_px) or run.length_px < 0:
        raise RuntimeError("Successful run has an invalid path length")

    for start, goal in zip(path_coordinates, path_coordinates[1:]):
        if collision(start, goal, safe_map):
            raise RuntimeError("Planner returned a path containing a collision")


def run_prm(case: dict, safe_map: np.ndarray, seed: int) -> PlannerRun:
    """Run PRM + A* once on a benchmark case."""
    start = np.array(case["start"], dtype=int)
    goal = np.array(case["goal"], dtype=int)

    set_seed(seed)

    start_time = time.perf_counter()

    nodes = sample_free_space(safe_map, PRM_SAMPLES, seed=seed)
    sampled_node_count = len(nodes)
    start_idx = len(nodes)
    goal_idx = len(nodes) + 1
    nodes = np.vstack([nodes, start, goal])

    edges, edge_count = build_roadmap(
        nodes,
        safe_map,
        PRM_NEIGHBORS,
        anchor_indices=[start_idx, goal_idx],
    )

    path, expanded_nodes = astar(edges, nodes, start_idx, goal_idx)
    elapsed = time.perf_counter() - start_time

    found = len(path) > 0
    length = path_length(path, nodes) if found else 0.0

    if expanded_nodes > len(nodes):
        raise RuntimeError("A* expanded-node count exceeds roadmap graph vertices")

    run = PlannerRun(
        seed=seed,
        found=found,
        length_px=length,
        time_s=elapsed,
        path=path,
        nodes=nodes,
        prm_samples=sampled_node_count,
        graph_vertices=len(nodes),
        graph_edges=edge_count,
        astar_expanded_nodes=expanded_nodes,
    )
    path_coordinates = nodes[path] if path else np.empty((0, 2))
    validate_run_path(run, path_coordinates, safe_map)
    return run


def run_rrt(case: dict, safe_map: np.ndarray, seed: int) -> PlannerRun:
    """Run RRT once on a benchmark case."""
    start = np.array(case["start"], dtype=float)
    goal = np.array(case["goal"], dtype=float)

    start_time = time.perf_counter()
    planner = RRTPlanner(
        max_iter=RRT_MAX_ITER,
        step_size=RRT_STEP_SIZE,
        goal_bias=RRT_GOAL_BIAS,
        goal_radius=RRT_GOAL_RADIUS,
        seed=seed,
    )

    path, _ = planner.plan(start, goal, safe_map)
    elapsed = time.perf_counter() - start_time

    found = len(path) > 0
    length = rrt_path_length(path) if found else 0.0
    tree_nodes = planner.nodes

    run = PlannerRun(
        seed=seed,
        found=found,
        length_px=length,
        time_s=elapsed,
        path=path,
        nodes=tree_nodes,
        tree_nodes_at_termination=len(tree_nodes),
    )
    validate_run_path(run, np.array(path), safe_map)
    return run


def summarize_runs(
    scenario: str,
    algorithm: str,
    runs: list[PlannerRun],
) -> SummaryResult:
    """Aggregate repeated planner runs into one summary row."""
    if not runs:
        raise ValueError("At least one planner run is required")

    successful_runs = [run for run in runs if run.found]

    attempted_runs = len(runs)
    successful_count = len(successful_runs)
    success_rate_percent = successful_count / attempted_runs * 100.0
    runtimes_ms = np.array([run.time_s * 1000 for run in runs], dtype=float)
    successful_lengths = np.array(
        [run.length_px for run in successful_runs],
        dtype=float,
    )

    if not 0.0 <= success_rate_percent <= 100.0:
        raise RuntimeError("Success rate is outside the range 0 to 100 percent")
    if not np.all(np.isfinite(runtimes_ms)) or np.any(runtimes_ms < 0):
        raise RuntimeError("Runtime summary contains invalid values")
    if successful_count and (
        not np.all(np.isfinite(successful_lengths))
        or np.any(successful_lengths < 0)
    ):
        raise RuntimeError("Successful-path summary contains invalid lengths")

    prm_samples_per_run: int | None = None
    mean_graph_vertices: float | None = None
    mean_graph_edges: float | None = None
    mean_tree_nodes: float | None = None

    if algorithm == "PRM + A*":
        if any(
            run.prm_samples is None
            or run.graph_vertices is None
            or run.graph_edges is None
            or run.astar_expanded_nodes is None
            for run in runs
        ):
            raise RuntimeError("PRM run is missing roadmap metrics")
        sample_counts = {run.prm_samples for run in runs}
        if len(sample_counts) != 1:
            raise RuntimeError("PRM runs used different sample counts")
        prm_samples_per_run = int(sample_counts.pop())
        mean_graph_vertices = float(
            np.mean([run.graph_vertices for run in runs])
        )
        mean_graph_edges = float(np.mean([run.graph_edges for run in runs]))
    elif algorithm == "RRT":
        if any(run.tree_nodes_at_termination is None for run in runs):
            raise RuntimeError("RRT run is missing tree-node metrics")
        mean_tree_nodes = float(
            np.mean([run.tree_nodes_at_termination for run in runs])
        )
    else:
        raise ValueError(f"Unknown algorithm label: {algorithm}")

    return SummaryResult(
        scenario=scenario,
        algorithm=algorithm,
        attempted_runs=attempted_runs,
        successful_runs=successful_count,
        success_rate_percent=success_rate_percent,
        mean_runtime_ms=float(np.mean(runtimes_ms)),
        runtime_std_ms=float(np.std(runtimes_ms)),
        mean_path_length_px=(
            float(np.mean(successful_lengths)) if successful_count else None
        ),
        path_length_std_px=(
            float(np.std(successful_lengths)) if successful_count else None
        ),
        prm_samples_per_run=prm_samples_per_run,
        mean_graph_vertices=mean_graph_vertices,
        mean_graph_edges=mean_graph_edges,
        mean_tree_nodes_at_termination=mean_tree_nodes,
    )


def select_representative_run(runs: list[PlannerRun]) -> PlannerRun | None:
    """Select the first successful run for visualization, or the final failed run."""
    for run in runs:
        if run.found:
            return run

    return runs[-1] if runs else None


def save_comparison_figure(
    case: dict,
    obstacle_map: np.ndarray,
    safe_map: np.ndarray,
    prm_run: PlannerRun,
    rrt_run: PlannerRun | None,
    output_path: Path,
) -> None:
    """Save a two-panel figure comparing PRM + A* and RRT."""
    start = np.array(case["start"], dtype=int)
    goal = np.array(case["goal"], dtype=int)

    if prm_run.nodes is None:
        raise ValueError("PRM visualization requires roadmap nodes")
    prm_nodes = prm_run.nodes
    start_idx = len(prm_nodes) - 2
    goal_idx = len(prm_nodes) - 1

    edges, _ = build_roadmap(
        prm_nodes,
        safe_map,
        PRM_NEIGHBORS,
        anchor_indices=[start_idx, goal_idx],
    )

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    for axis in axes:
        axis.imshow(obstacle_map, cmap="gray", origin="upper")
        axis.axis("off")

    prm_segments = [
        [[prm_nodes[i, 1], prm_nodes[i, 0]], [prm_nodes[j, 1], prm_nodes[j, 0]]]
        for i, neighbors in edges.items()
        for j in neighbors
        if i < j
    ]

    if prm_segments:
        axes[0].add_collection(
            LineCollection(
                prm_segments,
                colors="tab:blue",
                linewidths=0.35,
                alpha=0.45,
            )
        )

    axes[0].scatter(
        prm_nodes[:, 1],
        prm_nodes[:, 0],
        c="tab:red",
        s=5,
        alpha=0.75,
        zorder=3,
        label="PRM nodes",
    )

    if prm_run.path:
        path_coords = prm_nodes[prm_run.path]
        axes[0].plot(
            path_coords[:, 1],
            path_coords[:, 0],
            color="tab:green",
            linewidth=2.5,
            zorder=6,
            label="Path",
        )

    axes[0].scatter(start[1], start[0], c="tab:cyan", s=140, marker="*", zorder=8)
    axes[0].scatter(goal[1], goal[0], c="tab:orange", s=140, marker="*", zorder=8)
    axes[0].set_title(
        f"PRM + A* | length={prm_run.length_px:.0f}px"
        if prm_run.found
        else "PRM + A* | no path"
    )

    if rrt_run is not None and rrt_run.nodes is not None and len(rrt_run.nodes) > 0:
        axes[1].scatter(
            rrt_run.nodes[:, 1],
            rrt_run.nodes[:, 0],
            c="tab:blue",
            s=4,
            alpha=0.45,
            zorder=2,
            label="RRT tree",
        )

    if rrt_run is not None and rrt_run.path:
        rrt_path = np.array(rrt_run.path)
        axes[1].plot(
            rrt_path[:, 1],
            rrt_path[:, 0],
            color="tab:green",
            linewidth=2.5,
            zorder=6,
            label="Path",
        )

    axes[1].scatter(start[1], start[0], c="tab:cyan", s=140, marker="*", zorder=8)
    axes[1].scatter(goal[1], goal[0], c="tab:orange", s=140, marker="*", zorder=8)

    if rrt_run is not None and rrt_run.found:
        axes[1].set_title(f"RRT | length={rrt_run.length_px:.0f}px")
    else:
        axes[1].set_title("RRT | no path")

    figure.suptitle(f"Algorithm Comparison: {case['name']}", fontsize=13)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved figure: {output_path}")


def print_results_table(results: list[SummaryResult]) -> None:
    """Print comparison results in a compact terminal-friendly table."""
    print()
    print(
        f"{'Scenario':<20}"
        f"{'Algorithm':<10}"
        f"{'Runs':>6}"
        f"{'Success':>10}"
        f"{'Rate':>8}"
        f"{'Mean Len':>11}"
        f"{'Len SD':>10}"
        f"{'Mean ms':>10}"
        f"{'Time SD':>10}  "
        "Structure"
    )
    print("-" * 124)

    for result in results:
        success_count = f"{result.successful_runs}/{result.attempted_runs}"
        mean_length = (
            f"{result.mean_path_length_px:.1f}"
            if result.mean_path_length_px is not None
            else "n/a"
        )
        length_std = (
            f"{result.path_length_std_px:.1f}"
            if result.path_length_std_px is not None
            else "n/a"
        )
        if result.algorithm == "PRM + A*":
            structure = (
                f"{result.mean_graph_vertices:.1f} graph vertices; "
                f"{result.mean_graph_edges:.1f} graph edges"
            )
        else:
            structure = (
                f"{result.mean_tree_nodes_at_termination:.1f} "
                "tree nodes at termination"
            )

        print(
            f"{result.scenario[:20]:<20}"
            f"{result.algorithm:<10}"
            f"{result.attempted_runs:>6}"
            f"{success_count:>10}"
            f"{result.success_rate_percent:>7.1f}%"
            f"{mean_length:>11}"
            f"{length_std:>10}"
            f"{result.mean_runtime_ms:>10.1f}"
            f"{result.runtime_std_ms:>10.1f}  "
            f"{structure}"
        )


def save_csv(results: list[SummaryResult], output_path: Path) -> None:
    """Save comparison results as a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scenario",
                "algorithm",
                "attempted_runs",
                "successful_runs",
                "success_rate_percent",
                "mean_runtime_ms_all_runs",
                "runtime_std_ms_all_runs",
                "mean_path_length_px_successful_runs",
                "path_length_std_px_successful_runs",
                "prm_samples_per_run",
                "mean_prm_graph_vertices",
                "mean_prm_graph_edges",
                "mean_rrt_tree_nodes_at_termination",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "algorithm": result.algorithm,
                    "attempted_runs": result.attempted_runs,
                    "successful_runs": result.successful_runs,
                    "success_rate_percent": round(
                        result.success_rate_percent, 1
                    ),
                    "mean_runtime_ms_all_runs": round(
                        result.mean_runtime_ms, 1
                    ),
                    "runtime_std_ms_all_runs": round(
                        result.runtime_std_ms, 1
                    ),
                    "mean_path_length_px_successful_runs": (
                        round(result.mean_path_length_px, 1)
                        if result.mean_path_length_px is not None
                        else ""
                    ),
                    "path_length_std_px_successful_runs": (
                        round(result.path_length_std_px, 1)
                        if result.path_length_std_px is not None
                        else ""
                    ),
                    "prm_samples_per_run": result.prm_samples_per_run or "",
                    "mean_prm_graph_vertices": (
                        round(result.mean_graph_vertices, 1)
                        if result.mean_graph_vertices is not None
                        else ""
                    ),
                    "mean_prm_graph_edges": (
                        round(result.mean_graph_edges, 1)
                        if result.mean_graph_edges is not None
                        else ""
                    ),
                    "mean_rrt_tree_nodes_at_termination": (
                        round(result.mean_tree_nodes_at_termination, 1)
                        if result.mean_tree_nodes_at_termination is not None
                        else ""
                    ),
                }
            )

    print(f"\nComparison CSV saved: {output_path}")


def main() -> None:
    """Run PRM + A* and RRT comparison across all configured scenarios."""
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print("PRM + A* vs RRT Comparison")
    print(
        f"runs_per_algorithm={NUM_RUNS}  seeds={SEEDS}  clearance={CLEARANCE}\n"
        f"PRM: samples={PRM_SAMPLES}, k={PRM_NEIGHBORS}, "
        "anchor_neighbor_attempts=3*k\n"
        f"RRT: max_iter={RRT_MAX_ITER}, step_size={RRT_STEP_SIZE}, "
        f"goal_bias={RRT_GOAL_BIAS}, goal_radius={RRT_GOAL_RADIUS}"
    )
    print("=" * 64)

    all_results: list[SummaryResult] = []

    for case in TEST_CASES:
        map_path = root / case["map"]

        if not map_path.exists():
            print(f"\nSKIP: Map not found: {map_path}")
            continue

        obstacle_map = load_map(map_path)
        safe_map = apply_clearance(obstacle_map, CLEARANCE)

        try:
            case = validate_case(case, obstacle_map, safe_map)
        except ValueError as error:
            print(f"\nSKIP: {case['name']} - {error}")
            continue

        print(f"\nScenario: {case['name']}")

        prm_runs: list[PlannerRun] = []
        rrt_runs: list[PlannerRun] = []

        for run_number, seed in enumerate(SEEDS, start=1):
            prm_run = run_prm(case, safe_map, seed)
            rrt_run = run_rrt(case, safe_map, seed)
            prm_runs.append(prm_run)
            rrt_runs.append(rrt_run)

            print(
                f"  run {run_number:>2}/{NUM_RUNS}, seed={seed}: "
                f"PRM={'ok' if prm_run.found else 'fail'} "
                f"({prm_run.time_s * 1000:.1f} ms), "
                f"RRT={'ok' if rrt_run.found else 'fail'} "
                f"({rrt_run.time_s * 1000:.1f} ms)"
            )

        if len(prm_runs) != NUM_RUNS or len(rrt_runs) != NUM_RUNS:
            raise RuntimeError("Planner run count does not match NUM_RUNS")
        if [run.seed for run in prm_runs] != [run.seed for run in rrt_runs]:
            raise RuntimeError("PRM and RRT did not use matched seed lists")

        prm_summary = summarize_runs(case["name"], "PRM + A*", prm_runs)
        all_results.append(prm_summary)
        rrt_summary = summarize_runs(case["name"], "RRT", rrt_runs)
        all_results.append(rrt_summary)

        print(
            f"  PRM summary: {prm_summary.successful_runs}/{NUM_RUNS} successful, "
            f"mean_runtime={prm_summary.mean_runtime_ms:.1f} ms\n"
            f"  RRT summary: {rrt_summary.successful_runs}/{NUM_RUNS} successful, "
            f"mean_runtime={rrt_summary.mean_runtime_ms:.1f} ms"
        )

        representative_prm_run = select_representative_run(prm_runs)
        representative_rrt_run = select_representative_run(rrt_runs)
        figure_path = output_dir / f"compare_{sanitize_filename(case['name'])}.png"

        if representative_prm_run is None:
            raise RuntimeError("PRM comparison produced no runs to visualize")

        save_comparison_figure(
            case,
            obstacle_map,
            safe_map,
            representative_prm_run,
            representative_rrt_run,
            figure_path,
        )

    if all_results:
        print("\n" + "-" * 64)
        print("Comparison Results")
        print("-" * 64)
        print_results_table(all_results)
        save_csv(all_results, output_dir / "comparison_results.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
