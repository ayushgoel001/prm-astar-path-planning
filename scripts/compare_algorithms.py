"""
compare_algorithms.py - Compare PRM + A* and RRT on fixed map scenarios.

The script runs both planners on the same start-goal queries, reports aggregate
metrics, saves a CSV file, and exports comparison figures.

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
from src.map_loader import apply_clearance, load_map
from src.prm import build_roadmap, sample_free_space
from src.rrt import RRTPlanner, rrt_path_length
from src.utils import set_seed, validate_planning_point


SEED = 42
CLEARANCE = 3
NUM_RUNS = 5

PRM_NODES = 500
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

    found: bool
    length_px: float
    time_s: float
    size: int
    path: list
    nodes: np.ndarray | None


@dataclass
class SummaryResult:
    """Aggregated result for one planner on one scenario."""

    scenario: str
    algorithm: str
    success_rate: float
    avg_length_px: float
    avg_time_ms: float
    avg_size: int


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


def run_prm(case: dict, safe_map: np.ndarray, seed: int) -> PlannerRun:
    """Run PRM + A* once on a benchmark case."""
    start = np.array(case["start"], dtype=int)
    goal = np.array(case["goal"], dtype=int)

    set_seed(seed)

    start_time = time.perf_counter()

    nodes = sample_free_space(safe_map, PRM_NODES, seed=seed)
    start_idx = len(nodes)
    goal_idx = len(nodes) + 1
    nodes = np.vstack([nodes, start, goal])

    edges, _ = build_roadmap(
        nodes,
        safe_map,
        PRM_NEIGHBORS,
        anchor_indices=[start_idx, goal_idx],
    )

    path, _ = astar(edges, nodes, start_idx, goal_idx)
    elapsed = time.perf_counter() - start_time

    found = len(path) > 0
    length = path_length(path, nodes) if found else 0.0

    return PlannerRun(
        found=found,
        length_px=length,
        time_s=elapsed,
        size=len(nodes),
        path=path,
        nodes=nodes,
    )


def run_rrt(case: dict, safe_map: np.ndarray, seed: int) -> PlannerRun:
    """Run RRT once on a benchmark case."""
    start = np.array(case["start"], dtype=float)
    goal = np.array(case["goal"], dtype=float)

    planner = RRTPlanner(
        max_iter=RRT_MAX_ITER,
        step_size=RRT_STEP_SIZE,
        goal_bias=RRT_GOAL_BIAS,
        goal_radius=RRT_GOAL_RADIUS,
        seed=seed,
    )

    start_time = time.perf_counter()
    path, _ = planner.plan(start, goal, safe_map)
    elapsed = time.perf_counter() - start_time

    found = len(path) > 0
    length = rrt_path_length(path) if found else 0.0
    tree_nodes = planner.nodes

    return PlannerRun(
        found=found,
        length_px=length,
        time_s=elapsed,
        size=len(tree_nodes),
        path=path,
        nodes=tree_nodes,
    )


def summarize_runs(
    scenario: str,
    algorithm: str,
    runs: list[PlannerRun],
) -> SummaryResult:
    """Aggregate repeated planner runs into one summary row."""
    successful_runs = [run for run in runs if run.found]

    success_rate = len(successful_runs) / len(runs) if runs else 0.0
    avg_length = (
        float(np.mean([run.length_px for run in successful_runs]))
        if successful_runs
        else 0.0
    )
    avg_time_ms = float(np.mean([run.time_s for run in runs]) * 1000) if runs else 0.0
    avg_size = int(np.mean([run.size for run in runs])) if runs else 0

    return SummaryResult(
        scenario=scenario,
        algorithm=algorithm,
        success_rate=success_rate,
        avg_length_px=avg_length,
        avg_time_ms=avg_time_ms,
        avg_size=avg_size,
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
        f"{'Algorithm':<12}"
        f"{'Success':>9}"
        f"{'Length':>10}"
        f"{'Time':>10}"
        f"{'Size':>10}"
    )
    print("-" * 71)

    for result in results:
        print(
            f"{result.scenario[:20]:<20}"
            f"{result.algorithm:<12}"
            f"{result.success_rate * 100:>8.0f}%"
            f"{result.avg_length_px:>10.1f}"
            f"{result.avg_time_ms:>10.1f}"
            f"{result.avg_size:>10}"
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
                "success_rate",
                "avg_length_px",
                "avg_time_ms",
                "avg_size",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "scenario": result.scenario,
                    "algorithm": result.algorithm,
                    "success_rate": round(result.success_rate, 3),
                    "avg_length_px": round(result.avg_length_px, 1),
                    "avg_time_ms": round(result.avg_time_ms, 1),
                    "avg_size": result.avg_size,
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
        f"runs={NUM_RUNS}  clearance={CLEARANCE}  seed={SEED}  "
        f"prm_nodes={PRM_NODES}  prm_k={PRM_NEIGHBORS}"
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

        prm_run = run_prm(case, safe_map, SEED)
        prm_summary = summarize_runs(case["name"], "PRM + A*", [prm_run])
        all_results.append(prm_summary)

        print(
            f"PRM + A*: {'found' if prm_run.found else 'not found'}, "
            f"length={prm_run.length_px:.0f}px, "
            f"time={prm_run.time_s * 1000:.0f}ms"
        )

        rrt_runs = [
            run_rrt(case, safe_map, SEED + run_index)
            for run_index in range(NUM_RUNS)
        ]
        rrt_summary = summarize_runs(case["name"], "RRT", rrt_runs)
        all_results.append(rrt_summary)

        print(
            f"RRT: success={rrt_summary.success_rate * 100:.0f}%, "
            f"avg_length={rrt_summary.avg_length_px:.0f}px, "
            f"avg_time={rrt_summary.avg_time_ms:.0f}ms, "
            f"avg_tree_size={rrt_summary.avg_size}"
        )

        representative_rrt_run = select_representative_run(rrt_runs)
        figure_path = output_dir / f"compare_{sanitize_filename(case['name'])}.png"

        save_comparison_figure(
            case,
            obstacle_map,
            safe_map,
            prm_run,
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