import numpy as np
import pytest

from scripts.compare_algorithms import (
    NUM_RUNS,
    SEED,
    SEEDS,
    PlannerRun,
    summarize_runs,
)


def make_prm_run(found: bool, length: float, runtime: float, edges: int) -> PlannerRun:
    return PlannerRun(
        seed=42,
        found=found,
        length_px=length,
        time_s=runtime,
        path=[0] if found else [],
        nodes=np.zeros((502, 2)),
        prm_samples=500,
        graph_vertices=502,
        graph_edges=edges,
        astar_expanded_nodes=10,
    )


def make_rrt_run(found: bool, length: float, runtime: float, size: int) -> PlannerRun:
    return PlannerRun(
        seed=42,
        found=found,
        length_px=length,
        time_s=runtime,
        path=[np.zeros(2)] if found else [],
        nodes=np.zeros((size, 2)),
        tree_nodes_at_termination=size,
    )


def test_comparison_uses_ten_reproducible_seeds():
    assert NUM_RUNS == 10
    assert SEEDS == [SEED + index for index in range(NUM_RUNS)]


def test_prm_summary_excludes_failed_runs_from_path_length_average():
    runs = [
        make_prm_run(True, 10.0, 0.1, 3000),
        make_prm_run(False, 0.0, 0.2, 3100),
        make_prm_run(True, 20.0, 0.3, 3200),
    ]

    summary = summarize_runs("test", "PRM + A*", runs)

    assert summary.attempted_runs == 3
    assert summary.successful_runs == 2
    assert summary.success_rate_percent == pytest.approx(200 / 3)
    assert summary.mean_runtime_ms == pytest.approx(200.0)
    assert summary.mean_path_length_px == pytest.approx(15.0)
    assert summary.path_length_std_px == pytest.approx(5.0)
    assert summary.prm_samples_per_run == 500
    assert summary.mean_graph_vertices == pytest.approx(502.0)
    assert summary.mean_graph_edges == pytest.approx(3100.0)
    assert summary.mean_tree_nodes_at_termination is None


def test_rrt_summary_labels_tree_nodes_separately():
    runs = [
        make_rrt_run(True, 12.0, 0.05, 20),
        make_rrt_run(False, 0.0, 0.15, 40),
    ]

    summary = summarize_runs("test", "RRT", runs)

    assert summary.attempted_runs == 2
    assert summary.successful_runs == 1
    assert summary.mean_runtime_ms == pytest.approx(100.0)
    assert summary.mean_path_length_px == pytest.approx(12.0)
    assert summary.mean_tree_nodes_at_termination == pytest.approx(30.0)
    assert summary.mean_graph_vertices is None
    assert summary.mean_graph_edges is None


def test_summary_rejects_empty_run_list():
    with pytest.raises(ValueError, match="At least one planner run"):
        summarize_runs("test", "PRM + A*", [])
