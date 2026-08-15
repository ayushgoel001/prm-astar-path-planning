import numpy as np
import pytest

from src.map_loader import apply_clearance
from src.utils import PlanningStats, validate_planning_point


def test_validate_planning_point_accepts_free_point():
    obstacle_map = np.full((5, 5), 255, dtype=np.uint8)

    point = validate_planning_point(
        np.array([2, 3]), obstacle_map, obstacle_map, name="start"
    )

    assert np.array_equal(point, np.array([2, 3]))


def test_validate_planning_point_rejects_obstacle_point():
    obstacle_map = np.full((5, 5), 255, dtype=np.uint8)
    obstacle_map[2, 2] = 0

    with pytest.raises(ValueError, match="inside an obstacle"):
        validate_planning_point(
            np.array([2, 2]), obstacle_map, obstacle_map, name="goal"
        )


def test_validate_planning_point_rejects_out_of_bounds_point():
    obstacle_map = np.full((5, 5), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="out of bounds"):
        validate_planning_point(
            np.array([5, 1]), obstacle_map, obstacle_map, name="start"
        )


def test_validate_planning_point_rejects_point_removed_by_clearance():
    obstacle_map = np.full((5, 5), 255, dtype=np.uint8)
    obstacle_map[2, 2] = 0
    safe_map = apply_clearance(obstacle_map, clearance=1)

    with pytest.raises(ValueError, match="too close to an obstacle"):
        validate_planning_point(
            np.array([1, 1]),
            obstacle_map,
            safe_map,
            clearance=1,
            name="start",
        )


def test_validate_planning_point_accepts_free_boundary_point():
    obstacle_map = np.full((5, 5), 255, dtype=np.uint8)

    point = validate_planning_point(
        np.array([0, 4]), obstacle_map, obstacle_map, name="goal"
    )

    assert np.array_equal(point, np.array([0, 4]))


def test_planning_stats_separates_samples_from_graph_nodes():
    stats = PlanningStats(num_sampled_nodes=500, num_graph_nodes=502)

    values = stats.to_dict()

    assert values["sampled_nodes"] == 500
    assert values["graph_nodes"] == 502
    assert "PRM sampled nodes" in stats.report()
    assert "Roadmap graph nodes" in stats.report()
