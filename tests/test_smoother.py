import numpy as np
import pytest

from src.collision_checker import collision
from src.smoother import shortcut_smooth, smooth_path_length


def make_open_map(size: int = 10) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


@pytest.mark.parametrize("path", [[], [0], [0, 1]])
def test_shortcut_smooth_preserves_paths_with_at_most_two_points(path):
    nodes = np.array([[1, 1], [1, 8]])

    assert shortcut_smooth(path, nodes, make_open_map()) == path


def test_shortcut_smooth_removes_intermediate_waypoints():
    nodes = np.array([[1, 1], [1, 3], [1, 5], [1, 8]])

    smoothed = shortcut_smooth([0, 1, 2, 3], nodes, make_open_map())

    assert smoothed == [0, 3]


def obstacle_route() -> tuple[np.ndarray, np.ndarray, list[int]]:
    obstacle_map = make_open_map(7)
    obstacle_map[3, 3] = 0
    nodes = np.array([[3, 0], [0, 0], [0, 6], [3, 6]])
    return obstacle_map, nodes, [0, 1, 2, 3]


def test_shortcut_smooth_does_not_cross_blocking_obstacle():
    obstacle_map, nodes, path = obstacle_route()

    smoothed = shortcut_smooth(path, nodes, obstacle_map)

    assert collision(nodes[path[0]], nodes[path[-1]], obstacle_map) is True
    assert smoothed != [path[0], path[-1]]


def test_shortcut_smooth_preserves_start_and_goal():
    obstacle_map, nodes, path = obstacle_route()

    smoothed = shortcut_smooth(path, nodes, obstacle_map)

    assert smoothed[0] == path[0]
    assert smoothed[-1] == path[-1]


def test_shortcut_smooth_returns_only_collision_free_segments():
    obstacle_map, nodes, path = obstacle_route()

    smoothed = shortcut_smooth(path, nodes, obstacle_map)

    for start_idx, goal_idx in zip(smoothed, smoothed[1:]):
        assert collision(nodes[start_idx], nodes[goal_idx], obstacle_map) is False


def test_shortcut_smooth_does_not_increase_path_length():
    obstacle_map, nodes, path = obstacle_route()
    original_length = smooth_path_length(nodes[path])

    smoothed = shortcut_smooth(path, nodes, obstacle_map)
    smoothed_length = smooth_path_length(nodes[smoothed])

    assert smoothed_length <= original_length + 1e-9
