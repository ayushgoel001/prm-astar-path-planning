"""
tests/test_collision.py — Unit tests for the collision checker.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collision_checker import collision, is_in_free_space


def make_open_map(h: int = 100, w: int = 100) -> np.ndarray:
    """All-free map (255 everywhere)."""
    return np.full((h, w), 255, dtype=np.uint8)


def make_map_with_wall(h: int = 100, w: int = 100, wall_col: int = 50) -> np.ndarray:
    """Map with a vertical wall at wall_col."""
    m = np.full((h, w), 255, dtype=np.uint8)
    m[:, wall_col] = 0
    return m


# ── is_in_free_space ──────────────────────────────────────────────────────────

def test_free_space_open_map():
    m = make_open_map()
    assert is_in_free_space(np.array([50, 50]), m) == True


def test_obstacle_pixel_returns_false():
    m = make_open_map()
    m[50, 50] = 0
    assert is_in_free_space(np.array([50, 50]), m) == False


def test_out_of_bounds_returns_false():
    m = make_open_map(100, 100)
    assert is_in_free_space(np.array([200, 200]), m) is False
    assert is_in_free_space(np.array([-1, 50]), m) is False


# ── collision ─────────────────────────────────────────────────────────────────

def test_open_horizontal_segment():
    assert collision(np.array([10, 5]), np.array([10, 90]), make_open_map()) is False


def test_open_vertical_segment():
    assert collision(np.array([5, 10]), np.array([90, 10]), make_open_map()) is False


def test_open_diagonal_segment():
    assert collision(np.array([10, 10]), np.array([90, 90]), make_open_map()) is False


def test_collision_with_wall():
    m = make_map_with_wall(wall_col=50)
    n1 = np.array([50, 10])
    n2 = np.array([50, 90])
    assert collision(n1, n2, m) is True


def test_collision_with_one_pixel_obstacle():
    m = make_open_map(10, 10)
    m[4, 4] = 0
    assert collision(np.array([4, 1]), np.array([4, 8]), m) is True


def test_diagonal_corner_touch_is_conservative():
    m = make_open_map(5, 5)
    m[0, 1] = 0
    assert collision(np.array([0, 0]), np.array([2, 2]), m) is True


def test_segment_on_cell_boundary_checks_both_sides():
    m = make_open_map(5, 5)
    m[0, 2] = 0
    assert collision(np.array([0.5, 0]), np.array([0.5, 4]), m) is True


def test_supercover_regression_for_shallow_segment():
    m = make_open_map(6, 8)
    m[1, 1] = 0
    assert collision(np.array([0, 0]), np.array([2, 5]), m) is True


def test_no_collision_parallel_to_wall():
    m = make_map_with_wall(wall_col=50)
    # Both nodes on the same side of the wall
    n1 = np.array([10, 10])
    n2 = np.array([90, 10])
    assert collision(n1, n2, m) is False


def test_collision_same_node_in_obstacle():
    m = make_open_map()
    m[50, 50] = 0
    n = np.array([50, 50])
    assert collision(n, n, m) is True


def test_collision_same_node_in_free_space():
    m = make_open_map()
    n = np.array([50, 50])
    assert collision(n, n, m) is False


def test_collision_with_occupied_endpoint():
    m = make_open_map(10, 10)
    m[8, 8] = 0
    assert collision(np.array([1, 1]), np.array([8, 8]), m) is True


def test_collision_supports_floating_point_endpoints():
    m = make_open_map(6, 8)
    m[1, 1] = 0
    assert collision(np.array([0.25, 0.25]), np.array([2.25, 5.25]), m) is True


def test_half_integer_corner_endpoint_terminates_on_open_map():
    m = make_open_map(5, 5)
    assert collision(np.array([0.0, 1.0]), np.array([0.5, 0.5]), m) is False


def test_half_integer_corner_endpoint_reverse_terminates_on_open_map():
    m = make_open_map(5, 5)
    assert collision(np.array([0.5, 0.5]), np.array([0.0, 1.0]), m) is False


def test_half_integer_corner_endpoint_checks_every_touched_cell():
    m = make_open_map(5, 5)
    m[1, 0] = 0
    start = np.array([0.0, 1.0])
    goal = np.array([0.5, 0.5])
    assert collision(start, goal, m) is True
    assert collision(goal, start, m) is True


def test_single_half_integer_boundary_endpoint_is_symmetric():
    m = make_open_map(5, 5)
    start = np.array([1.0, 1.0])
    goal = np.array([1.5, 2.0])
    assert collision(start, goal, m) is False
    assert collision(goal, start, m) is False


def test_collision_symmetric():
    """Collision check should be symmetric: collision(a, b) == collision(b, a)."""
    m = make_map_with_wall(wall_col=50)
    n1 = np.array([50, 10])
    n2 = np.array([50, 90])
    assert collision(n1, n2, m) == collision(n2, n1, m)

    shallow_map = make_open_map(6, 8)
    shallow_map[1, 1] = 0
    a = np.array([0.25, 0.25])
    b = np.array([2.25, 5.25])
    assert collision(a, b, shallow_map) == collision(b, a, shallow_map)


def test_collision_rejects_out_of_bounds_coordinates():
    m = make_open_map(10, 10)
    assert collision(np.array([-0.01, 5]), np.array([5, 5]), m) is True
    assert collision(np.array([5, 5]), np.array([9.01, 5]), m) is True
