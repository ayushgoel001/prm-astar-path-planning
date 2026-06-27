"""
tests/test_collision.py — Unit tests for the collision checker.
"""

import numpy as np
import pytest
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

def test_no_collision_open_map():
    m = make_open_map()
    n1 = np.array([10, 10])
    n2 = np.array([90, 90])
    assert collision(n1, n2, m) is False


def test_collision_with_wall():
    m = make_map_with_wall(wall_col=50)
    n1 = np.array([50, 10])
    n2 = np.array([50, 90])
    assert collision(n1, n2, m) is True


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


def test_collision_symmetric():
    """Collision check should be symmetric: collision(a, b) == collision(b, a)."""
    m = make_map_with_wall(wall_col=50)
    n1 = np.array([50, 10])
    n2 = np.array([50, 90])
    assert collision(n1, n2, m) == collision(n2, n1, m)
