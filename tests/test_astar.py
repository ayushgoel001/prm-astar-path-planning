"""
tests/test_astar.py — Unit tests for the A* search implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.astar import astar, heuristic, path_length


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_linear_graph(n: int) -> tuple[dict, np.ndarray]:
    """Build a simple chain graph: 0 → 1 → 2 → ... → n-1."""
    graph = {i: [i + 1] for i in range(n - 1)}
    graph[n - 1] = []
    nodes = np.array([[float(i), 0.0] for i in range(n)])
    return graph, nodes


def make_triangle_graph() -> tuple[dict, np.ndarray]:
    """Triangle: 0-1-2-0. Shortest 0→2 is direct edge (length √2)."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    return graph, nodes


# ── Heuristic tests ───────────────────────────────────────────────────────────

def test_heuristic_zero_same_point():
    p = np.array([3.0, 4.0])
    assert heuristic(p, p) == pytest.approx(0.0)


def test_heuristic_known_distance():
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])
    assert heuristic(p1, p2) == pytest.approx(5.0)


# ── A* correctness tests ──────────────────────────────────────────────────────

def test_astar_linear_path():
    """A* should find path 0→1→2→3→4 on a chain graph."""
    graph, nodes = make_linear_graph(5)
    path, _ = astar(graph, nodes, start=0, goal=4)
    assert path == [0, 1, 2, 3, 4]


def test_astar_start_equals_goal():
    """When start == goal, path should be [start]."""
    graph, nodes = make_linear_graph(5)
    path, _ = astar(graph, nodes, start=2, goal=2)
    assert path == [2]


def test_astar_no_path_returns_empty():
    """Disconnected graph should return empty path."""
    # Two isolated nodes
    graph = {0: [], 1: []}
    nodes = np.array([[0.0, 0.0], [10.0, 10.0]])
    path, _ = astar(graph, nodes, start=0, goal=1)
    assert path == []


def test_astar_explored_count_positive():
    """A* should explore at least one node."""
    graph, nodes = make_linear_graph(5)
    _, explored = astar(graph, nodes, start=0, goal=4)
    assert explored >= 1


def test_astar_triangle_shortest():
    """A* should prefer the direct edge (0→2, cost √2) over the 2-step path (0→1→2)."""
    graph, nodes = make_triangle_graph()
    path, _ = astar(graph, nodes, start=0, goal=2)
    assert path == [0, 2]


def test_astar_path_starts_and_ends_correctly():
    """Path must begin at start and end at goal."""
    graph, nodes = make_linear_graph(10)
    path, _ = astar(graph, nodes, start=0, goal=9)
    assert path[0] == 0
    assert path[-1] == 9


# ── path_length tests ─────────────────────────────────────────────────────────

def test_path_length_zero_for_single_node():
    nodes = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert path_length([0], nodes) == pytest.approx(0.0)


def test_path_length_known_value():
    nodes = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert path_length([0, 1], nodes) == pytest.approx(5.0)


def test_path_length_three_nodes():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    # 0→1 = 1.0, 1→2 = 1.0, total = 2.0
    assert path_length([0, 1, 2], nodes) == pytest.approx(2.0)
