"""
astar.py — A* search on the PRM graph.
"""

from __future__ import annotations
import heapq
import numpy as np


def heuristic(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Euclidean distance between two [row, col] positions."""
    return float(np.linalg.norm(pos1 - pos2))


def astar(
    graph: dict[int, list[int]],
    nodes: np.ndarray,
    start: int,
    goal: int,
) -> tuple[list[int], int]:
    """
    A* search on the PRM graph to find the shortest path from start to goal.

    Uses a binary min-heap (heapq) for the priority queue. Heap entries carry
    their g-cost so stale entries can be discarded without re-expanding nodes.

    Args:
        graph: Adjacency list — maps node index to list of neighbor indices.
        nodes: Array of shape (N, 2) with node positions.
        start: Index of the start node.
        goal: Index of the goal node.

    Returns:
        Tuple of:
          - path: Ordered list of node indices from start to goal.
                  Empty list if no path exists.
          - expanded_nodes: Number of unique valid graph vertices expanded.
    """
    heap: list[tuple[float, int, float]] = [
        (heuristic(nodes[start], nodes[goal]), start, 0.0)
    ]
    came_from: dict[int, int | None] = {start: None}
    cost_so_far: dict[int, float] = {start: 0.0}
    expanded: set[int] = set()

    while heap:
        _, current, entry_cost = heapq.heappop(heap)

        if entry_cost > cost_so_far.get(current, float("inf")):
            continue
        if current in expanded:
            continue

        expanded.add(current)

        if current == goal:
            break

        for next_node in graph.get(current, []):
            new_cost = cost_so_far[current] + heuristic(nodes[current], nodes[next_node])

            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                priority = new_cost + heuristic(nodes[next_node], nodes[goal])
                heapq.heappush(heap, (priority, next_node, new_cost))
                came_from[next_node] = current

    if goal not in came_from:
        return [], len(expanded)

    path: list[int] = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]  # type: ignore[assignment]
    path.reverse()

    if path[0] != start:
        return [], len(expanded)

    return path, len(expanded)


def path_length(path: list[int], nodes: np.ndarray) -> float:
    """
    Compute the total Euclidean length of a path in pixels.

    Args:
        path: Ordered list of node indices.
        nodes: Array of node positions.

    Returns:
        Total path length in pixels.
    """
    if len(path) < 2:
        return 0.0
    coords = nodes[path]
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
