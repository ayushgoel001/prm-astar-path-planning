"""
utils.py — Shared utilities: reproducibility, nearest-node lookup, stats.
"""

from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
from scipy.spatial import KDTree


def set_seed(seed: int) -> None:
    """Set random seeds for numpy and Python's random module."""
    np.random.seed(seed)
    random.seed(seed)


def find_nearest_node(nodes: np.ndarray, coordinate: np.ndarray) -> int:
    """
    Find the index of the node in `nodes` closest to `coordinate`.

    Args:
        nodes: Array of shape (N, 2) representing node positions [row, col].
        coordinate: Target point [row, col].

    Returns:
        Index of the nearest node.
    """
    tree = KDTree(nodes)
    _, idx = tree.query(coordinate, k=1)
    return int(idx)

def validate_planning_point(
    point: np.ndarray,
    original_map: np.ndarray,
    safe_map: np.ndarray,
    clearance: int = 0,
    name: str = "point",
) -> np.ndarray:
    """
    Validate that a planning point is inside bounds and lies in safe free space.

    The point is checked against both the original map and the clearance-applied
    map. This prevents start/goal positions from being placed inside obstacles
    or too close to obstacles after clearance is applied.

    Args:
        point: Coordinate as [row, col].
        original_map: Original binary map where 0 = obstacle and 255 = free.
        safe_map: Clearance-applied binary map.
        clearance: Obstacle clearance used while creating safe_map.
        name: Name used in error messages, such as "start" or "goal".

    Returns:
        The validated point as an integer numpy array.

    Raises:
        ValueError: If the point is out of bounds, inside an obstacle, or
        invalid after obstacle clearance.
    """
    point = np.asarray(point, dtype=int)

    if point.shape != (2,):
        raise ValueError(f"{name} must be a coordinate [row, col], got {point.tolist()}")

    h, w = original_map.shape
    r, c = int(point[0]), int(point[1])

    if not (0 <= r < h and 0 <= c < w):
        raise ValueError(f"{name} {point.tolist()} is out of bounds for map size {h}x{w}")

    if original_map[r, c] == 0:
        raise ValueError(f"{name} {point.tolist()} is inside an obstacle")

    if safe_map[r, c] == 0:
        raise ValueError(
            f"{name} {point.tolist()} is too close to an obstacle "
            f"for clearance={clearance}"
        )

    return point

@dataclass
class PlanningStats:
    """Collects and reports runtime statistics for one planning run."""

    map_name: str = ""
    num_nodes: int = 0
    num_edges: int = 0
    astar_explored: int = 0
    path_found: bool = False
    path_length_px: float = 0.0
    build_time_s: float = 0.0
    search_time_s: float = 0.0

    @property
    def total_time_s(self) -> float:
        return self.build_time_s + self.search_time_s

    def report(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"{'Map':<22}: {self.map_name}",
            f"{'Nodes sampled':<22}: {self.num_nodes}",
            f"{'Edges built':<22}: {self.num_edges}",
            f"{'A* nodes explored':<22}: {self.astar_explored}",
            f"{'Path found':<22}: {'Yes' if self.path_found else 'No'}",
            f"{'Path length (px)':<22}: {self.path_length_px:.1f}",
            f"{'Roadmap build time':<22}: {self.build_time_s * 1000:.1f} ms",
            f"{'A* search time':<22}: {self.search_time_s * 1000:.1f} ms",
            f"{'Total time':<22}: {self.total_time_s * 1000:.1f} ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "map": self.map_name,
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "astar_explored": self.astar_explored,
            "path_found": self.path_found,
            "path_length_px": round(self.path_length_px, 1),
            "build_time_ms": round(self.build_time_s * 1000, 1),
            "search_time_ms": round(self.search_time_s * 1000, 1),
            "total_time_ms": round(self.total_time_s * 1000, 1),
        }
