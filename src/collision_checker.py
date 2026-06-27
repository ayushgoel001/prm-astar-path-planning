"""
collision_checker.py - Point and segment collision checks on binary obstacle maps.
"""

from __future__ import annotations

import numpy as np


def is_in_free_space(point: np.ndarray, obstacle_map: np.ndarray) -> bool:
    """
    Check whether a point lies inside the map and in free space.

    Args:
        point: Coordinate as [row, col].
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.

    Returns:
        True if the point is within bounds and lies in free space.
    """
    row, col = np.rint(point).astype(int)
    height, width = obstacle_map.shape

    if row < 0 or row >= height or col < 0 or col >= width:
        return False

    return bool(obstacle_map[row, col] == 255)


def collision(
    node1: np.ndarray,
    node2: np.ndarray,
    obstacle_map: np.ndarray,
    num_checks: int | None = None,
) -> bool:
    """
    Check whether the straight-line segment between two nodes crosses an obstacle.

    The segment is sampled at approximately one-pixel resolution by default.
    This avoids missing thin obstacles on long edges while keeping short-edge
    checks lightweight.

    Args:
        node1: Start coordinate as [row, col].
        node2: End coordinate as [row, col].
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.
        num_checks: Optional number of samples along the segment. If None,
            the number of samples is derived from the Euclidean distance.

    Returns:
        True if the segment intersects an obstacle or goes out of bounds.
        False if the full segment is collision-free.
    """
    node1 = np.asarray(node1, dtype=float)
    node2 = np.asarray(node2, dtype=float)

    if node1.shape != (2,) or node2.shape != (2,):
        raise ValueError("node1 and node2 must be coordinates of shape (2,)")

    if num_checks is None:
        distance = np.linalg.norm(node2 - node1)
        num_checks = max(int(np.ceil(distance)) + 1, 2)

    if num_checks < 2:
        raise ValueError("num_checks must be at least 2")

    coords = np.rint(np.linspace(node1, node2, num=num_checks)).astype(int)

    height, width = obstacle_map.shape
    rows = coords[:, 0]
    cols = coords[:, 1]

    out_of_bounds = (
        (rows < 0)
        | (rows >= height)
        | (cols < 0)
        | (cols >= width)
    )

    if np.any(out_of_bounds):
        return True

    return bool(np.any(obstacle_map[rows, cols] == 0))