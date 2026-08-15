"""
collision_checker.py - Point and segment collision checks on binary obstacle maps.
"""

from __future__ import annotations

import numpy as np


def _cell_for_point(point: np.ndarray) -> tuple[int, int]:
    """Return the raster cell containing a continuous [row, col] point."""
    cell = np.floor(point + 0.5).astype(int)
    return int(cell[0]), int(cell[1])


def _on_cell_boundary(value: float) -> bool:
    """Return whether a coordinate lies on a half-integer cell boundary."""
    shifted = value + 0.5
    return bool(np.isclose(shifted, round(shifted), rtol=0.0, atol=1e-12))


def _endpoint_cells(point: np.ndarray) -> list[tuple[int, int]]:
    """Return every raster cell touched by an endpoint."""
    row, col = _cell_for_point(point)
    row_offsets = (0, -1) if _on_cell_boundary(float(point[0])) else (0,)
    col_offsets = (0, -1) if _on_cell_boundary(float(point[1])) else (0,)
    return [(row + dr, col + dc) for dr in row_offsets for dc in col_offsets]


def _supercover_cells(
    node1: np.ndarray,
    node2: np.ndarray,
) -> list[tuple[int, int]]:
    """Traverse every raster cell touched by a continuous segment."""
    current = np.array(_cell_for_point(node1), dtype=int)
    delta = node2 - node1
    step = np.sign(delta).astype(int)

    shifted_start = node1 + 0.5
    t_max = np.full(2, np.inf, dtype=float)
    t_delta = np.full(2, np.inf, dtype=float)

    for axis in range(2):
        if step[axis] == 0:
            continue
        next_boundary = current[axis] + (1 if step[axis] > 0 else 0)
        t_max[axis] = (next_boundary - shifted_start[axis]) / delta[axis]
        t_delta[axis] = 1.0 / abs(delta[axis])

    row_boundary_line = step[0] == 0 and _on_cell_boundary(float(node1[0]))
    col_boundary_line = step[1] == 0 and _on_cell_boundary(float(node1[1]))
    cells: list[tuple[int, int]] = []

    def add_cell(row: int, col: int) -> None:
        cells.append((row, col))
        if row_boundary_line:
            cells.append((row - 1, col))
        if col_boundary_line:
            cells.append((row, col - 1))
        if row_boundary_line and col_boundary_line:
            cells.append((row - 1, col - 1))

    add_cell(int(current[0]), int(current[1]))

    while float(np.min(t_max)) <= 1.0 + 1e-12:
        if np.isclose(t_max[0], t_max[1], rtol=1e-12, atol=1e-12):
            add_cell(int(current[0] + step[0]), int(current[1]))
            add_cell(int(current[0]), int(current[1] + step[1]))
            current += step
            t_max += t_delta
        elif t_max[0] < t_max[1]:
            current[0] += step[0]
            t_max[0] += t_delta[0]
        else:
            current[1] += step[1]
            t_max[1] += t_delta[1]

        add_cell(int(current[0]), int(current[1]))

    return cells


def is_in_free_space(point: np.ndarray, obstacle_map: np.ndarray) -> bool:
    """
    Check whether a point lies inside the map and in free space.

    Args:
        point: Coordinate as [row, col].
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.

    Returns:
        True if the point is within bounds and lies in free space.
    """
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return False

    height, width = obstacle_map.shape

    if not (0 <= point[0] <= height - 1 and 0 <= point[1] <= width - 1):
        return False

    row, col = _cell_for_point(point)
    return bool(obstacle_map[row, col] == 255)


def collision(
    node1: np.ndarray,
    node2: np.ndarray,
    obstacle_map: np.ndarray,
) -> bool:
    """
    Check whether the straight-line segment between two nodes crosses an obstacle.

    Uses a conservative supercover grid traversal. Cells touched at grid corners
    or along grid boundaries are checked in addition to cells crossed through
    their interior.

    Args:
        node1: Start coordinate as [row, col].
        node2: End coordinate as [row, col].
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.
    Returns:
        True if the segment intersects an obstacle or goes out of bounds.
        False if the full segment is collision-free.
    """
    node1 = np.asarray(node1, dtype=float)
    node2 = np.asarray(node2, dtype=float)

    if node1.shape != (2,) or node2.shape != (2,):
        raise ValueError("node1 and node2 must be coordinates of shape (2,)")

    height, width = obstacle_map.shape
    if not np.all(np.isfinite(node1)) or not np.all(np.isfinite(node2)):
        return True

    endpoints = np.vstack([node1, node2])
    if np.any(endpoints[:, 0] < 0) or np.any(endpoints[:, 0] > height - 1):
        return True
    if np.any(endpoints[:, 1] < 0) or np.any(endpoints[:, 1] > width - 1):
        return True

    cells = _endpoint_cells(node1) + _endpoint_cells(node2)
    cells.extend(_supercover_cells(node1, node2))

    for row, col in cells:
        if row < 0 or row >= height or col < 0 or col >= width:
            return True
        if obstacle_map[row, col] == 0:
            return True

    return False
