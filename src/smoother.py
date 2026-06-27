"""
smoother.py — Post-process a PRM or RRT path to reduce waypoint count.

Two functions are provided:

  shortcut_smooth   Greedy shortcutting — iteratively removes waypoints
                    that can be bypassed without a collision.

  interpolate_path  Arc-length parameterization — resamples a set of
                    waypoints into a dense, evenly-spaced position sequence
                    suitable for robot controllers.
"""

from __future__ import annotations
import numpy as np
from .collision_checker import collision


def shortcut_smooth(
    path: list[int],
    nodes: np.ndarray,
    obstacle_map: np.ndarray,
    max_iterations: int = 100,
) -> list[int]:
    """
    Greedy shortcutting: remove redundant intermediate waypoints.

    On each pass, the algorithm tries to connect non-adjacent waypoints
    directly. If the direct segment is collision-free, all nodes between
    them are removed. Passes repeat until no further shortcuts are found
    or `max_iterations` is reached.

    Args:
        path:           Ordered list of node indices (from A* or RRT).
        nodes:          (N, 2) array of [row, col] node coordinates.
        obstacle_map:   Binary obstacle map (0 = obstacle, 255 = free).
        max_iterations: Maximum number of smoothing passes.

    Returns:
        Shortened list of node indices. Never longer than the input path.
    """
    if len(path) <= 2:
        return path

    smoothed = list(path)

    for _ in range(max_iterations):
        improved = False
        i = 0
        while i < len(smoothed) - 2:
            j = len(smoothed) - 1
            while j > i + 1:
                if not collision(nodes[smoothed[i]], nodes[smoothed[j]], obstacle_map):
                    smoothed = smoothed[: i + 1] + smoothed[j:]
                    improved = True
                    break
                j -= 1
            i += 1

        if not improved:
            break

    return smoothed


def interpolate_path(
    waypoints: np.ndarray,
    num_points: int = 200,
) -> np.ndarray:
    """
    Resample a waypoint sequence into a dense, evenly-spaced path.

    Uses arc-length parameterization so that points are distributed
    uniformly along the total path length rather than uniformly by
    waypoint index.

    Args:
        waypoints:  (M, 2) array of [row, col] positions.
        num_points: Total number of output points.

    Returns:
        (num_points, 2) array of interpolated positions.
    """
    if len(waypoints) < 2:
        return waypoints

    diffs     = np.diff(waypoints, axis=0)
    seg_lens  = np.linalg.norm(diffs, axis=1)
    cum_lens  = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    if total_len < 1e-9:
        return waypoints

    sample_lens = np.linspace(0, total_len, num_points)
    rows = np.interp(sample_lens, cum_lens, waypoints[:, 0])
    cols = np.interp(sample_lens, cum_lens, waypoints[:, 1])

    return np.column_stack([rows, cols])


def smooth_path_length(waypoints: np.ndarray) -> float:
    """Compute the total Euclidean length of a waypoint path in pixels."""
    if len(waypoints) < 2:
        return 0.0
    diffs = np.diff(waypoints, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))
