"""
rrt.py — Rapidly-exploring Random Tree (RRT) path planner.

RRT grows a tree from the start node by randomly sampling free-space points
and extending toward them. Unlike PRM, it is single-query: a new tree is
built for each start-goal pair, making it well-suited to environments where
only one path query is needed.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial import KDTree

from .collision_checker import collision


class RRTPlanner:
    """
    Goal-biased RRT planner.

    Builds a tree from `start` by sampling random free-space points and
    steering toward them step by step. Terminates when a node lands within
    `goal_radius` pixels of the goal and can connect to it collision-free.

    Args:
        max_iter:    Maximum number of extension attempts.
        step_size:   Maximum edge length (pixels) per extension step.
        goal_radius: Distance threshold at which the goal is considered reached.
        goal_bias:   Probability of sampling the goal directly on each iteration.
        seed:        Random seed for reproducibility.
    """

    def __init__(
        self,
        max_iter: int = 5000,
        step_size: float = 30.0,
        goal_radius: float = 25.0,
        goal_bias: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.max_iter    = max_iter
        self.step_size   = step_size
        self.goal_radius = goal_radius
        self.goal_bias   = goal_bias
        self.rng         = np.random.default_rng(seed)

        self._nodes:   list[np.ndarray] = []
        self._parents: list[int | None] = []

    @property
    def nodes(self) -> np.ndarray:
        """Return all tree nodes as an (N, 2) array."""
        return np.array(self._nodes)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> tuple[list[np.ndarray], int]:
        """
        Grow an RRT from start toward goal.

        Args:
            start: Start position [row, col].
            goal:  Goal position [row, col].
            obstacle_map: Binary map (0 = obstacle, 255 = free).

        Returns:
            Tuple of:
              - path: Ordered list of [row, col] waypoints from start to goal.
                      Empty list if no path was found within max_iter.
              - iterations: Number of extension attempts made.
        """
        h, w = obstacle_map.shape

        self._nodes   = [start.copy()]
        self._parents = [None]

        for iteration in range(self.max_iter):
            if self.rng.random() < self.goal_bias:
                sample = goal.astype(float)
            else:
                sample = np.array([
                    self.rng.uniform(0, h - 1),
                    self.rng.uniform(0, w - 1),
                ])

            nearest_idx = self._nearest(sample)
            nearest     = self._nodes[nearest_idx]

            direction = sample - nearest
            dist      = np.linalg.norm(direction)
            if dist < 1e-6:
                continue

            new_node = nearest + (direction / dist) * min(dist, self.step_size)
            new_node = np.clip(new_node, [0, 0], [h - 1, w - 1])

            if collision(nearest, new_node, obstacle_map):
                continue

            self._nodes.append(new_node)
            self._parents.append(nearest_idx)
            new_idx = len(self._nodes) - 1

            if np.linalg.norm(new_node - goal) <= self.goal_radius:
                if not collision(new_node, goal, obstacle_map):
                    self._nodes.append(goal.copy())
                    self._parents.append(new_idx)
                    return self._reconstruct_path(len(self._nodes) - 1), iteration + 1

        return [], self.max_iter

    def _nearest(self, point: np.ndarray) -> int:
        """Return the index of the closest node in the tree to `point`."""
        if len(self._nodes) == 1:
            return 0
        tree = KDTree(np.array(self._nodes))
        _, idx = tree.query(point, k=1)
        return int(idx)

    def _reconstruct_path(self, goal_idx: int) -> list[np.ndarray]:
        """Walk parent pointers from goal back to start and reverse."""
        path: list[np.ndarray] = []
        idx: int | None = goal_idx
        while idx is not None:
            path.append(self._nodes[idx])
            idx = self._parents[idx]
        path.reverse()
        return path


def rrt_path_length(path: list[np.ndarray]) -> float:
    """Compute the total Euclidean length of an RRT path in pixels."""
    if len(path) < 2:
        return 0.0
    coords = np.array(path)
    return float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
