"""
prm.py - Probabilistic Roadmap Method sampling and graph construction.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from .collision_checker import collision


def sample_free_space(
    obstacle_map: np.ndarray,
    num_samples: int,
    border_margin: int = 5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Uniformly sample collision-free nodes from a binary obstacle map.

    Args:
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.
        num_samples: Number of nodes to sample.
        border_margin: Minimum pixel distance from the image border.
        seed: Optional random seed for reproducible sampling.

    Returns:
        Array of shape (num_samples, 2) containing [row, col] coordinates.

    Raises:
        ValueError: If inputs are invalid or free space is insufficient.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    if border_margin < 0:
        raise ValueError("border_margin must be non-negative")

    height, width = obstacle_map.shape

    if 2 * border_margin >= height or 2 * border_margin >= width:
        raise ValueError("border_margin is too large for the given map dimensions")

    free_mask = obstacle_map == 255

    border_mask = np.zeros_like(free_mask, dtype=bool)
    border_mask[
        border_margin : height - border_margin,
        border_margin : width - border_margin,
    ] = True

    valid_mask = free_mask & border_mask
    valid_positions = np.argwhere(valid_mask)

    if len(valid_positions) < num_samples:
        raise ValueError(
            f"Requested {num_samples} samples, but only {len(valid_positions)} "
            "valid free-space pixels are available. Reduce the number of samples "
            "or decrease border_margin."
        )

    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(
        len(valid_positions),
        size=num_samples,
        replace=False,
    )

    return valid_positions[chosen_indices]


def build_roadmap(
    nodes: np.ndarray,
    obstacle_map: np.ndarray,
    k_neighbors: int,
    anchor_indices: list[int] | None = None,
) -> tuple[dict[int, list[int]], int]:
    """
    Build an undirected PRM roadmap using KD-Tree nearest-neighbor search.

    Each candidate edge is collision-checked before insertion. Anchor nodes,
    usually start and goal, use a wider neighbor search to improve connectivity
    in narrow regions.

    Args:
        nodes: Array of shape (N, 2) containing node coordinates as [row, col].
        obstacle_map: Binary map where 0 = obstacle and 255 = free space.
        k_neighbors: Number of nearest neighbors to attempt for each node.
        anchor_indices: Optional node indices that receive wider neighbor search.

    Returns:
        A tuple containing:
            - edges: adjacency list mapping node index to neighbor indices.
            - total_edges: number of undirected roadmap edges.

    Raises:
        ValueError: If k_neighbors is invalid.
    """
    num_nodes = len(nodes)

    if num_nodes == 0:
        return {}, 0

    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")

    kdtree = KDTree(nodes)
    anchor_set = set(anchor_indices or [])

    standard_k = min(k_neighbors + 1, num_nodes)
    anchor_k = min(k_neighbors * 3, num_nodes)

    _, standard_neighbors = kdtree.query(nodes, k=standard_k)

    edges_set: dict[int, set[int]] = {i: set() for i in range(num_nodes)}

    for i in range(num_nodes):
        if i in anchor_set:
            _, neighbors = kdtree.query(nodes[i], k=anchor_k)
            neighbors = np.atleast_1d(neighbors)[1:]
        else:
            neighbors = np.atleast_1d(standard_neighbors[i])[1:]

        for j in neighbors:
            j = int(j)

            if j == i:
                continue

            if collision(nodes[i], nodes[j], obstacle_map):
                continue

            edges_set[i].add(j)
            edges_set[j].add(i)

    edges: dict[int, list[int]] = {
        node: sorted(neighbors) for node, neighbors in edges_set.items()
    }

    total_edges = sum(len(neighbors) for neighbors in edges.values()) // 2
    return edges, total_edges