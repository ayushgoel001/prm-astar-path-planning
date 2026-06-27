import numpy as np
import pytest

from src.collision_checker import collision
from src.prm import build_roadmap, sample_free_space


def test_sample_free_space_returns_requested_number_of_nodes():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)

    nodes = sample_free_space(
        obstacle_map,
        num_samples=50,
        border_margin=2,
        seed=42,
    )

    assert nodes.shape == (50, 2)


def test_sample_free_space_samples_only_free_pixels():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)
    obstacle_map[5:15, 5:15] = 0

    nodes = sample_free_space(
        obstacle_map,
        num_samples=30,
        border_margin=1,
        seed=42,
    )

    for row, col in nodes:
        assert obstacle_map[row, col] == 255


def test_sample_free_space_is_reproducible_with_seed():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)

    nodes_a = sample_free_space(obstacle_map, 40, border_margin=1, seed=7)
    nodes_b = sample_free_space(obstacle_map, 40, border_margin=1, seed=7)

    assert np.array_equal(nodes_a, nodes_b)


def test_sample_free_space_rejects_invalid_sample_count():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)

    with pytest.raises(ValueError):
        sample_free_space(obstacle_map, num_samples=0)


def test_sample_free_space_rejects_excessive_samples():
    obstacle_map = np.full((10, 10), 255, dtype=np.uint8)
    obstacle_map[:, :] = 0
    obstacle_map[5, 5] = 255

    with pytest.raises(ValueError):
        sample_free_space(obstacle_map, num_samples=2, border_margin=0)


def test_build_roadmap_returns_undirected_edges():
    obstacle_map = np.full((30, 30), 255, dtype=np.uint8)

    nodes = np.array(
        [
            [5, 5],
            [5, 10],
            [10, 10],
            [15, 15],
        ]
    )

    edges, _ = build_roadmap(nodes, obstacle_map, k_neighbors=3)

    for node, neighbors in edges.items():
        for neighbor in neighbors:
            assert node in edges[neighbor]


def test_build_roadmap_edges_are_collision_free():
    obstacle_map = np.full((30, 30), 255, dtype=np.uint8)
    obstacle_map[:, 15] = 0

    nodes = np.array(
        [
            [5, 5],
            [10, 5],
            [20, 5],
            [5, 25],
            [10, 25],
            [20, 25],
        ]
    )

    edges, _ = build_roadmap(nodes, obstacle_map, k_neighbors=5)

    for node, neighbors in edges.items():
        for neighbor in neighbors:
            assert not collision(nodes[node], nodes[neighbor], obstacle_map)


def test_build_roadmap_rejects_invalid_neighbor_count():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)
    nodes = np.array([[5, 5], [10, 10]])

    with pytest.raises(ValueError):
        build_roadmap(nodes, obstacle_map, k_neighbors=0)