import numpy as np

from src.astar import astar
from src.collision_checker import collision
from src.map_loader import apply_clearance
from src.prm import build_roadmap, sample_free_space
from src.utils import validate_planning_point


def test_prm_astar_pipeline_finds_collision_free_path():
    obstacle_map = np.full((30, 30), 255, dtype=np.uint8)
    obstacle_map[:, 15] = 0
    obstacle_map[10:20, 15] = 255
    safe_map = apply_clearance(obstacle_map, clearance=1)

    start = validate_planning_point(
        np.array([15, 3]), obstacle_map, safe_map, clearance=1, name="start"
    )
    goal = validate_planning_point(
        np.array([15, 26]), obstacle_map, safe_map, clearance=1, name="goal"
    )

    nodes = sample_free_space(
        safe_map,
        num_samples=180,
        border_margin=1,
        seed=42,
    )
    start_idx = len(nodes)
    goal_idx = start_idx + 1
    nodes = np.vstack([nodes, start, goal])

    graph, edge_count = build_roadmap(
        nodes,
        safe_map,
        k_neighbors=15,
        anchor_indices=[start_idx, goal_idx],
    )
    path, expanded_nodes = astar(graph, nodes, start_idx, goal_idx)

    assert edge_count > 0
    assert path
    assert path[0] == start_idx
    assert path[-1] == goal_idx
    assert np.array_equal(nodes[path[0]], start)
    assert np.array_equal(nodes[path[-1]], goal)
    assert expanded_nodes <= len(graph)

    for current, following in zip(path, path[1:]):
        assert collision(nodes[current], nodes[following], safe_map) is False
