import numpy as np

from src.collision_checker import collision
from src.rrt import RRTPlanner


def make_open_map(size: int = 30) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def make_planner(seed: int = 42) -> RRTPlanner:
    return RRTPlanner(
        max_iter=200,
        step_size=5.0,
        goal_radius=4.0,
        goal_bias=0.25,
        seed=seed,
    )


def assert_path_is_collision_free(path: list[np.ndarray], obstacle_map: np.ndarray) -> None:
    for start, end in zip(path, path[1:]):
        assert collision(start, end, obstacle_map) is False


def test_rrt_start_equals_goal_returns_trivial_path_immediately():
    obstacle_map = make_open_map()
    point = np.array([10.0, 10.0])
    planner = make_planner()

    path, iterations = planner.plan(point, point, obstacle_map)

    assert iterations == 0
    assert len(path) == 1
    assert np.array_equal(path[0], point)
    assert np.array_equal(planner.nodes, np.array([point]))


def test_rrt_finds_collision_free_path_on_open_map():
    obstacle_map = make_open_map()
    start = np.array([2.0, 2.0])
    goal = np.array([27.0, 27.0])
    planner = make_planner(seed=7)

    path, iterations = planner.plan(start, goal, obstacle_map)

    assert path
    assert iterations <= planner.max_iter
    assert np.array_equal(path[0], start)
    assert np.array_equal(path[-1], goal)
    assert_path_is_collision_free(path, obstacle_map)


def test_rrt_failure_returns_empty_path_and_max_iterations():
    obstacle_map = make_open_map(20)
    obstacle_map[:, 10] = 0
    start = np.array([10.0, 2.0])
    goal = np.array([10.0, 17.0])
    planner = RRTPlanner(
        max_iter=80,
        step_size=3.0,
        goal_radius=2.0,
        goal_bias=0.2,
        seed=11,
    )

    path, iterations = planner.plan(start, goal, obstacle_map)

    assert path == []
    assert iterations == planner.max_iter


def test_rrt_tree_nodes_stay_within_map_bounds():
    obstacle_map = make_open_map(20)
    planner = make_planner(seed=3)

    planner.plan(
        np.array([1.0, 1.0]),
        np.array([18.0, 18.0]),
        obstacle_map,
    )

    assert np.all(planner.nodes[:, 0] >= 0)
    assert np.all(planner.nodes[:, 0] <= obstacle_map.shape[0] - 1)
    assert np.all(planner.nodes[:, 1] >= 0)
    assert np.all(planner.nodes[:, 1] <= obstacle_map.shape[1] - 1)


def test_rrt_is_deterministic_for_a_fixed_seed():
    obstacle_map = make_open_map()
    start = np.array([2.0, 2.0])
    goal = np.array([27.0, 27.0])
    first = make_planner(seed=19)
    second = make_planner(seed=19)

    first_path, first_iterations = first.plan(start, goal, obstacle_map)
    second_path, second_iterations = second.plan(start, goal, obstacle_map)

    assert first_iterations == second_iterations
    assert np.array_equal(first.nodes, second.nodes)
    assert np.array_equal(np.array(first_path), np.array(second_path))
