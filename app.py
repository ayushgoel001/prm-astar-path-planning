"""
app.py — Interactive Streamlit demo for PRM + A* and RRT path planning.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

# Standard library
import io
import sys
import time
from pathlib import Path

# Third-party — matplotlib backend must be configured before pyplot is imported
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.collections import LineCollection

# Local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.astar import astar, path_length
from src.map_loader import apply_clearance, load_map
from src.prm import build_roadmap, sample_free_space
from src.rrt import RRTPlanner, rrt_path_length
from src.smoother import shortcut_smooth, smooth_path_length
from src.utils import set_seed, validate_planning_point


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILTIN_MAPS: dict[str, str] = {
    "Maze 1 — Complex Maze": "maze_1.png",
    "Maze 2 — Grid World": "maze_2.png",
    "Maze 3 — Random Obstacles": "maze_3.png",
}

DEFAULT_START: dict[str, tuple[int, int]] = {
    "maze_1.png": (5, 235),
    "maze_2.png": (30, 30),
    "maze_3.png": (30, 30),
}

DEFAULT_GOAL: dict[str, tuple[int, int]] = {
    "maze_1.png": (350, 450),
    "maze_2.png": (460, 460),
    "maze_3.png": (460, 440),
}


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_builtin_map(filename: str) -> np.ndarray:
    """Load a built-in binary obstacle map from the examples directory."""
    return load_map(Path("examples") / filename)


def decode_uploaded_map(uploaded_file) -> np.ndarray | None:
    """Decode an uploaded image file into a binary obstacle map.

    Returns None if the image cannot be decoded by OpenCV.
    """
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    _, binary_map = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_map


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_map_preview(
    obstacle_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
) -> plt.Figure:
    """Render the obstacle map with start and goal markers."""
    height, width = obstacle_map.shape
    figure, axes = plt.subplots(figsize=(6, 4))
    axes.imshow(obstacle_map, cmap="gray", origin="upper")
    axes.scatter(start[1], start[0], c="tab:cyan", s=150, marker="*", label="Start")
    axes.scatter(goal[1], goal[0], c="tab:orange", s=150, marker="*", label="Goal")
    axes.set_title(f"Map preview  ({width} × {height} px)", fontsize=11)
    axes.axis("off")
    axes.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return figure


def render_prm_figure(
    obstacle_map: np.ndarray,
    nodes: np.ndarray,
    edges: dict[int, list[int]],
    path: list[int],
    start_idx: int,
    goal_idx: int,
    title: str,
) -> plt.Figure:
    """Render the PRM roadmap, sampled nodes, and the A* path."""
    figure, axes = plt.subplots(figsize=(9, 7))
    axes.imshow(obstacle_map, cmap="gray", origin="upper")
    axes.axis("off")

    edge_segments = [
        [[nodes[i, 1], nodes[i, 0]], [nodes[j, 1], nodes[j, 0]]]
        for i, neighbors in edges.items()
        for j in neighbors
        if i < j
    ]
    if edge_segments:
        axes.add_collection(
            LineCollection(edge_segments, colors="tab:blue", linewidths=0.35, alpha=0.45)
        )

    axes.scatter(
        nodes[:, 1], nodes[:, 0],
        c="tab:red", s=6, alpha=0.75, zorder=3, label="PRM nodes",
    )

    if path:
        path_coords = nodes[path]
        axes.plot(
            path_coords[:, 1], path_coords[:, 0],
            color="tab:green", linewidth=2.5, zorder=6, label="Path",
        )

    axes.scatter(
        nodes[start_idx, 1], nodes[start_idx, 0],
        c="tab:cyan", s=150, marker="*", zorder=8, label="Start",
    )
    axes.scatter(
        nodes[goal_idx, 1], nodes[goal_idx, 0],
        c="tab:orange", s=150, marker="*", zorder=8, label="Goal",
    )

    axes.set_title(title, fontsize=11)
    axes.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return figure


def render_rrt_figure(
    obstacle_map: np.ndarray,
    tree_nodes: np.ndarray,
    path: list,
    start: np.ndarray,
    goal: np.ndarray,
    title: str,
) -> plt.Figure:
    """Render RRT tree nodes and the planned path."""
    figure, axes = plt.subplots(figsize=(9, 7))
    axes.imshow(obstacle_map, cmap="gray", origin="upper")
    axes.axis("off")

    if tree_nodes is not None and len(tree_nodes) > 0:
        axes.scatter(
            tree_nodes[:, 1], tree_nodes[:, 0],
            c="tab:blue", s=4, alpha=0.45, zorder=2, label="RRT tree",
        )

    if path:
        path_arr = np.array(path)
        axes.plot(
            path_arr[:, 1], path_arr[:, 0],
            color="tab:green", linewidth=2.5, zorder=6, label="Path",
        )

    axes.scatter(start[1], start[0], c="tab:cyan", s=150, marker="*", zorder=8, label="Start")
    axes.scatter(goal[1], goal[0], c="tab:orange", s=150, marker="*", zorder=8, label="Goal")

    axes.set_title(title, fontsize=11)
    axes.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return figure


def figure_to_png_bytes(figure: plt.Figure) -> bytes:
    """Serialize a Matplotlib figure to PNG bytes."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Planner execution
# ---------------------------------------------------------------------------

def run_prm_planner(
    obstacle_map: np.ndarray,
    safe_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    num_nodes: int,
    k_neighbors: int,
    seed: int,
    apply_smoothing: bool,
) -> tuple[plt.Figure, dict]:
    """Build a PRM roadmap and run A* search from start to goal.

    Returns the result figure and a statistics dictionary.
    """
    set_seed(seed)
    total_start = time.perf_counter()

    build_start = time.perf_counter()
    nodes = sample_free_space(safe_map, num_nodes, seed=seed)
    start_idx = len(nodes)
    goal_idx = len(nodes) + 1
    nodes = np.vstack([nodes, start, goal])
    edges, total_edges = build_roadmap(
        nodes, safe_map, k_neighbors, anchor_indices=[start_idx, goal_idx]
    )
    build_ms = (time.perf_counter() - build_start) * 1000

    search_start = time.perf_counter()
    path, explored = astar(edges, nodes, start_idx, goal_idx)
    search_ms = (time.perf_counter() - search_start) * 1000

    raw_length = path_length(path, nodes) if path else 0.0

    if path and apply_smoothing:
        path = shortcut_smooth(path, nodes, safe_map)
        final_length = smooth_path_length(nodes[path])
    else:
        final_length = raw_length

    total_ms = (time.perf_counter() - total_start) * 1000

    length_label = f"{round(final_length)} px" if path else "no path"
    title = (
        f"PRM + A*  |  {len(nodes)} nodes  |  {total_edges} edges  |  "
        f"Path length: {length_label}"
    )

    figure = render_prm_figure(obstacle_map, nodes, edges, path, start_idx, goal_idx, title)

    stats = {
        "found": bool(path),
        "nodes": len(nodes),
        "edges": total_edges,
        "explored": explored,
        "path_length": final_length,
        "build_ms": build_ms,
        "search_ms": search_ms,
        "total_ms": total_ms,
    }

    return figure, stats


def run_rrt_planner(
    obstacle_map: np.ndarray,
    safe_map: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    max_iter: int,
    seed: int,
) -> tuple[plt.Figure, dict]:
    """Run RRT path planning from start to goal.

    Returns the result figure and a statistics dictionary.
    """
    planner = RRTPlanner(
        max_iter=max_iter,
        step_size=25.0,
        goal_bias=0.15,
        goal_radius=20.0,
        seed=seed,
    )

    start_time = time.perf_counter()
    path, iterations = planner.plan(start.astype(float), goal.astype(float), safe_map)
    total_ms = (time.perf_counter() - start_time) * 1000

    tree_nodes = planner.nodes
    final_length = rrt_path_length(path) if path else 0.0

    length_label = f"{round(final_length)} px" if path else "no path"
    title = (
        f"RRT  |  {len(tree_nodes)} tree nodes  |  {iterations} iterations  |  "
        f"Path length: {length_label}"
    )

    figure = render_rrt_figure(obstacle_map, tree_nodes, path, start, goal, title)

    stats = {
        "found": bool(path),
        "tree_nodes": len(tree_nodes),
        "iterations": iterations,
        "path_length": final_length,
        "total_ms": total_ms,
    }

    return figure, stats


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render the sidebar controls and return the current configuration as a dict."""
    with st.sidebar:
        st.header("Configuration")

        # Map selection
        st.subheader("Map")
        map_source = st.radio(
            "Map source", ["Built-in example", "Upload custom map"], horizontal=True
        )

        obstacle_map: np.ndarray | None = None
        selected_map_file = "maze_1.png"

        if map_source == "Built-in example":
            selected_map_label = st.selectbox("Map", list(BUILTIN_MAPS.keys()))
            selected_map_file = BUILTIN_MAPS[selected_map_label]
            try:
                obstacle_map = load_builtin_map(selected_map_file)
            except FileNotFoundError:
                st.error(
                    f"Map file not found: `examples/{selected_map_file}`. "
                    "Run `python scripts/generate_maps.py` to generate the example maps."
                )
        else:
            uploaded_file = st.file_uploader(
                "Upload a PNG or JPG map", type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                obstacle_map = decode_uploaded_map(uploaded_file)
                if obstacle_map is None:
                    st.error("Could not decode the uploaded image.")

        st.divider()

        # Algorithm settings
        st.subheader("Algorithm")
        algorithm = st.selectbox("Planner", ["PRM + A*", "RRT"])
        num_nodes = st.slider("PRM nodes / RRT iterations scale", 100, 1000, 400, step=50)
        k_neighbors = st.slider("PRM k-nearest neighbors", 5, 40, 15)
        clearance = st.slider("Obstacle clearance (px)", 0, 15, 3)
        seed = st.number_input("Random seed", value=42, min_value=0, max_value=9999)
        apply_smoothing = st.checkbox("Apply path smoothing", value=True)

        st.divider()

        # Start and goal coordinates
        st.subheader("Start and Goal")
        st.caption(
            "Coordinates are in (row, column) format. "
            "The top-left pixel is (0, 0)."
        )

        default_start = DEFAULT_START.get(selected_map_file, (5, 5))
        default_goal = DEFAULT_GOAL.get(selected_map_file, (350, 350))

        col_start, col_goal = st.columns(2)
        with col_start:
            start_row = st.number_input("Start row", value=default_start[0], min_value=0)
            start_col = st.number_input("Start column", value=default_start[1], min_value=0)
        with col_goal:
            goal_row = st.number_input("Goal row", value=default_goal[0], min_value=0)
            goal_col = st.number_input("Goal column", value=default_goal[1], min_value=0)

        run_button = st.button("Run planner", type="primary", width="stretch")

    return {
        "obstacle_map": obstacle_map,
        "selected_map_file": selected_map_file,
        "algorithm": algorithm,
        "num_nodes": int(num_nodes),
        "k_neighbors": int(k_neighbors),
        "clearance": int(clearance),
        "seed": int(seed),
        "apply_smoothing": bool(apply_smoothing),
        "start_coord": np.array([int(start_row), int(start_col)]),
        "goal_coord": np.array([int(goal_row), int(goal_col)]),
        "run_button": run_button,
    }


# ---------------------------------------------------------------------------
# Metrics display
# ---------------------------------------------------------------------------

def display_prm_metrics(stats: dict) -> None:
    """Display PRM + A* run statistics as Streamlit metric tiles."""
    cols = st.columns(5)
    cols[0].metric("Nodes", stats["nodes"])
    cols[1].metric("Edges", stats["edges"])
    cols[2].metric("A* explored", stats["explored"])
    cols[3].metric(
        "Path length",
        f"{stats['path_length']:.0f} px" if stats["found"] else "—",
    )
    cols[4].metric("Total time", f"{stats['total_ms']:.0f} ms")


def display_rrt_metrics(stats: dict) -> None:
    """Display RRT run statistics as Streamlit metric tiles."""
    cols = st.columns(4)
    cols[0].metric("Tree nodes", stats["tree_nodes"])
    cols[1].metric("Iterations", stats["iterations"])
    cols[2].metric(
        "Path length",
        f"{stats['path_length']:.0f} px" if stats["found"] else "—",
    )
    cols[3].metric("Total time", f"{stats['total_ms']:.0f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Path Planning Demo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Path Planning in 2D Environments")
    st.write(
        "Sampling-based path planning using PRM with A\\* and RRT on binary obstacle maps. "
        "Configure the planner in the sidebar, then click **Run planner**."
    )

    config = render_sidebar()

    obstacle_map: np.ndarray | None = config["obstacle_map"]
    if obstacle_map is None:
        st.info("Select a built-in map or upload a custom map to begin.")
        st.stop()

    safe_map = apply_clearance(obstacle_map, config["clearance"])

    try:
        start_coord = validate_planning_point(
            config["start_coord"],
            obstacle_map,
            safe_map,
            clearance=config["clearance"],
            name="start",
        )
        goal_coord = validate_planning_point(
            config["goal_coord"],
            obstacle_map,
            safe_map,
            clearance=config["clearance"],
            name="goal",
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if not config["run_button"]:
        preview = render_map_preview(obstacle_map, start_coord, goal_coord)
        st.pyplot(preview, width="content")
        plt.close(preview)
        st.stop()

    try:
        with st.spinner("Running planner..."):
            if config["algorithm"] == "PRM + A*":
                figure, stats = run_prm_planner(
                    obstacle_map=obstacle_map,
                    safe_map=safe_map,
                    start=start_coord,
                    goal=goal_coord,
                    num_nodes=config["num_nodes"],
                    k_neighbors=config["k_neighbors"],
                    seed=config["seed"],
                    apply_smoothing=config["apply_smoothing"],
                )
                display_prm_metrics(stats)
            else:
                figure, stats = run_rrt_planner(
                    obstacle_map=obstacle_map,
                    safe_map=safe_map,
                    start=start_coord,
                    goal=goal_coord,
                    max_iter=config["num_nodes"] * 10,
                    seed=config["seed"],
                )
                display_rrt_metrics(stats)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    if stats["found"]:
        st.success("Path found.")
    else:
        hint = (
            "Try increasing the number of nodes or k-nearest neighbors, "
            "or reducing the obstacle clearance."
            if config["algorithm"] == "PRM + A*"
            else "Try increasing the maximum iterations."
        )
        st.warning(f"No path found. {hint}")

    st.pyplot(figure, width="stretch")

    png_bytes = figure_to_png_bytes(figure)
    plt.close(figure)

    st.download_button(
        label="Download result image",
        data=png_bytes,
        file_name="path_planning_result.png",
        mime="image/png",
    )

    with st.expander("Method overview"):
        st.markdown(
            r"""
**PRM with A\***

1. Samples collision-free points uniformly from the free space.
2. Connects nearby nodes using KD-Tree nearest-neighbor search.
3. Validates roadmap edges with line-of-sight collision checking.
4. Runs A\* on the roadmap using Euclidean distance as the heuristic.

**RRT**

1. Grows a tree from the start position by sampling random free-space points.
2. Extends the tree toward each sample by a fixed step size.
3. Accepts extensions only when the connecting segment is collision-free.
4. Terminates when the tree reaches within the goal radius.

**Path smoothing**

Shortcut-based smoothing iteratively removes redundant waypoints by directly
connecting non-adjacent nodes whenever the segment is collision-free.
            """
        )


main()