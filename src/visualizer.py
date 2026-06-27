"""
visualizer.py — Publication-quality multi-panel visualizations.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection


def visualize(
    obstacle_map: np.ndarray,
    nodes: np.ndarray,
    edges: dict[int, list[int]],
    path: list[int],
    start_idx: int,
    goal_idx: int,
    save_path: str | Path | None = None,
    show: bool = True,
    title_prefix: str = "",
) -> None:
    """
    Render a four-panel figure:
      1. Original obstacle map
      2. Sampled free-space nodes
      3. Full PRM roadmap (edges + nodes)
      4. Final A* path highlighted

    Args:
        obstacle_map: Binary map (0 = obstacle, 255 = free).
        nodes: Array of shape (N, 2) with node positions.
        edges: Adjacency list from build_roadmap().
        path: List of node indices returned by astar().
        start_idx: Index of the start node (highlighted green).
        goal_idx: Index of the goal node (highlighted red).
        save_path: If given, save the figure to this path.
        show: If True, display the figure interactively.
        title_prefix: Optional prefix for all subplot titles.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#1a1a2e")

    ax_style = dict(facecolor="#16213e")
    for ax in axes:
        ax.set_facecolor(ax_style["facecolor"])
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

    cmap = "gray"

    # ── Panel 1: Original map ────────────────────────────────────────────────
    axes[0].imshow(obstacle_map, cmap=cmap, origin="upper")
    axes[0].set_title(f"{title_prefix}Obstacle Map", color="white", fontsize=11, pad=8)
    axes[0].axis("off")

    # ── Panel 2: Sampled nodes ───────────────────────────────────────────────
    axes[1].imshow(obstacle_map, cmap=cmap, origin="upper")
    axes[1].scatter(
        nodes[:, 1], nodes[:, 0],
        c="#e94560", s=6, alpha=0.8, zorder=3,
    )
    axes[1].scatter(
        nodes[start_idx, 1], nodes[start_idx, 0],
        c="#00d4aa", s=80, marker="*", zorder=5, label="Start",
    )
    axes[1].scatter(
        nodes[goal_idx, 1], nodes[goal_idx, 0],
        c="#ff6b35", s=80, marker="*", zorder=5, label="Goal",
    )
    axes[1].set_title(f"{title_prefix}Sampled Nodes ({len(nodes)})", color="white", fontsize=11, pad=8)
    axes[1].axis("off")

    # ── Panel 3: Full PRM roadmap ────────────────────────────────────────────
    axes[2].imshow(obstacle_map, cmap=cmap, origin="upper")

    # Build LineCollection for all edges — much faster than looping plt.plot
    edge_segments = []
    for i, neighbors in edges.items():
        for j in neighbors:
            edge_segments.append([
                [nodes[i, 1], nodes[i, 0]],
                [nodes[j, 1], nodes[j, 0]],
            ])
    if edge_segments:
        lc = LineCollection(edge_segments, colors="#5577aa", linewidths=0.4, alpha=0.6, zorder=2)
        axes[2].add_collection(lc)

    axes[2].scatter(nodes[:, 1], nodes[:, 0], c="#e94560", s=4, alpha=0.9, zorder=3)
    axes[2].scatter(nodes[start_idx, 1], nodes[start_idx, 0], c="#00d4aa", s=80, marker="*", zorder=5)
    axes[2].scatter(nodes[goal_idx, 1], nodes[goal_idx, 0], c="#ff6b35", s=80, marker="*", zorder=5)
    total_edges = sum(len(v) for v in edges.values()) // 2
    axes[2].set_title(f"{title_prefix}PRM Roadmap ({total_edges} edges)", color="white", fontsize=11, pad=8)
    axes[2].axis("off")

    # ── Panel 4: A* path ─────────────────────────────────────────────────────
    axes[3].imshow(obstacle_map, cmap=cmap, origin="upper")

    if edge_segments:
        lc2 = LineCollection(edge_segments, colors="#5577aa", linewidths=0.3, alpha=0.4, zorder=2)
        axes[3].add_collection(lc2)

    axes[3].scatter(nodes[:, 1], nodes[:, 0], c="#e94560", s=4, alpha=0.6, zorder=3)

    if path:
        path_coords = nodes[path]
        axes[3].plot(
            path_coords[:, 1], path_coords[:, 0],
            color="#00ff88", linewidth=2.5, zorder=6, label="A* Path",
        )
        # Mark waypoints along the path
        axes[3].scatter(
            path_coords[1:-1, 1], path_coords[1:-1, 0],
            c="#ffdd00", s=20, zorder=7, alpha=0.9,
        )

    axes[3].scatter(nodes[start_idx, 1], nodes[start_idx, 0], c="#00d4aa", s=100, marker="*", zorder=8)
    axes[3].scatter(nodes[goal_idx, 1], nodes[goal_idx, 0], c="#ff6b35", s=100, marker="*", zorder=8)

    path_label = f"A* Path ({len(path)} nodes)" if path else "A* Path (not found)"
    axes[3].set_title(f"{title_prefix}{path_label}", color="white", fontsize=11, pad=8)
    axes[3].axis("off")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color="#00d4aa", label="Start"),
        mpatches.Patch(color="#ff6b35", label="Goal"),
        mpatches.Patch(color="#00ff88", label="A* Path"),
        mpatches.Patch(color="#e94560", label="PRM Nodes"),
        mpatches.Patch(color="#5577aa", label="Roadmap Edges"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        framealpha=0.2,
        labelcolor="white",
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {out}")

    if show:
        plt.show()

    plt.close(fig)


def visualize_single(
    obstacle_map: np.ndarray,
    nodes: np.ndarray,
    edges: dict[int, list[int]],
    path: list[int],
    start_idx: int,
    goal_idx: int,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Compact single-panel visualization: map + roadmap + path overlaid.
    Useful for quick inspection without saving large 4-panel figures.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.imshow(obstacle_map, cmap="gray", origin="upper")

    # Edges
    edge_segments = [
        [[nodes[i, 1], nodes[i, 0]], [nodes[j, 1], nodes[j, 0]]]
        for i, nbrs in edges.items() for j in nbrs
    ]
    if edge_segments:
        lc = LineCollection(edge_segments, colors="#5577aa", linewidths=0.4, alpha=0.5)
        ax.add_collection(lc)

    ax.scatter(nodes[:, 1], nodes[:, 0], c="#e94560", s=8, alpha=0.8, zorder=3)

    if path:
        path_coords = nodes[path]
        ax.plot(path_coords[:, 1], path_coords[:, 0], color="#00ff88", linewidth=3, zorder=6)

    ax.scatter(nodes[start_idx, 1], nodes[start_idx, 0], c="#00d4aa", s=150, marker="*", zorder=8, label="Start")
    ax.scatter(nodes[goal_idx, 1], nodes[goal_idx, 0], c="#ff6b35", s=150, marker="*", zorder=8, label="Goal")

    ax.set_title("PRM Roadmap + A* Path", color="white", fontsize=13)
    ax.legend(labelcolor="white", framealpha=0.2, fontsize=10)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {out}")

    if show:
        plt.show()

    plt.close(fig)
