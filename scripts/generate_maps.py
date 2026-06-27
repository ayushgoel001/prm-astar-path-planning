"""
generate_maps.py — Programmatically generate additional test maps.

Run once to create examples/maze_2.png and examples/maze_3.png.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2


def make_grid_world(
    rows: int = 500,
    cols: int = 500,
    cell_size: int = 50,
    wall_thickness: int = 6,
    gap_fraction: float = 0.4,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a grid-world map with random gaps in each wall segment.

    Args:
        rows, cols: Image dimensions.
        cell_size: Size of each grid cell in pixels.
        wall_thickness: Thickness of walls in pixels.
        gap_fraction: Fraction of each wall segment that is open.
        seed: Random seed.

    Returns:
        Binary uint8 image (255 = free, 0 = obstacle).
    """
    rng = np.random.default_rng(seed)
    img = np.full((rows, cols), 255, dtype=np.uint8)

    half = wall_thickness // 2

    for r in range(cell_size, rows, cell_size):
        gap_size  = int(cell_size * gap_fraction)
        gap_start = rng.integers(0, cell_size - gap_size, endpoint=True)

        for c in range(0, cols, cell_size):
            seg_start = c
            seg_end   = min(c + cell_size, cols)
            wall_clear_start = c + gap_start
            wall_clear_end   = c + gap_start + gap_size
            for cc in range(seg_start, seg_end):
                if cc < wall_clear_start or cc >= wall_clear_end:
                    img[max(0, r - half): min(rows, r + half), cc] = 0

    for c in range(cell_size, cols, cell_size):
        gap_size  = int(cell_size * gap_fraction)
        gap_start = rng.integers(0, cell_size - gap_size, endpoint=True)

        for r in range(0, rows, cell_size):
            seg_start = r
            seg_end = min(r + cell_size, rows)
            wall_clear_start = r + gap_start
            wall_clear_end = r + gap_start + gap_size
            for rr in range(seg_start, seg_end):
                if rr < wall_clear_start or rr >= wall_clear_end:
                    img[rr, max(0, c - half): min(cols, c + half)] = 0

    return img


def make_random_obstacle_field(
    rows: int = 500,
    cols: int = 500,
    num_obstacles: int = 30,
    min_size: int = 20,
    max_size: int = 70,
    seed: int = 1,
) -> np.ndarray:
    """
    Generate a free-space map with randomly placed rectangular obstacles.

    Args:
        rows, cols: Image dimensions.
        num_obstacles: Number of rectangular obstacles.
        min_size, max_size: Min/max side length of each obstacle.
        seed: Random seed.

    Returns:
        Binary uint8 image (255 = free, 0 = obstacle).
    """
    rng = np.random.default_rng(seed)
    img = np.full((rows, cols), 255, dtype=np.uint8)

    margin = 40
    for _ in range(num_obstacles):
        h = int(rng.integers(min_size, max_size))
        w = int(rng.integers(min_size, max_size))
        r = int(rng.integers(margin, rows - margin - h))
        c = int(rng.integers(margin, cols - margin - w))
        img[r : r + h, c : c + w] = 0

    return img


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "examples"
    out_dir.mkdir(exist_ok=True)

    grid_map = make_grid_world(seed=7)
    cv2.imwrite(str(out_dir / "maze_2.png"), grid_map)
    print(f"Saved: {out_dir / 'maze_2.png'}")

    random_map = make_random_obstacle_field(seed=3)
    cv2.imwrite(str(out_dir / "maze_3.png"), random_map)
    print(f"Saved: {out_dir / 'maze_3.png'}")
