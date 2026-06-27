"""
map_loader.py — Load and preprocess binary maze/map images.
"""

from pathlib import Path
import cv2
import numpy as np


def load_map(image_path: str | Path) -> np.ndarray:
    """
    Load a maze image and convert it to a binary obstacle map.

    Pixels above threshold become free space (255); all others are obstacles (0).

    Args:
        image_path: Path to the input image (PNG, JPG, etc.).

    Returns:
        Binary uint8 numpy array of shape (H, W):
          - 0   → obstacle
          - 255 → free space

    Raises:
        FileNotFoundError: If the image does not exist.
        ValueError: If the image cannot be read by OpenCV.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Map image not found: {path}")

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"OpenCV could not read the image: {path}")

    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return binary


def apply_clearance(obstacle_map: np.ndarray, clearance: int) -> np.ndarray:
    """
    Inflate obstacles by `clearance` pixels using morphological erosion.

    Shrinking free space is equivalent to inflating obstacles by the robot's
    footprint radius, ensuring sampled nodes and edges maintain a safe margin
    from all walls.

    Args:
        obstacle_map: Binary map (0 = obstacle, 255 = free).
        clearance: Number of pixels to inflate each obstacle by.

    Returns:
        New binary map with inflated obstacles.
    """
    if clearance <= 0:
        return obstacle_map.copy()

    kernel_size = 2 * clearance + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(obstacle_map, kernel, iterations=1)
