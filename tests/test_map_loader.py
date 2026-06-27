import cv2
import numpy as np
import pytest

from src.map_loader import apply_clearance, load_map


def test_load_map_returns_binary_image(tmp_path):
    image = np.array(
        [
            [0, 100, 255],
            [30, 180, 240],
            [0, 255, 255],
        ],
        dtype=np.uint8,
    )

    image_path = tmp_path / "test_map.png"
    cv2.imwrite(str(image_path), image)

    loaded = load_map(image_path)

    assert loaded.shape == image.shape
    assert set(np.unique(loaded)).issubset({0, 255})


def test_load_map_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        load_map(missing_path)


def test_apply_clearance_zero_returns_same_map():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)
    obstacle_map[10, 10] = 0

    cleared = apply_clearance(obstacle_map, clearance=0)

    assert np.array_equal(cleared, obstacle_map)
    assert cleared is not obstacle_map


def test_apply_clearance_reduces_free_space():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)
    obstacle_map[10, 10] = 0

    cleared = apply_clearance(obstacle_map, clearance=2)

    original_free_pixels = np.count_nonzero(obstacle_map == 255)
    cleared_free_pixels = np.count_nonzero(cleared == 255)

    assert cleared_free_pixels < original_free_pixels


def test_apply_clearance_keeps_binary_values():
    obstacle_map = np.full((20, 20), 255, dtype=np.uint8)
    obstacle_map[8:12, 8:12] = 0

    cleared = apply_clearance(obstacle_map, clearance=2)

    assert set(np.unique(cleared)).issubset({0, 255})