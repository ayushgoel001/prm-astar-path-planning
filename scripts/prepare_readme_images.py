"""Prepare consistently sized result images for the README."""

from pathlib import Path

from PIL import Image, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
CANVAS_SIZE = (1600, 900)
IMAGE_PAIRS = (
    ("prm_maze_hard.png", "prm_maze_hard_readme.png"),
    ("prm_vs_rrt_maze_hard.png", "prm_vs_rrt_maze_hard_readme.png"),
)


def prepare_image(source_path: Path, output_path: Path) -> None:
    """Fit an image onto a centered white canvas without changing its aspect ratio."""
    with Image.open(source_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")
        image.thumbnail(CANVAS_SIZE, Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", CANVAS_SIZE, "white")
        offset = (
            (CANVAS_SIZE[0] - image.width) // 2,
            (CANVAS_SIZE[1] - image.height) // 2,
        )
        canvas.alpha_composite(image, offset)
        canvas.convert("RGB").save(output_path, format="PNG", optimize=True)

    print(f"Generated {output_path.relative_to(ROOT_DIR)} ({CANVAS_SIZE[0]}x{CANVAS_SIZE[1]})")


def main() -> None:
    for source_name, output_name in IMAGE_PAIRS:
        prepare_image(ASSETS_DIR / source_name, ASSETS_DIR / output_name)


if __name__ == "__main__":
    main()
