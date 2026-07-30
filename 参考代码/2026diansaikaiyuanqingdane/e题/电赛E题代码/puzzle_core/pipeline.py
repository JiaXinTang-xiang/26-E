from __future__ import annotations

from pathlib import Path

import cv2

from .config import A4_WIDTH_MM
from .image_io import read_image
from .simulation import render_movement
from .solver import solve_layout
from .vision import classify_mode, detect_pieces, draw_detection


def solve_image(
    input_path: Path | str,
    output_dir: Path | str,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Cannot read input image: {input_path}")
    image = read_image(input_path, cv2.IMREAD_COLOR)
    scale = image.shape[1] / A4_WIDTH_MM
    pieces, _ = detect_pieces(image, scale)
    mode = classify_mode(image, pieces)
    layout = solve_layout(pieces, mode, scale)
    detected = draw_detection(image, pieces, mode, scale)
    plan = render_movement(
        image,
        pieces,
        layout,
        output_dir,
        scale,
        detected,
    )
    plan["input"] = str(input_path.resolve())
    plan["output"] = str(output_dir.resolve())
    plan["detected_count"] = len(pieces)
    return plan
