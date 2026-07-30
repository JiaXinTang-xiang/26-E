from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BACKGROUND_BGR,
    DIVIDER_BGR,
    SPLIT_Y_MM,
    TARGET_CENTER_MM,
)
from .geometry import normalize_angle, polygon_centroid
from .image_io import write_image
from .models import DetectedPiece, SolvedLayout
from .solver import placement_rotation_deg


def _piece_affine_px(
    piece: DetectedPiece,
    final_rotation: np.ndarray,
    final_translation: np.ndarray,
    scale: float,
) -> np.ndarray:
    affine = np.zeros((2, 3), dtype=np.float64)
    affine[:, :2] = final_rotation
    affine[:, 2] = scale * (
        final_translation - final_rotation @ piece.center_mm
    )
    return affine


def _redraw_divider(image: np.ndarray, scale: float) -> None:
    split_px = int(round(SPLIT_Y_MM * scale))
    cv2.line(
        image,
        (0, split_px),
        (image.shape[1] - 1, split_px),
        DIVIDER_BGR,
        max(1, int(round(2.0 * scale))),
        cv2.LINE_AA,
    )


def build_movement_plan(
    pieces: list[DetectedPiece], layout: SolvedLayout
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for sequence, piece_index in enumerate(sorted(layout.aligned_polygons_mm), start=1):
        piece = pieces[piece_index]
        rotation = layout.final_rotations[piece_index]
        translation = layout.final_translations[piece_index]
        target_center = polygon_centroid(layout.aligned_polygons_mm[piece_index])
        target_pick = rotation @ (piece.pick_point_mm - piece.center_mm) + translation
        rotation_delta = placement_rotation_deg(rotation)
        target_angle = normalize_angle(piece.angle_deg + rotation_delta)
        records.append(
            {
                "id": piece.id,
                "source_center_mm": np.round(piece.center_mm, 4).tolist(),
                "pick_point_mm": np.round(piece.pick_point_mm, 4).tolist(),
                "source_angle_deg": round(float(piece.angle_deg), 4),
                "target_center_mm": np.round(target_center, 4).tolist(),
                "target_pick_point_mm": np.round(target_pick, 4).tolist(),
                "target_angle_deg": round(float(target_angle), 4),
                "translation_mm": np.round(target_center - piece.center_mm, 4).tolist(),
                "rotation_deg": round(float(rotation_delta), 4),
                "sequence": sequence,
            }
        )
    return {
        "a4_size_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
        "mode": layout.mode,
        "target_rect": {
            "center_mm": list(TARGET_CENTER_MM),
            "width_mm": round(float(layout.width_mm), 4),
            "height_mm": round(float(layout.height_mm), 4),
        },
        "quality": {
            "geometry_score": round(float(layout.geometry_score), 6),
            "texture_score": round(float(layout.texture_score), 6),
        },
        "pieces": records,
    }


def render_movement(
    input_image: np.ndarray,
    pieces: list[DetectedPiece],
    layout: SolvedLayout,
    output_dir: Path | str,
    scale: float,
    detected_image: np.ndarray,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    steps_dir = output_dir / "movement_steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    steps_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "detected.png", detected_image)
    write_image(steps_dir / "step_00.png", detected_image)

    current = input_image.copy()
    for sequence, piece_index in enumerate(sorted(layout.aligned_polygons_mm), start=1):
        piece = pieces[piece_index]
        removal_radius_px = max(3, int(round(1.8 * scale)))
        removal = cv2.dilate(
            piece.mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * removal_radius_px + 1,) * 2,
            ),
        )
        current[removal > 0] = BACKGROUND_BGR
        _redraw_divider(current, scale)

        affine = _piece_affine_px(
            piece,
            layout.final_rotations[piece_index],
            layout.final_translations[piece_index],
            scale,
        )
        width, height = current.shape[1], current.shape[0]
        warped_image = cv2.warpAffine(
            input_image,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BACKGROUND_BGR,
        )
        warped_mask = cv2.warpAffine(
            piece.mask,
            affine,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        current[warped_mask > 0] = warped_image[warped_mask > 0]

        annotated = current.copy()
        source = tuple(np.round(piece.center_mm * scale).astype(int))
        target_center = polygon_centroid(layout.aligned_polygons_mm[piece_index])
        target = tuple(np.round(target_center * scale).astype(int))
        cv2.arrowedLine(
            annotated,
            source,
            target,
            (50, 190, 255),
            3,
            cv2.LINE_AA,
            tipLength=0.035,
        )
        cv2.putText(
            annotated,
            f"move P{piece.id}: {sequence}/{len(pieces)}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (45, 48, 52),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"move P{piece.id}: {sequence}/{len(pieces)}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (40, 150, 235),
            2,
            cv2.LINE_AA,
        )
        write_image(steps_dir / f"step_{sequence:02d}.png", annotated)

    _redraw_divider(current, scale)
    write_image(output_dir / "solved.png", current)
    plan = build_movement_plan(pieces, layout)
    (output_dir / "movement_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return plan
