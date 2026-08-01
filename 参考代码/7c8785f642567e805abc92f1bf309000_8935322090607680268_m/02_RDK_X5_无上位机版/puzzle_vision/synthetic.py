from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import (
    rigid_align,
    polygon_centroid,
    polygons_overlap,
    rotation_matrix_row,
    transform_points,
)
from .detector import DetectionError
from .pipeline import PuzzleVisionPipeline


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def _place_test_pieces(
    templates: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    centre_hints = [
        np.array([48.0, 36.0]),
        np.array([150.0, 43.0]),
        np.array([53.0, 108.0]),
        np.array([153.0, 110.0]),
    ]
    placed: list[np.ndarray] = []
    for index, template in enumerate(templates):
        source = template - polygon_centroid(template)
        accepted = None
        for attempt in range(200):
            angle = math.radians(float(rng.uniform(-42.0, 42.0)))
            centre = centre_hints[index] + rng.normal(0.0, 2.0, size=2)
            candidate = transform_points(
                source, rotation_matrix_row(angle), centre
            )
            lower = np.min(candidate, axis=0)
            upper = np.max(candidate, axis=0)
            if lower[0] < 5 or upper[0] > 205 or lower[1] < 5 or upper[1] > 142:
                continue
            if any(polygons_overlap(candidate, other, 0.2) for other in placed):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError(f"Cannot place synthetic piece {index}")
        placed.append(accepted)
    return placed


def make_synthetic_scene(
    config: dict[str, Any],
    seed: int = 7,
    paper_bgr: tuple[int, int, int] = (145, 68, 28),
    piece_bgr: tuple[int, int, int] = (40, 220, 250),
    table_bgr: tuple[int, int, int] = (190, 190, 185),
    divider_bgr: tuple[int, int, int] | None = (8, 8, 8),
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    ppm = float(config["paper"]["pixels_per_mm"])
    paper_width = int(round(float(config["paper"]["width_mm"]) * ppm))
    paper_height = int(round(float(config["paper"]["height_mm"]) * ppm))
    paper = np.full((paper_height, paper_width, 3), paper_bgr, np.uint8)
    # Add a very light paper texture so the test is not a perfectly flat image.
    noise = rng.normal(0.0, 1.5, paper.shape[:2]).astype(np.int16)
    paper = np.clip(paper.astype(np.int16) + noise[:, :, None], 0, 255).astype(
        np.uint8
    )
    divider_y = int(round(float(config["paper"]["divider_y_mm"]) * ppm))
    if divider_bgr is not None:
        cv2.line(
            paper,
            (0, divider_y),
            (paper_width - 1, divider_y),
            divider_bgr,
            10,
        )

    templates = [
        np.asarray(piece["vertices_mm"], dtype=np.float64)
        for piece in config["fixed"]["pieces"]
    ]
    pieces = _place_test_pieces(templates, rng)
    for piece in pieces:
        polygon_px = np.rint(piece * ppm).astype(np.int32)
        cv2.fillPoly(paper, [polygon_px], piece_bgr, cv2.LINE_AA)

    canvas = np.full((1040, 1280, 3), table_bgr, np.uint8)
    destination = np.array(
        [[330, 70], [920, 48], [958, 940], [290, 958]], dtype=np.float32
    )
    source = np.array(
        [
            [0, 0],
            [paper_width - 1, 0],
            [paper_width - 1, paper_height - 1],
            [0, paper_height - 1],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(paper, homography, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(
        np.full(paper.shape[:2], 255, np.uint8),
        homography,
        (canvas.shape[1], canvas.shape[0]),
    )
    canvas[mask > 0] = warped[mask > 0]
    cv2.polylines(
        canvas, [np.rint(destination).astype(np.int32)], True, (30, 30, 30), 3
    )
    return canvas, pieces


def run_self_test(
    config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scene, _ = make_synthetic_scene(config)
    _write_image(output / "synthetic_scene.jpg", scene)

    pipeline = PuzzleVisionPipeline(config)
    fixed_result, fixed_debug, mask, rectified = pipeline.analyze(scene, "fixed")
    _write_image(output / "fixed_debug.jpg", fixed_debug)
    _write_image(output / "foreground_mask.png", mask)
    _write_image(output / "rectified_paper.jpg", rectified)
    with (output / "fixed_result.json").open("w", encoding="utf-8") as handle:
        json.dump(fixed_result, handle, ensure_ascii=False, indent=2)

    unknown_result, unknown_debug, _, _ = pipeline.analyze(
        scene, "unknown-white"
    )
    _write_image(output / "unknown_debug.jpg", unknown_debug)
    with (output / "unknown_result.json").open("w", encoding="utf-8") as handle:
        json.dump(unknown_result, handle, ensure_ascii=False, indent=2)

    no_divider_scene, _ = make_synthetic_scene(
        config, seed=8, divider_bgr=None
    )
    try:
        pipeline.rectify(no_divider_scene)
        missing_divider_rejected = False
    except DetectionError:
        missing_divider_rejected = True

    probe = np.array([[0.0, 0.0], [31.0, 2.0], [7.0, 19.0]])
    mirrored_probe = probe.copy()
    mirrored_probe[:, 0] *= -1.0
    mirror_fit_rotation, _, _ = rigid_align(probe, mirrored_probe)
    proper_rotation_only = float(np.linalg.det(mirror_fit_rotation)) > 0.999

    checks = {
        "fixed_piece_count": len(fixed_result["pieces"]) == 4,
        "fixed_plan_count": len(fixed_result["plan"]) == 4,
        "divider_detected_near_half": abs(
            fixed_result["paper"]["divider"]["detected_y_mm"]
            - float(config["paper"]["divider_y_mm"])
        )
        < 3.0,
        "divider_width_valid": fixed_result["paper"]["divider"]["width_mm"]
        <= float(config["paper"]["divider_max_width_mm"]),
        "missing_divider_rejected": missing_divider_rejected,
        "fixed_match_error": fixed_result["solver"]["max_match_error_mm"]
        < float(config["fixed"]["max_match_error_mm"]),
        "fixed_target_fully_covered": (
            fixed_result["solver"]["target_coverage_ratio"] > 0.999
            and fixed_result["solver"]["target_corners_covered"] is True
        ),
        "unknown_plan_count": len(unknown_result["plan"]) == 4,
        "unknown_fill_ratio": unknown_result["solver"]["fill_ratio"] > 0.94,
        "fixed_mirror_forbidden": (
            fixed_result["solver"]["mirror_allowed"] is False
            and all(item["mirrored"] is False for item in fixed_result["plan"])
        ),
        "unknown_mirror_forbidden": (
            unknown_result["solver"]["mirror_allowed"] is False
            and all(item["mirrored"] is False for item in unknown_result["plan"])
        ),
        "proper_rotation_only": proper_rotation_only,
    }
    summary = {
        "ok": all(checks.values()),
        "checks": checks,
        "fixed_timing_ms": fixed_result["timing_ms"],
        "unknown_timing_ms": unknown_result["timing_ms"],
        "unknown_solver": unknown_result["solver"],
        "output_dir": str(output.resolve()),
    }
    with (output / "self_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if not summary["ok"]:
        raise AssertionError(f"Self-test failed: {checks}")
    return summary
