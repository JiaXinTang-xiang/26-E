"""Plan requirement 1(1) using four fixed lower-half drop points."""

from __future__ import annotations

from datetime import datetime
import itertools
from typing import Callable

import cv2
import numpy as np

from .assembly import AssemblyConfig, a4_to_global_pixels


PulseMapper = Callable[[tuple[float, float]], tuple[int, int] | None]
FIXED_DROP_POINTS_A4_MM = (
    (35.0, 211.0),
    (76.0, 211.0),
    (117.0, 211.0),
    (158.0, 211.0),
)


def _xy(values, digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    first32 = np.asarray(first, dtype=np.float32)
    second32 = np.asarray(second, dtype=np.float32)
    if cv2.isContourConvex(first32) and cv2.isContourConvex(second32):
        area, _polygon = cv2.intersectConvexConvex(first32, second32)
        return float(area)
    minimum = np.floor(np.vstack([first32, second32]).min(axis=0) - 1).astype(int)
    maximum = np.ceil(np.vstack([first32, second32]).max(axis=0) + 1).astype(int)
    shape = np.maximum(maximum - minimum + 3, 3)
    masks = []
    for polygon in (first32, second32):
        mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon - minimum).astype(np.int32)], 1)
        masks.append(mask)
    return float(np.count_nonzero(masks[0] & masks[1]))


def _rotate_about(point: np.ndarray, center: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image points around *center*; positive angles are clockwise."""
    angle = np.deg2rad(float(angle_deg))
    matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return (np.asarray(point, dtype=np.float64) - center) @ matrix.T + center


def _canonical_rotation(angle_deg: float) -> float:
    """Return a servo-friendly equivalent in the range [-180, 180]."""
    value = ((float(angle_deg) + 180.0) % 360.0) - 180.0
    return 180.0 if abs(value + 180.0) < 1e-6 else value


def _fixed_assignment(
    pieces: list,
    fixed_points_px: np.ndarray,
    roi: tuple[int, int, int, int],
    split_y_px: float,
) -> tuple[tuple[int, ...], list[np.ndarray], list[np.ndarray], list[float]]:
    roi_x, roi_y, roi_width, roi_height = roi
    best = None
    for assignment in itertools.permutations(range(len(pieces))):
        for angle_choices in itertools.product((0.0, 90.0, 180.0, 270.0), repeat=len(pieces)):
            polygons = []
            fitted_points = []
            rotations = []
            outside = 0
            rotation_penalty = 0.0
            fitting_penalty = 0.0
            for fixed_point, piece_index, angle_deg in zip(
                fixed_points_px, assignment, angle_choices
            ):
                piece = pieces[piece_index]
                pick = np.asarray(piece.pick_point, dtype=np.float64)
                polygon = _rotate_about(
                    np.asarray(piece.polygon, dtype=np.float64), pick, angle_deg
                )
                delta = polygon - pick
                # The nominal point is the intended drop location.  If the
                # detected safe pick point is close to an edge, clamp only the
                # target point needed to keep the whole rigid polygon inside
                # the lower A4 region.
                min_point = np.array((roi_x, split_y_px), dtype=np.float64) - delta.min(axis=0)
                max_point = np.array((roi_x + roi_width, roi_y + roi_height), dtype=np.float64) - delta.max(axis=0)
                fitted = np.minimum(
                    np.maximum(np.asarray(fixed_point, dtype=np.float64), min_point),
                    max_point,
                )
                target = polygon + fitted - pick
                fitted_points.append(fitted)
                rotations.append(_canonical_rotation(angle_deg))
                fitting_penalty += float(np.linalg.norm(fitted - fixed_point)) * 0.10
                polygons.append(target)
                outside += int(np.any(
                    (target[:, 0] < roi_x)
                    | (target[:, 0] > roi_x + roi_width)
                    | (target[:, 1] < split_y_px)
                    | (target[:, 1] > roi_y + roi_height)
                ))
                rotation_penalty += abs(_canonical_rotation(angle_deg)) * 0.01
            overlap = sum(
                _intersection_area(polygons[first], polygons[second])
                for first, second in itertools.combinations(range(len(polygons)), 2)
            )
            score = outside * 1_000_000.0 + overlap + rotation_penalty + fitting_penalty
            if best is None or score < best[0]:
                best = (score, assignment, polygons, outside, overlap, fitted_points, rotations)
    if best is None or best[3] > 0:
        raise ValueError("4个固定放置点无法容纳当前碎片，请检查ROI和碎片摆放")
    if best[4] > 1.0:
        raise ValueError(f"4个固定放置点会造成碎片重叠 {best[4]:.0f} px²")
    return tuple(best[1]), best[2], best[5], best[6]


def build_transfer_plan(
    pieces: list,
    roi: tuple[int, int, int, int],
    pulse_mapper: PulseMapper,
    calibration_file: str | None = None,
    config: AssemblyConfig | None = None,
) -> tuple[dict[str, object], list[np.ndarray]]:
    """Assign four pieces to four fixed lower-half points without assembly."""
    cfg = config or AssemblyConfig()
    if len(pieces) != 4:
        raise ValueError("题目1（1）要求识别4块自备碎片")
    roi_x, roi_y, roi_width, roi_height = (int(value) for value in roi)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("A4 ROI无效")
    split_y_px = roi_y + roi_height * cfg.split_fraction
    for piece in pieces:
        polygon = np.asarray(piece.polygon, dtype=np.float64)
        if polygon[:, 1].max() > split_y_px + 1.0:
            raise ValueError(f"P{piece.piece_id}没有完全位于A4上半区")

    fixed_points_px = a4_to_global_pixels(
        np.asarray(FIXED_DROP_POINTS_A4_MM, dtype=np.float64), roi, cfg
    )
    assignment, target_polygons, fitted_points, rotations = _fixed_assignment(
        pieces, fixed_points_px, roi, split_y_px
    )
    records = []
    ordered_targets: list[np.ndarray] = []
    for sequence, (fixed_point, piece_index, target_polygon, rotation_deg) in enumerate(
        zip(fitted_points, assignment, target_polygons, rotations), start=1
    ):
        piece = pieces[piece_index]
        source_center = np.asarray(piece.center, dtype=np.float64)
        source_pick = np.asarray(piece.pick_point, dtype=np.float64)
        rotated_center = _rotate_about(source_center, source_pick, rotation_deg)
        target_center = rotated_center + fixed_point - source_pick
        source_pulse = pulse_mapper(tuple(source_pick))
        target_pulse = pulse_mapper(tuple(fixed_point))
        records.append({
            "piece_id": int(piece.piece_id),
            "sequence": sequence,
            "fixed_drop_index": sequence - 1,
            "source_center_px": _xy(source_center),
            "source_pick_px": _xy(source_pick),
            "source_pick_pulse": None if source_pulse is None else list(source_pulse),
            "target_center_px": _xy(target_center),
            "target_pick_px": _xy(fixed_point),
            "target_pick_pulse": None if target_pulse is None else list(target_pulse),
            "target_polygon_px": [_xy(point) for point in target_polygon],
            "source_angle_deg": round(float(piece.pca_angle_deg), 3),
            "target_angle_deg": round(float(piece.pca_angle_deg + rotation_deg), 3),
            "rotation_deg": round(float(rotation_deg), 3),
        })
        ordered_targets.append(target_polygon)

    document = {
        "format": "puzzle-device.transfer-movement-plan.v2",
        "created_local": datetime.now().astimezone().isoformat(),
        "operation_mode": "transfer_only",
        "motor_commands_sent": False,
        "rotation_axis_controlled": True,
        "calibration_file": calibration_file,
        "full_a4_roi_px": list(roi),
        "source_region_px": [roi_x, roi_y, roi_width, round(split_y_px - roi_y)],
        "target_region_px": [
            roi_x, round(split_y_px), roi_width, round(roi_y + roi_height - split_y_px),
        ],
        "quality": {
            "geometry_verified": True,
            "transfer_only": True,
            "piece_count": len(records),
            "fixed_drop_points_a4_mm": [list(point) for point in FIXED_DROP_POINTS_A4_MM],
            "fixed_drop_points_px": [_xy(point) for point in fixed_points_px],
            "fitted_drop_points_px": [_xy(point) for point in fitted_points],
            "pose_preserved": False,
            "rotation_candidates_deg": [0.0, 90.0, 180.0, -90.0],
        },
        "pieces": records,
    }
    return document, ordered_targets


def draw_transfer_preview(
    image: np.ndarray,
    pieces: list,
    target_polygons: list[np.ndarray],
) -> np.ndarray:
    """Draw fixed target outlines; pieces follow plan sequence order."""
    output = image.copy()
    fill = output.copy()
    colors = ((38, 142, 255), (66, 190, 95), (210, 110, 60), (180, 80, 205))
    rounded = [np.round(polygon).astype(np.int32) for polygon in target_polygons]
    for index, polygon in enumerate(rounded):
        cv2.fillPoly(fill, [polygon], colors[index % len(colors)])
    output = cv2.addWeighted(fill, 0.22, output, 0.78, 0.0)
    for index, polygon in enumerate(rounded, start=1):
        color = colors[(index - 1) % len(colors)]
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        target = tuple(np.round(target_polygons[index - 1].mean(axis=0)).astype(int))
        cv2.drawMarker(output, target, color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(
            output, f"fixed {index}", target,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return output
