"""Convert an assembly solution into a previewable gantry movement plan."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Callable, Iterable

import cv2
import numpy as np

from .assembly import (
    AssemblyConfig,
    AssemblyPlan,
    a4_to_global_pixels,
    global_pixels_to_a4,
    transform_global_points,
)


PulseMapper = Callable[[tuple[float, float]], tuple[int, int] | None]


def _xy(values: Iterable[float], digits: int = 3) -> list[float]:
    return [round(float(value), digits) for value in values]


def _rotation_degrees(transform: np.ndarray) -> float:
    angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
    return (angle + 180.0) % 360.0 - 180.0


def target_rectangle_pixels(
    roi: tuple[int, int, int, int], config: AssemblyConfig | None = None
) -> tuple[float, float, float, float]:
    """Return the lower-half target rectangle in global image pixels."""
    cfg = config or AssemblyConfig()
    split_y = cfg.a4_height_mm * cfg.split_fraction
    center = np.array([
        cfg.a4_width_mm / 2.0,
        split_y + (cfg.a4_height_mm - split_y) / 2.0,
    ])
    top_left = center - [cfg.target_width_mm / 2.0, cfg.target_height_mm / 2.0]
    bottom_right = center + [cfg.target_width_mm / 2.0, cfg.target_height_mm / 2.0]
    corners = a4_to_global_pixels(np.asarray([top_left, bottom_right]), roi, cfg)
    return (
        float(corners[0, 0]),
        float(corners[0, 1]),
        float(corners[1, 0] - corners[0, 0]),
        float(corners[1, 1] - corners[0, 1]),
    )


def build_movement_plan(
    pieces: list,
    assembly: AssemblyPlan,
    pulse_mapper: PulseMapper | None = None,
    calibration_file: str | None = None,
    config: AssemblyConfig | None = None,
) -> dict[str, object]:
    """Build a data-only pick/place plan; this function never sends commands."""
    cfg = config or AssemblyConfig()
    if len(pieces) != len(assembly.transforms):
        raise ValueError("碎片数量与拼接变换数量不一致")

    records = []
    for sequence, (piece, transform) in enumerate(
        zip(pieces, assembly.transforms), start=1
    ):
        source_center_px = np.asarray(piece.center, dtype=np.float64)
        source_pick_px = np.asarray(piece.pick_point, dtype=np.float64)
        target_center_px = transform_global_points(
            source_center_px.reshape(1, 2), assembly, transform, cfg
        )[0]
        target_pick_px = transform_global_points(
            source_pick_px.reshape(1, 2), assembly, transform, cfg
        )[0]
        target_polygon_px = transform_global_points(
            np.asarray(piece.polygon, dtype=np.float64), assembly, transform, cfg
        )
        source_center_mm = global_pixels_to_a4(
            source_center_px.reshape(1, 2), assembly.roi, cfg
        )[0]
        source_pick_mm = global_pixels_to_a4(
            source_pick_px.reshape(1, 2), assembly.roi, cfg
        )[0]
        target_center_mm = global_pixels_to_a4(
            target_center_px.reshape(1, 2), assembly.roi, cfg
        )[0]
        target_pick_mm = global_pixels_to_a4(
            target_pick_px.reshape(1, 2), assembly.roi, cfg
        )[0]
        rotation_deg = _rotation_degrees(transform)
        source_pulse = None if pulse_mapper is None else pulse_mapper(tuple(source_pick_px))
        target_pulse = None if pulse_mapper is None else pulse_mapper(tuple(target_pick_px))
        records.append({
            "piece_id": int(piece.piece_id),
            "source_center_px": _xy(source_center_px),
            "source_pick_px": _xy(source_pick_px),
            "source_pick_pulse": None if source_pulse is None else list(source_pulse),
            "source_center_a4_mm": _xy(source_center_mm),
            "source_pick_a4_mm": _xy(source_pick_mm),
            "source_angle_deg": round(float(piece.pca_angle_deg), 3),
            "target_center_px": _xy(target_center_px),
            "target_pick_px": _xy(target_pick_px),
            "target_pick_pulse": None if target_pulse is None else list(target_pulse),
            "target_center_a4_mm": _xy(target_center_mm),
            "target_pick_a4_mm": _xy(target_pick_mm),
            "target_polygon_px": [_xy(point) for point in target_polygon_px],
            "target_angle_deg": round(float(piece.pca_angle_deg + rotation_deg), 3),
            "rotation_deg": round(rotation_deg, 3),
            "transform_a4_mm": np.round(transform, 8).tolist(),
            "sequence": sequence,
        })

    matches = []
    for match in assembly.matches:
        penalty, first, first_edge, second, second_edge, fs, fe, ss, se = match
        matches.append({
            "piece_a": int(first),
            "edge_a": int(first_edge),
            "edge_a_fraction": [round(float(fs), 4), round(float(fe), 4)],
            "piece_b": int(second),
            "edge_b": int(second_edge),
            "edge_b_fraction": [round(float(ss), 4), round(float(se), 4)],
            "partial_edge": not (
                abs(fs) < 1e-6 and abs(fe - 1.0) < 1e-6
                and abs(ss) < 1e-6 and abs(se - 1.0) < 1e-6
            ),
            "penalty": round(float(penalty), 6),
        })

    roi_x, roi_y, roi_width, roi_height = assembly.roi
    split_y_px = roi_y + round(roi_height * cfg.split_fraction)
    target_px = target_rectangle_pixels(assembly.roi, cfg)
    return {
        "format": "puzzle-device.assembly-movement-plan.v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "motor_commands_sent": False,
        "rotation_axis_controlled": False,
        "calibration_file": calibration_file,
        "coordinate_system": {
            "pixels": "rotated camera frame; global x right, y down",
            "a4_mm": "A4 top-left origin; x right, y down",
            "pulses": "gantry absolute XY pulses after homing",
        },
        "a4_size_mm": [cfg.a4_width_mm, cfg.a4_height_mm],
        "full_a4_roi_px": list(assembly.roi),
        "source_region_px": [roi_x, roi_y, roi_width, split_y_px - roi_y],
        "target_region_px": [roi_x, split_y_px, roi_width, roi_y + roi_height - split_y_px],
        "target_rect": {
            "center_a4_mm": [cfg.a4_width_mm / 2.0,
                              cfg.a4_height_mm * (1.0 + cfg.split_fraction) / 2.0],
            "size_mm": [cfg.target_width_mm, cfg.target_height_mm],
            "rect_px": _xy(target_px),
        },
        "quality": {
            "geometry_verified": True,
            "geometry_score": round(float(assembly.score), 6),
            "recovered_size_mm": _xy(assembly.recovered_size_mm),
            "rectangle_fill_ratio": round(float(assembly.rectangle_fill_ratio), 6),
            "union_convexity_ratio": round(float(assembly.union_convexity_ratio), 6),
            "hull_rectangle_ratio": round(float(assembly.hull_rectangle_ratio), 6),
            "overlap_ratio": round(float(assembly.overlap_ratio), 6),
            "dimension_error_ratio": round(float(assembly.dimension_error_ratio), 6),
            "candidate_match_count": len(assembly.matches),
        },
        "matches": matches,
        "pieces": records,
    }


def draw_assembly_preview(
    image: np.ndarray,
    pieces: list,
    assembly: AssemblyPlan,
    config: AssemblyConfig | None = None,
) -> np.ndarray:
    """Draw target outlines and source-to-target arrows on a camera frame."""
    cfg = config or AssemblyConfig()
    output = image.copy()
    fill = output.copy()
    colors = ((38, 142, 255), (66, 190, 95), (210, 110, 60), (180, 80, 205))
    target_polygons = []
    for piece, transform in zip(pieces, assembly.transforms):
        target = transform_global_points(piece.polygon, assembly, transform, cfg)
        target_polygons.append(np.round(target).astype(np.int32))
    for index, polygon in enumerate(target_polygons):
        cv2.fillPoly(fill, [polygon], colors[index % len(colors)])
    output = cv2.addWeighted(fill, 0.24, output, 0.76, 0.0)

    for sequence, (piece, polygon, transform) in enumerate(
        zip(pieces, target_polygons, assembly.transforms), start=1
    ):
        color = colors[(sequence - 1) % len(colors)]
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        target_pick = transform_global_points(
            np.asarray(piece.pick_point).reshape(1, 2), assembly, transform, cfg
        )[0]
        source = tuple(np.round(piece.pick_point).astype(int))
        target = tuple(np.round(target_pick).astype(int))
        cv2.arrowedLine(output, source, target, color, 2, cv2.LINE_AA, tipLength=0.04)
        cv2.drawMarker(output, target, color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(output, f"P{piece.piece_id} #{sequence}", target,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output
