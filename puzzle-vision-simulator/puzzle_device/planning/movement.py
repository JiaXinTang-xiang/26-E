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


def _polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the minimum Euclidean distance between two polygon boundaries."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    minimum = math.inf
    for polygon_a, polygon_b in ((first, second), (second, first)):
        for point in polygon_a:
            for index, start in enumerate(polygon_b):
                end = polygon_b[(index + 1) % len(polygon_b)]
                edge = end - start
                length_squared = float(edge @ edge)
                if length_squared <= 1e-12:
                    distance = float(np.linalg.norm(point - start))
                else:
                    fraction = float(np.clip((point - start) @ edge / length_squared, 0.0, 1.0))
                    distance = float(np.linalg.norm(point - (start + fraction * edge)))
                minimum = min(minimum, distance)
    return minimum


def _placement_geometry(
    pieces: list,
    assembly: AssemblyPlan,
    config: AssemblyConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], float, float]:
    """Expand the ideal assembly radially to create a small non-overlap margin."""
    source_a4 = [
        global_pixels_to_a4(np.asarray(piece.polygon, dtype=np.float64), assembly.roi, config)
        for piece in pieces
    ]
    ideal = [
        np.c_[polygon, np.ones(len(polygon))] @ transform.T
        for polygon, transform in zip(source_a4, assembly.transforms)
    ]
    ideal = [polygon[:, :2] for polygon in ideal]
    if len(ideal) <= 1 or config.placement_gap_mm <= 0:
        return ideal, [np.zeros(2, dtype=np.float64) for _ in ideal], 0.0, 0.0

    centers = np.asarray([polygon.mean(axis=0) for polygon in ideal], dtype=np.float64)
    assembly_center = np.vstack(ideal).mean(axis=0)
    radial = centers - assembly_center
    maximum_radial_distance = max(float(np.linalg.norm(vector)) for vector in radial)
    if maximum_radial_distance <= 1e-6:
        raise ValueError("cannot create placement gap because a piece is centred on the assembly")

    neighbour_pairs = [(int(match[1]), int(match[3])) for match in assembly.matches]
    if not neighbour_pairs:
        return ideal, [np.zeros(2, dtype=np.float64) for _ in ideal], 0.0, 0.0

    def expanded(scale: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        offsets = [scale * vector for vector in radial]
        return [polygon + offset for polygon, offset in zip(ideal, offsets)], offsets

    def minimum_gap(polygons: list[np.ndarray]) -> float:
        return min(_polygon_distance(polygons[first], polygons[second])
                   for first, second in neighbour_pairs)

    high = config.maximum_piece_offset_mm / maximum_radial_distance
    high_polygons, high_offsets = expanded(high)
    if minimum_gap(high_polygons) + 1e-6 < config.placement_gap_mm:
        raise ValueError(
            f"cannot create {config.placement_gap_mm:.1f} mm placement gap within "
            f"the {config.maximum_piece_offset_mm:.1f} mm piece-offset limit"
        )
    low = 0.0
    for _ in range(32):
        middle = (low + high) / 2.0
        polygons, _offsets = expanded(middle)
        if minimum_gap(polygons) >= config.placement_gap_mm:
            high = middle
        else:
            low = middle
    placed, offsets = expanded(high)

    maximum_vertex_distance = 0.0
    for match in assembly.matches:
        first, second = int(match[1]), int(match[3])
        maximum_vertex_distance = max(
            maximum_vertex_distance,
            float(np.linalg.norm(offsets[first] - offsets[second])),
        )
    if maximum_vertex_distance > config.maximum_corresponding_vertex_distance_mm + 1e-6:
        raise ValueError(
            f"placement expansion separates corresponding vertices by "
            f"{maximum_vertex_distance:.1f} mm, exceeding 20 mm"
        )

    margin = 2.0
    bounds = np.vstack(placed)
    if (
        bounds[:, 0].min() < margin
        or bounds[:, 0].max() > config.a4_width_mm - margin
        or bounds[:, 1].min() < assembly.split_y_mm + margin
        or bounds[:, 1].max() > config.a4_height_mm - margin
    ):
        raise ValueError("placement gap would move a piece outside the lower A4 work area")
    return placed, offsets, minimum_gap(placed), maximum_vertex_distance


def target_rectangle_pixels(
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    target_rect_mm: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float]:
    """Return the lower-half target rectangle in global image pixels."""
    cfg = config or AssemblyConfig()
    if target_rect_mm is None:
        split_y = cfg.a4_height_mm * cfg.split_fraction
        center = np.array([
            cfg.a4_width_mm / 2.0,
            split_y + (cfg.a4_height_mm - split_y) / 2.0,
        ])
        top_left = center - [cfg.target_width_mm / 2.0, cfg.target_height_mm / 2.0]
        bottom_right = center + [cfg.target_width_mm / 2.0, cfg.target_height_mm / 2.0]
    else:
        x, y, width, height = target_rect_mm
        top_left = np.array([x, y], dtype=np.float64)
        bottom_right = top_left + [width, height]
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

    target_polygons_a4, target_offsets_a4, actual_gap, maximum_vertex_distance = (
        _placement_geometry(pieces, assembly, cfg)
    )

    records = []
    pixel_scale = np.asarray([
        assembly.roi[2] / cfg.a4_width_mm,
        assembly.roi[3] / cfg.a4_height_mm,
    ])
    for sequence, (piece, transform, target_polygon_a4, target_offset_a4) in enumerate(
        zip(pieces, assembly.transforms, target_polygons_a4, target_offsets_a4), start=1
    ):
        source_center_px = np.asarray(piece.center, dtype=np.float64)
        source_pick_px = np.asarray(piece.pick_point, dtype=np.float64)
        target_center_px = transform_global_points(
            source_center_px.reshape(1, 2), assembly, transform, cfg
        )[0] + target_offset_a4 * pixel_scale
        target_pick_px = transform_global_points(
            source_pick_px.reshape(1, 2), assembly, transform, cfg
        )[0] + target_offset_a4 * pixel_scale
        target_polygon_px = a4_to_global_pixels(target_polygon_a4, assembly.roi, cfg)
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
            "target_offset_a4_mm": _xy(target_offset_a4),
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
    target_px = target_rectangle_pixels(assembly.roi, cfg, assembly.target_rect_mm)
    target_x, target_y, target_width, target_height = assembly.target_rect_mm
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
            "center_a4_mm": _xy([
                target_x + target_width / 2.0,
                target_y + target_height / 2.0,
            ]),
            "size_mm": _xy([target_width, target_height]),
            "rect_px": _xy(target_px),
        },
        "quality": {
            "geometry_verified": True,
            "texture_verified": assembly.texture_score is not None,
            "texture_score": (
                None if assembly.texture_score is None
                else round(float(assembly.texture_score), 6)
            ),
            "texture_seam_scores": [
                round(float(score), 6) for score in assembly.texture_seam_scores
            ],
            "geometry_score": round(float(assembly.score), 6),
            "recovered_size_mm": _xy(assembly.recovered_size_mm),
            "target_size_range_mm": {
                "width": [cfg.minimum_target_width_mm, cfg.maximum_target_width_mm],
                "height": [cfg.minimum_target_height_mm, cfg.maximum_target_height_mm],
            },
            "target_size_tolerance_mm": round(float(cfg.target_size_tolerance_mm), 3),
            "recovered_size_in_range": True,
            "rectangle_fill_ratio": round(float(assembly.rectangle_fill_ratio), 6),
            "union_convexity_ratio": round(float(assembly.union_convexity_ratio), 6),
            "hull_rectangle_ratio": round(float(assembly.hull_rectangle_ratio), 6),
            "overlap_ratio": round(float(assembly.overlap_ratio), 6),
            "dimension_error_ratio": round(float(assembly.dimension_error_ratio), 6),
            "candidate_match_count": len(assembly.matches),
            "placement_gap_requested_mm": round(float(cfg.placement_gap_mm), 3),
            "placement_gap_actual_mm": round(float(actual_gap), 3),
            "maximum_corresponding_vertex_distance_mm": round(
                float(maximum_vertex_distance), 3
            ),
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
    target_polygons_a4, target_offsets_a4, _actual_gap, _maximum_vertex_distance = (
        _placement_geometry(pieces, assembly, cfg)
    )
    target_polygons = [
        np.round(a4_to_global_pixels(polygon, assembly.roi, cfg)).astype(np.int32)
        for polygon in target_polygons_a4
    ]
    target_x, target_y, target_width, target_height = target_rectangle_pixels(
        assembly.roi, cfg, assembly.target_rect_mm
    )
    target_box = np.round(np.array([
        [target_x, target_y],
        [target_x + target_width, target_y],
        [target_x + target_width, target_y + target_height],
        [target_x, target_y + target_height],
    ])).astype(np.int32)
    cv2.polylines(output, [target_box], True, (80, 255, 120), 2, cv2.LINE_AA)
    label_origin = (int(target_x) + 5, max(18, int(target_y) - 7))
    cv2.putText(
        output,
        f"{assembly.recovered_size_mm[0]:.1f} x {assembly.recovered_size_mm[1]:.1f} mm",
        label_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (80, 255, 120),
        2,
        cv2.LINE_AA,
    )
    if assembly.texture_score is not None:
        cv2.putText(
            output,
            f"texture seam: {assembly.texture_score:.3f} (lower is better)",
            (label_origin[0], label_origin[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 255, 120),
            1,
            cv2.LINE_AA,
        )
    for index, polygon in enumerate(target_polygons):
        cv2.fillPoly(fill, [polygon], colors[index % len(colors)])
    output = cv2.addWeighted(fill, 0.24, output, 0.76, 0.0)

    pixel_scale = np.asarray([
        assembly.roi[2] / cfg.a4_width_mm,
        assembly.roi[3] / cfg.a4_height_mm,
    ])
    for sequence, (piece, polygon, transform, target_offset_a4) in enumerate(
        zip(pieces, target_polygons, assembly.transforms, target_offsets_a4), start=1
    ):
        color = colors[(sequence - 1) % len(colors)]
        cv2.polylines(output, [polygon], True, color, 2, cv2.LINE_AA)
        target_pick = transform_global_points(
            np.asarray(piece.pick_point).reshape(1, 2), assembly, transform, cfg
        )[0] + target_offset_a4 * pixel_scale
        source = tuple(np.round(piece.pick_point).astype(int))
        target = tuple(np.round(target_pick).astype(int))
        cv2.arrowedLine(output, source, target, color, 2, cv2.LINE_AA, tipLength=0.04)
        cv2.drawMarker(output, target, color, cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(output, f"P{piece.piece_id} #{sequence}", target,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return output
