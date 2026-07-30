from __future__ import annotations

import math
from collections.abc import Iterable

import cv2
import numpy as np


def ensure_ccw(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if cv2.contourArea(pts.astype(np.float32), oriented=True) < 0:
        pts = pts[::-1].copy()
    return pts


def polygon_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(points, dtype=np.float32))))


def polygon_centroid(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    moments = cv2.moments(pts.astype(np.float32))
    if abs(moments["m00"]) < 1e-9:
        return pts.mean(axis=0)
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def edge_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)


def rotation_matrix(angle_deg: float) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )


def normalize_angle(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def longest_edge_angle(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    vectors = np.roll(pts, -1, axis=0) - pts
    idx = int(np.argmax(np.linalg.norm(vectors, axis=1)))
    return normalize_angle(math.degrees(math.atan2(vectors[idx, 1], vectors[idx, 0])))


def self_piece_polygons() -> tuple[list[np.ndarray], float, float]:
    width, height = 100.0, 60.0
    p0 = np.array([0.0, 0.0])
    p1 = np.array([20.0, 0.0])
    p2 = np.array([100.0, 0.0])
    p3 = np.array([100.0, 60.0])
    p4 = np.array([0.0, 60.0])
    left_a = np.array([0.0, 20.0])
    left_b = np.array([0.0, 30.0])
    diag_a = np.array([36.0, 12.0])
    diag_b = np.array([76.0, 42.0])
    pieces = [
        ensure_ccw(np.array([p0, p1, diag_a, left_a])),
        ensure_ccw(np.array([left_a, diag_a, diag_b, left_b])),
        ensure_ccw(np.array([left_b, diag_b, p3, p4])),
        ensure_ccw(np.array([p1, p2, p3, diag_b, diag_a])),
    ]
    return pieces, width, height


def perimeter_point(width: float, height: float, distance: float) -> np.ndarray:
    perimeter = 2.0 * (width + height)
    s = distance % perimeter
    if s <= width:
        return np.array([s, 0.0])
    if s <= width + height:
        return np.array([width, s - width])
    if s <= 2.0 * width + height:
        return np.array([2.0 * width + height - s, height])
    return np.array([0.0, perimeter - s])


def _sample_side_distance(
    rng: np.random.Generator, side: int, width: float, height: float
) -> float:
    lengths = [width, height, width, height]
    starts = [0.0, width, width + height, 2.0 * width + height]
    side_len = lengths[side]
    margin = 20.0
    if side_len < 2.0 * margin:
        raise ValueError("Rectangle side is too short for the 20 mm edge constraint")
    offset = float(rng.uniform(margin, side_len - margin))
    return starts[side] + offset


def fan_partition(
    width: float, height: float, count: int, rng: np.random.Generator
) -> list[np.ndarray]:
    if count == 1:
        return [
            np.array(
                [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
                dtype=np.float64,
            )
        ]
    if count not in (2, 3, 4):
        raise ValueError("Field piece count must be between 1 and 4")

    perimeter = 2.0 * (width + height)
    corners = [
        (0.0, np.array([0.0, 0.0])),
        (width, np.array([width, 0.0])),
        (width + height, np.array([width, height])),
        (2.0 * width + height, np.array([0.0, height])),
    ]

    for _ in range(500):
        if count == 2:
            side_sets = ([0, 2], [1, 3])
            sides = list(side_sets[int(rng.integers(0, len(side_sets)))])
        elif count == 3:
            missing = int(rng.integers(0, 4))
            sides = [side for side in range(4) if side != missing]
        else:
            sides = [0, 1, 2, 3]

        boundary_distances = sorted(
            _sample_side_distance(rng, side, width, height) for side in sides
        )
        hub = np.array(
            [
                float(rng.uniform(0.40 * width, 0.60 * width)),
                float(rng.uniform(0.40 * height, 0.60 * height)),
            ]
        )
        pieces: list[np.ndarray] = []
        for index, start in enumerate(boundary_distances):
            end = boundary_distances[(index + 1) % len(boundary_distances)]
            if end <= start:
                end += perimeter
            polygon = [hub.copy(), perimeter_point(width, height, start)]
            for shift in (0.0, perimeter, 2.0 * perimeter):
                for corner_s, corner_point in corners:
                    candidate_s = corner_s + shift
                    if start + 1e-8 < candidate_s < end - 1e-8:
                        polygon.append(corner_point.copy())
            polygon.append(perimeter_point(width, height, end))
            pieces.append(ensure_ccw(np.array(polygon, dtype=np.float64)))

        valid = True
        for polygon in pieces:
            if len(polygon) > 5 or np.min(edge_lengths(polygon)) < 19.999:
                valid = False
                break
            # Keep the common hub visibly different from a straight boundary
            # chord. Otherwise raster contour simplification can legitimately
            # collapse a very shallow concave hub into one long edge.
            hub_index = int(np.argmin(np.linalg.norm(polygon - hub, axis=1)))
            previous = polygon[hub_index - 1]
            following = polygon[(hub_index + 1) % len(polygon)]
            chord = following - previous
            offset = hub - previous
            cross_2d = chord[0] * offset[1] - chord[1] * offset[0]
            depth = abs(float(cross_2d)) / max(
                float(np.linalg.norm(chord)), 1e-9
            )
            if depth < 8.0:
                valid = False
                break
        if valid:
            return pieces
    raise RuntimeError("Unable to generate a well-conditioned fan partition")


def rasterized_overlap_area(
    polygons: Iterable[np.ndarray], scale: float = 3.0
) -> tuple[float, float, np.ndarray, tuple[float, float]]:
    polygons = [np.asarray(poly, dtype=np.float64) for poly in polygons]
    all_points = np.vstack(polygons)
    min_xy = np.floor(all_points.min(axis=0) - 2.0)
    max_xy = np.ceil(all_points.max(axis=0) + 2.0)
    size = np.maximum(np.ceil((max_xy - min_xy) * scale).astype(int) + 1, 2)
    masks: list[np.ndarray] = []
    for polygon in polygons:
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        pixels = np.round((polygon - min_xy) * scale).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 1)
        masks.append(mask)
    stacked = np.stack(masks, axis=0)
    union = np.any(stacked > 0, axis=0)
    sum_area = float(stacked.sum()) / (scale * scale)
    union_area = float(union.sum()) / (scale * scale)
    return sum_area - union_area, union_area, union.astype(np.uint8), tuple(min_xy)


def rigid_edge_alignment(
    source_a: np.ndarray,
    source_b: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_vec = np.asarray(source_b) - np.asarray(source_a)
    target_vec = np.asarray(target_b) - np.asarray(target_a)
    angle = math.atan2(target_vec[1], target_vec[0]) - math.atan2(
        source_vec[1], source_vec[0]
    )
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    translation = np.asarray(target_a, dtype=np.float64) - rotation @ np.asarray(
        source_a, dtype=np.float64
    )
    return rotation, translation
