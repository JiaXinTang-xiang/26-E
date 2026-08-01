from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np


def signed_area(polygon: np.ndarray) -> float:
    p = np.asarray(polygon, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    cross_sum = float(
        np.dot(x[:-1], y[1:])
        - np.dot(y[:-1], x[1:])
        + x[-1] * y[0]
        - y[-1] * x[0]
    )
    return 0.5 * cross_sum


def polygon_area(polygon: np.ndarray) -> float:
    return abs(signed_area(polygon))


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    p = np.asarray(polygon, dtype=np.float64)
    x = p[:, 0]
    y = p[:, 1]
    cross = np.empty(len(p), dtype=np.float64)
    cross[:-1] = x[:-1] * y[1:] - y[:-1] * x[1:]
    cross[-1] = x[-1] * y[0] - y[-1] * x[0]
    a2 = float(np.sum(cross))
    if abs(a2) < 1e-8:
        return np.mean(p, axis=0)
    cx = (
        np.sum((x[:-1] + x[1:]) * cross[:-1])
        + (x[-1] + x[0]) * cross[-1]
    ) / (3.0 * a2)
    cy = (
        np.sum((y[:-1] + y[1:]) * cross[:-1])
        + (y[-1] + y[0]) * cross[-1]
    ) / (3.0 * a2)
    return np.array([cx, cy], dtype=np.float64)


def normalize_winding(polygon: np.ndarray, positive: bool = True) -> np.ndarray:
    p = np.asarray(polygon, dtype=np.float64)
    if (signed_area(p) > 0) != positive:
        p = p[::-1].copy()
    return p


def rotation_matrix_row(theta_rad: float) -> np.ndarray:
    """Row-vector rotation matrix.

    With image coordinates (x right, y down), positive theta is clockwise.
    """

    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def is_proper_rotation(rotation: np.ndarray, tolerance: float = 1e-5) -> bool:
    """Return True only for a 2-D rotation, never a reflection or scaling."""

    r = np.asarray(rotation, dtype=np.float64)
    return bool(
        r.shape == (2, 2)
        and np.allclose(r.T @ r, np.eye(2), atol=tolerance, rtol=0.0)
        and abs(float(np.linalg.det(r)) - 1.0) <= tolerance
    )


def transform_points(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ rotation + translation


def compose_transforms(
    first_r: np.ndarray,
    first_t: np.ndarray,
    second_r: np.ndarray,
    second_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose row transforms: apply first, then second."""

    return first_r @ second_r, first_t @ second_r + second_t


def invert_transform(
    rotation: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inv_r = rotation.T
    inv_t = -translation @ inv_r
    return inv_r, inv_t


def transform_angle_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def rigid_align(
    source: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Rigidly align equal-length ordered point lists without reflection."""

    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(destination, dtype=np.float64)
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    x = src - src_mean
    y = dst - dst_mean
    u, _, vt = np.linalg.svd(x.T @ y)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    if not is_proper_rotation(r):
        raise ValueError("Rigid alignment produced a reflected/non-rigid transform")
    t = dst_mean - src_mean @ r
    residual = transform_points(src, r, t) - dst
    rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return r, t, rms


def best_cyclic_alignment(
    source: np.ndarray, destination: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, int]:
    src = normalize_winding(source)
    dst = normalize_winding(destination)
    if len(src) != len(dst):
        raise ValueError("Polygons must have the same vertex count")
    best: tuple[np.ndarray, np.ndarray, float, int] | None = None
    for shift in range(len(dst)):
        ordered = np.roll(dst, -shift, axis=0)
        r, t, error = rigid_align(src, ordered)
        if best is None or error < best[2]:
            best = (r, t, error, shift)
    assert best is not None
    return best


def edge_lengths(polygon: np.ndarray) -> np.ndarray:
    p = np.asarray(polygon, dtype=np.float64)
    return np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)


def min_area_rectangle(
    polygons: Iterable[np.ndarray],
) -> tuple[np.ndarray, float, float, float]:
    points = np.vstack([np.asarray(p, np.float32) for p in polygons])
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float64)
    lengths = np.linalg.norm(np.roll(box, -1, axis=0) - box, axis=1)
    idx = int(np.argmax(lengths))
    start = box[idx]
    end = box[(idx + 1) % 4]
    theta = math.atan2(end[1] - start[1], end[0] - start[0])
    long_side = float(max(rect[1]))
    short_side = float(min(rect[1]))
    return box, long_side, short_side, theta


def safe_interior_point(polygon: np.ndarray, resolution_mm: float = 0.5) -> np.ndarray:
    """Approximate the pole of inaccessibility using a distance transform."""

    p = np.asarray(polygon, dtype=np.float64)
    lower = np.floor(np.min(p, axis=0) - 2.0)
    upper = np.ceil(np.max(p, axis=0) + 2.0)
    size = np.maximum(
        3, np.ceil((upper - lower) / resolution_mm).astype(np.int32) + 1
    )
    mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    local = np.rint((p - lower) / resolution_mm).astype(np.int32)
    cv2.fillPoly(mask, [local], 255)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, _, _, maximum = cv2.minMaxLoc(distance)
    return lower + np.array(maximum, dtype=np.float64) * resolution_mm


def segments_properly_intersect(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    epsilon: float,
) -> bool:
    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    o1 = cross(b - a, c - a)
    o2 = cross(b - a, d - a)
    o3 = cross(d - c, a - c)
    o4 = cross(d - c, b - c)
    return o1 * o2 < -(epsilon**2) and o3 * o4 < -(epsilon**2)


def polygon_intersection_area(
    first: np.ndarray,
    second: np.ndarray,
    pixels_per_mm: float = 2.0,
) -> float:
    """Return polygon intersection area without OpenCV convex-clip quirks.

    Contest pieces are normally convex.  For that fast path use a small
    Sutherland-Hodgman clip implemented in double precision.  It is invariant
    to translation and treats a shared seam as zero area.  Concave outlines
    use the deterministic local raster fallback.
    """

    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if (
        np.max(a[:, 0]) < np.min(b[:, 0])
        or np.max(b[:, 0]) < np.min(a[:, 0])
        or np.max(a[:, 1]) < np.min(b[:, 1])
        or np.max(b[:, 1]) < np.min(a[:, 1])
    ):
        return 0.0
    contour_a = a.astype(np.float32).reshape(-1, 1, 2)
    contour_b = b.astype(np.float32).reshape(-1, 1, 2)
    if cv2.isContourConvex(contour_a) and cv2.isContourConvex(contour_b):
        # intersectConvexConvex has a rotation-dependent numerical failure for
        # a small class of almost complementary quadrilaterals: at one angle it
        # reports zero, while the same rigidly rotated pair reports that one
        # polygon is almost fully covered.  Such a false zero lets a completely
        # overlapping assembly win the rectangle search.  Cross-check at a
        # fixed non-axis angle; the common case stays in compiled OpenCV, and
        # only disagreement falls through to the reliable local raster below.
        area = float(cv2.intersectConvexConvex(contour_a, contour_b)[0])
        check_rotation = np.asarray(
            [
                [0.9925461516, -0.1218693434],
                [0.1218693434, 0.9925461516],
            ],
            dtype=np.float32,
        )
        rotated_a = (a.astype(np.float32) @ check_rotation).reshape(-1, 1, 2)
        rotated_b = (b.astype(np.float32) @ check_rotation).reshape(-1, 1, 2)
        rotated_area = float(
            cv2.intersectConvexConvex(rotated_a, rotated_b)[0]
        )
        if math.isfinite(area) and math.isfinite(rotated_area):
            disagreement = abs(area - rotated_area)
            agreement_limit = max(
                0.75, 0.02 * max(abs(area), abs(rotated_area))
            )
            if disagreement <= agreement_limit:
                return max(0.0, 0.5 * (area + rotated_area))

    scale = max(float(pixels_per_mm), 0.5)
    lower = np.floor(np.minimum(np.min(a, axis=0), np.min(b, axis=0)) - 1.0)
    upper = np.ceil(np.maximum(np.max(a, axis=0), np.max(b, axis=0)) + 1.0)
    size = np.maximum(3, np.ceil((upper - lower) * scale).astype(int) + 1)
    mask_a = np.zeros((int(size[1]), int(size[0])), np.uint8)
    mask_b = np.zeros_like(mask_a)
    local_a = np.rint((a - lower) * scale).astype(np.int32)
    local_b = np.rint((b - lower) * scale).astype(np.int32)
    cv2.fillPoly(mask_a, [local_a], 1)
    cv2.fillPoly(mask_b, [local_b], 1)
    return float(
        cv2.countNonZero(cv2.bitwise_and(mask_a, mask_b)) / scale**2
    )


def polygons_overlap(
    first: np.ndarray, second: np.ndarray, tolerance_mm: float = 0.4
) -> bool:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if (
        np.max(a[:, 0]) < np.min(b[:, 0])
        or np.max(b[:, 0]) < np.min(a[:, 0])
        or np.max(a[:, 1]) < np.min(b[:, 1])
        or np.max(b[:, 1]) < np.min(a[:, 1])
    ):
        return False

    # Use the rotation-checked convex path (and the local raster fallback for
    # concave/ambiguous pairs) so search-time and final overlap decisions use
    # the same rotation-invariant authority.
    intersection_area = polygon_intersection_area(a, b)
    allowed_area = max(5.0, 15.0 * tolerance_mm)
    return intersection_area > allowed_area


def edge_alignment_transforms(
    source_edge: np.ndarray,
    destination_edge: np.ndarray,
    length_tolerance_mm: float,
    relative_tolerance: float,
    minimum_partial_remainder_mm: float = 18.0,
    internal_partial_fractions: tuple[float, ...] | list[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Map a source edge against a destination edge with opposite directions.

    Equal edges are centre-aligned.  Unequal edges may share either endpoint.
    When ``internal_partial_fractions`` is supplied, a shorter edge may also
    occupy an interior section of a longer edge.  The latter is required for
    legal three/four-piece cuts where a long straight side is divided by two
    T-junctions.  Every returned transform is still a proper rotation plus a
    translation; reflection and scaling are never introduced.
    """

    u0, u1 = np.asarray(source_edge, dtype=np.float64)
    a, b = np.asarray(destination_edge, dtype=np.float64)
    src_vec = u1 - u0
    dst_vec = b - a
    src_len = float(np.linalg.norm(src_vec))
    dst_len = float(np.linalg.norm(dst_vec))
    allowed = max(
        length_tolerance_mm,
        relative_tolerance * min(src_len, dst_len),
    )
    difference = abs(src_len - dst_len)
    # The formal advanced task uses sides >= 20 mm.  A guided practice-piece
    # solve may explicitly lower this limit when camera simplification leaves
    # a shorter apparent remainder; autonomous mode keeps the strict default.
    if (
        difference > allowed
        and difference < minimum_partial_remainder_mm
    ):
        return []
    if max(src_len, dst_len) / min(src_len, dst_len) > 4.0:
        return []
    if min(src_len, dst_len) < 1e-6:
        return []

    source_angle = math.atan2(src_vec[1], src_vec[0])
    destination_reverse_angle = math.atan2(-dst_vec[1], -dst_vec[0])
    r = rotation_matrix_row(destination_reverse_angle - source_angle)
    unit = dst_vec / dst_len

    if difference <= allowed:
        positions = [(dst_len + src_len) * 0.5]
    else:
        # q0 is the transformed source u0 coordinate along destination a->b.
        positions = [dst_len, src_len]
        if internal_partial_fractions:
            lower_position = min(dst_len, src_len)
            upper_position = max(dst_len, src_len)
            span = upper_position - lower_position
            for fraction in internal_partial_fractions:
                value = float(fraction)
                if 0.0 < value < 1.0:
                    positions.append(lower_position + value * span)
        # Preserve deterministic search order while avoiding duplicate
        # endpoint/centre placements after floating-point conversion.
        positions = list(dict.fromkeys(round(value, 8) for value in positions))

    results: list[tuple[np.ndarray, np.ndarray, float]] = []
    for q0_coordinate in positions:
        q0 = a + unit * q0_coordinate
        t = q0 - u0 @ r
        overlap = min(src_len, dst_len)
        results.append((r, t, overlap))
    return results
