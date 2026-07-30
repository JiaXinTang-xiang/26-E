from __future__ import annotations

import cv2
import numpy as np

from .config import DEFAULT_SCALE, SPLIT_Y_MM
from .geometry import ensure_ccw, longest_edge_angle, polygon_centroid
from .models import DetectedPiece


def _refined_longest_edge_angle(
    contour: np.ndarray, polygon_px: np.ndarray
) -> float:
    polygon = np.asarray(polygon_px, dtype=np.float64)
    vectors = np.roll(polygon, -1, axis=0) - polygon
    edge_index = int(np.argmax(np.linalg.norm(vectors, axis=1)))
    a = polygon[edge_index]
    direction = vectors[edge_index]
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return longest_edge_angle(polygon)
    unit = direction / length
    points = contour.reshape(-1, 2).astype(np.float64)
    relative = points - a
    projection = relative @ unit
    distance = np.abs(unit[0] * relative[:, 1] - unit[1] * relative[:, 0])
    selected = points[
        (projection >= 0.08 * length)
        & (projection <= 0.92 * length)
        & (distance <= 3.0)
    ]
    if len(selected) < 8:
        return longest_edge_angle(polygon)
    vx, vy, _, _ = cv2.fitLine(
        selected.astype(np.float32),
        cv2.DIST_L2,
        0,
        0.01,
        0.01,
    ).reshape(-1)
    return (float(np.degrees(np.arctan2(vy, vx))) + 180.0) % 180.0


def _adaptive_polygon(contour: np.ndarray) -> np.ndarray:
    # Keep concave hub vertices: a fan sector that follows more than half of the
    # rectangle perimeter is a legal concave piece. Card artwork holes are handled
    # by RETR_EXTERNAL plus the closed gray outline, not by replacing the contour
    # with a convex hull.
    perimeter = cv2.arcLength(contour, True)
    chosen = None
    for fraction in np.linspace(0.002, 0.035, 34):
        approx = cv2.approxPolyDP(contour, float(fraction * perimeter), True)
        count = len(approx)
        if 3 <= count <= 5:
            chosen = approx
            break
    if chosen is None:
        chosen = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    points = chosen.reshape(-1, 2).astype(np.float64)
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            current = points[index]
            following = points[(index + 1) % len(points)]
            if np.linalg.norm(current - previous) < 30.0:
                points = np.delete(points, index, axis=0)
                changed = True
                break
            line = following - previous
            line_length = np.linalg.norm(line)
            if line_length > 1e-6:
                offset = current - previous
                cross_2d = line[0] * offset[1] - line[1] * offset[0]
                distance = abs(cross_2d) / line_length
                if distance < 1.2:
                    points = np.delete(points, index, axis=0)
                    changed = True
                    break
    return ensure_ccw(points)


def foreground_mask(image: np.ndarray, scale: float = DEFAULT_SCALE) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    # White pieces are deliberately a little brighter than the warm-white A4.
    # Colored self-prepared pieces are selected by saturation. The dark branch
    # keeps clipped card artwork and the gray cut edge connected as one piece;
    # the visible A4 border and piece shadow use mid tones and remain excluded.
    mask = (
        (
            (value >= 251)
            | (value <= 135)
            | ((saturation > 70) & (value > 65))
        )
        * 255
    ).astype(np.uint8)
    split_px = int(round(SPLIT_Y_MM * scale))
    mask[max(0, split_px - int(round(3.0 * scale))) :, :] = 0
    kernel_size = max(3, int(round(1.2 * scale)) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def detect_pieces(
    image: np.ndarray, scale: float = DEFAULT_SCALE
) -> tuple[list[DetectedPiece], np.ndarray]:
    if image is None or image.ndim != 3:
        raise ValueError("Input image could not be read")
    mask = foreground_mask(image, scale)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minimum_area = (12.0 * scale) ** 2
    contours = [contour for contour in contours if cv2.contourArea(contour) >= minimum_area]
    contours.sort(
        key=lambda contour: (
            cv2.boundingRect(contour)[1],
            cv2.boundingRect(contour)[0],
        )
    )
    pieces: list[DetectedPiece] = []
    for index, contour in enumerate(contours, start=1):
        polygon_px = _adaptive_polygon(contour)
        component = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(
            component,
            [np.round(polygon_px).astype(np.int32)],
            255,
        )
        polygon_mm = polygon_px / scale
        center_mm = polygon_centroid(polygon_mm)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        _, _, _, max_location = cv2.minMaxLoc(distance)
        pick_point_mm = np.array(max_location, dtype=np.float64) / scale
        pieces.append(
            DetectedPiece(
                id=index,
                polygon_mm=polygon_mm,
                center_mm=center_mm,
                pick_point_mm=pick_point_mm,
                angle_deg=_refined_longest_edge_angle(contour, polygon_px),
                contour_px=contour,
                mask=component,
                source_image=image,
            )
        )
    if not pieces:
        raise RuntimeError("No puzzle pieces were detected in the A4 upper half")
    if len(pieces) > 4:
        raise RuntimeError(f"Detected {len(pieces)} pieces; the problem allows at most 4")
    return pieces, mask


def classify_mode(image: np.ndarray, pieces: list[DetectedPiece]) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation_values: list[np.ndarray] = []
    texture_values: list[np.ndarray] = []
    scale = image.shape[1] / 210.0
    for piece in pieces:
        erosion_px = max(9, int(round(3.0 * scale)))
        eroded = cv2.erode(
            piece.mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (erosion_px * 2 + 1, erosion_px * 2 + 1)
            ),
        )
        valid = eroded > 0
        if not np.any(valid):
            valid = piece.mask > 0
        saturation_values.append(hsv[:, :, 1][valid])
        texture_values.append(gray[valid])
    saturation = np.concatenate(saturation_values)
    texture = np.concatenate(texture_values)
    if float(np.median(saturation)) > 65.0:
        return "self"
    low_fraction = float(np.mean(texture < 185))
    texture_std = float(np.std(texture))
    if low_fraction > 0.012 or texture_std > 18.0:
        return "field-card"
    return "field-white"


def draw_detection(
    image: np.ndarray, pieces: list[DetectedPiece], mode: str, scale: float
) -> np.ndarray:
    output = image.copy()
    palette = [
        (60, 220, 60),
        (255, 150, 50),
        (80, 120, 255),
        (220, 80, 220),
    ]
    for index, piece in enumerate(pieces):
        color = palette[index % len(palette)]
        polygon_px = np.round(piece.polygon_mm * scale).astype(np.int32)
        cv2.polylines(output, [polygon_px], True, color, 3, cv2.LINE_AA)
        center = tuple(np.round(piece.center_mm * scale).astype(int))
        pick = tuple(np.round(piece.pick_point_mm * scale).astype(int))
        cv2.drawMarker(output, center, (30, 30, 240), cv2.MARKER_CROSS, 22, 3)
        cv2.circle(output, pick, 8, (40, 230, 40), 3, cv2.LINE_AA)
        text = (
            f"P{piece.id} C=({piece.center_mm[0]:.1f},"
            f"{piece.center_mm[1]:.1f}) A={piece.angle_deg:.1f}"
        )
        origin = (max(4, center[0] + 10), max(24, center[1] - 10))
        cv2.putText(
            output,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        output,
        f"mode: {mode}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (45, 48, 52),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"mode: {mode}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (30, 120, 205),
        1,
        cv2.LINE_AA,
    )
    return output
