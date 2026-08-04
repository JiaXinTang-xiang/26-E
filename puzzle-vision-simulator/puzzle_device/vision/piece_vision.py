#!/usr/bin/env python3
"""OpenCV piece segmentation and geometric feature extraction."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.vision.cuda_ops import (
    _Fallback,
    _check_cuda,
    _disable_cuda,
    segment_pieces_gpu,
)


@dataclass
class DetectionConfig:
    """Tunable thresholds for a rectified top-down work-area image."""

    segmentation_method: str = "background"
    min_area_px: float = 800.0
    max_area_px: float = 0.0
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.75
    min_vertices: int = 3
    max_vertices: int = 5
    max_pieces: int = 4
    border_fraction: float = 0.04
    color_distance_threshold: int | None = None
    white_saturation_max: int = 85
    white_value_min: int = 170
    brightness_min: int = 185
    morphology_size: int = 3
    gaussian_blur_size: int = 5
    canny_lower: int = 50
    canny_upper: int = 150
    polygon_epsilon_min: float = 0.0025
    polygon_epsilon_preferred: float = 0.02
    polygon_epsilon_max: float = 0.06
    polygon_epsilon_steps: int = 60
    min_edge_length_px: float = 3.0
    min_edge_length_ratio: float = 0.025
    minimum_pick_clearance_px: float = 8.0
    polygon_vertex_strategy: str = "current"

    def validate(self) -> None:
        if self.segmentation_method not in {"background", "white_hsv", "brightness"}:
            raise ValueError("unknown segmentation method")
        if self.min_area_px <= 0 or not 0 <= self.min_area_ratio < self.max_area_ratio <= 1:
            raise ValueError("invalid piece area limits")
        if self.max_area_px < 0 or 0 < self.max_area_px <= self.min_area_px:
            raise ValueError("max_area_px must be zero or greater than min_area_px")
        if not 3 <= self.min_vertices <= self.max_vertices:
            raise ValueError("vertex limits must start at three")
        if self.max_pieces <= 0:
            raise ValueError("max_pieces must be positive")
        if self.color_distance_threshold is not None and not 0 <= self.color_distance_threshold <= 255:
            raise ValueError("color distance threshold must be 0..255 or null")
        if not 0 <= self.white_saturation_max <= 255 or not 0 <= self.white_value_min <= 255:
            raise ValueError("white HSV thresholds must be 0..255")
        if not 0 <= self.brightness_min <= 255:
            raise ValueError("brightness threshold must be 0..255")
        if self.morphology_size < 1 or self.gaussian_blur_size < 1:
            raise ValueError("kernel sizes must be positive")
        if not 0 <= self.canny_lower < self.canny_upper <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= lower < upper <= 255")
        if not (0 < self.polygon_epsilon_min <= self.polygon_epsilon_preferred
                <= self.polygon_epsilon_max < 0.2):
            raise ValueError("invalid polygon epsilon range")
        if self.polygon_epsilon_steps < 2:
            raise ValueError("polygon_epsilon_steps must be at least two")
        if self.min_edge_length_px < 0 or not 0 <= self.min_edge_length_ratio < 1:
            raise ValueError("invalid minimum edge length")
        if self.minimum_pick_clearance_px < 0:
            raise ValueError("minimum pick clearance cannot be negative")
        if self.polygon_vertex_strategy not in {"current", "legacy_4"}:
            raise ValueError("unknown polygon vertex strategy")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "DetectionConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown detection parameters: {', '.join(sorted(unknown))}")
        config = cls(**values)
        config.validate()
        return config


def load_detection_config(path: Path) -> DetectionConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("parameters", document)
    if not isinstance(values, dict):
        raise ValueError("detection configuration must contain an object")
    return DetectionConfig.from_dict(values)


def save_detection_config(path: Path, config: DetectionConfig) -> None:
    config.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "format": "puzzle-device.vision-detection-config.v1",
        "parameters": config.to_dict(),
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class PieceObservation:
    """Geometry measured from one piece in image coordinates."""

    piece_id: int
    contour: np.ndarray
    polygon: np.ndarray
    mask: np.ndarray
    center: tuple[float, float]
    pick_point: tuple[float, float]
    pick_clearance_px: float
    area_px: float
    pca_angle_deg: float
    longest_edge_angle_deg: float
    bounding_box: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "piece_id": self.piece_id,
            "center_px": [round(self.center[0], 3), round(self.center[1], 3)],
            "pick_point_px": [round(self.pick_point[0], 3), round(self.pick_point[1], 3)],
            "pick_clearance_px": round(self.pick_clearance_px, 3),
            "area_px": round(self.area_px, 3),
            "vertex_count": len(self.polygon),
            "vertices_px": np.round(self.polygon, 3).tolist(),
            "pca_angle_deg": round(self.pca_angle_deg, 3),
            "longest_edge_angle_deg": round(self.longest_edge_angle_deg, 3),
            "bounding_box": list(self.bounding_box),
            "confidence": round(self.confidence, 4),
        }


def _normalize_axis_angle(angle_deg: float) -> float:
    """Normalize an undirected axis angle to [-90, 90)."""
    return (angle_deg + 90.0) % 180.0 - 90.0


def order_clockwise(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    ordered = vertices[np.argsort(angles)]
    start = np.lexsort((ordered[:, 0], ordered[:, 1]))[0]
    return np.roll(ordered, -start, axis=0)


def _border_pixels(lab: np.ndarray, fraction: float) -> np.ndarray:
    h, w = lab.shape[:2]
    band = max(2, int(round(min(h, w) * fraction)))
    return np.concatenate([
        lab[:band].reshape(-1, 3),
        lab[-band:].reshape(-1, 3),
        lab[band:-band, :band].reshape(-1, 3),
        lab[band:-band, -band:].reshape(-1, 3),
    ])


def segment_pieces(
    image: np.ndarray,
    background: np.ndarray | None = None,
    config: DetectionConfig | None = None,
) -> np.ndarray:
    """Return a binary piece mask using an empty frame or border color model.

    When a CUDA device is available the common background-subtraction path
    chains blur → Lab convert → threshold → morphology on the GPU to
    minimise kernel-launch and transfer overhead.  Other segmentation
    methods and the no-background fallback run on the CPU.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a non-empty BGR image")
    cfg = config or DetectionConfig()
    cfg.validate()
    if background is not None and background.shape != image.shape:
        raise ValueError("background and image must have the same shape")

    # -- try composite GPU pipeline (background subtraction only) -----------
    # GPU kernel-launch overhead dominates on small frames; only use it
    # when the image area exceeds ~0.5 MPix where the per-pixel savings
    # outweigh the upload/download cost.
    if (
        cfg.segmentation_method == "background"
        and _check_cuda()
        and image.shape[0] * image.shape[1] >= 500_000
    ):
        try:
            return segment_pieces_gpu(
                image, background,
                blur_size=max(1, cfg.gaussian_blur_size | 1),
                color_distance_threshold=cfg.color_distance_threshold,
                morph_size=cfg.morphology_size,
            )
        except (_Fallback, cv2.error, AttributeError, RuntimeError, TypeError) as exc:
            # Some CUDA-enabled OpenCV builds expose the device but omit one
            # or more Python operators. Disable the optional path for the rest
            # of this process and continue with the proven CPU implementation.
            _disable_cuda(exc)

    # -- CPU path -----------------------------------------------------------
    blur_size = max(1, cfg.gaussian_blur_size | 1)
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    if cfg.segmentation_method == "white_hsv":
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0, 0, cfg.white_value_min], dtype=np.uint8),
            np.array([179, cfg.white_saturation_max, 255], dtype=np.uint8),
        )
    elif cfg.segmentation_method == "brightness":
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, cfg.brightness_min, 255, cv2.THRESH_BINARY)
    else:
        if background is not None:
            reference = cv2.GaussianBlur(background, (blur_size, blur_size), 0)
            distance = _color_distance(blurred, reference)
        else:
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            background_color = np.median(
                _border_pixels(lab, cfg.border_fraction), axis=0
            ).astype(np.uint8)
            reference = np.empty_like(blurred)
            reference[:] = cv2.cvtColor(
                background_color.reshape(1, 1, 3), cv2.COLOR_LAB2BGR
            )[0, 0]
            distance = _color_distance(blurred, reference)
        if cfg.color_distance_threshold is None:
            otsu_threshold, mask = cv2.threshold(
                distance, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )
            if otsu_threshold < 8:
                _, mask = cv2.threshold(distance, 8, 255, cv2.THRESH_BINARY)
        else:
            _, mask = cv2.threshold(
                distance, cfg.color_distance_threshold, 255, cv2.THRESH_BINARY
            )

    size = max(3, cfg.morphology_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Fill texture holes so playing-card markings do not split a piece.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _color_distance(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Euclidean Lab colour distance between two BGR images (CPU)."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab - ref
    return np.sqrt(np.mean(diff * diff, axis=2)).clip(0, 255).astype(np.uint8)


def _polygon_from_contour(
    contour: np.ndarray, config: DetectionConfig
) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    # Do not use a convex hull here: legal field pieces may be concave. Remove
    # only tiny anti-aliasing corners after each Douglas-Peucker approximation.
    preferred = config.polygon_epsilon_preferred
    candidates = np.linspace(
        config.polygon_epsilon_min,
        config.polygon_epsilon_max,
        config.polygon_epsilon_steps,
    )
    ratios = sorted(candidates, key=lambda value: abs(value - preferred))
    if config.polygon_vertex_strategy == "legacy_4":
        for ratio in ratios:
            approx = cv2.approxPolyDP(contour, ratio * perimeter, True).reshape(-1, 2)
            approx = _remove_tiny_edges(approx.astype(np.float64), perimeter, config)
            if config.min_vertices <= len(approx) <= config.max_vertices:
                return order_clockwise(approx)
        return None

    valid = []
    contour_area = abs(float(cv2.contourArea(contour)))
    minimum_edge = max(
        config.min_edge_length_px,
        config.min_edge_length_ratio * perimeter,
    )
    seen = set()
    for ratio in ratios:
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True).reshape(-1, 2)
        approx = _remove_tiny_edges(approx.astype(np.float64), perimeter, config)
        count = len(approx)
        if not config.min_vertices <= count <= config.max_vertices:
            continue
        ordered = order_clockwise(approx)
        signature = tuple(np.round(ordered, 2).reshape(-1))
        if signature in seen:
            continue
        seen.add(signature)

        polygon_area = abs(float(cv2.contourArea(ordered.astype(np.float32))))
        area_error = abs(polygon_area - contour_area) / max(contour_area, 1.0)
        edges = np.roll(ordered, -1, axis=0) - ordered
        lengths = np.linalg.norm(edges, axis=1)
        short_edge_penalty = max(0.0, minimum_edge * 1.5 - float(lengths.min()))
        short_edge_penalty /= max(minimum_edge * 1.5, 1.0)

        previous = np.roll(ordered, 1, axis=0) - ordered
        following = np.roll(ordered, -1, axis=0) - ordered
        cosine = np.sum(previous * following, axis=1) / np.maximum(
            np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1), 1e-9
        )
        turn_angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        collinear_penalty = max(0.0, float(turn_angles.max()) - 168.0) / 12.0
        preferred_penalty = abs(ratio - preferred) / max(
            config.polygon_epsilon_max - config.polygon_epsilon_min, 1e-9
        )
        score = (
            5.0 * area_error
            + 2.0 * short_edge_penalty
            + 1.5 * collinear_penalty
            + 0.08 * preferred_penalty
        )
        valid.append((score, area_error, preferred_penalty, ordered))

    if not valid:
        return None
    # Compare every valid epsilon instead of accepting the first 3-5 vertex result.
    return min(valid, key=lambda item: item[:3])[3]


def _remove_tiny_edges(
    vertices: np.ndarray, perimeter: float, config: DetectionConfig
) -> np.ndarray:
    vertices = vertices.copy()
    minimum = max(config.min_edge_length_px, config.min_edge_length_ratio * perimeter)
    while len(vertices) > 3:
        lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
        index = int(np.argmin(lengths))
        if lengths[index] >= minimum:
            break
        # Merge a tiny raster edge by retaining the endpoint that better
        # preserves its neighboring long edges.
        remove = (index + 1) % len(vertices)
        vertices = np.delete(vertices, remove, axis=0)
    return vertices


def _center_from_contour(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    points = contour.reshape(-1, 2).astype(float)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def _safe_pick_point(mask: np.ndarray) -> tuple[tuple[float, float], float]:
    """Return the interior point with maximum clearance from a piece boundary."""
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _minimum, maximum, _minimum_location, maximum_location = cv2.minMaxLoc(distance)
    if maximum <= 0:
        raise ValueError("piece mask has no valid interior pick point")
    return (float(maximum_location[0]), float(maximum_location[1])), float(maximum)


def _pca_angle(contour: np.ndarray) -> float:
    points = contour.reshape(-1, 2).astype(np.float64)
    points -= points.mean(axis=0)
    covariance = points.T @ points / max(1, len(points) - 1)
    values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, np.argmax(values)]
    return _normalize_axis_angle(math.degrees(math.atan2(axis[1], axis[0])))


def _longest_edge_angle(polygon: np.ndarray) -> float:
    following = np.roll(polygon, -1, axis=0)
    vectors = following - polygon
    edge = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
    return _normalize_axis_angle(math.degrees(math.atan2(edge[1], edge[0])))


def detect_piece_observations(
    image: np.ndarray,
    background: np.ndarray | None = None,
    config: DetectionConfig | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[list[PieceObservation], np.ndarray]:
    """Segment pieces and extract contour, vertices, center, and orientation."""
    cfg = config or DetectionConfig()
    cfg.validate()
    if image is None or image.ndim != 3:
        raise ValueError("image must be a non-empty BGR image")

    origin_x = origin_y = 0
    detection_image = image
    detection_background = background
    if roi is not None:
        origin_x, origin_y, width, height = (int(value) for value in roi)
        if origin_x < 0 or origin_y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI must describe a positive image rectangle")
        if origin_x + width > image.shape[1] or origin_y + height > image.shape[0]:
            raise ValueError("ROI extends outside the image")
        detection_image = image[origin_y:origin_y + height, origin_x:origin_x + width]
        if background is not None:
            if background.shape != image.shape:
                raise ValueError("background and image must have the same shape")
            detection_background = background[
                origin_y:origin_y + height, origin_x:origin_x + width
            ]

    local_mask = segment_pieces(detection_image, detection_background, cfg)
    image_area = detection_image.shape[0] * detection_image.shape[1]
    min_area = max(cfg.min_area_px, cfg.min_area_ratio * image_area)
    max_area = cfg.max_area_ratio * image_area
    if cfg.max_area_px > 0:
        max_area = min(max_area, cfg.max_area_px)
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    raw = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        polygon = _polygon_from_contour(contour, cfg)
        if polygon is None:
            continue
        polygon_area = abs(float(cv2.contourArea(polygon.astype(np.float32))))
        area_agreement = max(0.0, 1.0 - abs(polygon_area - area) / max(area, 1.0))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0 else 0.0
        confidence = float(np.clip(0.55 * area_agreement + 0.45 * solidity, 0, 1))
        local_piece_mask = np.zeros_like(local_mask)
        cv2.drawContours(local_piece_mask, [contour], -1, 255, cv2.FILLED)
        local_pick_point, pick_clearance = _safe_pick_point(local_piece_mask)
        offset = np.array([origin_x, origin_y], dtype=np.float64)
        global_contour = contour + np.array([[[origin_x, origin_y]]], dtype=contour.dtype)
        global_polygon = polygon + offset
        local_center = _center_from_contour(contour)
        global_center = (local_center[0] + origin_x, local_center[1] + origin_y)
        global_pick_point = (
            local_pick_point[0] + origin_x,
            local_pick_point[1] + origin_y,
        )
        piece_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        piece_mask[origin_y:origin_y + local_mask.shape[0],
                   origin_x:origin_x + local_mask.shape[1]] = local_piece_mask
        box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
        raw.append({
            "contour": global_contour,
            "polygon": global_polygon,
            "mask": piece_mask,
            "center": global_center,
            "pick_point": global_pick_point,
            "pick_clearance": pick_clearance,
            "area": area,
            "pca_angle": _pca_angle(contour),
            "edge_angle": _longest_edge_angle(polygon),
            "bbox": (box_x + origin_x, box_y + origin_y, box_width, box_height),
            "confidence": confidence,
        })

    raw.sort(key=lambda item: (item["center"][1], item["center"][0]))
    if len(raw) > cfg.max_pieces:
        raise RuntimeError(f"detected {len(raw)} pieces, expected at most {cfg.max_pieces}")

    observations = [
        PieceObservation(
            piece_id=index,
            contour=item["contour"],
            polygon=item["polygon"],
            mask=item["mask"],
            center=item["center"],
            pick_point=item["pick_point"],
            pick_clearance_px=item["pick_clearance"],
            area_px=item["area"],
            pca_angle_deg=item["pca_angle"],
            longest_edge_angle_deg=item["edge_angle"],
            bounding_box=item["bbox"],
            confidence=item["confidence"],
        )
        for index, item in enumerate(raw)
    ]
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[origin_y:origin_y + local_mask.shape[0],
         origin_x:origin_x + local_mask.shape[1]] = local_mask
    return observations, mask


def extract_piece_edges(mask: np.ndarray, config: DetectionConfig | None = None) -> np.ndarray:
    """Extract clean outer edges from the segmentation mask, excluding piece texture."""
    if mask is None or mask.ndim != 2:
        raise ValueError("mask must be a non-empty single-channel image")
    cfg = config or DetectionConfig()
    cfg.validate()
    blur_size = max(1, cfg.gaussian_blur_size | 1)
    smoothed = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return cv2.Canny(smoothed, cfg.canny_lower, cfg.canny_upper, apertureSize=3)


def draw_piece_observations(
    image: np.ndarray, observations: list[PieceObservation]
) -> np.ndarray:
    output = image.copy()
    for piece in observations:
        polygon = np.round(piece.polygon).astype(np.int32)
        cv2.polylines(output, [polygon], True, (0, 220, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(polygon):
            cv2.circle(output, tuple(point), 5, (0, 0, 255), -1)
            cv2.putText(output, str(index), tuple(point + [6, -6]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA)

        center = np.round(piece.center).astype(int)
        length = max(30, int(math.sqrt(piece.area_px) * 0.28))
        angle = math.radians(piece.pca_angle_deg)
        delta = np.array([math.cos(angle), math.sin(angle)]) * length
        start = tuple(np.round(center - delta).astype(int))
        end = tuple(np.round(center + delta).astype(int))
        cv2.line(output, start, end, (255, 80, 30), 2, cv2.LINE_AA)
        cv2.drawMarker(output, tuple(center), (40, 255, 40), cv2.MARKER_CROSS, 15, 2)
        pick = np.round(piece.pick_point).astype(int)
        cv2.circle(output, tuple(pick), max(5, round(piece.pick_clearance_px)), (255, 0, 255), 1,
                   cv2.LINE_AA)
        cv2.drawMarker(output, tuple(pick), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
        label = (
            f"P{piece.piece_id} PCA:{piece.pca_angle_deg:.1f} deg "
            f"pick:{piece.pick_clearance_px:.0f}px"
        )
        cv2.putText(output, label, tuple(center + [10, 20]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(output, label, tuple(center + [10, 20]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="input BGR image")
    parser.add_argument("--background", type=Path, help="optional empty work-area image")
    parser.add_argument("--output", type=Path, default=Path("output/piece_detection"))
    parser.add_argument(
        "--roi", metavar="X,Y,W,H",
        help="optional source-area crop; reported coordinates are relative to this ROI",
    )
    parser.add_argument("--min-area", type=float, default=800.0)
    parser.add_argument("--max-area", type=float, default=0.0,
                        help="maximum contour area in pixels; zero disables this extra limit")
    parser.add_argument("--threshold", type=int, help="fixed Lab color-distance threshold")
    parser.add_argument("--config", type=Path, help="shared vision parameter JSON")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")
    background = None
    if args.background:
        background = cv2.imread(str(args.background), cv2.IMREAD_COLOR)
        if background is None:
            raise SystemExit(f"cannot read background: {args.background}")

    roi_origin = (0, 0)
    if args.roi:
        try:
            x, y, width, height = (int(value) for value in args.roi.split(","))
        except ValueError as exc:
            raise SystemExit("--roi must use X,Y,W,H integers") from exc
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise SystemExit("--roi values must describe a positive image rectangle")
        if x + width > image.shape[1] or y + height > image.shape[0]:
            raise SystemExit("--roi extends outside the input image")
        roi_origin = (x, y)
        image = image[y:y + height, x:x + width]
        if background is not None:
            background = background[y:y + height, x:x + width]

    config = load_detection_config(args.config) if args.config else DetectionConfig()
    config.min_area_px = args.min_area
    config.max_area_px = args.max_area
    if args.threshold is not None:
        config.color_distance_threshold = args.threshold
    config.validate()
    observations, mask = detect_piece_observations(image, background, config)
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output / "mask.png"), mask)
    cv2.imwrite(str(args.output / "detected.png"),
                draw_piece_observations(image, observations))
    result = {
        "coordinate_system": "ROI image pixels; x right, y down",
        "roi_origin_in_input_px": list(roi_origin),
        "pieces": [piece.to_dict() for piece in observations],
    }
    (args.output / "pieces.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"detected {len(observations)} piece(s); output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
