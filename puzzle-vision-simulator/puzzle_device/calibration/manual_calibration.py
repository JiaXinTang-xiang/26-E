"""Manual camera-pixel to gantry-pulse calibration primitives."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationPoint:
    """One manually clicked tool-centre and its known gantry position."""

    pixel_x: float
    pixel_y: float
    pulse_x: float
    pulse_y: float
    phase: str = "manual"


@dataclass(frozen=True)
class CalibrationMetrics:
    point_count: int
    inlier_count: int
    mean_error_pulse: float
    median_error_pulse: float
    max_error_pulse: float


class PixelToGantryCalibration:
    """Fit and apply a projective mapping from camera pixels to XY pulses."""

    def __init__(self, points: list[CalibrationPoint] | None = None):
        self.points = list(points or [])
        self.matrix: np.ndarray | None = None
        self.inliers: np.ndarray | None = None
        self.metrics: CalibrationMetrics | None = None

    def add_point(self, point: CalibrationPoint) -> None:
        self.points.append(point)
        self.matrix = None
        self.inliers = None
        self.metrics = None

    def remove_point(self, index: int) -> CalibrationPoint:
        point = self.points.pop(index)
        self.matrix = None
        self.inliers = None
        self.metrics = None
        return point

    def fit(self, ransac_threshold_pulse: float = 12.0) -> CalibrationMetrics:
        """Fit the mapping, rejecting accidental clicks with RANSAC."""
        if len(self.points) < 4:
            raise ValueError("at least four calibration points are required")
        source = np.array([[p.pixel_x, p.pixel_y] for p in self.points], np.float32)
        target = np.array([[p.pulse_x, p.pulse_y] for p in self.points], np.float32)
        matrix, inlier_mask = cv2.findHomography(
            source, target, cv2.RANSAC, ransac_threshold_pulse
        )
        if matrix is None or inlier_mask is None:
            raise RuntimeError("homography fitting failed; check point distribution")
        self.matrix = matrix
        self.inliers = inlier_mask.reshape(-1).astype(bool)
        errors = np.linalg.norm(self.transform_pixels(source) - target, axis=1)
        inlier_errors = errors[self.inliers]
        if len(inlier_errors) < 4:
            raise RuntimeError("fewer than four inliers remain after RANSAC")
        self.metrics = CalibrationMetrics(
            point_count=len(self.points),
            inlier_count=int(self.inliers.sum()),
            mean_error_pulse=float(inlier_errors.mean()),
            median_error_pulse=float(np.median(inlier_errors)),
            max_error_pulse=float(inlier_errors.max()),
        )
        return self.metrics

    def transform_pixels(self, pixels: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("fit calibration before transforming coordinates")
        values = np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(values, self.matrix).reshape(-1, 2)

    def predict_pulse(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        pulse = self.transform_pixels(np.array([[pixel_x, pixel_y]], np.float32))[0]
        return float(pulse[0]), float(pulse[1])

    def point_errors(self) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("fit calibration before requesting errors")
        pixels = np.array([[p.pixel_x, p.pixel_y] for p in self.points], np.float32)
        pulses = np.array([[p.pulse_x, p.pulse_y] for p in self.points], np.float32)
        return np.linalg.norm(self.transform_pixels(pixels) - pulses, axis=1)

    def save(self, path: Path, metadata: dict | None = None) -> None:
        if self.matrix is None or self.inliers is None or self.metrics is None:
            raise RuntimeError("fit calibration before saving")
        document = {
            "format": "puzzle-device.pixel-to-gantry-calibration.v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": {
                "source": "undistorted camera pixels; x right, y down",
                "target": "gantry absolute pulses after XY homing",
            },
            "matrix_pixel_to_pulse": self.matrix.tolist(),
            "points": [asdict(point) for point in self.points],
            "inliers": self.inliers.astype(int).tolist(),
            "metrics": asdict(self.metrics),
            "metadata": metadata or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

