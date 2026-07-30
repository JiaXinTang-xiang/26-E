from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedPiece:
    id: int
    polygon_mm: np.ndarray
    center_mm: np.ndarray
    pick_point_mm: np.ndarray
    angle_deg: float
    contour_px: np.ndarray
    mask: np.ndarray
    source_image: np.ndarray
    local_polygon_mm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.polygon_mm = np.asarray(self.polygon_mm, dtype=np.float64)
        self.center_mm = np.asarray(self.center_mm, dtype=np.float64)
        self.pick_point_mm = np.asarray(self.pick_point_mm, dtype=np.float64)
        self.local_polygon_mm = self.polygon_mm - self.center_mm


@dataclass
class Placement:
    piece_index: int
    rotation: np.ndarray
    translation: np.ndarray

    def transform(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return points @ self.rotation.T + self.translation


@dataclass
class SolvedLayout:
    placements: dict[int, Placement]
    aligned_polygons_mm: dict[int, np.ndarray]
    final_rotations: dict[int, np.ndarray]
    final_translations: dict[int, np.ndarray]
    width_mm: float
    height_mm: float
    geometry_score: float
    texture_score: float
    mode: str
