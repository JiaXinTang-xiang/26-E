"""Save and load repeatable real-camera vision cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.vision.piece_vision import DetectionConfig, PieceObservation
from puzzle_device.vision.image_io import read_image, write_image


@dataclass(frozen=True)
class VisionCase:
    path: Path
    frame: np.ndarray
    background: np.ndarray | None
    config: DetectionConfig
    roi: tuple[int, int, int, int] | None


def save_vision_case(
    root: Path,
    frame: np.ndarray,
    background: np.ndarray | None,
    config: DetectionConfig,
    roi: tuple[int, int, int, int] | None,
    pieces: list[PieceObservation] | None = None,
    mask: np.ndarray | None = None,
    overlay: np.ndarray | None = None,
    edges: np.ndarray | None = None,
) -> Path:
    """Save raw inputs, parameters, and optional diagnostic outputs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    case_path = Path(root) / stamp
    case_path.mkdir(parents=True, exist_ok=False)
    images = {"frame.png": frame, "background.png": background,
              "mask.png": mask, "overlay.png": overlay, "edges.png": edges}
    for name, image in images.items():
        if image is not None and not write_image(case_path / name, image):
            raise OSError(f"cannot save vision case image: {case_path / name}")
    document = {
        "format": "puzzle-device.vision-case.v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "roi": None if roi is None else list(roi),
        "detection_parameters": config.to_dict(),
        "pieces": [] if pieces is None else [piece.to_dict() for piece in pieces],
    }
    (case_path / "case.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return case_path


def load_vision_case(path: Path) -> VisionCase:
    path = Path(path)
    document = json.loads((path / "case.json").read_text(encoding="utf-8"))
    if document.get("format") != "puzzle-device.vision-case.v1":
        raise ValueError("unsupported vision case format")
    frame = read_image(path / "frame.png", cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("vision case has no readable frame.png")
    background_path = path / "background.png"
    background = read_image(background_path, cv2.IMREAD_COLOR)
    if background_path.exists() and background is None:
        raise ValueError("vision case background.png cannot be read")
    roi_values = document.get("roi")
    roi = None if roi_values is None else tuple(int(value) for value in roi_values)
    if roi is not None and len(roi) != 4:
        raise ValueError("vision case ROI must contain four integers")
    config = DetectionConfig.from_dict(document["detection_parameters"])
    return VisionCase(path, frame, background, config, roi)
