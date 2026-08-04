"""OpenCV image I/O that also supports non-ASCII Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image without passing a Unicode path through ``cv2.imread``."""
    image_path = Path(path)
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, flags)
    except cv2.error:
        return None


def write_image(path: str | Path, image: np.ndarray) -> bool:
    """Write an image through ``imencode`` so Chinese paths work on Windows."""
    image_path = Path(path)
    suffix = image_path.suffix or ".png"
    try:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            return False
        encoded.tofile(image_path)
        return True
    except (OSError, cv2.error):
        return False
