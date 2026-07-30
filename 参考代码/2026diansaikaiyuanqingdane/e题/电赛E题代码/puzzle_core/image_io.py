from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from a Windows path that may contain Chinese characters."""
    image_path = Path(path)
    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image to a Windows path that may contain Chinese characters."""
    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {suffix}: {image_path}")
    encoded.tofile(image_path)
