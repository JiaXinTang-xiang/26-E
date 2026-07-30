"""OpenCV UVC camera initialization shared by calibration and detection apps."""

from __future__ import annotations

from dataclasses import dataclass
import sys

import cv2


@dataclass(frozen=True)
class CameraInfo:
    backend: str
    width: int
    height: int
    fps: float
    fourcc: str

    def describe(self) -> str:
        fps_text = f"{self.fps:.0f}" if self.fps > 0 else "未知"
        return (
            f"{self.backend} / {self.fourcc or '未知格式'} / "
            f"{self.width}x{self.height} / {fps_text} FPS"
        )


def _decode_fourcc(value: float) -> str:
    number = int(round(value))
    text = "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4))
    return "".join(character for character in text if character.isprintable()).strip()


def _backend_candidates() -> list[int]:
    if sys.platform.startswith("win"):
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform.startswith("linux"):
        return [cv2.CAP_V4L2, cv2.CAP_ANY]
    return [cv2.CAP_ANY]


def open_uvc_camera(
    camera_index: int,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
) -> tuple[cv2.VideoCapture | None, CameraInfo | None]:
    """Open a USB/UVC camera and request a low-latency MJPG stream."""
    capture: cv2.VideoCapture | None = None
    for backend in _backend_candidates():
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()
    if capture is None:
        return None, None

    # MJPG usually avoids saturating USB bandwidth at 1280x720.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        backend_name = capture.getBackendName()
    except cv2.error:
        backend_name = "OpenCV"
    info = CameraInfo(
        backend=backend_name,
        width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(capture.get(cv2.CAP_PROP_FPS)),
        fourcc=_decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
    )
    return capture, info
