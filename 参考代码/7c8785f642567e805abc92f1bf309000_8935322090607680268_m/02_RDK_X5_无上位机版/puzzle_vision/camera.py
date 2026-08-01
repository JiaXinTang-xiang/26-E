from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


def _read_image(path: str) -> np.ndarray:
    # np.fromfile + imdecode also supports non-ASCII paths on Windows.
    data = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise CameraError(f"Cannot read image: {path}")
    return frame


class UsbCamera:
    def __init__(self, device: str | int, settings: dict[str, Any]):
        self.device = device
        self.settings = settings
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            raise CameraError(f"Cannot open USB camera: {self.device}")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.settings["width"]))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.settings["height"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(self.settings["fps"]))
        for _ in range(int(self.settings.get("warmup_frames", 8))):
            self.cap.read()

    def read(self) -> np.ndarray:
        if self.cap is None:
            self.open()
        assert self.cap is not None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError(f"Failed to read USB camera: {self.device}")
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MipiCamera:
    """RDK X5 MIPI source using the board-provided hobot_vio binding."""

    def __init__(self, settings: dict[str, Any]):
        self.settings = settings
        self.cam: Any = None

    def open(self) -> None:
        try:
            from hobot_vio import libsrcampy
        except ImportError as exc:
            raise CameraError("hobot_vio.libsrcampy is not installed") from exc

        self.cam = libsrcampy.Camera()
        ret = self.cam.open_cam(
            0,
            -1,
            int(self.settings["fps"]),
            int(self.settings["width"]),
            int(self.settings["height"]),
        )
        if ret:
            self.cam = None
            raise CameraError(
                "Failed to open MIPI camera. Check the ribbon cable, sensor "
                "compatibility, and that only one MIPI sensor is connected."
            )
        for _ in range(int(self.settings.get("warmup_frames", 8))):
            self._read_nv12()

    def _read_nv12(self) -> bytes:
        if self.cam is None:
            self.open()
        assert self.cam is not None
        width = int(self.settings["width"])
        height = int(self.settings["height"])
        data = self.cam.get_img(2, width, height)
        if data is None:
            # Some hobot_vio versions expose the source channel as channel 1.
            data = self.cam.get_img(1)
        if data is None:
            raise CameraError("MIPI camera returned no image")
        return data

    def read(self) -> np.ndarray:
        width = int(self.settings["width"])
        height = int(self.settings["height"])
        data = self._read_nv12()
        raw = np.frombuffer(data, dtype=np.uint8)
        expected = width * height * 3 // 2
        if raw.size < expected:
            raise CameraError(
                f"Unexpected MIPI frame size: {raw.size}, expected {expected}"
            )
        nv12 = raw[:expected].reshape(height * 3 // 2, width)
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

    def close(self) -> None:
        if self.cam is not None:
            self.cam.close_cam()
            self.cam = None


def _first_working_usb(settings: dict[str, Any]) -> UsbCamera | None:
    candidates: list[str | int] = sorted(glob.glob("/dev/video*"))
    if os.name == "nt":
        candidates.extend(range(5))
    for candidate in candidates:
        source = UsbCamera(candidate, settings)
        try:
            source.open()
            return source
        except CameraError:
            source.close()
    return None


def capture_frame(source_spec: str, settings: dict[str, Any]) -> np.ndarray:
    if Path(source_spec).is_file():
        return _read_image(source_spec)

    source: UsbCamera | MipiCamera | None = None
    try:
        if source_spec == "auto":
            source = _first_working_usb(settings)
            if source is None:
                source = MipiCamera(settings)
                source.open()
        elif source_spec == "mipi":
            source = MipiCamera(settings)
            source.open()
        elif source_spec.startswith("usb:"):
            value = source_spec.split(":", 1)[1]
            device: str | int = int(value) if value.isdigit() else value
            source = UsbCamera(device, settings)
            source.open()
        else:
            raise CameraError(
                "Unknown source. Use auto, mipi, usb:/dev/video0, usb:0, or an image path."
            )
        return source.read()
    finally:
        if source is not None:
            source.close()
