#!/usr/bin/env python3
"""Print the Jetson runtime capabilities needed by the puzzle program."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import cv2
import numpy as np

from puzzle_device.vision.camera import open_uvc_camera
from puzzle_device.vision.cuda_ops import cuda_status
from puzzle_device.paths import LOCAL_CONFIG_DIR, LOCAL_DATA_DIR, OUTPUT_DIR


def _check_runtime_files() -> None:
    required = (
        LOCAL_CONFIG_DIR / "a4_roi.json",
        LOCAL_CONFIG_DIR / "calibration.json",
        LOCAL_CONFIG_DIR / "vision_detection.json",
        LOCAL_DATA_DIR / "empty_work_area.png",
    )
    for path in required:
        print(f"Runtime file: {'OK' if path.exists() else 'MISSING'} - {path}")
    roi_path = LOCAL_CONFIG_DIR / "a4_roi.json"
    calibration_path = LOCAL_CONFIG_DIR / "calibration.json"
    try:
        roi = json.loads(roi_path.read_text(encoding="utf-8"))
        print(f"ROI rotation: {roi.get('camera_rotation_degrees', 'unknown')} degrees")
    except (OSError, json.JSONDecodeError):
        pass
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        metadata = calibration.get("metadata", {})
        print(
            "Calibration source: "
            f"camera={metadata.get('camera_index', 'unknown')}, "
            f"rotation={metadata.get('camera_rotation_degrees', 'unknown')} degrees"
        )
    except (OSError, json.JSONDecodeError):
        pass
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = OUTPUT_DIR / ".jetson_write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        print(f"Output directory: WRITABLE - {OUTPUT_DIR}")
    except OSError as exc:
        print(f"Output directory: NOT WRITABLE ({exc}) - {OUTPUT_DIR}")


def _print_display_info() -> None:
    display = os.environ.get("DISPLAY")
    print(f"DISPLAY: {display or 'MISSING'}")
    if not display:
        print("GUI display: FAILED (start from the Jetson desktop, not a plain SSH shell)")
        return
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        print(f"Screen: {root.winfo_screenwidth()}x{root.winfo_screenheight()}")
        root.destroy()
    except Exception as exc:
        print(f"Screen query: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args()

    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print(
            "Python compatibility: FAILED (this project needs Python 3.10+; "
            "Jetson Nano JetPack 4 normally provides Python 3.6)"
        )
    else:
        print("Python compatibility: OK (3.10+)")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")

    available, reason = cuda_status()
    try:
        device_count = cv2.cuda.getCudaEnabledDeviceCount()
    except (AttributeError, cv2.error):
        device_count = 0
    print(f"OpenCV CUDA devices: {device_count}")
    print(f"Puzzle CUDA pipeline: {'available' if available else 'CPU fallback'} ({reason})")

    try:
        import tkinter  # noqa: F401
        print("Tkinter: OK")
    except ImportError as exc:
        print(f"Tkinter: MISSING ({exc})")

    try:
        import serial
        print(f"pyserial: {getattr(serial, '__version__', 'installed')}")
    except ImportError:
        print("pyserial: MISSING")

    _print_display_info()
    _check_runtime_files()

    if args.skip_camera:
        return
    capture, info = open_uvc_camera(args.camera)
    if capture is None:
        print(f"Camera {args.camera}: OPEN FAILED")
        return
    try:
        ok, frame = capture.read()
        print(f"Camera {args.camera}: {info.describe() if info else 'opened'}")
        print(f"Camera frame: {frame.shape[1]}x{frame.shape[0]}" if ok else "Camera frame: READ FAILED")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
