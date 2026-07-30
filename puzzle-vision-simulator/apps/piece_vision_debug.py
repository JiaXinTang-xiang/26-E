#!/usr/bin/env python3
"""OpenCV-only live debugger for white puzzle pieces on an orange A4 sheet.

This tool never opens a serial port and never moves the gantry.  It exposes
the segmentation and contour stages as independent windows so thresholds can
be tuned against the real camera before changing the production detector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.vision.camera import open_uvc_camera


WINDOW_MAIN = "Puzzle Debug - Main"
WINDOW_BG = "Puzzle Debug - Background Difference"
WINDOW_WHITE = "Puzzle Debug - White HSV"
WINDOW_BRIGHT = "Puzzle Debug - Brightness"
WINDOW_EDGES = "Puzzle Debug - Canny Edges"
WINDOW_MASK = "Puzzle Debug - Selected Mask"
WINDOW_CONTROLS = "Puzzle Debug - Controls"
SETTINGS_PATH = Path("configs/local/piece_vision_debug.json")


@dataclass
class DebugSettings:
    """Runtime sliders; all values are intentionally camera-pixel based."""

    method: int = 0  # 0 background difference, 1 white HSV, 2 brightness.
    difference_threshold: int = 25
    white_s_max: int = 85
    white_v_min: int = 170
    brightness_min: int = 185
    blur_size: int = 5
    morphology_size: int = 3
    canny_lower: int = 50
    canny_upper: int = 150
    minimum_area: int = 1000
    epsilon_percent: int = 2

    def validate(self) -> None:
        self.method = int(np.clip(self.method, 0, 2))
        self.difference_threshold = int(np.clip(self.difference_threshold, 1, 255))
        self.white_s_max = int(np.clip(self.white_s_max, 0, 255))
        self.white_v_min = int(np.clip(self.white_v_min, 0, 255))
        self.brightness_min = int(np.clip(self.brightness_min, 0, 255))
        self.blur_size = max(1, int(self.blur_size) | 1)
        self.morphology_size = max(1, int(self.morphology_size) | 1)
        self.canny_lower = int(np.clip(self.canny_lower, 0, 254))
        self.canny_upper = int(np.clip(self.canny_upper, self.canny_lower + 1, 255))
        self.minimum_area = max(1, int(self.minimum_area))
        self.epsilon_percent = int(np.clip(self.epsilon_percent, 1, 10))


def _empty_callback(_value: int) -> None:
    pass


def _make_trackbars(settings: DebugSettings) -> None:
    cv2.namedWindow(WINDOW_CONTROLS, cv2.WINDOW_NORMAL)
    for name, value, maximum in (
        ("method 0=BG 1=HSV 2=bright", settings.method, 2),
        ("background difference", settings.difference_threshold, 255),
        ("white HSV S max", settings.white_s_max, 255),
        ("white HSV V min", settings.white_v_min, 255),
        ("brightness min", settings.brightness_min, 255),
        ("blur size", settings.blur_size, 31),
        ("morphology size", settings.morphology_size, 31),
        ("Canny lower", settings.canny_lower, 254),
        ("Canny upper", settings.canny_upper, 255),
        ("minimum contour area", settings.minimum_area, 30000),
        ("polygon epsilon percent", settings.epsilon_percent, 10),
    ):
        cv2.createTrackbar(name, WINDOW_CONTROLS, value, maximum, _empty_callback)


def _read_trackbars() -> DebugSettings:
    settings = DebugSettings(
        method=cv2.getTrackbarPos("method 0=BG 1=HSV 2=bright", WINDOW_CONTROLS),
        difference_threshold=cv2.getTrackbarPos("background difference", WINDOW_CONTROLS),
        white_s_max=cv2.getTrackbarPos("white HSV S max", WINDOW_CONTROLS),
        white_v_min=cv2.getTrackbarPos("white HSV V min", WINDOW_CONTROLS),
        brightness_min=cv2.getTrackbarPos("brightness min", WINDOW_CONTROLS),
        blur_size=cv2.getTrackbarPos("blur size", WINDOW_CONTROLS),
        morphology_size=cv2.getTrackbarPos("morphology size", WINDOW_CONTROLS),
        canny_lower=cv2.getTrackbarPos("Canny lower", WINDOW_CONTROLS),
        canny_upper=cv2.getTrackbarPos("Canny upper", WINDOW_CONTROLS),
        minimum_area=cv2.getTrackbarPos("minimum contour area", WINDOW_CONTROLS),
        epsilon_percent=cv2.getTrackbarPos("polygon epsilon percent", WINDOW_CONTROLS),
    )
    settings.validate()
    return settings


def _lab_difference(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = frame_lab - background_lab
    return np.sqrt(np.mean(delta * delta, axis=2)).clip(0, 255).astype(np.uint8)


def _morphology(mask: np.ndarray, size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cleaned)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


def _safe_pick(mask: np.ndarray) -> tuple[tuple[int, int], float] | None:
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _min_value, maximum, _min_loc, maximum_location = cv2.minMaxLoc(distance)
    if maximum <= 0:
        return None
    return maximum_location, float(maximum)


def _draw_contours(frame: np.ndarray, mask: np.ndarray, settings: DebugSettings) -> tuple[np.ndarray, int]:
    output = frame.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kept = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings.minimum_area:
            continue
        kept += 1
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, perimeter * settings.epsilon_percent / 100.0, True)
        cv2.drawContours(output, [contour], -1, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.polylines(output, [polygon], True, (0, 0, 255), 2, cv2.LINE_AA)
        for point in polygon.reshape(-1, 2):
            cv2.circle(output, tuple(point), 5, (0, 0, 255), -1, cv2.LINE_AA)

        component = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, cv2.FILLED)
        moments = cv2.moments(contour)
        if moments["m00"]:
            center = (round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"]))
            cv2.drawMarker(output, center, (40, 255, 40), cv2.MARKER_CROSS, 16, 2)
        pick = _safe_pick(component)
        if pick is not None:
            point, clearance = pick
            cv2.circle(output, point, max(5, round(clearance)), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(output, point, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(output, f"P{kept} {len(polygon)} corners safe={clearance:.0f}px",
                        (point[0] + 10, max(20, point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(output, f"P{kept} {len(polygon)} corners safe={clearance:.0f}px",
                        (point[0] + 10, max(20, point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output, kept


def _save_settings(settings: DebugSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_settings() -> DebugSettings:
    if not SETTINGS_PATH.exists():
        return DebugSettings()
    try:
        settings = DebugSettings(**json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        settings.validate()
        return settings
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return DebugSettings()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--no-rotate-180", action="store_true", help="use raw camera orientation")
    args = parser.parse_args()

    settings = _load_settings()
    _make_trackbars(settings)
    capture, camera_info = open_uvc_camera(args.camera)
    if capture is None:
        raise SystemExit(f"cannot open camera {args.camera}")
    if camera_info is not None:
        print(f"Camera: {camera_info.describe()}")

    background: np.ndarray | None = None
    print("Keys: G=capture empty orange-A4 background, S=save sliders, Q/Esc=quit")
    print("Method: 0=background difference (recommended), 1=white HSV, 2=brightness")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue
            if not args.no_rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            settings = _read_trackbars()

            blur = cv2.GaussianBlur(frame, (settings.blur_size, settings.blur_size), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            if background is not None:
                difference = _lab_difference(blur, background)
                _, background_mask = cv2.threshold(
                    difference, settings.difference_threshold, 255, cv2.THRESH_BINARY
                )
            else:
                difference = np.zeros(gray.shape, np.uint8)
                background_mask = np.zeros(gray.shape, np.uint8)
            white_mask = cv2.inRange(
                hsv, np.array([0, 0, settings.white_v_min]),
                np.array([179, settings.white_s_max, 255]),
            )
            _, brightness_mask = cv2.threshold(gray, settings.brightness_min, 255, cv2.THRESH_BINARY)
            raw_mask = (background_mask, white_mask, brightness_mask)[settings.method]
            mask = _morphology(raw_mask, settings.morphology_size)
            edges = cv2.Canny(gray, settings.canny_lower, settings.canny_upper, apertureSize=3)
            annotated, count = _draw_contours(frame, mask, settings)

            method_name = ("BG difference", "white HSV", "brightness")[settings.method]
            state = "background captured" if background is not None else "press G with empty orange A4"
            cv2.putText(annotated, f"{method_name} | pieces={count} | {state}", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"{method_name} | pieces={count} | {state}", (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_MAIN, annotated)
            cv2.imshow(WINDOW_BG, difference)
            cv2.imshow(WINDOW_WHITE, white_mask)
            cv2.imshow(WINDOW_BRIGHT, brightness_mask)
            cv2.imshow(WINDOW_EDGES, edges)
            cv2.imshow(WINDOW_MASK, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("g"):
                background = blur.copy()
                print("Background captured: remove all white pieces before pressing G.")
            if key == ord("s"):
                _save_settings(settings)
                print(f"Saved: {SETTINGS_PATH.resolve()}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
