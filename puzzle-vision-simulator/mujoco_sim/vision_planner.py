"""Camera-only visual detection and puzzle planning for the MuJoCo scene."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle_sim import analyze_camera_frame, apply_h  # noqa: E402
from mujoco_sim.scene_builder import (  # noqa: E402
    CAMERA_FOVY, PIXEL_TO_METER, TARGET_CENTER)


@dataclass
class PiecePlan:
    visual_id: int
    pickup_xy: np.ndarray
    target_xy: np.ndarray
    target_yaw: float


def pixel_to_world(pixel_xy, width=900, height=1200, camera_z=.650,
                   plane_z=.010, fovy_deg=CAMERA_FOVY):
    """Analytic projection for the fixed vertical MuJoCo camera."""
    vertical_span = 2 * (camera_z - plane_z) * math.tan(
        math.radians(fovy_deg) / 2)
    scale = vertical_span / height
    x = (pixel_xy[0] - width / 2) * scale
    y = (height / 2 - pixel_xy[1]) * scale
    return np.array([x, y])


def plan_from_camera_rgb(rgb_image: np.ndarray, cut_mode: str = "auto"):
    """No simulator state or generated geometry is accepted by this function."""
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    # MuJoCo's headlight can saturate the blue material to V=255. Apply a
    # camera-like exposure correction before the unchanged core detector.
    vision_bgr = np.clip(bgr.astype(np.float32) * .85, 0, 255).astype(np.uint8)
    pieces, transforms, matches = analyze_camera_frame(vision_bgr, cut_mode)
    plans = plans_from_solution(
        pieces, transforms, rgb_image.shape[1], rgb_image.shape[0])
    return plans, pieces, transforms, matches, vision_bgr


def plans_from_solution(pieces, transforms, image_width: int,
                        image_height: int):
    """Convert one camera-only SE(2) solution to calibrated robot plans.

    Keeping this conversion separate lets the GUI and MuJoCo executor consume
    the exact same visual solution instead of solving an ambiguous geometry
    twice and potentially choosing different equivalent piece orders.
    """
    plans = []
    restored = [apply_h(piece, transform)
                for piece, transform in zip(pieces, transforms)]
    restored_points = np.vstack(restored)
    restored_min = restored_points.min(axis=0)
    restored_max = restored_points.max(axis=0)
    target_pixel_center = (restored_min + restored_max) / 2
    for i, (piece, transform) in enumerate(zip(pieces, transforms)):
        current_center_px = piece.mean(axis=0)
        target_center_px = apply_h(current_center_px[None], transform)[0]
        pickup_xy = pixel_to_world(
            current_center_px, width=image_width, height=image_height)
        # Preserve one uniform physical scale. Rounded-corner recovery and
        # antialiasing can make the measured union a few pixels narrower or
        # taller than 400x240. Independently normalizing those two measured
        # dimensions would anisotropically stretch the planned body centers,
        # while the rigid MuJoCo meshes keep their original size; adjacent
        # fragments would then show gaps or overlap. The source card and
        # MuJoCo meshes both use the exact 10 cm / 400 px scale.
        target_xy = TARGET_CENTER + np.array([
            (target_center_px[0] - target_pixel_center[0])
            * PIXEL_TO_METER,
            -(target_center_px[1] - target_pixel_center[1])
            * PIXEL_TO_METER,
        ])
        # Image y points downward while MuJoCo world y points upward, so a
        # positive image-plane rotation is a negative world yaw.
        yaw = -math.atan2(transform[1, 0], transform[0, 0])
        plans.append(PiecePlan(i, pickup_xy, target_xy, yaw))
    return plans
