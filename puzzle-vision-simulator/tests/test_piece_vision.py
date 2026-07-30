"""Regression tests for piece segmentation and geometry extraction."""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.simulation.puzzle_sim import DIVIDER_Y, generate_camera_frame
from puzzle_device.vision.piece_vision import (
    DetectionConfig,
    detect_piece_observations,
    extract_piece_edges,
    load_detection_config,
    save_detection_config,
)


class PieceVisionTest(unittest.TestCase):
    def test_detects_all_simulated_piece_counts(self):
        for piece_count in range(1, 5):
            for seed in range(10, 20):
                with self.subTest(piece_count=piece_count, seed=seed):
                    image = generate_camera_frame(seed, piece_count)[:DIVIDER_Y]
                    pieces, mask = detect_piece_observations(
                        image, config=DetectionConfig(min_area_px=3000.0)
                    )
                    self.assertEqual(len(pieces), piece_count)
                    self.assertEqual(mask.shape, image.shape[:2])
                    for piece in pieces:
                        self.assertGreaterEqual(len(piece.polygon), 3)
                        self.assertLessEqual(len(piece.polygon), 5)
                        self.assertTrue(np.isfinite(piece.pca_angle_deg))
                        self.assertTrue(np.isfinite(piece.longest_edge_angle_deg))
                        cx, cy = np.round(piece.center).astype(int)
                        self.assertGreater(piece.mask[cy, cx], 0)
                        px, py = np.round(piece.pick_point).astype(int)
                        self.assertGreater(piece.mask[py, px], 0)
                        self.assertGreater(piece.pick_clearance_px, 0)

    def test_background_subtraction_detects_white_textured_pieces(self):
        background = np.full((480, 640, 3), (45, 70, 80), np.uint8)
        image = background.copy()
        polygons = [
            np.array([[70, 70], [250, 55], [225, 185], [90, 205]], np.int32),
            np.array([[350, 235], [550, 255], [520, 410], [330, 385]], np.int32),
        ]
        for polygon in polygons:
            cv2.fillPoly(image, [polygon], (240, 240, 240))
            texture = image.copy()
            cv2.line(texture, tuple(polygon[0] + [12, 12]),
                     tuple(polygon[2] - [12, 12]), (20, 20, 20), 5)
            polygon_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(polygon_mask, [polygon], 255)
            image[polygon_mask > 0] = texture[polygon_mask > 0]

        pieces, _mask = detect_piece_observations(image, background)
        self.assertEqual(len(pieces), 2)
        self.assertTrue(all(len(piece.polygon) == 4 for piece in pieces))
        self.assertTrue(all(piece.pick_clearance_px > 20 for piece in pieces))

    def test_safe_pick_point_avoids_concave_notch(self):
        background = np.full((300, 300, 3), (40, 70, 90), np.uint8)
        image = background.copy()
        # A concave L-shape whose polygon centroid lies near the missing corner.
        polygon = np.array(
            [[40, 40], [250, 40], [250, 110], [120, 110], [120, 250], [40, 250]],
            np.int32,
        )
        cv2.fillPoly(image, [polygon], (245, 245, 245))
        pieces, _mask = detect_piece_observations(
            image, background, DetectionConfig(min_area_px=500, max_vertices=6)
        )
        self.assertEqual(len(pieces), 1)
        piece = pieces[0]
        self.assertGreater(piece.pick_clearance_px, 35)
        pick = tuple(np.round(piece.pick_point).astype(int))
        self.assertGreater(piece.mask[pick[1], pick[0]], 0)

    def test_white_hsv_and_brightness_modes_on_orange_background(self):
        background = np.full((300, 420, 3), (20, 145, 225), np.uint8)
        image = background.copy()
        cv2.rectangle(image, (80, 70), (320, 235), (245, 245, 245), cv2.FILLED)
        for method in ("white_hsv", "brightness"):
            with self.subTest(method=method):
                config = DetectionConfig(
                    segmentation_method=method,
                    min_area_px=500,
                    white_saturation_max=90,
                    white_value_min=170,
                    brightness_min=185,
                )
                pieces, mask = detect_piece_observations(image, None, config)
                self.assertEqual(len(pieces), 1)
                self.assertGreater(mask[150, 200], 0)
                self.assertEqual(mask[20, 20], 0)

    def test_roi_excludes_distractor_and_preserves_global_coordinates(self):
        image = np.full((360, 600, 3), (20, 145, 225), np.uint8)
        cv2.rectangle(image, (120, 90), (260, 240), (245, 245, 245), cv2.FILLED)
        cv2.rectangle(image, (430, 80), (560, 230), (245, 245, 245), cv2.FILLED)
        config = DetectionConfig(
            segmentation_method="white_hsv",
            min_area_px=500,
            white_saturation_max=90,
            white_value_min=170,
        )
        pieces, mask = detect_piece_observations(image, None, config, roi=(80, 50, 240, 240))
        self.assertEqual(len(pieces), 1)
        self.assertGreater(pieces[0].center[0], 120)
        self.assertGreater(pieces[0].center[1], 90)
        self.assertGreater(mask[150, 180], 0)
        self.assertEqual(mask[150, 490], 0)
        self.assertEqual(pieces[0].mask.shape, image.shape[:2])

    def test_maximum_area_excludes_large_false_region(self):
        image = np.full((360, 600, 3), (20, 145, 225), np.uint8)
        cv2.rectangle(image, (40, 60), (190, 210), (245, 245, 245), cv2.FILLED)
        cv2.rectangle(image, (280, 35), (570, 325), (245, 245, 245), cv2.FILLED)
        config = DetectionConfig(
            segmentation_method="white_hsv",
            min_area_px=500,
            max_area_px=30000,
            white_saturation_max=90,
            white_value_min=170,
        )
        pieces, _mask = detect_piece_observations(image, None, config)
        self.assertEqual(len(pieces), 1)
        self.assertLess(pieces[0].area_px, config.max_area_px)

    def test_shared_detection_config_round_trip(self):
        config = DetectionConfig(
            segmentation_method="white_hsv",
            color_distance_threshold=28,
            white_saturation_max=70,
            white_value_min=180,
            brightness_min=190,
            morphology_size=5,
            canny_lower=35,
            canny_upper=110,
            polygon_epsilon_min=0.006,
            polygon_epsilon_preferred=0.02,
            polygon_epsilon_max=0.045,
            minimum_pick_clearance_px=10.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vision.json"
            save_detection_config(path, config)
            loaded = load_detection_config(path)
        self.assertEqual(loaded.to_dict(), config.to_dict())

    def test_edges_follow_mask_boundary_not_piece_texture(self):
        mask = np.zeros((200, 260), np.uint8)
        cv2.rectangle(mask, (40, 35), (220, 165), 255, cv2.FILLED)
        edges = extract_piece_edges(mask)
        self.assertGreater(np.count_nonzero(edges), 0)
        self.assertEqual(np.count_nonzero(edges[60:140, 70:190]), 0)


if __name__ == "__main__":
    unittest.main()
