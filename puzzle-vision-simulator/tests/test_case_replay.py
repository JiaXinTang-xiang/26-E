"""Tests for saving and loading real-camera replay cases."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.vision.case_replay import load_vision_case, save_vision_case
from puzzle_device.vision.piece_vision import DetectionConfig, detect_piece_observations


class VisionCaseReplayTest(unittest.TestCase):
    def test_case_round_trip_can_be_detected_offline(self):
        background = np.full((240, 360, 3), (20, 145, 225), np.uint8)
        frame = background.copy()
        cv2.rectangle(frame, (80, 60), (250, 180), (245, 245, 245), cv2.FILLED)
        config = DetectionConfig(segmentation_method="white_hsv", min_area_px=500)
        with tempfile.TemporaryDirectory() as directory:
            path = save_vision_case(
                Path(directory), frame, background, config, (30, 20, 280, 190)
            )
            case = load_vision_case(path)
            pieces, _mask = detect_piece_observations(
                case.frame, case.background, case.config, roi=case.roi
            )
        self.assertEqual(len(pieces), 1)
        self.assertEqual(case.roi, (30, 20, 280, 190))
        self.assertEqual(case.config.to_dict(), config.to_dict())


if __name__ == "__main__":
    unittest.main()
