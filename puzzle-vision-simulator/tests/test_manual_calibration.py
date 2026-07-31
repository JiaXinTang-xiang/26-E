"""Tests for camera-pixel to gantry-pulse homography fitting."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from puzzle_device.calibration.manual_calibration import (
    CalibrationPoint,
    PixelToGantryCalibration,
)


class ManualCalibrationTest(unittest.TestCase):
    def test_recovers_projective_mapping_and_rejects_bad_click(self):
        pixels = np.array([
            [50, 40], [300, 50], [610, 60], [60, 220], [310, 230], [600, 215],
            [45, 440], [300, 430], [620, 445],
        ], dtype=np.float32)
        matrix = np.array([
            [4.2, 0.18, 130.0], [-0.12, 3.6, 90.0], [0.00035, -0.00022, 1.0],
        ])
        homogeneous = np.c_[pixels, np.ones(len(pixels))] @ matrix.T
        pulses = homogeneous[:, :2] / homogeneous[:, 2, None]
        calibration = PixelToGantryCalibration([
            CalibrationPoint(*pixel, *pulse, "source")
            for pixel, pulse in zip(pixels, pulses)
        ])
        calibration.add_point(CalibrationPoint(500, 350, 20, 20, "destination"))
        metrics = calibration.fit(ransac_threshold_pulse=3.0)

        self.assertEqual(metrics.inlier_count, len(pixels))
        predicted = calibration.predict_pulse(410, 150)
        expected_h = np.array([410, 150, 1.0]) @ matrix.T
        expected = expected_h[:2] / expected_h[2]
        np.testing.assert_allclose(predicted, expected, atol=0.01)

    def test_saved_calibration_contains_mapping_and_points(self):
        calibration = PixelToGantryCalibration([
            CalibrationPoint(0, 0, 10, 20), CalibrationPoint(100, 0, 110, 20),
            CalibrationPoint(100, 100, 110, 120), CalibrationPoint(0, 100, 10, 120),
        ])
        calibration.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path, {"camera": "test"})
            self.assertIn("matrix_pixel_to_pulse", path.read_text(encoding="utf-8"))

    def test_affine_average_stays_finite_outside_measured_points(self):
        calibration = PixelToGantryCalibration([
            CalibrationPoint(0, 0, 100, 200),
            CalibrationPoint(100, 0, 100, 500),
            CalibrationPoint(0, 100, 500, 200),
            CalibrationPoint(100, 100, 500, 500),
        ])
        metrics = calibration.fit_affine_average()
        self.assertEqual(metrics.inlier_count, 4)
        np.testing.assert_allclose(calibration.predict_pulse(50, 50), [300, 350])
        prediction = np.asarray(calibration.predict_pulse(500, 500))
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
