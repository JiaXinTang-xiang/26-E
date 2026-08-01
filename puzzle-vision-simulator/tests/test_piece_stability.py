"""Tests for multi-frame piece stability and averaging."""

import unittest

import numpy as np

from puzzle_device.vision.piece_vision import PieceObservation
from puzzle_device.vision.stability import PieceStabilityTracker


def make_piece(
    x: float, y: float, angle: float = 12.0, area: float = 12000.0,
    vertex_noise: np.ndarray | None = None,
):
    polygon = np.array([[x - 10, y - 10], [x + 10, y - 10], [x, y + 10]])
    if vertex_noise is not None:
        polygon = polygon + vertex_noise
    return PieceObservation(
        piece_id=0,
        contour=polygon.astype(np.int32).reshape(-1, 1, 2),
        polygon=polygon,
        mask=np.zeros((200, 200), np.uint8),
        center=(x, y),
        pick_point=(x + 1, y + 1),
        pick_clearance_px=15.0,
        area_px=area,
        pca_angle_deg=angle,
        longest_edge_angle_deg=angle,
        bounding_box=(round(x - 10), round(y - 10), 20, 20),
        confidence=0.95,
    )


class PieceStabilityTest(unittest.TestCase):
    def test_default_requires_four_frames(self):
        tracker = PieceStabilityTracker()
        self.assertEqual(tracker.required_frames, 4)

    def test_stable_frames_can_be_averaged(self):
        tracker = PieceStabilityTracker(required_frames=4)
        for offset in (0.0, 1.0, -1.0, 0.5):
            status = tracker.update([make_piece(100 + offset, 80 - offset, 12 + offset)])
        self.assertTrue(status.stable)
        averaged = tracker.averaged_observations()
        self.assertEqual(len(averaged), 1)
        self.assertAlmostEqual(averaged[0].center[0], 100.125)
        self.assertAlmostEqual(averaged[0].center[1], 79.875)

    def test_motion_restarts_stability_count(self):
        tracker = PieceStabilityTracker(required_frames=4, center_tolerance_px=3.0)
        tracker.update([make_piece(100, 80)])
        tracker.update([make_piece(101, 80)])
        status = tracker.update([make_piece(120, 80)])
        self.assertFalse(status.stable)
        self.assertEqual(status.sample_count, 1)
        self.assertIn("中心波动", status.reason)

    def test_polygon_vertices_are_aligned_and_median_stabilized(self):
        tracker = PieceStabilityTracker(required_frames=4, polygon_tolerance_px=4.0)
        noises = (
            np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
            np.zeros((3, 2)),
            np.array([[-1.0, 1.0], [0.0, -1.0], [1.0, 0.0]]),
        )
        for index, noise in enumerate(noises):
            piece = make_piece(100, 80, vertex_noise=noise)
            piece.polygon = np.roll(piece.polygon, index % 3, axis=0)
            status = tracker.update([piece])
        self.assertTrue(status.stable)
        polygon = tracker.averaged_observations()[0].polygon
        expected = make_piece(100, 80).polygon
        self.assertLess(np.linalg.norm(polygon - expected, axis=1).max(), 1.1)

    def test_large_vertex_jitter_restarts_sampling(self):
        tracker = PieceStabilityTracker(required_frames=3, polygon_tolerance_px=3.0)
        tracker.update([make_piece(100, 80)])
        noise = np.array([[8.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        status = tracker.update([make_piece(100, 80, vertex_noise=noise)])
        self.assertFalse(status.stable)
        self.assertEqual(status.sample_count, 1)
        self.assertIn("顶点波动", status.reason)


if __name__ == "__main__":
    unittest.main()
