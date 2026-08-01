"""Tests for fixed four-piece self-template matching."""

import math
import unittest

import numpy as np

from puzzle_device.planning import AssemblyConfig, build_movement_plan, solve_self_assembly
from puzzle_device.planning.assembly import a4_to_global_pixels, global_pixels_to_a4
from puzzle_device.planning.template_assembly import load_self_piece_template
from puzzle_device.vision.piece_vision import PieceObservation


ROI = (100, 20, 420, 594)


def _rigid(angle: float, translation: tuple[float, float]) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, translation[0]],
        [s, c, translation[1]],
        [0.0, 0.0, 1.0],
    ])


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (np.c_[points, np.ones(len(points))] @ transform.T)[:, :2]


class TemplateAssemblyTest(unittest.TestCase):
    def test_template_recovers_rotated_and_permuted_pieces(self):
        template = load_self_piece_template()
        order = (2, 0, 3, 1)
        observed_a4 = []
        for index, template_index in enumerate(order):
            transform = _rigid(
                math.radians((index + 1) * 31.0),
                (35.0 + index * 38.0, 28.0 + (index % 2) * 52.0),
            )
            polygon = template.pieces[template_index]
            center = polygon.mean(axis=0)
            observed_a4.append(_apply(polygon - center, transform))
        observed_px = [a4_to_global_pixels(polygon, ROI) for polygon in observed_a4]
        plan = solve_self_assembly(observed_px, ROI, require_upper_half=False)
        placed = [
            _apply(global_pixels_to_a4(polygon, ROI), transform)
            for polygon, transform in zip(observed_px, plan.transforms)
        ]
        bounds = np.vstack(placed)
        self.assertEqual(len(plan.transforms), 4)
        self.assertTrue(np.allclose(bounds.max(0) - bounds.min(0), [100, 60], atol=0.1))
        self.assertGreater(plan.rectangle_fill_ratio, 0.95)
        self.assertLess(plan.overlap_ratio, 0.02)

    def test_template_tolerates_one_extra_contour_vertex(self):
        template = load_self_piece_template()
        polygons = [piece.copy() for piece in template.pieces]
        midpoint = (polygons[0][0] + polygons[0][1]) / 2.0
        polygons[0] = np.insert(polygons[0], 1, midpoint + [0.0, 0.4], axis=0)
        observed_px = [a4_to_global_pixels(polygon + [45, 25], ROI) for polygon in polygons]
        plan = solve_self_assembly(observed_px, ROI, require_upper_half=False)
        self.assertEqual(len(plan.transforms), 4)
        self.assertGreater(plan.rectangle_fill_ratio, 0.95)

    def test_template_target_is_rotated_180_degrees_in_place(self):
        template = load_self_piece_template()
        polygons = [a4_to_global_pixels(piece + [45, 25], ROI) for piece in template.pieces]
        plan = solve_self_assembly(polygons, ROI, require_upper_half=False)
        placed = [
            _apply(global_pixels_to_a4(polygon, ROI), transform)
            for polygon, transform in zip(polygons, plan.transforms)
        ]
        # fixed_1 normally occupies the top-left corner; 180 degrees moves it
        # to the matching bottom-right corner while the target rectangle stays.
        self.assertGreater(placed[0].mean(axis=0)[0], 130.0)
        self.assertGreater(placed[0].mean(axis=0)[1], 235.0)
        self.assertAlmostEqual(plan.target_rect_mm[0], 47.0, places=3)

    def test_template_movement_plan_creates_safe_gap_within_limits(self):
        template = load_self_piece_template()
        polygons = [a4_to_global_pixels(piece + [45, 25], ROI) for piece in template.pieces]
        pieces = []
        for index, polygon in enumerate(polygons):
            center = tuple(polygon.mean(axis=0))
            pieces.append(PieceObservation(
                piece_id=index,
                contour=np.round(polygon).astype(np.int32).reshape(-1, 1, 2),
                polygon=polygon,
                mask=np.zeros((720, 1280), np.uint8),
                center=center,
                pick_point=center,
                pick_clearance_px=20.0,
                area_px=10000.0,
                pca_angle_deg=0.0,
                longest_edge_angle_deg=0.0,
                bounding_box=(0, 0, 1, 1),
                confidence=1.0,
            ))
        plan = solve_self_assembly(polygons, ROI, require_upper_half=False)
        document = build_movement_plan(pieces, plan, config=AssemblyConfig())
        quality = document["quality"]
        self.assertAlmostEqual(quality["placement_gap_actual_mm"], 5.0, places=3)
        self.assertGreaterEqual(quality["maximum_corresponding_vertex_distance_mm"], 0.0)

    def test_template_rejects_wrong_piece_count(self):
        template = load_self_piece_template()
        observed_px = [a4_to_global_pixels(piece, ROI) for piece in template.pieces[:3]]
        with self.assertRaisesRegex(ValueError, "4块"):
            solve_self_assembly(observed_px, ROI, require_upper_half=False)


if __name__ == "__main__":
    unittest.main()
