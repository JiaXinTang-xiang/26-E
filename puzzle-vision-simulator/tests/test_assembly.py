"""Tests for physical A4 assembly and movement-plan generation."""

import unittest

import numpy as np

from puzzle_device.planning import AssemblyConfig, build_movement_plan, solve_assembly
from puzzle_device.planning.assembly import (
    a4_to_global_pixels,
    transform_global_points,
)
from puzzle_device.simulation.puzzle_sim import apply_h, random_cut, rigid
from puzzle_device.vision.piece_vision import PieceObservation


ROI = (100, 20, 420, 594)


def _place(polygons, config):
    placed = []
    for index, polygon in enumerate(polygons):
        center = polygon.mean(axis=0)
        transform = rigid(0.35 + index * 0.43, 35 + index * 40, 40 + index * 10)
        placed.append(a4_to_global_pixels(apply_h(polygon - center, transform), ROI, config))
    return placed


class AssemblyTest(unittest.TestCase):
    def test_non_rectangular_arrangement_is_rejected(self):
        config = AssemblyConfig()
        polygons = [a4_to_global_pixels(
            np.array([[0, 0], [100, 0], [50, 60]], float), ROI, config
        )]
        with self.assertRaises(RuntimeError):
            solve_assembly(polygons, ROI, config, require_upper_half=False)

    def test_oversized_rectangle_is_accepted(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [120, 0], [120, 72], [0, 72]], float), ROI, config
        )
        plan = solve_assembly([polygon], ROI, config, require_upper_half=False)
        self.assertTrue(np.allclose(plan.recovered_size_mm, [120, 72], atol=0.1))
        self.assertAlmostEqual(plan.rectangle_fill_ratio, 1.0, places=5)

    def test_random_common_cuts_recover_target_rectangle(self):
        config = AssemblyConfig()
        for count in range(1, 5):
            with self.subTest(count=count):
                if count == 1:
                    source = [np.array([[0, 0], [100, 0], [100, 60], [0, 60]], float)]
                else:
                    source = [polygon / 4.0 for polygon in random_cut(
                        np.random.default_rng(700 + count), count
                    )]
                placed = _place(source, config)
                plan = solve_assembly(placed, ROI, config, require_upper_half=False)
                output = [
                    transform_global_points(polygon, plan, transform, config)
                    for polygon, transform in zip(placed, plan.transforms)
                ]
                a4_output = [
                    (polygon - [ROI[0], ROI[1]])
                    * [config.a4_width_mm / ROI[2], config.a4_height_mm / ROI[3]]
                    for polygon in output
                ]
                points = np.vstack(a4_output)
                self.assertTrue(np.allclose(points.max(0) - points.min(0), [100, 60], atol=0.1))

    def test_partial_edges_support_t_junction(self):
        config = AssemblyConfig()
        source = [
            np.array([[0, 0], [100, 0], [100, 30], [0, 30]], float),
            np.array([[0, 30], [50, 30], [50, 60], [0, 60]], float),
            np.array([[50, 30], [100, 30], [100, 60], [50, 60]], float),
        ]
        placed = _place(source, config)
        plan = solve_assembly(placed, ROI, config, require_upper_half=False)
        self.assertEqual(len(plan.matches), 2)
        self.assertTrue(any(
            match[6] - match[5] < 0.99 or match[8] - match[7] < 0.99
            for match in plan.matches
        ))
        output = [
            transform_global_points(polygon, plan, transform, config)
            for polygon, transform in zip(placed, plan.transforms)
        ]
        points = np.vstack([
            (polygon - [ROI[0], ROI[1]]) * [0.5, 0.5] for polygon in output
        ])
        self.assertTrue(np.allclose(points.max(0) - points.min(0), [100, 60], atol=0.1))

    def test_lower_half_piece_is_rejected(self):
        config = AssemblyConfig()
        lower = a4_to_global_pixels(
            np.array([[80, 180], [130, 180], [130, 220], [80, 220]], float), ROI, config
        )
        with self.assertRaisesRegex(ValueError, "上半区"):
            solve_assembly([lower], ROI, config)

    def test_movement_plan_contains_pixels_pulses_rotation_and_safety_flag(self):
        config = AssemblyConfig()
        source = np.array([[30, 30], [130, 30], [130, 90], [30, 90]], float)
        polygon = a4_to_global_pixels(source, ROI, config)
        mask = np.zeros((720, 1280), np.uint8)
        observation = PieceObservation(
            piece_id=0,
            contour=np.round(polygon).astype(np.int32),
            polygon=polygon,
            mask=mask,
            center=tuple(polygon.mean(axis=0)),
            pick_point=tuple(polygon.mean(axis=0)),
            pick_clearance_px=20.0,
            area_px=10000.0,
            pca_angle_deg=0.0,
            longest_edge_angle_deg=0.0,
            bounding_box=(0, 0, 1, 1),
            confidence=1.0,
        )
        assembly = solve_assembly([polygon], ROI, config)
        document = build_movement_plan(
            [observation], assembly,
            pulse_mapper=lambda point: (round(point[0] * 2), round(point[1] * 3)),
            calibration_file="calibration.json", config=config,
        )
        self.assertFalse(document["motor_commands_sent"])
        self.assertFalse(document["rotation_axis_controlled"])
        self.assertEqual(len(document["pieces"]), 1)
        record = document["pieces"][0]
        self.assertIsNotNone(record["source_pick_pulse"])
        self.assertIsNotNone(record["target_pick_pulse"])
        self.assertIn("rotation_deg", record)


if __name__ == "__main__":
    unittest.main()
