"""Tests for physical A4 assembly and movement-plan generation."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from puzzle_device.planning import (
    AssemblyConfig,
    build_movement_plan,
    draw_assembly_preview,
    draw_card_candidate_gallery,
    solve_composite_card_assembly,
    solve_assembly,
    solve_textured_assembly,
)
from puzzle_device.planning.assembly import (
    _apply,
    _assembly_worker_count,
    _choose_candidate,
    _edge_candidates,
    _matched_seams_are_valid,
    _texture_seam_scores,
    a4_to_global_pixels,
    transform_global_points,
)
from puzzle_device.planning.composite_card import _rerank_geometry_qualified_layouts
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
    def test_jetson_defaults_to_two_geometry_workers(self):
        with patch.dict("os.environ", {}, clear=True), \
                patch("puzzle_device.planning.assembly.platform.machine", return_value="aarch64"), \
                patch("puzzle_device.planning.assembly.os.cpu_count", return_value=6):
            self.assertEqual(_assembly_worker_count(4), 2)
            self.assertEqual(_assembly_worker_count(3), 2)
            self.assertEqual(_assembly_worker_count(2), 1)

    def test_worker_environment_override_wins_on_jetson(self):
        with patch.dict("os.environ", {"PUZZLE_ASSEMBLY_WORKERS": "1"}, clear=True), \
                patch("puzzle_device.planning.assembly.platform.machine", return_value="aarch64"):
            self.assertEqual(_assembly_worker_count(4), 1)
        with patch.dict("os.environ", {"PUZZLE_ASSEMBLY_WORKERS": "2"}, clear=True):
            self.assertEqual(_assembly_worker_count(2), 1)

    def test_relaxed_card_profile_accepts_noisy_card_ratio_candidate(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [89, 0], [89, 58], [0, 58]], float), ROI, config
        )
        image = np.full((720, 1280, 3), 230, np.uint8)
        plan = solve_textured_assembly(
            image, [polygon], ROI, config, require_upper_half=False
        )
        self.assertTrue(np.allclose(plan.recovered_size_mm, [89, 58], atol=0.2))
        self.assertIsNotNone(plan.texture_score)

    def test_card_candidate_gallery_renders_ranked_diagnostics(self):
        config = AssemblyConfig()
        polygons_a4 = [
            np.array([[0, 0], [44, 0], [44, 58], [0, 58]], float),
            np.array([[44, 0], [89, 0], [89, 58], [44, 58]], float),
        ]
        polygons = [a4_to_global_pixels(polygon, ROI, config) for polygon in polygons_a4]
        image = np.full((720, 1280, 3), 230, np.uint8)
        plan = solve_textured_assembly(
            image, polygons, ROI, config, require_upper_half=False
        )
        self.assertGreaterEqual(len(plan.candidate_diagnostics), 1)
        self.assertEqual(plan.candidate_diagnostics[0]["rank"], 1)
        gallery = draw_card_candidate_gallery(image, plan, config)
        self.assertEqual(gallery.ndim, 3)
        self.assertGreater(gallery.shape[0], 0)
        self.assertGreater(gallery.shape[1], 0)

    def test_composite_card_method_is_separate_and_solves_simple_sliding_case(self):
        config = AssemblyConfig()
        polygons_a4 = [
            np.array([[0, 0], [44, 0], [44, 57], [0, 57]], float),
            np.array([[44, 0], [88, 0], [88, 57], [44, 57]], float),
        ]
        polygons = [a4_to_global_pixels(polygon, ROI, config) for polygon in polygons_a4]
        image = np.full((720, 1280, 3), 230, np.uint8)
        with patch.dict("os.environ", {"PUZZLE_CARD2_WORKERS": "1"}):
            plan = solve_composite_card_assembly(
                image, polygons, ROI, config, require_upper_half=False
            )
        self.assertTrue(np.allclose(plan.recovered_size_mm, [88, 57], atol=0.2))
        self.assertGreater(plan.rectangle_fill_ratio, 0.99)
        self.assertGreaterEqual(len(plan.candidate_diagnostics), 1)

    def test_composite_card_texture_breaks_tie_between_strong_rectangles(self):
        identity = np.eye(3)
        placed = [np.zeros((3, 2), dtype=float) for _ in range(4)]

        def item(total, geometry, texture, metrics):
            return (
                total, geometry, (identity,) * 4, (), placed,
                metrics, texture, (texture,) * 3,
            )

        geometrically_best_but_wrong = item(
            154.6, 117.9, 0.663,
            ((85.7, 55.8), 0.971, 0.991, 0.979, 0.0, 0.079),
        )
        artwork_continuous = item(
            283.8, 247.1, 0.625,
            ((88.5, 55.8), 0.940, 0.985, 0.954, 0.0, 0.167),
        )
        weak_outer_shape = item(
            360.0, 330.0, 0.610,
            ((86.0, 55.8), 0.965, 1.0, 0.850, 0.0, 0.29),
        )
        ranked = _rerank_geometry_qualified_layouts([
            geometrically_best_but_wrong, artwork_continuous, weak_outer_shape,
        ])
        self.assertIs(ranked[0], artwork_continuous)
        self.assertIs(ranked[2], weak_outer_shape)

    def test_card_ranking_prefers_repeated_layout_family_over_isolated_candidate(self):
        config = AssemblyConfig(
            card_family_bonus_per_support=12.0,
            card_family_bonus_cap=36.0,
        )
        identity = np.eye(3)
        square = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]], float)

        def candidate(score, centers):
            placed = tuple(square + np.asarray(center) for center in centers)
            return (
                score,
                (identity, identity, identity, identity),
                placed,
                (88.0, 57.0),
                0.94,
                0.97,
                0.96,
                0.0,
                0.0,
                (),
            )

        isolated = candidate(10.0, [(0, 0), (28, 0), (0, 20), (28, 20)])
        correct_family = [
            candidate(
                20.0 + offset,
                [(0, 0), (20 + offset * 0.05, 0), (0, 28), (20, 28)],
            )
            for offset in (0.0, 1.0, 2.0, 3.0)
        ]
        image = np.full((720, 1280, 3), 230, np.uint8)
        selected, _texture, _seams, diagnostics = _choose_candidate(
            [isolated, *correct_family],
            [square.copy() for _ in range(4)],
            ROI,
            config,
            image,
            prefer_fixed_card_shape=False,
            card_mode=True,
        )
        self.assertIs(selected, correct_family[0])
        self.assertEqual(diagnostics[0]["family_support"], 4)
        self.assertEqual(diagnostics[0]["consensus_bonus"], 36.0)

    def test_two_piece_search_keeps_all_complete_edge_pairs(self):
        config = AssemblyConfig(two_piece_edge_relative_tolerance=1.0)
        first = np.array([[0, 0], [5, 0], [5, 4], [0, 4]], dtype=float)
        second = np.array([[7, 0], [12, 0], [12, 4], [7, 4]], dtype=float)
        candidates = _edge_candidates([first, second], config)[(0, 1)]
        full = [candidate for candidate in candidates if candidate[5:] == (0.0, 1.0, 0.0, 1.0)]
        self.assertEqual(len(full), 16)

    def test_seam_validation_requires_real_opposing_contact(self):
        config = AssemblyConfig()
        first = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        second = np.array([[2, 0], [4, 0], [4, 2], [2, 2]], dtype=float)
        match = ((0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0),)
        self.assertTrue(_matched_seams_are_valid(
            [first, second], [np.eye(3), np.eye(3)], [first, second], match, config,
        ))
        shifted = np.eye(3)
        shifted[0, 2] = 2.0
        self.assertFalse(_matched_seams_are_valid(
            [first, second], [np.eye(3), shifted], [first, _apply(second, shifted)], match, config,
        ))

    def test_self_mode_prefers_original_100_by_60_shape_without_rejecting(self):
        config = AssemblyConfig()
        identity = np.eye(3)
        full_matches = ((0.01, 0, 0, 1, 0, 0.0, 1.0, 0.0, 1.0),)
        skinny = (8.0, (identity, identity), (), (105.0, 49.0),
                  0.9, 0.95, 0.9, 0.0, 0.0, full_matches)
        original_shape = (12.0, (identity, identity), (), (100.0, 60.0),
                          0.9, 0.95, 0.9, 0.0, 0.0, full_matches)
        selected, _texture, _seams, _diagnostics = _choose_candidate(
            [skinny, original_shape],
            [np.zeros((4, 2)), np.zeros((4, 2))],
            ROI,
            config,
            None,
            prefer_fixed_card_shape=True,
        )
        self.assertEqual(selected[3], (100.0, 60.0))

    def test_full_edge_candidate_is_preferred_over_partial_edge(self):
        config = AssemblyConfig()
        identity = np.eye(3)
        full = ((0.05, 0, 0, 1, 0, 0.0, 1.0, 0.0, 1.0),)
        partial = ((0.01, 0, 0, 1, 0, 0.0, 0.6, 0.0, 1.0),)
        full_candidate = (10.0, (identity, identity), (), (100.0, 60.0),
                          0.9, 0.95, 0.9, 0.0, 0.0, full)
        partial_candidate = (1.0, (identity, identity), (), (100.0, 60.0),
                             0.9, 0.95, 0.9, 0.0, 0.0, partial)
        selected, _texture, _seams, _diagnostics = _choose_candidate(
            [partial_candidate, full_candidate],
            [np.zeros((4, 2)), np.zeros((4, 2))],
            ROI,
            config,
            None,
            prefer_fixed_card_shape=False,
        )
        self.assertIs(selected, full_candidate)

    def test_card_texture_prefers_continuous_seam_direction(self):
        config = AssemblyConfig()
        image = np.full((720, 1280, 3), 230, np.uint8)
        polygons = [
            np.array([[10, 10], [60, 10], [60, 70], [10, 70]], float),
            np.array([[80, 10], [130, 10], [130, 70], [80, 70]], float),
        ]
        # Give both source edges the same asymmetric printed pattern from top
        # to bottom. Reversing one edge must therefore produce a worse seam.
        for y_mm in np.linspace(10, 70, 121):
            color = int(30 + (y_mm - 10) / 60 * 190)
            for x_mm in (57.5, 82.5):
                x_px, y_px = a4_to_global_pixels(
                    np.array([[x_mm, y_mm]], float), ROI, config
                )[0].astype(int)
                cv2.circle(image, (x_px, y_px), 5, (color, color, color), -1)
        continuous = ((0.0, 0, 1, 1, 3, 0.0, 1.0, 0.0, 1.0),)
        reversed_pattern = ((0.0, 0, 1, 1, 1, 0.0, 1.0, 0.0, 1.0),)
        continuous_score = _texture_seam_scores(
            image, polygons, continuous, ROI, config
        )[0]
        reversed_score = _texture_seam_scores(
            image, polygons, reversed_pattern, ROI, config
        )[0]
        self.assertLess(continuous_score, reversed_score * 0.45)

    def test_non_rectangular_arrangement_is_rejected(self):
        config = AssemblyConfig()
        polygons = [a4_to_global_pixels(
            np.array([[0, 0], [100, 0], [50, 60]], float), ROI, config
        )]
        with self.assertRaises(RuntimeError):
            solve_assembly(polygons, ROI, config, require_upper_half=False)

    def test_120_by_72_rectangle_is_accepted_and_becomes_target(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [120, 0], [120, 72], [0, 72]], float), ROI, config
        )
        plan = solve_assembly([polygon], ROI, config, require_upper_half=False)
        self.assertTrue(np.allclose(plan.recovered_size_mm, [120, 72], atol=0.1))
        self.assertTrue(np.allclose(plan.target_rect_mm[2:], [120, 72], atol=0.1))
        self.assertAlmostEqual(plan.rectangle_fill_ratio, 1.0, places=5)

    def test_non_five_to_three_rectangle_is_accepted(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [90, 0], [90, 80], [0, 80]], float), ROI, config
        )
        plan = solve_assembly([polygon], ROI, config, require_upper_half=False)
        self.assertTrue(np.allclose(plan.recovered_size_mm, [90, 80], atol=0.1))
        self.assertTrue(np.allclose(plan.target_rect_mm[2:], [90, 80], atol=0.1))

    def test_width_outside_task_range_is_rejected(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [130, 0], [130, 70], [0, 70]], float), ROI, config
        )
        with self.assertRaisesRegex(RuntimeError, "尺寸范围"):
            solve_assembly([polygon], ROI, config, require_upper_half=False)

    def test_height_outside_task_range_is_rejected(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [100, 0], [100, 95], [0, 95]], float), ROI, config
        )
        with self.assertRaisesRegex(RuntimeError, "尺寸范围"):
            solve_assembly([polygon], ROI, config, require_upper_half=False)

    def test_visual_size_tolerance_accepts_small_measurement_error(self):
        config = AssemblyConfig()
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [88.1, 0], [88.1, 58.6], [0, 58.6]], float), ROI, config
        )
        plan = solve_assembly([polygon], ROI, config, require_upper_half=False)
        self.assertTrue(np.allclose(plan.recovered_size_mm, [88.1, 58.6], atol=0.1))

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
        self.assertTrue(np.allclose(
            document["target_rect"]["size_mm"], assembly.recovered_size_mm, atol=0.01
        ))
        self.assertEqual(
            document["quality"]["target_size_range_mm"],
            {"width": [90.0, 120.0], "height": [50.0, 90.0]},
        )
        self.assertTrue(document["quality"]["recovered_size_in_range"])

    def test_execution_gap_separates_neighbours_without_changing_rotation(self):
        config = AssemblyConfig(placement_gap_mm=5.0)
        source = [
            np.array([[0, 0], [50, 0], [50, 60], [0, 60]], float),
            np.array([[50, 0], [100, 0], [100, 60], [50, 60]], float),
        ]
        placed = _place(source, config)
        observations = []
        for index, polygon in enumerate(placed):
            observations.append(PieceObservation(
                piece_id=index,
                contour=np.round(polygon).astype(np.int32),
                polygon=polygon,
                mask=np.zeros((720, 1280), np.uint8),
                center=tuple(polygon.mean(axis=0)),
                pick_point=tuple(polygon.mean(axis=0)),
                pick_clearance_px=20.0,
                area_px=5000.0,
                pca_angle_deg=0.0,
                longest_edge_angle_deg=0.0,
                bounding_box=(0, 0, 1, 1),
                confidence=1.0,
            ))
        assembly = solve_assembly(placed, ROI, config, require_upper_half=False)
        document = build_movement_plan(observations, assembly, config=config)
        quality = document["quality"]
        self.assertGreaterEqual(quality["placement_gap_actual_mm"], 4.99)
        self.assertLessEqual(quality["maximum_corresponding_vertex_distance_mm"], 20.0)
        offsets = [np.asarray(piece["target_offset_a4_mm"]) for piece in document["pieces"]]
        self.assertTrue(all(np.linalg.norm(offset) <= 12.01 for offset in offsets))
        self.assertTrue(all(
            abs(piece["rotation_deg"] - angle) < 1e-3
            for piece, angle in zip(
                document["pieces"],
                [
                    np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
                    for transform in assembly.transforms
                ],
            )
        ))

    def test_three_strips_keep_centre_piece_fixed_and_create_gap(self):
        config = AssemblyConfig(placement_gap_mm=5.0)
        source = [
            np.array([[0, 0], [100 / 3, 0], [100 / 3, 60], [0, 60]], float),
            np.array([[100 / 3, 0], [200 / 3, 0], [200 / 3, 60], [100 / 3, 60]], float),
            np.array([[200 / 3, 0], [100, 0], [100, 60], [200 / 3, 60]], float),
        ]
        placed = _place(source, config)
        observations = [
            PieceObservation(
                piece_id=index,
                contour=np.round(polygon).astype(np.int32),
                polygon=polygon,
                mask=np.zeros((720, 1280), np.uint8),
                center=tuple(polygon.mean(axis=0)),
                pick_point=tuple(polygon.mean(axis=0)),
                pick_clearance_px=20.0,
                area_px=5000.0,
                pca_angle_deg=0.0,
                longest_edge_angle_deg=0.0,
                bounding_box=(0, 0, 1, 1),
                confidence=1.0,
            )
            for index, polygon in enumerate(placed)
        ]
        assembly = solve_assembly(placed, ROI, config, require_upper_half=False)
        document = build_movement_plan(observations, assembly, config=config)

        offsets = [np.asarray(piece["target_offset_a4_mm"]) for piece in document["pieces"]]
        centre_index = min(range(3), key=lambda index: np.linalg.norm(offsets[index]))
        self.assertTrue(np.allclose(offsets[centre_index], [0.0, 0.0], atol=1e-3))
        self.assertGreaterEqual(document["quality"]["placement_gap_actual_mm"], 4.99)
        self.assertLessEqual(
            document["quality"]["maximum_corresponding_vertex_distance_mm"], 20.0
        )

    def test_zero_execution_gap_preserves_ideal_targets(self):
        config = AssemblyConfig(placement_gap_mm=0.0)
        polygon = a4_to_global_pixels(
            np.array([[0, 0], [100, 0], [100, 60], [0, 60]], float), ROI, config
        )
        observation = PieceObservation(
            piece_id=0,
            contour=np.round(polygon).astype(np.int32),
            polygon=polygon,
            mask=np.zeros((720, 1280), np.uint8),
            center=tuple(polygon.mean(axis=0)),
            pick_point=tuple(polygon.mean(axis=0)),
            pick_clearance_px=20.0,
            area_px=6000.0,
            pca_angle_deg=0.0,
            longest_edge_angle_deg=0.0,
            bounding_box=(0, 0, 1, 1),
            confidence=1.0,
        )
        assembly = solve_assembly([polygon], ROI, config, require_upper_half=False)
        document = build_movement_plan([observation], assembly, config=config)
        self.assertEqual(document["pieces"][0]["target_offset_a4_mm"], [0.0, 0.0])
        self.assertEqual(document["quality"]["placement_gap_actual_mm"], 0.0)


if __name__ == "__main__":
    unittest.main()
