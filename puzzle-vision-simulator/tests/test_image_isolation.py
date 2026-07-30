"""Regression test: planning must work from saved pixels without generator state."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from puzzle_device.simulation.puzzle_sim import (
    CARD_H,
    CARD_W,
    DIVIDER_Y,
    PAPER_BGR,
    PIXELS_PER_CM,
    analyze_camera_frame,
    annotate_detection,
    detect_pieces,
    generate_camera_frame,
    random_cut,
    render_piece_poses,
    validate_cut_layout,
)


class ImageIsolationTest(unittest.TestCase):
    def test_saved_image_is_sufficient_for_planning(self):
        for piece_count in range(2, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(seed=100 + piece_count,
                                              piece_count=piece_count)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "camera.png"
                    self.assertTrue(cv2.imwrite(str(path), image))
                    reloaded = cv2.imread(str(path), cv2.IMREAD_COLOR)

                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)

    def test_joker_texture_is_detected_from_pixels_only(self):
        for piece_count in range(2, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(
                    seed=200 + piece_count,
                    piece_count=piece_count,
                    material_mode="joker")
                reloaded = cv2.imdecode(
                    cv2.imencode(".png", image)[1], cv2.IMREAD_COLOR)
                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)
                for transform in transforms:
                    # Each planned pose remains a proper rigid transform.
                    self.assertAlmostEqual(
                        float(np.linalg.det(transform[:2, :2])), 1.0, places=5)

    def test_rounded_card_corners_and_print_are_normalized(self):
        image = np.full((1200, 900, 3), PAPER_BGR, np.uint8)

        def add_card(center, size, radius, angle):
            width, height = size
            local = np.zeros((height + 80, width + 80), np.uint8)
            x, y = 40, 40
            cv2.rectangle(
                local, (x + radius, y),
                (x + width - radius, y + height), 255, -1)
            cv2.rectangle(
                local, (x, y + radius),
                (x + width, y + height - radius), 255, -1)
            for cx, cy in (
                    (x + radius, y + radius),
                    (x + width - radius, y + radius),
                    (x + radius, y + height - radius),
                    (x + width - radius, y + height - radius)):
                cv2.circle(local, (cx, cy), radius, 255, -1)
            rotation = cv2.getRotationMatrix2D(
                (local.shape[1] / 2, local.shape[0] / 2), angle, 1.0)
            rotated = cv2.warpAffine(
                local, rotation, local.shape[::-1],
                flags=cv2.INTER_LINEAR)
            ys, xs = np.where(rotated > 0)
            left, top = int(xs.min()), int(ys.min())
            right, bottom = int(xs.max()) + 1, int(ys.max()) + 1
            crop = rotated[top:bottom, left:right]
            ox = int(center[0] - crop.shape[1] / 2)
            oy = int(center[1] - crop.shape[0] / 2)
            region = image[oy:oy + crop.shape[0], ox:ox + crop.shape[1]]
            region[crop > 96] = (245, 245, 245)
            # Dense black/red face artwork remains strictly inside the stock.
            cv2.line(region, (35, crop.shape[0] // 2),
                     (crop.shape[1] - 35, crop.shape[0] // 2),
                     (5, 5, 5), 8, cv2.LINE_AA)
            cv2.circle(region, (crop.shape[1] // 2, crop.shape[0] // 2),
                       24, (20, 20, 150), -1, cv2.LINE_AA)

        add_card((230, 190), (260, 150), 20, 27)
        add_card((650, 360), (230, 140), 18, -31)
        self.assertLess(360 + 180, DIVIDER_Y)
        pieces = detect_pieces(image, expected_count=2)
        self.assertEqual([len(piece) for piece in pieces], [4, 4])
        recovered_sizes = []
        for piece in pieces:
            width, height = cv2.minAreaRect(
                piece.astype(np.float32))[1]
            recovered_sizes.append(sorted((width, height)))
        expected_sizes = [sorted((260, 150)), sorted((230, 140))]
        for recovered, expected in zip(recovered_sizes, expected_sizes):
            self.assertTrue(np.allclose(recovered, expected, atol=5.0),
                            (recovered, expected))

    def test_rounded_joker_is_solved_for_every_cut_topology(self):
        modes = (
            "common", "boundary_fan", "strips", "equal_rectangles",
            "t_junction", "corner", "concave")
        for mode in modes:
            with self.subTest(mode=mode):
                image = generate_camera_frame(
                    seed=17, piece_count=4,
                    material_mode="joker", cut_mode=mode)
                pieces, transforms, _ = analyze_camera_frame(image, mode)
                self.assertEqual(len(pieces), 4)
                self.assertEqual(len(transforms), 4)

    def test_joker_texture_and_both_contours_remain_visible(self):
        image = generate_camera_frame(
            31, 4, "joker", "equal_rectangles")
        pieces = detect_pieces(image, expected_count=4)
        annotated = annotate_detection(image, pieces)
        motion = render_piece_poses(
            image, pieces, [np.eye(3)] * 4, preserve_texture=True)

        def chromatic_pixels(frame):
            upper = frame[:DIVIDER_Y].astype(np.int16)
            return int(np.count_nonzero(
                upper.max(axis=2) - upper.min(axis=2) > 35))

        original_color = chromatic_pixels(image)
        self.assertGreater(original_color, 1000)
        self.assertGreater(chromatic_pixels(annotated), original_color * .75)
        self.assertGreater(chromatic_pixels(motion), original_color * .75)
        # Cyan is the measured rounded silhouette; yellow is the recovered
        # virtual polygon used by the geometric solver.
        cyan = cv2.inRange(
            annotated, np.array([245, 195, 0]), np.array([255, 225, 20]))
        yellow = cv2.inRange(
            annotated, np.array([0, 205, 245]), np.array([20, 235, 255]))
        self.assertGreater(cv2.countNonZero(cyan), 100)
        self.assertGreater(cv2.countNonZero(yellow), 100)

    def test_sequential_cuts_cover_non_common_vertex_layouts(self):
        found_without_global_vertex = False
        for seed in range(20):
            pieces = random_cut(np.random.default_rng(seed), 4)
            for piece in pieces:
                self.assertLessEqual(len(piece), 5)
                lengths = np.linalg.norm(
                    np.roll(piece, -1, axis=0) - piece, axis=1)
                self.assertGreaterEqual(lengths.min(), 2 * PIXELS_PER_CM - 1e-6)
            candidates = pieces[0]
            common = [
                point for point in candidates
                if all(any(np.linalg.norm(point - other) < 1e-5
                           for other in piece)
                       for piece in pieces[1:])
            ]
            if not common:
                found_without_global_vertex = True
                break
        self.assertTrue(found_without_global_vertex)

    def test_all_cut_categories_are_visually_solvable(self):
        for mode in (
                "common", "boundary_fan", "strips", "equal_rectangles",
                "t_junction", "corner", "concave"):
            with self.subTest(mode=mode):
                image = generate_camera_frame(
                    seed=7, piece_count=4, material_mode="color",
                    cut_mode=mode)
                pieces, transforms, matches = analyze_camera_frame(image, mode)
                self.assertEqual(len(pieces), 4)
                self.assertEqual(len(transforms), 4)
                partial_count = sum(
                    tuple(match[5:]) != (0., 1., 0., 1.)
                    for match in matches)
                self.assertEqual(partial_count, 1 if mode == "t_junction" else 0)
                if mode == "concave":
                    self.assertTrue(any(
                        not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2))
                        for p in pieces))

    def test_field_geometry_requirements_for_every_category(self):
        modes = (
            "common", "boundary_fan", "strips", "equal_rectangles",
            "t_junction", "corner", "concave")
        for mode in modes:
            for piece_count in range(2, 5):
                for seed in range(30):
                    with self.subTest(
                            mode=mode, pieces=piece_count, seed=seed):
                        validate_cut_layout(random_cut(
                            np.random.default_rng(seed), piece_count, mode))

    def test_concave_outline_survives_pixel_only_detection(self):
        seen_vertex_counts = set()
        for piece_count in range(2, 5):
            for seed in range(30):
                with self.subTest(pieces=piece_count, seed=seed):
                    image = generate_camera_frame(
                        seed, piece_count, "color", "concave")
                    pieces, transforms, _ = analyze_camera_frame(
                        image, "concave")
                    self.assertEqual(len(pieces), piece_count)
                    self.assertEqual(len(transforms), piece_count)
                    self.assertTrue(any(
                        not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2))
                        for p in pieces))
                    seen_vertex_counts.update(
                        len(p) for p in pieces
                        if not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2)))
        self.assertTrue({4, 5}.issubset(seen_vertex_counts))

    def test_four_identical_rectangles_are_solved_without_texture(self):
        source = random_cut(np.random.default_rng(0), 4, "equal_rectangles")
        widths = []
        heights = []
        for piece in source:
            low, high = piece.min(axis=0), piece.max(axis=0)
            widths.append(high[0] - low[0])
            heights.append(high[1] - low[1])
        self.assertTrue(np.allclose(widths, CARD_W / 2))
        self.assertTrue(np.allclose(heights, CARD_H / 2))
        for seed in range(20):
            image = generate_camera_frame(
                seed, 4, "color", "equal_rectangles")
            pieces, transforms, _ = analyze_camera_frame(
                image, "equal_rectangles")
            self.assertEqual(len(pieces), 4)
            self.assertEqual(len(transforms), 4)


if __name__ == "__main__":
    unittest.main()
