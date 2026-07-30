from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from puzzle_core.generation import generate_case
from puzzle_core.geometry import edge_lengths, polygon_area, self_piece_polygons
from puzzle_core.image_io import read_image
from puzzle_core.pipeline import solve_image
from puzzle_core.solver import solve_layout
from puzzle_core.vision import classify_mode, detect_pieces


class GeometryTests(unittest.TestCase):
    def test_fixed_self_geometry(self) -> None:
        pieces, width, height = self_piece_polygons()
        self.assertEqual(len(pieces), 4)
        self.assertEqual((width, height), (100.0, 60.0))
        self.assertAlmostEqual(
            sum(polygon_area(piece) for piece in pieces),
            width * height,
            places=3,
        )
        all_points = np.vstack(pieces)
        self.assertTrue(np.any(np.all(np.isclose(all_points, [36.0, 12.0]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(all_points, [76.0, 42.0]), axis=1)))

    def test_invalid_field_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for count in (0, 5):
                with self.subTest(count=count):
                    with self.assertRaises(ValueError):
                        generate_case("field-white", 1, count, temp)


class PipelineTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = generate_case("field-card", 20260729, 4, root / "a")
            second = generate_case("field-card", 20260729, 4, root / "b")
            digest_a = hashlib.sha256((first / "input.png").read_bytes()).digest()
            digest_b = hashlib.sha256((second / "input.png").read_bytes()).digest()
            self.assertEqual(digest_a, digest_b)
            self.assertEqual(
                (first / "ground_truth.json").read_text(encoding="utf-8"),
                (second / "ground_truth.json").read_text(encoding="utf-8"),
            )

    def test_counts_one_to_four(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for count in (1, 2, 3, 4):
                with self.subTest(count=count):
                    case = generate_case("field-white", 3100 + count, count, root)
                    image = read_image(case / "input.png")
                    pieces, _ = detect_pieces(image, 5.0)
                    mode = classify_mode(image, pieces)
                    layout = solve_layout(pieces, mode, 5.0)
                    truth = json.loads(
                        (case / "ground_truth.json").read_text(encoding="utf-8")
                    )
                    target = truth["target_rect"]
                    self.assertEqual(len(pieces), count)
                    self.assertEqual(mode, "field-white")
                    self.assertLess(abs(layout.width_mm - target["width_mm"]), 4.0)
                    self.assertLess(abs(layout.height_mm - target["height_mm"]), 4.0)
                    for record in truth["pieces"]:
                        self.assertLessEqual(
                            len(record["target_polygon_local_mm"]), 5
                        )
                        self.assertGreaterEqual(
                            min(record["edge_lengths_mm"]), 19.999
                        )

    def test_three_default_modes_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, count in (
                ("self", 4),
                ("field-white", 4),
                ("field-card", 4),
            ):
                with self.subTest(kind=kind):
                    case = generate_case(kind, 20260729, count, root / "images")
                    # The pipeline receives only the image path and output path.
                    output = root / "outputs" / case.name
                    result = solve_image(case / "input.png", output)
                    self.assertEqual(result["mode"], kind)
                    self.assertEqual(result["detected_count"], count)
                    self.assertTrue((output / "detected.png").is_file())
                    self.assertTrue((output / "solved.png").is_file())
                    self.assertTrue((output / "movement_plan.json").is_file())
                    self.assertEqual(
                        len(list((output / "movement_steps").glob("step_*.png"))),
                        count + 1,
                    )
                    plan = json.loads(
                        (output / "movement_plan.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(len(plan["pieces"]), count)
                    self.assertEqual(plan["target_rect"]["center_mm"], [105.0, 222.75])


if __name__ == "__main__":
    unittest.main()
