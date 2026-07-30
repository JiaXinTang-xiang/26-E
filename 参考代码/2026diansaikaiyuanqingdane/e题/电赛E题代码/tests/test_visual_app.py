from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from visual_app import MODE_OPTIONS, collect_result_bundle


class VisualAppTests(unittest.TestCase):
    def test_mode_options_cover_all_solvers(self) -> None:
        self.assertEqual(
            set(MODE_OPTIONS.values()),
            {"self", "field-white", "field-card"},
        )

    def test_result_bundle_orders_movement_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.png"
            output = root / "output"
            steps = output / "movement_steps"
            steps.mkdir(parents=True)
            for path in (
                input_path,
                output / "detected.png",
                output / "solved.png",
                steps / "step_02.png",
                steps / "step_00.png",
                steps / "step_01.png",
            ):
                path.write_bytes(b"test")
            (output / "movement_plan.json").write_text(
                json.dumps({"mode": "field-white", "pieces": []}),
                encoding="utf-8",
            )
            bundle = collect_result_bundle(input_path, output)
            self.assertEqual(
                [path.name for path in bundle.movement_steps],
                ["step_00.png", "step_01.png", "step_02.png"],
            )
            self.assertEqual(bundle.plan["mode"], "field-white")


if __name__ == "__main__":
    unittest.main()
