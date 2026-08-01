"""Tests for requirement 1(1) upper-to-lower transfer planning."""

import unittest

import numpy as np

from puzzle_device.planning import AssemblyConfig, build_execution_tasks, build_transfer_plan
from puzzle_device.planning.transfer import FIXED_DROP_POINTS_A4_MM
from puzzle_device.vision.piece_vision import PieceObservation


ROI = (100, 20, 420, 594)


def _piece(piece_id: int, x: float, y: float) -> PieceObservation:
    polygon = np.array([[x, y], [x + 30, y], [x + 30, y + 24], [x, y + 24]], float)
    center = tuple(polygon.mean(axis=0))
    return PieceObservation(
        piece_id=piece_id,
        contour=np.round(polygon).astype(np.int32),
        polygon=polygon,
        mask=np.zeros((720, 1280), np.uint8),
        center=center,
        pick_point=center,
        pick_clearance_px=10.0,
        area_px=720.0,
        pca_angle_deg=12.0,
        longest_edge_angle_deg=0.0,
        bounding_box=(round(x), round(y), 30, 24),
        confidence=1.0,
    )


class TransferPlanTest(unittest.TestCase):
    def test_four_pieces_use_four_fixed_drop_points_without_rotation(self):
        pieces = [
            _piece(0, 130, 50),
            _piece(1, 210, 70),
            _piece(2, 300, 100),
            _piece(3, 400, 120),
        ]
        document, targets = build_transfer_plan(
            pieces, ROI, pulse_mapper=lambda point: (round(point[0]), round(point[1]))
        )
        self.assertEqual(document["operation_mode"], "transfer_only")
        self.assertEqual(len(document["pieces"]), 4)
        self.assertEqual(
            document["quality"]["fixed_drop_points_a4_mm"],
            [list(point) for point in FIXED_DROP_POINTS_A4_MM],
        )
        self.assertEqual(len(targets), 4)
        for record in document["pieces"]:
            self.assertEqual(record["rotation_deg"], 0.0)
            self.assertEqual(record["fixed_drop_index"], record["sequence"] - 1)
        tasks = build_execution_tasks(document)
        self.assertTrue(all(task.pick_angle_deg == task.place_angle_deg == 135 for task in tasks))

    def test_current_machine_fixed_points_stay_inside_pulse_range(self):
        roi = (425, 36, 458, 667)
        matrix = np.array([
            [-0.3638243881, 4.5634218094, -447.029927],
            [3.2170126206, 0.1693902852, -1260.8691854],
            [0.0, 0.0, 1.0],
        ])
        pieces = [
            _piece(0, 470, 100),
            _piece(1, 560, 120),
            _piece(2, 650, 140),
            _piece(3, 760, 160),
        ]
        def mapper(point):
            pulse = matrix @ np.array([point[0], point[1], 1.0])
            return round(pulse[0]), round(pulse[1])
        document, _targets = build_transfer_plan(pieces, roi, mapper)
        for record in document["pieces"]:
            x, y = record["target_pick_pulse"]
            self.assertTrue(0 <= x <= 2350)
            self.assertTrue(0 <= y <= 1350)

    def test_transfer_requires_exactly_four_pieces(self):
        with self.assertRaisesRegex(ValueError, "4块"):
            build_transfer_plan(
                [_piece(0, 130, 50)], ROI, pulse_mapper=lambda point: (0, 0)
            )

    def test_piece_crossing_divider_is_rejected(self):
        config = AssemblyConfig()
        divider = ROI[1] + ROI[3] * config.split_fraction
        pieces = [
            _piece(0, 130, divider - 10),
            _piece(1, 210, 70),
            _piece(2, 300, 100),
            _piece(3, 400, 120),
        ]
        with self.assertRaisesRegex(ValueError, "上半区"):
            build_transfer_plan(pieces, ROI, pulse_mapper=lambda point: (0, 0))


if __name__ == "__main__":
    unittest.main()
