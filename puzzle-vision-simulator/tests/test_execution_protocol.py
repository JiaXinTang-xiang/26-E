"""Tests for rotation-aware gantry execution and status parsing."""

import struct
import unittest

from puzzle_device.calibration.gantry_protocol import (
    CMD_PICK_AND_PLACE_DUAL_ANGLE,
    GantryStatusParser,
    SERVO_COMMAND_MARKER,
    STATUS_ACTION_FAILED,
    STATUS_ACTION_COMPLETE,
    STATUS_COMMAND_ACCEPTED,
    build_dual_angle_pick_and_place_frame,
    build_pick_and_place_frame,
)
from puzzle_device.planning import build_execution_tasks


class ExecutionProtocolTest(unittest.TestCase):
    def test_rotation_angle_uses_legacy_z_fields(self):
        frame = build_pick_and_place_frame(
            100, 200, 300, 400, rotation_angle_deg=123
        )
        self.assertEqual(len(frame), 17)
        command, source_x, source_y, source_z, target_x, target_y, target_z = struct.unpack(
            ">BHHHHHH", frame[2:15]
        )
        self.assertEqual(command, 0xA1)
        self.assertEqual((source_x, source_y), (100, 200))
        self.assertEqual(source_z, SERVO_COMMAND_MARKER)
        self.assertEqual((target_x, target_y, target_z), (300, 400, 123))

    def test_dual_angle_command_keeps_17_byte_frame(self):
        frame = build_dual_angle_pick_and_place_frame(
            100, 200, 300, 400, pick_angle_deg=93, place_angle_deg=178
        )
        self.assertEqual(len(frame), 17)
        values = struct.unpack(">BHHHHHH", frame[2:15])
        self.assertEqual(values, (
            CMD_PICK_AND_PLACE_DUAL_ANGLE, 100, 200, 93, 300, 400, 178
        ))

    def test_status_parser_ignores_echo_and_handles_fragmentation(self):
        accepted = bytes([0xAA, 0x03, STATUS_COMMAND_ACCEPTED,
                          0x03 ^ STATUS_COMMAND_ACCEPTED, 0x55])
        complete = bytes([0xAA, 0x03, STATUS_ACTION_COMPLETE,
                          0x03 ^ STATUS_ACTION_COMPLETE, 0x55])
        parser = GantryStatusParser()
        self.assertEqual(parser.feed(b"\xAA\x0F\xA1" + accepted[:2]), [])
        self.assertEqual(parser.feed(accepted[2:] + b"\x00\x01" + complete), [
            STATUS_COMMAND_ACCEPTED, STATUS_ACTION_COMPLETE,
        ])

    def test_status_parser_accepts_action_failure(self):
        failed = bytes([0xAA, 0x03, STATUS_ACTION_FAILED,
                        0x03 ^ STATUS_ACTION_FAILED, 0x55])
        self.assertEqual(GantryStatusParser().feed(failed), [STATUS_ACTION_FAILED])

    def test_status_parser_reset_discards_partial_frame(self):
        parser = GantryStatusParser()
        parser.feed(bytes([0xAA, 0x03, STATUS_COMMAND_ACCEPTED]))
        parser.reset()
        tail = bytes([0x03 ^ STATUS_COMMAND_ACCEPTED, 0x55])
        self.assertEqual(parser.feed(tail), [])

    def test_plan_rotation_maps_around_135_degree_home(self):
        document = {
            "pieces": [{
                "sequence": 1,
                "piece_id": 2,
                "source_pick_pulse": [500, 600],
                "target_pick_pulse": [700, 800],
                "rotation_deg": -45.0,
            }]
        }
        forward = build_execution_tasks(document, servo_direction=1)[0]
        reverse = build_execution_tasks(document, servo_direction=-1)[0]
        self.assertEqual((forward.pick_angle_deg, forward.place_angle_deg), (158, 112))
        self.assertEqual((reverse.pick_angle_deg, reverse.place_angle_deg), (112, 158))
        self.assertAlmostEqual(
            forward.place_angle_deg - forward.pick_angle_deg, -45, delta=1
        )

    def test_execution_tasks_accept_every_competition_piece_count(self):
        for piece_count in range(1, 5):
            with self.subTest(piece_count=piece_count):
                document = {
                    "pieces": [{
                        "sequence": index + 1,
                        "piece_id": index,
                        "source_pick_pulse": [100 + index, 200 + index],
                        "target_pick_pulse": [300 + index, 400 + index],
                        "rotation_deg": 0.0,
                    } for index in range(piece_count)]
                }
                tasks = build_execution_tasks(document)
                self.assertEqual(len(tasks), piece_count)
                self.assertEqual([task.sequence for task in tasks], list(range(1, piece_count + 1)))

    def test_out_of_range_error_shows_endpoint_value_and_pixel(self):
        record = {
            "sequence": 1,
            "piece_id": 0,
            "source_pick_px": [641.2, 142.5],
            "source_pick_pulse": [-34, 830],
            "target_pick_px": [683.2, 472.8],
            "target_pick_pulse": [1462, 1017],
            "rotation_deg": 0.0,
        }
        with self.assertRaisesRegex(
            ValueError, r"P0 取料 X 脉冲=-34.*取料像素=\[641\.2, 142\.5\]"
        ):
            build_execution_tasks({"pieces": [record]})


if __name__ == "__main__":
    unittest.main()
