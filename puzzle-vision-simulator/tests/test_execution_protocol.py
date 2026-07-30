"""Tests for rotation-aware gantry execution and status parsing."""

import struct
import unittest

from puzzle_device.calibration.gantry_protocol import (
    GantryStatusParser,
    SERVO_COMMAND_MARKER,
    STATUS_ACTION_FAILED,
    STATUS_ACTION_COMPLETE,
    STATUS_COMMAND_ACCEPTED,
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
        self.assertEqual(forward.servo_angle_deg, 90)
        self.assertEqual(reverse.servo_angle_deg, 180)


if __name__ == "__main__":
    unittest.main()
