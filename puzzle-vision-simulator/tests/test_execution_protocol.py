"""Tests for rotation-aware gantry execution and status parsing."""

import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from puzzle_device.calibration.gantry_protocol import (
    CMD_PICK_AND_PLACE_DUAL_ANGLE,
    GantryStatusParser,
    SERVO_COMMAND_MARKER,
    STATUS_ACTION_FAILED,
    STATUS_ACTION_COMPLETE,
    STATUS_COMMAND_ACCEPTED,
    build_dual_angle_pick_and_place_frame,
    build_pick_and_place_frame,
    build_serial_health_check_frame,
    OptionalSerialPort,
    select_ch340_port,
)
from puzzle_device.planning import build_execution_tasks
from apps.puzzle_control_gui import PuzzleControlApp


class ExecutionProtocolTest(unittest.TestCase):
    def test_transient_read_error_retries_without_closing_connection(self):
        app = PuzzleControlApp.__new__(PuzzleControlApp)
        app.serial_read_error_count = 0
        app.serial_read_retry_after = 0.0
        app.controller_state = Mock()
        app.status = Mock()
        app._append_log = Mock()
        app._handle_serial_fault = Mock()
        PuzzleControlApp._handle_serial_read_error(app, OSError("temporary"))
        self.assertEqual(app.serial_read_error_count, 1)
        app._handle_serial_fault.assert_not_called()
        app.controller_state.set.assert_called_once()

    def test_repeated_read_errors_escalate_to_safe_disconnect(self):
        app = PuzzleControlApp.__new__(PuzzleControlApp)
        app.serial_read_error_count = 0
        app.serial_read_retry_after = 0.0
        app.controller_state = Mock()
        app.status = Mock()
        app._append_log = Mock()
        app._handle_serial_fault = Mock()
        for _ in range(3):
            PuzzleControlApp._handle_serial_read_error(app, OSError("temporary"))
        self.assertEqual(app.serial_read_error_count, 3)
        app._handle_serial_fault.assert_called_once()
        self.assertEqual(app._handle_serial_fault.call_args.args[1], "读取")

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

    def test_serial_health_check_frame_is_invalid_without_changing_coordinates(self):
        valid = build_pick_and_place_frame(0, 0, 0, 0)
        health = build_serial_health_check_frame()
        self.assertEqual(len(health), 17)
        self.assertEqual(health[:-2], valid[:-2])
        self.assertNotEqual(health[-2], valid[-2])
        self.assertEqual(health[-1], valid[-1])

    def test_optional_serial_rejects_send_when_disconnected(self):
        port = OptionalSerialPort("COM_TEST")
        with self.assertRaisesRegex(RuntimeError, "not connected"):
            port.send(b"test")

    def test_optional_serial_close_clears_handle_even_if_close_fails(self):
        port = OptionalSerialPort("COM_TEST")
        handle = Mock()
        handle.close.side_effect = OSError("device removed")
        port._serial = handle
        with self.assertRaises(OSError):
            port.close()
        self.assertIsNone(port._serial)

    def test_optional_serial_connect_closes_stale_handle_first(self):
        port = OptionalSerialPort("COM_TEST")
        stale = Mock()
        port._serial = stale
        opened = Mock(is_open=True)
        serial_module = Mock()
        serial_module.Serial.return_value = opened
        with patch.dict("sys.modules", {"serial": serial_module}):
            port.connect()
        stale.close.assert_called_once()
        opened.reset_input_buffer.assert_called_once()
        opened.reset_output_buffer.assert_called_once()

    def test_ch340_selection_prefers_existing_configured_port(self):
        ports = [
            SimpleNamespace(device="COM8", description="USB-SERIAL CH340", hwid=""),
            SimpleNamespace(device="COM30", description="CH340", hwid="USB VID:PID=1A86"),
        ]
        self.assertEqual(select_ch340_port(ports, "COM8"), "COM8")

    def test_ch340_selection_prefers_linux_port(self):
        ports = [
            SimpleNamespace(device="/dev/ttyUSB0", description="USB-SERIAL CH340", hwid=""),
            SimpleNamespace(device="/dev/ttyUSB1", description="CH341", hwid=""),
        ]
        self.assertEqual(select_ch340_port(ports, "/dev/ttyUSB0"), "/dev/ttyUSB0")

    def test_ch340_selection_follows_single_ch340_after_com_change(self):
        ports = [
            SimpleNamespace(device="COM31", description="USB-SERIAL CH340", hwid=""),
            SimpleNamespace(device="COM5", description="Bluetooth Link", hwid=""),
        ]
        self.assertEqual(select_ch340_port(ports, "COM30"), "COM31")

    def test_ch340_selection_follows_linux_port_change(self):
        ports = [
            SimpleNamespace(device="/dev/ttyUSB1", description="QINHENG CH340", hwid=""),
            SimpleNamespace(device="/dev/ttyACM0", description="Modem", hwid=""),
        ]
        self.assertEqual(select_ch340_port(ports, "/dev/ttyUSB0"), "/dev/ttyUSB1")

    def test_ch340_selection_does_not_guess_between_multiple_devices(self):
        ports = [
            SimpleNamespace(device="COM30", description="CH340", hwid=""),
            SimpleNamespace(device="COM31", description="CH341", hwid=""),
        ]
        self.assertIsNone(select_ch340_port(ports, "COM8"))

    def test_ch340_selection_does_not_guess_between_multiple_linux_devices(self):
        ports = [
            SimpleNamespace(device="/dev/ttyUSB0", description="CH340", hwid=""),
            SimpleNamespace(device="/dev/ttyUSB1", description="CH341", hwid=""),
        ]
        self.assertIsNone(select_ch340_port(ports, "/dev/ttyUSB2"))

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
