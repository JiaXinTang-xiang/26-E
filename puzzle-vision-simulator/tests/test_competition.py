"""Tests for competition modes and the 120-second display helpers."""

import unittest
from unittest.mock import Mock

from apps.competition_gui import CompetitionApp
from puzzle_device.competition import (
    COMPETITION_LIMIT_SECONDS,
    FIELD_WHITE_MODE,
    PLAYING_CARD_MODE,
    PLAYING_CARD_V2_MODE,
    SELF_ASSEMBLY_MODE,
    SELF_TRANSFER_MODE,
    format_competition_time,
)


class CompetitionTest(unittest.TestCase):
    def test_modes_use_fixed_four_or_auto_count(self):
        self.assertEqual(SELF_TRANSFER_MODE.expected_piece_count, 4)
        self.assertEqual(SELF_TRANSFER_MODE.planning_method, "transfer")
        self.assertEqual(SELF_ASSEMBLY_MODE.expected_piece_count, 4)
        self.assertEqual(SELF_ASSEMBLY_MODE.planning_method, "self_assembly")
        self.assertIsNone(FIELD_WHITE_MODE.expected_piece_count)
        self.assertTrue(SELF_TRANSFER_MODE.implemented)
        self.assertTrue(SELF_ASSEMBLY_MODE.implemented)
        self.assertTrue(FIELD_WHITE_MODE.implemented)
        self.assertTrue(PLAYING_CARD_MODE.implemented)
        self.assertEqual(PLAYING_CARD_MODE.planning_method, "texture")
        self.assertTrue(PLAYING_CARD_V2_MODE.implemented)
        self.assertEqual(PLAYING_CARD_V2_MODE.planning_method, "texture_v2")

    def test_time_format_and_limit(self):
        self.assertEqual(COMPETITION_LIMIT_SECONDS, 120.0)
        self.assertEqual(format_competition_time(0), "00:00.0")
        self.assertEqual(format_competition_time(61.27), "01:01.2")
        self.assertEqual(format_competition_time(120), "02:00.0")

    def test_full_stop_clears_pc_state_without_sending_controller_frame(self):
        app = CompetitionApp.__new__(CompetitionApp)
        app.competition_active = True
        app.planning_active = True
        app.waiting_for_completion = True
        app.auto_run_enabled = True
        app.tasks = [object(), object()]
        app.current_task_index = 1
        app.competition_waiting_for_serial = True
        app.competition_finished_elapsed = None
        app.competition_result = "running"
        app.planning_expected_piece_count = 4
        app.planning_generation = 7
        app.planning_future = Mock()
        app.planning_future_generation = 7
        app.planning_tracker = Mock()
        app.waiting_for_accept = True
        app.ignore_controller_status_until_next_run = False
        app.serial = Mock(connected=True)
        app.status_parser = Mock()
        app.run_log_path = object()
        app.plan_state = Mock()
        app.task_state = Mock()
        app.controller_state = Mock()
        app.status = Mock()
        app._elapsed_seconds = Mock(return_value=12.3)
        app._cancel_accept_timeout = Mock()
        app._cancel_auto_continue = Mock()
        app._cancel_serial_health_check = Mock()
        app._refresh_task_list = Mock()
        app._set_piece_count_controls_enabled = Mock()
        app._set_mode_buttons_enabled = Mock()
        app._set_competition_state = Mock()
        app._append_log = Mock()

        app._stop_competition()

        self.assertFalse(app.competition_active)
        self.assertFalse(app.planning_active)
        self.assertFalse(app.waiting_for_completion)
        self.assertFalse(app.waiting_for_accept)
        self.assertFalse(app.auto_run_enabled)
        self.assertEqual(app.tasks, [])
        self.assertTrue(app.ignore_controller_status_until_next_run)
        app.serial.send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
