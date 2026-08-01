"""Tests for competition modes and the 120-second display helpers."""

import unittest

from puzzle_device.competition import (
    COMPETITION_LIMIT_SECONDS,
    FIELD_WHITE_MODE,
    PLAYING_CARD_MODE,
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

    def test_time_format_and_limit(self):
        self.assertEqual(COMPETITION_LIMIT_SECONDS, 120.0)
        self.assertEqual(format_competition_time(0), "00:00.0")
        self.assertEqual(format_competition_time(61.27), "01:01.2")
        self.assertEqual(format_competition_time(120), "02:00.0")


if __name__ == "__main__":
    unittest.main()
