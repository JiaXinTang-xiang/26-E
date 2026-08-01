"""Small state helpers shared by the competition GUI and its tests."""

from __future__ import annotations

from dataclasses import dataclass


COMPETITION_LIMIT_SECONDS = 120.0


@dataclass(frozen=True)
class CompetitionMode:
    key: str
    title: str
    expected_piece_count: int | None
    planning_method: str = "assembly"
    implemented: bool = True


SELF_TRANSFER_MODE = CompetitionMode(
    "requirement_1_1", "1（1）自备4块只搬运", 4, planning_method="transfer"
)
SELF_ASSEMBLY_MODE = CompetitionMode(
    "requirement_1_2", "1（2）自备4块拼接", 4, planning_method="self_assembly"
)
FIELD_WHITE_MODE = CompetitionMode("requirement_2_1", "2（1）现场白色碎片", None)
PLAYING_CARD_MODE = CompetitionMode(
    "requirement_2_2", "2（2）扑克牌碎片", None, planning_method="texture"
)
COMPETITION_MODES = (
    SELF_TRANSFER_MODE,
    SELF_ASSEMBLY_MODE,
    FIELD_WHITE_MODE,
    PLAYING_CARD_MODE,
)


def format_competition_time(elapsed_seconds: float) -> str:
    elapsed = max(0.0, float(elapsed_seconds))
    minutes, seconds = divmod(int(elapsed), 60)
    tenths = int((elapsed - int(elapsed)) * 10)
    return f"{minutes:02d}:{seconds:02d}.{tenths}"
