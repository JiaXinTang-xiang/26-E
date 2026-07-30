"""Multi-frame stability checks and averaging for detected puzzle pieces."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from itertools import permutations
import math

import numpy as np

from puzzle_device.vision.piece_vision import PieceObservation


@dataclass(frozen=True)
class StabilityStatus:
    sample_count: int
    required_frames: int
    stable: bool
    reason: str


def _axis_angle_difference(first: float, second: float) -> float:
    return abs((first - second + 90.0) % 180.0 - 90.0)


def _mean_axis_angle(values: list[float]) -> float:
    doubled = np.radians(np.asarray(values, dtype=np.float64) * 2.0)
    angle = math.degrees(math.atan2(np.sin(doubled).mean(), np.cos(doubled).mean())) / 2.0
    return (angle + 90.0) % 180.0 - 90.0


class PieceStabilityTracker:
    """Track up to four stationary pieces across consecutive detection frames."""

    def __init__(
        self,
        required_frames: int = 8,
        center_tolerance_px: float = 4.0,
        angle_tolerance_deg: float = 4.0,
        area_tolerance_ratio: float = 0.08,
    ) -> None:
        if required_frames < 2:
            raise ValueError("required_frames must be at least two")
        self.required_frames = required_frames
        self.center_tolerance_px = center_tolerance_px
        self.angle_tolerance_deg = angle_tolerance_deg
        self.area_tolerance_ratio = area_tolerance_ratio
        self._history: deque[list[PieceObservation]] = deque(maxlen=required_frames)
        self.status = StabilityStatus(0, required_frames, False, "等待识别")

    def reset(self, reason: str = "等待稳定采样") -> StabilityStatus:
        self._history.clear()
        self.status = StabilityStatus(0, self.required_frames, False, reason)
        return self.status

    def update(self, pieces: list[PieceObservation]) -> StabilityStatus:
        if not pieces:
            return self.reset("没有识别到碎片")

        ordered = list(pieces)
        if self._history:
            if len(ordered) != len(self._history[-1]):
                self._history.clear()
            else:
                ordered = self._match_to_previous(self._history[-1], ordered)

        candidate_history = [*self._history, ordered]
        reason = self._instability_reason(candidate_history)
        if reason is not None:
            self._history.clear()
            self._history.append(ordered)
            self.status = StabilityStatus(1, self.required_frames, False, reason)
            return self.status

        self._history.append(ordered)
        count = len(self._history)
        stable = count >= self.required_frames
        reason = "稳定，可确认锁定" if stable else f"稳定采样 {count}/{self.required_frames}"
        self.status = StabilityStatus(count, self.required_frames, stable, reason)
        return self.status

    @staticmethod
    def _match_to_previous(
        previous: list[PieceObservation], current: list[PieceObservation]
    ) -> list[PieceObservation]:
        best_order = current
        best_cost = math.inf
        for order in permutations(current):
            cost = sum(
                math.dist(old.center, new.center) for old, new in zip(previous, order)
            )
            if cost < best_cost:
                best_cost = cost
                best_order = list(order)
        return list(best_order)

    def _instability_reason(
        self, history: list[list[PieceObservation]]
    ) -> str | None:
        if len(history) < 2:
            return None
        piece_count = len(history[0])
        if any(len(frame) != piece_count for frame in history):
            return "碎片数量变化，重新计数"
        for index in range(piece_count):
            samples = [frame[index] for frame in history]
            if len({len(piece.polygon) for piece in samples}) != 1:
                return f"P{index} 顶点数变化"
            centers = np.asarray([piece.center for piece in samples], dtype=np.float64)
            center_deviation = np.linalg.norm(centers - centers.mean(axis=0), axis=1).max()
            if center_deviation > self.center_tolerance_px:
                return f"P{index} 中心波动 {center_deviation:.1f}px"
            areas = np.asarray([piece.area_px for piece in samples], dtype=np.float64)
            relative_area_range = float(np.ptp(areas) / max(areas.mean(), 1.0))
            if relative_area_range > self.area_tolerance_ratio:
                return f"P{index} 面积波动过大"
            mean_angle = _mean_axis_angle([piece.pca_angle_deg for piece in samples])
            if max(
                _axis_angle_difference(piece.pca_angle_deg, mean_angle) for piece in samples
            ) > self.angle_tolerance_deg:
                return f"P{index} 方向波动过大"
        return None

    def averaged_observations(self) -> list[PieceObservation]:
        if not self.status.stable or not self._history:
            raise RuntimeError("stable observations are not available")
        averaged: list[PieceObservation] = []
        for index in range(len(self._history[-1])):
            samples = [frame[index] for frame in self._history]
            latest = samples[-1]
            centers = np.asarray([piece.center for piece in samples], dtype=np.float64)
            picks = np.asarray([piece.pick_point for piece in samples], dtype=np.float64)
            averaged.append(replace(
                latest,
                piece_id=index,
                center=tuple(centers.mean(axis=0)),
                pick_point=tuple(picks.mean(axis=0)),
                pick_clearance_px=float(np.mean([piece.pick_clearance_px for piece in samples])),
                area_px=float(np.mean([piece.area_px for piece in samples])),
                pca_angle_deg=_mean_axis_angle([piece.pca_angle_deg for piece in samples]),
                longest_edge_angle_deg=_mean_axis_angle(
                    [piece.longest_edge_angle_deg for piece in samples]
                ),
                confidence=float(np.mean([piece.confidence for piece in samples])),
            ))
        return averaged
