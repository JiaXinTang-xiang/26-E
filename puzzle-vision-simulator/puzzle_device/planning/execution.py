"""Validate assembly records and convert rotations into servo commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionTask:
    sequence: int
    piece_id: int
    source_x: int
    source_y: int
    target_x: int
    target_y: int
    rotation_deg: float
    servo_angle_deg: int


def build_execution_tasks(
    document: dict[str, object],
    *,
    servo_home_angle: int = 135,
    servo_direction: int = 1,
    max_x: int = 2350,
    max_y: int = 1350,
) -> list[ExecutionTask]:
    """Create safe controller tasks from an in-memory assembly plan."""
    if servo_direction not in (-1, 1):
        raise ValueError("servo_direction must be 1 or -1")
    records = document.get("pieces")
    if not isinstance(records, list) or not 1 <= len(records) <= 4:
        raise ValueError("拼接方案必须包含 1-4 块碎片")

    tasks = []
    for record in sorted(records, key=lambda item: int(item["sequence"])):
        source = record.get("source_pick_pulse")
        target = record.get("target_pick_pulse")
        if not isinstance(source, list) or len(source) != 2:
            raise ValueError(f"P{record.get('piece_id')} 缺少源抓取脉冲")
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError(f"P{record.get('piece_id')} 缺少目标放置脉冲")
        source_x, source_y = (int(round(value)) for value in source)
        target_x, target_y = (int(round(value)) for value in target)
        if not (0 <= source_x <= max_x and 0 <= target_x <= max_x):
            raise ValueError(f"P{record.get('piece_id')} 的 X 脉冲超出 0-{max_x}")
        if not (0 <= source_y <= max_y and 0 <= target_y <= max_y):
            raise ValueError(f"P{record.get('piece_id')} 的 Y 脉冲超出 0-{max_y}")

        rotation = float(record["rotation_deg"])
        servo_angle = round(servo_home_angle + servo_direction * rotation)
        if not 0 <= servo_angle <= 270:
            raise ValueError(
                f"P{record.get('piece_id')} 需要旋转 {rotation:.1f} 度，"
                f"换算舵机角度 {servo_angle} 度，超出 0-270 度"
            )
        tasks.append(ExecutionTask(
            sequence=int(record["sequence"]),
            piece_id=int(record["piece_id"]),
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y,
            rotation_deg=rotation,
            servo_angle_deg=servo_angle,
        ))
    return tasks
