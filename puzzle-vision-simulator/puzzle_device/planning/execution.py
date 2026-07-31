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
    pick_angle_deg: int
    place_angle_deg: int

    @property
    def servo_angle_deg(self) -> int:
        """Retain the old single-angle API for compatibility."""
        return self.place_angle_deg


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
        piece_id = record.get("piece_id")
        source = record.get("source_pick_pulse")
        target = record.get("target_pick_pulse")
        if not isinstance(source, list) or len(source) != 2:
            raise ValueError(f"P{piece_id} 缺少源抓取脉冲")
        if not isinstance(target, list) or len(target) != 2:
            raise ValueError(f"P{piece_id} 缺少目标放置脉冲")
        source_x, source_y = (int(round(value)) for value in source)
        target_x, target_y = (int(round(value)) for value in target)
        source_px = record.get("source_pick_px")
        target_px = record.get("target_pick_px")
        if not 0 <= source_x <= max_x:
            raise ValueError(
                f"P{piece_id} 取料 X 脉冲={source_x}，超出 0-{max_x}；"
                f"取料像素={source_px}"
            )
        if not 0 <= target_x <= max_x:
            raise ValueError(
                f"P{piece_id} 放料 X 脉冲={target_x}，超出 0-{max_x}；"
                f"放料像素={target_px}"
            )
        if not 0 <= source_y <= max_y:
            raise ValueError(
                f"P{piece_id} 取料 Y 脉冲={source_y}，超出 0-{max_y}；"
                f"取料像素={source_px}"
            )
        if not 0 <= target_y <= max_y:
            raise ValueError(
                f"P{piece_id} 放料 Y 脉冲={target_y}，超出 0-{max_y}；"
                f"放料像素={target_px}"
            )

        rotation = float(record["rotation_deg"])
        directed_rotation = servo_direction * rotation
        pick_angle = round(servo_home_angle - directed_rotation / 2.0)
        place_angle = round(servo_home_angle + directed_rotation / 2.0)
        if not (0 <= pick_angle <= 270 and 0 <= place_angle <= 270):
            raise ValueError(
                f"P{piece_id} 需要旋转 {rotation:.1f} 度，"
                f"换算舵机角度 {pick_angle}/{place_angle} 度，超出 0-270 度"
            )
        tasks.append(ExecutionTask(
            sequence=int(record["sequence"]),
            piece_id=int(record["piece_id"]),
            source_x=source_x,
            source_y=source_y,
            target_x=target_x,
            target_y=target_y,
            rotation_deg=rotation,
            pick_angle_deg=pick_angle,
            place_angle_deg=place_angle,
        ))
    return tasks
