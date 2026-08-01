#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛版现场任意 1~4 片拼图执行节点（SwiftPro，无第四轴）。

订阅 /puzzle/piece_coordinates，支持 onsite_puzzle_solver_final.py 输出的动态片数和求解状态。
只有调用 /onsite_puzzle_executor/start 后才动作。

限制：没有第四轴时，每片 required_rotation_deg_clockwise 必须在允许范围内；
否则服务拒绝执行，需要人工预先摆正。
"""

import json
import math
import statistics
import threading
import time
from collections import deque
from typing import Dict, Optional, Tuple

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from swiftpro.msg import position, status


class CompetitionPuzzleExecutorNoWrist:
    def __init__(self) -> None:
        rospy.init_node("onsite_puzzle_executor", anonymous=False)

        self.coords_topic = rospy.get_param("~coords_topic", "/puzzle/piece_coordinates")
        self.arm_topic = rospy.get_param("~arm_topic", "position_write_topic")
        self.pump_topic = rospy.get_param("~pump_topic", "pump_topic")

        self.arm_pub = rospy.Publisher(self.arm_topic, position, queue_size=1)
        self.pump_pub = rospy.Publisher(self.pump_topic, status, queue_size=1)
        self.coords_sub = rospy.Subscriber(
            self.coords_topic, String, self.coords_callback, queue_size=20
        )
        self.solver_status_topic = rospy.get_param(
            "~solver_status_topic", "/puzzle/solver_status"
        )
        self.latest_solver_status = "UNKNOWN"
        self.latest_solver_status_time = 0.0
        self.solver_status_sub = rospy.Subscriber(
            self.solver_status_topic,
            String,
            self.solver_status_callback,
            queue_size=5,
        )

        self.start_srv = rospy.Service("~start", Trigger, self.start_callback)
        self.stop_srv = rospy.Service("~stop", Trigger, self.stop_callback)
        self.pump_off_srv = rospy.Service("~pump_off", Trigger, self.pump_off_callback)
        self.reset_srv = rospy.Service("~reset", Trigger, self.reset_callback)

        self.pick_z = float(rospy.get_param("~pick_z", -35.0))
        self.place_z = float(rospy.get_param("~place_z", -30.0))
        self.safe_z = float(rospy.get_param("~safe_z", 80.0))
        self.finish_xyz = tuple(
            float(v) for v in rospy.get_param("~finish_xyz", [100.0, -100.0, 35.0])
        )
        if len(self.finish_xyz) != 3:
            raise ValueError("~finish_xyz 必须包含三个数")

        self.max_rotation_deg = float(rospy.get_param("~max_rotation_deg", 18.0))
        self.ignore_rotation = bool(rospy.get_param("~ignore_rotation", False))
        self.required_frames = int(rospy.get_param("~required_frames", 5))
        self.max_data_age_s = float(rospy.get_param("~max_data_age_s", 2.0))
        self.max_solver_status_age_s = float(
            rospy.get_param("~max_solver_status_age_s", 3.0)
        )
        self.require_solver_found = bool(
            rospy.get_param("~require_solver_found", True)
        )
        self.max_xy_jitter_mm = float(rospy.get_param("~max_xy_jitter_mm", 3.0))

        self.move_wait_s = float(rospy.get_param("~move_wait_s", 1.2))
        self.pump_wait_s = float(rospy.get_param("~pump_wait_s", 1.0))
        self.release_wait_s = float(rospy.get_param("~release_wait_s", 0.8))
        self.between_piece_wait_s = float(rospy.get_param("~between_piece_wait_s", 0.4))
        self.dry_run = bool(rospy.get_param("~dry_run", False))

        self.x_min = float(rospy.get_param("~x_min", 0.0))
        self.x_max = float(rospy.get_param("~x_max", 450.0))
        self.y_min = float(rospy.get_param("~y_min", -250.0))
        self.y_max = float(rospy.get_param("~y_max", 250.0))
        self.z_min = float(rospy.get_param("~z_min", -100.0))
        self.z_max = float(rospy.get_param("~z_max", 220.0))

        self.frames = deque(maxlen=max(12, self.required_frames * 2))
        self.frames_lock = threading.Lock()
        self.worker_lock = threading.Lock()
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo(
            "比赛执行节点启动：pick_z=%.1f place_z=%.1f max_rot=%.1f dry_run=%s",
            self.pick_z,
            self.place_z,
            self.max_rotation_deg,
            self.dry_run,
        )

    @staticmethod
    def parse_label(value) -> Optional[int]:
        if isinstance(value, str) and value.upper().startswith("P"):
            value = value[1:]
        try:
            label = int(value)
        except (TypeError, ValueError):
            return None
        return label if 1 <= label <= 4 else None

    @staticmethod
    def parse_xyz(value) -> Optional[Tuple[float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            xyz = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return None
        return xyz if all(math.isfinite(v) for v in xyz) else None

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    def solver_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.latest_solver_status = str(payload.get("status", "UNKNOWN"))
            self.latest_solver_status_time = time.monotonic()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "解析求解状态失败: %s", exc)

    def coords_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not bool(payload.get("solution_found", True)):
                return
            count = int(payload.get("count", 0))
            if not 1 <= count <= 4:
                return

            frame: Dict[int, Dict[str, object]] = {}
            for item in payload.get("pieces", []):
                label = self.parse_label(item.get("label"))
                pick = self.parse_xyz(item.get("pick_command_xyz"))
                place = self.parse_xyz(item.get("place_command_xyz"))
                if label is None or pick is None or place is None:
                    return
                rotation = float(item.get("required_rotation_deg_clockwise", 0.0))
                area = float(item.get("area_mm2", 0.0))
                if not math.isfinite(rotation) or not math.isfinite(area):
                    return
                frame[label] = {
                    "pick": (pick[0], pick[1], self.pick_z),
                    "place": (place[0], place[1], self.place_z),
                    "rotation": self.wrap_angle(rotation),
                    "area": area,
                }
            if len(frame) != count:
                return
            with self.frames_lock:
                self.frames.append((time.monotonic(), frame))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "解析现场拼图结果失败: %s", exc)

    def stable_plan(self):
        if self.require_solver_found:
            age = time.monotonic() - self.latest_solver_status_time
            if self.latest_solver_status_time <= 0.0 or age > self.max_solver_status_age_s:
                return None, "求解器状态缺失或已过期"
            if self.latest_solver_status != "FOUND":
                return None, "求解器状态不是 FOUND，而是 {}".format(
                    self.latest_solver_status
                )

        with self.frames_lock:
            frames = list(self.frames)[-self.required_frames:]
        if len(frames) < self.required_frames:
            return None, "稳定帧不足 {}/{}".format(len(frames), self.required_frames)
        if time.monotonic() - frames[-1][0] > self.max_data_age_s:
            return None, "识别数据已过期"

        labels = sorted(frames[-1][1].keys())
        if not labels:
            return None, "没有拼图片"
        for _, frame in frames:
            if sorted(frame.keys()) != labels:
                return None, "连续帧片数或标签发生变化"

        plan = {}
        for label in labels:
            pick_x = [frame[label]["pick"][0] for _, frame in frames]
            pick_y = [frame[label]["pick"][1] for _, frame in frames]
            place_x = [frame[label]["place"][0] for _, frame in frames]
            place_y = [frame[label]["place"][1] for _, frame in frames]
            rotations = [frame[label]["rotation"] for _, frame in frames]
            areas = [frame[label]["area"] for _, frame in frames]
            jitter = max(
                max(pick_x) - min(pick_x),
                max(pick_y) - min(pick_y),
                max(place_x) - min(place_x),
                max(place_y) - min(place_y),
            )
            if jitter > self.max_xy_jitter_mm:
                return None, "P{} 坐标抖动 {:.1f}mm".format(label, jitter)
            rotation = statistics.median(rotations)
            if not self.ignore_rotation and abs(rotation) > self.max_rotation_deg:
                return None, (
                    "P{} 仍需旋转 {:.1f} 度；无第四轴，请人工摆正"
                ).format(label, rotation)
            plan[label] = {
                "pick": (
                    statistics.median(pick_x),
                    statistics.median(pick_y),
                    self.pick_z,
                ),
                "place": (
                    statistics.median(place_x),
                    statistics.median(place_y),
                    self.place_z,
                ),
                "rotation": rotation,
                "area": statistics.median(areas),
            }

        # 大片先放，减少后续碰撞。
        order = sorted(labels, key=lambda label: plan[label]["area"], reverse=True)
        return (plan, order), "OK"

    def xyz_in_workspace(self, xyz) -> bool:
        return (
            self.x_min <= xyz[0] <= self.x_max
            and self.y_min <= xyz[1] <= self.y_max
            and self.z_min <= xyz[2] <= self.z_max
        )

    def move(self, xyz, description: str) -> bool:
        if not self.xyz_in_workspace(xyz):
            rospy.logerr("坐标超工作空间 %s: %s", description, xyz)
            return False
        if self.stop_event.is_set():
            return False
        rospy.loginfo("MOVE %-18s X=%.1f Y=%.1f Z=%.1f", description, *xyz)
        if not self.dry_run:
            msg = position()
            msg.x, msg.y, msg.z = xyz
            self.arm_pub.publish(msg)
        time.sleep(self.move_wait_s)
        return not self.stop_event.is_set()

    def pump(self, enabled: bool) -> None:
        rospy.loginfo("PUMP %s", "ON" if enabled else "OFF")
        if not self.dry_run:
            msg = status()
            msg.status = 1 if enabled else 0
            self.pump_pub.publish(msg)

    def execute_piece(self, label: int, item: Dict[str, object]) -> bool:
        pick = item["pick"]
        place = item["place"]
        sequence = [
            ((pick[0], pick[1], self.safe_z), "P{}抓取上方".format(label)),
            (pick, "P{}抓取".format(label)),
        ]
        for xyz, text in sequence:
            if not self.move(xyz, text):
                return False
        self.pump(True)
        time.sleep(self.pump_wait_s)
        if not self.move((pick[0], pick[1], self.safe_z), "P{}抬升".format(label)):
            return False
        if not self.move((place[0], place[1], self.safe_z), "P{}放置上方".format(label)):
            return False
        if not self.move(place, "P{}放置".format(label)):
            return False
        self.pump(False)
        time.sleep(self.release_wait_s)
        if not self.move((place[0], place[1], self.safe_z), "P{}离开".format(label)):
            return False
        time.sleep(self.between_piece_wait_s)
        return True

    def worker_main(self, plan, order) -> None:
        try:
            for label in order:
                if self.stop_event.is_set() or not self.execute_piece(label, plan[label]):
                    break
            if not self.stop_event.is_set():
                self.move(self.finish_xyz, "完成停靠位")
        except Exception as exc:
            rospy.logerr("执行异常: %s", exc)
        finally:
            self.pump(False)
            with self.worker_lock:
                self.worker = None

    def start_callback(self, _request):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                return TriggerResponse(False, "正在执行")
            stable, reason = self.stable_plan()
            if stable is None:
                return TriggerResponse(False, reason)
            plan, order = stable
            self.stop_event.clear()
            self.worker = threading.Thread(
                target=self.worker_main, args=(plan, order), daemon=True
            )
            self.worker.start()
        return TriggerResponse(
            True,
            "开始执行 {}，顺序 {}".format(
                "DRY-RUN" if self.dry_run else "LIVE",
                " -> ".join("P{}".format(v) for v in order),
            ),
        )

    def stop_callback(self, _request):
        self.stop_event.set()
        self.pump(False)
        return TriggerResponse(True, "已停止后续动作并关闭气泵")

    def pump_off_callback(self, _request):
        self.pump(False)
        return TriggerResponse(True, "气泵关闭命令已发送")

    def reset_callback(self, _request):
        with self.frames_lock:
            self.frames.clear()
        return TriggerResponse(True, "稳定帧已清除")

    def on_shutdown(self) -> None:
        self.stop_event.set()
        try:
            self.pump(False)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        CompetitionPuzzleExecutorNoWrist()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
