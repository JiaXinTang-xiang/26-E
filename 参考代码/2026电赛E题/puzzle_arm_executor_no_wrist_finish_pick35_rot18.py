#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 SwiftPro 拼图执行节点（无第四轴版）。

订阅：
  /puzzle/piece_coordinates    std_msgs/String，识别节点输出的 JSON

发布：
  position_write_topic         swiftpro/position
  pump_topic                   swiftpro/status

服务：
  /puzzle_arm_executor/start       执行四片
  /puzzle_arm_executor/stop        停止后续动作并关闭气泵
  /puzzle_arm_executor/pump_off    立即发送关闭气泵命令
  /puzzle_arm_executor/reset       清除稳定帧

特点：
  1. 抓取 Z 默认固定为 -35 mm，放置 Z 默认固定为 -30 mm；
  2. 不使用第四轴；纸片必须事先人工摆正；
  3. 启动时取最近若干帧的中位数，避免单帧抖动；
  4. 只有调用 start 服务后才运动，不会开机自动动作；
  5. 四片完成后默认移动到 (100, -100, 35) mm。
"""

import json
import math
import statistics
import threading
import time
from collections import deque

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from swiftpro.msg import position, status


class PuzzleArmExecutorNoWrist:
    def __init__(self):
        rospy.init_node("puzzle_arm_executor", anonymous=False)

        # ROS 接口
        self.coords_topic = rospy.get_param(
            "~coords_topic", "/puzzle/piece_coordinates"
        )
        self.arm_topic = rospy.get_param("~arm_topic", "position_write_topic")
        self.pump_topic = rospy.get_param("~pump_topic", "pump_topic")

        self.arm_pub = rospy.Publisher(
            self.arm_topic, position, queue_size=1
        )
        self.pump_pub = rospy.Publisher(
            self.pump_topic, status, queue_size=1
        )
        self.coords_sub = rospy.Subscriber(
            self.coords_topic, String, self.coords_callback, queue_size=20
        )

        self.start_srv = rospy.Service(
            "~start", Trigger, self.start_callback
        )
        self.stop_srv = rospy.Service(
            "~stop", Trigger, self.stop_callback
        )
        self.pump_off_srv = rospy.Service(
            "~pump_off", Trigger, self.pump_off_callback
        )
        self.reset_srv = rospy.Service(
            "~reset", Trigger, self.reset_callback
        )

        # 已实测并按现场要求调整：抓取 Z=-35 mm，放置 Z=-30 mm
        self.pick_z = float(rospy.get_param("~pick_z", -35.0))
        self.place_z = float(rospy.get_param("~place_z", -30.0))
        self.safe_z = float(rospy.get_param("~safe_z", 80.0))
        self.home_xyz = tuple(
            float(v)
            for v in rospy.get_param("~home_xyz", [170.0, 0.0, 80.0])
        )
        if len(self.home_xyz) != 3:
            raise ValueError("~home_xyz 必须包含 x、y、z 三个数")

        # 四片全部执行完成后的最终停靠位置。
        self.finish_xyz = tuple(
            float(v)
            for v in rospy.get_param("~finish_xyz", [100.0, -100.0, 35.0])
        )
        if len(self.finish_xyz) != 3:
            raise ValueError("~finish_xyz 必须包含 x、y、z 三个数")

        # 固定等待时间。当前 SwiftPro 驱动没有可靠的到位反馈。
        self.move_wait_s = float(rospy.get_param("~move_wait_s", 1.2))
        self.pump_wait_s = float(rospy.get_param("~pump_wait_s", 1.0))
        self.release_wait_s = float(rospy.get_param("~release_wait_s", 0.8))
        self.between_piece_wait_s = float(
            rospy.get_param("~between_piece_wait_s", 0.5)
        )

        # 大块先放，降低后续吸盘碰到已放纸片的概率。
        self.piece_order = [
            int(v) for v in rospy.get_param("~piece_order", [2, 4, 3, 1])
        ]
        if sorted(self.piece_order) != [1, 2, 3, 4]:
            raise ValueError("~piece_order 必须恰好包含 1、2、3、4")

        # 无第四轴时必须预先摆正。调试纯平移时可设置 ignore_rotation=true。
        self.ignore_rotation = bool(
            rospy.get_param("~ignore_rotation", False)
        )
        self.max_rotation_deg = float(
            rospy.get_param("~max_rotation_deg", 18.0)
        )

        # 稳定性参数
        self.required_frames = int(rospy.get_param("~required_frames", 5))
        self.max_data_age_s = float(rospy.get_param("~max_data_age_s", 2.0))
        self.max_xy_jitter_mm = float(
            rospy.get_param("~max_xy_jitter_mm", 3.0)
        )
        self.frames = deque(maxlen=max(self.required_frames, 10))
        self.frames_lock = threading.Lock()

        # 工作空间保护，可按实际机械臂修改
        self.x_min = float(rospy.get_param("~x_min", 0.0))
        self.x_max = float(rospy.get_param("~x_max", 450.0))
        self.y_min = float(rospy.get_param("~y_min", -250.0))
        self.y_max = float(rospy.get_param("~y_max", 250.0))
        self.z_min = float(rospy.get_param("~z_min", -100.0))
        self.z_max = float(rospy.get_param("~z_max", 220.0))

        # 默认是真机模式，但仍需显式调用 start 服务才动作。
        self.dry_run = bool(rospy.get_param("~dry_run", False))
        self.stop_event = threading.Event()
        self.worker = None
        self.worker_lock = threading.Lock()

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            "无第四轴拼图执行节点已启动：dry_run=%s pick_z=%.1f "
            "place_z=%.1f safe_z=%.1f",
            self.dry_run,
            self.pick_z,
            self.place_z,
            self.safe_z,
        )
        rospy.logwarn(
            "无第四轴：四片必须人工预先摆正；软件急停不能替代实体急停。"
        )

    @staticmethod
    def parse_label(value):
        if isinstance(value, str) and value.upper().startswith("P"):
            value = value[1:]
        try:
            label = int(value)
        except (TypeError, ValueError):
            return None
        return label if label in (1, 2, 3, 4) else None

    @staticmethod
    def parse_xyz(value):
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            xyz = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in xyz):
            return None
        return xyz

    @staticmethod
    def wrap_angle(angle):
        return (float(angle) + 180.0) % 360.0 - 180.0

    def coords_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            if int(payload.get("count", 0)) != 4:
                return

            frame = {}
            for item in payload.get("pieces", []):
                label = self.parse_label(item.get("label"))
                pick = self.parse_xyz(item.get("pick_command_xyz"))
                place = self.parse_xyz(item.get("place_command_xyz"))
                if label is None or pick is None or place is None:
                    return

                rotation = item.get("required_rotation_deg_clockwise", 0.0)
                rotation = 0.0 if rotation is None else float(rotation)
                if not math.isfinite(rotation):
                    return

                # 只采用识别得到的 X、Y；Z 使用现场实测固定值。
                frame[label] = {
                    "pick": (pick[0], pick[1], self.pick_z),
                    "place": (place[0], place[1], self.place_z),
                    "rotation": self.wrap_angle(rotation),
                }

            if sorted(frame.keys()) != [1, 2, 3, 4]:
                return

            with self.frames_lock:
                self.frames.append((time.monotonic(), frame))
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "解析识别结果失败: %s", exc)

    def stable_plan(self):
        with self.frames_lock:
            frames = list(self.frames)[-self.required_frames:]

        if len(frames) < self.required_frames:
            return None, "稳定帧不足 {}/{}".format(
                len(frames), self.required_frames
            )
        if time.monotonic() - frames[-1][0] > self.max_data_age_s:
            return None, "识别数据已过期"

        plan = {}
        for label in (1, 2, 3, 4):
            pick_xs = [f[1][label]["pick"][0] for f in frames]
            pick_ys = [f[1][label]["pick"][1] for f in frames]
            place_xs = [f[1][label]["place"][0] for f in frames]
            place_ys = [f[1][label]["place"][1] for f in frames]
            rotations = [f[1][label]["rotation"] for f in frames]

            jitter = max(
                max(pick_xs) - min(pick_xs),
                max(pick_ys) - min(pick_ys),
                max(place_xs) - min(place_xs),
                max(place_ys) - min(place_ys),
            )
            if jitter > self.max_xy_jitter_mm:
                return None, "P{} 坐标抖动 {:.1f} mm，超过限制".format(
                    label, jitter
                )

            rotation = statistics.median(rotations)
            if (
                not self.ignore_rotation
                and abs(rotation) > self.max_rotation_deg
            ):
                return None, (
                    "P{} 仍需旋转 {:.1f} 度；无第四轴，请先人工摆正"
                ).format(label, rotation)

            plan[label] = {
                "pick": (
                    statistics.median(pick_xs),
                    statistics.median(pick_ys),
                    self.pick_z,
                ),
                "place": (
                    statistics.median(place_xs),
                    statistics.median(place_ys),
                    self.place_z,
                ),
                "rotation": rotation,
            }

        return plan, "OK"

    def in_workspace(self, xyz):
        x, y, z = xyz
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def sleep_interruptible(self, seconds):
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end:
            if self.stop_event.is_set() or rospy.is_shutdown():
                raise RuntimeError("收到停止请求")
            time.sleep(min(0.05, end - time.monotonic()))

    def move(self, xyz, description):
        if not self.in_workspace(xyz):
            raise ValueError(
                "{} 坐标超出工作空间: ({:.1f}, {:.1f}, {:.1f})".format(
                    description, *xyz
                )
            )
        if self.stop_event.is_set():
            raise RuntimeError("收到停止请求")

        rospy.loginfo(
            "MOVE %-18s -> X=%.1f Y=%.1f Z=%.1f",
            description,
            *xyz
        )
        if not self.dry_run:
            msg = position()
            msg.x, msg.y, msg.z = xyz
            self.arm_pub.publish(msg)
        self.sleep_interruptible(self.move_wait_s)

    def pump(self, enabled):
        rospy.loginfo("PUMP %s", "ON" if enabled else "OFF")
        if self.dry_run:
            return
        msg = status()
        msg.status = 1 if enabled else 0
        self.pump_pub.publish(msg)

    def execute_piece(self, label, item):
        pick_x, pick_y, _ = item["pick"]
        place_x, place_y, _ = item["place"]

        rospy.loginfo(
            "开始 P%d：pick=(%.1f, %.1f, %.1f)，place=(%.1f, %.1f, %.1f)，"
            "未执行旋转 %.1f°",
            label,
            pick_x,
            pick_y,
            self.pick_z,
            place_x,
            place_y,
            self.place_z,
            item["rotation"],
        )

        self.move((pick_x, pick_y, self.safe_z), "P{} 抓取上方".format(label))
        self.move((pick_x, pick_y, self.pick_z), "P{} 下降抓取".format(label))
        self.pump(True)
        self.sleep_interruptible(self.pump_wait_s)
        self.move((pick_x, pick_y, self.safe_z), "P{} 垂直抬起".format(label))

        self.move((place_x, place_y, self.safe_z), "P{} 放置上方".format(label))
        self.move((place_x, place_y, self.place_z), "P{} 下降放置".format(label))
        self.pump(False)
        self.sleep_interruptible(self.release_wait_s)
        self.move((place_x, place_y, self.safe_z), "P{} 放置后抬起".format(label))

    def run_plan(self, plan):
        try:
            self.move(self.home_xyz, "安全位")
            for label in self.piece_order:
                self.execute_piece(label, plan[label])
                self.sleep_interruptible(self.between_piece_wait_s)
            self.move(self.finish_xyz, "执行完成停靠位")
            rospy.loginfo(
                "四片执行完成，机械臂已移动到 X=%.1f Y=%.1f Z=%.1f",
                *self.finish_xyz
            )
        except Exception as exc:
            rospy.logerr("执行中止: %s", exc)
            self.pump(False)
        finally:
            with self.worker_lock:
                self.worker = None

    def start_callback(self, _req):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                return TriggerResponse(False, "机械臂正在执行")

        plan, reason = self.stable_plan()
        if plan is None:
            return TriggerResponse(False, reason)

        for label in (1, 2, 3, 4):
            if not self.in_workspace(plan[label]["pick"]):
                return TriggerResponse(False, "P{} 抓取坐标超限".format(label))
            if not self.in_workspace(plan[label]["place"]):
                return TriggerResponse(False, "P{} 放置坐标超限".format(label))

        self.stop_event.clear()
        with self.worker_lock:
            self.worker = threading.Thread(
                target=self.run_plan,
                args=(plan,),
                daemon=True,
            )
            self.worker.start()

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        return TriggerResponse(
            True,
            "已冻结稳定坐标并开始执行 [{}]，顺序 P{}".format(
                mode, " -> P".join(str(v) for v in self.piece_order)
            ),
        )

    def stop_callback(self, _req):
        self.stop_event.set()
        self.pump(False)
        return TriggerResponse(
            True, "已停止后续动作并关闭气泵；仍需使用实体急停保障安全"
        )

    def pump_off_callback(self, _req):
        self.pump(False)
        return TriggerResponse(True, "已发送关闭气泵命令")

    def reset_callback(self, _req):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                return TriggerResponse(False, "执行中不能清除稳定帧")
        with self.frames_lock:
            self.frames.clear()
        return TriggerResponse(True, "已清除稳定帧")

    def on_shutdown(self):
        self.stop_event.set()
        try:
            self.pump(False)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        PuzzleArmExecutorNoWrist()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
