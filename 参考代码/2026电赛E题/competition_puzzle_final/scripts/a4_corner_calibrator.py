#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS1 黑色 A4 纸四角标定工具。

操作：
1. 运行后显示彩色相机画面；
2. 鼠标依次点击：左上、右上、右下、左下；
3. 检查右侧透视预览；
4. 按 S 或 Enter 保存；
5. 按 U 撤销，R 重置，Q/Esc 退出。

默认读取：/camera/color/image_raw
默认保存：~/.ros/puzzle_a4_corners.yaml
"""

import json
import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


PAPER_W_MM = 210.0
PAPER_H_MM = 297.0
WARP_W = 840
WARP_H = 1188
POINT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
POINT_NAMES_ZH = ("左上", "右上", "右下", "左下")


class A4CornerCalibrator:
    def __init__(self) -> None:
        rospy.init_node("a4_corner_calibrator", anonymous=False)
        self.bridge = CvBridge()
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.output_file = os.path.expanduser(
            rospy.get_param("~output_file", "~/.ros/puzzle_a4_corners.yaml")
        )
        self.window_name = "A4 corner calibration - click TL,TR,BR,BL"
        self.preview_name = "A4 warped preview"
        self.display_max_width = int(rospy.get_param("~display_max_width", 1200))
        self.display_max_height = int(rospy.get_param("~display_max_height", 800))
        self.display_scale = 1.0

        self.lock = threading.Lock()
        self.latest_bgr: Optional[np.ndarray] = None
        self.points: List[Tuple[float, float]] = []
        self.last_message = "等待图像..."

        self.sub = rospy.Subscriber(
            self.color_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.namedWindow(self.preview_name, cv2.WINDOW_AUTOSIZE)

        rospy.loginfo("A4 corner calibrator started")
        rospy.loginfo("Color topic: %s", self.color_topic)
        rospy.loginfo("Output file: %s", self.output_file)
        rospy.loginfo("Click order: top-left, top-right, bottom-right, bottom-left")

    def image_callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_bgr = bgr.copy()
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Image callback error: %s", exc)

    def mouse_callback(self, event, x, y, flags, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 4:
            self.last_message = "已经有4个点；按 U 撤销或 R 重置"
            return
        scale = max(self.display_scale, 1e-9)
        image_x = float(x) / scale
        image_y = float(y) / scale
        self.points.append((image_x, image_y))
        idx = len(self.points) - 1
        self.last_message = "已记录 {} ({:.1f}, {:.1f})".format(
            POINT_NAMES_ZH[idx], image_x, image_y
        )

    @staticmethod
    def validate_quad(points: List[Tuple[float, float]]) -> Tuple[bool, str]:
        if len(points) != 4:
            return False, "必须恰好点击4个点"
        quad = np.asarray(points, dtype=np.float32)
        contour = quad.reshape(-1, 1, 2)
        area = abs(float(cv2.contourArea(contour)))
        if area < 10000.0:
            return False, "四边形面积过小，可能点错"
        if not cv2.isContourConvex(np.round(contour).astype(np.int32)):
            return False, "四点顺序错误或四边形不凸；请按左上、右上、右下、左下点击"

        edges = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if min(edges) < 30.0:
            return False, "存在过短边，可能重复点击"
        return True, "OK"

    @staticmethod
    def compute_warp(frame: np.ndarray, points: List[Tuple[float, float]]) -> Optional[np.ndarray]:
        valid, _ = A4CornerCalibrator.validate_quad(points)
        if not valid:
            return None
        src = np.asarray(points, dtype=np.float32)
        dst = np.array(
            [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
            dtype=np.float32,
        )
        h = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, h, (WARP_W, WARP_H))

    def save(self, frame: np.ndarray) -> bool:
        valid, reason = self.validate_quad(self.points)
        if not valid:
            self.last_message = "无法保存：" + reason
            rospy.logwarn(self.last_message)
            return False

        h, w = frame.shape[:2]
        corners = {
            name: [round(float(p[0]), 3), round(float(p[1]), 3)]
            for name, p in zip(POINT_NAMES, self.points)
        }
        data = {
            "version": 1,
            "color_topic": self.color_topic,
            "image_width": int(w),
            "image_height": int(h),
            "paper_width_mm": PAPER_W_MM,
            "paper_height_mm": PAPER_H_MM,
            "click_order": list(POINT_NAMES),
            "corners": corners,
        }

        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
        tmp_file = self.output_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                if yaml is not None:
                    yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
                else:
                    # JSON 是 YAML 1.2 的合法子集，避免额外依赖。
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.write("\n")
            os.replace(tmp_file, self.output_file)
            self.last_message = "保存成功：{}".format(self.output_file)
            rospy.loginfo(self.last_message)
            return True
        except Exception as exc:
            self.last_message = "保存失败：{}".format(exc)
            rospy.logerr(self.last_message)
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                pass
            return False

    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        shown = frame.copy()
        colors = ((0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0))

        for idx, point in enumerate(self.points):
            p = (int(round(point[0])), int(round(point[1])))
            cv2.circle(shown, p, 7, colors[idx], -1)
            cv2.circle(shown, p, 11, (0, 0, 0), 2)
            cv2.putText(
                shown,
                "{} {}".format(idx + 1, POINT_NAMES[idx]),
                (p[0] + 12, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colors[idx],
                2,
                cv2.LINE_AA,
            )

        if len(self.points) >= 2:
            pts = np.round(np.asarray(self.points)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(shown, [pts], len(self.points) == 4, (0, 255, 0), 2)

        next_name = POINT_NAMES_ZH[len(self.points)] if len(self.points) < 4 else "完成"
        lines = [
            "点击黑色A4有效区域四角，不要点击外侧白框/托板",
            "依次点击：左上 -> 右上 -> 右下 -> 左下",
            "当前：{}/4；下一点：{}".format(len(self.points), next_name),
            "S/Enter 保存 | U 撤销 | R 重置 | Q/Esc 退出",
            self.last_message,
        ]
        y = 30
        for line in lines:
            (tw, th), base = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.63, 2)
            cv2.rectangle(shown, (8, y - th - 6), (16 + tw, y + base + 3), (25, 25, 25), -1)
            cv2.putText(
                shown,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 31
        return shown

    def resize_for_display(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(
            1.0,
            float(self.display_max_width) / max(w, 1),
            float(self.display_max_height) / max(h, 1),
        )
        self.display_scale = scale
        if abs(scale - 1.0) < 1e-9:
            return image
        return cv2.resize(
            image,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def resize_preview(image: np.ndarray, max_height: int = 760) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, float(max_height) / max(h, 1))
        if abs(scale - 1.0) < 1e-9:
            return image
        return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                          interpolation=cv2.INTER_AREA)

    def run(self) -> None:
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            with self.lock:
                frame = None if self.latest_bgr is None else self.latest_bgr.copy()

            if frame is not None:
                overlay = self.draw_overlay(frame)
                cv2.imshow(self.window_name, self.resize_for_display(overlay))
                warped = self.compute_warp(frame, self.points)
                if warped is None:
                    preview = np.zeros((600, 430, 3), dtype=np.uint8)
                    cv2.putText(
                        preview,
                        "click 4 corners first",
                        (35, 295),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    preview = warped
                    cv2.line(preview, (0, WARP_H // 2), (WARP_W - 1, WARP_H // 2), (0, 0, 255), 2)
                cv2.imshow(self.preview_name, self.resize_preview(preview))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("r"), ord("R")):
                self.points.clear()
                self.last_message = "已重置"
            elif key in (ord("u"), ord("U"), 8):
                if self.points:
                    self.points.pop()
                    self.last_message = "已撤销最后一个点"
            elif key in (ord("s"), ord("S"), 13, 10):
                if frame is not None:
                    self.save(frame)

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        A4CornerCalibrator().run()
    except rospy.ROSInterruptException:
        pass
