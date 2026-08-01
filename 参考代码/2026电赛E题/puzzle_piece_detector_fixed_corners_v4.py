#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E题自备四片碎片识别与机械臂坐标计算（ROS1）

功能：
1. 订阅彩色图和对齐到彩色图的深度图；
2. 从 YAML 读取人工点击保存的黑色 A4 四角，并进行固定透视矫正；
3. 在黑纸区域内用“相对亮度 + Otsu + 低饱和度”识别白色碎片；
4. 根据图2固定几何模板识别 P1~P4；
5. 优先选择碎片几何中心作为吸取点，并计算当前机器人坐标、带间隔的目标放置坐标和所需平面旋转量；
6. 在图像上绘制 A4 边界、四片轮廓、外接框和吸取点；面积、角度、坐标统一显示在 A4 外的图像角落信息面板；
7. 发布原图标注、俯视图、二值图、PoseArray 和 JSON 坐标。

注意：
- 本节点默认不直接驱动机械臂，避免检测抖动造成误动作。
- 机械臂命令 Z 默认沿用原程序：抓取 -52 mm，放置 -45 mm，可用 ROS 参数修改。
- 现有 swiftpro.msg.position 只有 x/y/z，不能命令碎片绕竖直轴旋转；
  本节点会输出 required_rotation_deg，真正拼图仍需末端旋转轴或其他转向机构。
"""

import itertools
import json
import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import String


# ---------- 与原程序一致的彩色相机内参 ----------
CAMERA_MATRIX = np.array(
    [
        [605.2310180664062, 0.0, 316.76287841796875],
        [0.0, 604.7352294921875, 253.9280548095703],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# A4 纸实际尺寸，纵向放置
PAPER_W_MM = 210.0
PAPER_H_MM = 297.0

# 标准俯视图：4 px/mm
WARP_W = 840
WARP_H = 1188
PX_PER_MM_X = WARP_W / PAPER_W_MM
PX_PER_MM_Y = WARP_H / PAPER_H_MM

# 图2固定碎片模板，坐标原点为目标矩形左上角，单位 mm。
# 目标矩形 100 mm × 60 mm。
# 主对角线从 (20, 0) 到 (100, 60)，长度 100 mm；
# 对角线分点分别距起点 20 mm、距终点 30 mm。
TEMPLATE_POINTS_MM: Dict[int, np.ndarray] = {
    1: np.array([[0.0, 0.0], [20.0, 0.0], [36.0, 12.0], [0.0, 20.0]], dtype=np.float64),
    2: np.array([[20.0, 0.0], [100.0, 0.0], [100.0, 60.0]], dtype=np.float64),
    3: np.array([[0.0, 20.0], [36.0, 12.0], [76.0, 42.0], [0.0, 30.0]], dtype=np.float64),
    4: np.array([[0.0, 30.0], [76.0, 42.0], [100.0, 60.0], [0.0, 60.0]], dtype=np.float64),
}

EXPECTED_AREA_MM2 = {
    1: 480.0,
    2: 2400.0,
    3: 1080.0,
    4: 2040.0,
}
EXPECTED_VERTICES = {1: 4, 2: 3, 3: 4, 4: 4}

# cv2.matchShapes 使用的模板轮廓
TEMPLATE_CONTOURS = {
    label: np.round(points * 10.0).astype(np.int32).reshape(-1, 1, 2)
    for label, points in TEMPLATE_POINTS_MM.items()
}


@dataclass
class Candidate:
    contour_px: np.ndarray
    polygon_px: np.ndarray
    area_px: float
    area_mm2: float
    bbox_px: Tuple[int, int, int, int]
    grasp_warp_px: Tuple[float, float]
    shape_costs: Dict[int, float]


@dataclass
class PieceResult:
    label: int
    candidate: Candidate
    assignment_cost: float
    current_yaw_deg_clockwise: Optional[float]
    required_rotation_deg_clockwise: Optional[float]
    pose_scale: Optional[float]
    pose_rmse_mm: Optional[float]
    template_to_observed_R: Optional[np.ndarray]
    template_to_observed_t: Optional[np.ndarray]
    canonical_grasp_mm: Optional[np.ndarray]
    pick_pixel: Tuple[float, float]
    place_pixel: Optional[Tuple[float, float]]
    depth_pick_mm: Optional[float]
    depth_place_mm: Optional[float]
    measured_pick_robot_xyz: Optional[Tuple[float, float, float]]
    measured_place_robot_xyz: Optional[Tuple[float, float, float]]
    pick_command_xyz: Optional[Tuple[float, float, float]]
    place_command_xyz: Optional[Tuple[float, float, float]]


class PuzzlePieceDetector:
    def __init__(self) -> None:
        rospy.init_node("puzzle_piece_detector", anonymous=False)
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_depth: Optional[np.ndarray] = None
        self.last_process_time = 0.0

        # ---------- 与原程序一致的话题默认值 ----------
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )

        # ---------- A4 四角文件 ----------
        self.paper_corner_file = os.path.expanduser(
            rospy.get_param("~paper_corner_file", "~/.ros/puzzle_a4_corners.yaml")
        )
        self.allow_auto_a4_fallback = bool(
            rospy.get_param("~allow_auto_a4_fallback", False)
        )
        self.paper_quad_saved: Optional[np.ndarray] = None
        self.paper_calib_image_size: Optional[Tuple[int, int]] = None
        self.paper_source = "none"
        self.load_paper_corners()

        # ---------- 处理与阈值参数 ----------
        self.process_hz = float(rospy.get_param("~process_hz", 5.0))
        self.black_gray_max = int(rospy.get_param("~black_gray_max", 105))
        self.white_margin = int(rospy.get_param("~white_margin", 40))
        self.white_sat_max = int(rospy.get_param("~white_sat_max", 165))
        self.paper_border_ignore_mm = float(
            rospy.get_param("~paper_border_ignore_mm", 8.0)
        )
        self.divider_ignore_mm = float(
            rospy.get_param("~divider_ignore_mm", 0.0)
        )
        # 识别区域默认使用整张 A4。只有明确需要时才设为 upper/lower。
        # 之前默认 upper 会把位于几何中线以下的正常碎片直接清零。
        self.source_region = str(rospy.get_param("~source_region", "full")).lower()
        self.min_piece_area_mm2 = float(rospy.get_param("~min_piece_area_mm2", 260.0))
        self.max_piece_area_mm2 = float(rospy.get_param("~max_piece_area_mm2", 3000.0))
        self.max_candidate_count = int(rospy.get_param("~max_candidate_count", 9))
        self.depth_patch_radius = int(rospy.get_param("~depth_patch_radius", 5))
        self.plane_inlier_mm = float(rospy.get_param("~plane_inlier_mm", 5.0))
        self.plane_ransac_iterations = int(rospy.get_param("~plane_ransac_iterations", 120))
        self.min_valid_depth_mm = float(rospy.get_param("~min_valid_depth_mm", 100.0))
        self.max_valid_depth_mm = float(rospy.get_param("~max_valid_depth_mm", 3000.0))

        # 沿用原程序的机械臂 Z 命令
        self.grasp_z = float(rospy.get_param("~grasp_z", -52.0))
        self.place_z = float(rospy.get_param("~place_z", -45.0))

        # 目标矩形默认在“碎片所在半区的相反半区”中央。
        self.target_rect_w_mm = 100.0
        self.target_rect_h_mm = 60.0

        # 放置时给相邻碎片留出少量间隔，避免机械臂定位误差导致重叠。
        # 该值表示各碎片相对理论拼合位置的平移量级，默认 2.5 mm。
        self.placement_gap_mm = float(
            rospy.get_param("~placement_gap_mm", 2.5)
        )

        # 吸取点优先使用轮廓几何中心；若中心离边缘过近，则移动到
        # 距中心最近且具有足够边缘余量的位置。
        self.grasp_edge_margin_mm = float(
            rospy.get_param("~grasp_edge_margin_mm", 3.0)
        )

        self.calib_file = os.path.expanduser(
            rospy.get_param("~calib_file", "~/thefile.txt")
        )
        self.calib_mode = "none"
        self.T_cam_to_robot: Optional[np.ndarray] = None
        self.x_kb: Optional[Sequence[float]] = None
        self.y_kb: Optional[Sequence[float]] = None
        self.load_calibration()

        self.annotated_pub = rospy.Publisher(
            "/puzzle/annotated_image", Image, queue_size=1
        )
        self.mask_pub = rospy.Publisher("/puzzle/piece_mask", Image, queue_size=1)
        self.warped_pub = rospy.Publisher("/puzzle/warped_image", Image, queue_size=1)
        self.warped_mask_pub = rospy.Publisher(
            "/puzzle/warped_piece_mask", Image, queue_size=1
        )
        self.pose_pub = rospy.Publisher(
            "/puzzle/piece_command_poses", PoseArray, queue_size=1
        )
        self.coords_pub = rospy.Publisher(
            "/puzzle/piece_coordinates", String, queue_size=1
        )

        self.depth_sub = rospy.Subscriber(
            self.depth_topic, Image, self.depth_callback, queue_size=1
        )
        self.color_sub = rospy.Subscriber(
            self.color_topic,
            Image,
            self.color_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo("Puzzle detector started")
        rospy.loginfo("Color topic: %s", self.color_topic)
        rospy.loginfo("Depth topic: %s", self.depth_topic)
        rospy.loginfo("Calibration: %s (%s)", self.calib_file, self.calib_mode)
        rospy.loginfo(
            "Paper corners: %s (%s)",
            self.paper_corner_file,
            "loaded" if self.paper_quad_saved is not None else "not loaded",
        )
        rospy.loginfo(
            "Grasp center margin: %.1f mm; placement gap: %.1f mm",
            self.grasp_edge_margin_mm,
            self.placement_gap_mm,
        )

    # ------------------------------------------------------------------
    # 标定和坐标变换
    # ------------------------------------------------------------------
    def load_paper_corners(self) -> None:
        """读取 a4_corner_calibrator.py 保存的 YAML/JSON 文件。"""
        try:
            with open(self.paper_corner_file, "r", encoding="utf-8") as f:
                if yaml is not None:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            corners_obj = data.get("corners") if isinstance(data, dict) else None
            if isinstance(corners_obj, dict):
                names = ("top_left", "top_right", "bottom_right", "bottom_left")
                points = [corners_obj[name] for name in names]
            elif isinstance(corners_obj, list):
                points = corners_obj
            else:
                raise ValueError("corners 字段不存在或格式错误")

            quad = np.asarray(points, dtype=np.float32).reshape(4, 2)
            if not cv2.isContourConvex(np.round(quad).astype(np.int32).reshape(-1, 1, 2)):
                raise ValueError("四角不是凸四边形，顺序应为左上、右上、右下、左下")
            if abs(cv2.contourArea(quad.reshape(-1, 1, 2))) < 10000.0:
                raise ValueError("四角面积过小")

            width = int(data.get("image_width", 0)) if isinstance(data, dict) else 0
            height = int(data.get("image_height", 0)) if isinstance(data, dict) else 0
            self.paper_quad_saved = quad
            self.paper_calib_image_size = (width, height) if width > 0 and height > 0 else None
            rospy.loginfo("Loaded fixed A4 corners: %s", quad.tolist())
        except Exception as exc:
            self.paper_quad_saved = None
            self.paper_calib_image_size = None
            rospy.logerr(
                "Failed to load A4 corner file %s: %s", self.paper_corner_file, exc
            )
            if not self.allow_auto_a4_fallback:
                rospy.logerr(
                    "Automatic A4 fallback is disabled. Run a4_corner_calibrator.py first."
                )

    def calibrated_quad_for_frame(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.paper_quad_saved is None:
            return None
        quad = self.paper_quad_saved.copy()
        current_h, current_w = bgr.shape[:2]
        if self.paper_calib_image_size is not None:
            saved_w, saved_h = self.paper_calib_image_size
            if saved_w > 0 and saved_h > 0 and (saved_w != current_w or saved_h != current_h):
                quad[:, 0] *= float(current_w) / float(saved_w)
                quad[:, 1] *= float(current_h) / float(saved_h)
                rospy.logwarn_throttle(5.0,
                    "Camera resolution differs from corner calibration; corners are scaled from %dx%d to %dx%d",
                    saved_w, saved_h, current_w, current_h)
        return quad.astype(np.float32)

    def load_calibration(self) -> None:
        try:
            with open(self.calib_file, "r", encoding="utf-8") as f:
                data = [float(x) for line in f for x in line.strip().split()]

            if len(data) == 16:
                self.T_cam_to_robot = np.array(data, dtype=np.float64).reshape(4, 4)
                self.calib_mode = "matrix"
                rospy.loginfo("Loaded 4x4 camera-to-robot matrix:\n%s", self.T_cam_to_robot)
            elif len(data) == 4:
                self.x_kb = [data[0], data[1]]
                self.y_kb = [data[2], data[3]]
                self.calib_mode = "linear"
                rospy.logwarn(
                    "Loaded legacy linear calibration: x=kx*u+bx, y=ky*v+by"
                )
            else:
                raise ValueError("calibration file must contain 16 or 4 numbers")
        except Exception as exc:
            self.calib_mode = "none"
            rospy.logerr("Failed to load calibration %s: %s", self.calib_file, exc)


    def get_depth_snapshot(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.latest_depth is None:
                return None
            return self.latest_depth.copy()

    def fit_support_plane_camera(
        self, depth: Optional[np.ndarray], quad: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        在相机坐标系中拟合 A4/桌面支撑平面 n·P+d=0。
        黑纸可能让部分深度为 0，因此 A4 内有效点不足时退化到整幅图的主平面。
        """
        if depth is None or depth.ndim != 2:
            return None
        h, w = depth.shape

        def collect(mask: np.ndarray, step: int) -> np.ndarray:
            ys, xs = np.where(mask[::step, ::step] > 0)
            ys = ys * step
            xs = xs * step
            z = depth[ys, xs].astype(np.float64)
            valid = (z >= self.min_valid_depth_mm) & (z <= self.max_valid_depth_mm)
            xs = xs[valid].astype(np.float64)
            ys = ys[valid].astype(np.float64)
            z = z[valid]
            if z.size == 0:
                return np.empty((0, 3), dtype=np.float64)
            x = (xs - CAMERA_MATRIX[0, 2]) * z / CAMERA_MATRIX[0, 0]
            y = (ys - CAMERA_MATRIX[1, 2]) * z / CAMERA_MATRIX[1, 1]
            return np.column_stack([x, y, z])

        paper_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(paper_mask, np.round(quad).astype(np.int32), 255)
        points = collect(paper_mask, 6)
        if len(points) < 100:
            # 黑色表面深度稀疏时，从整幅图寻找占比最大的桌面平面。
            full_mask = np.full((h, w), 255, dtype=np.uint8)
            points = collect(full_mask, 10)
        if len(points) < 60:
            return None

        if len(points) > 6000:
            rng = np.random.default_rng(12345)
            points = points[rng.choice(len(points), 6000, replace=False)]

        rng = np.random.default_rng(2026)
        best_inliers = None
        best_count = 0
        for _ in range(max(20, self.plane_ransac_iterations)):
            ids = rng.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[ids]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue
            normal /= norm
            d = -float(np.dot(normal, p1))
            distances = np.abs(points @ normal + d)
            inliers = distances <= self.plane_inlier_mm
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < 50:
            return None

        inlier_points = points[best_inliers]
        center = inlier_points.mean(axis=0)
        _, _, vt = np.linalg.svd(inlier_points - center, full_matrices=False)
        normal = vt[-1]
        normal /= np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        d = -float(np.dot(normal, center))
        return normal, d

    @staticmethod
    def plane_depth_at_pixel(
        u: float, v: float, plane: Optional[Tuple[np.ndarray, float]]
    ) -> Optional[float]:
        if plane is None:
            return None
        normal, d = plane
        ray = np.linalg.inv(CAMERA_MATRIX) @ np.array([u, v, 1.0], dtype=np.float64)
        denom = float(np.dot(normal, ray))
        if abs(denom) < 1e-9:
            return None
        lam = -d / denom
        if not math.isfinite(lam) or lam <= 0:
            return None
        # ray 的第三维为 1，因此 lam 就是相机 Z 深度。
        return float(lam)

    def robust_depth_at(self, u: float, v: float) -> Optional[float]:
        with self.lock:
            if self.latest_depth is None:
                return None
            depth = self.latest_depth.copy()

        h, w = depth.shape[:2]
        x = int(round(u))
        y = int(round(v))
        r = max(1, self.depth_patch_radius)
        x1, x2 = max(0, x - r), min(w, x + r + 1)
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        patch = depth[y1:y2, x1:x2].astype(np.float64).reshape(-1)
        valid = patch[
            (patch >= self.min_valid_depth_mm)
            & (patch <= self.max_valid_depth_mm)
            & np.isfinite(patch)
        ]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def pixel_to_robot_xyz(
        self, u: float, v: float, depth_mm: Optional[float]
    ) -> Optional[Tuple[float, float, float]]:
        if self.calib_mode == "matrix":
            if self.T_cam_to_robot is None or depth_mm is None:
                return None
            zc = float(depth_mm)
            xc = (u - CAMERA_MATRIX[0, 2]) * zc / CAMERA_MATRIX[0, 0]
            yc = (v - CAMERA_MATRIX[1, 2]) * zc / CAMERA_MATRIX[1, 1]
            p_cam = np.array([xc, yc, zc, 1.0], dtype=np.float64)
            p_robot = self.T_cam_to_robot @ p_cam
            return float(p_robot[0]), float(p_robot[1]), float(p_robot[2])

        if self.calib_mode == "linear" and self.x_kb and self.y_kb:
            x = self.x_kb[0] * u + self.x_kb[1]
            y = self.y_kb[0] * v + self.y_kb[1]
            return float(x), float(y), float("nan")

        return None

    # ------------------------------------------------------------------
    # ROS 回调
    # ------------------------------------------------------------------
    def depth_callback(self, msg: Image) -> None:
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            with self.lock:
                self.latest_depth = depth
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Depth callback error: %s", exc)

    def color_callback(self, msg: Image) -> None:
        now = rospy.get_time()
        if now - self.last_process_time < 1.0 / max(self.process_hz, 0.1):
            return
        self.last_process_time = now

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated, mask, warped, warped_mask, results = self.process_frame(bgr)
            self.publish_outputs(
                msg.header, annotated, mask, warped, warped_mask, results
            )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Color processing error: %s", exc)

    # ------------------------------------------------------------------
    # A4 检测
    # ------------------------------------------------------------------
    @staticmethod
    def order_quad(points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
        result = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        result[0] = pts[np.argmin(s)]      # top-left
        result[2] = pts[np.argmax(s)]      # bottom-right
        result[1] = pts[np.argmin(d)]      # top-right
        result[3] = pts[np.argmax(d)]      # bottom-left
        return result

    @staticmethod
    def orient_portrait(quad: np.ndarray) -> np.ndarray:
        q = quad.copy()
        width = 0.5 * (
            np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])
        )
        height = 0.5 * (
            np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])
        )
        if width > height:
            q = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
        return q

    def score_a4_quad(self, quad: np.ndarray, image_area: float) -> float:
        q = self.order_quad(quad)
        sides = [
            np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)
        ]
        short = max(1.0, min(np.mean([sides[0], sides[2]]), np.mean([sides[1], sides[3]])))
        long = max(np.mean([sides[0], sides[2]]), np.mean([sides[1], sides[3]]))
        ratio = long / short
        area = abs(cv2.contourArea(q.reshape(-1, 1, 2)))
        if area < 0.08 * image_area or not (1.15 <= ratio <= 1.85):
            return -1.0
        ratio_score = math.exp(-3.0 * abs(math.log(ratio / math.sqrt(2.0))))
        return (area / image_area) * ratio_score

    def find_a4_quad(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        image_area = float(gray.shape[0] * gray.shape[1])

        masks = []
        dark = cv2.inRange(blur, 0, self.black_gray_max)
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
            iterations=2,
        )
        masks.append(dark)

        edges = cv2.Canny(blur, 40, 130)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        masks.append(edges)

        best_score = -1.0
        best_quad = None
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                hull = cv2.convexHull(contour)
                perimeter = cv2.arcLength(hull, True)
                if perimeter <= 0:
                    continue
                approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = approx.reshape(4, 2).astype(np.float32)
                score = self.score_a4_quad(quad, image_area)
                if score > best_score:
                    best_score = score
                    best_quad = quad

        if best_quad is None:
            return None
        return self.orient_portrait(self.order_quad(best_quad))

    def get_a4_quad(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        fixed = self.calibrated_quad_for_frame(bgr)
        if fixed is not None:
            self.paper_source = "yaml"
            return fixed
        if self.allow_auto_a4_fallback:
            auto = self.find_a4_quad(bgr)
            self.paper_source = "auto" if auto is not None else "none"
            return auto
        self.paper_source = "none"
        return None

    # ------------------------------------------------------------------
    # 白色碎片分割与固定模板识别
    # ------------------------------------------------------------------
    @staticmethod
    def fill_external_contours(mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def segment_white_pieces(self, warped: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # 黑纸上的白片反差很大，但不使用单一固定阈值：
        # 1) Otsu 根据当前帧自动分割；2) 与纸面中值保持至少 white_margin 的亮度差。
        paper_median = float(np.median(gray))
        otsu_t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(max(float(otsu_t), paper_median + self.white_margin))
        threshold = max(35, min(threshold, 245))

        mask = np.zeros_like(gray, dtype=np.uint8)
        white = (gray >= threshold) & (sat <= self.white_sat_max)
        mask[white] = 255

        # 固定四角后，纸面在标准坐标中稳定。扩大边缘屏蔽，彻底去除白色托板/纸边。
        border_x = int(round(self.paper_border_ignore_mm * PX_PER_MM_X))
        border_y = int(round(self.paper_border_ignore_mm * PX_PER_MM_Y))
        mask[:border_y, :] = 0
        mask[-border_y:, :] = 0
        mask[:, :border_x] = 0
        mask[:, -border_x:] = 0

        mid = WARP_H // 2

        # 只有在明确设置 divider_ignore_mm > 0 时才屏蔽中线。
        # 你的当前纸面没有可见分界线；默认值为 0，避免切掉靠近中线的碎片。
        line_band = int(round(max(0.0, self.divider_ignore_mm) * PX_PER_MM_Y))
        if line_band > 0:
            mask[max(0, mid - line_band):min(WARP_H, mid + line_band), :] = 0

        # 默认识别整张 A4。upper/lower 仅作为可选模式，不能作为默认假设。
        # 题面虽说碎片在上半区域，但实际摆放、标定误差和碎片尺寸都可能使轮廓
        # 跨越几何中线；直接 mask[mid:, :] = 0 会把完整碎片截断。
        if self.source_region == "upper":
            mask[mid:, :] = 0
        elif self.source_region == "lower":
            mask[:mid, :] = 0
        elif self.source_region not in ("full", "auto"):
            rospy.logwarn_throttle(5.0, "Unknown source_region=%s; using full", self.source_region)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        return self.fill_external_contours(mask)

    def grasp_point_from_contour(
        self, contour: np.ndarray, shape: Tuple[int, int]
    ) -> Tuple[float, float]:
        """返回中心优先、同时远离边缘的吸取点。

        旧版本直接选取最大内接圆圆心。对于不规则或细长碎片，这个点
        可能明显偏向较宽的一端，看起来像“靠边抓取”。新版先计算轮廓
        几何中心；若中心处的边缘余量不足，再选择距离几何中心最近的
        安全内部点。
        """
        local_mask = np.zeros(shape, dtype=np.uint8)
        cv2.drawContours(local_mask, [contour], -1, 255, cv2.FILLED)
        distance = cv2.distanceTransform(local_mask, cv2.DIST_L2, 5)

        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-6:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + 0.5 * w
            center_y = y + 0.5 * h

        cx = int(round(np.clip(center_x, 0, shape[1] - 1)))
        cy = int(round(np.clip(center_y, 0, shape[0] - 1)))
        required_margin_px = max(
            1.0,
            self.grasp_edge_margin_mm
            * 0.5
            * (PX_PER_MM_X + PX_PER_MM_Y),
        )

        # 几何中心本身安全时，直接使用中心。
        if local_mask[cy, cx] != 0 and distance[cy, cx] >= required_margin_px:
            return center_x, center_y

        # 在满足边缘余量的内部区域中，选择离几何中心最近的点。
        safe_yx = np.argwhere(distance >= required_margin_px)
        if safe_yx.size > 0:
            dx = safe_yx[:, 1].astype(np.float64) - center_x
            dy = safe_yx[:, 0].astype(np.float64) - center_y
            index = int(np.argmin(dx * dx + dy * dy))
            return float(safe_yx[index, 1]), float(safe_yx[index, 0])

        # 极窄碎片没有满足余量的区域时，退回最大内接圆圆心。
        _, _, _, max_loc = cv2.minMaxLoc(distance)
        return float(max_loc[0]), float(max_loc[1])

    def extract_candidates(self, piece_mask: np.ndarray) -> List[Candidate]:
        contours, _ = cv2.findContours(piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        candidates: List[Candidate] = []
        px_area_per_mm2 = PX_PER_MM_X * PX_PER_MM_Y

        for contour in contours:
            hull = cv2.convexHull(contour)
            area_px = float(cv2.contourArea(hull))
            area_mm2 = area_px / px_area_per_mm2
            if not (self.min_piece_area_mm2 <= area_mm2 <= self.max_piece_area_mm2):
                continue

            rect = cv2.minAreaRect(hull)
            rw, rh = rect[1]
            if min(rw, rh) < 1.0:
                continue
            elongation = max(rw, rh) / min(rw, rh)
            if elongation > 5.0:
                # 典型的分界线或反光长条
                continue

            perimeter = cv2.arcLength(hull, True)
            polygon = cv2.approxPolyDP(hull, 0.012 * perimeter, True)
            bbox = cv2.boundingRect(hull)
            grasp = self.grasp_point_from_contour(hull, piece_mask.shape)

            shape_costs = {
                label: float(cv2.matchShapes(hull, template, cv2.CONTOURS_MATCH_I1, 0.0))
                for label, template in TEMPLATE_CONTOURS.items()
            }
            candidates.append(
                Candidate(
                    contour_px=hull,
                    polygon_px=polygon,
                    area_px=area_px,
                    area_mm2=area_mm2,
                    bbox_px=bbox,
                    grasp_warp_px=grasp,
                    shape_costs=shape_costs,
                )
            )

        # 过多候选时，只保留最像任一模板的若干项，避免组合爆炸。
        def candidate_quality(c: Candidate) -> float:
            best = float("inf")
            for label in (1, 2, 3, 4):
                area_cost = abs(math.log(max(c.area_mm2, 1.0) / EXPECTED_AREA_MM2[label]))
                shape_cost = c.shape_costs[label]
                best = min(best, 0.75 * area_cost + 2.0 * shape_cost)
            return best

        candidates.sort(key=candidate_quality)
        return candidates[: self.max_candidate_count]

    @staticmethod
    def assignment_pair_cost(candidate: Candidate, label: int) -> float:
        area_cost = abs(math.log(max(candidate.area_mm2, 1.0) / EXPECTED_AREA_MM2[label]))
        vertex_count = len(candidate.polygon_px)
        vertex_cost = abs(vertex_count - EXPECTED_VERTICES[label])
        shape_cost = candidate.shape_costs[label]
        return 0.8 * area_cost + 2.2 * shape_cost + 0.18 * vertex_cost

    def assign_four_pieces(
        self, candidates: List[Candidate]
    ) -> Tuple[Optional[Dict[int, Candidate]], float]:
        if len(candidates) < 4:
            return None, float("inf")

        labels = (1, 2, 3, 4)
        best_assignment = None
        best_cost = float("inf")

        for indices in itertools.combinations(range(len(candidates)), 4):
            subset = [candidates[i] for i in indices]
            for label_perm in itertools.permutations(labels):
                total = 0.0
                mapping: Dict[int, Candidate] = {}
                for candidate, label in zip(subset, label_perm):
                    total += self.assignment_pair_cost(candidate, label)
                    mapping[label] = candidate
                if total < best_cost:
                    best_cost = total
                    best_assignment = mapping

        return best_assignment, best_cost

    # ------------------------------------------------------------------
    # 固定模板到观测多边形的二维相似变换，用于角度和目标放置点
    # ------------------------------------------------------------------
    @staticmethod
    def polygon_signed_area(points: np.ndarray) -> float:
        p = np.asarray(points, dtype=np.float64)
        return 0.5 * float(
            np.dot(p[:, 0], np.roll(p[:, 1], -1))
            - np.dot(p[:, 1], np.roll(p[:, 0], -1))
        )

    @classmethod
    def normalize_polygon_order(cls, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).copy()
        if cls.polygon_signed_area(p) < 0:
            p = p[::-1]
        return p

    @staticmethod
    def approximate_contour_to_n(contour: np.ndarray, n: int) -> Optional[np.ndarray]:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        exact = []
        for eps_ratio in np.linspace(0.003, 0.065, 80):
            approx = cv2.approxPolyDP(hull, float(eps_ratio * perimeter), True)
            if len(approx) == n:
                exact.append(approx.reshape(-1, 2).astype(np.float64))
        if not exact:
            return None
        # 取 epsilon 较小得到的第一组，尽量保留真实顶点。
        return exact[0]

    @staticmethod
    def fit_similarity(template: np.ndarray, observed: np.ndarray):
        """返回 scale, R, t, rmse，使 observed ~= scale * template @ R.T + t。"""
        a = np.asarray(template, dtype=np.float64)
        b = np.asarray(observed, dtype=np.float64)
        ca = a.mean(axis=0)
        cb = b.mean(axis=0)
        aa = a - ca
        bb = b - cb
        h = aa.T @ bb
        u, s, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        denom = float(np.sum(aa * aa))
        if denom <= 1e-9:
            return None
        scale = float(np.sum(s) / denom)
        t = cb - scale * (r @ ca)
        predicted = (scale * (r @ a.T)).T + t
        rmse = float(np.sqrt(np.mean(np.sum((predicted - b) ** 2, axis=1))))
        return scale, r, t, rmse

    def estimate_piece_pose(
        self, label: int, contour_px: np.ndarray
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray, float]]:
        n = EXPECTED_VERTICES[label]
        observed_px = self.approximate_contour_to_n(contour_px, n)
        if observed_px is None:
            return None
        observed_mm = observed_px / np.array([PX_PER_MM_X, PX_PER_MM_Y])
        observed_mm = self.normalize_polygon_order(observed_mm)
        template_mm = self.normalize_polygon_order(TEMPLATE_POINTS_MM[label])

        best = None
        for shift in range(n):
            rolled = np.roll(observed_mm, -shift, axis=0)
            fit = self.fit_similarity(template_mm, rolled)
            if fit is None:
                continue
            if best is None or fit[3] < best[3]:
                best = fit
        return best

    @staticmethod
    def warp_to_original(point_warp: Tuple[float, float], h_inv: np.ndarray) -> Tuple[float, float]:
        p = np.array([[[point_warp[0], point_warp[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(p, h_inv)[0, 0]
        return float(result[0]), float(result[1])

    @staticmethod
    def warp_contour_to_original(contour: np.ndarray, h_inv: np.ndarray) -> np.ndarray:
        points = contour.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(points, h_inv)

    @staticmethod
    def target_origin_mm(source_half: str) -> np.ndarray:
        x = (PAPER_W_MM - 100.0) / 2.0
        if source_half == "upper":
            y = PAPER_H_MM / 2.0 + (PAPER_H_MM / 2.0 - 60.0) / 2.0
        else:
            y = (PAPER_H_MM / 2.0 - 60.0) / 2.0
        return np.array([x, y], dtype=np.float64)

    def placement_offset_mm(self, label: int) -> np.ndarray:
        """为四片设置轻微分离的目标平移，减少放置重叠。"""
        gap = max(0.0, self.placement_gap_mm)
        offsets = {
            1: np.array([-gap, -gap], dtype=np.float64),
            2: np.array([ gap, -gap], dtype=np.float64),
            3: np.array([-gap,  0.0], dtype=np.float64),
            4: np.array([ gap,  gap], dtype=np.float64),
        }
        return offsets[label]

    # ------------------------------------------------------------------
    # 主处理
    # ------------------------------------------------------------------
    def process_frame(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[PieceResult]]:
        annotated = bgr.copy()
        empty_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        empty_warped = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
        empty_warp_mask = np.zeros((WARP_H, WARP_W), dtype=np.uint8)

        quad = self.get_a4_quad(bgr)
        if quad is None:
            self.draw_info_panel(
                annotated,
                ["A4 NOT FOUND"],
                quad=None,
                text_color=(80, 80, 255),
            )
            return annotated, empty_mask, empty_warped, empty_warp_mask, []

        depth_snapshot = self.get_depth_snapshot()
        support_plane = self.fit_support_plane_camera(depth_snapshot, quad)

        dst = np.array(
            [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
            dtype=np.float32,
        )
        h = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        h_inv = np.linalg.inv(h)
        warped = cv2.warpPerspective(bgr, h, (WARP_W, WARP_H))
        piece_mask_warp = self.segment_white_pieces(warped)
        candidates = self.extract_candidates(piece_mask_warp)
        rospy.loginfo_throttle(
            2.0,
            "piece segmentation: source_region=%s divider_ignore_mm=%.1f candidates=%d areas_mm2=%s",
            self.source_region,
            self.divider_ignore_mm,
            len(candidates),
            [round(c.area_mm2, 1) for c in candidates],
        )
        assignment, assignment_cost = self.assign_four_pieces(candidates)

        quad_i = np.round(quad).astype(np.int32)
        cv2.polylines(
            annotated,
            [quad_i.reshape(-1, 1, 2)],
            True,
            (0, 255, 0),
            2,
        )
        for idx in range(4):
            p = tuple(quad_i[idx])
            cv2.circle(annotated, p, 5, (0, 255, 0), -1)

        # 将归一化二值图投回原图，只用于发布和观察。
        mask_original = cv2.warpPerspective(
            piece_mask_warp,
            h_inv,
            (bgr.shape[1], bgr.shape[0]),
            flags=cv2.INTER_NEAREST,
        )

        if assignment is None:
            self.draw_info_panel(
                annotated,
                ["pieces: {}/4".format(len(candidates))],
                quad=quad,
                text_color=(80, 80, 255),
            )
            cv2.putText(
                warped,
                "candidates: {}/4".format(len(candidates)),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            return annotated, mask_original, warped, piece_mask_warp, []

        mean_y = float(np.mean([c.grasp_warp_px[1] for c in assignment.values()]))
        source_half = "upper" if mean_y < WARP_H / 2.0 else "lower"
        target_origin = self.target_origin_mm(source_half)

        # 在原图上画目标矩形
        target_rect_mm = np.array(
            [
                target_origin,
                target_origin + [100.0, 0.0],
                target_origin + [100.0, 60.0],
                target_origin + [0.0, 60.0],
            ],
            dtype=np.float32,
        )
        target_rect_warp = target_rect_mm * np.array([PX_PER_MM_X, PX_PER_MM_Y], dtype=np.float32)
        target_rect_orig = self.warp_contour_to_original(
            target_rect_warp.reshape(-1, 1, 2), h_inv
        )
        cv2.polylines(
            annotated,
            [np.round(target_rect_orig).astype(np.int32)],
            True,
            (255, 180, 0),
            2,
        )

        results: List[PieceResult] = []
        info_lines: List[str] = []
        for label in (1, 2, 3, 4):
            candidate = assignment[label]
            pose_fit = self.estimate_piece_pose(label, candidate.contour_px)

            yaw_deg = None
            required_deg = None
            pose_scale = None
            pose_rmse = None
            r = None
            t = None
            canonical_grasp = None
            place_warp = None
            place_pixel = None

            grasp_obs_mm = np.array(candidate.grasp_warp_px, dtype=np.float64) / np.array(
                [PX_PER_MM_X, PX_PER_MM_Y]
            )

            if pose_fit is not None:
                pose_scale, r, t, pose_rmse = pose_fit
                yaw_deg = math.degrees(math.atan2(r[1, 0], r[0, 0]))
                required_deg = -yaw_deg
                canonical_grasp = (r.T @ (grasp_obs_mm - t)) / pose_scale
                target_grasp_mm = (
                    target_origin
                    + canonical_grasp
                    + self.placement_offset_mm(label)
                )
                place_warp = (
                    float(target_grasp_mm[0] * PX_PER_MM_X),
                    float(target_grasp_mm[1] * PX_PER_MM_Y),
                )
                place_pixel = self.warp_to_original(place_warp, h_inv)

            pick_pixel = self.warp_to_original(candidate.grasp_warp_px, h_inv)
            # 记录传感器原始深度用于诊断；坐标计算优先使用拟合平面深度。
            # 这样即使黑色 A4 纸在目标位置返回 0 深度，也能得到放置 XY。
            depth_pick = self.robust_depth_at(*pick_pixel)
            plane_depth_pick = self.plane_depth_at_pixel(*pick_pixel, support_plane)
            coordinate_depth_pick = plane_depth_pick if plane_depth_pick is not None else depth_pick
            pick_robot = self.pixel_to_robot_xyz(*pick_pixel, coordinate_depth_pick)

            depth_place = self.robust_depth_at(*place_pixel) if place_pixel else None
            plane_depth_place = (
                self.plane_depth_at_pixel(*place_pixel, support_plane) if place_pixel else None
            )
            coordinate_depth_place = (
                plane_depth_place if plane_depth_place is not None else depth_place
            )
            place_robot = (
                self.pixel_to_robot_xyz(*place_pixel, coordinate_depth_place)
                if place_pixel
                else None
            )

            pick_command = None
            if pick_robot is not None:
                pick_command = (pick_robot[0], pick_robot[1], self.grasp_z)

            place_command = None
            if place_robot is not None:
                place_command = (place_robot[0], place_robot[1], self.place_z)

            result = PieceResult(
                label=label,
                candidate=candidate,
                assignment_cost=self.assignment_pair_cost(candidate, label),
                current_yaw_deg_clockwise=yaw_deg,
                required_rotation_deg_clockwise=required_deg,
                pose_scale=pose_scale,
                pose_rmse_mm=pose_rmse,
                template_to_observed_R=r,
                template_to_observed_t=t,
                canonical_grasp_mm=canonical_grasp,
                pick_pixel=pick_pixel,
                place_pixel=place_pixel,
                depth_pick_mm=depth_pick,
                depth_place_mm=depth_place,
                measured_pick_robot_xyz=pick_robot,
                measured_place_robot_xyz=place_robot,
                pick_command_xyz=pick_command,
                place_command_xyz=place_command,
            )
            results.append(result)

            # 当前碎片轮廓、框和吸取点
            contour_orig = self.warp_contour_to_original(candidate.contour_px, h_inv)
            contour_i = np.round(contour_orig).astype(np.int32)
            cv2.polylines(annotated, [contour_i], True, (0, 255, 255), 2)
            x, y, w, hh = cv2.boundingRect(contour_i)
            cv2.rectangle(annotated, (x, y), (x + w, y + hh), (255, 0, 255), 2)
            px, py = int(round(pick_pixel[0])), int(round(pick_pixel[1]))
            cv2.circle(annotated, (px, py), 5, (0, 0, 255), -1)

            # 目标碎片轮廓：加入轻微分离偏移，显示结果与实际放置一致。
            target_poly_mm = (
                TEMPLATE_POINTS_MM[label]
                + target_origin
                + self.placement_offset_mm(label)
            )
            target_poly_warp = target_poly_mm * np.array([PX_PER_MM_X, PX_PER_MM_Y])
            target_poly_orig = self.warp_contour_to_original(
                target_poly_warp.reshape(-1, 1, 2).astype(np.float32), h_inv
            )
            cv2.polylines(
                annotated,
                [np.round(target_poly_orig).astype(np.int32)],
                True,
                (255, 120, 0),
                1,
            )
            if place_pixel is not None:
                cv2.circle(
                    annotated,
                    (int(round(place_pixel[0])), int(round(place_pixel[1]))),
                    4,
                    (255, 0, 0),
                    -1,
                )

            angle_text = "--" if yaw_deg is None else "{:.1f}deg".format(yaw_deg)
            info_lines.append(
                "P{}  A={:.0f}mm2  yaw={}".format(
                    label, candidate.area_mm2, angle_text
                )
            )
            if pick_command is not None:
                info_lines.append(
                    "    pick X={:.1f} Y={:.1f} Z={:.1f}".format(
                        pick_command[0], pick_command[1], pick_command[2]
                    )
                )
            else:
                info_lines.append("    pick: no calibrated depth")

        status_line = "4/4 cost={:.3f} {} plane={}".format(
            assignment_cost, self.paper_source,
            "OK" if support_plane is not None else "NO"
        )
        self.draw_info_panel(
            annotated,
            [status_line] + info_lines,
            quad=quad,
            text_color=(255, 255, 255),
        )
        # 在俯视图中同步绘制轮廓，便于确认透视和分割是否正确。
        for result in results:
            cv2.polylines(
                warped,
                [np.round(result.candidate.contour_px).astype(np.int32)],
                True,
                (0, 255, 255),
                3,
            )
            gx, gy = result.candidate.grasp_warp_px
            cv2.circle(warped, (int(round(gx)), int(round(gy))), 7, (0, 0, 255), -1)
            cv2.putText(warped, "P{}".format(result.label),
                        (int(round(gx)) + 10, int(round(gy)) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        return annotated, mask_original, warped, piece_mask_warp, results

    @staticmethod
    def _panel_rect_for_corner(
        image_shape: Tuple[int, ...],
        panel_w: int,
        panel_h: int,
        corner: str,
        margin: int = 10,
    ) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        if corner == "top_right":
            x1, y1 = w - margin - panel_w, margin
        elif corner == "bottom_left":
            x1, y1 = margin, h - margin - panel_h
        elif corner == "bottom_right":
            x1, y1 = w - margin - panel_w, h - margin - panel_h
        else:
            x1, y1 = margin, margin
        x1 = max(0, min(x1, max(0, w - panel_w)))
        y1 = max(0, min(y1, max(0, h - panel_h)))
        return x1, y1, x1 + panel_w, y1 + panel_h

    @staticmethod
    def _rect_quad_overlap_area(
        rect: Tuple[int, int, int, int],
        quad: Optional[np.ndarray],
    ) -> float:
        if quad is None:
            return 0.0
        x1, y1, x2, y2 = rect
        rect_poly = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.float32,
        )
        try:
            area, _ = cv2.intersectConvexConvex(
                rect_poly, quad.astype(np.float32)
            )
            return float(area)
        except cv2.error:
            return float("inf")

    def draw_info_panel(
        self,
        image: np.ndarray,
        lines: Sequence[str],
        quad: Optional[np.ndarray],
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """在与 A4 重叠最小的图像角落绘制统一信息面板。

        面板只承载文字，A4 内部不再绘制面积、角度或机械臂坐标。
        """
        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.47
        thickness = 1
        line_gap = 7
        pad_x = 9
        pad_y = 8

        text_sizes = [
            cv2.getTextSize(str(line), font, scale, thickness)[0]
            for line in lines
        ]
        max_tw = max((size[0] for size in text_sizes), default=1)
        max_th = max((size[1] for size in text_sizes), default=12)
        line_step = max_th + line_gap
        panel_w = min(image.shape[1] - 2, max_tw + 2 * pad_x)
        panel_h = min(
            image.shape[0] - 2,
            2 * pad_y + line_step * len(lines),
        )

        corners = ("top_left", "top_right", "bottom_left", "bottom_right")
        candidates = []
        for order, corner in enumerate(corners):
            rect = self._panel_rect_for_corner(
                image.shape, panel_w, panel_h, corner
            )
            overlap = self._rect_quad_overlap_area(rect, quad)
            candidates.append((overlap, order, rect))
        _, _, (x1, y1, x2, y2) = min(candidates, key=lambda item: (item[0], item[1]))

        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(overlay, 0.78, image, 0.22, 0.0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (135, 135, 135), 1)

        baseline_y = y1 + pad_y + max_th
        for index, line in enumerate(lines):
            y = baseline_y + index * line_step
            if y > y2 - pad_y:
                break
            cv2.putText(
                image,
                str(line),
                (x1 + pad_x, y),
                font,
                scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def draw_text_box(
        image: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        background: Tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        cv2.rectangle(
            image,
            (x, y - th - baseline - 2),
            (x + tw + 3, y + 2),
            background,
            cv2.FILLED,
        )
        cv2.putText(image, text, (x + 1, y - 2), font, scale, (255, 255, 255), thickness)

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    @staticmethod
    def finite_or_none(value: Optional[float]):
        if value is None or not math.isfinite(value):
            return None
        return float(value)

    def publish_outputs(
        self,
        header,
        annotated: np.ndarray,
        mask: np.ndarray,
        warped: np.ndarray,
        warped_mask: np.ndarray,
        results: List[PieceResult],
    ) -> None:
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        annotated_msg.header = header
        self.annotated_pub.publish(annotated_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

        warped_msg = self.bridge.cv2_to_imgmsg(warped, encoding="bgr8")
        warped_msg.header = header
        self.warped_pub.publish(warped_msg)

        warped_mask_msg = self.bridge.cv2_to_imgmsg(warped_mask, encoding="mono8")
        warped_mask_msg.header = header
        self.warped_mask_pub.publish(warped_mask_msg)

        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = "robot_base"

        payload = {
            "count": len(results),
            "pose_order": "P1,P2,P3,P4",
            "paper_corner_source": self.paper_source,
            "paper_corner_file": self.paper_corner_file,
            "note": "command Z uses configured grasp_z/place_z; measured Z is only diagnostic",
            "pieces": [],
        }

        for result in sorted(results, key=lambda x: x.label):
            if result.pick_command_xyz is not None:
                pose = Pose()
                pose.position.x = result.pick_command_xyz[0]
                pose.position.y = result.pick_command_xyz[1]
                pose.position.z = result.pick_command_xyz[2]
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)

            item = {
                "label": "P{}".format(result.label),
                "area_mm2": round(result.candidate.area_mm2, 2),
                "pick_pixel_uv": [round(result.pick_pixel[0], 2), round(result.pick_pixel[1], 2)],
                "place_pixel_uv": (
                    [round(result.place_pixel[0], 2), round(result.place_pixel[1], 2)]
                    if result.place_pixel is not None
                    else None
                ),
                "pick_depth_mm": self.finite_or_none(result.depth_pick_mm),
                "place_depth_mm": self.finite_or_none(result.depth_place_mm),
                "pick_measured_robot_xyz": (
                    [self.finite_or_none(x) for x in result.measured_pick_robot_xyz]
                    if result.measured_pick_robot_xyz is not None
                    else None
                ),
                "place_measured_robot_xyz": (
                    [self.finite_or_none(x) for x in result.measured_place_robot_xyz]
                    if result.measured_place_robot_xyz is not None
                    else None
                ),
                "pick_command_xyz": (
                    [round(x, 3) for x in result.pick_command_xyz]
                    if result.pick_command_xyz is not None
                    else None
                ),
                "place_command_xyz": (
                    [round(x, 3) for x in result.place_command_xyz]
                    if result.place_command_xyz is not None
                    else None
                ),
                "current_yaw_deg_clockwise": self.finite_or_none(
                    result.current_yaw_deg_clockwise
                ),
                "required_rotation_deg_clockwise": self.finite_or_none(
                    result.required_rotation_deg_clockwise
                ),
                "pose_rmse_mm": self.finite_or_none(result.pose_rmse_mm),
                "assignment_cost": round(result.assignment_cost, 5),
            }
            payload["pieces"].append(item)

        self.pose_pub.publish(pose_array)
        self.coords_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        if len(results) == 4:
            summary = []
            for r in results:
                if r.pick_command_xyz and r.place_command_xyz:
                    summary.append(
                        "P{} pick({:.1f},{:.1f},{:.1f}) place({:.1f},{:.1f},{:.1f}) rot={}".format(
                            r.label,
                            *r.pick_command_xyz,
                            *r.place_command_xyz,
                            "{:.1f}deg".format(r.required_rotation_deg_clockwise)
                            if r.required_rotation_deg_clockwise is not None
                            else "NA",
                        )
                    )
            rospy.loginfo_throttle(2.0, "\n" + "\n".join(summary))


if __name__ == "__main__":
    try:
        PuzzlePieceDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
