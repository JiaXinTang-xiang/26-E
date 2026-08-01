#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现场任意 1~4 片矩形拼图识别与装配规划（ROS1 / SwiftPro）

依赖：
  puzzle_piece_detector_fixed_corners_v4.py
  两个文件放在同一个 ROS 包 scripts 目录中。

输入：
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw
  ~/.ros/puzzle_a4_corners.yaml
  ~/thefile.txt

输出：
  /puzzle/annotated_image
  /puzzle/warped_image
  /puzzle/warped_piece_mask
  /puzzle/piece_mask
  /puzzle/piece_coordinates
  /puzzle/piece_command_poses

算法：
1. 固定 A4 四角透视矫正；
2. 在黑纸上提取白底纸片，彩色/黑色扑克牌花纹作为片内孔洞填充；
3. 将每片轮廓拟合为 3~5 边多边形；
4. 枚举一片外边作为矩形上边，通过边到边的刚体匹配递归拼接其余片；
5. 按“无重叠、并集接近矩形、矩形尺寸范围、边界完整度”评分；
6. 对扑克牌牌面增加切缝两侧 Lab 颜色连续性评分，消除几何歧义；
7. 输出每片中心优先抓取点、目标位置和所需旋转角。

重要限制：
- 当前机械臂没有第四轴供电时，只能执行 required_rotation 接近 0° 的方案。
- 本程序负责识别和规划，不会自动驱动机械臂。
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import String
from sensor_msgs.msg import Image

# 允许直接从同目录加载现有固定四角识别程序。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from puzzle_piece_detector_fixed_corners_v4 import (  # noqa: E402
    CAMERA_MATRIX,
    PAPER_H_MM,
    PAPER_W_MM,
    PX_PER_MM_X,
    PX_PER_MM_Y,
    WARP_H,
    WARP_W,
    PuzzlePieceDetector,
)


@dataclass
class GenericPiece:
    piece_id: int
    contour_px: np.ndarray
    polygon_px: np.ndarray
    polygon_cart_mm: np.ndarray
    centroid_cart_mm: np.ndarray
    area_mm2: float
    grasp_warp_px: Tuple[float, float]
    grasp_local_cart_mm: np.ndarray
    bbox_px: Tuple[int, int, int, int]


@dataclass
class Placement:
    piece_index: int
    R: np.ndarray
    t: np.ndarray
    polygon_cart_mm: np.ndarray


@dataclass
class AssemblySolution:
    placements: Dict[int, Placement]
    width_mm: float
    height_mm: float
    fill_ratio: float
    perimeter_ratio: float
    texture_score: float
    geometry_score: float
    total_score: float


@dataclass
class GridPlacementCandidate:
    piece_index: int
    R: np.ndarray
    t: np.ndarray
    polygon_cart_mm: np.ndarray
    interior_bits: int
    full_bits: int
    border_bits: int
    full_cell_count: int


@dataclass
class GenericResult:
    label: int
    piece: GenericPiece
    pick_pixel: Tuple[float, float]
    place_pixel: Optional[Tuple[float, float]]
    pick_command_xyz: Optional[Tuple[float, float, float]]
    place_command_xyz: Optional[Tuple[float, float, float]]
    pick_measured_robot_xyz: Optional[Tuple[float, float, float]]
    place_measured_robot_xyz: Optional[Tuple[float, float, float]]
    pick_depth_mm: Optional[float]
    place_depth_mm: Optional[float]
    required_rotation_deg_clockwise: float
    target_polygon_warp_px: Optional[np.ndarray]


class OnsitePuzzleSolver(PuzzlePieceDetector):
    def __init__(self) -> None:
        # 基类会在 __init__ 中立即创建订阅者；用 ready 标志避免相机首帧抢在
        # 现场求解参数初始化完成前进入回调。
        self.onsite_ready = False
        super().__init__()

        # 现场题约束与求解参数。
        self.min_piece_count = int(rospy.get_param("~min_piece_count", 1))
        self.max_piece_count = int(rospy.get_param("~max_piece_count", 4))
        self.min_polygon_vertices = int(rospy.get_param("~min_polygon_vertices", 3))
        self.max_polygon_vertices = int(rospy.get_param("~max_polygon_vertices", 5))
        self.true_min_edge_mm = float(rospy.get_param("~true_min_edge_mm", 20.0))
        self.detected_min_edge_mm = float(rospy.get_param("~detected_min_edge_mm", 14.0))

        # 目标矩形长边 90~120 mm，短边 50~90 mm。
        self.rect_long_min_mm = float(rospy.get_param("~rect_long_min_mm", 90.0))
        self.rect_long_max_mm = float(rospy.get_param("~rect_long_max_mm", 120.0))
        self.rect_short_min_mm = float(rospy.get_param("~rect_short_min_mm", 50.0))
        self.rect_short_max_mm = float(rospy.get_param("~rect_short_max_mm", 90.0))
        self.dimension_slack_mm = float(rospy.get_param("~dimension_slack_mm", 7.0))

        self.min_piece_area_mm2 = float(rospy.get_param("~min_piece_area_mm2", 120.0))
        self.max_piece_area_mm2 = float(rospy.get_param("~max_piece_area_mm2", 11000.0))
        self.max_candidate_count = int(rospy.get_param("~max_candidate_count", 6))

        # 边匹配允许轮廓提取误差；支持一条长边与两条短边形成 T 形接缝。
        self.edge_angle_tol_deg = float(rospy.get_param("~edge_angle_tol_deg", 7.0))
        self.edge_line_tol_mm = float(rospy.get_param("~edge_line_tol_mm", 2.5))
        self.min_edge_overlap_mm = float(rospy.get_param("~min_edge_overlap_mm", 10.0))
        self.max_overlap_area_mm2 = float(rospy.get_param("~max_overlap_area_mm2", 10.0))
        self.search_node_limit = int(rospy.get_param("~search_node_limit", 7000))
        self.max_full_solutions = int(rospy.get_param("~max_full_solutions", 50))
        self.raster_scale = float(rospy.get_param("~solver_raster_px_per_mm", 2.0))
        # 主求解器采用“每片至少有一条矩形外边”的边界栅格装箱。
        self.packing_grid_mm = float(rospy.get_param("~packing_grid_mm", 1.0))
        self.packing_scale = float(rospy.get_param("~packing_raster_px_per_mm", 2.0))
        self.rect_area_tolerance_ratio = float(
            rospy.get_param("~rect_area_tolerance_ratio", 0.12)
        )
        self.max_rectangle_candidates = int(
            rospy.get_param("~max_rectangle_candidates", 70)
        )
        self.packing_min_contact_mm = float(
            rospy.get_param("~packing_min_contact_mm", 6.0)
        )
        self.packing_solution_limit = int(
            rospy.get_param("~packing_solution_limit", 24)
        )
        self.packing_node_limit = int(
            rospy.get_param("~packing_node_limit", 250000)
        )

        # 纹理评分。白片时自动弱化；扑克牌花纹明显时自动提高权重。
        self.texture_weight = float(rospy.get_param("~texture_weight", 0.045))
        self.texture_sample_offset_mm = float(
            rospy.get_param("~texture_sample_offset_mm", 0.8)
        )
        self.texture_variance_threshold = float(
            rospy.get_param("~texture_variance_threshold", 12.0)
        )

        # 放置时整体居中到碎片所在半区的另一半。
        self.placement_gap_mm = float(rospy.get_param("~placement_gap_mm", 1.5))
        self.grasp_edge_margin_mm = float(rospy.get_param("~grasp_edge_margin_mm", 3.0))

        # 现场分割调试参数。白底是唯一的全局前景种子；不能把所有高饱和度
        # 像素直接并入掩膜，否则灰纸上的彩色噪声会把多片碎片连接成一片。
        self.seed_min_area_mm2 = float(rospy.get_param("~seed_min_area_mm2", 8.0))
        self.segment_close_mm = float(rospy.get_param("~segment_close_mm", 1.2))
        self.segment_open_mm = float(rospy.get_param("~segment_open_mm", 0.5))
        self.last_white_seed = np.zeros((WARP_H, WARP_W), dtype=np.uint8)
        self.white_seed_pub = rospy.Publisher(
            "/puzzle/warped_white_seed", Image, queue_size=1
        )

        self.search_nodes = 0
        self.full_solutions: List[AssemblySolution] = []

        self.onsite_ready = True
        rospy.loginfo(
            "Onsite arbitrary puzzle solver enabled: pieces=%d..%d, rectangle long %.0f..%.0f mm, short %.0f..%.0f mm",
            self.min_piece_count,
            self.max_piece_count,
            self.rect_long_min_mm,
            self.rect_long_max_mm,
            self.rect_short_min_mm,
            self.rect_short_max_mm,
        )

    # ------------------------------------------------------------------
    # 分割：白底纸片 + 红黑扑克牌图案
    # ------------------------------------------------------------------
    def segment_white_pieces(self, warped: np.ndarray) -> np.ndarray:
        """提取白底碎片。

        关键原则：
        1. 只用高亮、低饱和度的白底作为前景种子；
        2. 扑克牌的红黑图案作为片内孔洞，后续由外轮廓填充；
        3. 不再把全图所有“彩色像素”直接加入掩膜，避免纸面色噪声形成桥接；
        4. 形态学核按毫米设置，并保持较小，避免相邻碎片被粘连。
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # 排除A4边缘后估计纸面亮度。使用中位数能适应当前灰黑纸面和缓慢光照变化。
        border_x = int(round(self.paper_border_ignore_mm * PX_PER_MM_X))
        border_y = int(round(self.paper_border_ignore_mm * PX_PER_MM_Y))
        y0 = min(max(border_y, 0), max(0, WARP_H - 1))
        y1 = max(y0 + 1, WARP_H - max(border_y, 0))
        x0 = min(max(border_x, 0), max(0, WARP_W - 1))
        x1 = max(x0 + 1, WARP_W - max(border_x, 0))
        roi_gray = gray[y0:y1, x0:x1]
        paper_median = float(np.median(roi_gray)) if roi_gray.size else float(np.median(gray))

        otsu_t, _ = cv2.threshold(
            roi_gray if roi_gray.size else gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        # 与纸面至少相差white_margin，同时不低于Otsu阈值。
        white_t = int(max(float(otsu_t), paper_median + float(self.white_margin)))
        white_t = max(35, min(white_t, 245))

        white_seed = (gray >= white_t) & (sat <= self.white_sat_max)
        seed = np.zeros_like(gray, dtype=np.uint8)
        seed[white_seed] = 255

        # 屏蔽纸边、分界线及可选半区。
        if border_y > 0:
            seed[:border_y, :] = 0
            seed[-border_y:, :] = 0
        if border_x > 0:
            seed[:, :border_x] = 0
            seed[:, -border_x:] = 0

        mid = WARP_H // 2
        line_band = int(round(max(0.0, self.divider_ignore_mm) * PX_PER_MM_Y))
        if line_band > 0:
            seed[max(0, mid - line_band):min(WARP_H, mid + line_band), :] = 0

        if self.source_region == "upper":
            seed[mid:, :] = 0
        elif self.source_region == "lower":
            seed[:mid, :] = 0
        elif self.source_region not in ("full", "auto"):
            rospy.logwarn_throttle(
                5.0, "Unknown source_region=%s; using full", self.source_region
            )

        # 先小幅开运算去除纸面亮点。
        open_px = max(1, int(round(self.segment_open_mm * 0.5 * (PX_PER_MM_X + PX_PER_MM_Y))))
        if open_px % 2 == 0:
            open_px += 1
        if open_px > 1:
            seed = cv2.morphologyEx(
                seed,
                cv2.MORPH_OPEN,
                np.ones((open_px, open_px), np.uint8),
                iterations=1,
            )

        # 删除面积很小的白色噪声，防止后续闭运算把噪声串成桥。
        min_seed_px = max(
            4,
            int(round(self.seed_min_area_mm2 * PX_PER_MM_X * PX_PER_MM_Y)),
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
        clean_seed = np.zeros_like(seed)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_seed_px:
                clean_seed[labels == label] = 255
        seed = clean_seed
        self.last_white_seed = seed.copy()

        # 仅用约1mm的小闭运算修补锯齿；原来的9x9、两次闭运算容易连接相邻碎片。
        close_px = max(1, int(round(self.segment_close_mm * 0.5 * (PX_PER_MM_X + PX_PER_MM_Y))))
        if close_px % 2 == 0:
            close_px += 1
        if close_px > 1:
            seed = cv2.morphologyEx(
                seed,
                cv2.MORPH_CLOSE,
                np.ones((close_px, close_px), np.uint8),
                iterations=1,
            )

        # 红黑牌面图案作为孔洞：只填充每个白色外轮廓内部，不将彩色噪声加入前景。
        contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(seed)
        min_outer_px = max(4.0, 0.25 * self.min_piece_area_mm2 * PX_PER_MM_X * PX_PER_MM_Y)
        max_outer_px = self.max_piece_area_mm2 * PX_PER_MM_X * PX_PER_MM_Y
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if min_outer_px <= area <= max_outer_px:
                cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

        rospy.loginfo_throttle(
            2.0,
            "onsite segmentation: paper_median=%.1f otsu=%.1f white_t=%d seed_components=%d",
            paper_median,
            float(otsu_t),
            white_t,
            max(0, count - 1),
        )
        return filled

    # ------------------------------------------------------------------
    # 多边形提取
    # ------------------------------------------------------------------
    @staticmethod
    def signed_area(points: np.ndarray) -> float:
        p = np.asarray(points, dtype=np.float64)
        return 0.5 * float(
            np.dot(p[:, 0], np.roll(p[:, 1], -1))
            - np.dot(p[:, 1], np.roll(p[:, 0], -1))
        )

    @classmethod
    def ensure_ccw(cls, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).copy()
        if cls.signed_area(p) < 0:
            p = p[::-1]
        return p

    @staticmethod
    def polygon_centroid(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        a = OnsitePuzzleSolver.signed_area(p)
        if abs(a) < 1e-9:
            return p.mean(axis=0)
        cross = p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]
        cx = np.sum((p[:, 0] + np.roll(p[:, 0], -1)) * cross) / (6.0 * a)
        cy = np.sum((p[:, 1] + np.roll(p[:, 1], -1)) * cross) / (6.0 * a)
        return np.array([cx, cy], dtype=np.float64)

    @staticmethod
    def edge_lengths(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)

    @staticmethod
    def remove_nearly_collinear(points: np.ndarray, angle_deg: float = 176.0) -> np.ndarray:
        p = list(np.asarray(points, dtype=np.float64))
        changed = True
        while changed and len(p) > 3:
            changed = False
            for i in range(len(p)):
                a = p[(i - 1) % len(p)] - p[i]
                b = p[(i + 1) % len(p)] - p[i]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na < 1e-9 or nb < 1e-9:
                    del p[i]
                    changed = True
                    break
                cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                angle = math.degrees(math.acos(cosv))
                if angle >= angle_deg:
                    del p[i]
                    changed = True
                    break
        return np.asarray(p, dtype=np.float64)

    def approximate_piece_polygon(self, contour: np.ndarray) -> Optional[np.ndarray]:
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = abs(float(cv2.contourArea(contour)))
        if perimeter <= 1e-6 or contour_area <= 1e-6:
            return None

        best = None
        best_score = float("inf")
        for eps_ratio in np.linspace(0.0025, 0.045, 120):
            approx = cv2.approxPolyDP(contour, eps_ratio * perimeter, True)
            n = len(approx)
            if not (self.min_polygon_vertices <= n <= self.max_polygon_vertices):
                continue
            pts_px = approx.reshape(-1, 2).astype(np.float64)
            pts_mm_img = pts_px / np.array([PX_PER_MM_X, PX_PER_MM_Y])
            # 转换到 y 向上的笛卡尔坐标。
            pts_cart = np.column_stack([pts_mm_img[:, 0], -pts_mm_img[:, 1]])
            pts_cart = self.remove_nearly_collinear(self.ensure_ccw(pts_cart))
            if not (self.min_polygon_vertices <= len(pts_cart) <= self.max_polygon_vertices):
                continue

            lengths = self.edge_lengths(pts_cart)
            min_edge = float(np.min(lengths))
            poly_px = np.column_stack(
                [pts_cart[:, 0] * PX_PER_MM_X, -pts_cart[:, 1] * PX_PER_MM_Y]
            ).astype(np.float32)
            poly_area = abs(float(cv2.contourArea(poly_px.reshape(-1, 1, 2))))
            area_error = abs(poly_area - contour_area) / max(contour_area, 1.0)
            edge_penalty = max(0.0, self.detected_min_edge_mm - min_edge) / max(
                self.detected_min_edge_mm, 1.0
            )
            # 面积贴合优先，顶点少量偏好，避免噪声产生多余顶点。
            score = 5.0 * area_error + 1.6 * edge_penalty + 0.025 * len(pts_cart)
            if score < best_score:
                best_score = score
                best = pts_cart

        if best is None:
            return None
        return self.ensure_ccw(best)

    def extract_generic_pieces(self, piece_mask: np.ndarray) -> List[GenericPiece]:
        contours, _ = cv2.findContours(
            piece_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        raw: List[Tuple[float, np.ndarray, np.ndarray, Tuple[int, int, int, int], Tuple[float, float]]] = []
        px_area_per_mm2 = PX_PER_MM_X * PX_PER_MM_Y

        for contour in contours:
            area_px = abs(float(cv2.contourArea(contour)))
            area_mm2 = area_px / px_area_per_mm2
            if not (self.min_piece_area_mm2 <= area_mm2 <= self.max_piece_area_mm2):
                continue

            polygon_cart = self.approximate_piece_polygon(contour)
            if polygon_cart is None:
                continue
            if len(polygon_cart) > self.max_polygon_vertices:
                continue

            lengths = self.edge_lengths(polygon_cart)
            if float(np.min(lengths)) < self.detected_min_edge_mm:
                continue

            grasp = self.grasp_point_from_contour(contour, piece_mask.shape)
            raw.append((area_mm2, contour, polygon_cart, cv2.boundingRect(contour), grasp))

        # 优先保留大且规则的候选；现场最多四片。
        raw.sort(key=lambda x: x[0], reverse=True)
        raw = raw[: self.max_candidate_count]

        # 标签固定为从上到下、从左到右，便于执行节点稳定跟踪。
        raw.sort(key=lambda item: (item[4][1], item[4][0]))
        pieces: List[GenericPiece] = []
        for idx, (area_mm2, contour, polygon_cart, bbox, grasp) in enumerate(raw, start=1):
            centroid = self.polygon_centroid(polygon_cart)
            grasp_mm_img = np.array(
                [grasp[0] / PX_PER_MM_X, grasp[1] / PX_PER_MM_Y], dtype=np.float64
            )
            grasp_cart = np.array([grasp_mm_img[0], -grasp_mm_img[1]], dtype=np.float64)
            pieces.append(
                GenericPiece(
                    piece_id=idx,
                    contour_px=contour,
                    polygon_px=np.column_stack(
                        [polygon_cart[:, 0] * PX_PER_MM_X, -polygon_cart[:, 1] * PX_PER_MM_Y]
                    ).astype(np.float32).reshape(-1, 1, 2),
                    polygon_cart_mm=polygon_cart - centroid,
                    centroid_cart_mm=centroid,
                    area_mm2=area_mm2,
                    grasp_warp_px=grasp,
                    grasp_local_cart_mm=grasp_cart - centroid,
                    bbox_px=bbox,
                )
            )
        return pieces

    # ------------------------------------------------------------------
    # 几何拼接搜索
    # ------------------------------------------------------------------
    @staticmethod
    def rotation_matrix(angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array([[c, -s], [s, c]], dtype=np.float64)

    @staticmethod
    def transform_polygon(piece: GenericPiece, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        return (R @ piece.polygon_cart_mm.T).T + t

    @staticmethod
    def polygon_edges(points: np.ndarray) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
        for i in range(len(points)):
            yield i, points[i], points[(i + 1) % len(points)]

    @staticmethod
    def rigid_align_direction(
        src_a: np.ndarray,
        src_b: np.ndarray,
        target_direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        src_vec = src_b - src_a
        ns = float(np.linalg.norm(src_vec))
        nt = float(np.linalg.norm(target_direction))
        if ns < 1e-9 or nt < 1e-9:
            return None
        angle = math.atan2(target_direction[1], target_direction[0]) - math.atan2(
            src_vec[1], src_vec[0]
        )
        return OnsitePuzzleSolver.rotation_matrix(angle)

    @staticmethod
    def segment_overlap_on_line(
        a0: np.ndarray,
        a1: np.ndarray,
        b0: np.ndarray,
        b1: np.ndarray,
        angle_tol_deg: float,
        line_tol_mm: float,
    ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        va = a1 - a0
        vb = b1 - b0
        la = float(np.linalg.norm(va))
        lb = float(np.linalg.norm(vb))
        if la < 1e-9 or lb < 1e-9:
            return 0.0, None, None
        ua = va / la
        ub = vb / lb
        # 内部接缝的边方向应相反。
        if float(np.dot(ua, ub)) > -math.cos(math.radians(angle_tol_deg)):
            return 0.0, None, None
        normal = np.array([-ua[1], ua[0]], dtype=np.float64)
        if max(abs(float(np.dot(b0 - a0, normal))), abs(float(np.dot(b1 - a0, normal)))) > line_tol_mm:
            return 0.0, None, None

        aa0, aa1 = 0.0, la
        bb0 = float(np.dot(b0 - a0, ua))
        bb1 = float(np.dot(b1 - a0, ua))
        low = max(min(aa0, aa1), min(bb0, bb1))
        high = min(max(aa0, aa1), max(bb0, bb1))
        overlap = max(0.0, high - low)
        if overlap <= 0:
            return 0.0, None, None
        return overlap, a0 + low * ua, a0 + high * ua

    def pair_overlap_area_mm2(self, poly_a: np.ndarray, poly_b: np.ndarray) -> float:
        # 先做包围盒快速判断。
        min_a, max_a = poly_a.min(axis=0), poly_a.max(axis=0)
        min_b, max_b = poly_b.min(axis=0), poly_b.max(axis=0)
        if np.any(max_a <= min_b) or np.any(max_b <= min_a):
            return 0.0

        # 大多数现场直线切割碎片为凸多边形；凸多边形用 OpenCV 精确求交，速度远快于栅格。
        pa = poly_a.astype(np.float32).reshape(-1, 1, 2)
        pb = poly_b.astype(np.float32).reshape(-1, 1, 2)
        if cv2.isContourConvex(pa.astype(np.int32)) and cv2.isContourConvex(pb.astype(np.int32)):
            try:
                area, _ = cv2.intersectConvexConvex(pa, pb)
                return float(area)
            except cv2.error:
                pass

        # 凹多边形退回低分辨率栅格求交。
        all_points = np.vstack([poly_a, poly_b])
        min_xy = np.floor(all_points.min(axis=0) - 1.0)
        max_xy = np.ceil(all_points.max(axis=0) + 1.0)
        scale = max(1.0, min(self.raster_scale, 2.0))
        size = np.maximum(8, np.ceil((max_xy - min_xy) * scale).astype(int) + 3)
        if size[0] > 700 or size[1] > 700:
            return float("inf")

        def to_px(poly: np.ndarray) -> np.ndarray:
            q = (poly - min_xy) * scale
            q[:, 1] = size[1] - 1 - q[:, 1]
            return np.round(q).astype(np.int32).reshape(-1, 1, 2)

        mask_a = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
        mask_b = np.zeros_like(mask_a)
        cv2.fillPoly(mask_a, [to_px(poly_a)], 255)
        cv2.fillPoly(mask_b, [to_px(poly_b)], 255)
        overlap_px = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
        return overlap_px / (scale * scale)

    def placement_is_valid(
        self,
        new_poly: np.ndarray,
        placements: Dict[int, Placement],
    ) -> bool:
        for placement in placements.values():
            overlap = self.pair_overlap_area_mm2(new_poly, placement.polygon_cart_mm)
            if overlap > self.max_overlap_area_mm2:
                return False

        all_points = [new_poly] + [p.polygon_cart_mm for p in placements.values()]
        stacked = np.vstack(all_points)
        extent = stacked.max(axis=0) - stacked.min(axis=0)
        if max(extent) > self.rect_long_max_mm + 2.0 * self.dimension_slack_mm + 20.0:
            return False
        if min(extent) > self.rect_short_max_mm + 2.0 * self.dimension_slack_mm + 20.0:
            return False
        return True

    def candidate_placements_against_edge(
        self,
        piece: GenericPiece,
        target_a: np.ndarray,
        target_b: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        target_vec = target_b - target_a
        results: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        seen = set()

        for _, src_a, src_b in self.polygon_edges(piece.polygon_cart_mm):
            # 新片边与已放片边方向相反。
            R = self.rigid_align_direction(src_a, src_b, -target_vec)
            if R is None:
                continue
            ra = R @ src_a
            rb = R @ src_b
            mid_target = 0.5 * (target_a + target_b)
            mid_src = 0.5 * (ra + rb)
            translations = [
                target_a - ra,
                target_b - ra,
                target_a - rb,
                target_b - rb,
                mid_target - mid_src,
            ]
            for t in translations:
                poly = self.transform_polygon(piece, R, t)
                ea = R @ src_a + t
                eb = R @ src_b + t
                overlap, _, _ = self.segment_overlap_on_line(
                    target_a,
                    target_b,
                    ea,
                    eb,
                    self.edge_angle_tol_deg,
                    self.edge_line_tol_mm,
                )
                if overlap < self.min_edge_overlap_mm:
                    continue
                angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                key = (
                    int(round(angle * 2.0)),
                    int(round(t[0] * 2.0)),
                    int(round(t[1] * 2.0)),
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append((R.copy(), t.copy(), poly, overlap))

        results.sort(key=lambda item: item[3], reverse=True)
        return results

    def all_contact_edges(
        self, placements: Dict[int, Placement]
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        edges: List[Tuple[np.ndarray, np.ndarray]] = []
        for placement in placements.values():
            for _, a, b in self.polygon_edges(placement.polygon_cart_mm):
                edges.append((a, b))
        return edges

    def recursive_search(
        self,
        pieces: List[GenericPiece],
        placements: Dict[int, Placement],
        remaining: Sequence[int],
        warped_lab: np.ndarray,
    ) -> None:
        if self.search_nodes >= self.search_node_limit:
            return
        if len(self.full_solutions) >= self.max_full_solutions:
            return
        self.search_nodes += 1

        if not remaining:
            solution = self.evaluate_solution(pieces, placements, warped_lab)
            if solution is not None:
                self.full_solutions.append(solution)
            return

        contacts = self.all_contact_edges(placements)
        # 尝试面积较大的片先放，分支更少。
        ordered_remaining = sorted(remaining, key=lambda idx: pieces[idx].area_mm2, reverse=True)
        for piece_index in ordered_remaining:
            piece = pieces[piece_index]
            generated: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
            for target_a, target_b in contacts:
                generated.extend(
                    self.candidate_placements_against_edge(piece, target_a, target_b)
                )
            # 去重并保留优质分支。边重合越长越优先。
            generated.sort(key=lambda item: item[3], reverse=True)
            generated = generated[:60]
            branch_seen = set()
            branch_count = 0
            for R, t, poly, overlap in generated:
                angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                sig = (
                    int(round(angle)),
                    int(round(t[0])),
                    int(round(t[1])),
                )
                if sig in branch_seen:
                    continue
                branch_seen.add(sig)
                if not self.placement_is_valid(poly, placements):
                    continue
                new_placements = dict(placements)
                new_placements[piece_index] = Placement(piece_index, R, t, poly)
                new_remaining = [idx for idx in remaining if idx != piece_index]
                self.recursive_search(pieces, new_placements, new_remaining, warped_lab)
                branch_count += 1
                if branch_count >= 18:
                    break

    def dimension_penalty(self, long_side: float, short_side: float) -> float:
        def interval_distance(v: float, low: float, high: float) -> float:
            if v < low:
                return low - v
            if v > high:
                return v - high
            return 0.0

        return (
            interval_distance(long_side, self.rect_long_min_mm, self.rect_long_max_mm)
            + interval_distance(short_side, self.rect_short_min_mm, self.rect_short_max_mm)
        )

    def raster_union_metrics(
        self, polygons: Sequence[np.ndarray]
    ) -> Tuple[float, float, float, float, float]:
        stacked = np.vstack(polygons)
        min_xy = stacked.min(axis=0)
        max_xy = stacked.max(axis=0)
        width = float(max_xy[0] - min_xy[0])
        height = float(max_xy[1] - min_xy[1])
        size = np.maximum(8, np.ceil((max_xy - min_xy + 4.0) * self.raster_scale).astype(int) + 3)
        if size[0] > 1200 or size[1] > 1200:
            return width, height, 0.0, 10.0, float("inf")

        mask = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
        offset = min_xy - 2.0
        for poly in polygons:
            q = (poly - offset) * self.raster_scale
            q[:, 1] = size[1] - 1 - q[:, 1]
            cv2.fillPoly(mask, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 255)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return width, height, 0.0, 10.0, float("inf")
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bbox_area_px = max(1, (x2 - x1 + 1) * (y2 - y1 + 1))
        union_area_px = int(np.count_nonzero(mask[y1:y2 + 1, x1:x2 + 1]))
        fill_ratio = union_area_px / float(bbox_area_px)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return width, height, fill_ratio, 10.0, float("inf")
        largest = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(largest, True)) / self.raster_scale
        expected_perimeter = max(1e-6, 2.0 * (width + height))
        perimeter_ratio = perimeter / expected_perimeter
        gap_area_mm2 = (bbox_area_px - union_area_px) / (self.raster_scale ** 2)
        return width, height, fill_ratio, perimeter_ratio, gap_area_mm2

    @staticmethod
    def bilinear_sample(image: np.ndarray, u: float, v: float) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        if u < 0 or v < 0 or u >= w - 1 or v >= h - 1:
            return None
        x0 = int(math.floor(u))
        y0 = int(math.floor(v))
        dx = u - x0
        dy = v - y0
        p00 = image[y0, x0].astype(np.float64)
        p10 = image[y0, x0 + 1].astype(np.float64)
        p01 = image[y0 + 1, x0].astype(np.float64)
        p11 = image[y0 + 1, x0 + 1].astype(np.float64)
        return (
            (1 - dx) * (1 - dy) * p00
            + dx * (1 - dy) * p10
            + (1 - dx) * dy * p01
            + dx * dy * p11
        )

    def sample_piece_lab(
        self,
        piece: GenericPiece,
        placement: Placement,
        assembly_point: np.ndarray,
        warped_lab: np.ndarray,
    ) -> Optional[np.ndarray]:
        local = placement.R.T @ (assembly_point - placement.t)
        source_cart = local + piece.centroid_cart_mm
        u = float(source_cart[0] * PX_PER_MM_X)
        v = float(-source_cart[1] * PX_PER_MM_Y)
        return self.bilinear_sample(warped_lab, u, v)

    def texture_continuity_score(
        self,
        pieces: List[GenericPiece],
        placements: Dict[int, Placement],
        warped_lab: np.ndarray,
    ) -> float:
        scores: List[float] = []
        keys = sorted(placements.keys())
        for ai in range(len(keys)):
            pa = placements[keys[ai]]
            piece_a = pieces[keys[ai]]
            for bi in range(ai + 1, len(keys)):
                pb = placements[keys[bi]]
                piece_b = pieces[keys[bi]]
                for _, a0, a1 in self.polygon_edges(pa.polygon_cart_mm):
                    va = a1 - a0
                    la = float(np.linalg.norm(va))
                    if la < 1e-9:
                        continue
                    ua = va / la
                    normal_a = np.array([-ua[1], ua[0]], dtype=np.float64)
                    for _, b0, b1 in self.polygon_edges(pb.polygon_cart_mm):
                        overlap, q0, q1 = self.segment_overlap_on_line(
                            a0,
                            a1,
                            b0,
                            b1,
                            self.edge_angle_tol_deg,
                            self.edge_line_tol_mm,
                        )
                        if overlap < self.min_edge_overlap_mm or q0 is None or q1 is None:
                            continue
                        vb = b1 - b0
                        lb = float(np.linalg.norm(vb))
                        if lb < 1e-9:
                            continue
                        ub = vb / lb
                        normal_b = np.array([-ub[1], ub[0]], dtype=np.float64)
                        sample_count = max(3, min(18, int(overlap / 3.0)))
                        for alpha in np.linspace(0.08, 0.92, sample_count):
                            q = (1.0 - alpha) * q0 + alpha * q1
                            ca = self.sample_piece_lab(
                                piece_a,
                                pa,
                                q + normal_a * self.texture_sample_offset_mm,
                                warped_lab,
                            )
                            cb = self.sample_piece_lab(
                                piece_b,
                                pb,
                                q + normal_b * self.texture_sample_offset_mm,
                                warped_lab,
                            )
                            if ca is None or cb is None:
                                continue
                            # L 通道权重略低，a/b 色彩差异对红黑牌面更有判别力。
                            diff = ca - cb
                            score = math.sqrt(
                                0.55 * diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
                            )
                            scores.append(float(score))
        if not scores:
            return 80.0
        scores = sorted(scores)
        # 去掉少量轮廓采样异常。
        keep = scores[: max(1, int(0.9 * len(scores)))]
        return float(np.mean(keep))

    def overall_texture_variance(
        self, pieces: List[GenericPiece], warped: np.ndarray
    ) -> float:
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        values: List[float] = []
        for piece in pieces:
            mask = np.zeros(warped.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [piece.contour_px], -1, 255, cv2.FILLED)
            pixels = lab[mask > 0]
            if len(pixels) < 20:
                continue
            values.append(float(np.mean(np.std(pixels.astype(np.float64), axis=0))))
        return float(np.mean(values)) if values else 0.0

    def normalize_solution(
        self, placements: Dict[int, Placement]
    ) -> Dict[int, Placement]:
        all_points = np.vstack([p.polygon_cart_mm for p in placements.values()])
        extent = all_points.max(axis=0) - all_points.min(axis=0)
        global_R = np.eye(2, dtype=np.float64)
        if extent[1] > extent[0]:
            # 将长边转为水平。
            global_R = self.rotation_matrix(-math.pi / 2.0)

        transformed: Dict[int, Placement] = {}
        for idx, placement in placements.items():
            R = global_R @ placement.R
            t = global_R @ placement.t
            poly = (global_R @ placement.polygon_cart_mm.T).T
            transformed[idx] = Placement(idx, R, t, poly)

        all_points = np.vstack([p.polygon_cart_mm for p in transformed.values()])
        min_xy = all_points.min(axis=0)
        shift = -min_xy
        for idx, placement in list(transformed.items()):
            transformed[idx] = Placement(
                idx,
                placement.R,
                placement.t + shift,
                placement.polygon_cart_mm + shift,
            )
        return transformed

    def solution_signature(self, placements: Dict[int, Placement]) -> Tuple[int, ...]:
        vals: List[int] = []
        for idx in sorted(placements):
            p = placements[idx]
            center = p.polygon_cart_mm.mean(axis=0)
            angle = math.degrees(math.atan2(p.R[1, 0], p.R[0, 0]))
            vals.extend(
                [int(round(center[0])), int(round(center[1])), int(round(angle / 2.0))]
            )
        return tuple(vals)

    def evaluate_solution(
        self,
        pieces: List[GenericPiece],
        placements: Dict[int, Placement],
        warped_lab: np.ndarray,
    ) -> Optional[AssemblySolution]:
        normalized = self.normalize_solution(placements)
        polygons = [normalized[i].polygon_cart_mm for i in sorted(normalized)]
        width, height, fill_ratio, perimeter_ratio, gap_area = self.raster_union_metrics(polygons)
        long_side = max(width, height)
        short_side = min(width, height)
        dim_penalty = self.dimension_penalty(long_side, short_side)
        if dim_penalty > self.dimension_slack_mm * 2.0:
            return None
        if fill_ratio < 0.78:
            return None
        if perimeter_ratio > 1.45:
            return None

        texture_score = 0.0
        geometry_score = (
            22.0 * (1.0 - fill_ratio)
            + 4.0 * abs(perimeter_ratio - 1.0)
            + 0.12 * gap_area
            + 0.30 * dim_penalty
        )
        total_score = geometry_score
        return AssemblySolution(
            placements=normalized,
            width_mm=long_side,
            height_mm=short_side,
            fill_ratio=fill_ratio,
            perimeter_ratio=perimeter_ratio,
            texture_score=texture_score,
            geometry_score=geometry_score,
            total_score=total_score,
        )

    @staticmethod
    def mask_to_bits(mask: np.ndarray) -> int:
        packed = np.packbits((mask > 0).reshape(-1), bitorder="little")
        return int.from_bytes(packed.tobytes(), "little")

    def unique_axis_orientations(
        self, piece: GenericPiece
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """枚举使某一条边与矩形水平/竖直边平行的旋转。"""
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        angles: List[float] = []
        for _, a, b in self.polygon_edges(piece.polygon_cart_mm):
            vec = b - a
            if np.linalg.norm(vec) < 1e-9:
                continue
            edge_angle = math.atan2(vec[1], vec[0])
            for quarter in range(4):
                angle = -edge_angle + quarter * math.pi / 2.0
                normalized = (math.degrees(angle) + 360.0) % 360.0
                if any(
                    abs((normalized - existing + 180.0) % 360.0 - 180.0) < 0.35
                    for existing in angles
                ):
                    continue
                R = self.rotation_matrix(angle)
                q = (R @ piece.polygon_cart_mm.T).T
                angles.append(normalized)
                out.append((R, q))
        return out

    def rasterize_grid_placement(
        self,
        polygon: np.ndarray,
        width_mm: float,
        height_mm: float,
    ) -> Tuple[int, int, int, int]:
        scale = self.packing_scale
        width_px = max(2, int(round(width_mm * scale)))
        height_px = max(2, int(round(height_mm * scale)))
        mask = np.zeros((height_px, width_px), dtype=np.uint8)
        q = polygon.copy() * scale
        q[:, 1] = height_px - q[:, 1]
        q[:, 0] = np.clip(q[:, 0], 0, width_px - 1)
        q[:, 1] = np.clip(q[:, 1], 0, height_px - 1)
        cv2.fillPoly(
            mask,
            [np.round(q).astype(np.int32).reshape(-1, 1, 2)],
            255,
        )
        # 共享边在两个 fillPoly 中会同时占一列像素；侵蚀后的内部掩膜用于判重叠。
        interior = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        border = np.zeros_like(mask)
        border[(dilated > 0) & (mask == 0)] = 255
        return (
            self.mask_to_bits(interior),
            self.mask_to_bits(mask),
            self.mask_to_bits(border),
            int(np.count_nonzero(mask)),
        )

    def generate_boundary_placements(
        self,
        piece_index: int,
        piece: GenericPiece,
        width_mm: float,
        height_mm: float,
    ) -> List[GridPlacementCandidate]:
        """只生成至少一条真实边落在目标矩形外边上的放置。"""
        step = max(0.5, self.packing_grid_mm)
        placements: List[GridPlacementCandidate] = []
        seen_geometry = set()
        seen_masks = set()
        axis_tol = math.sin(math.radians(self.edge_angle_tol_deg))

        for R, oriented in self.unique_axis_orientations(piece):
            for _, a, b in self.polygon_edges(oriented):
                vec = b - a
                length = float(np.linalg.norm(vec))
                if length < self.detected_min_edge_mm:
                    continue

                # 水平边可作为上边或下边。
                if abs(vec[1]) <= axis_tol * max(length, 1e-9):
                    for boundary_y in (0.0, height_mm):
                        ty = boundary_y - a[1]
                        shifted = oriented + np.array([0.0, ty])
                        min_x = float(np.min(shifted[:, 0]))
                        max_x = float(np.max(shifted[:, 0]))
                        start = math.ceil((-min_x - 1e-6) / step) * step
                        stop = math.floor((width_mm - max_x + 1e-6) / step) * step
                        tx = start
                        while tx <= stop + 1e-9:
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -0.7
                                and np.max(poly[:, 0]) <= width_mm + 0.7
                                and np.min(poly[:, 1]) >= -0.7
                                and np.max(poly[:, 1]) <= height_mm + 0.7
                            ):
                                angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                                key = (
                                    int(round(angle * 2.0)),
                                    int(round(t[0] / step)),
                                    int(round(t[1] / step)),
                                )
                                if key not in seen_geometry:
                                    seen_geometry.add(key)
                                    interior, full, border, cells = self.rasterize_grid_placement(
                                        poly, width_mm, height_mm
                                    )
                                    if full not in seen_masks:
                                        seen_masks.add(full)
                                        placements.append(
                                            GridPlacementCandidate(
                                                piece_index, R.copy(), t.copy(), poly,
                                                interior, full, border, cells
                                            )
                                        )
                            tx += step

                # 竖直边可作为左边或右边。
                if abs(vec[0]) <= axis_tol * max(length, 1e-9):
                    for boundary_x in (0.0, width_mm):
                        tx = boundary_x - a[0]
                        shifted = oriented + np.array([tx, 0.0])
                        min_y = float(np.min(shifted[:, 1]))
                        max_y = float(np.max(shifted[:, 1]))
                        start = math.ceil((-min_y - 1e-6) / step) * step
                        stop = math.floor((height_mm - max_y + 1e-6) / step) * step
                        ty = start
                        while ty <= stop + 1e-9:
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -0.7
                                and np.max(poly[:, 0]) <= width_mm + 0.7
                                and np.min(poly[:, 1]) >= -0.7
                                and np.max(poly[:, 1]) <= height_mm + 0.7
                            ):
                                angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                                key = (
                                    int(round(angle * 2.0)),
                                    int(round(t[0] / step)),
                                    int(round(t[1] / step)),
                                )
                                if key not in seen_geometry:
                                    seen_geometry.add(key)
                                    interior, full, border, cells = self.rasterize_grid_placement(
                                        poly, width_mm, height_mm
                                    )
                                    if full not in seen_masks:
                                        seen_masks.add(full)
                                        placements.append(
                                            GridPlacementCandidate(
                                                piece_index, R.copy(), t.copy(), poly,
                                                interior, full, border, cells
                                            )
                                        )
                            ty += step
        return placements

    def rectangle_dimension_candidates(
        self, pieces: List[GenericPiece]
    ) -> List[Tuple[float, float, float]]:
        total_area = float(sum(piece.area_mm2 for piece in pieces))
        candidates: List[Tuple[float, float, float]] = []
        long_low = int(math.floor(self.rect_long_min_mm - self.dimension_slack_mm))
        long_high = int(math.ceil(self.rect_long_max_mm + self.dimension_slack_mm))
        short_low = int(math.floor(self.rect_short_min_mm - self.dimension_slack_mm))
        short_high = int(math.ceil(self.rect_short_max_mm + self.dimension_slack_mm))
        for width in range(max(1, long_low), long_high + 1):
            for height in range(max(1, short_low), min(short_high, width) + 1):
                rect_area = float(width * height)
                area_error = abs(rect_area - total_area) / max(total_area, 1.0)
                if area_error > self.rect_area_tolerance_ratio:
                    continue
                official_penalty = self.dimension_penalty(float(width), float(height))
                score = 4.0 * area_error + 0.02 * official_penalty
                candidates.append((score, float(width), float(height)))
        candidates.sort(key=lambda item: item[0])
        return candidates[: self.max_rectangle_candidates]

    def solve_one_rectangle_grid(
        self,
        pieces: List[GenericPiece],
        width_mm: float,
        height_mm: float,
    ) -> Tuple[List[Tuple[float, Dict[int, Placement]]], int]:
        placement_lists: Dict[int, List[GridPlacementCandidate]] = {}
        for index, piece in enumerate(pieces):
            options = self.generate_boundary_placements(
                index, piece, width_mm, height_mm
            )
            if not options:
                return [], 0
            placement_lists[index] = options

        # 候选最少的片先放；通常是大块或窄边块。
        order = sorted(range(len(pieces)), key=lambda idx: len(placement_lists[idx]))
        total_cells = max(1, int(round(width_mm * self.packing_scale)) * int(round(height_mm * self.packing_scale)))
        min_contact_cells = max(3, int(round(self.packing_min_contact_mm * self.packing_scale)))
        overlap_allow_cells = max(2, int(round(0.8 * self.packing_scale ** 2)))
        nodes = 0
        found: List[Tuple[float, Dict[int, Placement]]] = []

        def recurse(
            level: int,
            occupied_interior: int,
            occupied_full: int,
            chosen: Dict[int, GridPlacementCandidate],
        ) -> None:
            nonlocal nodes
            if nodes >= self.packing_node_limit:
                return
            if len(found) >= self.packing_solution_limit:
                return
            nodes += 1

            if level == len(order):
                fill_ratio = bin(occupied_full).count("1") / float(total_cells)
                if fill_ratio < 0.82:
                    return
                placements = {
                    idx: Placement(
                        idx, option.R.copy(), option.t.copy(), option.polygon_cart_mm.copy()
                    )
                    for idx, option in chosen.items()
                }
                found.append((fill_ratio, placements))
                return

            index = order[level]
            options = placement_lists[index]
            # 大面积候选和靠近已占区域的候选优先。
            if level > 0:
                options = sorted(
                    options,
                    key=lambda option: bin(option.border_bits & occupied_full).count("1"),
                    reverse=True,
                )
            for option in options:
                if bin(option.interior_bits & occupied_interior).count("1") > overlap_allow_cells:
                    continue
                if level > 0:
                    contact = bin(option.border_bits & occupied_full).count("1")
                    if contact < min_contact_cells:
                        continue
                chosen[index] = option
                recurse(
                    level + 1,
                    occupied_interior | option.interior_bits,
                    occupied_full | option.full_bits,
                    chosen,
                )
                chosen.pop(index, None)
                if len(found) >= self.packing_solution_limit:
                    return

        recurse(0, 0, 0, {})
        found.sort(key=lambda item: item[0], reverse=True)
        return found, nodes

    def solve_assembly(
        self, pieces: List[GenericPiece], warped: np.ndarray
    ) -> Optional[AssemblySolution]:
        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            return None

        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        texture_variance = self.overall_texture_variance(pieces, warped)
        effective_texture_weight = self.texture_weight
        if texture_variance < self.texture_variance_threshold:
            effective_texture_weight *= 0.15

        dimension_candidates = self.rectangle_dimension_candidates(pieces)
        if not dimension_candidates:
            rospy.logwarn_throttle(2.0, "No rectangle dimensions match total piece area")
            return None

        geometry_solutions: List[AssemblySolution] = []
        total_nodes = 0
        total_piece_area = float(sum(piece.area_mm2 for piece in pieces))
        for _, width_mm, height_mm in dimension_candidates:
            packed, nodes = self.solve_one_rectangle_grid(
                pieces, width_mm, height_mm
            )
            total_nodes += nodes
            for fill_ratio_grid, placements in packed:
                polygons = [placements[i].polygon_cart_mm for i in sorted(placements)]
                width, height, fill_ratio, perimeter_ratio, gap_area = self.raster_union_metrics(
                    polygons
                )
                long_side = max(width, height)
                short_side = min(width, height)
                dim_penalty = self.dimension_penalty(long_side, short_side)
                area_error_ratio = abs(width_mm * height_mm - total_piece_area) / max(
                    total_piece_area, 1.0
                )
                geometry_score = (
                    24.0 * (1.0 - fill_ratio_grid)
                    + 12.0 * (1.0 - fill_ratio)
                    + 3.0 * abs(perimeter_ratio - 1.0)
                    + 0.08 * gap_area
                    + 0.25 * dim_penalty
                    + 18.0 * area_error_ratio
                )
                geometry_solutions.append(
                    AssemblySolution(
                        placements=placements,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        fill_ratio=min(fill_ratio_grid, fill_ratio),
                        perimeter_ratio=perimeter_ratio,
                        texture_score=0.0,
                        geometry_score=geometry_score,
                        total_score=geometry_score,
                    )
                )
            # 已得到高填充方案时，不必继续遍历大量较差尺寸。
            if any(solution.fill_ratio >= 0.965 for solution in geometry_solutions):
                if len(geometry_solutions) >= 8:
                    break

        if not geometry_solutions:
            rospy.logwarn_throttle(
                2.0,
                "No boundary-packing solution: rectangles=%d nodes=%d pieces=%d",
                len(dimension_candidates),
                total_nodes,
                len(pieces),
            )
            return None

        # 去重后只对几何最好的方案计算牌面连续性。
        unique: Dict[Tuple[int, ...], AssemblySolution] = {}
        for solution in geometry_solutions:
            sig = self.solution_signature(solution.placements)
            if sig not in unique or solution.geometry_score < unique[sig].geometry_score:
                unique[sig] = solution
        finalists = sorted(unique.values(), key=lambda item: item.geometry_score)[:20]
        for solution in finalists:
            solution.texture_score = self.texture_continuity_score(
                pieces, solution.placements, warped_lab
            )
            solution.total_score = (
                solution.geometry_score
                + effective_texture_weight * solution.texture_score
            )
        best = min(finalists, key=lambda item: item.total_score)
        rospy.loginfo_throttle(
            2.0,
            "Solved onsite puzzle(grid): pieces=%d rectangles=%d nodes=%d unique=%d rect=%.1fx%.1f fill=%.3f geometry=%.3f texture=%.2f total=%.3f texture_var=%.2f",
            len(pieces),
            len(dimension_candidates),
            total_nodes,
            len(unique),
            best.width_mm,
            best.height_mm,
            best.fill_ratio,
            best.geometry_score,
            best.texture_score,
            best.total_score,
            texture_variance,
        )
        return best

    # ------------------------------------------------------------------
    # 目标放置与机器人坐标
    # ------------------------------------------------------------------
    def target_origin_mm_for_rect(
        self, source_half: str, width_mm: float, height_mm: float
    ) -> np.ndarray:
        x = (PAPER_W_MM - width_mm) / 2.0
        half_h = PAPER_H_MM / 2.0
        if source_half == "upper":
            y = half_h + (half_h - height_mm) / 2.0
        else:
            y = (half_h - height_mm) / 2.0
        return np.array([x, y], dtype=np.float64)

    def piece_gap_offset(
        self,
        placement: Placement,
        width_mm: float,
        height_mm: float,
    ) -> np.ndarray:
        if self.placement_gap_mm <= 0:
            return np.zeros(2, dtype=np.float64)
        centroid = self.polygon_centroid(placement.polygon_cart_mm)
        direction = centroid - np.array([width_mm / 2.0, height_mm / 2.0])
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float64)
        return direction / norm * self.placement_gap_mm

    def assembly_cart_to_warp_mm(
        self,
        point: np.ndarray,
        target_origin: np.ndarray,
        height_mm: float,
    ) -> np.ndarray:
        # 求解坐标 y 向上；A4 俯视坐标 y 向下。
        return np.array(
            [target_origin[0] + point[0], target_origin[1] + height_mm - point[1]],
            dtype=np.float64,
        )

    def process_frame(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[GenericResult], Optional[AssemblySolution]]:
        annotated = bgr.copy()
        empty_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        empty_warped = np.zeros((WARP_H, WARP_W, 3), dtype=np.uint8)
        empty_warp_mask = np.zeros((WARP_H, WARP_W), dtype=np.uint8)

        quad = self.get_a4_quad(bgr)
        if quad is None:
            self.draw_info_panel(annotated, ["A4 NOT FOUND"], quad=None, text_color=(80, 80, 255))
            return annotated, empty_mask, empty_warped, empty_warp_mask, [], None

        depth_snapshot = self.get_depth_snapshot()
        support_plane = self.fit_support_plane_camera(depth_snapshot, quad)
        dst = np.array(
            [[0, 0], [WARP_W - 1, 0], [WARP_W - 1, WARP_H - 1], [0, WARP_H - 1]],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        H_inv = np.linalg.inv(H)
        warped = cv2.warpPerspective(bgr, H, (WARP_W, WARP_H))
        piece_mask_warp = self.segment_white_pieces(warped)
        pieces = self.extract_generic_pieces(piece_mask_warp)
        solution = self.solve_assembly(pieces, warped)

        quad_i = np.round(quad).astype(np.int32)
        cv2.polylines(annotated, [quad_i.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for p in quad_i:
            cv2.circle(annotated, tuple(p), 5, (0, 255, 0), -1)

        mask_original = cv2.warpPerspective(
            piece_mask_warp,
            H_inv,
            (bgr.shape[1], bgr.shape[0]),
            flags=cv2.INTER_NEAREST,
        )

        for piece in pieces:
            contour_orig = self.warp_contour_to_original(piece.contour_px, H_inv)
            cv2.polylines(
                annotated,
                [np.round(contour_orig).astype(np.int32)],
                True,
                (0, 255, 255),
                2,
            )
            pick_orig = self.warp_to_original(piece.grasp_warp_px, H_inv)
            cv2.circle(
                annotated,
                (int(round(pick_orig[0])), int(round(pick_orig[1]))),
                5,
                (0, 0, 255),
                -1,
            )
            cv2.polylines(
                warped,
                [np.round(piece.polygon_px).astype(np.int32)],
                True,
                (0, 255, 255),
                3,
            )
            cv2.circle(
                warped,
                (int(round(piece.grasp_warp_px[0])), int(round(piece.grasp_warp_px[1]))),
                7,
                (0, 0, 255),
                -1,
            )

        if solution is None:
            lines = [
                "onsite pieces: {}".format(len(pieces)),
                "rectangle solution: NOT FOUND",
            ]
            for piece in pieces:
                lines.append(
                    "P{} A={:.0f}mm2 edges={}".format(
                        piece.piece_id, piece.area_mm2, len(piece.polygon_cart_mm)
                    )
                )
            self.draw_info_panel(annotated, lines, quad=quad, text_color=(80, 80, 255))
            return annotated, mask_original, warped, piece_mask_warp, [], None

        mean_y = float(np.mean([p.grasp_warp_px[1] for p in pieces]))
        source_half = "upper" if mean_y < WARP_H / 2.0 else "lower"
        target_origin = self.target_origin_mm_for_rect(
            source_half, solution.width_mm, solution.height_mm
        )

        target_rect_cart = np.array(
            [
                [0.0, 0.0],
                [solution.width_mm, 0.0],
                [solution.width_mm, solution.height_mm],
                [0.0, solution.height_mm],
            ],
            dtype=np.float64,
        )
        target_rect_warp_mm = np.array(
            [
                self.assembly_cart_to_warp_mm(q, target_origin, solution.height_mm)
                for q in target_rect_cart
            ],
            dtype=np.float64,
        )
        target_rect_warp_px = target_rect_warp_mm * np.array([PX_PER_MM_X, PX_PER_MM_Y])
        target_rect_orig = self.warp_contour_to_original(
            target_rect_warp_px.astype(np.float32).reshape(-1, 1, 2), H_inv
        )
        cv2.polylines(
            annotated,
            [np.round(target_rect_orig).astype(np.int32)],
            True,
            (255, 180, 0),
            2,
        )

        results: List[GenericResult] = []
        info_lines = [
            "{} pcs rect={:.1f}x{:.1f}mm fill={:.3f}".format(
                len(pieces), solution.width_mm, solution.height_mm, solution.fill_ratio
            ),
            "geom={:.2f} texture={:.1f} plane={}".format(
                solution.geometry_score,
                solution.texture_score,
                "OK" if support_plane is not None else "NO",
            ),
        ]

        for idx, piece in enumerate(pieces):
            placement = solution.placements[idx]
            gap_offset = self.piece_gap_offset(
                placement, solution.width_mm, solution.height_mm
            )
            target_grasp_cart = placement.R @ piece.grasp_local_cart_mm + placement.t + gap_offset
            target_poly_cart = placement.polygon_cart_mm + gap_offset

            target_grasp_warp_mm = self.assembly_cart_to_warp_mm(
                target_grasp_cart, target_origin, solution.height_mm
            )
            target_grasp_warp_px = target_grasp_warp_mm * np.array(
                [PX_PER_MM_X, PX_PER_MM_Y]
            )
            place_pixel = self.warp_to_original(
                (float(target_grasp_warp_px[0]), float(target_grasp_warp_px[1])), H_inv
            )
            pick_pixel = self.warp_to_original(piece.grasp_warp_px, H_inv)

            target_poly_warp_mm = np.array(
                [
                    self.assembly_cart_to_warp_mm(q, target_origin, solution.height_mm)
                    for q in target_poly_cart
                ],
                dtype=np.float64,
            )
            target_poly_warp_px = target_poly_warp_mm * np.array(
                [PX_PER_MM_X, PX_PER_MM_Y]
            )
            target_poly_orig = self.warp_contour_to_original(
                target_poly_warp_px.astype(np.float32).reshape(-1, 1, 2), H_inv
            )
            cv2.polylines(
                annotated,
                [np.round(target_poly_orig).astype(np.int32)],
                True,
                (255, 120, 0),
                2,
            )
            cv2.circle(
                annotated,
                (int(round(place_pixel[0])), int(round(place_pixel[1]))),
                4,
                (255, 0, 0),
                -1,
            )

            raw_pick_depth = self.robust_depth_at(*pick_pixel)
            plane_pick_depth = self.plane_depth_at_pixel(*pick_pixel, support_plane)
            pick_depth_for_xyz = plane_pick_depth if plane_pick_depth is not None else raw_pick_depth
            pick_robot = self.pixel_to_robot_xyz(*pick_pixel, pick_depth_for_xyz)

            raw_place_depth = self.robust_depth_at(*place_pixel)
            plane_place_depth = self.plane_depth_at_pixel(*place_pixel, support_plane)
            place_depth_for_xyz = plane_place_depth if plane_place_depth is not None else raw_place_depth
            place_robot = self.pixel_to_robot_xyz(*place_pixel, place_depth_for_xyz)

            pick_command = None if pick_robot is None else (pick_robot[0], pick_robot[1], self.grasp_z)
            place_command = None if place_robot is None else (place_robot[0], place_robot[1], self.place_z)
            angle_ccw = math.degrees(math.atan2(placement.R[1, 0], placement.R[0, 0]))
            rotation_clockwise = -angle_ccw
            rotation_clockwise = (rotation_clockwise + 180.0) % 360.0 - 180.0

            results.append(
                GenericResult(
                    label=piece.piece_id,
                    piece=piece,
                    pick_pixel=pick_pixel,
                    place_pixel=place_pixel,
                    pick_command_xyz=pick_command,
                    place_command_xyz=place_command,
                    pick_measured_robot_xyz=pick_robot,
                    place_measured_robot_xyz=place_robot,
                    pick_depth_mm=raw_pick_depth,
                    place_depth_mm=raw_place_depth,
                    required_rotation_deg_clockwise=rotation_clockwise,
                    target_polygon_warp_px=target_poly_warp_px,
                )
            )

            info_lines.append(
                "P{} A={:.0f} e={} rot={:.1f}deg".format(
                    piece.piece_id,
                    piece.area_mm2,
                    len(piece.polygon_cart_mm),
                    rotation_clockwise,
                )
            )
            if pick_command is not None:
                info_lines.append(
                    "  pick X={:.1f} Y={:.1f} Z={:.1f}".format(*pick_command)
                )

        self.draw_info_panel(annotated, info_lines, quad=quad)
        return annotated, mask_original, warped, piece_mask_warp, results, solution

    # ------------------------------------------------------------------
    # ROS 发布
    # ------------------------------------------------------------------
    @staticmethod
    def finite_or_none_generic(value: Optional[float]):
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
        results: List[GenericResult],
        solution: Optional[AssemblySolution],
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

        white_seed_msg = self.bridge.cv2_to_imgmsg(self.last_white_seed, encoding="mono8")
        white_seed_msg.header = header
        self.white_seed_pub.publish(white_seed_msg)

        pose_array = PoseArray()
        pose_array.header = header
        pose_array.header.frame_id = "robot_base"

        payload = {
            "mode": "onsite_arbitrary_rectangle",
            "count": len(results),
            "solution_found": solution is not None,
            "paper_corner_source": self.paper_source,
            "target_rectangle_mm": (
                [round(solution.width_mm, 2), round(solution.height_mm, 2)]
                if solution is not None
                else None
            ),
            "fill_ratio": (
                round(solution.fill_ratio, 5) if solution is not None else None
            ),
            "geometry_score": (
                round(solution.geometry_score, 5) if solution is not None else None
            ),
            "texture_score": (
                round(solution.texture_score, 5) if solution is not None else None
            ),
            "note": "required_rotation is mandatory for arbitrary pieces; no powered fourth axis means manual pre-alignment is required",
            "pieces": [],
        }

        for result in sorted(results, key=lambda r: r.label):
            if result.pick_command_xyz is not None:
                pose = Pose()
                pose.position.x = result.pick_command_xyz[0]
                pose.position.y = result.pick_command_xyz[1]
                pose.position.z = result.pick_command_xyz[2]
                pose.orientation.w = 1.0
                pose_array.poses.append(pose)

            payload["pieces"].append(
                {
                    "label": "P{}".format(result.label),
                    "area_mm2": round(result.piece.area_mm2, 2),
                    "edge_count": len(result.piece.polygon_cart_mm),
                    "edge_lengths_mm": [
                        round(float(v), 2)
                        for v in self.edge_lengths(result.piece.polygon_cart_mm)
                    ],
                    "pick_pixel_uv": [round(result.pick_pixel[0], 2), round(result.pick_pixel[1], 2)],
                    "place_pixel_uv": (
                        [round(result.place_pixel[0], 2), round(result.place_pixel[1], 2)]
                        if result.place_pixel is not None
                        else None
                    ),
                    "pick_depth_mm": self.finite_or_none_generic(result.pick_depth_mm),
                    "place_depth_mm": self.finite_or_none_generic(result.place_depth_mm),
                    "pick_measured_robot_xyz": (
                        [self.finite_or_none_generic(x) for x in result.pick_measured_robot_xyz]
                        if result.pick_measured_robot_xyz is not None
                        else None
                    ),
                    "place_measured_robot_xyz": (
                        [self.finite_or_none_generic(x) for x in result.place_measured_robot_xyz]
                        if result.place_measured_robot_xyz is not None
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
                    "required_rotation_deg_clockwise": round(
                        result.required_rotation_deg_clockwise, 3
                    ),
                }
            )

        self.pose_pub.publish(pose_array)
        self.coords_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    # 基类 color_callback 的返回值数量是固定的，因此覆盖。
    def color_callback(self, msg) -> None:
        if not getattr(self, "onsite_ready", False):
            return
        now = rospy.get_time()
        if now - self.last_process_time < 1.0 / max(self.process_hz, 0.1):
            return
        self.last_process_time = now
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            annotated, mask, warped, warped_mask, results, solution = self.process_frame(bgr)
            self.publish_outputs(
                msg.header, annotated, mask, warped, warped_mask, results, solution
            )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "Onsite puzzle processing error: %s", exc)


if __name__ == "__main__":
    try:
        OnsitePuzzleSolver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
