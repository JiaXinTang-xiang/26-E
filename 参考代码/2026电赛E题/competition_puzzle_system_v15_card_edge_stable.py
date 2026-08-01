#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现场比赛一体化拼图系统 v15（扑克牌边缘时域稳定 + 场景防抖 + 持续深化搜索，ROS1 / SwiftPro）。

一个文件同时完成：
1. 固定 A4 四角透视矫正；
2. 现场任意 1~4 片白底/扑克牌碎片分割；
3. 3~5 边多边形提取、中心安全抓取点计算；
4. 按赛题约束进行边角色枚举、联合位姿/矩形优化、持续深化搜索和纹理接缝评分；
5. 发布图像、坐标和求解状态；
6. 提供机械臂抓放服务，抓取 Z=-35 mm、放置 Z=-30 mm；
7. 全部完成后回到 (100, -100, 35) mm。

只需本文件和 a4_corner_calibrator.py。所有比赛参数都在 INLINE_CONFIG 中，
无需 YAML 参数文件或 launch 文件。仍可用 ROS 私有参数覆盖某个值。

v15 对扑克牌边缘采用多帧二值投票，并对场景变化进行连续帧确认；
轮廓的单帧跳变不会取消或重新启动求解。旋转限制仍按 INLINE_CONFIG 设置。
"""

import copy
import itertools
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rospy

try:
    from scipy.optimize import least_squares  # type: ignore
except Exception:  # pragma: no cover
    least_squares = None

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from swiftpro.msg import position, status


# ============================================================================
# 所有可调参数集中在这里。比赛前只需要改此字典。
# ============================================================================
INLINE_CONFIG = {
    # 比赛版以代码内参数为唯一真值，覆盖 ROS 参数服务器中的旧值。
    "force_inline_config": True,
    # 相机、A4 和手眼标定
    "color_topic": "/camera/color/image_raw",
    "depth_topic": "/camera/aligned_depth_to_color/image_raw",
    "paper_corner_file": "~/.ros/puzzle_a4_corners.yaml",
    "allow_auto_a4_fallback": False,
    "calib_file": "~/thefile.txt",
    "process_hz": 5.0,
    "source_region": "full",
    "divider_ignore_mm": 0.0,
    "paper_border_ignore_mm": 8.0,

    # 白底/扑克牌碎片分割
    "white_margin": 40,
    "white_sat_max": 165,
    "seed_min_area_mm2": 8.0,
    "segment_open_mm": 0.5,
    "segment_close_mm": 1.2,
    # 扑克牌边缘时域稳定：对最近若干帧的最终碎片掩膜做多数投票。
    # 红黑花纹碰到边缘、曝光微变或 approxPolyDP 顶点跳变时，不会立即改变轮廓。
    "temporal_mask_window": 5,
    "temporal_mask_vote_ratio": 0.60,
    "temporal_mask_reset_iou": 0.55,
    "temporal_mask_close_mm": 0.8,
    "min_piece_area_mm2": 120.0,
    "max_piece_area_mm2": 11000.0,
    "max_candidate_count": 6,

    # 现场碎片约束
    "min_piece_count": 1,
    "max_piece_count": 4,
    "min_polygon_vertices": 3,
    "max_polygon_vertices": 5,
    "true_min_edge_mm": 20.0,
    "detected_min_edge_mm": 14.0,
    "min_edge_overlap_mm": 10.0,

    # 目标矩形范围 9x5 cm ~ 12x9 cm
    "rect_long_min_mm": 90.0,
    "rect_long_max_mm": 120.0,
    "rect_short_min_mm": 50.0,
    "rect_short_max_mm": 90.0,
    "dimension_slack_mm": 0.0,
    "rect_area_tolerance_ratio": 0.10,
    "rectangle_step_mm": 2.0,

    # 限时求解、后台线程与缓存
    "solver_timeout_sec": 10.0,
    "solver_stable_frames": 3,
    "solver_retry_sec": 0.2,
    # v12：失败后不停止。后台线程会换碎片根节点/排列并逐轮扩大候选集合，
    # 一直搜索到 FOUND、场景发生真实变化、手动 reset，或达到可选总时限。
    "solver_auto_retry_failed": True,
    "solver_continuous_search": True,
    # 0 表示不设总时限，持续搜索。比赛前若希望给机械臂保留固定时间，
    # 可改成 60~80 秒；达到时限后会自动重新启动下一搜索周期。
    "solver_total_search_timeout_sec": 0.0,
    # 0 表示不限轮数。每轮包含 3 个不同深度、不同碎片排列的尝试。
    "solver_max_search_rounds": 0,
    "solver_round_pause_sec": 0.10,
    # 每完成一轮排列循环后，增加初值、边配对、优化候选和迭代次数，
    # 但不放宽最终安全验收条件，避免长搜变成接受伪解。
    "solver_round_growth": 0.35,
    "solver_round_scale_cap": 3.0,
    "solver_round_timeout_growth": 0.20,
    "solver_round_timeout_scale_cap": 2.5,
    "solver_diversify_piece_order": True,
    # 同一稳定场景不是重复执行完全相同的搜索，而是自动进行 3 级递进搜索。
    # 第 1 级快速粗搜；第 2 级常规搜索；第 3 级扩大候选和节点数。
    # 三次均失败后才缓存失败结果，等待场景真实变化或手动 reset。
    "solver_attempt_timeouts_sec": [3.0, 5.0, 8.0],
    "solver_attempt_rect_candidates": [4, 8, 14],
    "solver_attempt_grid_mm": [4.0, 2.5, 2.0],
    "solver_attempt_node_limits": [20000, 90000, 280000],
    "solver_attempt_placements_per_piece": [260, 750, 1500],
    "solver_attempt_branch_options": [100, 300, 700],
    # 识别轮廓不可能与真实铁片完全一致，因此求解采用软约束。
    # 第 1 级偏严格，第 2/3 级逐步接受小空隙、小模型重叠和轻微越界。
    "solver_attempt_min_fill_ratio": [0.88, 0.84, 0.80],
    "solver_attempt_min_contact_mm": [0.0, 0.0, 0.0],
    "solver_attempt_area_tolerance_ratio": [0.10, 0.13, 0.16],
    # 目标是矩形：四个外角必须由碎片的凸近直角顶点构成。第三级允许
    # 有一个角因轮廓拟合失真未被识别，但仍对缺角施加强惩罚。
    "solver_attempt_min_rect_corners": [4, 4, 3],
    "right_angle_prior_enabled": True,
    "right_angle_tolerance_deg": 28.0,
    "right_angle_alignment_tolerance_deg": 24.0,
    "right_angle_min_adjacent_edge_mm": 11.0,
    "right_angle_corner_snap_mm": 8.0,
    "right_angle_missing_corner_penalty": 90.0,
    "right_angle_error_penalty": 0.45,
    "solution_min_accept_fill_ratio": 0.82,
    "approx_boundary_tolerance_mm": 6.0,
    "approx_contact_tolerance_mm": 10.0,
    "approx_max_overlap_area_mm2": 120.0,
    "approx_overlap_penalty_per_mm2": 0.16,
    "approx_outside_penalty_per_mm2": 0.24,
    "approx_gap_penalty_per_mm2": 0.045,
    # 场景变化判定：位置、边长在 5 mm 内的抖动视为同一场景
    "solver_change_tolerance_mm": 5.0,
    "solver_angle_tolerance_deg": 12.0,
    "solver_area_tolerance_ratio": 0.15,
    # 只有连续多帧都确认发生明显变化，才认为换了场景。
    # 5 mm 内直接视为未变化；5~12 mm 的偶发跳变也先按抖动处理。
    "solver_hard_change_tolerance_mm": 12.0,
    "scene_change_confirm_frames": 5,
    "max_rectangle_candidates": 6,
    "packing_grid_mm": 2.0,
    "packing_raster_px_per_mm": 1.0,
    "packing_node_limit": 60000,
    "packing_solution_limit": 8,
    "max_placements_per_piece": 650,
    "max_branch_options": 220,
    "packing_min_fill_ratio": 0.80,
    "packing_min_contact_mm": 6.0,
    "finalist_count": 10,

    # v10 拓扑求解器：先建立边兼容矩阵，再枚举最多 16 种生成树拓扑。
    # 这些参数全部内置在代码中，不需要 YAML。
    "topology_merge_collinear_deg": 18.0,
    "topology_chain_min_length_mm": 8.0,
    "topology_pair_options_per_level": [7, 14, 24],
    "topology_edge_abs_tol_mm_per_level": [5.0, 8.0, 12.0],
    "topology_edge_ratio_tol_per_level": [0.16, 0.25, 0.36],
    "topology_partial_edge_max_ratio_per_level": [2.2, 4.5, 6.0],
    "topology_interval_reuse_tolerance": 0.06,
    "topology_pair_overlap_limit_mm2_per_level": [35.0, 70.0, 130.0],
    "topology_complete_limit_per_level": [180, 650, 1800],
    "topology_orientation_limit": 10,
    "topology_dimension_candidates": 4,
    "topology_accept_fill_per_level": [0.82, 0.77, 0.72],
    "topology_accept_area_error_per_level": [0.12, 0.18, 0.25],
    "topology_accept_overlap_mm2_per_level": [70.0, 120.0, 180.0],
    "topology_accept_outside_mm2_per_level": [60.0, 120.0, 200.0],
    "topology_accept_unmatched_ratio_per_level": [0.34, 0.44, 0.54],
    "topology_corner_tolerance_mm_per_level": [7.0, 10.0, 14.0],
    "topology_outer_line_tolerance_mm_per_level": [3.0, 5.0, 8.0],
    "topology_line_tolerance_mm_per_level": [2.5, 4.0, 6.0],
    "topology_angle_tolerance_deg_per_level": [8.0, 13.0, 20.0],
    "topology_texture_finalists": 8,

    # v11 赛题联合优化求解器。先用少量边兼容拓扑生成初值，再把所有
    # 碎片位姿和目标矩形一次性联合优化，避免生成树传播误差累积。
    "joint_initial_assemblies_per_level": [45, 90, 160],
    "joint_pair_options_per_tree_edge": [4, 6, 9],
    "joint_optimize_candidates_per_level": [18, 35, 60],
    "joint_max_nfev_per_level": [55, 85, 120],
    "joint_pose_angle_window_deg_per_level": [22.0, 35.0, 50.0],
    "joint_pose_translation_window_mm_per_level": [12.0, 20.0, 30.0],
    "joint_seam_sigma_mm_per_level": [2.2, 3.2, 4.5],
    "joint_outer_sigma_mm_per_level": [2.5, 3.8, 5.5],
    "joint_corner_sigma_mm_per_level": [5.0, 8.0, 12.0],
    "joint_area_sigma_ratio_per_level": [0.045, 0.070, 0.10],
    "joint_outer_seed_max_cost_per_level": [10.0, 15.0, 22.0],
    "joint_accept_area_error_per_level": [0.065, 0.085, 0.11],
    "joint_accept_fill_per_level": [0.86, 0.82, 0.78],
    "joint_accept_overlap_mm2_per_level": [55.0, 85.0, 120.0],
    "joint_accept_outside_mm2_per_level": [45.0, 80.0, 120.0],
    "joint_accept_seam_rms_mm_per_level": [3.5, 5.0, 7.0],
    "joint_extra_seam_line_tol_mm_per_level": [3.0, 5.0, 7.5],
    "joint_extra_seam_angle_tol_deg_per_level": [9.0, 14.0, 21.0],
    "joint_texture_finalists": 6,

    # 扑克牌纹理接缝评分
    "texture_weight": 0.045,
    "texture_sample_offset_mm": 0.8,
    "texture_variance_threshold": 12.0,

    # 抓取/放置几何
    "grasp_edge_margin_mm": 3.0,
    "placement_gap_mm": 2.5,
    "grasp_z": -35.0,
    "place_z": -30.0,

    # SwiftPro 执行
    "coords_topic": "/puzzle/piece_coordinates",
    "solver_status_topic": "/puzzle/solver_status",
    "arm_topic": "position_write_topic",
    "pump_topic": "pump_topic",
    "pick_z": -35.0,
    "safe_z": 80.0,
    "finish_xyz": [100.0, -100.0, 35.0],
    "max_rotation_deg": 18.0,
    "ignore_rotation": True,
    "require_solver_found": True,
    # 调用 start 时若求解器仍在 SOLVING/STABILIZING，不再返回失败。
    # 执行线程会等待 FOUND 和稳定坐标，然后自动开始机械臂动作。
    "start_wait_for_found": True,
    # 0 表示无限等待；比赛时可改成例如 60.0。
    "start_wait_timeout_s": 0.0,
    "start_wait_poll_s": 0.20,
    "required_frames": 5,
    "max_data_age_s": 2.0,
    "max_solver_status_age_s": 3.0,
    "max_xy_jitter_mm": 3.0,
    "move_wait_s": 1.2,
    "pump_wait_s": 1.0,
    "release_wait_s": 0.8,
    "between_piece_wait_s": 0.4,
    "dry_run": False,

    # 机械臂工作空间保护
    "x_min": 0.0,
    "x_max": 450.0,
    "y_min": -250.0,
    "y_max": 250.0,
    "z_min": -100.0,
    "z_max": 220.0,
}


def apply_inline_config() -> None:
    """把代码内参数强制写入当前节点私有命名空间。

    ROS 参数服务器在节点退出后仍会保留旧参数。此前只在参数不存在时写入，
    会导致旧版的 ``solver_auto_retry_failed``、阈值等继续生效。比赛版要求
    参数全部放在本文件内，因此这里每次启动都覆盖旧私有参数。
    """
    for key, value in INLINE_CONFIG.items():
        if key == "force_inline_config":
            continue
        rospy.set_param("~" + key, value)



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
        rospy.init_node("competition_puzzle_system", anonymous=False)
        apply_inline_config()
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
    # 该候选中，哪些目标矩形角由“凸近直角顶点”占据；bit0..bit3
    corner_mask: int = 0
    # 顶点到矩形角、角度及边方向的综合误差，越小越可信。
    corner_error: float = 1.0e9


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

        # 识别误差会让本应接触的两条边相差数毫米。这里用“近邻接触带”
        # 代替必须逐像素贴合的硬条件。
        contact_tol_mm = float(getattr(self, "approx_contact_tolerance_mm", 1.0))
        radius = max(1, int(round(contact_tol_mm * scale)))
        kernel_size = 2 * radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(mask, kernel, iterations=1)
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
            # 普通现场版在这里显示 NOT FOUND；比赛版由子类根据后台状态
            # 显示 STABILIZING / SOLVING / TIMEOUT / NOT_FOUND。
            # 否则同一帧会绘制两次半透明面板，形成 “SOLVING ND” 重影。
            if not getattr(self, "competition_ready", False):
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
                self.draw_info_panel(
                    annotated, lines, quad=quad, text_color=(80, 80, 255)
                )
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




class CompetitionOnsitePuzzleSolver(OnsitePuzzleSolver):
    """不阻塞相机回调、带超时和缓存的比赛版求解器。"""

    def __init__(self) -> None:
        self.competition_ready = False
        super().__init__()

        # ---------- 比赛求解限制 ----------
        self.solver_timeout_sec = float(
            rospy.get_param("~solver_timeout_sec", 4.5)
        )
        self.solver_stable_frames = int(
            rospy.get_param("~solver_stable_frames", 3)
        )
        self.solver_retry_sec = float(
            rospy.get_param("~solver_retry_sec", 4.0)
        )
        self.solver_auto_retry_failed = bool(
            rospy.get_param("~solver_auto_retry_failed", True)
        )
        self.solver_continuous_search = bool(
            rospy.get_param("~solver_continuous_search", True)
        )
        self.solver_total_search_timeout_sec = float(
            rospy.get_param("~solver_total_search_timeout_sec", 0.0)
        )
        self.solver_max_search_rounds = int(
            rospy.get_param("~solver_max_search_rounds", 0)
        )
        self.solver_round_pause_sec = float(
            rospy.get_param("~solver_round_pause_sec", 0.10)
        )
        self.solver_round_growth = float(
            rospy.get_param("~solver_round_growth", 0.35)
        )
        self.solver_round_scale_cap = float(
            rospy.get_param("~solver_round_scale_cap", 3.0)
        )
        self.solver_round_timeout_growth = float(
            rospy.get_param("~solver_round_timeout_growth", 0.20)
        )
        self.solver_round_timeout_scale_cap = float(
            rospy.get_param("~solver_round_timeout_scale_cap", 2.5)
        )
        self.solver_diversify_piece_order = bool(
            rospy.get_param("~solver_diversify_piece_order", True)
        )

        # 同一场景采用不同搜索宽度的分级尝试，而不是无限重复同一搜索。
        self.solver_attempt_timeouts = [
            float(v) for v in rospy.get_param(
                "~solver_attempt_timeouts_sec", [3.5, 7.0, 12.0]
            )
        ]
        self.solver_attempt_rect_candidates = [
            int(v) for v in rospy.get_param(
                "~solver_attempt_rect_candidates", [4, 6, 10]
            )
        ]
        self.solver_attempt_grid_mm = [
            float(v) for v in rospy.get_param(
                "~solver_attempt_grid_mm", [3.0, 2.0, 1.5]
            )
        ]
        self.solver_attempt_node_limits = [
            int(v) for v in rospy.get_param(
                "~solver_attempt_node_limits", [25000, 80000, 220000]
            )
        ]
        self.solver_attempt_placements_per_piece = [
            int(v) for v in rospy.get_param(
                "~solver_attempt_placements_per_piece", [300, 650, 1100]
            )
        ]
        self.solver_attempt_branch_options = [
            int(v) for v in rospy.get_param(
                "~solver_attempt_branch_options", [100, 220, 380]
            )
        ]
        self.solver_attempt_min_fill_ratio = [
            float(v) for v in rospy.get_param(
                "~solver_attempt_min_fill_ratio", [0.78, 0.70, 0.62]
            )
        ]
        self.solver_attempt_min_contact_mm = [
            float(v) for v in rospy.get_param(
                "~solver_attempt_min_contact_mm", [3.0, 1.0, 1.0]
            )
        ]
        self.solver_attempt_area_tolerance_ratio = [
            float(v) for v in rospy.get_param(
                "~solver_attempt_area_tolerance_ratio", [0.10, 0.15, 0.20]
            )
        ]

        # 近似拼接参数：检测轮廓允许少量越界、间隙和模型重叠。
        # 这些误差只用于几何求解，实际下放位置仍会通过 placement_gap_mm
        # 略微分开，避免实体薄片真正重叠。
        self.approx_boundary_tolerance_mm = float(
            rospy.get_param("~approx_boundary_tolerance_mm", 4.0)
        )
        self.approx_contact_tolerance_mm = float(
            rospy.get_param("~approx_contact_tolerance_mm", 8.0)
        )
        self.approx_max_overlap_area_mm2 = float(
            rospy.get_param("~approx_max_overlap_area_mm2", 160.0)
        )
        self.approx_overlap_penalty_per_mm2 = float(
            rospy.get_param("~approx_overlap_penalty_per_mm2", 0.10)
        )
        self.approx_outside_penalty_per_mm2 = float(
            rospy.get_param("~approx_outside_penalty_per_mm2", 0.18)
        )
        self.approx_gap_penalty_per_mm2 = float(
            rospy.get_param("~approx_gap_penalty_per_mm2", 0.035)
        )

        # 矩形拓扑先验。不是把“所有直角”都当作外角，而是要求最终矩形的
        # 四个角由某一碎片的凸近直角顶点支撑，并优先生成这类候选。
        self.right_angle_prior_enabled = bool(
            rospy.get_param("~right_angle_prior_enabled", True)
        )
        self.right_angle_tolerance_deg = float(
            rospy.get_param("~right_angle_tolerance_deg", 28.0)
        )
        self.right_angle_alignment_tolerance_deg = float(
            rospy.get_param("~right_angle_alignment_tolerance_deg", 24.0)
        )
        self.right_angle_min_adjacent_edge_mm = float(
            rospy.get_param("~right_angle_min_adjacent_edge_mm", 11.0)
        )
        self.right_angle_corner_snap_mm = float(
            rospy.get_param("~right_angle_corner_snap_mm", 8.0)
        )
        self.right_angle_missing_corner_penalty = float(
            rospy.get_param("~right_angle_missing_corner_penalty", 90.0)
        )
        self.right_angle_error_penalty = float(
            rospy.get_param("~right_angle_error_penalty", 0.45)
        )
        self.solution_min_accept_fill_ratio = float(
            rospy.get_param("~solution_min_accept_fill_ratio", 0.76)
        )
        self.solver_attempt_min_rect_corners = [
            int(v) for v in rospy.get_param(
                "~solver_attempt_min_rect_corners", [4, 4, 3]
            )
        ]
        self.packing_min_rect_corners = int(
            rospy.get_param("~packing_min_rect_corners", 4)
        )
        attempt_lengths = [
            len(self.solver_attempt_timeouts),
            len(self.solver_attempt_rect_candidates),
            len(self.solver_attempt_grid_mm),
            len(self.solver_attempt_node_limits),
            len(self.solver_attempt_placements_per_piece),
            len(self.solver_attempt_branch_options),
            len(self.solver_attempt_min_fill_ratio),
            len(self.solver_attempt_min_contact_mm),
            len(self.solver_attempt_area_tolerance_ratio),
            len(self.solver_attempt_min_rect_corners),
        ]
        self.solver_attempt_count = max(1, min(attempt_lengths))
        self.solver_attempt_index = 0
        self.solver_round_index = 0
        self.solver_round_piece_order = []
        self.solver_cancel_event = threading.Event()

        # 轮廓在相机中会有少量抖动。位置或边长变化不超过该值时，
        # 继续视为同一个场景，不重新启动求解器。
        self.solver_change_tolerance_mm = float(
            rospy.get_param("~solver_change_tolerance_mm", 5.0)
        )
        self.solver_angle_tolerance_deg = float(
            rospy.get_param("~solver_angle_tolerance_deg", 6.0)
        )
        self.solver_area_tolerance_ratio = float(
            rospy.get_param("~solver_area_tolerance_ratio", 0.10)
        )
        self.solver_hard_change_tolerance_mm = float(
            rospy.get_param("~solver_hard_change_tolerance_mm", 12.0)
        )
        self.scene_change_confirm_frames = max(1, int(
            rospy.get_param("~scene_change_confirm_frames", 5)
        ))

        # 扑克牌边缘多帧稳定。掩膜发生大幅变化时自动清空历史，
        # 所以机械臂搬走碎片后不会留下长时间重影。
        self.temporal_mask_window = max(1, int(
            rospy.get_param("~temporal_mask_window", 5)
        ))
        self.temporal_mask_vote_ratio = float(
            rospy.get_param("~temporal_mask_vote_ratio", 0.60)
        )
        self.temporal_mask_reset_iou = float(
            rospy.get_param("~temporal_mask_reset_iou", 0.55)
        )
        self.temporal_mask_close_mm = float(
            rospy.get_param("~temporal_mask_close_mm", 0.8)
        )
        self.temporal_mask_history = deque(maxlen=self.temporal_mask_window)
        self.temporal_last_raw_mask = None
        self.rectangle_step_mm = float(
            rospy.get_param("~rectangle_step_mm", 2.0)
        )
        self.max_placements_per_piece = int(
            rospy.get_param("~max_placements_per_piece", 650)
        )
        self.max_branch_options = int(
            rospy.get_param("~max_branch_options", 220)
        )
        self.packing_min_fill_ratio = float(
            rospy.get_param("~packing_min_fill_ratio", 0.80)
        )
        self.finalist_count = int(
            rospy.get_param("~finalist_count", 10)
        )

        # 覆盖旧版偏大的默认搜索量。仍允许在 launch/YAML 中修改。
        self.max_rectangle_candidates = int(
            rospy.get_param("~max_rectangle_candidates", 12)
        )
        self.packing_grid_mm = float(
            rospy.get_param("~packing_grid_mm", 2.0)
        )
        self.packing_scale = float(
            rospy.get_param("~packing_raster_px_per_mm", 1.0)
        )
        self.packing_node_limit = int(
            rospy.get_param("~packing_node_limit", 30000)
        )
        self.packing_solution_limit = int(
            rospy.get_param("~packing_solution_limit", 8)
        )
        self.dimension_slack_mm = float(
            rospy.get_param("~dimension_slack_mm", 3.0)
        )
        self.rect_area_tolerance_ratio = float(
            rospy.get_param("~rect_area_tolerance_ratio", 0.10)
        )

        # ---------- 后台线程与缓存 ----------
        self.solver_lock = threading.Lock()
        self.solver_thread: Optional[threading.Thread] = None
        self.solver_running_signature = None
        self.last_seen_signature = None
        self.signature_stable_count = 0
        self.scene_change_candidate_count = 0
        self.scene_change_candidate_signature = None

        self.cached_signature = None
        self.cached_solution: Optional[AssemblySolution] = None
        self.cached_status = "WAITING"
        self.cached_finished_at = 0.0
        self.cached_solve_sec = 0.0
        self.cached_stats: Dict[str, object] = {}
        self.last_piece_summaries: List[str] = []
        self.last_piece_count = 0

        self.status_pub = rospy.Publisher(
            "/puzzle/solver_status", String, queue_size=1
        )
        self.solver_reset_srv = rospy.Service(
            "/puzzle/solver_reset", Trigger, self.solver_reset_callback
        )
        self.competition_ready = True

        rospy.loginfo(
            "Card-edge stabilization: mask_window=%d vote=%.2f reset_iou=%.2f "
            "scene_soft=%.1fmm scene_hard=%.1fmm confirm=%d",
            self.temporal_mask_window,
            self.temporal_mask_vote_ratio,
            self.temporal_mask_reset_iou,
            self.solver_change_tolerance_mm,
            self.solver_hard_change_tolerance_mm,
            self.scene_change_confirm_frames,
        )
        rospy.loginfo(
            "Competition solver ready: timeout=%.1fs stable_frames=%d rects=%d "
            "grid=%.1fmm nodes=%d placements/piece=%d change_tol=%.1fmm",
            self.solver_timeout_sec,
            self.solver_stable_frames,
            self.max_rectangle_candidates,
            self.packing_grid_mm,
            self.packing_node_limit,
            self.max_placements_per_piece,
            self.solver_change_tolerance_mm,
        )
        rospy.loginfo(
            "Rectangle corner prior: enabled=%s angle_tol=%.1fdeg snap=%.1fmm min_corners=%s",
            self.right_angle_prior_enabled,
            self.right_angle_tolerance_deg,
            self.right_angle_corner_snap_mm,
            self.solver_attempt_min_rect_corners,
        )
        rospy.loginfo(
            "Continuous search: enabled=%s total_timeout=%.1fs max_rounds=%d "
            "growth=%.2f cap=%.2f diversify_order=%s",
            self.solver_continuous_search,
            self.solver_total_search_timeout_sec,
            self.solver_max_search_rounds,
            self.solver_round_growth,
            self.solver_round_scale_cap,
            self.solver_diversify_piece_order,
        )

    # ------------------------------------------------------------------
    # 扑克牌边缘时域稳定
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = np.asarray(mask_a) > 0
        b = np.asarray(mask_b) > 0
        union = int(np.count_nonzero(a | b))
        if union <= 0:
            return 1.0
        return float(np.count_nonzero(a & b)) / float(union)

    @staticmethod
    def _mask_component_count(mask: np.ndarray) -> int:
        count, _ = cv2.connectedComponents((np.asarray(mask) > 0).astype(np.uint8), 8)
        return max(0, int(count) - 1)

    def temporal_stabilize_piece_mask(self, raw_mask: np.ndarray) -> np.ndarray:
        """用最近多帧多数投票稳定扑克牌外轮廓。

        红色/黑色花纹接触碎片边缘时，单帧白色阈值会产生凹口，
        进而令多边形边数和外接尺寸跳变。多数投票保留持续存在的实体边界，
        删除只出现一两帧的锯齿。若整体 IoU 很低或连通块数量变化，说明
        碎片确实被移动，立即清空历史，避免产生拖影。
        """
        raw = np.where(np.asarray(raw_mask) > 0, 255, 0).astype(np.uint8)
        if self.temporal_mask_window <= 1:
            return raw

        if self.temporal_last_raw_mask is not None:
            iou = self._mask_iou(self.temporal_last_raw_mask, raw)
            # 不因单帧连通块数量跳变清空历史；扑克牌花纹触边时恰好可能
            # 让一个轮廓短暂分裂。只有整体重叠度明显下降才判定为真实移动。
            if iou < self.temporal_mask_reset_iou:
                self.temporal_mask_history.clear()

        self.temporal_last_raw_mask = raw.copy()
        self.temporal_mask_history.append(raw.copy())

        # 前两帧直接输出当前结果，避免启动时等待太久。
        if len(self.temporal_mask_history) < 3:
            return raw

        stack = np.stack(
            [(item > 0).astype(np.uint8) for item in self.temporal_mask_history],
            axis=0,
        )
        required = max(1, int(math.ceil(
            len(self.temporal_mask_history)
            * min(max(self.temporal_mask_vote_ratio, 0.01), 1.0)
        )))
        voted = np.where(np.sum(stack, axis=0) >= required, 255, 0).astype(np.uint8)

        close_px = max(1, int(round(
            self.temporal_mask_close_mm
            * 0.5 * (PX_PER_MM_X + PX_PER_MM_Y)
        )))
        if close_px % 2 == 0:
            close_px += 1
        if close_px > 1:
            voted = cv2.morphologyEx(
                voted, cv2.MORPH_CLOSE,
                np.ones((close_px, close_px), np.uint8), iterations=1
            )

        # 再次按外轮廓填满，确保扑克牌花纹孔洞不会影响边缘。
        contours, _ = cv2.findContours(
            voted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(voted)
        min_area_px = max(4.0, self.min_piece_area_mm2 * PX_PER_MM_X * PX_PER_MM_Y)
        max_area_px = self.max_piece_area_mm2 * PX_PER_MM_X * PX_PER_MM_Y
        for contour in contours:
            area = abs(float(cv2.contourArea(contour)))
            if min_area_px <= area <= max_area_px:
                cv2.drawContours(filled, [contour], -1, 255, cv2.FILLED)
        return filled

    def segment_white_pieces(self, warped: np.ndarray) -> np.ndarray:
        raw = super().segment_white_pieces(warped)
        return self.temporal_stabilize_piece_mask(raw)

    # ------------------------------------------------------------------
    # 场景签名与缓存
    # ------------------------------------------------------------------
    @staticmethod
    def _popcount(value: int) -> int:
        # 兼容 Python 3.6/3.7/3.8，不使用 int.bit_count()。
        return bin(int(value)).count("1")

    @staticmethod
    def _quantize(value: float, step: float) -> int:
        return int(round(float(value) / max(float(step), 1e-9)))

    @staticmethod
    def _angle_difference_deg(a: float, b: float) -> float:
        """返回两个角度之间的最小绝对差，范围 0~180 度。"""
        return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)

    def piece_signature(self, pieces: List[GenericPiece]):
        """生成抗抖动场景描述。

        使用最终填充轮廓的矩、面积和最小外接矩形，不使用 approxPolyDP
        顶点。这样扑克牌花纹造成的 3/4/5 边跳变不会改变场景身份。
        """
        scene = []
        for piece in pieces:
            contour = np.asarray(piece.contour_px, dtype=np.float32).reshape(-1, 1, 2)
            moments = cv2.moments(contour)
            if abs(float(moments.get("m00", 0.0))) > 1e-9:
                cx_px = float(moments["m10"] / moments["m00"])
                cy_px = float(moments["m01"] / moments["m00"])
            else:
                points = contour.reshape(-1, 2)
                cx_px = float(np.mean(points[:, 0]))
                cy_px = float(np.mean(points[:, 1]))

            rect = cv2.minAreaRect(contour)
            rw = float(rect[1][0]) / PX_PER_MM_X
            rh = float(rect[1][1]) / PX_PER_MM_Y
            angle = float(rect[2])
            if rw < rh:
                rw, rh = rh, rw
                angle += 90.0
            angle %= 180.0
            anisotropy = rw / max(rh, 1e-6)

            contour_area_mm2 = abs(float(cv2.contourArea(contour))) / (
                PX_PER_MM_X * PX_PER_MM_Y
            )
            scene.append(
                {
                    "centroid": (cx_px / PX_PER_MM_X, -cy_px / PX_PER_MM_Y),
                    "area_mm2": contour_area_mm2,
                    "rect_size": (rw, rh),
                    "axis_angle_deg": angle,
                    "anisotropy": anisotropy,
                }
            )
        return tuple(scene)

    @staticmethod
    def _axis_angle_difference_deg(a: float, b: float) -> float:
        """无方向主轴角差，范围 0~90 度。"""
        return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)

    def _piece_descriptors_equivalent(
        self, old_piece, new_piece, tolerance_scale: float = 1.0
    ) -> bool:
        """判断两帧中的描述是否为同一块碎片。

        中心和外接尺寸在 5 mm 内、面积在 15% 内视为未变化。只有细长
        碎片主轴真实转动超过 12 度才触发重求解；接近方形的碎片方向本来
        就不稳定，因此不使用角度判定。
        """
        scale = max(1.0, float(tolerance_scale))
        tol_mm = max(0.1, float(self.solver_change_tolerance_mm) * scale)
        old_center = np.asarray(old_piece["centroid"], dtype=np.float64)
        new_center = np.asarray(new_piece["centroid"], dtype=np.float64)
        if float(np.linalg.norm(old_center - new_center)) > tol_mm:
            return False

        old_area = float(old_piece["area_mm2"])
        new_area = float(new_piece["area_mm2"])
        area_ratio = abs(old_area - new_area) / max(old_area, new_area, 1.0)
        if area_ratio > max(0.01, float(self.solver_area_tolerance_ratio) * scale):
            return False

        old_size = np.asarray(old_piece["rect_size"], dtype=np.float64)
        new_size = np.asarray(new_piece["rect_size"], dtype=np.float64)
        if float(np.max(np.abs(old_size - new_size))) > tol_mm:
            return False

        # 只有明显细长的碎片才用主轴角判断真实旋转。
        if min(float(old_piece["anisotropy"]), float(new_piece["anisotropy"])) >= 1.25:
            if self._axis_angle_difference_deg(
                old_piece["axis_angle_deg"], new_piece["axis_angle_deg"]
            ) > max(1.0, float(self.solver_angle_tolerance_deg) * scale):
                return False
        return True

    def scene_equivalent(
        self, old_scene, new_scene, tolerance_scale: float = 1.0
    ) -> bool:
        """判断两帧是否属于同一场景。

        最多只有 4 片，直接枚举匹配即可。这样即使碎片标签因检测排序偶尔互换，
        只要每片中心移动不超过 5 mm、边长变化不超过 5 mm，仍不会重新求解。
        """
        if old_scene is None or new_scene is None:
            return False
        if len(old_scene) != len(new_scene):
            return False

        import itertools

        indices = range(len(new_scene))
        for permutation in itertools.permutations(indices):
            if all(
                self._piece_descriptors_equivalent(
                    old_scene[i], new_scene[permutation[i]], tolerance_scale
                )
                for i in range(len(old_scene))
            ):
                return True
        return False

    def current_solver_snapshot(self) -> Dict[str, object]:
        with self.solver_lock:
            running = self.solver_thread is not None and self.solver_thread.is_alive()
            return {
                "status": "SOLVING" if running else self.cached_status,
                "running": running,
                "stable_frames": self.signature_stable_count,
                "required_stable_frames": self.solver_stable_frames,
                "scene_change_pending_frames": int(self.scene_change_candidate_count),
                "scene_change_confirm_frames": int(self.scene_change_confirm_frames),
                "temporal_mask_frames": int(len(self.temporal_mask_history)),
                "solve_sec": round(float(self.cached_solve_sec), 3),
                "attempt_index": int(self.solver_attempt_index),
                "attempt_count": int(self.solver_attempt_count),
                "round_index": int(getattr(self, "solver_round_index", 0)),
                "piece_order": list(getattr(self, "solver_round_piece_order", [])),
                "stats": dict(self.cached_stats),
            }

    # ------------------------------------------------------------------
    # 快速矩形尺寸候选
    # ------------------------------------------------------------------
    def rectangle_dimension_candidates(
        self, pieces: List[GenericPiece]
    ) -> List[Tuple[float, float, float]]:
        total_area = float(sum(piece.area_mm2 for piece in pieces))
        step = max(1.0, float(self.rectangle_step_mm))
        long_min = self.rect_long_min_mm - self.dimension_slack_mm
        long_max = self.rect_long_max_mm + self.dimension_slack_mm
        short_min = self.rect_short_min_mm - self.dimension_slack_mm
        short_max = self.rect_short_max_mm + self.dimension_slack_mm

        candidates: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

        # 规则网格候选。
        long_side = math.ceil(long_min / step) * step
        while long_side <= long_max + 1e-9:
            short_side = math.ceil(short_min / step) * step
            while short_side <= min(short_max, long_side) + 1e-9:
                rect_area = long_side * short_side
                area_error = abs(rect_area - total_area) / max(total_area, 1.0)
                if area_error <= self.rect_area_tolerance_ratio:
                    penalty = self.dimension_penalty(long_side, short_side)
                    score = 4.0 * area_error + 0.02 * penalty
                    key = (
                        self._quantize(long_side, step),
                        self._quantize(short_side, step),
                    )
                    candidates[key] = (score, long_side, short_side)
                short_side += step
            long_side += step

        # 面积反推候选，避免目标尺寸落在规则步长中间。
        long_side = math.ceil(long_min)
        while long_side <= long_max + 1e-9:
            short_side = total_area / max(long_side, 1e-9)
            short_side = round(short_side / step) * step
            if short_min <= short_side <= min(short_max, long_side):
                rect_area = long_side * short_side
                area_error = abs(rect_area - total_area) / max(total_area, 1.0)
                if area_error <= self.rect_area_tolerance_ratio:
                    penalty = self.dimension_penalty(long_side, short_side)
                    score = 3.5 * area_error + 0.02 * penalty
                    key = (
                        self._quantize(long_side, step),
                        self._quantize(short_side, step),
                    )
                    old = candidates.get(key)
                    if old is None or score < old[0]:
                        candidates[key] = (score, float(long_side), float(short_side))
            long_side += 1.0

        ordered = sorted(candidates.values(), key=lambda item: item[0])
        return ordered[: self.max_rectangle_candidates]

    # ------------------------------------------------------------------
    # 矩形直角先验
    # ------------------------------------------------------------------
    @staticmethod
    def _angle_between_vectors_deg(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-9 or nb < 1e-9:
            return 180.0
        value = float(np.dot(a, b) / (na * nb))
        value = max(-1.0, min(1.0, value))
        return math.degrees(math.acos(value))

    def right_angle_vertices(self, polygon: np.ndarray):
        """返回凸的近 90 度顶点及其两条相邻边方向。"""
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        count = len(points)
        if count < 3:
            return []
        signed_area = float(self.polygon_signed_area(points))
        found = []
        for index in range(count):
            vertex = points[index]
            prev_vec = points[(index - 1) % count] - vertex
            next_vec = points[(index + 1) % count] - vertex
            prev_len = float(np.linalg.norm(prev_vec))
            next_len = float(np.linalg.norm(next_vec))
            if min(prev_len, next_len) < self.right_angle_min_adjacent_edge_mm:
                continue
            cross = float(prev_vec[0] * next_vec[1] - prev_vec[1] * next_vec[0])
            # 对 CCW 多边形，凸顶点在上述向量顺序下 cross<0；CW 相反。
            if abs(signed_area) > 1e-6 and signed_area * cross >= 0.0:
                continue
            angle = self._angle_between_vectors_deg(prev_vec, next_vec)
            error = abs(angle - 90.0)
            if error <= self.right_angle_tolerance_deg:
                found.append((index, error, prev_vec, next_vec))
        return found

    @staticmethod
    def rectangle_corner_targets(width_mm: float, height_mm: float):
        return [
            (0, np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            (1, np.array([width_mm, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0])),
            (2, np.array([width_mm, height_mm]), np.array([-1.0, 0.0]), np.array([0.0, -1.0])),
            (3, np.array([0.0, height_mm]), np.array([1.0, 0.0]), np.array([0.0, -1.0])),
        ]

    def candidate_corner_metrics(
        self, polygon: np.ndarray, width_mm: float, height_mm: float
    ) -> Tuple[int, float]:
        """统计候选中由凸近直角顶点支撑的目标矩形角。"""
        mask = 0
        total_error = 0.0
        vertices = self.right_angle_vertices(polygon)
        for corner_index, target, direction_a, direction_b in self.rectangle_corner_targets(
            width_mm, height_mm
        ):
            best = None
            for vertex_index, angle_error, prev_vec, next_vec in vertices:
                point = polygon[vertex_index]
                distance = float(np.linalg.norm(point - target))
                if distance > self.right_angle_corner_snap_mm:
                    continue
                align_direct = max(
                    self._angle_between_vectors_deg(prev_vec, direction_a),
                    self._angle_between_vectors_deg(next_vec, direction_b),
                )
                align_swapped = max(
                    self._angle_between_vectors_deg(prev_vec, direction_b),
                    self._angle_between_vectors_deg(next_vec, direction_a),
                )
                alignment_error = min(align_direct, align_swapped)
                if alignment_error > self.right_angle_alignment_tolerance_deg:
                    continue
                error = distance + 0.20 * angle_error + 0.12 * alignment_error
                if best is None or error < best:
                    best = error
            if best is not None:
                mask |= 1 << corner_index
                total_error += float(best)
        if mask == 0:
            return 0, 1.0e6
        return mask, total_error

    def solution_corner_metrics(
        self, polygons: Sequence[np.ndarray], width_mm: float, height_mm: float
    ) -> Tuple[int, float]:
        mask = 0
        error = 0.0
        for polygon in polygons:
            piece_mask, piece_error = self.candidate_corner_metrics(
                polygon, width_mm, height_mm
            )
            mask |= piece_mask
            if piece_mask:
                error += piece_error
        return self._popcount(mask), error

    @staticmethod
    def _point_segment_distance(point, a, b) -> float:
        point = np.asarray(point, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(point - a))
        t = float(np.dot(point - a, ab) / denom)
        t = max(0.0, min(1.0, t))
        return float(np.linalg.norm(point - (a + t * ab)))

    def polygon_distance_mm(self, first: np.ndarray, second: np.ndarray) -> float:
        # 有重叠时距离为 0。
        if self.pair_overlap_area_mm2(first, second) > 0.01:
            return 0.0
        best = float("inf")
        for point in first:
            for _, a, b in self.polygon_edges(second):
                best = min(best, self._point_segment_distance(point, a, b))
        for point in second:
            for _, a, b in self.polygon_edges(first):
                best = min(best, self._point_segment_distance(point, a, b))
        return best

    def assembly_connected(self, polygons: Sequence[np.ndarray]) -> bool:
        count = len(polygons)
        if count <= 1:
            return True
        graph = [[] for _ in range(count)]
        tolerance = max(1.0, float(self.approx_contact_tolerance_mm))
        for i in range(count):
            for j in range(i + 1, count):
                if self.polygon_distance_mm(polygons[i], polygons[j]) <= tolerance:
                    graph[i].append(j)
                    graph[j].append(i)
        seen = {0}
        stack = [0]
        while stack:
            current = stack.pop()
            for other in graph[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return len(seen) == count

    # ------------------------------------------------------------------
    # 带截止时间的候选生成
    # ------------------------------------------------------------------
    def generate_boundary_placements_timed(
        self,
        piece_index: int,
        piece: GenericPiece,
        width_mm: float,
        height_mm: float,
        deadline: float,
    ) -> Tuple[List[GridPlacementCandidate], bool]:
        step = max(0.5, self.packing_grid_mm)
        placements: List[GridPlacementCandidate] = []
        seen_geometry = set()
        seen_masks = set()
        axis_tol = math.sin(math.radians(self.edge_angle_tol_deg))
        timed_out = False
        soft_cap = max(self.max_placements_per_piece * 3, self.max_placements_per_piece)

        def append_candidate(R, t, poly) -> None:
            angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            key = (
                int(round(angle * 2.0)),
                int(round(t[0] / step)),
                int(round(t[1] / step)),
            )
            if key in seen_geometry:
                return
            seen_geometry.add(key)
            interior, full, border, cells = self.rasterize_grid_placement(
                poly, width_mm, height_mm
            )
            if full in seen_masks:
                return
            seen_masks.add(full)
            corner_mask, corner_error = self.candidate_corner_metrics(
                poly, width_mm, height_mm
            )
            placements.append(
                GridPlacementCandidate(
                    piece_index=piece_index,
                    R=R.copy(),
                    t=t.copy(),
                    polygon_cart_mm=poly.copy(),
                    interior_bits=interior,
                    full_bits=full,
                    border_bits=border,
                    full_cell_count=cells,
                    corner_mask=corner_mask,
                    corner_error=corner_error,
                )
            )

        # 先直接把碎片的凸近直角顶点吸附到目标矩形四个角。这是最重要的
        # 候选来源，避免正确角点在后续沿边滑动候选中被均匀抽样丢掉。
        if self.right_angle_prior_enabled:
            source_polygon = piece.polygon_cart_mm
            for vertex_index, _, prev_vec, next_vec in self.right_angle_vertices(
                source_polygon
            ):
                source_vertex = source_polygon[vertex_index]
                for _, target, direction_a, direction_b in self.rectangle_corner_targets(
                    width_mm, height_mm
                ):
                    for target_prev, target_next in (
                        (direction_a, direction_b),
                        (direction_b, direction_a),
                    ):
                        angle = math.atan2(target_prev[1], target_prev[0]) - math.atan2(
                            prev_vec[1], prev_vec[0]
                        )
                        R = self.rotation_matrix(angle)
                        rotated_next = R @ next_vec
                        alignment = self._angle_between_vectors_deg(
                            rotated_next, target_next
                        )
                        if alignment > self.right_angle_alignment_tolerance_deg:
                            continue
                        t = target - R @ source_vertex
                        poly = self.transform_polygon(piece, R, t)
                        tol = float(self.approx_boundary_tolerance_mm)
                        if (
                            np.min(poly[:, 0]) >= -tol
                            and np.max(poly[:, 0]) <= width_mm + tol
                            and np.min(poly[:, 1]) >= -tol
                            and np.max(poly[:, 1]) <= height_mm + tol
                        ):
                            append_candidate(R, t, poly)

        for R, oriented in self.unique_axis_orientations(piece):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            for _, a, b in self.polygon_edges(oriented):
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                vec = b - a
                length = float(np.linalg.norm(vec))
                if length < self.detected_min_edge_mm:
                    continue

                if abs(vec[1]) <= axis_tol * max(length, 1e-9):
                    for boundary_y in (0.0, height_mm):
                        ty = boundary_y - a[1]
                        shifted = oriented + np.array([0.0, ty])
                        min_x = float(np.min(shifted[:, 0]))
                        max_x = float(np.max(shifted[:, 0]))
                        tol = float(getattr(self, "approx_boundary_tolerance_mm", 0.7))
                        start = math.ceil((-min_x - tol) / step) * step
                        stop = math.floor((width_mm - max_x + tol) / step) * step
                        tx = start
                        while tx <= stop + 1e-9:
                            if time.monotonic() >= deadline:
                                timed_out = True
                                break
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -tol
                                and np.max(poly[:, 0]) <= width_mm + tol
                                and np.min(poly[:, 1]) >= -tol
                                and np.max(poly[:, 1]) <= height_mm + tol
                            ):
                                append_candidate(R, t, poly)
                            if len(placements) >= soft_cap:
                                break
                            tx += step
                        if timed_out or len(placements) >= soft_cap:
                            break

                if timed_out or len(placements) >= soft_cap:
                    break

                if abs(vec[0]) <= axis_tol * max(length, 1e-9):
                    for boundary_x in (0.0, width_mm):
                        tx = boundary_x - a[0]
                        shifted = oriented + np.array([tx, 0.0])
                        min_y = float(np.min(shifted[:, 1]))
                        max_y = float(np.max(shifted[:, 1]))
                        tol = float(getattr(self, "approx_boundary_tolerance_mm", 0.7))
                        start = math.ceil((-min_y - tol) / step) * step
                        stop = math.floor((height_mm - max_y + tol) / step) * step
                        ty = start
                        while ty <= stop + 1e-9:
                            if time.monotonic() >= deadline:
                                timed_out = True
                                break
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -tol
                                and np.max(poly[:, 0]) <= width_mm + tol
                                and np.min(poly[:, 1]) >= -tol
                                and np.max(poly[:, 1]) <= height_mm + tol
                            ):
                                append_candidate(R, t, poly)
                            if len(placements) >= soft_cap:
                                break
                            ty += step
                        if timed_out or len(placements) >= soft_cap:
                            break
                if timed_out or len(placements) >= soft_cap:
                    break
            if timed_out or len(placements) >= soft_cap:
                break

        # 直角落在矩形角的候选必须优先保留；其余候选再均匀抽样。
        placements.sort(
            key=lambda option: (
                -self._popcount(option.corner_mask),
                option.corner_error,
                -option.full_cell_count,
            )
        )
        if len(placements) > self.max_placements_per_piece:
            priority = [option for option in placements if option.corner_mask]
            others = [option for option in placements if not option.corner_mask]
            keep = priority[: self.max_placements_per_piece]
            remaining = self.max_placements_per_piece - len(keep)
            if remaining > 0 and others:
                indices = np.linspace(
                    0, len(others) - 1, min(remaining, len(others)), dtype=np.int32
                )
                keep.extend(others[int(index)] for index in indices)
            placements = keep

        return placements, timed_out

    # ------------------------------------------------------------------
    # 带截止时间的递归装箱
    # ------------------------------------------------------------------
    def solve_one_rectangle_grid_timed(
        self,
        pieces: List[GenericPiece],
        width_mm: float,
        height_mm: float,
        deadline: float,
    ) -> Tuple[List[Tuple[float, Dict[int, Placement]]], int, bool]:
        placement_lists: Dict[int, List[GridPlacementCandidate]] = {}
        generation_timed_out = False
        for index, piece in enumerate(pieces):
            options, timed_out = self.generate_boundary_placements_timed(
                index, piece, width_mm, height_mm, deadline
            )
            generation_timed_out = generation_timed_out or timed_out
            if not options:
                return [], 0, generation_timed_out
            placement_lists[index] = options
            if time.monotonic() >= deadline:
                return [], 0, True

        # 有直角落角候选的碎片优先放；在同类中候选少的先放。
        def order_key(index: int):
            corner_count = sum(1 for option in placement_lists[index] if option.corner_mask)
            return (0 if corner_count else 1, corner_count if corner_count else len(placement_lists[index]), len(placement_lists[index]))

        order = sorted(range(len(pieces)), key=order_key)
        width_px = max(2, int(round(width_mm * self.packing_scale)))
        height_px = max(2, int(round(height_mm * self.packing_scale)))
        total_cells = width_px * height_px
        min_fill_cells = int(math.floor(self.packing_min_fill_ratio * total_cells))
        min_contact_cells = max(
            0, int(round(self.packing_min_contact_mm * self.packing_scale))
        )
        overlap_allow_cells = max(
            1,
            int(round(
                float(self.approx_max_overlap_area_mm2) * self.packing_scale ** 2
            )),
        )

        max_cells_by_piece = {
            idx: max(option.full_cell_count for option in placement_lists[idx])
            for idx in order
        }
        remaining_max = [0] * (len(order) + 1)
        for level in range(len(order) - 1, -1, -1):
            remaining_max[level] = remaining_max[level + 1] + max_cells_by_piece[order[level]]

        nodes = 0
        timed_out = generation_timed_out
        # rank, fill, placements
        found = []
        best_fill = 0.0

        def recurse(
            level: int,
            occupied_interior: int,
            occupied_full: int,
            occupied_count: int,
            occupied_corner_mask: int,
            corner_error_sum: float,
            chosen: Dict[int, GridPlacementCandidate],
        ) -> None:
            nonlocal nodes, timed_out, best_fill
            if timed_out or nodes >= self.packing_node_limit or len(found) >= self.packing_solution_limit:
                return
            if nodes % 64 == 0 and time.monotonic() >= deadline:
                timed_out = True
                return
            if occupied_count + remaining_max[level] < min_fill_cells:
                return

            nodes += 1
            if level == len(order):
                fill_ratio = occupied_count / float(max(total_cells, 1))
                corner_count = self._popcount(occupied_corner_mask)
                if fill_ratio < self.packing_min_fill_ratio:
                    return
                if self.right_angle_prior_enabled and corner_count < self.packing_min_rect_corners:
                    return
                placements = {
                    idx: Placement(
                        piece_index=idx,
                        R=option.R.copy(),
                        t=option.t.copy(),
                        polygon_cart_mm=option.polygon_cart_mm.copy(),
                    )
                    for idx, option in chosen.items()
                }
                polygons = [placements[idx].polygon_cart_mm for idx in sorted(placements)]
                if not self.assembly_connected(polygons):
                    return
                rank = (
                    fill_ratio
                    + 0.055 * corner_count
                    - 0.0008 * corner_error_sum
                )
                found.append((rank, fill_ratio, placements))
                best_fill = max(best_fill, fill_ratio)
                return

            index = order[level]
            valid = []
            for option in placement_lists[index]:
                overlap = self._popcount(option.interior_bits & occupied_interior)
                if overlap > overlap_allow_cells:
                    continue
                contact = 0
                if level > 0:
                    contact = self._popcount(option.border_bits & occupied_full)
                    if min_contact_cells > 0 and contact < min_contact_cells:
                        continue
                new_corner_count = self._popcount(
                    option.corner_mask & (~occupied_corner_mask)
                )
                valid.append((new_corner_count, contact, option))

            valid.sort(
                key=lambda item: (
                    item[0],
                    self._popcount(item[2].corner_mask),
                    item[1],
                    -item[2].corner_error,
                    item[2].full_cell_count,
                ),
                reverse=True,
            )
            if len(valid) > self.max_branch_options:
                valid = valid[: self.max_branch_options]

            for _, _, option in valid:
                if timed_out:
                    return
                new_full = occupied_full | option.full_bits
                new_count = self._popcount(new_full)
                chosen[index] = option
                recurse(
                    level + 1,
                    occupied_interior | option.interior_bits,
                    new_full,
                    new_count,
                    occupied_corner_mask | option.corner_mask,
                    corner_error_sum + (option.corner_error if option.corner_mask else 0.0),
                    chosen,
                )
                chosen.pop(index, None)
                if len(found) >= self.packing_solution_limit or (best_fill >= 0.975 and len(found) >= 2):
                    return

        recurse(0, 0, 0, 0, 0, 0.0, {})
        found.sort(key=lambda item: item[0], reverse=True)
        return [(fill, placements) for _, fill, placements in found], nodes, timed_out

    def approximate_target_metrics(
        self,
        polygons: Sequence[np.ndarray],
        width_mm: float,
        height_mm: float,
    ) -> Tuple[float, float, float, float]:
        """计算相对目标矩形的软几何误差。

        返回：目标内部填充率、模型重叠面积、越界面积、目标空隙面积。
        这些量都只用于给候选排序，不再作为“必须为零”的硬条件。
        """
        scale = max(0.5, float(self.packing_scale))
        width_px = max(2, int(round(width_mm * scale)))
        height_px = max(2, int(round(height_mm * scale)))
        count = np.zeros((height_px, width_px), dtype=np.uint8)
        inside_sum_cells = 0
        polygon_area_mm2 = 0.0

        for polygon in polygons:
            polygon = np.asarray(polygon, dtype=np.float64)
            polygon_area_mm2 += abs(float(cv2.contourArea(
                polygon.astype(np.float32).reshape(-1, 1, 2)
            )))
            mask = np.zeros((height_px, width_px), dtype=np.uint8)
            q = polygon.copy() * scale
            q[:, 1] = height_px - q[:, 1]
            cv2.fillPoly(
                mask,
                [np.round(q).astype(np.int32).reshape(-1, 1, 2)],
                255,
            )
            inside = mask > 0
            inside_sum_cells += int(np.count_nonzero(inside))
            count[inside] = np.minimum(count[inside] + 1, 255)

        union_cells = int(np.count_nonzero(count > 0))
        overlap_cells = int(np.sum(np.maximum(count.astype(np.int16) - 1, 0)))
        target_cells = max(1, width_px * height_px)
        fill_ratio = union_cells / float(target_cells)
        overlap_area_mm2 = overlap_cells / (scale * scale)
        inside_sum_area_mm2 = inside_sum_cells / (scale * scale)
        outside_area_mm2 = max(0.0, polygon_area_mm2 - inside_sum_area_mm2)
        gap_area_mm2 = max(0.0, width_mm * height_mm - union_cells / (scale * scale))
        return fill_ratio, overlap_area_mm2, outside_area_mm2, gap_area_mm2

    # ------------------------------------------------------------------
    # 真正的后台求解实现
    # ------------------------------------------------------------------
    def solve_assembly_timed(
        self,
        pieces: List[GenericPiece],
        warped: np.ndarray,
        timeout_sec: float,
    ) -> Tuple[Optional[AssemblySolution], str, Dict[str, object]]:
        start = time.monotonic()
        deadline = start + max(0.2, float(timeout_sec))
        stats: Dict[str, object] = {
            "rectangles_tested": 0,
            "nodes": 0,
            "geometry_solutions": 0,
        }

        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            return None, "INVALID_PIECE_COUNT", stats

        dimensions = self.rectangle_dimension_candidates(pieces)
        stats["rectangle_candidates"] = len(dimensions)
        if not dimensions:
            return None, "NO_DIMENSION", stats

        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        texture_variance = self.overall_texture_variance(pieces, warped)
        effective_texture_weight = self.texture_weight
        if texture_variance < self.texture_variance_threshold:
            effective_texture_weight *= 0.15
        stats["texture_variance"] = round(float(texture_variance), 3)

        geometry_solutions: List[AssemblySolution] = []
        total_piece_area = float(sum(piece.area_mm2 for piece in pieces))
        timed_out = False

        for _, width_mm, height_mm in dimensions:
            if time.monotonic() >= deadline:
                timed_out = True
                break
            stats["rectangles_tested"] = int(stats["rectangles_tested"]) + 1
            packed, nodes, rect_timeout = self.solve_one_rectangle_grid_timed(
                pieces, width_mm, height_mm, deadline
            )
            stats["nodes"] = int(stats["nodes"]) + int(nodes)
            timed_out = timed_out or rect_timeout

            for fill_ratio_grid, placements in packed:
                polygons = [
                    placements[index].polygon_cart_mm
                    for index in sorted(placements)
                ]
                width, height, bbox_fill_ratio, perimeter_ratio, bbox_gap_area = (
                    self.raster_union_metrics(polygons)
                )
                (
                    target_fill_ratio,
                    overlap_area_mm2,
                    outside_area_mm2,
                    target_gap_area_mm2,
                ) = self.approximate_target_metrics(
                    polygons, width_mm, height_mm
                )
                long_side = max(width, height)
                short_side = min(width, height)
                dim_penalty = self.dimension_penalty(long_side, short_side)
                area_error_ratio = abs(
                    width_mm * height_mm - total_piece_area
                ) / max(total_piece_area, 1.0)
                corner_count, corner_error = self.solution_corner_metrics(
                    polygons, width_mm, height_mm
                )
                connected = self.assembly_connected(polygons)
                if self.right_angle_prior_enabled and corner_count < self.packing_min_rect_corners:
                    continue
                if not connected:
                    continue

                # 软评分：允许少量缝隙、轮廓模型重叠和轻微越界，
                # 但误差越大分数越差。这样不会因为 1~3 mm 的视觉误差
                # 把本来正确的近似矩形直接判为无解。
                geometry_score = (
                    20.0 * (1.0 - target_fill_ratio)
                    + 8.0 * (1.0 - bbox_fill_ratio)
                    + 2.0 * abs(perimeter_ratio - 1.0)
                    + self.approx_gap_penalty_per_mm2 * target_gap_area_mm2
                    + self.approx_overlap_penalty_per_mm2 * overlap_area_mm2
                    + self.approx_outside_penalty_per_mm2 * outside_area_mm2
                    + 0.20 * dim_penalty
                    + 12.0 * area_error_ratio
                    + self.right_angle_missing_corner_penalty * max(0, 4 - corner_count)
                    + self.right_angle_error_penalty * corner_error
                )
                geometry_solutions.append(
                    AssemblySolution(
                        placements=placements,
                        width_mm=width_mm,
                        height_mm=height_mm,
                        fill_ratio=min(fill_ratio_grid, target_fill_ratio),
                        perimeter_ratio=perimeter_ratio,
                        texture_score=0.0,
                        geometry_score=geometry_score,
                        total_score=geometry_score,
                    )
                )

            # 找到高质量方案后优先结束，给机械臂留时间。
            if any(s.fill_ratio >= 0.965 for s in geometry_solutions):
                if len(geometry_solutions) >= 5:
                    break
            if timed_out:
                break

        stats["geometry_solutions"] = len(geometry_solutions)
        if not geometry_solutions:
            return None, "TIMEOUT" if timed_out else "NOT_FOUND", stats

        unique: Dict[Tuple[int, ...], AssemblySolution] = {}
        for solution in geometry_solutions:
            signature = self.solution_signature(solution.placements)
            old = unique.get(signature)
            if old is None or solution.geometry_score < old.geometry_score:
                unique[signature] = solution

        finalists = sorted(
            unique.values(), key=lambda item: item.geometry_score
        )[: self.finalist_count]
        stats["unique_solutions"] = len(unique)
        stats["finalists"] = len(finalists)

        # 剩余时间不足时直接使用几何最优解；避免纹理评分突破总超时。
        for solution in finalists:
            if time.monotonic() >= deadline - 0.08:
                timed_out = True
                break
            solution.texture_score = self.texture_continuity_score(
                pieces, solution.placements, warped_lab
            )
            solution.total_score = (
                solution.geometry_score
                + effective_texture_weight * solution.texture_score
            )

        best = min(finalists, key=lambda item: item.total_score)
        best_polygons = [
            best.placements[index].polygon_cart_mm
            for index in sorted(best.placements)
        ]
        (
            best_target_fill,
            best_overlap_area,
            best_outside_area,
            best_gap_area,
        ) = self.approximate_target_metrics(
            best_polygons, best.width_mm, best.height_mm
        )
        best_corner_count, best_corner_error = self.solution_corner_metrics(
            best_polygons, best.width_mm, best.height_mm
        )
        best_connected = self.assembly_connected(best_polygons)
        stats.update(
            {
                "approximate": True,
                "fill_ratio": round(float(best.fill_ratio), 5),
                "target_fill_ratio": round(float(best_target_fill), 5),
                "overlap_area_mm2": round(float(best_overlap_area), 2),
                "outside_area_mm2": round(float(best_outside_area), 2),
                "gap_area_mm2": round(float(best_gap_area), 2),
                "rectangle_mm": [
                    round(float(best.width_mm), 2),
                    round(float(best.height_mm), 2),
                ],
                "geometry_score": round(float(best.geometry_score), 5),
                "texture_score": round(float(best.texture_score), 3),
                "rectangle_corner_count": int(best_corner_count),
                "rectangle_corner_error": round(float(best_corner_error), 3),
                "assembly_connected": bool(best_connected),
            }
        )
        accepted = (
            best_connected
            and best_corner_count >= self.packing_min_rect_corners
            and best_target_fill >= self.solution_min_accept_fill_ratio
        )
        return best, "FOUND" if accepted else "LOW_QUALITY", stats

    # ------------------------------------------------------------------
    # v8：基于“内部切边配对”的求解器
    # ------------------------------------------------------------------
    @staticmethod
    def _edge_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        v = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
        return math.degrees(math.atan2(float(v[1]), float(v[0])))

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        while angle <= -180.0:
            angle += 360.0
        while angle > 180.0:
            angle -= 360.0
        return angle

    def _edge_match_attempt_params(self) -> Dict[str, float]:
        # solver_worker 在进入每一级搜索前会设置 solver_attempt_index=1..3。
        level = max(0, min(2, int(getattr(self, "solver_attempt_index", 1)) - 1))
        return {
            # 三级只扩大搜索量，不把几何质量放宽到“任意拼法”。
            "min_overlap_mm": [12.0, 10.0, 8.0][level],
            "min_overlap_ratio": [0.55, 0.45, 0.35][level],
            "line_tol_mm": [3.0, 4.0, 5.5][level],
            "angle_tol_deg": [10.0, 14.0, 18.0][level],
            "max_pair_overlap_mm2": [30.0, 55.0, 85.0][level],
            "max_total_overlap_mm2": [45.0, 80.0, 120.0][level],
            "max_outside_mm2": [55.0, 85.0, 120.0][level],
            "min_fill": [0.88, 0.84, 0.80][level],
            "max_unmatched_ratio": [0.16, 0.21, 0.27][level],
            "corner_tol_mm": [6.0, 8.0, 10.0][level],
            "outer_line_tol_mm": [3.5, 4.5, 6.0][level],
            "max_branch": [55, 130, 300][level],
            "max_complete": [80, 240, 650][level],
        }

    def _transform_from_edge_match(
        self,
        moving_piece: GenericPiece,
        moving_edge_index: int,
        fixed_a: np.ndarray,
        fixed_b: np.ndarray,
        alignment_mode: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        source = moving_piece.polygon_cart_mm
        c = source[moving_edge_index]
        d = source[(moving_edge_index + 1) % len(source)]
        fixed_vec = np.asarray(fixed_b, dtype=np.float64) - np.asarray(fixed_a, dtype=np.float64)
        moving_vec = d - c
        lf = float(np.linalg.norm(fixed_vec))
        lm = float(np.linalg.norm(moving_vec))
        if lf < 1e-6 or lm < 1e-6:
            return None

        # 两片共边时，按各自 CCW 边界遍历方向应相反。
        target_angle = math.atan2(-fixed_vec[1], -fixed_vec[0])
        source_angle = math.atan2(moving_vec[1], moving_vec[0])
        R = self.rotation_matrix(target_angle - source_angle)
        rc = R @ c
        rd = R @ d

        if alignment_mode == 0:
            # moving c 对 fixed b
            t = np.asarray(fixed_b, dtype=np.float64) - rc
        elif alignment_mode == 1:
            # moving d 对 fixed a
            t = np.asarray(fixed_a, dtype=np.float64) - rd
        elif alignment_mode == 2:
            # 中点对齐，用于识别边长有系统误差的情况。
            t = 0.5 * (np.asarray(fixed_a) + np.asarray(fixed_b)) - 0.5 * (rc + rd)
        elif alignment_mode == 3:
            # moving 中点落在 fixed 1/3 位置。
            target = np.asarray(fixed_a) + fixed_vec / 3.0
            t = target - 0.5 * (rc + rd)
        else:
            # moving 中点落在 fixed 2/3 位置。
            target = np.asarray(fixed_a) + 2.0 * fixed_vec / 3.0
            t = target - 0.5 * (rc + rd)

        polygon = self.transform_polygon(moving_piece, R, t)
        return R, t, polygon

    def _placed_edge_match_candidates(
        self,
        piece_index: int,
        piece: GenericPiece,
        placed: Dict[int, Placement],
        params: Dict[str, float],
        deadline: float,
    ) -> List[Tuple[float, float, Placement]]:
        candidates: List[Tuple[float, float, Placement]] = []
        seen = set()

        for fixed_index, fixed_placement in placed.items():
            fixed_polygon = fixed_placement.polygon_cart_mm
            for _, fixed_a, fixed_b in self.polygon_edges(fixed_polygon):
                lf = float(np.linalg.norm(fixed_b - fixed_a))
                if lf < 5.0:
                    continue
                for moving_edge_index, source_a, source_b in self.polygon_edges(
                    piece.polygon_cart_mm
                ):
                    if time.monotonic() >= deadline:
                        return candidates
                    lm = float(np.linalg.norm(source_b - source_a))
                    if lm < 5.0:
                        continue
                    # 完全不相近的边不必尝试；仍允许一条长边被两条短边分段覆盖。
                    if max(lf, lm) / max(min(lf, lm), 1e-6) > 2.8:
                        continue

                    required_overlap = max(
                        params["min_overlap_mm"],
                        params["min_overlap_ratio"] * min(lf, lm),
                    )
                    for alignment_mode in range(5):
                        transformed = self._transform_from_edge_match(
                            piece,
                            moving_edge_index,
                            fixed_a,
                            fixed_b,
                            alignment_mode,
                        )
                        if transformed is None:
                            continue
                        R, t, polygon = transformed
                        moved_a = polygon[moving_edge_index]
                        moved_b = polygon[(moving_edge_index + 1) % len(polygon)]
                        seam_overlap, _, _ = self.segment_overlap_on_line(
                            fixed_a,
                            fixed_b,
                            moved_a,
                            moved_b,
                            params["angle_tol_deg"],
                            params["line_tol_mm"],
                        )
                        if seam_overlap + 1e-6 < required_overlap:
                            continue

                        total_overlap = 0.0
                        rejected = False
                        for other in placed.values():
                            area = self.pair_overlap_area_mm2(
                                polygon, other.polygon_cart_mm
                            )
                            if area > params["max_pair_overlap_mm2"]:
                                rejected = True
                                break
                            total_overlap += area
                        if rejected or total_overlap > params["max_total_overlap_mm2"]:
                            continue

                        all_points = np.vstack(
                            [polygon]
                            + [other.polygon_cart_mm for other in placed.values()]
                        )
                        span = all_points.max(axis=0) - all_points.min(axis=0)
                        # 官方尺寸上限外只给视觉误差留少量余量。
                        if max(span) > self.rect_long_max_mm + 14.0:
                            continue
                        if min(span) > self.rect_short_max_mm + 14.0:
                            continue

                        angle = math.degrees(math.atan2(R[1, 0], R[0, 0]))
                        centroid = polygon.mean(axis=0)
                        key = (
                            int(round(centroid[0] / 1.5)),
                            int(round(centroid[1] / 1.5)),
                            int(round(angle / 2.0)),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        # seam 越长越好；重叠越小越好。
                        quality = seam_overlap - 0.35 * total_overlap
                        candidates.append(
                            (
                                -quality,
                                total_overlap,
                                Placement(
                                    piece_index=piece_index,
                                    R=R.copy(),
                                    t=t.copy(),
                                    polygon_cart_mm=polygon.copy(),
                                ),
                            )
                        )

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[: int(params["max_branch"])]

    @staticmethod
    def _rotate_assembly_90(
        placements: Dict[int, Placement], clockwise: bool = False
    ) -> Dict[int, Placement]:
        angle = -math.pi / 2.0 if clockwise else math.pi / 2.0
        c = math.cos(angle)
        s = math.sin(angle)
        G = np.array([[c, -s], [s, c]], dtype=np.float64)
        result: Dict[int, Placement] = {}
        for index, placement in placements.items():
            result[index] = Placement(
                piece_index=index,
                R=G @ placement.R,
                t=G @ placement.t,
                polygon_cart_mm=(G @ placement.polygon_cart_mm.T).T,
            )
        return result

    @staticmethod
    def _shift_assembly(
        placements: Dict[int, Placement], shift: np.ndarray
    ) -> Dict[int, Placement]:
        shift = np.asarray(shift, dtype=np.float64)
        return {
            index: Placement(
                piece_index=index,
                R=placement.R.copy(),
                t=placement.t + shift,
                polygon_cart_mm=placement.polygon_cart_mm + shift,
            )
            for index, placement in placements.items()
        }

    def _strict_dimension_candidates(
        self, polygons: Sequence[np.ndarray], total_area: float
    ) -> List[Tuple[float, float]]:
        points = np.vstack(polygons)
        span = points.max(axis=0) - points.min(axis=0)
        bw = float(span[0])
        bh = float(span[1])
        candidates = []
        for width in np.arange(self.rect_long_min_mm, self.rect_long_max_mm + 0.1, 1.0):
            for height in np.arange(self.rect_short_min_mm, self.rect_short_max_mm + 0.1, 1.0):
                area_error = abs(width * height - total_area) / max(total_area, 1.0)
                # 拼图保证完整覆盖目标矩形；视觉面积误差可以有，但不能达到 30% 以上。
                # 这里限制为 16%，避免 8034mm² 的碎片被塞进 118x89=10502mm² 的伪矩形。
                if area_error > 0.16:
                    continue
                bbox_error = abs(width - bw) + abs(height - bh)
                score = bbox_error + 45.0 * area_error
                candidates.append((score, float(width), float(height)))
        candidates.sort(key=lambda item: item[0])
        return [(w, h) for _, w, h in candidates[:12]]

    @staticmethod
    def _edge_samples(a: np.ndarray, b: np.ndarray, step_mm: float = 1.0) -> np.ndarray:
        length = float(np.linalg.norm(b - a))
        count = max(2, int(math.ceil(length / max(step_mm, 0.25))) + 1)
        values = np.linspace(0.0, 1.0, count)
        return a[None, :] + values[:, None] * (b - a)[None, :]

    def _outer_edge_coverage(
        self,
        a: np.ndarray,
        b: np.ndarray,
        width: float,
        height: float,
        tolerance: float,
    ) -> float:
        samples = self._edge_samples(a, b, 1.0)
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            return 0.0
        coverages = []
        conditions = [
            np.abs(samples[:, 0]) <= tolerance,
            np.abs(samples[:, 0] - width) <= tolerance,
            np.abs(samples[:, 1]) <= tolerance,
            np.abs(samples[:, 1] - height) <= tolerance,
        ]
        for condition in conditions:
            coverages.append(length * float(np.mean(condition)))
        return max(coverages)

    def _internal_edge_coverage(
        self,
        piece_index: int,
        a: np.ndarray,
        b: np.ndarray,
        polygons: Sequence[np.ndarray],
        params: Dict[str, float],
    ) -> float:
        length = float(np.linalg.norm(b - a))
        intervals: List[Tuple[float, float]] = []
        direction = b - a
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return 0.0
        unit = direction / norm
        for other_index, other in enumerate(polygons):
            if other_index == piece_index:
                continue
            for _, c, d in self.polygon_edges(other):
                overlap, p0, p1 = self.segment_overlap_on_line(
                    a,
                    b,
                    c,
                    d,
                    params["angle_tol_deg"],
                    params["line_tol_mm"],
                )
                if overlap <= 0.0 or p0 is None or p1 is None:
                    continue
                x0 = float(np.dot(p0 - a, unit))
                x1 = float(np.dot(p1 - a, unit))
                intervals.append((max(0.0, min(x0, x1)), min(length, max(x0, x1))))
        if not intervals:
            return 0.0
        intervals.sort()
        covered = 0.0
        start, end = intervals[0]
        for low, high in intervals[1:]:
            if low <= end + 1e-6:
                end = max(end, high)
            else:
                covered += max(0.0, end - start)
                start, end = low, high
        covered += max(0.0, end - start)
        return min(length, covered)

    def _seam_graph_metrics(
        self,
        polygons: Sequence[np.ndarray],
        params: Dict[str, float],
    ) -> Dict[str, object]:
        """只用真实的反向共线切边建立邻接图。

        与 assembly_connected() 不同，单纯模型重叠不再算作“连接”。
        每对碎片至少要有一组足够长的共线边，才在拼接图中连边。
        """
        count = len(polygons)
        graph = [[] for _ in range(count)]
        pair_seams = []
        seam_count = 0
        total_seam_overlap = 0.0

        for i in range(count):
            for j in range(i + 1, count):
                best_overlap = 0.0
                best_pair = None
                for edge_i, a, b in self.polygon_edges(polygons[i]):
                    len_i = float(np.linalg.norm(b - a))
                    if len_i < 1e-6:
                        continue
                    for edge_j, c, d in self.polygon_edges(polygons[j]):
                        len_j = float(np.linalg.norm(d - c))
                        if len_j < 1e-6:
                            continue
                        overlap, _, _ = self.segment_overlap_on_line(
                            a, b, c, d,
                            params["angle_tol_deg"],
                            params["line_tol_mm"],
                        )
                        required = max(
                            6.0,
                            0.28 * min(len_i, len_j),
                        )
                        if overlap >= required and overlap > best_overlap:
                            best_overlap = float(overlap)
                            best_pair = (int(edge_i), int(edge_j))
                if best_pair is not None:
                    graph[i].append(j)
                    graph[j].append(i)
                    seam_count += 1
                    total_seam_overlap += best_overlap
                    pair_seams.append({
                        "pieces": [int(i), int(j)],
                        "edges": [int(best_pair[0]), int(best_pair[1])],
                        "overlap_mm": float(best_overlap),
                    })

        if count <= 1:
            connected = True
        else:
            seen = {0}
            stack = [0]
            while stack:
                current = stack.pop()
                for other in graph[current]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            connected = len(seen) == count

        return {
            "connected": bool(connected),
            "seam_count": int(seam_count),
            "required_seam_count": max(0, count - 1),
            "total_seam_overlap_mm": float(total_seam_overlap),
            "pair_seams": pair_seams,
        }

    def _edge_assembly_metrics(
        self,
        placements: Dict[int, Placement],
        width: float,
        height: float,
        params: Dict[str, float],
    ) -> Dict[str, object]:
        ordered_indices = sorted(placements)
        polygons = [placements[index].polygon_cart_mm for index in ordered_indices]
        scale = 1.0
        margin = 20
        canvas_w = int(math.ceil(width + 2 * margin + 2))
        canvas_h = int(math.ceil(height + 2 * margin + 2))
        target = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.rectangle(
            target,
            (margin, margin),
            (margin + int(round(width)), margin + int(round(height))),
            255,
            -1,
        )
        masks = []
        for polygon in polygons:
            points = np.round(polygon + np.array([margin, margin])).astype(np.int32)
            mask = np.zeros_like(target)
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
            masks.append(mask)
        if not masks:
            return {}
        stack = np.stack([(mask > 0).astype(np.uint8) for mask in masks], axis=0)
        counts = np.sum(stack, axis=0)
        union = counts > 0
        target_bool = target > 0
        union_area = int(np.count_nonzero(union))
        target_area = int(np.count_nonzero(target_bool))
        inside_area = int(np.count_nonzero(union & target_bool))
        outside_area = int(np.count_nonzero(union & (~target_bool)))
        overlap_area = int(np.sum(np.maximum(counts.astype(np.int32) - 1, 0)))
        gap_area = int(np.count_nonzero(target_bool & (~union)))
        fill = inside_area / float(max(target_area, 1))

        corner_points = [
            np.array([0.0, 0.0]),
            np.array([width, 0.0]),
            np.array([width, height]),
            np.array([0.0, height]),
        ]
        vertices = np.vstack(polygons)
        corner_distances = [
            float(np.min(np.linalg.norm(vertices - corner, axis=1)))
            for corner in corner_points
        ]
        corner_count = sum(distance <= params["corner_tol_mm"] for distance in corner_distances)

        missing_outer = 0
        unmatched_length = 0.0
        total_edge_length = 0.0
        outer_contact_lengths = []
        for local_index, polygon in enumerate(polygons):
            best_outer = 0.0
            for _, a, b in self.polygon_edges(polygon):
                edge_length = float(np.linalg.norm(b - a))
                total_edge_length += edge_length
                outer = self._outer_edge_coverage(
                    a, b, width, height, params["outer_line_tol_mm"]
                )
                internal = self._internal_edge_coverage(
                    local_index, a, b, polygons, params
                )
                best_outer = max(best_outer, outer)
                unmatched_length += max(0.0, edge_length - max(outer, internal))
            outer_contact_lengths.append(best_outer)
            if best_outer < max(8.0, 0.45 * self.detected_min_edge_mm):
                missing_outer += 1

        unmatched_ratio = unmatched_length / max(total_edge_length, 1.0)
        seam_graph = self._seam_graph_metrics(polygons, params)
        connected = bool(seam_graph["connected"])
        return {
            "fill": fill,
            "overlap": float(overlap_area),
            "outside": float(outside_area),
            "gap": float(gap_area),
            "corner_count": int(corner_count),
            "corner_error": float(sum(corner_distances)),
            "corner_distances": corner_distances,
            "missing_outer": int(missing_outer),
            "outer_contact_lengths": outer_contact_lengths,
            "unmatched_length": float(unmatched_length),
            "unmatched_ratio": float(unmatched_ratio),
            "connected": bool(connected),
            "seam_count": int(seam_graph["seam_count"]),
            "required_seam_count": int(seam_graph["required_seam_count"]),
            "total_seam_overlap_mm": float(seam_graph["total_seam_overlap_mm"]),
            "pair_seams": seam_graph["pair_seams"],
            "union_area": float(union_area),
        }

    def _evaluate_complete_edge_assembly(
        self,
        placements: Dict[int, Placement],
        total_area: float,
        params: Dict[str, float],
    ) -> List[Tuple[float, AssemblySolution, Dict[str, object]]]:
        evaluated = []
        for rotated in (placements, self._rotate_assembly_90(placements)):
            polygons0 = [rotated[index].polygon_cart_mm for index in sorted(rotated)]
            min_xy = np.vstack(polygons0).min(axis=0)
            normalized = self._shift_assembly(rotated, -min_xy)
            polygons = [normalized[index].polygon_cart_mm for index in sorted(normalized)]
            points = np.vstack(polygons)
            span = points.max(axis=0) - points.min(axis=0)
            bw, bh = float(span[0]), float(span[1])
            for width, height in self._strict_dimension_candidates(polygons, total_area):
                # 只测试左/中/右、下/中/上的少量对齐方式。
                dx_values = sorted(set([0.0, width - bw, 0.5 * (width - bw)]))
                dy_values = sorted(set([0.0, height - bh, 0.5 * (height - bh)]))
                for dx in dx_values:
                    for dy in dy_values:
                        shifted = self._shift_assembly(
                            normalized, np.array([dx, dy], dtype=np.float64)
                        )
                        metrics = self._edge_assembly_metrics(
                            shifted, width, height, params
                        )
                        if not metrics:
                            continue
                        if not metrics["connected"]:
                            continue
                        if metrics["seam_count"] < metrics["required_seam_count"]:
                            continue
                        if metrics["missing_outer"] > 0:
                            continue
                        # 目标是矩形，四个角都必须由碎片顶点支撑；不再接受只有 3 个角的伪解。
                        if metrics["corner_count"] < 4:
                            continue
                        if metrics["fill"] < params["min_fill"]:
                            continue
                        if metrics["overlap"] > params["max_total_overlap_mm2"]:
                            continue
                        if metrics["outside"] > params["max_outside_mm2"]:
                            continue
                        if metrics["unmatched_ratio"] > params["max_unmatched_ratio"]:
                            continue

                        area_error = abs(width * height - total_area) / max(total_area, 1.0)
                        score = (
                            0.045 * metrics["gap"]
                            + 0.32 * metrics["overlap"]
                            + 0.28 * metrics["outside"]
                            + 2.8 * metrics["unmatched_length"]
                            + 55.0 * max(0, 4 - metrics["corner_count"])
                            + 1.4 * metrics["corner_error"]
                            + 95.0 * area_error
                            + 140.0 * metrics["missing_outer"]
                        )
                        solution = AssemblySolution(
                            placements=shifted,
                            width_mm=float(width),
                            height_mm=float(height),
                            fill_ratio=float(metrics["fill"]),
                            perimeter_ratio=1.0,
                            texture_score=0.0,
                            geometry_score=float(score),
                            total_score=float(score),
                        )
                        evaluated.append((score, solution, metrics))
        evaluated.sort(key=lambda item: item[0])
        return evaluated[:12]

    def solve_assembly_timed(
        self,
        pieces: List[GenericPiece],
        warped: np.ndarray,
        timeout_sec: float,
    ) -> Tuple[Optional[AssemblySolution], str, Dict[str, object]]:
        """按内部切边配对拼接，而不是把四片独立吸附到矩形边界。"""
        start = time.monotonic()
        deadline = start + max(0.3, float(timeout_sec))
        params = self._edge_match_attempt_params()
        stats: Dict[str, object] = {
            "solver": "edge_matching_v8",
            "nodes": 0,
            "complete_assemblies": 0,
            "evaluated_solutions": 0,
        }
        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            return None, "INVALID_PIECE_COUNT", stats

        total_area = float(sum(piece.area_mm2 for piece in pieces))
        root_index = max(range(len(pieces)), key=lambda idx: pieces[idx].area_mm2)
        root = pieces[root_index]
        all_indices = set(range(len(pieces)))
        best_candidates: List[Tuple[float, AssemblySolution, Dict[str, object]]] = []
        seen_complete = set()
        nodes = 0
        complete_count = 0
        timed_out = False

        def complete_signature(placements: Dict[int, Placement]):
            values = []
            for index in sorted(placements):
                placement = placements[index]
                centroid = placement.polygon_cart_mm.mean(axis=0)
                angle = math.degrees(math.atan2(placement.R[1, 0], placement.R[0, 0]))
                values.extend(
                    [
                        int(round(centroid[0] / 2.0)),
                        int(round(centroid[1] / 2.0)),
                        int(round(angle / 3.0)),
                    ]
                )
            return tuple(values)

        def recurse(placed: Dict[int, Placement]) -> None:
            nonlocal nodes, complete_count, timed_out, best_candidates
            if timed_out:
                return
            if nodes >= self.packing_node_limit:
                return
            if nodes % 32 == 0 and time.monotonic() >= deadline:
                timed_out = True
                return
            nodes += 1
            if len(placed) == len(pieces):
                signature = complete_signature(placed)
                if signature in seen_complete:
                    return
                seen_complete.add(signature)
                complete_count += 1
                evaluated = self._evaluate_complete_edge_assembly(
                    placed, total_area, params
                )
                best_candidates.extend(evaluated)
                best_candidates.sort(key=lambda item: item[0])
                best_candidates = best_candidates[:16]
                if complete_count >= int(params["max_complete"]):
                    return
                return

            remaining = list(all_indices - set(placed))
            piece_options = []
            for index in remaining:
                options = self._placed_edge_match_candidates(
                    index, pieces[index], placed, params, deadline
                )
                if options:
                    piece_options.append((len(options), index, options))
            if not piece_options:
                return
            piece_options.sort(key=lambda item: item[0])
            _, next_index, options = piece_options[0]
            for _, _, candidate in options:
                if timed_out or time.monotonic() >= deadline:
                    timed_out = True
                    return
                placed[next_index] = candidate
                recurse(placed)
                del placed[next_index]
                if complete_count >= int(params["max_complete"]):
                    return

        # 枚举根片每条边作为一个外边方向。全局平移/旋转自由度被固定后，
        # 后续只通过切边配对增长拼图，不再允许四片独立在矩形中任意滑动。
        for edge_index, a, b in self.polygon_edges(root.polygon_cart_mm):
            if time.monotonic() >= deadline:
                timed_out = True
                break
            edge = b - a
            if float(np.linalg.norm(edge)) < self.detected_min_edge_mm:
                continue
            angle = -math.atan2(edge[1], edge[0])
            R = self.rotation_matrix(angle)
            rotated_a = R @ a
            t = -rotated_a
            polygon = self.transform_polygon(root, R, t)
            placed = {
                root_index: Placement(
                    piece_index=root_index,
                    R=R.copy(),
                    t=t.copy(),
                    polygon_cart_mm=polygon.copy(),
                )
            }
            recurse(placed)

        stats["nodes"] = int(nodes)
        stats["complete_assemblies"] = int(complete_count)
        stats["evaluated_solutions"] = int(len(best_candidates))
        if not best_candidates:
            return None, "TIMEOUT" if timed_out else "NOT_FOUND", stats

        # 纹理只在几何前几名上计算。
        finalists = best_candidates[: min(8, len(best_candidates))]
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        texture_variance = self.overall_texture_variance(pieces, warped)
        texture_weight = self.texture_weight if texture_variance >= self.texture_variance_threshold else 0.0
        for _, solution, _ in finalists:
            if time.monotonic() >= deadline - 0.05:
                break
            solution.texture_score = self.texture_continuity_score(
                pieces, solution.placements, warped_lab
            )
            solution.total_score = solution.geometry_score + texture_weight * solution.texture_score
        finalists.sort(key=lambda item: item[1].total_score)
        _, best, metrics = finalists[0]
        stats.update(
            {
                "rectangle_mm": [round(best.width_mm, 2), round(best.height_mm, 2)],
                "fill_ratio": round(best.fill_ratio, 5),
                "geometry_score": round(best.geometry_score, 4),
                "texture_score": round(best.texture_score, 3),
                "overlap_area_mm2": round(float(metrics["overlap"]), 2),
                "outside_area_mm2": round(float(metrics["outside"]), 2),
                "gap_area_mm2": round(float(metrics["gap"]), 2),
                "unmatched_edge_ratio": round(float(metrics["unmatched_ratio"]), 4),
                "unmatched_edge_length_mm": round(float(metrics["unmatched_length"]), 2),
                "rectangle_corner_count": int(metrics["corner_count"]),
                "corner_distances_mm": [round(float(v), 2) for v in metrics["corner_distances"]],
                "all_pieces_have_outer_edge": bool(metrics["missing_outer"] == 0),
                "assembly_connected": bool(metrics["connected"]),
                "seam_count": int(metrics.get("seam_count", 0)),
                "required_seam_count": int(metrics.get("required_seam_count", max(0, len(pieces) - 1))),
                "total_seam_overlap_mm": round(float(metrics.get("total_seam_overlap_mm", 0.0)), 2),
                "pair_seams": metrics.get("pair_seams", []),
            }
        )

        long_side = max(float(best.width_mm), float(best.height_mm))
        short_side = min(float(best.width_mm), float(best.height_mm))
        area_error = abs(best.width_mm * best.height_mm - total_area) / max(total_area, 1.0)
        overlap_limit = min(120.0, 0.015 * max(total_area, 1.0))
        accepted = (
            self.rect_long_min_mm <= long_side <= self.rect_long_max_mm
            and self.rect_short_min_mm <= short_side <= self.rect_short_max_mm
            and area_error <= 0.16
            and float(metrics["fill"]) >= max(0.82, float(self.solution_min_accept_fill_ratio))
            and float(metrics["overlap"]) <= overlap_limit
            and float(metrics["outside"]) <= 120.0
            and float(metrics["unmatched_ratio"]) <= 0.27
            and int(metrics["corner_count"]) == 4
            and int(metrics["missing_outer"]) == 0
            and bool(metrics["connected"])
            and int(metrics.get("seam_count", 0)) >= max(0, len(pieces) - 1)
        )
        stats["accepted"] = bool(accepted)
        stats["area_error_ratio"] = round(float(area_error), 5)
        stats["overlap_accept_limit_mm2"] = round(float(overlap_limit), 2)
        if not accepted:
            # 关键修复：v8 在这里无条件返回 FOUND，导致 fill=0.687 的伪解也被执行。
            return None, "LOW_QUALITY", stats
        return best, "FOUND", stats


    @staticmethod
    def _remap_solution_from_order(solution, order):
        """把按重排碎片索引求出的方案映射回原始碎片索引。"""
        if solution is None:
            return None
        mapped = {}
        for local_index, placement in solution.placements.items():
            original_index = int(order[int(local_index)])
            mapped[original_index] = Placement(
                piece_index=original_index,
                R=np.asarray(placement.R, dtype=np.float64).copy(),
                t=np.asarray(placement.t, dtype=np.float64).copy(),
                polygon_cart_mm=np.asarray(
                    placement.polygon_cart_mm, dtype=np.float64
                ).copy(),
            )
        return AssemblySolution(
            placements=mapped,
            width_mm=float(solution.width_mm),
            height_mm=float(solution.height_mm),
            fill_ratio=float(solution.fill_ratio),
            perimeter_ratio=float(solution.perimeter_ratio),
            texture_score=float(solution.texture_score),
            geometry_score=float(solution.geometry_score),
            total_score=float(solution.total_score),
        )

    def _continuous_piece_orders(self, count):
        """返回确定性的多样化碎片顺序，优先轮换根片，再覆盖所有排列。"""
        base = tuple(range(count))
        if count <= 1 or not self.solver_diversify_piece_order:
            return [base]
        orders = []
        # 先轮换和反向轮换，使不同碎片尽快成为根片。
        for shift in range(count):
            cyc = base[shift:] + base[:shift]
            orders.append(cyc)
            orders.append(tuple(reversed(cyc)))
        # 最多 4 片，完整排列不超过 24 个。
        for item in itertools.permutations(base):
            if item not in orders:
                orders.append(item)
        return orders

    @staticmethod
    def _scale_int_list(values, scale, minimum=1):
        return [max(minimum, int(round(float(v) * scale))) for v in values]

    def solver_worker(
        self,
        signature,
        pieces: List[GenericPiece],
        warped: np.ndarray,
    ) -> None:
        """持续深化搜索，失败后自动换排列、扩大候选并继续。

        与旧版不同，三级搜索全部失败不会结束线程。下一轮会：
        1. 改变碎片排列，使不同碎片成为联合优化的根片；
        2. 增大边配对初值、联合优化候选数和 least_squares 迭代数；
        3. 适度增加单次尝试时间。

        最终验收阈值不随轮次放宽，因此长时间搜索不会靠接受伪解结束。
        """
        started_all = time.monotonic()
        solution = None
        final_status = "NOT_FOUND"
        attempt_records = []
        round_records = []
        piece_orders = self._continuous_piece_orders(len(pieces))
        terminal_statuses = {
            "INVALID_PIECE_COUNT", "NOT_READY", "SCIPY_MISSING"
        }

        original = {
            "solver_timeout_sec": self.solver_timeout_sec,
            "max_rectangle_candidates": self.max_rectangle_candidates,
            "packing_grid_mm": self.packing_grid_mm,
            "packing_node_limit": self.packing_node_limit,
            "max_placements_per_piece": self.max_placements_per_piece,
            "max_branch_options": self.max_branch_options,
            "packing_min_fill_ratio": self.packing_min_fill_ratio,
            "packing_min_contact_mm": self.packing_min_contact_mm,
            "rect_area_tolerance_ratio": self.rect_area_tolerance_ratio,
            "packing_min_rect_corners": self.packing_min_rect_corners,
            "joint_initial_assemblies_per_level": list(
                getattr(self, "joint_initial_assemblies_per_level", [])
            ),
            "joint_pair_options_per_tree_edge": list(
                getattr(self, "joint_pair_options_per_tree_edge", [])
            ),
            "joint_optimize_candidates_per_level": list(
                getattr(self, "joint_optimize_candidates_per_level", [])
            ),
            "joint_max_nfev_per_level": list(
                getattr(self, "joint_max_nfev_per_level", [])
            ),
        }

        self.solver_cancel_event.clear()
        round_index = 0
        try:
            while not rospy.is_shutdown():
                if self.solver_cancel_event.is_set():
                    final_status = "CANCELLED"
                    break
                elapsed_total = time.monotonic() - started_all
                if (
                    self.solver_total_search_timeout_sec > 0.0
                    and elapsed_total >= self.solver_total_search_timeout_sec
                ):
                    final_status = "SEARCH_TIMEOUT"
                    break
                if (
                    self.solver_max_search_rounds > 0
                    and round_index >= self.solver_max_search_rounds
                ):
                    final_status = "SEARCH_ROUND_LIMIT"
                    break

                # 每 2 轮扩大一次搜索规模；碎片排列仍在全部排列中轮换。
                # 不等待 24 个排列全部跑完才加深，否则第一次扩容会过慢。
                order_cycle = round_index // 2
                size_scale = min(
                    self.solver_round_scale_cap,
                    1.0 + self.solver_round_growth * order_cycle,
                )
                timeout_scale = min(
                    self.solver_round_timeout_scale_cap,
                    1.0 + self.solver_round_timeout_growth * order_cycle,
                )
                round_started = time.monotonic()
                round_attempts = []

                with self.solver_lock:
                    self.solver_round_index = round_index + 1

                for attempt in range(self.solver_attempt_count):
                    if rospy.is_shutdown() or self.solver_cancel_event.is_set():
                        final_status = "CANCELLED"
                        break
                    elapsed_total = time.monotonic() - started_all
                    remaining = None
                    if self.solver_total_search_timeout_sec > 0.0:
                        remaining = self.solver_total_search_timeout_sec - elapsed_total
                        if remaining <= 0.05:
                            final_status = "SEARCH_TIMEOUT"
                            break

                    order_index = (
                        round_index * self.solver_attempt_count + attempt
                    ) % len(piece_orders)
                    order = piece_orders[order_index]
                    ordered_pieces = [pieces[i] for i in order]

                    with self.solver_lock:
                        self.solver_attempt_index = attempt + 1
                        self.solver_round_piece_order = [int(i) + 1 for i in order]

                    base_timeout = self.solver_attempt_timeouts[attempt]
                    attempt_timeout = max(0.5, base_timeout * timeout_scale)
                    if remaining is not None:
                        attempt_timeout = max(0.25, min(attempt_timeout, remaining))

                    self.solver_timeout_sec = attempt_timeout
                    self.max_rectangle_candidates = max(
                        1,
                        int(round(
                            self.solver_attempt_rect_candidates[attempt]
                            * size_scale
                        )),
                    )
                    self.packing_grid_mm = self.solver_attempt_grid_mm[attempt]
                    self.packing_node_limit = max(
                        1000,
                        int(round(
                            self.solver_attempt_node_limits[attempt]
                            * size_scale
                        )),
                    )
                    self.max_placements_per_piece = max(
                        20,
                        int(round(
                            self.solver_attempt_placements_per_piece[attempt]
                            * size_scale
                        )),
                    )
                    self.max_branch_options = max(
                        20,
                        int(round(
                            self.solver_attempt_branch_options[attempt]
                            * size_scale
                        )),
                    )
                    self.packing_min_fill_ratio = self.solver_attempt_min_fill_ratio[attempt]
                    self.packing_min_contact_mm = self.solver_attempt_min_contact_mm[attempt]
                    self.rect_area_tolerance_ratio = self.solver_attempt_area_tolerance_ratio[attempt]
                    self.packing_min_rect_corners = self.solver_attempt_min_rect_corners[attempt]

                    # v11 联合优化器的主要搜索宽度也按轮次扩大。
                    if original["joint_initial_assemblies_per_level"]:
                        self.joint_initial_assemblies_per_level = self._scale_int_list(
                            original["joint_initial_assemblies_per_level"],
                            size_scale,
                        )
                    if original["joint_pair_options_per_tree_edge"]:
                        self.joint_pair_options_per_tree_edge = self._scale_int_list(
                            original["joint_pair_options_per_tree_edge"],
                            min(size_scale, 2.2),
                        )
                    if original["joint_optimize_candidates_per_level"]:
                        self.joint_optimize_candidates_per_level = self._scale_int_list(
                            original["joint_optimize_candidates_per_level"],
                            size_scale,
                        )
                    if original["joint_max_nfev_per_level"]:
                        self.joint_max_nfev_per_level = self._scale_int_list(
                            original["joint_max_nfev_per_level"],
                            min(size_scale, 2.0),
                            minimum=20,
                        )

                    attempt_started = time.monotonic()
                    try:
                        current_solution, status, stats = self.solve_assembly_timed(
                            ordered_pieces, warped, attempt_timeout
                        )
                    except Exception as exc:
                        rospy.logerr(
                            "Continuous solver round %d attempt %d/%d error: %s",
                            round_index + 1,
                            attempt + 1,
                            self.solver_attempt_count,
                            exc,
                        )
                        current_solution = None
                        status = "ERROR"
                        stats = {"error": str(exc)}

                    attempt_elapsed = time.monotonic() - attempt_started
                    record = {
                        "round": round_index + 1,
                        "attempt": attempt + 1,
                        "status": status,
                        "seconds": round(float(attempt_elapsed), 3),
                        "timeout_sec": round(float(attempt_timeout), 3),
                        "piece_order": [int(i) + 1 for i in order],
                        "size_scale": round(float(size_scale), 3),
                        "timeout_scale": round(float(timeout_scale), 3),
                        "stats": stats,
                    }
                    attempt_records.append(record)
                    round_attempts.append(record)

                    # 在运行中发布最新进度，rqt/rostopic 可看到轮次和累计时间。
                    with self.solver_lock:
                        self.cached_stats = {
                            "continuous": True,
                            "round": round_index + 1,
                            "attempt": attempt + 1,
                            "piece_order": [int(i) + 1 for i in order],
                            "elapsed_total_sec": round(
                                float(time.monotonic() - started_all), 3
                            ),
                            "last_attempt": record,
                        }

                    rospy.loginfo(
                        "Continuous solve R%d A%d/%d order=%s status=%s "
                        "time=%.3fs scale=%.2f",
                        round_index + 1,
                        attempt + 1,
                        self.solver_attempt_count,
                        [int(i) + 1 for i in order],
                        status,
                        attempt_elapsed,
                        size_scale,
                    )

                    if current_solution is not None and status == "FOUND":
                        solution = self._remap_solution_from_order(
                            current_solution, order
                        )
                        final_status = "FOUND"
                        break
                    final_status = status
                    if status in terminal_statuses:
                        break

                round_records.append(
                    {
                        "round": round_index + 1,
                        "seconds": round(float(time.monotonic() - round_started), 3),
                        "attempts": round_attempts,
                    }
                )
                if solution is not None or final_status in terminal_statuses:
                    break
                if final_status in ("CANCELLED", "SEARCH_TIMEOUT", "SEARCH_ROUND_LIMIT"):
                    break

                round_index += 1
                if not self.solver_continuous_search:
                    break
                if self.solver_round_pause_sec > 0.0:
                    self.solver_cancel_event.wait(self.solver_round_pause_sec)

        finally:
            for key, value in original.items():
                if key.startswith("joint_"):
                    if value:
                        setattr(self, key, list(value))
                else:
                    setattr(self, key, value)

        elapsed_all = time.monotonic() - started_all
        combined_stats = {
            "continuous": bool(self.solver_continuous_search),
            "rounds_used": len(round_records),
            "attempts_used": len(attempt_records),
            "total_seconds": round(float(elapsed_all), 3),
            # 防止状态消息无限增大，只保留最近 12 次尝试和最近 4 轮。
            "attempts": attempt_records[-12:],
            "rounds": round_records[-4:],
        }
        if attempt_records:
            combined_stats["last_attempt"] = attempt_records[-1]

        with self.solver_lock:
            self.cached_signature = signature
            self.cached_solution = solution
            self.cached_status = final_status
            self.cached_finished_at = time.monotonic()
            self.cached_solve_sec = elapsed_all
            self.cached_stats = combined_stats
            self.solver_running_signature = None
            self.solver_attempt_index = 0
            self.solver_round_index = 0
            self.solver_round_piece_order = []
            self.solver_thread = None

        rospy.loginfo(
            "Continuous solve finished: status=%s total=%.3fs rounds=%d attempts=%d",
            final_status,
            elapsed_all,
            len(round_records),
            len(attempt_records),
        )

    def solve_assembly(
        self, pieces: List[GenericPiece], warped: np.ndarray
    ) -> Optional[AssemblySolution]:
        """非阻塞入口：相机回调只管理缓存和启动后台线程。"""
        self.last_piece_count = len(pieces)
        self.last_piece_summaries = [
            "P{} A={:.0f}mm2 edges={}".format(
                piece.piece_id, piece.area_mm2, len(piece.polygon_cart_mm)
            )
            for piece in pieces
        ]

        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            with self.solver_lock:
                self.cached_status = "WAITING_PIECES"
            return None

        signature = self.piece_signature(pieces)
        now = time.monotonic()
        with self.solver_lock:
            hard_scale = max(1.0,
                self.solver_hard_change_tolerance_mm
                / max(self.solver_change_tolerance_mm, 1e-6)
            )
            just_initialized_scene = self.last_seen_signature is None
            if just_initialized_scene:
                # 第一帧立即建立基准场景；不能把 None 传给后台求解器。
                self.last_seen_signature = signature
                self.signature_stable_count = 1
                self.scene_change_candidate_count = 0
                self.scene_change_candidate_signature = None

            same_as_last_seen = self.scene_equivalent(
                self.last_seen_signature, signature, tolerance_scale=1.0
            )
            same_as_last_seen_hard = self.scene_equivalent(
                self.last_seen_signature, signature, tolerance_scale=hard_scale
            )

            if same_as_last_seen or same_as_last_seen_hard:
                # 5~12 mm 的单帧跳变仍按扑克牌边缘抖动处理。
                if not just_initialized_scene:
                    self.signature_stable_count += 1
                self.scene_change_candidate_count = 0
                self.scene_change_candidate_signature = None
                if self.last_seen_signature is None:
                    self.last_seen_signature = signature
            else:
                # 只有连续多帧都确认变化，才真正切换场景。
                if self.scene_equivalent(
                    self.scene_change_candidate_signature, signature,
                    tolerance_scale=hard_scale
                ):
                    self.scene_change_candidate_count += 1
                else:
                    self.scene_change_candidate_signature = signature
                    self.scene_change_candidate_count = 1

                if self.scene_change_candidate_count >= self.scene_change_confirm_frames:
                    self.last_seen_signature = signature
                    self.signature_stable_count = 1
                    self.scene_change_candidate_count = 0
                    self.scene_change_candidate_signature = None
                else:
                    # 尚未确认是真实移动，继续沿用旧场景，不重启求解。
                    self.signature_stable_count += 1
                    signature = self.last_seen_signature

            same_as_cached = self.scene_equivalent(
                self.cached_signature, signature, tolerance_scale=hard_scale
            )
            if same_as_cached and self.cached_solution is not None:
                return self.cached_solution

            running = self.solver_thread is not None and self.solver_thread.is_alive()
            if running:
                # 只有连续确认的新场景才会到达这里并取消旧搜索。
                if (
                    signature is not None
                    and not self.scene_equivalent(
                        self.solver_running_signature, signature,
                        tolerance_scale=hard_scale
                    )
                    and self.scene_change_candidate_count == 0
                ):
                    self.solver_cancel_event.set()
                return None

            # 若配置了有限总时限，周期结束后自动再次启动下一周期；
            # 无总时限时正常情况下线程会一直运行到 FOUND。
            # 场景比较，而不是逐帧完全相等。
            same_failed_scene = (
                same_as_cached
                and self.cached_solution is None
                and self.cached_finished_at > 0.0
                and self.cached_status in (
                    "TIMEOUT", "NOT_FOUND", "NO_DIMENSION", "NO_TOPOLOGY",
                    "LOW_QUALITY", "ERROR", "SEARCH_TIMEOUT",
                    "SEARCH_ROUND_LIMIT", "CANCELLED"
                )
            )
            if same_failed_scene:
                if not self.solver_auto_retry_failed:
                    return None
                if now - self.cached_finished_at < self.solver_retry_sec:
                    return None

            if self.signature_stable_count < self.solver_stable_frames:
                self.cached_status = "STABILIZING"
                return None

            pieces_snapshot = copy.deepcopy(pieces)
            warped_snapshot = warped.copy()
            self.cached_status = "SOLVING"
            self.solver_cancel_event.clear()
            # 启动时立即冻结本次场景身份。求解完成后即使轮廓边数/周长抖动，
            # 只要中心和尺寸在 5 mm 容差内，也不会被误判成新场景。
            self.cached_signature = signature
            self.solver_running_signature = signature
            worker = threading.Thread(
                target=self.solver_worker,
                args=(signature, pieces_snapshot, warped_snapshot),
                daemon=True,
            )
            self.solver_thread = worker
            worker.start()
        return None

    def solver_reset_callback(self, _request):
        """取消当前持续搜索并清除缓存；新场景稳定后自动重新开始。"""
        with self.solver_lock:
            running = self.solver_thread is not None and self.solver_thread.is_alive()
            if running:
                self.solver_cancel_event.set()
                self.last_seen_signature = None
                self.signature_stable_count = 0
                self.scene_change_candidate_count = 0
                self.scene_change_candidate_signature = None
                self.temporal_mask_history.clear()
                self.temporal_last_raw_mask = None
                self.cached_status = "CANCELLING"
                return TriggerResponse(True, "已请求取消当前搜索；线程退出后将重新识别并搜索")
            self.solver_running_signature = None
            self.last_seen_signature = None
            self.signature_stable_count = 0
            self.scene_change_candidate_count = 0
            self.scene_change_candidate_signature = None
            self.temporal_mask_history.clear()
            self.temporal_last_raw_mask = None
            self.cached_signature = None
            self.cached_solution = None
            self.cached_status = "WAITING"
            self.cached_finished_at = 0.0
            self.cached_solve_sec = 0.0
            self.cached_stats = {}
            self.solver_round_index = 0
            self.solver_round_piece_order = []
        return TriggerResponse(True, "求解缓存已清除，将在轮廓稳定后重新求解")

    # ------------------------------------------------------------------
    # 显示与状态发布
    # ------------------------------------------------------------------
    def process_frame(self, bgr: np.ndarray):
        output = super().process_frame(bgr)
        annotated, mask, warped, warped_mask, results, solution = output
        snapshot = self.current_solver_snapshot()
        status = str(snapshot["status"])

        if solution is None and status in (
            "SOLVING",
            "STABILIZING",
            "TIMEOUT",
            "NOT_FOUND",
            "NO_DIMENSION",
            "ERROR",
            "SEARCH_TIMEOUT",
            "SEARCH_ROUND_LIMIT",
            "CANCELLED",
            "CANCELLING",
            "NO_TOPOLOGY",
            "LOW_QUALITY",
            "WAITING_PIECES",
        ):
            quad = self.get_a4_quad(bgr)
            if status == "SOLVING":
                attempt_index = int(snapshot.get("attempt_index", 0))
                attempt_count = int(snapshot.get("attempt_count", 0))
                round_index = int(snapshot.get("round_index", 0))
                piece_order = snapshot.get("piece_order", [])
                if attempt_index > 0 and attempt_count > 0:
                    state_text = "rectangle solution: SOLVING R{} A{}/{}".format(
                        max(1, round_index), attempt_index, attempt_count
                    )
                    if piece_order:
                        state_text += " order=" + "-".join(str(v) for v in piece_order)
                else:
                    state_text = "rectangle solution: SOLVING"
                color = (0, 220, 255)
            elif status == "STABILIZING":
                state_text = "rectangle solution: STABILIZING {}/{}".format(
                    snapshot["stable_frames"], snapshot["required_stable_frames"]
                )
                color = (0, 220, 255)
            else:
                state_text = "rectangle solution: {}".format(status)
                color = (80, 80, 255)
            lines = ["onsite pieces: {}".format(self.last_piece_count), state_text]
            if status != "SOLVING" and float(snapshot["solve_sec"]) > 0:
                lines.append("solve time: {:.2f}s".format(snapshot["solve_sec"]))
            lines.extend(self.last_piece_summaries[:4])
            self.draw_info_panel(annotated, lines, quad=quad, text_color=color)

        return annotated, mask, warped, warped_mask, results, solution

    def publish_outputs(
        self,
        header,
        annotated,
        mask,
        warped,
        warped_mask,
        results,
        solution,
    ) -> None:
        super().publish_outputs(
            header, annotated, mask, warped, warped_mask, results, solution
        )
        snapshot = self.current_solver_snapshot()
        payload = {
            "status": snapshot["status"],
            "running": snapshot["running"],
            "piece_count": self.last_piece_count,
            "stable_frames": snapshot["stable_frames"],
            "required_stable_frames": snapshot["required_stable_frames"],
            "last_solve_sec": snapshot["solve_sec"],
            "attempt_index": snapshot.get("attempt_index", 0),
            "attempt_count": snapshot.get("attempt_count", 0),
            "attempt_timeouts_sec": list(self.solver_attempt_timeouts),
            "solution_available": solution is not None,
            "stats": snapshot["stats"],
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def color_callback(self, msg) -> None:
        if not getattr(self, "competition_ready", False):
            return
        super().color_callback(msg)





class TopologyCompetitionOnsitePuzzleSolver(CompetitionOnsitePuzzleSolver):
    """v10：边兼容矩阵 + 生成树拓扑 + 刚体姿态传播。

    与 v8/v9 的自由递归摆放不同，本求解器不会在矩形内部任意滑动碎片。
    它先为每对碎片只保留少量高质量切边匹配，再枚举 n<=4 时最多
    16 种生成树拓扑。固定一片后，其余碎片的姿态由所选切边直接传播，
    因此搜索规模从数十万节点下降为几百到几千个组合。
    """

    def __init__(self) -> None:
        # 防止父类刚完成初始化时相机首帧抢先进入拓扑求解。
        self.topology_ready = False
        super().__init__()
        self.competition_ready = False

        self.topology_merge_collinear_deg = float(
            rospy.get_param("~topology_merge_collinear_deg", 18.0)
        )
        self.topology_chain_min_length_mm = float(
            rospy.get_param("~topology_chain_min_length_mm", 8.0)
        )
        self.topology_pair_options_per_level = [
            int(v) for v in rospy.get_param(
                "~topology_pair_options_per_level", [5, 9, 14]
            )
        ]
        self.topology_edge_abs_tol_mm_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_edge_abs_tol_mm_per_level", [5.0, 8.0, 12.0]
            )
        ]
        self.topology_edge_ratio_tol_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_edge_ratio_tol_per_level", [0.16, 0.25, 0.36]
            )
        ]
        self.topology_partial_edge_max_ratio_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_partial_edge_max_ratio_per_level", [2.2, 4.5, 6.0]
            )
        ]
        self.topology_interval_reuse_tolerance = float(
            rospy.get_param("~topology_interval_reuse_tolerance", 0.06)
        )
        self.topology_pair_overlap_limit_mm2_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_pair_overlap_limit_mm2_per_level", [35.0, 70.0, 130.0]
            )
        ]
        self.topology_complete_limit_per_level = [
            int(v) for v in rospy.get_param(
                "~topology_complete_limit_per_level", [180, 650, 1800]
            )
        ]
        self.topology_orientation_limit = int(
            rospy.get_param("~topology_orientation_limit", 10)
        )
        self.topology_dimension_candidates = int(
            rospy.get_param("~topology_dimension_candidates", 4)
        )
        self.topology_accept_fill_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_accept_fill_per_level", [0.82, 0.77, 0.72]
            )
        ]
        self.topology_accept_area_error_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_accept_area_error_per_level", [0.12, 0.18, 0.25]
            )
        ]
        self.topology_accept_overlap_mm2_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_accept_overlap_mm2_per_level", [70.0, 120.0, 180.0]
            )
        ]
        self.topology_accept_outside_mm2_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_accept_outside_mm2_per_level", [60.0, 120.0, 200.0]
            )
        ]
        self.topology_accept_unmatched_ratio_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_accept_unmatched_ratio_per_level", [0.34, 0.44, 0.54]
            )
        ]
        self.topology_corner_tolerance_mm_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_corner_tolerance_mm_per_level", [7.0, 10.0, 14.0]
            )
        ]
        self.topology_outer_line_tolerance_mm_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_outer_line_tolerance_mm_per_level", [3.0, 5.0, 8.0]
            )
        ]
        self.topology_line_tolerance_mm_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_line_tolerance_mm_per_level", [2.5, 4.0, 6.0]
            )
        ]
        self.topology_angle_tolerance_deg_per_level = [
            float(v) for v in rospy.get_param(
                "~topology_angle_tolerance_deg_per_level", [8.0, 13.0, 20.0]
            )
        ]
        self.topology_texture_finalists = int(
            rospy.get_param("~topology_texture_finalists", 8)
        )

        self.topology_ready = True
        self.competition_ready = True
        rospy.loginfo(
            "v10 topology solver ready: edge options=%s, change_tol=%.1fmm",
            self.topology_pair_options_per_level,
            self.solver_change_tolerance_mm,
        )

    @staticmethod
    def _topo_level_value(values, level):
        if not values:
            raise ValueError("empty topology level list")
        return values[min(max(int(level), 0), len(values) - 1)]

    def _topo_edge_chains(self, piece: GenericPiece):
        """返回单边以及由两条近共线相邻边合并而成的边链。

        这样可以容忍“真实一条边被 approxPolyDP 拆成两条短边”的情况。
        """
        poly = np.asarray(piece.polygon_cart_mm, dtype=np.float64)
        count = len(poly)
        outer_likelihood = [0.0] * count
        try:
            for vertex_index, angle_error, _, _ in self.right_angle_vertices(poly):
                strength = max(0.0, 1.0 - angle_error / max(
                    self.right_angle_tolerance_deg, 1.0
                ))
                outer_likelihood[(vertex_index - 1) % count] = max(
                    outer_likelihood[(vertex_index - 1) % count], strength
                )
                outer_likelihood[vertex_index] = max(
                    outer_likelihood[vertex_index], strength
                )
        except Exception:
            pass
        chains = []
        for i in range(count):
            a = poly[i]
            b = poly[(i + 1) % count]
            length = float(np.linalg.norm(b - a))
            if length >= self.topology_chain_min_length_mm:
                chains.append({
                    "ids": (int(i),),
                    "a": a.copy(),
                    "b": b.copy(),
                    "length": length,
                    "merge_penalty": 0.0,
                    "outer_likelihood": float(outer_likelihood[i]),
                })

        for i in range(count):
            a = poly[i]
            m = poly[(i + 1) % count]
            b = poly[(i + 2) % count]
            v1 = m - a
            v2 = b - m
            l1 = float(np.linalg.norm(v1))
            l2 = float(np.linalg.norm(v2))
            if l1 < 1e-6 or l2 < 1e-6:
                continue
            dot = float(np.clip(np.dot(v1 / l1, v2 / l2), -1.0, 1.0))
            turn = math.degrees(math.acos(dot))
            combined = float(np.linalg.norm(b - a))
            if (
                turn <= self.topology_merge_collinear_deg
                and combined >= self.topology_chain_min_length_mm
            ):
                chains.append({
                    "ids": (int(i), int((i + 1) % count)),
                    "a": a.copy(),
                    "b": b.copy(),
                    "length": combined,
                    "merge_penalty": 0.04 * turn + 0.15,
                    "outer_likelihood": float(max(
                        outer_likelihood[i], outer_likelihood[(i + 1) % count]
                    )),
                })
        return chains

    @staticmethod
    def _topo_relative_transform(src_a, src_b, dst_a, dst_b):
        """求把 src 反向贴到 dst 的二维刚体变换。"""
        src_vec = np.asarray(src_b) - np.asarray(src_a)
        dst_vec = np.asarray(dst_a) - np.asarray(dst_b)  # 反向
        if np.linalg.norm(src_vec) < 1e-8 or np.linalg.norm(dst_vec) < 1e-8:
            return None, None
        angle = math.atan2(dst_vec[1], dst_vec[0]) - math.atan2(
            src_vec[1], src_vec[0]
        )
        R = OnsitePuzzleSolver.rotation_matrix(angle)
        return R, angle

    def _topo_partition_starts(
        self, pieces, long_piece_index, short_piece_index,
        long_length, short_length, level
    ):
        """为“长边被多块短边分段占用”生成有意义的起点。

        除均匀采样外，还使用其他碎片边长的子集和。四片题中真实分段位置
        往往正好是前面一到两段接缝长度之和，例如 20、20+50=70 mm。
        """
        slack = max(0.0, float(long_length) - float(short_length))
        if slack <= 1e-6:
            return [0.0]
        lengths = []
        for index, piece in enumerate(pieces):
            if index == long_piece_index or index == short_piece_index:
                continue
            for _, a, b in self.polygon_edges(piece.polygon_cart_mm):
                value = float(np.linalg.norm(b - a))
                if 5.0 <= value <= long_length + 1e-6:
                    lengths.append(value)
        # 1 mm 去重，防止相似边产生大量几乎相同的滑动位置。
        lengths = sorted(set(round(value, 1) for value in lengths))
        starts = {0.0, round(slack, 3), round(0.5 * slack, 3)}
        for fraction in ([0.25, 0.75] if level >= 1 else []):
            starts.add(round(fraction * slack, 3))
        for value in lengths:
            if value <= slack + 1e-6:
                starts.add(round(value, 3))
        # 最多四片，长边前面最多由另外两段组成；对子集和枚举到 2 已足够。
        for ai in range(len(lengths)):
            for bi in range(ai + 1, len(lengths)):
                value = lengths[ai] + lengths[bi]
                if value <= slack + 1e-6:
                    starts.add(round(value, 3))
        values = sorted(max(0.0, min(slack, value)) for value in starts)
        # 三级分别限制起点数；优先保留两端、中心及子集和。
        limit = [12, 22, 36][min(max(level, 0), 2)]
        if len(values) <= limit:
            return values
        important = [0.0, slack, 0.5 * slack]
        values.sort(key=lambda value: min(abs(value - point) for point in important))
        return sorted(values[:limit])

    def _topo_pair_options(self, pieces, parent_index, child_index, level):
        parent = pieces[parent_index]
        child = pieces[child_index]
        parent_chains = self._topo_edge_chains(parent)
        child_chains = self._topo_edge_chains(child)
        abs_tol = self._topo_level_value(
            self.topology_edge_abs_tol_mm_per_level, level
        )
        ratio_tol = self._topo_level_value(
            self.topology_edge_ratio_tol_per_level, level
        )
        partial_ratio_limit = self._topo_level_value(
            self.topology_partial_edge_max_ratio_per_level, level
        )
        overlap_limit = self._topo_level_value(
            self.topology_pair_overlap_limit_mm2_per_level, level
        )
        line_tol = self._topo_level_value(
            self.topology_line_tolerance_mm_per_level, level
        )
        max_keep = self._topo_level_value(
            self.topology_pair_options_per_level, level
        )

        options = []

        for fixed in parent_chains:
            for moving in child_chains:
                la = float(fixed["length"])
                lb = float(moving["length"])
                shorter = min(la, lb)
                longer = max(la, lb)
                if shorter < self.topology_chain_min_length_mm:
                    continue
                length_diff = abs(la - lb)
                ratio = longer / max(shorter, 1e-6)
                near_equal = length_diff <= max(abs_tol, ratio_tol * longer)
                partial_match = ratio <= partial_ratio_limit
                if not near_equal and not partial_match:
                    continue

                R, angle = self._topo_relative_transform(
                    moving["a"], moving["b"], fixed["a"], fixed["b"]
                )
                if R is None:
                    continue
                rb0 = R @ moving["a"]
                rb1 = R @ moving["b"]
                fixed_mid = 0.5 * (fixed["a"] + fixed["b"])
                moving_mid = 0.5 * (rb0 + rb1)
                fixed_vec = fixed["b"] - fixed["a"]
                fixed_len = max(float(np.linalg.norm(fixed_vec)), 1e-9)
                fixed_unit = fixed_vec / fixed_len
                fixed_left = np.array([-fixed_unit[1], fixed_unit[0]], dtype=np.float64)

                base_translations = []
                if la >= lb:
                    starts = self._topo_partition_starts(
                        pieces, parent_index, child_index, la, lb, level
                    )
                    for start in starts:
                        target_mid = fixed["a"] + fixed_unit * (start + 0.5 * lb)
                        base_translations.append((
                            "parent_start_{:.1f}".format(start),
                            target_mid - moving_mid,
                        ))
                else:
                    child_vec = rb1 - rb0
                    child_unit = child_vec / max(float(np.linalg.norm(child_vec)), 1e-9)
                    starts = self._topo_partition_starts(
                        pieces, child_index, parent_index, lb, la, level
                    )
                    for start in starts:
                        moving_segment_mid = rb0 + child_unit * (start + 0.5 * la)
                        base_translations.append((
                            "child_start_{:.1f}".format(start),
                            fixed_mid - moving_segment_mid,
                        ))

                gap_values = [0.0]
                if level >= 1:
                    gap_values.append(0.8)
                if level >= 2:
                    gap_values.append(1.6)

                for align_name, base_t in base_translations:
                    for gap in gap_values:
                        t = base_t - fixed_left * gap
                        child_poly = self.transform_polygon(child, R, t)
                        pair_overlap = self.pair_overlap_area_mm2(
                            parent.polygon_cart_mm, child_poly
                        )
                        if not math.isfinite(pair_overlap) or pair_overlap > overlap_limit:
                            continue
                        cb0 = R @ moving["a"] + t
                        cb1 = R @ moving["b"] + t
                        seam_overlap, q0, q1 = self.segment_overlap_on_line(
                            fixed["a"], fixed["b"], cb0, cb1,
                            2.0, max(line_tol, gap + 0.8),
                        )
                        min_required = max(6.0, 0.62 * shorter)
                        if seam_overlap < min_required or q0 is None or q1 is None:
                            continue

                        def normalized_interval(a, b, p0, p1):
                            vec = b - a
                            length = max(float(np.linalg.norm(vec)), 1e-9)
                            unit = vec / length
                            v0 = float(np.dot(p0 - a, unit)) / length
                            v1 = float(np.dot(p1 - a, unit)) / length
                            return (max(0.0, min(v0, v1)), min(1.0, max(v0, v1)))

                        parent_interval = normalized_interval(
                            fixed["a"], fixed["b"], q0, q1
                        )
                        child_interval = normalized_interval(cb0, cb1, q0, q1)
                        uncovered = max(0.0, shorter - seam_overlap)
                        # 长边剩余部分可能与第三块拼接，不再把长短差本身当作重罚。
                        partial_penalty = 0.02 * max(0.0, longer - shorter)
                        score = (
                            0.25 * length_diff if near_equal else partial_penalty
                        ) + (
                            0.38 * uncovered
                            + 0.18 * pair_overlap
                            + 0.7 * gap
                            + float(fixed["merge_penalty"])
                            + float(moving["merge_penalty"])
                            + 3.0 * float(fixed.get("outer_likelihood", 0.0))
                            + 3.0 * float(moving.get("outer_likelihood", 0.0))
                        )
                        options.append({
                            "parent": int(parent_index),
                            "child": int(child_index),
                            "parent_edges": tuple(fixed["ids"]),
                            "child_edges": tuple(moving["ids"]),
                            "parent_interval": tuple(parent_interval),
                            "child_interval": tuple(child_interval),
                            "R": R.copy(),
                            "t": np.asarray(t, dtype=np.float64).copy(),
                            "angle": float(angle),
                            "score": float(score),
                            "seam_overlap": float(seam_overlap),
                            "length_diff": float(length_diff),
                            "pair_overlap": float(pair_overlap),
                            "align": align_name,
                            "gap": float(gap),
                        })

        # 同一边链对的不同滑动区段必须保留；同时先为每个边链对保留
        # 至少一个候选，防止“长边-短边”的真实接缝被大量低分滑动候选淹没。
        best_by_slot = {}
        for option in options:
            pc = 0.5 * sum(option["parent_interval"])
            cc = 0.5 * sum(option["child_interval"])
            key = (
                option["parent_edges"], option["child_edges"],
                round(pc, 1), round(cc, 1), round(option["gap"], 1),
            )
            current = best_by_slot.get(key)
            if current is None or option["score"] < current["score"]:
                best_by_slot[key] = option
        groups = {}
        for option in best_by_slot.values():
            groups.setdefault(
                (option["parent_edges"], option["child_edges"]), []
            ).append(option)
        for values in groups.values():
            values.sort(key=lambda item: item["score"])
        selected = []
        group_order = sorted(groups, key=lambda key: groups[key][0]["score"])
        # 第一轮：每种物理边链配对一个。
        for key in group_order:
            if len(selected) >= max_keep:
                break
            selected.append(groups[key][0])
        # 后续轮次：按组轮询添加不同滑动区段。
        depth = 1
        while len(selected) < max_keep:
            added = False
            for key in group_order:
                values = groups[key]
                if depth < len(values):
                    selected.append(values[depth])
                    added = True
                    if len(selected) >= max_keep:
                        break
            if not added:
                break
            depth += 1
        selected.sort(key=lambda item: item["score"])
        return selected

    @staticmethod
    def _topo_prufer_tree(sequence, count):
        degree = [1] * count
        for value in sequence:
            degree[value] += 1
        sequence = list(sequence)
        edges = []
        for value in sequence:
            leaf = min(index for index in range(count) if degree[index] == 1)
            edges.append((leaf, value))
            degree[leaf] -= 1
            degree[value] -= 1
        remaining = [index for index in range(count) if degree[index] == 1]
        if len(remaining) == 2:
            edges.append((remaining[0], remaining[1]))
        return edges

    def _topo_spanning_trees(self, count):
        if count <= 1:
            return [[]]
        if count == 2:
            return [[(0, 1)]]
        trees = []
        for sequence in itertools.product(range(count), repeat=count - 2):
            trees.append(self._topo_prufer_tree(sequence, count))
        return trees

    @staticmethod
    def _topo_orient_tree(edges, count, root=0):
        graph = [[] for _ in range(count)]
        for a, b in edges:
            graph[a].append(b)
            graph[b].append(a)
        oriented = []
        parent = {root: -1}
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for other in graph[current]:
                if other in parent:
                    continue
                parent[other] = current
                oriented.append((current, other))
                queue.append(other)
        return oriented

    def _topo_compose_child(self, parent_placement, child_piece, option):
        R = parent_placement.R @ option["R"]
        t = parent_placement.R @ option["t"] + parent_placement.t
        poly = self.transform_polygon(child_piece, R, t)
        return Placement(
            piece_index=int(option["child"]),
            R=R,
            t=t,
            polygon_cart_mm=poly,
        )

    def _topo_usage_available(self, usage, edge_ids, interval):
        """检查边区段是否仍可使用。单边允许多个不重叠区段。"""
        if len(edge_ids) > 1:
            return all(not usage.get(edge_id) for edge_id in edge_ids)
        edge_id = edge_ids[0]
        low, high = interval
        for old_low, old_high in usage.get(edge_id, []):
            overlap = min(high, old_high) - max(low, old_low)
            if overlap > self.topology_interval_reuse_tolerance:
                return False
        return True

    @staticmethod
    def _topo_usage_add(usage, edge_ids, interval):
        token = []
        if len(edge_ids) > 1:
            for edge_id in edge_ids:
                usage.setdefault(edge_id, []).append((0.0, 1.0))
                token.append((edge_id, (0.0, 1.0)))
        else:
            edge_id = edge_ids[0]
            value = (float(interval[0]), float(interval[1]))
            usage.setdefault(edge_id, []).append(value)
            token.append((edge_id, value))
        return token

    @staticmethod
    def _topo_usage_remove(usage, token):
        for edge_id, value in reversed(token):
            values = usage.get(edge_id, [])
            for index in range(len(values) - 1, -1, -1):
                if values[index] == value:
                    del values[index]
                    break
            if not values and edge_id in usage:
                del usage[edge_id]

    def _topo_quick_valid(self, new_placement, placements, pair_limit):
        overlap_sum = 0.0
        for existing in placements.values():
            overlap = self.pair_overlap_area_mm2(
                new_placement.polygon_cart_mm, existing.polygon_cart_mm
            )
            if not math.isfinite(overlap):
                return False
            overlap_sum += overlap
            if overlap > pair_limit:
                return False
        polygons = [new_placement.polygon_cart_mm] + [
            item.polygon_cart_mm for item in placements.values()
        ]
        extent = np.vstack(polygons).max(axis=0) - np.vstack(polygons).min(axis=0)
        if max(extent) > self.rect_long_max_mm + 25.0:
            return False
        if min(extent) > self.rect_short_max_mm + 25.0:
            return False
        return overlap_sum <= pair_limit * max(1, len(placements))

    def _topo_orientation_angles(self, placements, used_edges):
        angles = []
        for index, placement in placements.items():
            poly = placement.polygon_cart_mm
            used = used_edges.get(index, set())
            for edge_index, a, b in self.polygon_edges(poly):
                if edge_index in used:
                    continue
                vector = b - a
                if np.linalg.norm(vector) < 1e-6:
                    continue
                angle = math.atan2(vector[1], vector[0])
                angles.extend([-angle, math.pi / 2.0 - angle])

        points = np.vstack([p.polygon_cart_mm for p in placements.values()])
        try:
            rect = cv2.minAreaRect(points.astype(np.float32).reshape(-1, 1, 2))
            min_rect_angle = math.radians(float(rect[2]))
            angles.extend([-min_rect_angle, math.pi / 2.0 - min_rect_angle])
        except cv2.error:
            pass
        angles.append(0.0)

        unique = []
        for angle in angles:
            # 矩形方向每 90 度等价。
            canonical = (angle + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
            if all(abs(math.atan2(math.sin(canonical - old), math.cos(canonical - old)))
                   > math.radians(1.5) for old in unique):
                unique.append(canonical)
        return unique[: self.topology_orientation_limit]

    def _topo_rotate_placements(self, placements, angle):
        Q = self.rotation_matrix(angle)
        rotated = {}
        for index, placement in placements.items():
            R = Q @ placement.R
            t = Q @ placement.t
            poly = (Q @ placement.polygon_cart_mm.T).T
            rotated[index] = Placement(index, R, t, poly)
        return rotated

    def _topo_target_dimensions(self, span, total_area, level):
        span_w, span_h = float(span[0]), float(span[1])
        landscape = span_w >= span_h
        step = max(1.0, float(self.rectangle_step_mm))
        candidates = []
        long_values = np.arange(
            self.rect_long_min_mm,
            self.rect_long_max_mm + 0.5 * step,
            step,
        )
        short_values = np.arange(
            self.rect_short_min_mm,
            self.rect_short_max_mm + 0.5 * step,
            step,
        )
        max_area_error = self._topo_level_value(
            self.topology_accept_area_error_per_level, level
        )
        for long_side in long_values:
            for short_side in short_values:
                width, height = (
                    (float(long_side), float(short_side))
                    if landscape else
                    (float(short_side), float(long_side))
                )
                target_area = width * height
                area_error = abs(target_area - total_area) / max(target_area, 1.0)
                if area_error > max_area_error + 0.04:
                    continue
                # 正确拓扑的包围盒应接近目标矩形；允许识别误差但不允许
                # 像旧版那样用明显过大的矩形换取高填充假象。
                span_error = abs(width - span_w) + abs(height - span_h)
                if width + 10.0 < span_w or height + 10.0 < span_h:
                    continue
                score = 4.0 * area_error + 0.018 * span_error
                candidates.append((score, width, height, area_error))
        candidates.sort(key=lambda item: item[0])
        return candidates[: self.topology_dimension_candidates]

    def _topo_metric_params(self, level):
        return {
            "angle_tol_deg": self._topo_level_value(
                self.topology_angle_tolerance_deg_per_level, level
            ),
            "line_tol_mm": self._topo_level_value(
                self.topology_line_tolerance_mm_per_level, level
            ),
            "corner_tol_mm": self._topo_level_value(
                self.topology_corner_tolerance_mm_per_level, level
            ),
            "outer_line_tol_mm": self._topo_level_value(
                self.topology_outer_line_tolerance_mm_per_level, level
            ),
        }

    def _topo_evaluate_complete(
        self, pieces, placements, used_edges, seam_score, total_area, level
    ):
        params = self._topo_metric_params(level)
        accept_fill = self._topo_level_value(
            self.topology_accept_fill_per_level, level
        )
        accept_area_error = self._topo_level_value(
            self.topology_accept_area_error_per_level, level
        )
        accept_overlap = self._topo_level_value(
            self.topology_accept_overlap_mm2_per_level, level
        )
        accept_outside = self._topo_level_value(
            self.topology_accept_outside_mm2_per_level, level
        )
        accept_unmatched = self._topo_level_value(
            self.topology_accept_unmatched_ratio_per_level, level
        )
        candidates = []

        for angle in self._topo_orientation_angles(placements, used_edges):
            rotated = self._topo_rotate_placements(placements, angle)
            all_points = np.vstack([
                item.polygon_cart_mm for item in rotated.values()
            ])
            min_xy = all_points.min(axis=0)
            normalized = self._shift_assembly(rotated, -min_xy)
            points = np.vstack([
                item.polygon_cart_mm for item in normalized.values()
            ])
            span = points.max(axis=0) - points.min(axis=0)
            for _, width, height, area_error in self._topo_target_dimensions(
                span, total_area, level
            ):
                dx = width - float(span[0])
                dy = height - float(span[1])
                x_offsets = sorted(set([0.0, 0.5 * dx, dx]))
                y_offsets = sorted(set([0.0, 0.5 * dy, dy]))
                for ox in x_offsets:
                    for oy in y_offsets:
                        shifted = self._shift_assembly(
                            normalized,
                            np.array([ox, oy], dtype=np.float64),
                        )
                        metrics = self._edge_assembly_metrics(
                            shifted, width, height, params
                        )
                        if not metrics:
                            continue
                        # _edge_assembly_metrics 使用 1 px/mm 栅格，公共边上的
                        # 像素会被重复计入 overlap。这里改用多边形真实交面积，
                        # 避免完美拼接也出现约一百多 mm2 的假重叠。
                        ordered_polys = [
                            shifted[index].polygon_cart_mm
                            for index in sorted(shifted)
                        ]
                        true_overlap = 0.0
                        for ai in range(len(ordered_polys)):
                            for bi in range(ai + 1, len(ordered_polys)):
                                true_overlap += self.pair_overlap_area_mm2(
                                    ordered_polys[ai], ordered_polys[bi]
                                )
                        metrics["raster_overlap"] = float(metrics["overlap"])
                        metrics["overlap"] = float(true_overlap)
                        # 先做硬性拓扑门控，再进行软评分。
                        if not metrics["connected"]:
                            continue
                        if metrics["seam_count"] < max(0, len(pieces) - 1):
                            continue
                        if metrics["missing_outer"] > 0:
                            continue
                        # 比赛保证目标是矩形。允许角点位置有更大误差，但四个
                        # 外角必须全部存在，避免 3 角的斜四边形伪解。
                        if metrics["corner_count"] < 4:
                            continue
                        if metrics["fill"] < accept_fill:
                            continue
                        if metrics["overlap"] > accept_overlap:
                            continue
                        if metrics["outside"] > accept_outside:
                            continue
                        if metrics["unmatched_ratio"] > accept_unmatched:
                            continue
                        if area_error > accept_area_error:
                            continue

                        geometry_score = (
                            4.0 * seam_score
                            + 260.0 * (1.0 - metrics["fill"])
                            + 0.75 * metrics["overlap"]
                            + 0.90 * metrics["outside"]
                            + 120.0 * metrics["unmatched_ratio"]
                            + 50.0 * (4 - metrics["corner_count"])
                            + 0.16 * metrics["corner_error"]
                            + 110.0 * area_error
                        )
                        solution = AssemblySolution(
                            placements=shifted,
                            width_mm=float(width),
                            height_mm=float(height),
                            fill_ratio=float(metrics["fill"]),
                            perimeter_ratio=1.0 + float(metrics["unmatched_ratio"]),
                            texture_score=0.0,
                            geometry_score=float(geometry_score),
                            total_score=float(geometry_score),
                        )
                        candidates.append((geometry_score, solution, metrics, area_error))
        candidates.sort(key=lambda item: item[0])
        return candidates

    def solve_assembly_timed(self, pieces, warped, timeout_sec):
        """拓扑求解主入口。

        每次父类分级搜索会调用本函数。level 0/1/2 只逐步增加边配对容差
        和每对碎片保留的候选数，不再改变为另一套自由装箱算法。
        """
        started = time.monotonic()
        deadline = started + max(float(timeout_sec), 0.1)
        level = max(0, min(2, int(getattr(self, "solver_attempt_index", 1)) - 1))
        stats = {
            "solver": "topology_v10",
            "level": int(level + 1),
            "piece_count": int(len(pieces)),
            "trees": 0,
            "tree_branches": 0,
            "complete_assemblies": 0,
            "evaluated_candidates": 0,
            "pair_option_counts": {},
            "accepted": False,
        }
        if not self.topology_ready:
            return None, "NOT_READY", stats
        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            return None, "INVALID_PIECE_COUNT", stats
        if len(pieces) == 1:
            # 单片题无需切边拓扑；沿最长边拟合矩形仅在该片本身近矩形时接受。
            piece = pieces[0]
            placement = Placement(
                0, np.eye(2), np.zeros(2), piece.polygon_cart_mm.copy()
            )
            candidates = self._topo_evaluate_complete(
                pieces, {0: placement}, {0: set()}, 0.0, piece.area_mm2, level
            )
            if not candidates:
                return None, "LOW_QUALITY", stats
            best = candidates[0][1]
            stats["accepted"] = True
            return best, "FOUND", stats

        # 为所有有向碎片对建立少量高质量边兼容选项。
        pair_options = {}
        for parent in range(len(pieces)):
            for child in range(len(pieces)):
                if parent == child:
                    continue
                if time.monotonic() >= deadline:
                    return None, "TIMEOUT", stats
                options = self._topo_pair_options(pieces, parent, child, level)
                pair_options[(parent, child)] = options
                stats["pair_option_counts"]["{}->{}".format(parent, child)] = len(options)

        # 若某片与任何其他片都没有兼容切边，当前级别不可能形成连通拼图。
        for index in range(len(pieces)):
            if not any(
                pair_options.get((index, other)) or pair_options.get((other, index))
                for other in range(len(pieces)) if other != index
            ):
                return None, "NO_COMPATIBLE_EDGE", stats

        trees = self._topo_spanning_trees(len(pieces))
        stats["trees"] = len(trees)
        complete_limit = self._topo_level_value(
            self.topology_complete_limit_per_level, level
        )
        pair_limit = self._topo_level_value(
            self.topology_pair_overlap_limit_mm2_per_level, level
        )
        total_area = float(sum(piece.area_mm2 for piece in pieces))
        geometric_finalists = []
        timed_out = False

        for tree_index, tree in enumerate(trees):
            if time.monotonic() >= deadline or stats["complete_assemblies"] >= complete_limit:
                timed_out = time.monotonic() >= deadline
                break
            oriented = self._topo_orient_tree(tree, len(pieces), root=0)
            if len(oriented) != len(pieces) - 1:
                continue
            root_piece = pieces[0]
            placements = {
                0: Placement(
                    0,
                    np.eye(2, dtype=np.float64),
                    np.zeros(2, dtype=np.float64),
                    root_piece.polygon_cart_mm.copy(),
                )
            }
            used_intervals = {index: {} for index in range(len(pieces))}
            seam_records = []

            def recurse(edge_pos, seam_score):
                nonlocal timed_out
                if timed_out:
                    return
                if time.monotonic() >= deadline:
                    timed_out = True
                    return
                if stats["complete_assemblies"] >= complete_limit:
                    return
                if edge_pos >= len(oriented):
                    stats["complete_assemblies"] += 1
                    evaluated = self._topo_evaluate_complete(
                        pieces, placements,
                        {index: set(used_intervals[index].keys()) for index in used_intervals},
                        seam_score, total_area, level,
                    )
                    stats["evaluated_candidates"] += len(evaluated)
                    geometric_finalists.extend(evaluated[:3])
                    return

                parent, child = oriented[edge_pos]
                options = pair_options.get((parent, child), [])
                if not options:
                    return
                for option in options:
                    stats["tree_branches"] += 1
                    if time.monotonic() >= deadline:
                        timed_out = True
                        return
                    parent_ids = tuple(option["parent_edges"])
                    child_ids = tuple(option["child_edges"])
                    if not self._topo_usage_available(
                        used_intervals[parent], parent_ids, option["parent_interval"]
                    ):
                        continue
                    if not self._topo_usage_available(
                        used_intervals[child], child_ids, option["child_interval"]
                    ):
                        continue
                    parent_placement = placements.get(parent)
                    if parent_placement is None:
                        continue
                    child_placement = self._topo_compose_child(
                        parent_placement, pieces[child], option
                    )
                    if not self._topo_quick_valid(
                        child_placement, placements, pair_limit
                    ):
                        continue
                    placements[child] = child_placement
                    parent_token = self._topo_usage_add(
                        used_intervals[parent], parent_ids, option["parent_interval"]
                    )
                    child_token = self._topo_usage_add(
                        used_intervals[child], child_ids, option["child_interval"]
                    )
                    seam_records.append(option)
                    recurse(edge_pos + 1, seam_score + float(option["score"]))
                    seam_records.pop()
                    self._topo_usage_remove(used_intervals[parent], parent_token)
                    self._topo_usage_remove(used_intervals[child], child_token)
                    del placements[child]

            recurse(0, 0.0)

        if not geometric_finalists:
            stats["seconds"] = round(time.monotonic() - started, 3)
            return None, "TIMEOUT" if timed_out else "NOT_FOUND", stats

        geometric_finalists.sort(key=lambda item: item[0])
        # 去掉几乎相同的目标尺寸/姿态候选，避免纹理评分重复。
        unique = []
        seen = set()
        for item in geometric_finalists:
            solution = item[1]
            key = (
                round(solution.width_mm, 1),
                round(solution.height_mm, 1),
                tuple(
                    round(math.degrees(math.atan2(
                        solution.placements[i].R[1, 0],
                        solution.placements[i].R[0, 0]
                    )), 1)
                    for i in sorted(solution.placements)
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= self.topology_texture_finalists:
                break

        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        texture_variance = self.overall_texture_variance(pieces, warped)
        best_solution = None
        best_total = float("inf")
        best_metrics = None
        best_area_error = None
        for geometry_score, solution, metrics, area_error in unique:
            texture_score = 0.0
            if texture_variance >= self.texture_variance_threshold:
                texture_score = self.texture_continuity_score(
                    pieces, solution.placements, warped_lab
                )
            total_score = geometry_score + self.texture_weight * texture_score
            solution.texture_score = float(texture_score)
            solution.total_score = float(total_score)
            if total_score < best_total:
                best_total = total_score
                best_solution = solution
                best_metrics = metrics
                best_area_error = area_error

        if best_solution is None:
            stats["seconds"] = round(time.monotonic() - started, 3)
            return None, "LOW_QUALITY", stats

        stats.update({
            "accepted": True,
            "seconds": round(time.monotonic() - started, 3),
            "fill_ratio": round(float(best_solution.fill_ratio), 5),
            "width_mm": round(float(best_solution.width_mm), 2),
            "height_mm": round(float(best_solution.height_mm), 2),
            "area_error_ratio": round(float(best_area_error), 5),
            "overlap_area_mm2": round(float(best_metrics["overlap"]), 2),
            "outside_area_mm2": round(float(best_metrics["outside"]), 2),
            "unmatched_edge_ratio": round(float(best_metrics["unmatched_ratio"]), 5),
            "rectangle_corner_count": int(best_metrics["corner_count"]),
            "seam_count": int(best_metrics["seam_count"]),
            "required_seam_count": int(best_metrics["required_seam_count"]),
            "texture_variance": round(float(texture_variance), 3),
        })
        return best_solution, "FOUND", stats


class JointRoleCompetitionOnsitePuzzleSolver(TopologyCompetitionOnsitePuzzleSolver):
    """v11：针对赛题约束的联合优化求解器。

    赛题保证：片数不大于 4、每片不超过 5 条边、每片至少有一条目标
    矩形外边、目标矩形尺寸在 90~120 mm × 50~90 mm。算法不使用任何
    固定碎片模板：

    1. 用边兼容矩阵和最多 16 棵生成树产生少量拓扑初值；
    2. 从未作为内部接缝的边中，为每片选择至少一条矩形外边候选；
    3. 同时优化所有碎片的旋转/平移以及矩形中心、方向、长和宽；
    4. 再检测闭环接缝并进行第二次优化；
    5. 最后用面积、填充、真实多边形重叠、越界、四角、外边和接缝
       RMS 共同验收，不能靠重叠获得高填充率。
    """

    def __init__(self) -> None:
        self.joint_ready = False
        super().__init__()
        self.competition_ready = False
        self.joint_initial_assemblies_per_level = [int(v) for v in rospy.get_param(
            "~joint_initial_assemblies_per_level", [45, 90, 160]
        )]
        self.joint_pair_options_per_tree_edge = [int(v) for v in rospy.get_param(
            "~joint_pair_options_per_tree_edge", [4, 6, 9]
        )]
        self.joint_optimize_candidates_per_level = [int(v) for v in rospy.get_param(
            "~joint_optimize_candidates_per_level", [18, 35, 60]
        )]
        self.joint_max_nfev_per_level = [int(v) for v in rospy.get_param(
            "~joint_max_nfev_per_level", [55, 85, 120]
        )]
        self.joint_pose_angle_window_deg_per_level = [float(v) for v in rospy.get_param(
            "~joint_pose_angle_window_deg_per_level", [22.0, 35.0, 50.0]
        )]
        self.joint_pose_translation_window_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_pose_translation_window_mm_per_level", [12.0, 20.0, 30.0]
        )]
        self.joint_seam_sigma_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_seam_sigma_mm_per_level", [2.2, 3.2, 4.5]
        )]
        self.joint_outer_sigma_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_outer_sigma_mm_per_level", [2.5, 3.8, 5.5]
        )]
        self.joint_corner_sigma_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_corner_sigma_mm_per_level", [5.0, 8.0, 12.0]
        )]
        self.joint_area_sigma_ratio_per_level = [float(v) for v in rospy.get_param(
            "~joint_area_sigma_ratio_per_level", [0.045, 0.070, 0.10]
        )]
        self.joint_outer_seed_max_cost_per_level = [float(v) for v in rospy.get_param(
            "~joint_outer_seed_max_cost_per_level", [10.0, 15.0, 22.0]
        )]
        self.joint_accept_area_error_per_level = [float(v) for v in rospy.get_param(
            "~joint_accept_area_error_per_level", [0.065, 0.085, 0.11]
        )]
        self.joint_accept_fill_per_level = [float(v) for v in rospy.get_param(
            "~joint_accept_fill_per_level", [0.86, 0.82, 0.78]
        )]
        self.joint_accept_overlap_mm2_per_level = [float(v) for v in rospy.get_param(
            "~joint_accept_overlap_mm2_per_level", [55.0, 85.0, 120.0]
        )]
        self.joint_accept_outside_mm2_per_level = [float(v) for v in rospy.get_param(
            "~joint_accept_outside_mm2_per_level", [45.0, 80.0, 120.0]
        )]
        self.joint_accept_seam_rms_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_accept_seam_rms_mm_per_level", [3.5, 5.0, 7.0]
        )]
        self.joint_extra_seam_line_tol_mm_per_level = [float(v) for v in rospy.get_param(
            "~joint_extra_seam_line_tol_mm_per_level", [3.0, 5.0, 7.5]
        )]
        self.joint_extra_seam_angle_tol_deg_per_level = [float(v) for v in rospy.get_param(
            "~joint_extra_seam_angle_tol_deg_per_level", [9.0, 14.0, 21.0]
        )]
        self.joint_texture_finalists = int(rospy.get_param("~joint_texture_finalists", 6))
        self.joint_ready = True
        self.competition_ready = True
        rospy.loginfo(
            "v12 continuous joint-role solver ready: 5mm cache, continuous=%s scipy=%s",
            self.solver_continuous_search,
            "OK" if least_squares is not None else "MISSING",
        )

    @staticmethod
    def _jr_angle_from_R(R):
        return float(math.atan2(R[1, 0], R[0, 0]))

    @staticmethod
    def _jr_rect_frame(points, center, phi):
        c = math.cos(phi)
        s = math.sin(phi)
        Rm = np.array([[c, s], [-s, c]], dtype=np.float64)
        return (Rm @ (np.asarray(points, dtype=np.float64) - center).T).T

    @staticmethod
    def _jr_rect_seed_from_points(points):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float64)
        edges = [box[(i + 1) % 4] - box[i] for i in range(4)]
        lengths = [float(np.linalg.norm(v)) for v in edges]
        idx = int(np.argmax(lengths))
        long_vec = edges[idx]
        long_side = lengths[idx]
        short_side = lengths[(idx + 1) % 4]
        phi = math.atan2(long_vec[1], long_vec[0])
        return np.asarray(rect[0], dtype=np.float64), phi, long_side, short_side

    def _jr_rect_dimension_seeds(self, long0, short0, total_area, level):
        step = max(1.0, float(self.rectangle_step_mm))
        values = []
        max_area_error = self._topo_level_value(
            self.joint_accept_area_error_per_level, level
        ) + 0.05
        for W in np.arange(self.rect_long_min_mm, self.rect_long_max_mm + 0.1, step):
            for H in np.arange(self.rect_short_min_mm, self.rect_short_max_mm + 0.1, step):
                area_error = abs(W * H - total_area) / max(total_area, 1.0)
                if area_error > max_area_error:
                    continue
                shape_error = abs(W - long0) + abs(H - short0)
                values.append((5.0 * area_error + 0.012 * shape_error, float(W), float(H)))
        values.sort(key=lambda x: x[0])
        return values[:4]

    def _jr_chain_lookup(self, piece):
        return {tuple(item["ids"]): item for item in self._topo_edge_chains(piece)}

    def _jr_make_seam_descriptor(self, pieces, placements, option):
        pi = int(option["parent"])
        ci = int(option["child"])
        pchain = self._jr_chain_lookup(pieces[pi]).get(tuple(option["parent_edges"]))
        cchain = self._jr_chain_lookup(pieces[ci]).get(tuple(option["child_edges"]))
        if pchain is None or cchain is None:
            return None
        pp = placements[pi]
        cp = placements[ci]
        pa = pp.R @ pchain["a"] + pp.t
        pb = pp.R @ pchain["b"] + pp.t
        ca = cp.R @ cchain["a"] + cp.t
        cb = cp.R @ cchain["b"] + cp.t
        overlap, q0, q1 = self.segment_overlap_on_line(pa, pb, ca, cb, 3.0, 4.0)
        if overlap <= 0.0 or q0 is None or q1 is None:
            # 退回到两条边的中间公共长度，仍保持部分边匹配。
            p0, p1 = option["parent_interval"]
            c0, c1 = option["child_interval"]
            qpa = pchain["a"] + (pchain["b"] - pchain["a"]) * p0
            qpb = pchain["a"] + (pchain["b"] - pchain["a"]) * p1
            qca = cchain["a"] + (cchain["b"] - cchain["a"]) * c0
            qcb = cchain["a"] + (cchain["b"] - cchain["a"]) * c1
            P0 = pp.R @ qpa + pp.t
            P1 = pp.R @ qpb + pp.t
            C0 = cp.R @ qca + cp.t
            C1 = cp.R @ qcb + cp.t
            if np.linalg.norm(P0 - C0) + np.linalg.norm(P1 - C1) <= np.linalg.norm(P0 - C1) + np.linalg.norm(P1 - C0):
                return {"a": pi, "b": ci, "a0": qpa, "a1": qpb, "b0": qca, "b1": qcb,
                        "a_edges": tuple(option["parent_edges"]), "b_edges": tuple(option["child_edges"])}
            return {"a": pi, "b": ci, "a0": qpa, "a1": qpb, "b0": qcb, "b1": qca,
                    "a_edges": tuple(option["parent_edges"]), "b_edges": tuple(option["child_edges"])}
        a0 = pp.R.T @ (q0 - pp.t)
        a1 = pp.R.T @ (q1 - pp.t)
        b0 = cp.R.T @ (q0 - cp.t)
        b1 = cp.R.T @ (q1 - cp.t)
        return {"a": pi, "b": ci, "a0": a0, "a1": a1, "b0": b0, "b1": b1,
                "a_edges": tuple(option["parent_edges"]), "b_edges": tuple(option["child_edges"])}

    def _jr_initial_assemblies(self, pieces, level, deadline, stats):
        pair_options = {}
        keep = self._topo_level_value(self.joint_pair_options_per_tree_edge, level)
        for a in range(len(pieces)):
            for b in range(len(pieces)):
                if a == b:
                    continue
                options = self._topo_pair_options(pieces, a, b, level)[:keep]
                pair_options[(a, b)] = options
                stats["pair_option_counts"]["{}->{}".format(a, b)] = len(options)
        trees = self._topo_spanning_trees(len(pieces))
        stats["trees"] = len(trees)
        max_initial = self._topo_level_value(self.joint_initial_assemblies_per_level, level)
        pair_limit = self._topo_level_value(self.topology_pair_overlap_limit_mm2_per_level, level)
        candidates = []
        for tree in trees:
            if time.monotonic() >= deadline:
                break
            oriented = self._topo_orient_tree(tree, len(pieces), root=0)
            if len(oriented) != len(pieces) - 1:
                continue
            placements = {0: Placement(0, np.eye(2), np.zeros(2), pieces[0].polygon_cart_mm.copy())}
            usage = {i: {} for i in range(len(pieces))}
            seams = []

            def recurse(k, seam_score):
                if time.monotonic() >= deadline:
                    return
                if k == len(oriented):
                    true_overlap = 0.0
                    for i in range(len(pieces)):
                        for j in range(i + 1, len(pieces)):
                            true_overlap += self.pair_overlap_area_mm2(
                                placements[i].polygon_cart_mm, placements[j].polygon_cart_mm
                            )
                    points = np.vstack([placements[i].polygon_cart_mm for i in sorted(placements)])
                    _, _, long0, short0 = self._jr_rect_seed_from_points(points)
                    total_area = sum(p.area_mm2 for p in pieces)
                    nearest_area_error = 1.0
                    for W in np.arange(self.rect_long_min_mm, self.rect_long_max_mm + 0.1, self.rectangle_step_mm):
                        for H in np.arange(self.rect_short_min_mm, self.rect_short_max_mm + 0.1, self.rectangle_step_mm):
                            nearest_area_error = min(nearest_area_error, abs(W * H - total_area) / max(total_area, 1.0))
                    cheap = float(seam_score + 0.45 * true_overlap + 70.0 * nearest_area_error + 0.02 * (abs(long0 - self.rect_long_max_mm) + abs(short0 - self.rect_short_max_mm)))
                    candidates.append((cheap, copy.deepcopy(placements), copy.deepcopy(seams)))
                    return
                parent, child = oriented[k]
                if parent not in placements:
                    return
                for option in pair_options.get((parent, child), []):
                    if time.monotonic() >= deadline:
                        return
                    if not self._topo_usage_available(usage[parent], tuple(option["parent_edges"]), option["parent_interval"]):
                        continue
                    if not self._topo_usage_available(usage[child], tuple(option["child_edges"]), option["child_interval"]):
                        continue
                    child_p = self._topo_compose_child(placements[parent], pieces[child], option)
                    if not self._topo_quick_valid(child_p, placements, pair_limit):
                        continue
                    placements[child] = child_p
                    ta = self._topo_usage_add(usage[parent], tuple(option["parent_edges"]), option["parent_interval"])
                    tb = self._topo_usage_add(usage[child], tuple(option["child_edges"]), option["child_interval"])
                    seams.append(option)
                    recurse(k + 1, seam_score + float(option["score"]))
                    seams.pop()
                    self._topo_usage_remove(usage[parent], ta)
                    self._topo_usage_remove(usage[child], tb)
                    del placements[child]

            recurse(0, 0.0)
            if len(candidates) > 3 * max_initial:
                candidates.sort(key=lambda x: x[0])
                del candidates[2 * max_initial:]
        candidates.sort(key=lambda x: x[0])
        return candidates[:max_initial]

    def _jr_pose_arrays(self, pieces, placements):
        theta = []
        trans = []
        for i in range(len(pieces)):
            theta.append(self._jr_angle_from_R(placements[i].R))
            trans.append(np.asarray(placements[i].t, dtype=np.float64).copy())
        return theta, trans

    def _jr_transform_local(self, point, theta, trans):
        return self.rotation_matrix(theta) @ np.asarray(point, dtype=np.float64) + trans

    def _jr_outer_assignments(self, pieces, theta, trans, center, phi, W, H, used_edges, level):
        max_cost = self._topo_level_value(self.joint_outer_seed_max_cost_per_level, level)
        assignments = []
        per_piece = []
        c = math.cos(phi); s = math.sin(phi)
        Rm = np.array([[c, s], [-s, c]], dtype=np.float64)
        side_specs = {
            0: ("left", 0), 1: ("right", 0), 2: ("bottom", 1), 3: ("top", 1)
        }
        for i, piece in enumerate(pieces):
            poly = piece.polygon_cart_mm
            options = []
            for edge_id, a, b in self.polygon_edges(poly):
                if edge_id in used_edges.get(i, set()):
                    continue
                A = Rm @ (self._jr_transform_local(a, theta[i], trans[i]) - center)
                B = Rm @ (self._jr_transform_local(b, theta[i], trans[i]) - center)
                v = B - A
                length = max(float(np.linalg.norm(v)), 1e-9)
                u = v / length
                for side in range(4):
                    name, axis = side_specs[side]
                    if side == 0:
                        d = 0.5 * (abs(A[0] + W / 2.0) + abs(B[0] + W / 2.0)); desired = np.array([0.0, 1.0])
                    elif side == 1:
                        d = 0.5 * (abs(A[0] - W / 2.0) + abs(B[0] - W / 2.0)); desired = np.array([0.0, 1.0])
                    elif side == 2:
                        d = 0.5 * (abs(A[1] + H / 2.0) + abs(B[1] + H / 2.0)); desired = np.array([1.0, 0.0])
                    else:
                        d = 0.5 * (abs(A[1] - H / 2.0) + abs(B[1] - H / 2.0)); desired = np.array([1.0, 0.0])
                    angle_err = math.degrees(math.acos(min(1.0, abs(float(np.dot(u, desired))))))
                    cost = d + 0.11 * angle_err
                    options.append((cost, edge_id, side))
            options.sort(key=lambda x: x[0])
            if not options or options[0][0] > max_cost:
                return None
            best = options[0]
            selected = [best]
            # 若第二条相邻边落在相邻矩形边且代价也较小，保留为角片的第二条外边。
            for candidate in options[1:]:
                if candidate[0] > max_cost + 3.0:
                    break
                if candidate[1] == best[1] or candidate[2] == best[2]:
                    continue
                if candidate[1] in ((best[1] - 1) % len(poly), (best[1] + 1) % len(poly)):
                    if {candidate[2], best[2]} in ({0, 2}, {0, 3}, {1, 2}, {1, 3}):
                        selected.append(candidate)
                        break
            for cost, edge_id, side in selected:
                assignments.append((i, int(edge_id), int(side)))
            per_piece.append(float(best[0]))
        return assignments, per_piece

    def _jr_corner_assignments(self, pieces, theta, trans, center, phi, W, H):
        c = math.cos(phi); s = math.sin(phi)
        Rp = np.array([[c, -s], [s, c]], dtype=np.float64)
        local_corners = [np.array([-W/2, -H/2]), np.array([W/2, -H/2]), np.array([W/2, H/2]), np.array([-W/2, H/2])]
        global_corners = [center + Rp @ q for q in local_corners]
        pool = []
        for i, piece in enumerate(pieces):
            for vid, point in enumerate(piece.polygon_cart_mm):
                gp = self._jr_transform_local(point, theta[i], trans[i])
                pool.append((i, vid, gp))
        assigned = []
        used = set()
        for corner in global_corners:
            candidates = sorted((float(np.linalg.norm(gp - corner)), i, vid) for i, vid, gp in pool if (i, vid) not in used)
            if not candidates:
                return None
            d, i, vid = candidates[0]
            used.add((i, vid))
            assigned.append((i, vid, corner.copy(), d))
        return assigned

    def _jr_pack_initial(self, pieces, placements, center, phi, W, H):
        x = []
        for i in range(1, len(pieces)):
            x.extend([self._jr_angle_from_R(placements[i].R), placements[i].t[0], placements[i].t[1]])
        x.extend([center[0], center[1], phi, W, H])
        return np.asarray(x, dtype=np.float64)

    def _jr_unpack(self, x, pieces, root_placement):
        theta = [self._jr_angle_from_R(root_placement.R)]
        trans = [np.asarray(root_placement.t, dtype=np.float64).copy()]
        k = 0
        for _ in range(1, len(pieces)):
            theta.append(float(x[k])); trans.append(np.array([x[k+1], x[k+2]], dtype=np.float64)); k += 3
        center = np.array([x[k], x[k+1]], dtype=np.float64)
        phi = float(x[k+2]); W = float(x[k+3]); H = float(x[k+4])
        return theta, trans, center, phi, W, H

    def _jr_bounds(self, x0, pieces, level):
        angle_window = math.radians(self._topo_level_value(self.joint_pose_angle_window_deg_per_level, level))
        trans_window = self._topo_level_value(self.joint_pose_translation_window_mm_per_level, level)
        lo = []; hi = []; k = 0
        for _ in range(1, len(pieces)):
            lo += [x0[k] - angle_window, x0[k+1] - trans_window, x0[k+2] - trans_window]
            hi += [x0[k] + angle_window, x0[k+1] + trans_window, x0[k+2] + trans_window]
            k += 3
        lo += [x0[k] - 25.0, x0[k+1] - 25.0, x0[k+2] - math.radians(25.0), self.rect_long_min_mm, self.rect_short_min_mm]
        hi += [x0[k] + 25.0, x0[k+1] + 25.0, x0[k+2] + math.radians(25.0), self.rect_long_max_mm, self.rect_short_max_mm]
        return np.asarray(lo), np.asarray(hi)

    def _jr_residual(self, x, pieces, root_placement, seams, outer_assignments, corner_assignments, total_area, level):
        theta, trans, center, phi, W, H = self._jr_unpack(x, pieces, root_placement)
        seam_sigma = self._topo_level_value(self.joint_seam_sigma_mm_per_level, level)
        outer_sigma = self._topo_level_value(self.joint_outer_sigma_mm_per_level, level)
        corner_sigma = self._topo_level_value(self.joint_corner_sigma_mm_per_level, level)
        area_sigma = self._topo_level_value(self.joint_area_sigma_ratio_per_level, level) * max(total_area, 1.0)
        residual = []
        for seam in seams:
            A0 = self._jr_transform_local(seam["a0"], theta[seam["a"]], trans[seam["a"]])
            A1 = self._jr_transform_local(seam["a1"], theta[seam["a"]], trans[seam["a"]])
            B0 = self._jr_transform_local(seam["b0"], theta[seam["b"]], trans[seam["b"]])
            B1 = self._jr_transform_local(seam["b1"], theta[seam["b"]], trans[seam["b"]])
            residual.extend(((A0 - B0) / seam_sigma).tolist())
            residual.extend(((A1 - B1) / seam_sigma).tolist())
        c = math.cos(phi); s = math.sin(phi)
        Rm = np.array([[c, s], [-s, c]], dtype=np.float64)
        for i, edge_id, side in outer_assignments:
            poly = pieces[i].polygon_cart_mm
            a = poly[edge_id]; b = poly[(edge_id + 1) % len(poly)]
            A = Rm @ (self._jr_transform_local(a, theta[i], trans[i]) - center)
            B = Rm @ (self._jr_transform_local(b, theta[i], trans[i]) - center)
            if side == 0:
                residual += [(A[0] + W/2) / outer_sigma, (B[0] + W/2) / outer_sigma]
                desired = np.array([0.0, 1.0])
            elif side == 1:
                residual += [(A[0] - W/2) / outer_sigma, (B[0] - W/2) / outer_sigma]
                desired = np.array([0.0, 1.0])
            elif side == 2:
                residual += [(A[1] + H/2) / outer_sigma, (B[1] + H/2) / outer_sigma]
                desired = np.array([1.0, 0.0])
            else:
                residual += [(A[1] - H/2) / outer_sigma, (B[1] - H/2) / outer_sigma]
                desired = np.array([1.0, 0.0])
            v = B - A; nv = max(float(np.linalg.norm(v)), 1e-9)
            residual.append(2.0 * float(v[0] * desired[1] - v[1] * desired[0]) / nv)
        # 角点对应关系按列表顺序对应左下、右下、右上、左上。
        Rp = np.array([[c, -s], [s, c]], dtype=np.float64)
        rect_local_corners = [np.array([-W/2, -H/2]), np.array([W/2, -H/2]), np.array([W/2, H/2]), np.array([-W/2, H/2])]
        for corner_index, (i, vid, _, _) in enumerate(corner_assignments):
            P = self._jr_transform_local(pieces[i].polygon_cart_mm[vid], theta[i], trans[i])
            C = center + Rp @ rect_local_corners[corner_index]
            residual.extend(((P - C) / corner_sigma).tolist())
        residual.append((W * H - total_area) / max(area_sigma, 1.0))
        # 所有顶点应位于矩形内；只对越界量加软惩罚。
        for i, piece in enumerate(pieces):
            poly = (self.rotation_matrix(theta[i]) @ piece.polygon_cart_mm.T).T + trans[i]
            q = (Rm @ (poly - center).T).T
            for point in q:
                residual.append(max(0.0, abs(point[0]) - W/2) / max(outer_sigma, 1.0))
                residual.append(max(0.0, abs(point[1]) - H/2) / max(outer_sigma, 1.0))
        return np.asarray(residual, dtype=np.float64)

    def _jr_additional_seams(self, pieces, theta, trans, existing, level):
        present_pairs = {tuple(sorted((s["a"], s["b"]))) for s in existing}
        line_tol = self._topo_level_value(self.joint_extra_seam_line_tol_mm_per_level, level)
        angle_tol = self._topo_level_value(self.joint_extra_seam_angle_tol_deg_per_level, level)
        extras = []
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                if (i, j) in present_pairs:
                    continue
                best = None
                for ai in self._topo_edge_chains(pieces[i]):
                    A0 = self._jr_transform_local(ai["a"], theta[i], trans[i]); A1 = self._jr_transform_local(ai["b"], theta[i], trans[i])
                    for bj in self._topo_edge_chains(pieces[j]):
                        B0 = self._jr_transform_local(bj["a"], theta[j], trans[j]); B1 = self._jr_transform_local(bj["b"], theta[j], trans[j])
                        overlap, q0, q1 = self.segment_overlap_on_line(A0, A1, B0, B1, angle_tol, line_tol)
                        if q0 is None or overlap < max(8.0, 0.48 * min(ai["length"], bj["length"])):
                            continue
                        gap = min(float(np.linalg.norm(A0 - B1)), float(np.linalg.norm(A1 - B0)))
                        score = -overlap + 0.2 * gap
                        if best is None or score < best[0]:
                            a0 = self.rotation_matrix(theta[i]).T @ (q0 - trans[i]); a1 = self.rotation_matrix(theta[i]).T @ (q1 - trans[i])
                            b0 = self.rotation_matrix(theta[j]).T @ (q0 - trans[j]); b1 = self.rotation_matrix(theta[j]).T @ (q1 - trans[j])
                            best = (score, {"a": i, "b": j, "a0": a0, "a1": a1, "b0": b0, "b1": b1,
                                            "a_edges": tuple(ai["ids"]), "b_edges": tuple(bj["ids"])})
                if best is not None:
                    extras.append(best[1])
        return extras

    def _jr_solution_from_x(self, x, pieces, root_placement):
        theta, trans, center, phi, W, H = self._jr_unpack(x, pieces, root_placement)
        c = math.cos(phi); s = math.sin(phi)
        Rm = np.array([[c, s], [-s, c]], dtype=np.float64)
        placements = {}
        offset = np.array([W/2.0, H/2.0], dtype=np.float64)
        for i, piece in enumerate(pieces):
            Ri = self.rotation_matrix(theta[i])
            Rout = Rm @ Ri
            tout = Rm @ (trans[i] - center) + offset
            poly = (Rout @ piece.polygon_cart_mm.T).T + tout
            placements[i] = Placement(i, Rout, tout, poly)
        return placements, float(W), float(H), theta, trans, center, phi

    def _jr_true_overlap(self, placements):
        total = 0.0
        keys = sorted(placements)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                total += self.pair_overlap_area_mm2(
                    placements[keys[a]].polygon_cart_mm,
                    placements[keys[b]].polygon_cart_mm,
                )
        return float(total)

    def _jr_seam_rms(self, seams, theta, trans):
        values = []
        for seam in seams:
            A0 = self._jr_transform_local(seam["a0"], theta[seam["a"]], trans[seam["a"]])
            A1 = self._jr_transform_local(seam["a1"], theta[seam["a"]], trans[seam["a"]])
            B0 = self._jr_transform_local(seam["b0"], theta[seam["b"]], trans[seam["b"]])
            B1 = self._jr_transform_local(seam["b1"], theta[seam["b"]], trans[seam["b"]])
            values += [float(np.linalg.norm(A0-B0)), float(np.linalg.norm(A1-B1))]
        return math.sqrt(sum(v*v for v in values) / max(len(values), 1))

    def _jr_optimize_candidate(self, pieces, placements0, seam_options, warped, total_area, level, deadline):
        if least_squares is None:
            return None
        seams = []
        for option in seam_options:
            desc = self._jr_make_seam_descriptor(pieces, placements0, option)
            if desc is not None:
                seams.append(desc)
        if len(seams) < max(0, len(pieces)-1):
            return None
        used_edges = {i: set() for i in range(len(pieces))}
        for seam in seams:
            used_edges[seam["a"]].update(seam.get("a_edges", ()))
            used_edges[seam["b"]].update(seam.get("b_edges", ()))
        points = np.vstack([placements0[i].polygon_cart_mm for i in sorted(placements0)])
        center0, phi0, long0, short0 = self._jr_rect_seed_from_points(points)
        seeds = self._jr_rect_dimension_seeds(long0, short0, total_area, level)
        if not seeds:
            return None
        theta0, trans0 = self._jr_pose_arrays(pieces, placements0)
        root = placements0[0]
        best = None
        for _, W0, H0 in seeds:
            if time.monotonic() >= deadline:
                break
            outer = self._jr_outer_assignments(pieces, theta0, trans0, center0, phi0, W0, H0, used_edges, level)
            if outer is None:
                continue
            outer_assignments, outer_seed_costs = outer
            corner_assignments = self._jr_corner_assignments(pieces, theta0, trans0, center0, phi0, W0, H0)
            if corner_assignments is None or max(item[3] for item in corner_assignments) > 28.0:
                continue
            x0 = self._jr_pack_initial(pieces, placements0, center0, phi0, W0, H0)
            lo, hi = self._jr_bounds(x0, pieces, level)
            try:
                result = least_squares(
                    lambda x: self._jr_residual(x, pieces, root, seams, outer_assignments, corner_assignments, total_area, level),
                    x0, bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=1.0,
                    max_nfev=self._topo_level_value(self.joint_max_nfev_per_level, level),
                    xtol=1e-5, ftol=1e-5, gtol=1e-5,
                )
            except Exception:
                continue
            theta1, trans1, center1, phi1, W1, H1 = self._jr_unpack(result.x, pieces, root)
            extras = self._jr_additional_seams(pieces, theta1, trans1, seams, level)
            all_seams = seams + extras
            if extras and time.monotonic() < deadline:
                try:
                    result = least_squares(
                        lambda x: self._jr_residual(x, pieces, root, all_seams, outer_assignments, corner_assignments, total_area, level),
                        result.x, bounds=(lo, hi), method="trf", loss="soft_l1", f_scale=1.0,
                        max_nfev=max(25, self._topo_level_value(self.joint_max_nfev_per_level, level)//2),
                        xtol=1e-5, ftol=1e-5, gtol=1e-5,
                    )
                except Exception:
                    pass
            placements, W, H, theta, trans, center, phi = self._jr_solution_from_x(result.x, pieces, root)
            params = {
                "angle_tol_deg": self._topo_level_value(self.topology_angle_tolerance_deg_per_level, level),
                "line_tol_mm": self._topo_level_value(self.topology_line_tolerance_mm_per_level, level),
                "corner_tol_mm": 18.0,
                "outer_line_tol_mm": 7.0,
            }
            metrics = self._edge_assembly_metrics(placements, W, H, params)
            if not metrics:
                continue
            true_overlap = self._jr_true_overlap(placements)
            metrics["raster_overlap"] = metrics["overlap"]
            metrics["overlap"] = true_overlap
            area_error = abs(W*H-total_area)/max(total_area, 1.0)
            seam_rms = self._jr_seam_rms(all_seams, theta, trans)
            accept = (
                self.rect_long_min_mm <= W <= self.rect_long_max_mm
                and self.rect_short_min_mm <= H <= self.rect_short_max_mm
                and area_error <= self._topo_level_value(self.joint_accept_area_error_per_level, level)
                and metrics["fill"] >= self._topo_level_value(self.joint_accept_fill_per_level, level)
                and true_overlap <= self._topo_level_value(self.joint_accept_overlap_mm2_per_level, level)
                and metrics["outside"] <= self._topo_level_value(self.joint_accept_outside_mm2_per_level, level)
                and seam_rms <= self._topo_level_value(self.joint_accept_seam_rms_mm_per_level, level)
                and metrics["corner_count"] >= 4
                and metrics["missing_outer"] == 0
                and metrics["connected"]
                and metrics["seam_count"] >= max(0, len(pieces)-1)
            )
            score = (
                320.0*(1.0-metrics["fill"]) + 900.0*area_error
                + 1.6*true_overlap + 1.5*metrics["outside"]
                + 16.0*seam_rms + 0.35*metrics["corner_error"]
                + 5.0*sum(outer_seed_costs)
            )
            item = {
                "accepted": bool(accept), "score": float(score), "placements": placements,
                "W": W, "H": H, "metrics": metrics, "area_error": area_error,
                "seam_rms": seam_rms, "seam_count_model": len(all_seams),
            }
            if best is None or (item["accepted"], -item["score"]) > (best["accepted"], -best["score"]):
                best = item
        return best

    def solve_assembly_timed(self, pieces, warped, timeout_sec):
        started = time.monotonic()
        deadline = started + max(0.5, float(timeout_sec))
        level = max(0, min(2, int(getattr(self, "solver_attempt_index", 1))-1))
        stats = {
            "solver": "joint_roles_v12_continuous", "level": level+1, "piece_count": len(pieces),
            "trees": 0, "initial_assemblies": 0, "optimized": 0,
            "pair_option_counts": {}, "accepted": False,
            "scene_change_tolerance_mm": float(self.solver_change_tolerance_mm),
        }
        if not self.joint_ready:
            return None, "NOT_READY", stats
        if least_squares is None:
            return None, "SCIPY_MISSING", stats
        if not (self.min_piece_count <= len(pieces) <= self.max_piece_count):
            return None, "INVALID_PIECE_COUNT", stats
        total_area = float(sum(p.area_mm2 for p in pieces))
        if len(pieces) == 1:
            placement = {0: Placement(0, np.eye(2), np.zeros(2), pieces[0].polygon_cart_mm.copy())}
            points = placement[0].polygon_cart_mm
            center, phi, long0, short0 = self._jr_rect_seed_from_points(points)
            seeds = self._jr_rect_dimension_seeds(long0, short0, total_area, level)
            if not seeds:
                return None, "NO_DIMENSION", stats
            # 单片必须本身近矩形；使用最小外接矩形直接输出。
            _, W, H = seeds[0]
            c=math.cos(phi); s=math.sin(phi); Rm=np.array([[c,s],[-s,c]])
            Rout=Rm; tout=Rm@(-center)+np.array([W/2,H/2]); poly=(Rout@pieces[0].polygon_cart_mm.T).T+tout
            sol=AssemblySolution({0:Placement(0,Rout,tout,poly)},W,H,min(1.0,total_area/(W*H)),1.0,0.0,0.0,0.0)
            return sol, "FOUND", stats
        initial = self._jr_initial_assemblies(pieces, level, deadline, stats)
        stats["initial_assemblies"] = len(initial)
        if not initial:
            stats["seconds"] = round(time.monotonic()-started,3)
            return None, "TIMEOUT" if time.monotonic()>=deadline else "NO_TOPOLOGY", stats
        limit = self._topo_level_value(self.joint_optimize_candidates_per_level, level)
        finalists = []
        for _, placements0, seam_options in initial[:limit]:
            if time.monotonic() >= deadline:
                break
            result = self._jr_optimize_candidate(pieces, placements0, seam_options, warped, total_area, level, deadline)
            stats["optimized"] += 1
            if result is not None:
                finalists.append(result)
        if not finalists:
            stats["seconds"] = round(time.monotonic()-started,3)
            return None, "TIMEOUT" if time.monotonic()>=deadline else "LOW_QUALITY", stats
        finalists.sort(key=lambda item: (not item["accepted"], item["score"]))
        accepted = [item for item in finalists if item["accepted"]]
        if not accepted:
            best = finalists[0]
            stats.update({
                "seconds": round(time.monotonic()-started,3), "accepted": False,
                "best_fill": round(float(best["metrics"]["fill"]),4),
                "best_area_error": round(float(best["area_error"]),4),
                "best_overlap_mm2": round(float(best["metrics"]["overlap"]),2),
                "best_outside_mm2": round(float(best["metrics"]["outside"]),2),
                "best_seam_rms_mm": round(float(best["seam_rms"]),2),
                "best_corner_count": int(best["metrics"]["corner_count"]),
            })
            return None, "TIMEOUT" if time.monotonic()>=deadline else "LOW_QUALITY", stats
        warped_lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        texture_variance = self.overall_texture_variance(pieces, warped)
        best_solution = None; best_item = None; best_total = float("inf")
        for item in accepted[:self.joint_texture_finalists]:
            texture = 0.0
            if texture_variance >= self.texture_variance_threshold:
                texture = self.texture_continuity_score(pieces, item["placements"], warped_lab)
            total = item["score"] + self.texture_weight * texture
            if total < best_total:
                best_total = total; best_item = item
                best_solution = AssemblySolution(
                    placements=item["placements"], width_mm=item["W"], height_mm=item["H"],
                    fill_ratio=float(item["metrics"]["fill"]), perimeter_ratio=1.0,
                    texture_score=float(texture), geometry_score=float(item["score"]), total_score=float(total)
                )
        if best_solution is None:
            return None, "LOW_QUALITY", stats
        stats.update({
            "seconds": round(time.monotonic()-started,3), "accepted": True,
            "width_mm": round(float(best_item["W"]),2), "height_mm": round(float(best_item["H"]),2),
            "fill_ratio": round(float(best_item["metrics"]["fill"]),5),
            "area_error_ratio": round(float(best_item["area_error"]),5),
            "overlap_area_mm2": round(float(best_item["metrics"]["overlap"]),2),
            "outside_area_mm2": round(float(best_item["metrics"]["outside"]),2),
            "seam_rms_mm": round(float(best_item["seam_rms"]),3),
            "seam_count_model": int(best_item["seam_count_model"]),
            "seam_count_detected": int(best_item["metrics"]["seam_count"]),
            "rectangle_corner_count": int(best_item["metrics"]["corner_count"]),
            "texture_variance": round(float(texture_variance),3),
        })
        return best_solution, "FOUND", stats



class CompetitionPuzzleExecutorNoWrist:
    def __init__(self) -> None:
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

        self.start_srv = rospy.Service("/onsite_puzzle_executor/start", Trigger, self.start_callback)
        self.stop_srv = rospy.Service("/onsite_puzzle_executor/stop", Trigger, self.stop_callback)
        self.pump_off_srv = rospy.Service("/onsite_puzzle_executor/pump_off", Trigger, self.pump_off_callback)
        self.reset_srv = rospy.Service("/onsite_puzzle_executor/reset", Trigger, self.reset_callback)

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
        self.start_wait_for_found = bool(
            rospy.get_param("~start_wait_for_found", True)
        )
        self.start_wait_timeout_s = float(
            rospy.get_param("~start_wait_timeout_s", 0.0)
        )
        self.start_wait_poll_s = max(
            0.05, float(rospy.get_param("~start_wait_poll_s", 0.20))
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
            "比赛执行节点启动：pick_z=%.1f place_z=%.1f max_rot=%.1f "
            "ignore_rotation=%s wait_for_found=%s wait_timeout=%.1fs dry_run=%s",
            self.pick_z,
            self.place_z,
            self.max_rotation_deg,
            self.ignore_rotation,
            self.start_wait_for_found,
            self.start_wait_timeout_s,
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

            # 当前机械臂第四轴未供电：无论剩余旋转角多大，都不再阻止执行。
            # 这里只记录警告，机械臂仍按求解出的 X/Y 放置坐标执行平移抓放。
            if abs(rotation) > self.max_rotation_deg:
                rospy.logwarn_throttle(
                    2.0,
                    "P%d 仍需旋转 %.1f 度，但已启用忽略旋转限制，将继续执行 XYZ 抓放",
                    label,
                    rotation,
                )

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

    def wait_for_solution_and_execute(self) -> None:
        """等待求解器 FOUND 和坐标稳定后自动执行。

        /start 可以在 SOLVING 阶段调用。等待期间不会发送机械臂或气泵命令；
        只有 stable_plan() 同时通过 FOUND、数据时效和连续帧抖动检查后才执行。
        """
        begin = time.monotonic()
        last_reason = None
        rospy.loginfo(
            "已进入等待求解状态；FOUND 且坐标稳定后自动执行（timeout=%.1fs，0为不限）",
            self.start_wait_timeout_s,
        )
        try:
            while not rospy.is_shutdown() and not self.stop_event.is_set():
                stable, reason = self.stable_plan()
                if stable is not None:
                    plan, order = stable
                    rospy.loginfo(
                        "求解完成且坐标稳定，开始执行，顺序 %s",
                        " -> ".join("P{}".format(v) for v in order),
                    )
                    self.worker_main(plan, order)
                    return

                if reason != last_reason:
                    rospy.loginfo("等待执行：%s", reason)
                    last_reason = reason

                if (
                    self.start_wait_timeout_s > 0.0
                    and time.monotonic() - begin >= self.start_wait_timeout_s
                ):
                    rospy.logwarn(
                        "等待求解超时 %.1fs，未执行机械臂",
                        self.start_wait_timeout_s,
                    )
                    return
                time.sleep(self.start_wait_poll_s)
        finally:
            # 若已经进入 worker_main，它会自行关闭气泵并清理 worker；
            # 这里重复清理是幂等的，用于等待期间被 stop 或超时的情况。
            if self.stop_event.is_set():
                self.pump(False)
            with self.worker_lock:
                if self.worker is threading.current_thread():
                    self.worker = None

    def start_callback(self, _request):
        with self.worker_lock:
            if self.worker is not None and self.worker.is_alive():
                return TriggerResponse(False, "正在执行或等待求解")

            self.stop_event.clear()
            stable, reason = self.stable_plan()
            if stable is not None:
                plan, order = stable
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

            if not self.start_wait_for_found:
                return TriggerResponse(False, reason)

            # 求解尚未完成时接受启动请求，但绝不使用 SOLVING 阶段的临时/旧坐标。
            # 后台线程只等待，不会在 FOUND 前发布机械臂命令。
            self.worker = threading.Thread(
                target=self.wait_for_solution_and_execute, daemon=True
            )
            self.worker.start()
            return TriggerResponse(
                True,
                "求解器当前为 {}；已进入等待，FOUND 且坐标稳定后自动执行".format(
                    self.latest_solver_status
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
        solver = JointRoleCompetitionOnsitePuzzleSolver()
        executor = CompetitionPuzzleExecutorNoWrist()
        rospy.loginfo("比赛一体化系统 v11（赛题边角色+联合位姿优化）已启动")
        rospy.loginfo("执行服务: /onsite_puzzle_executor/start")
        rospy.loginfo("求解重置服务: /puzzle/solver_reset")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
