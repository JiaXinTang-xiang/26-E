#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛版现场任意 1~4 片矩形拼图求解器（ROS1）。

本文件是在 onsite_puzzle_solver_v3.py 的稳定分割、轮廓、纹理和坐标计算
基础上增加的比赛保护层：

1. 求解在后台线程执行，相机画面不会因为搜索而卡死；
2. 每次求解有墙钟超时，默认 4.5 秒；
3. 碎片没有变化时复用缓存，不重复暴力搜索；
4. 只在轮廓连续稳定若干帧后启动求解；
5. 限制矩形候选、每片放置候选、递归节点和分支数；
6. Python 3.6+ 兼容，不使用 int.bit_count()；
7. 额外发布 /puzzle/solver_status，便于比赛时判断 SOLVING/FOUND/TIMEOUT。

同目录必须存在：
  - onsite_puzzle_solver_v3.py
  - puzzle_piece_detector_fixed_corners_v4.py
"""

import copy
import json
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from std_msgs.msg import String

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from onsite_puzzle_solver_v3 import (  # noqa: E402
    AssemblySolution,
    GenericPiece,
    GridPlacementCandidate,
    OnsitePuzzleSolver,
    Placement,
)


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
        self.competition_ready = True

        rospy.loginfo(
            "Competition solver ready: timeout=%.1fs stable_frames=%d rects=%d "
            "grid=%.1fmm nodes=%d placements/piece=%d",
            self.solver_timeout_sec,
            self.solver_stable_frames,
            self.max_rectangle_candidates,
            self.packing_grid_mm,
            self.packing_node_limit,
            self.max_placements_per_piece,
        )

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

    def piece_signature(self, pieces: List[GenericPiece]):
        """生成对平移不敏感、对旋转敏感的稳定签名。

        只移动碎片而不旋转时，无需重新求解；旋转、换片、片数变化时必须重算。
        """
        signature = []
        for piece in pieces:
            poly = np.asarray(piece.polygon_cart_mm, dtype=np.float64)
            edge_features = []
            for i in range(len(poly)):
                vec = poly[(i + 1) % len(poly)] - poly[i]
                length = float(np.linalg.norm(vec))
                # polygon 已经被核心程序统一为逆时针，因此保留 0~360° 的有向角。
                # 再对所有循环起点取字典序最小值：既消除轮廓起点变化，又能区分 180°旋转。
                angle = math.degrees(math.atan2(vec[1], vec[0])) % 360.0
                edge_features.append(
                    (
                        self._quantize(length, 1.5),
                        self._quantize(angle, 2.0),
                    )
                )
            rotations = [
                tuple(edge_features[offset:] + edge_features[:offset])
                for offset in range(len(edge_features))
            ]
            canonical_edges = min(rotations) if rotations else tuple()
            signature.append(
                (
                    int(piece.piece_id),
                    len(poly),
                    self._quantize(piece.area_mm2, 20.0),
                    canonical_edges,
                )
            )
        return tuple(signature)

    def current_solver_snapshot(self) -> Dict[str, object]:
        with self.solver_lock:
            running = self.solver_thread is not None and self.solver_thread.is_alive()
            return {
                "status": "SOLVING" if running else self.cached_status,
                "running": running,
                "stable_frames": self.signature_stable_count,
                "required_stable_frames": self.solver_stable_frames,
                "solve_sec": round(float(self.cached_solve_sec), 3),
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
                )
            )

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
                        start = math.ceil((-min_x - 1e-6) / step) * step
                        stop = math.floor((width_mm - max_x + 1e-6) / step) * step
                        tx = start
                        while tx <= stop + 1e-9:
                            if time.monotonic() >= deadline:
                                timed_out = True
                                break
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -0.7
                                and np.max(poly[:, 0]) <= width_mm + 0.7
                                and np.min(poly[:, 1]) >= -0.7
                                and np.max(poly[:, 1]) <= height_mm + 0.7
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
                        start = math.ceil((-min_y - 1e-6) / step) * step
                        stop = math.floor((height_mm - max_y + 1e-6) / step) * step
                        ty = start
                        while ty <= stop + 1e-9:
                            if time.monotonic() >= deadline:
                                timed_out = True
                                break
                            t = np.array([tx, ty], dtype=np.float64)
                            poly = oriented + t
                            if (
                                np.min(poly[:, 0]) >= -0.7
                                and np.max(poly[:, 0]) <= width_mm + 0.7
                                and np.min(poly[:, 1]) >= -0.7
                                and np.max(poly[:, 1]) <= height_mm + 0.7
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

        # 均匀抽样而不是只保留前若干项，避免偏向某一个边界或朝向。
        if len(placements) > self.max_placements_per_piece:
            indices = np.linspace(
                0,
                len(placements) - 1,
                self.max_placements_per_piece,
                dtype=np.int32,
            )
            placements = [placements[int(index)] for index in indices]

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

        order = sorted(range(len(pieces)), key=lambda idx: len(placement_lists[idx]))
        width_px = max(2, int(round(width_mm * self.packing_scale)))
        height_px = max(2, int(round(height_mm * self.packing_scale)))
        total_cells = width_px * height_px
        min_fill_cells = int(math.floor(self.packing_min_fill_ratio * total_cells))
        min_contact_cells = max(
            2, int(round(self.packing_min_contact_mm * self.packing_scale))
        )
        overlap_allow_cells = max(
            1, int(round(0.8 * self.packing_scale ** 2))
        )

        # 乐观面积上界，用于提前剪枝。
        max_cells_by_piece = {
            idx: max(option.full_cell_count for option in placement_lists[idx])
            for idx in order
        }
        remaining_max = [0] * (len(order) + 1)
        for level in range(len(order) - 1, -1, -1):
            remaining_max[level] = (
                remaining_max[level + 1] + max_cells_by_piece[order[level]]
            )

        nodes = 0
        timed_out = generation_timed_out
        found: List[Tuple[float, Dict[int, Placement]]] = []
        best_fill = 0.0

        def recurse(
            level: int,
            occupied_interior: int,
            occupied_full: int,
            occupied_count: int,
            chosen: Dict[int, GridPlacementCandidate],
        ) -> None:
            nonlocal nodes, timed_out, best_fill
            if timed_out:
                return
            if nodes >= self.packing_node_limit:
                return
            if len(found) >= self.packing_solution_limit:
                return
            if nodes % 64 == 0 and time.monotonic() >= deadline:
                timed_out = True
                return
            if occupied_count + remaining_max[level] < min_fill_cells:
                return

            nodes += 1
            if level == len(order):
                fill_ratio = occupied_count / float(max(total_cells, 1))
                if fill_ratio < self.packing_min_fill_ratio:
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
                found.append((fill_ratio, placements))
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
                    if contact < min_contact_cells:
                        continue
                valid.append((contact, option))

            if level > 0:
                valid.sort(key=lambda item: item[0], reverse=True)
            if len(valid) > self.max_branch_options:
                valid = valid[: self.max_branch_options]

            for _, option in valid:
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
                    chosen,
                )
                chosen.pop(index, None)
                if (
                    len(found) >= self.packing_solution_limit
                    or (best_fill >= 0.975 and len(found) >= 2)
                ):
                    return

        recurse(0, 0, 0, 0, {})
        found.sort(key=lambda item: item[0], reverse=True)
        return found, nodes, timed_out

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
                width, height, fill_ratio, perimeter_ratio, gap_area = (
                    self.raster_union_metrics(polygons)
                )
                long_side = max(width, height)
                short_side = min(width, height)
                dim_penalty = self.dimension_penalty(long_side, short_side)
                area_error_ratio = abs(
                    width_mm * height_mm - total_piece_area
                ) / max(total_piece_area, 1.0)
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
        stats.update(
            {
                "fill_ratio": round(float(best.fill_ratio), 5),
                "rectangle_mm": [
                    round(float(best.width_mm), 2),
                    round(float(best.height_mm), 2),
                ],
                "geometry_score": round(float(best.geometry_score), 5),
                "texture_score": round(float(best.texture_score), 3),
            }
        )
        return best, "FOUND", stats

    def solver_worker(
        self,
        signature,
        pieces: List[GenericPiece],
        warped: np.ndarray,
    ) -> None:
        started = time.monotonic()
        try:
            solution, status, stats = self.solve_assembly_timed(
                pieces, warped, self.solver_timeout_sec
            )
        except Exception as exc:
            rospy.logerr("Competition solver worker error: %s", exc)
            solution = None
            status = "ERROR"
            stats = {"error": str(exc)}
        elapsed = time.monotonic() - started

        with self.solver_lock:
            self.cached_signature = signature
            self.cached_solution = solution
            self.cached_status = status
            self.cached_finished_at = time.monotonic()
            self.cached_solve_sec = elapsed
            self.cached_stats = stats
            self.solver_running_signature = None
            self.solver_thread = None

        rospy.loginfo(
            "Competition solve finished: status=%s time=%.3fs stats=%s",
            status,
            elapsed,
            json.dumps(stats, ensure_ascii=False),
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
            if signature == self.last_seen_signature:
                self.signature_stable_count += 1
            else:
                self.last_seen_signature = signature
                self.signature_stable_count = 1

            if self.cached_signature == signature and self.cached_solution is not None:
                return self.cached_solution

            running = self.solver_thread is not None and self.solver_thread.is_alive()
            if running:
                return None

            # 同一个失败场景在短时间内不立即重搜，避免持续占满CPU。
            recent_same_failure = (
                self.cached_signature == signature
                and self.cached_solution is None
                and now - self.cached_finished_at < self.solver_retry_sec
            )
            if recent_same_failure:
                return None

            if self.signature_stable_count < self.solver_stable_frames:
                self.cached_status = "STABILIZING"
                return None

            pieces_snapshot = copy.deepcopy(pieces)
            warped_snapshot = warped.copy()
            self.cached_status = "SOLVING"
            self.solver_running_signature = signature
            worker = threading.Thread(
                target=self.solver_worker,
                args=(signature, pieces_snapshot, warped_snapshot),
                daemon=True,
            )
            self.solver_thread = worker
            worker.start()
        return None

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
            "WAITING_PIECES",
        ):
            quad = self.get_a4_quad(bgr)
            if status == "SOLVING":
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
            if float(snapshot["solve_sec"]) > 0:
                lines.append("last solve: {:.2f}s".format(snapshot["solve_sec"]))
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
            "timeout_sec": self.solver_timeout_sec,
            "solution_available": solution is not None,
            "stats": snapshot["stats"],
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def color_callback(self, msg) -> None:
        if not getattr(self, "competition_ready", False):
            return
        super().color_callback(msg)


if __name__ == "__main__":
    try:
        CompetitionOnsitePuzzleSolver()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
