from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import signal
import threading
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from puzzle_vision.camera import CameraError, MipiCamera, UsbCamera
from puzzle_vision.config import load_config
from puzzle_vision.detector import DetectionError
from puzzle_vision.pipeline import PuzzleVisionPipeline
from puzzle_vision.solver import SolveError


PROJECT_DIR = Path(__file__).resolve().parent
INDEX_HTML = (PROJECT_DIR / "web" / "index.html").read_bytes()
cv2.setUseOptimized(True)
cv2.setNumThreads(max(2, os.cpu_count() or 4))


def hex_to_bgr(value: str) -> list[int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError("Colour must be #RRGGBB")
    red, green, blue = (int(text[index : index + 2], 16) for index in (0, 2, 4))
    return [blue, green, red]


# Runtime-adjustable playing-card parameters.  The allow-list prevents the web
# UI from changing unrelated detector/actuator safety settings.  All values are
# persisted in config.json and affect only unknown-pattern card solving.
CARD_SETTING_SPECS: dict[str, dict[str, Any]] = {
    "card_minimum_long_side_mm": {
        "label": "长边最小值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 60.0, "maximum": 115.0, "step": 0.5,
    },
    "card_maximum_long_side_mm": {
        "label": "长边最大值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 65.0, "maximum": 125.0, "step": 0.5,
    },
    "card_minimum_short_side_mm": {
        "label": "短边最小值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 38.0, "maximum": 78.0, "step": 0.5,
    },
    "card_maximum_short_side_mm": {
        "label": "短边最大值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 42.0, "maximum": 85.0, "step": 0.5,
    },
    "card_aspect_ratio": {
        "label": "长短边比例", "group": "尺寸与圆角",
        "type": "float", "minimum": 1.20, "maximum": 1.90, "step": 0.005,
    },
    "card_maximum_aspect_error": {
        "label": "比例允许误差", "group": "尺寸与圆角",
        "type": "float", "minimum": 0.03, "maximum": 0.45, "step": 0.01,
    },
    "card_rounded_chord_min_mm": {
        "label": "圆角弦最小值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 0.0, "maximum": 10.0, "step": 0.25,
    },
    "card_rounded_chord_max_mm": {
        "label": "圆角弦最大值 (mm)", "group": "尺寸与圆角",
        "type": "float", "minimum": 1.0, "maximum": 16.0, "step": 0.25,
    },
    "card_pattern_lightness_drop_lab": {
        "label": "暗花纹阈值 (Lab)", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 6.0, "maximum": 70.0, "step": 1.0,
    },
    "card_pattern_chroma_delta_lab": {
        "label": "彩色花纹阈值 (Lab)", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 4.0, "maximum": 60.0, "step": 1.0,
    },
    "card_minimum_pattern_pixel_ratio": {
        "label": "最小花纹占比", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 0.0001, "maximum": 0.05, "step": 0.0001,
    },
    "card_glare_overexposed_ratio": {
        "label": "反光判定占比", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 0.10, "maximum": 0.95, "step": 0.01,
    },
    "card_corner_patch_min_confidence": {
        "label": "角标识别置信度", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 0.10, "maximum": 0.98, "step": 0.01,
    },
    "card_component_rank_min_confidence": {
        "label": "数字/字母置信度", "group": "花纹、颜色与反光",
        "type": "float", "minimum": 0.10, "maximum": 0.98, "step": 0.01,
    },
    "card_pattern_profile_weight": {
        "label": "沿边花纹权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 5.0, "step": 0.05,
    },
    "card_pattern_transition_weight": {
        "label": "花纹变化趋势权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 5.0, "step": 0.05,
    },
    "card_pattern_one_sided_penalty": {
        "label": "单侧断纹惩罚", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 20.0, "step": 0.25,
    },
    "card_pattern_match_bonus": {
        "label": "连续花纹奖励", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 10.0, "step": 0.25,
    },
    "card_pattern_shift_samples": {
        "label": "沿边错位容许 (采样点)", "group": "同形碎片消歧",
        "type": "int", "minimum": 0, "maximum": 5, "step": 1,
    },
    "card_pair_pattern_weight": {
        "label": "搜索阶段花纹权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 3.0, "step": 0.05,
    },
    "card_texture_weight": {
        "label": "最终花纹评分权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 2.0, "step": 0.05,
    },
    "card_symmetry_score_weight": {
        "label": "整牌 180° 对称权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 3.0, "step": 0.05,
    },
    "card_rank_score_weight": {
        "label": "角标位置权重", "group": "同形碎片消歧",
        "type": "float", "minimum": 0.0, "maximum": 3.0, "step": 0.05,
    },
    "card_search_seconds": {
        "label": "总搜索时限 (s)", "group": "速度与候选",
        "type": "float", "minimum": 0.25, "maximum": 12.0, "step": 0.05,
    },
    "card_exact_search_seconds": {
        "label": "精确边搜索时限 (s)", "group": "速度与候选",
        "type": "float", "minimum": 0.10, "maximum": 5.0, "step": 0.05,
    },
    "card_fallback_search_seconds": {
        "label": "补充搜索时限 (s)", "group": "速度与候选",
        "type": "float", "minimum": 0.10, "maximum": 6.0, "step": 0.05,
    },
    "card_max_pair_options_exact": {
        "label": "每对精确候选上限", "group": "速度与候选",
        "type": "int", "minimum": 8, "maximum": 128, "step": 1,
    },
    "card_minimum_pattern_candidates": {
        "label": "花纹复选方案数", "group": "速度与候选",
        "type": "int", "minimum": 1, "maximum": 12, "step": 1,
    },
}

WHITE_SETTING_SPECS: dict[str, dict[str, Any]] = {
    "target_min_width_mm": {
        "label": "目标长边最小值 (mm)", "group": "目标矩形尺寸",
        "section": "unknown", "key": "min_width_mm",
        "type": "float", "minimum": 55.0, "maximum": 130.0, "step": 0.5,
    },
    "target_max_width_mm": {
        "label": "目标长边最大值 (mm)", "group": "目标矩形尺寸",
        "section": "unknown", "key": "max_width_mm",
        "type": "float", "minimum": 65.0, "maximum": 150.0, "step": 0.5,
    },
    "target_min_height_mm": {
        "label": "目标短边最小值 (mm)", "group": "目标矩形尺寸",
        "section": "unknown", "key": "min_height_mm",
        "type": "float", "minimum": 30.0, "maximum": 100.0, "step": 0.5,
    },
    "target_max_height_mm": {
        "label": "目标短边最大值 (mm)", "group": "目标矩形尺寸",
        "section": "unknown", "key": "max_height_mm",
        "type": "float", "minimum": 40.0, "maximum": 125.0, "step": 0.5,
    },
    "dimension_tolerance_mm": {
        "label": "尺寸测量容差 (mm)", "group": "目标矩形尺寸",
        "section": "unknown", "key": "dimension_measurement_tolerance_mm",
        "type": "float", "minimum": 0.0, "maximum": 6.0, "step": 0.25,
    },
    "piece_min_area_mm2": {
        "label": "单片最小面积 (mm²)", "group": "碎片大小与轮廓",
        "section": "segmentation", "key": "min_area_mm2",
        "type": "float", "minimum": 20.0, "maximum": 2500.0, "step": 10.0,
    },
    "piece_max_area_mm2": {
        "label": "单片最大面积 (mm²)", "group": "碎片大小与轮廓",
        "section": "segmentation", "key": "max_area_mm2",
        "type": "float", "minimum": 500.0, "maximum": 12000.0, "step": 50.0,
    },
    "polygon_epsilon_mm": {
        "label": "直边拟合精度 (mm)", "group": "碎片大小与轮廓",
        "section": "segmentation", "key": "polygon_epsilon_mm",
        "type": "float", "minimum": 0.2, "maximum": 4.0, "step": 0.1,
    },
    "morph_open_mm": {
        "label": "去除小噪点 (mm)", "group": "碎片大小与轮廓",
        "section": "segmentation", "key": "morph_open_mm",
        "type": "float", "minimum": 0.0, "maximum": 3.0, "step": 0.1,
    },
    "morph_close_mm": {
        "label": "填补碎片缺口 (mm)", "group": "碎片大小与轮廓",
        "section": "segmentation", "key": "morph_close_mm",
        "type": "float", "minimum": 0.2, "maximum": 8.0, "step": 0.1,
    },
    "lab_distance_threshold": {
        "label": "综合色差阈值 (Lab)", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "lab_distance_threshold",
        "type": "float", "minimum": 5.0, "maximum": 80.0, "step": 1.0,
    },
    "lab_lightness_threshold": {
        "label": "明暗差阈值 (Lab)", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "lab_lightness_threshold",
        "type": "float", "minimum": 3.0, "maximum": 65.0, "step": 1.0,
    },
    "lab_chroma_threshold": {
        "label": "色度差阈值 (Lab)", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "lab_chroma_threshold",
        "type": "float", "minimum": 2.0, "maximum": 55.0, "step": 1.0,
    },
    "background_difference_threshold": {
        "label": "背景差阈值", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "background_difference_threshold",
        "type": "float", "minimum": 4.0, "maximum": 90.0, "step": 1.0,
    },
    "piece_color_tolerance_lab": {
        "label": "指定颜色综合色差容差", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "piece_color_tolerance_lab",
        "type": "float", "minimum": 5.0, "maximum": 90.0, "step": 1.0,
    },
    "piece_color_tolerance_chroma": {
        "label": "指定颜色色度容差", "group": "环境光与颜色分割",
        "section": "segmentation", "key": "piece_color_tolerance_chroma",
        "type": "float", "minimum": 2.0, "maximum": 60.0, "step": 1.0,
    },
    "a4_color_hint_weight": {
        "label": "A4 颜色提示权重", "group": "环境光与颜色分割",
        "section": "paper", "key": "color_hint_weight",
        "type": "float", "minimum": 0.0, "maximum": 3.0, "step": 0.05,
    },
    "divider_min_contrast_lab": {
        "label": "分界线最小对比度", "group": "环境光与颜色分割",
        "section": "paper", "key": "divider_min_contrast_lab",
        "type": "float", "minimum": 3.0, "maximum": 70.0, "step": 1.0,
    },
    "edge_length_tolerance_mm": {
        "label": "拼接边绝对容差 (mm)", "group": "白片求解器",
        "section": "unknown", "key": "edge_length_tolerance_mm",
        "type": "float", "minimum": 1.0, "maximum": 12.0, "step": 0.25,
    },
    "edge_length_relative_tolerance": {
        "label": "拼接边相对容差", "group": "白片求解器",
        "section": "unknown", "key": "edge_length_relative_tolerance",
        "type": "float", "minimum": 0.02, "maximum": 0.35, "step": 0.01,
    },
    "minimum_fill_ratio": {
        "label": "最小矩形填充率", "group": "白片求解器",
        "section": "unknown", "key": "minimum_accepted_fill_ratio",
        "type": "float", "minimum": 0.70, "maximum": 1.02, "step": 0.01,
    },
    "maximum_geometry_score": {
        "label": "最大几何误差", "group": "白片求解器",
        "section": "unknown", "key": "maximum_accepted_geometry_score",
        "type": "float", "minimum": 3.0, "maximum": 50.0, "step": 0.5,
    },
    "max_search_seconds": {
        "label": "主搜索时限 (s)", "group": "白片求解器",
        "section": "unknown", "key": "max_search_seconds",
        "type": "float", "minimum": 0.20, "maximum": 12.0, "step": 0.05,
    },
    "exact_search_seconds": {
        "label": "精确边搜索时限 (s)", "group": "白片求解器",
        "section": "unknown", "key": "exact_search_seconds",
        "type": "float", "minimum": 0.10, "maximum": 5.0, "step": 0.05,
    },
    "fallback_search_seconds": {
        "label": "补充搜索时限 (s)", "group": "白片求解器",
        "section": "unknown", "key": "fallback_search_seconds",
        "type": "float", "minimum": 0.10, "maximum": 8.0, "step": 0.05,
    },
    "max_pair_options_exact": {
        "label": "每对接缝候选上限", "group": "白片求解器",
        "section": "unknown", "key": "max_pair_options_exact",
        "type": "int", "minimum": 16, "maximum": 160, "step": 1,
    },
    "exact_search_node_floor": {
        "label": "精确搜索最低节点数", "group": "白片求解器",
        "section": "unknown", "key": "exact_search_node_floor",
        "type": "int", "minimum": 0, "maximum": 500, "step": 1,
    },
    "partial_search_node_floor": {
        "label": "部分边搜索最低节点数", "group": "白片求解器",
        "section": "unknown", "key": "partial_search_node_floor",
        "type": "int", "minimum": 0, "maximum": 1000, "step": 1,
    },
    "fallback_search_node_floor": {
        "label": "补充搜索最低节点数", "group": "白片求解器",
        "section": "unknown", "key": "fallback_search_node_floor",
        "type": "int", "minimum": 0, "maximum": 1500, "step": 1,
    },
    "max_search_nodes": {
        "label": "单阶段节点硬上限", "group": "白片求解器",
        "section": "unknown", "key": "max_search_nodes",
        "type": "int", "minimum": 500, "maximum": 100000, "step": 100,
    },
}


class ImageReplayCamera:
    """Repeat a still image so the complete upper-computer UI can be tested."""

    def __init__(self, path: str):
        self.path = path
        self.frame: np.ndarray | None = None

    def open(self) -> None:
        data = np.fromfile(self.path, dtype=np.uint8)
        self.frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if self.frame is None:
            raise CameraError(f"Cannot read replay image: {self.path}")

    def read(self) -> np.ndarray:
        if self.frame is None:
            self.open()
        time.sleep(1.0 / 15.0)
        assert self.frame is not None
        return self.frame.copy()

    def close(self) -> None:
        self.frame = None


def open_camera(
    source: str, settings: dict[str, Any]
) -> UsbCamera | MipiCamera | ImageReplayCamera:
    if Path(source).is_file():
        camera: UsbCamera | MipiCamera | ImageReplayCamera = ImageReplayCamera(
            source
        )
    elif source == "mipi":
        camera = MipiCamera(settings)
    elif source.startswith("usb:"):
        value = source.split(":", 1)[1]
        device: str | int = int(value) if value.isdigit() else value
        camera = UsbCamera(device, settings)
    else:
        raise CameraError("Live server source must be mipi or usb:<device>")
    camera.open()
    return camera


class LiveVision:
    def __init__(
        self,
        config: dict[str, Any],
        source: str,
        mode: str,
        source_region: str,
        analysis_interval: float,
        jpeg_quality: int,
        use_color_hints: bool = False,
        paper_color: str = "#00a8bd",
        piece_color: str = "#f4f4ee",
        config_path: str | Path | None = None,
    ):
        self.base_config = config
        self.config = deepcopy(config)
        self.source = source
        self.mode = mode
        self.source_region = source_region
        self.use_color_hints = use_color_hints
        self.paper_color = paper_color
        self.piece_color = piece_color
        self.config_path = (
            Path(config_path).resolve() if config_path is not None else None
        )
        self.card_setting_defaults = {
            key: deepcopy(config["unknown"].get(key))
            for key in CARD_SETTING_SPECS
        }
        self.white_setting_defaults = {
            setting_id: deepcopy(
                config[spec["section"]].get(spec["key"])
            )
            for setting_id, spec in WHITE_SETTING_SPECS.items()
        }
        self.white_settings = deepcopy(self.white_setting_defaults)
        persisted_white_settings = config.get("white_tuning", {})
        if isinstance(persisted_white_settings, dict):
            for setting_id, value in persisted_white_settings.items():
                if setting_id in WHITE_SETTING_SPECS:
                    self.white_settings[setting_id] = deepcopy(value)
        self.analysis_interval = max(0.05, analysis_interval)
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))
        self._apply_color_hints()
        self.pipeline = PuzzleVisionPipeline(self.config)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.new_frame = threading.Event()
        self.raw_frame: np.ndarray | None = None
        self.latest_jpeg: bytes | None = None
        self.latest_result: dict[str, Any] | None = None
        self.latest_motion_export: dict[str, Any] | None = None
        self.latest_motion_json: bytes | None = None
        self.latest_motion_csv: bytes | None = None
        self.export_dir = PROJECT_DIR / "exports"
        self.last_error: str | None = "正在等待首帧"
        self.input_mode = "camera"
        self.capture_fps = 0.0
        self.analysis_ms = 0.0
        self.frame_sequence = 0
        self.result_sequence = 0
        self.solution_cache_hits = 0
        self.geometry_lock_hits = 0
        self.solution_locked = False
        self.solution_lock_reason = "waiting_for_good_detection"
        self._solution_signature: np.ndarray | None = None
        self._scene_change_streak = 0
        self._unchanged_cache_run = 0
        self._last_scene_change_ratio = 1.0
        # Run a complete detector pass periodically even when the scene is
        # unchanged.  Cheap signature passes keep the live UI responsive,
        # while periodic full passes ensure recognition never freezes.
        self._maximum_unchanged_cache_runs = 2
        self._scene_change_ratio_threshold = 0.035
        self._scene_change_confirmations = 2
        self._quality_confirmation_count = 0
        self._required_quality_confirmations = 3
        self.started_at = time.time()
        self.capture_thread = threading.Thread(
            target=self._capture_loop, name="capture", daemon=True
        )
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop, name="analysis", daemon=True
        )

    def start(self) -> None:
        self.capture_thread.start()
        self.analysis_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.new_frame.set()
        self.capture_thread.join(timeout=3.0)
        self.analysis_thread.join(timeout=3.0)

    def _apply_color_hints(self) -> None:
        self.config = deepcopy(self.base_config)
        # Ordinary-white tuning is deliberately a mode-local overlay.  The
        # underlying paper/segmentation/solver defaults remain untouched, so
        # changing illumination or piece-size thresholds cannot perturb the
        # fixed-template or playing-card pipelines.
        if self.mode == "unknown-white":
            for setting_id, value in self.white_settings.items():
                spec = WHITE_SETTING_SPECS[setting_id]
                self.config[spec["section"]][spec["key"]] = value
        if self.use_color_hints:
            self.config["paper"]["color_bgr"] = hex_to_bgr(self.paper_color)
            self.config["segmentation"]["piece_color_bgr"] = hex_to_bgr(
                self.piece_color
            )

    def update_settings(
        self,
        mode: str,
        source_region: str,
        use_color_hints: bool | None = None,
        paper_color: str | None = None,
        piece_color: str | None = None,
    ) -> None:
        if mode not in PuzzleVisionPipeline.MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        if source_region not in ("upper", "lower", "auto"):
            raise ValueError(f"Unsupported source region: {source_region}")
        with self.lock:
            self.mode = mode
            self.source_region = source_region
            if use_color_hints is not None:
                self.use_color_hints = bool(use_color_hints)
            if paper_color is not None:
                hex_to_bgr(paper_color)
                self.paper_color = paper_color
            if piece_color is not None:
                hex_to_bgr(piece_color)
                self.piece_color = piece_color
            self._apply_color_hints()
            self.pipeline = PuzzleVisionPipeline(self.config)
            self.latest_result = None
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self._solution_signature = None
            self.solution_locked = False
            self.solution_lock_reason = "settings_changed"
            self._scene_change_streak = 0
            self._unchanged_cache_run = 0
            self._last_scene_change_ratio = 1.0
            self._quality_confirmation_count = 0
            self.last_error = "设置已更新，正在重新计算"

    def card_settings_payload(self) -> dict[str, Any]:
        with self.lock:
            values = {
                key: deepcopy(self.base_config["unknown"].get(key))
                for key in CARD_SETTING_SPECS
            }
        return {
            "ok": True,
            "values": values,
            "defaults": deepcopy(self.card_setting_defaults),
            "schema": deepcopy(CARD_SETTING_SPECS),
            "persisted": self.config_path is not None,
        }

    def _validate_card_settings(
        self, raw_settings: Any
    ) -> dict[str, int | float]:
        if not isinstance(raw_settings, dict):
            raise ValueError("card settings must be a JSON object")
        unknown_keys = sorted(set(raw_settings) - set(CARD_SETTING_SPECS))
        if unknown_keys:
            raise ValueError(
                "Unsupported card setting: " + ", ".join(unknown_keys)
            )
        validated: dict[str, int | float] = {}
        for key, raw_value in raw_settings.items():
            spec = CARD_SETTING_SPECS[key]
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{spec['label']}必须是数字") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"{spec['label']}必须是有限数字")
            minimum = float(spec["minimum"])
            maximum = float(spec["maximum"])
            if numeric < minimum or numeric > maximum:
                raise ValueError(
                    f"{spec['label']}必须在 {minimum:g}～{maximum:g} 之间"
                )
            validated[key] = (
                int(round(numeric))
                if spec["type"] == "int"
                else float(numeric)
            )
        combined = {
            key: self.base_config["unknown"].get(key)
            for key in CARD_SETTING_SPECS
        }
        combined.update(validated)
        if float(combined["card_minimum_long_side_mm"]) >= float(
            combined["card_maximum_long_side_mm"]
        ):
            raise ValueError("扑克牌长边最小值必须小于最大值")
        if float(combined["card_minimum_short_side_mm"]) >= float(
            combined["card_maximum_short_side_mm"]
        ):
            raise ValueError("扑克牌短边最小值必须小于最大值")
        if float(combined["card_rounded_chord_min_mm"]) >= float(
            combined["card_rounded_chord_max_mm"]
        ):
            raise ValueError("圆角弦最小值必须小于最大值")
        if float(combined["card_exact_search_seconds"]) > float(
            combined["card_search_seconds"]
        ):
            raise ValueError("精确边搜索时限不能大于总搜索时限")
        return validated

    def _persist_card_settings(
        self, values: dict[str, int | float]
    ) -> None:
        if self.config_path is None:
            return
        try:
            document = json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
            if not isinstance(document, dict):
                raise ValueError("config root is not an object")
            unknown = document.setdefault("unknown", {})
            if not isinstance(unknown, dict):
                raise ValueError("config unknown section is not an object")
            unknown.update(values)
            temporary = self.config_path.with_name(
                self.config_path.name + ".tmp"
            )
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.config_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"保存扑克牌参数失败：{exc}") from exc

    def update_card_settings(
        self, raw_settings: Any, reset: bool = False
    ) -> dict[str, Any]:
        values = (
            deepcopy(self.card_setting_defaults)
            if reset
            else self._validate_card_settings(raw_settings)
        )
        # Validate reset defaults too, while ignoring missing legacy keys.
        values = self._validate_card_settings(
            {
                key: value
                for key, value in values.items()
                if value is not None
            }
        )
        self._persist_card_settings(values)
        with self.lock:
            self.base_config["unknown"].update(values)
            self._apply_color_hints()
            self.pipeline = PuzzleVisionPipeline(self.config)
            self.latest_result = None
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self._solution_signature = None
            self.solution_locked = False
            self.solution_lock_reason = "card_settings_changed"
            self._scene_change_streak = 0
            self._unchanged_cache_run = 0
            self._last_scene_change_ratio = 1.0
            self._quality_confirmation_count = 0
            self.last_error = "扑克牌参数已更新，正在重新计算"
        return self.card_settings_payload()

    def white_settings_payload(self) -> dict[str, Any]:
        with self.lock:
            values = deepcopy(self.white_settings)
        return {
            "ok": True,
            "values": values,
            "defaults": deepcopy(self.white_setting_defaults),
            "schema": deepcopy(WHITE_SETTING_SPECS),
            "persisted": self.config_path is not None,
        }

    def _validate_white_settings(
        self, raw_settings: Any
    ) -> dict[str, int | float]:
        if not isinstance(raw_settings, dict):
            raise ValueError("white-piece settings must be a JSON object")
        unsupported = sorted(set(raw_settings) - set(WHITE_SETTING_SPECS))
        if unsupported:
            raise ValueError(
                "Unsupported white-piece setting: "
                + ", ".join(unsupported)
            )
        validated: dict[str, int | float] = {}
        for setting_id, raw_value in raw_settings.items():
            spec = WHITE_SETTING_SPECS[setting_id]
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{spec['label']}必须是数字") from exc
            if not np.isfinite(numeric):
                raise ValueError(f"{spec['label']}必须是有限数字")
            minimum = float(spec["minimum"])
            maximum = float(spec["maximum"])
            if numeric < minimum or numeric > maximum:
                raise ValueError(
                    f"{spec['label']}必须在 {minimum:g}～{maximum:g} 之间"
                )
            validated[setting_id] = (
                int(round(numeric))
                if spec["type"] == "int"
                else float(numeric)
            )
        combined = deepcopy(self.white_settings)
        combined.update(validated)
        for minimum_key, maximum_key, label in (
            ("target_min_width_mm", "target_max_width_mm", "目标长边"),
            ("target_min_height_mm", "target_max_height_mm", "目标短边"),
            ("piece_min_area_mm2", "piece_max_area_mm2", "单片面积"),
        ):
            if float(combined[minimum_key]) >= float(combined[maximum_key]):
                raise ValueError(f"{label}最小值必须小于最大值")
        if float(combined["exact_search_seconds"]) > float(
            combined["max_search_seconds"]
        ):
            raise ValueError("精确边搜索时限不能大于总搜索时限")
        return validated

    def _persist_white_settings(
        self, values: dict[str, int | float]
    ) -> None:
        if self.config_path is None:
            return
        try:
            document = json.loads(
                self.config_path.read_text(encoding="utf-8")
            )
            if not isinstance(document, dict):
                raise ValueError("config root is not an object")
            section = document.setdefault("white_tuning", {})
            if not isinstance(section, dict):
                raise ValueError("config white_tuning section is not an object")
            section.update(values)
            temporary = self.config_path.with_name(
                self.config_path.name + ".tmp"
            )
            temporary.write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.config_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"保存白片参数失败：{exc}") from exc

    def update_white_settings(
        self, raw_settings: Any, reset: bool = False
    ) -> dict[str, Any]:
        values = (
            deepcopy(self.white_setting_defaults)
            if reset
            else self._validate_white_settings(raw_settings)
        )
        values = self._validate_white_settings(
            {
                setting_id: value
                for setting_id, value in values.items()
                if value is not None
            }
        )
        self._persist_white_settings(values)
        with self.lock:
            self.white_settings.update(values)
            self._apply_color_hints()
            self.pipeline = PuzzleVisionPipeline(self.config)
            self.latest_result = None
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self._solution_signature = None
            self.solution_locked = False
            self.solution_lock_reason = "white_settings_changed"
            self._scene_change_streak = 0
            self._unchanged_cache_run = 0
            self._last_scene_change_ratio = 1.0
            self._quality_confirmation_count = 0
            self.last_error = "纯色白片参数已更新，正在重新计算"
        return self.white_settings_payload()

    @staticmethod
    def _scene_signature(
        frame: np.ndarray,
        result: dict[str, Any],
    ) -> np.ndarray:
        """Return a lighting-tolerant foreground signature in A4 millimetres."""

        paper = result["paper"]
        ppm = float(paper["pixels_per_mm"])
        camera_to_paper = np.asarray(
            paper["homography_camera_to_paper_px"], dtype=np.float64
        )
        to_mm = np.asarray(
            [[1.0 / ppm, 0.0, 0.0], [0.0, 1.0 / ppm, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        paper_bgr = cv2.warpPerspective(
            frame,
            to_mm @ camera_to_paper,
            (210, 297),
            flags=cv2.INTER_AREA,
        )
        divider = int(
            round(float(paper["divider"]["detected_y_mm"]))
        )
        margin = 3
        if result.get("source_region") == "lower":
            source = paper_bgr[
                min(296, divider + margin) : 294, 3:207
            ]
        else:
            source = paper_bgr[
                3 : max(4, divider - margin), 3:207
            ]
        source = cv2.GaussianBlur(source, (5, 5), 0)
        lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        background = np.median(lab.reshape(-1, 3), axis=0)
        distance = np.linalg.norm(lab - background, axis=2)
        distance_u8 = np.clip(distance * 4.0, 0, 255).astype(np.uint8)
        otsu, _ = cv2.threshold(
            distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        threshold = float(np.clip(otsu / 4.0, 10.0, 30.0))
        mask = (distance >= threshold).astype(np.uint8) * 255
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        cleaned = np.zeros_like(mask)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= 12:
                cleaned[labels == label] = 255
        return cleaned

    @staticmethod
    def _signature_change_ratio(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        if first.shape != second.shape or first.size == 0:
            return 1.0
        kernel = np.ones((3, 3), dtype=np.uint8)
        first_dilated = cv2.dilate(first, kernel)
        second_dilated = cv2.dilate(second, kernel)
        first_edges = first > 0
        second_edges = second > 0
        unmatched = np.count_nonzero(
            first_edges & (second_dilated == 0)
        ) + np.count_nonzero(second_edges & (first_dilated == 0))
        edge_count = np.count_nonzero(first_edges) + np.count_nonzero(
            second_edges
        )
        return float(unmatched / max(edge_count, 1))

    @staticmethod
    def _sample_polygon(points: Any, count: int = 48) -> np.ndarray:
        polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(polygon) < 2:
            return polygon.copy()
        following = np.roll(polygon, -1, axis=0)
        vectors = following - polygon
        lengths = np.linalg.norm(vectors, axis=1)
        perimeter = float(np.sum(lengths))
        if perimeter <= 1e-9:
            return np.repeat(polygon[:1], count, axis=0)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        positions = np.linspace(0.0, perimeter, count, endpoint=False)
        samples = np.empty((count, 2), dtype=np.float64)
        for sample_index, position in enumerate(positions):
            edge_index = min(
                int(np.searchsorted(cumulative, position, side="right") - 1),
                len(polygon) - 1,
            )
            fraction = (
                position - cumulative[edge_index]
            ) / max(float(lengths[edge_index]), 1e-9)
            samples[sample_index] = (
                polygon[edge_index] + fraction * vectors[edge_index]
            )
        return samples

    @classmethod
    def _match_source_geometry(
        cls,
        previous: dict[str, Any],
        current: dict[str, Any],
        centroid_tolerance_mm: float = 3.0,
        contour_tolerance_mm: float = 3.0,
    ) -> list[int] | None:
        """Map unchanged physical pieces without relying on detector IDs."""

        old_pieces = list(previous.get("pieces", []))
        new_pieces = list(current.get("pieces", []))
        if (
            not old_pieces
            or len(old_pieces) != len(new_pieces)
            or previous.get("source_region") != current.get("source_region")
        ):
            return None
        count = len(old_pieces)
        old_samples = [
            cls._sample_polygon(item.get("polygon_mm", [])) for item in old_pieces
        ]
        new_samples = [
            cls._sample_polygon(item.get("polygon_mm", [])) for item in new_pieces
        ]
        costs = np.full((count, count), np.inf, dtype=np.float64)
        for old_index, old_piece in enumerate(old_pieces):
            old_centroid = np.asarray(
                old_piece.get("centroid_mm", []), dtype=np.float64
            )
            old_area = float(old_piece.get("area_mm2", 0.0))
            if old_centroid.shape != (2,) or len(old_samples[old_index]) < 3:
                continue
            for new_index, new_piece in enumerate(new_pieces):
                new_centroid = np.asarray(
                    new_piece.get("centroid_mm", []), dtype=np.float64
                )
                new_area = float(new_piece.get("area_mm2", 0.0))
                if new_centroid.shape != (2,) or len(new_samples[new_index]) < 3:
                    continue
                centroid_error = float(np.linalg.norm(old_centroid - new_centroid))
                area_error = abs(new_area - old_area) / max(old_area, 1.0)
                if centroid_error > centroid_tolerance_mm or area_error > 0.12:
                    continue
                distances = np.linalg.norm(
                    old_samples[old_index][:, None, :]
                    - new_samples[new_index][None, :, :],
                    axis=2,
                )
                boundary_error = max(
                    float(np.percentile(np.min(distances, axis=1), 90)),
                    float(np.percentile(np.min(distances, axis=0), 90)),
                )
                if boundary_error <= contour_tolerance_mm:
                    costs[old_index, new_index] = (
                        centroid_error + boundary_error + 8.0 * area_error
                    )
        best_mapping: list[int] | None = None
        best_cost = float("inf")
        for permutation in itertools.permutations(range(count)):
            values = [costs[index, permutation[index]] for index in range(count)]
            if all(np.isfinite(value) for value in values):
                total = float(sum(values))
                if total < best_cost:
                    best_cost = total
                    best_mapping = list(permutation)
        return best_mapping

    def _result_is_lockable(self, result: dict[str, Any]) -> tuple[bool, str]:
        """Require a genuinely good recognition before freezing robot poses."""

        candidate_ready = result.get(
            "recognition_candidate_ready", result.get("motion_ready")
        )
        if not candidate_ready:
            return False, "solver_not_ready"
        pieces = list(result.get("pieces", []))
        plan = list(result.get("plan", []))
        expected = 4 if result.get("mode") == "fixed" else None
        if expected is not None and len(pieces) != expected:
            return False, "fixed_piece_count"
        if expected is None and not (2 <= len(pieces) <= 4):
            return False, "advanced_piece_count"
        if len(plan) != len(pieces):
            return False, "incomplete_plan"
        recognition = result.get("recognition", {})
        if int(recognition.get("segmentation_component_count", len(pieces))) != len(
            pieces
        ):
            return False, "segmentation_components_incomplete"
        if float(recognition.get("mask_area_error_ratio", 0.0)) > 0.16:
            return False, "segmentation_area_unstable"
        solver = result.get("solver", {})
        if (
            solver.get("mirror_allowed") is not False
            or solver.get("piece_reflection_used", False)
            or any(item.get("mirrored", False) for item in plan)
        ):
            return False, "reflection_forbidden"
        if not solver.get("target_non_overlapping", True):
            return False, "target_overlap"
        card_mode = result.get("mode") == "unknown-pattern"
        minimum_stable_edge = (
            float(
                self.config.get("unknown", {}).get(
                    "card_rounded_chord_min_mm", 2.0
                )
            )
            if card_mode
            else 7.0
        )
        maximum_rounded_chord = float(
            self.config.get("unknown", {}).get(
                "card_rounded_chord_max_mm", 8.0
            )
        )
        for piece in pieces:
            polygon = np.asarray(piece.get("polygon_mm", []), dtype=np.float64)
            edges = np.asarray(piece.get("edge_lengths_mm", []), dtype=np.float64)
            if (
                polygon.ndim != 2
                or polygon.shape[1:] != (2,)
                or not (3 <= len(polygon) <= 5)
                or not np.all(np.isfinite(polygon))
                or len(edges) != len(polygon)
                or float(np.min(edges, initial=np.inf))
                < minimum_stable_edge
            ):
                return False, "unstable_piece_contour"
            if card_mode:
                # One fragment may own one or two original card corners.  Each
                # rounded corner appears in the structural polygon as a short
                # chord; it is valid geometry, not a noisy extra cut edge.
                # More than two sub-7-mm edges is still treated as a damaged
                # contour caused by print/glare touching the segmentation
                # boundary.
                rounded_chords = edges[
                    (edges >= minimum_stable_edge)
                    & (edges < min(7.0, maximum_rounded_chord + 1e-9))
                ]
                if len(rounded_chords) > 2:
                    return False, "too_many_rounded_corner_chords"
        if result.get("mode") == "fixed":
            if float(solver.get("max_match_error_mm", np.inf)) > 8.0:
                return False, "fixed_match_not_precise"
        else:
            fill = float(solver.get("fill_ratio", 0.0))
            geometry = float(solver.get("geometry_score", np.inf))
            if not (0.94 <= fill <= 1.025):
                return False, "rectangle_not_filled"
            if geometry > 8.0:
                return False, "geometry_not_precise"
            if solver.get("solution_quality") not in (
                "high",
                "taught_template_match",
            ):
                return False, "solution_not_high_quality"
        return True, "quality_confirmed"

    @staticmethod
    def _target_quality_score(result: dict[str, Any]) -> float:
        """Comparable lower-is-better score for replacing a cached tiling."""

        solver = result.get("solver", {})
        if not result.get(
            "recognition_candidate_ready", result.get("motion_ready")
        ):
            return float("inf")
        if result.get("mode") == "fixed":
            return float(solver.get("max_match_error_mm", np.inf))
        fill_error = 100.0 * abs(1.0 - float(solver.get("fill_ratio", 0.0)))
        geometry_error = float(
            solver.get(
                "geometry_score",
                solver.get("max_match_error_mm", np.inf),
            )
        )
        return fill_error + geometry_error

    @classmethod
    def _new_target_is_materially_better(
        cls,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> bool:
        old_score = cls._target_quality_score(previous)
        new_score = cls._target_quality_score(current)
        if not np.isfinite(old_score):
            return np.isfinite(new_score)
        improvement_needed = max(0.75, 0.15 * old_score)
        return new_score + improvement_needed < old_score

    def _add_recognition_diagnostics(
        self,
        result: dict[str, Any],
        mask: np.ndarray,
    ) -> None:
        ppm = float(result["paper"]["pixels_per_mm"])
        minimum_px = (
            0.60
            * float(self.config["segmentation"]["min_area_mm2"])
            * ppm
            * ppm
        )
        maximum_px = (
            2.0
            * float(self.config["segmentation"]["max_area_mm2"])
            * ppm
            * ppm
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8)
        )
        component_areas = [
            float(stats[label, cv2.CC_STAT_AREA])
            for label in range(1, count)
            if minimum_px <= float(stats[label, cv2.CC_STAT_AREA]) <= maximum_px
        ]
        mask_area_mm2 = sum(component_areas) / max(ppm * ppm, 1e-9)
        polygon_area_mm2 = sum(
            float(piece.get("area_mm2", 0.0))
            for piece in result.get("pieces", [])
        )
        result["recognition"] = {
            "segmentation_component_count": len(component_areas),
            "mask_area_mm2": round(mask_area_mm2, 3),
            "polygon_area_mm2": round(polygon_area_mm2, 3),
            "mask_area_error_ratio": round(
                abs(mask_area_mm2 - polygon_area_mm2)
                / max(polygon_area_mm2, 1.0),
                5,
            ),
        }

    @staticmethod
    def _reuse_stable_target(
        previous: dict[str, Any],
        current: dict[str, Any],
        mapping: list[int],
    ) -> dict[str, Any]:
        """Keep the chosen target tiling while refreshing source observations."""

        merged = deepcopy(current)
        old_pieces = list(previous.get("pieces", []))
        new_pieces = list(current.get("pieces", []))
        old_to_new_id = {
            old_pieces[old_index]["piece_id"]: new_pieces[new_index]["piece_id"]
            for old_index, new_index in enumerate(mapping)
        }
        new_by_id = {item["piece_id"]: item for item in new_pieces}
        stable_plan: list[dict[str, Any]] = []
        for old_pose in previous.get("plan", []):
            old_id = old_pose.get("piece_id")
            new_id = old_to_new_id.get(old_id)
            if new_id is None:
                return previous
            pose = deepcopy(old_pose)
            pose["piece_id"] = new_id
            pose["pick_mm"] = deepcopy(new_by_id[new_id].get("pickup_mm"))
            stable_plan.append(pose)
        merged["plan"] = stable_plan
        merged["solver"] = deepcopy(previous.get("solver", {}))
        merged["motion_ready"] = bool(previous.get("motion_ready"))
        return merged

    @staticmethod
    def _paper_mm_to_camera_px(
        point_mm: list[float],
        result: dict[str, Any],
    ) -> list[float]:
        paper = result["paper"]
        ppm = float(paper["pixels_per_mm"])
        camera_to_paper = np.asarray(
            paper["homography_camera_to_paper_px"], dtype=np.float64
        )
        paper_to_camera = np.linalg.inv(camera_to_paper)
        point = (
            np.asarray(point_mm, dtype=np.float32).reshape(1, 1, 2) * ppm
        )
        projected = cv2.perspectiveTransform(
            point, paper_to_camera
        ).reshape(2)
        return [round(float(value), 2) for value in projected]

    def _build_motion_export(
        self, result: dict[str, Any]
    ) -> tuple[dict[str, Any], bytes, bytes]:
        pieces = {
            item["piece_id"]: item for item in result.get("pieces", [])
        }
        commands: list[dict[str, Any]] = []
        for sequence, pose in enumerate(result.get("plan", []), start=1):
            piece = pieces.get(pose["piece_id"], {})
            pick_mm = [float(value) for value in pose["pick_mm"]]
            place_mm = [float(value) for value in pose["place_mm"]]
            commands.append(
                {
                    "sequence": sequence,
                    "piece_id": pose["piece_id"],
                    "pick_a4_mm": pick_mm,
                    "pick_camera_px": self._paper_mm_to_camera_px(
                        pick_mm, result
                    ),
                    "source_centroid_a4_mm": piece.get("centroid_mm"),
                    "place_a4_mm": place_mm,
                    "place_camera_px": self._paper_mm_to_camera_px(
                        place_mm, result
                    ),
                    "rotate_deg_clockwise": float(pose["rotate_deg"]),
                    "mirrored": bool(pose.get("mirrored", False)),
                    "template_id": pose.get("template_id"),
                }
            )
        export = {
            "schema": "a4-puzzle-motion-plan/v1",
            "generated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime()
            ),
            "motion_ready": bool(result.get("motion_ready")),
            "mode": result.get("mode"),
            "coordinate_frame": result.get("coordinate_frame"),
            "paper": result.get("paper"),
            "source_region": result.get("source_region"),
            "destination_region": result.get("destination_region"),
            "target_origin_a4_mm": result.get("solver", {}).get(
                "target_origin_mm"
            ),
            "target_size_mm": result.get("solver", {}).get(
                "target_size_mm"
            ),
            "commands": commands,
        }
        json_bytes = json.dumps(
            export, ensure_ascii=False, indent=2
        ).encode("utf-8")
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(
            [
                "sequence",
                "piece_id",
                "pick_x_a4_mm",
                "pick_y_a4_mm",
                "pick_x_camera_px",
                "pick_y_camera_px",
                "place_x_a4_mm",
                "place_y_a4_mm",
                "place_x_camera_px",
                "place_y_camera_px",
                "rotate_deg_clockwise",
                "mirrored",
            ]
        )
        for command in commands:
            writer.writerow(
                [
                    command["sequence"],
                    command["piece_id"],
                    *command["pick_a4_mm"],
                    *command["pick_camera_px"],
                    *command["place_a4_mm"],
                    *command["place_camera_px"],
                    command["rotate_deg_clockwise"],
                    str(command["mirrored"]).lower(),
                ]
            )
        csv_bytes = ("\ufeff" + text.getvalue()).encode("utf-8")
        return export, json_bytes, csv_bytes

    def _publish_motion_export(self, result: dict[str, Any]) -> None:
        export, json_bytes, csv_bytes = self._build_motion_export(result)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        (self.export_dir / "latest_motion_plan.json").write_bytes(json_bytes)
        (self.export_dir / "latest_motion_plan.csv").write_bytes(csv_bytes)
        with self.lock:
            self.latest_motion_export = export
            self.latest_motion_json = json_bytes
            self.latest_motion_csv = csv_bytes

    def _capture_loop(self) -> None:
        camera: UsbCamera | MipiCamera | ImageReplayCamera | None = None
        frame_count = 0
        fps_started = time.perf_counter()
        try:
            camera = open_camera(self.source, self.config["camera"])
            while not self.stop_event.is_set():
                frame = camera.read()
                frame_count += 1
                elapsed = time.perf_counter() - fps_started
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    fps_started = time.perf_counter()
                    with self.lock:
                        self.capture_fps = fps

                with self.lock:
                    self.raw_frame = frame.copy()
                    result = self.latest_result
                    error = self.last_error
                    mode = self.mode
                    source_region = self.source_region
                    input_mode = self.input_mode
                    self.frame_sequence += 1
                self.new_frame.set()

                if input_mode != "camera":
                    continue
                display = (
                    self.pipeline.draw_camera_overlay(frame, result)
                    if result is not None
                    else frame.copy()
                )
                self._draw_runtime_status(display, mode, source_region, error)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    display,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
        except Exception as exc:
            with self.lock:
                self.last_error = f"摄像头错误: {exc}"
        finally:
            if camera is not None:
                camera.close()

    def _publish_uploaded_progress(
        self,
        frame: np.ndarray,
        mode: str,
        source_region: str,
        progress_result: dict[str, Any],
    ) -> None:
        """Show A4/piece geometry while an uploaded image is being solved."""

        display = self.pipeline.draw_camera_overlay(frame, progress_result)
        self._draw_runtime_status(
            display,
            mode,
            source_region,
            "A4与碎片已定位，正在快速拼接",
        )
        ok, encoded = cv2.imencode(
            ".jpg",
            display,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        with self.lock:
            if self.input_mode == "uploaded_image":
                self.latest_result = progress_result
                self.last_error = "A4与碎片已定位，正在快速拼接"
                if ok:
                    self.latest_jpeg = encoded.tobytes()

    def _analysis_loop(self) -> None:
        last_analyzed_sequence = -1
        while not self.stop_event.is_set():
            self.new_frame.wait(timeout=0.5)
            self.new_frame.clear()
            with self.lock:
                frame = None if self.raw_frame is None else self.raw_frame.copy()
                sequence = self.frame_sequence
                mode = self.mode
                source_region = self.source_region
                input_mode = self.input_mode
                previous_result = self.latest_result
                previous_signature = (
                    None
                    if self._solution_signature is None
                    else self._solution_signature.copy()
                )
            if frame is None or sequence == last_analyzed_sequence:
                continue
            if input_mode != "camera":
                last_analyzed_sequence = sequence
                continue
            started = time.perf_counter()
            force_paper_redetection = False
            scene_was_unchanged = False
            if (
                previous_result is not None
                and previous_signature is not None
                and bool(previous_result.get("target_layout_locked", False))
                and bool(previous_result.get("motion_ready", False))
            ):
                try:
                    current_signature = self._scene_signature(
                        frame, previous_result
                    )
                    change_ratio = self._signature_change_ratio(
                        previous_signature, current_signature
                    )
                except (cv2.error, KeyError, TypeError, ValueError):
                    change_ratio = 1.0
                with self.lock:
                    self._last_scene_change_ratio = change_ratio
                if change_ratio <= self._scene_change_ratio_threshold:
                    scene_was_unchanged = True
                    with self.lock:
                        self._scene_change_streak = 0
                        use_cache = (
                            self._unchanged_cache_run
                            < self._maximum_unchanged_cache_runs
                        )
                        if use_cache:
                            self._unchanged_cache_run += 1
                            self.solution_cache_hits += 1
                            self.solution_lock_reason = (
                                "unchanged_scene_target_preserved"
                            )
                            self.analysis_ms = (
                                time.perf_counter() - started
                            ) * 1000.0
                        else:
                            self._unchanged_cache_run = 0
                    if use_cache:
                        last_analyzed_sequence = sequence
                        remaining = self.analysis_interval - (
                            time.perf_counter() - started
                        )
                        if remaining > 0:
                            self.stop_event.wait(remaining)
                        continue
                else:
                    with self.lock:
                        self._unchanged_cache_run = 0
                        self._scene_change_streak += 1
                        change_confirmed = (
                            self._scene_change_streak
                            >= self._scene_change_confirmations
                        )
                        force_paper_redetection = change_confirmed
                        if not change_confirmed:
                            self.solution_cache_hits += 1
                            self.solution_lock_reason = (
                                "scene_change_confirmation_pending"
                            )
                            self.analysis_ms = (
                                time.perf_counter() - started
                            ) * 1000.0
                    if not change_confirmed:
                        last_analyzed_sequence = sequence
                        remaining = self.analysis_interval - (
                            time.perf_counter() - started
                        )
                        if remaining > 0:
                            self.stop_event.wait(remaining)
                        continue
            if force_paper_redetection:
                self.pipeline.invalidate_paper_cache()
            try:
                def publish_progress(progress_result: dict[str, Any]) -> None:
                    if (
                        previous_result is not None
                        and bool(previous_result.get("motion_ready", False))
                        and not force_paper_redetection
                    ):
                        # Periodic verification of an unchanged, already
                        # solved scene must not replace the stable target with
                        # a transient "detecting" frame.
                        return
                    display = self.pipeline.draw_camera_overlay(
                        frame, progress_result
                    )
                    self._draw_runtime_status(
                        display,
                        mode,
                        source_region,
                        "A4与碎片已定位，正在快速拼接",
                    )
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        display,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    with self.lock:
                        self.latest_result = progress_result
                        self.last_error = (
                            "A4与碎片已定位，正在快速拼接"
                        )
                        if ok:
                            self.latest_jpeg = encoded.tobytes()

                result, _, mask, _ = self.pipeline.analyze(
                    frame,
                    mode,
                    source_region=source_region,
                    allow_unsolved=True,
                    progress_callback=publish_progress,
                )
                self._add_recognition_diagnostics(result, mask)
                result["recognition_candidate_ready"] = bool(
                    result.get("motion_ready")
                )
                current_motion_ready = bool(result.get("motion_ready"))
                if not current_motion_ready:
                    # A stale but still divider-valid homography can include a
                    # narrow strip outside the A4 and make the solver reject
                    # every later frame.  Do not let a failed recognition
                    # poison the live loop: the next frame must redetect the
                    # complete A4 boundary.
                    self.pipeline.invalidate_paper_cache()
                current_lockable, lock_reason = self._result_is_lockable(result)
                mapping = (
                    self._match_source_geometry(previous_result, result)
                    if previous_result is not None
                    else None
                )
                previous_target_locked = bool(
                    previous_result
                    and previous_result.get("target_layout_locked", False)
                )
                geometry_reused = (
                    mapping is not None and previous_target_locked
                )
                signature_reused = (
                    previous_target_locked
                    and scene_was_unchanged
                    and mapping is None
                )
                target_replaced_by_better = False
                if (
                    geometry_reused
                    and not scene_was_unchanged
                    and current_lockable
                    and self._new_target_is_materially_better(
                        previous_result, result
                    )
                ):
                    geometry_reused = False
                    target_replaced_by_better = True
                if signature_reused:
                    # The lighting-tolerant full-scene signature is the final
                    # authority for "the pieces did not move".  A periodic
                    # contour pass may shift a printed/glare boundary enough
                    # that polygon-to-polygon matching fails, but that must
                    # never select a different rectangle for an unchanged
                    # physical scene.  Keep the entire last validated result;
                    # a genuine move is handled by the confirmed scene-change
                    # branch above.
                    result = deepcopy(previous_result)
                    result["recognition_candidate_ready"] = (
                        current_motion_ready
                    )
                    current_motion_ready = bool(result.get("motion_ready"))
                    result["target_layout_locked"] = True
                    locked = True
                    lock_reason = "target_layout_preserved_by_scene"
                elif geometry_reused:
                    result = self._reuse_stable_target(
                        previous_result, result, mapping
                    )
                    # Keep recording whether this particular verification
                    # frame solved independently, but do not invalidate an
                    # already accepted robot plan when the source polygons
                    # still match the same unmoved physical pieces.  This is
                    # the core anti-flicker rule: illumination/glare may make
                    # one card frame fail while the last strictly validated,
                    # non-overlapping rigid placement remains valid.
                    result["recognition_candidate_ready"] = current_motion_ready
                    current_motion_ready = bool(result.get("motion_ready"))
                    result["target_layout_locked"] = True
                    locked = True
                    lock_reason = "target_layout_preserved"
                else:
                    locked = current_lockable
                    result["target_layout_locked"] = locked
                    if target_replaced_by_better:
                        lock_reason = "better_target_layout_selected"
                    elif current_lockable:
                        lock_reason = "target_layout_selected"
                result["motion_ready"] = current_motion_ready
                duration_ms = (time.perf_counter() - started) * 1000.0
                should_publish = current_motion_ready
                committed = False
                with self.lock:
                    if (
                        mode == self.mode
                        and source_region == self.source_region
                        and self.input_mode == "camera"
                    ):
                        committed = True
                        self.latest_result = result
                        self._solution_signature = self._scene_signature(
                            frame, result
                        )
                        self._scene_change_streak = 0
                        self._unchanged_cache_run = 0
                        self._quality_confirmation_count = (
                            self._required_quality_confirmations
                            if locked
                            else 0
                        )
                        self.solution_locked = locked
                        self.solution_lock_reason = lock_reason
                        if geometry_reused:
                            self.geometry_lock_hits += 1
                        self.last_error = None
                        self.analysis_ms = duration_ms
                        self.result_sequence += 1
                        if not current_motion_ready:
                            self.latest_motion_export = None
                            self.latest_motion_json = None
                            self.latest_motion_csv = None
                if should_publish and committed:
                    self._publish_motion_export(result)
            except (DetectionError, SolveError, RuntimeError, ValueError) as exc:
                # Detection errors are often caused by movement of the A4 or
                # camera.  Force a clean paper search on the next frame so the
                # upper computer can recover without a service restart.
                self.pipeline.invalidate_paper_cache()
                duration_ms = (time.perf_counter() - started) * 1000.0
                with self.lock:
                    target_was_locked = bool(
                        self.latest_result
                        and self.latest_result.get(
                            "target_layout_locked", False
                        )
                    )
                    if self.latest_result is not None:
                        self.latest_result = deepcopy(self.latest_result)
                        self.latest_result["motion_ready"] = False
                        self.latest_result[
                            "recognition_candidate_ready"
                        ] = False
                        self.latest_result[
                            "target_layout_locked"
                        ] = target_was_locked
                    self.latest_motion_export = None
                    self.latest_motion_json = None
                    self.latest_motion_csv = None
                    if not target_was_locked:
                        self._solution_signature = None
                    self._unchanged_cache_run = 0
                    self._quality_confirmation_count = 0
                    self.solution_locked = target_was_locked
                    self.solution_lock_reason = (
                        "detection_failed_target_preserved"
                        if target_was_locked
                        else "analysis_failed"
                    )
                    self.last_error = str(exc)
                    self.analysis_ms = duration_ms
            last_analyzed_sequence = sequence
            remaining = self.analysis_interval - (time.perf_counter() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)

    def analyze_uploaded_image(self, payload: bytes) -> dict[str, Any]:
        if not payload:
            raise ValueError("上传图片为空")
        if len(payload) > 25 * 1024 * 1024:
            raise ValueError("图片超过 25 MB 限制")
        encoded = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码图片，请选择 JPG、PNG 或 BMP")

        with self.lock:
            self.input_mode = "uploaded_image"
            mode = self.mode
            source_region = self.source_region
            pipeline = self.pipeline
            # Uploaded photographs do not share the live camera's pixel
            # coordinate system. Reusing live A4 corners can warp a valid
            # black sheet onto the wrong area and reduce an obvious bright
            # divider to a near-zero row contrast. Every uploaded image must
            # therefore redetect its own A4 boundary.
            pipeline.invalidate_paper_cache()
            self.latest_result = None
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self.last_error = "正在识别上传图片"

        started = time.perf_counter()
        try:
            result, _, mask, _ = pipeline.analyze(
                frame,
                mode,
                source_region=source_region,
                allow_unsolved=True,
                progress_callback=lambda progress: self._publish_uploaded_progress(
                    frame, mode, source_region, progress
                ),
            )
            self._add_recognition_diagnostics(result, mask)
            if result.get("motion_ready"):
                self._publish_motion_export(result)
            duration_ms = (time.perf_counter() - started) * 1000.0
            display = pipeline.draw_camera_overlay(frame, result)
            self._draw_runtime_status(
                display, mode, source_region, None
            )
            ok, jpeg = cv2.imencode(
                ".jpg",
                display,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            with self.lock:
                if self.input_mode == "uploaded_image":
                    self.latest_result = result
                    self._solution_signature = None
                    self._scene_change_streak = 0
                    self._quality_confirmation_count = 0
                    self.solution_locked = False
                    self.solution_lock_reason = "uploaded_image"
                    solve_error = (
                        result.get("solver", {}).get("solve_error")
                        if isinstance(result.get("solver"), dict)
                        else None
                    )
                    # ``allow_unsolved`` deliberately returns a diagnostic
                    # result instead of raising.  Keep its actual reason
                    # visible in the UI; clearing it here previously left the
                    # operator with only the vague "PLAN REJECTED" overlay.
                    self.last_error = (
                        str(solve_error) if solve_error else None
                    )
                    self.analysis_ms = duration_ms
                    self.result_sequence += 1
                    if ok:
                        self.latest_jpeg = jpeg.tobytes()
            return result
        except (DetectionError, SolveError, RuntimeError, ValueError) as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            display = frame.copy()
            self._draw_runtime_status(
                display, mode, source_region, str(exc)
            )
            ok, jpeg = cv2.imencode(
                ".jpg",
                display,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            with self.lock:
                if self.input_mode == "uploaded_image":
                    self.latest_result = None
                    self.latest_motion_export = None
                    self.latest_motion_json = None
                    self.latest_motion_csv = None
                    self.last_error = str(exc)
                    self.analysis_ms = duration_ms
                    if ok:
                        self.latest_jpeg = jpeg.tobytes()
            raise

    def resume_camera(self) -> None:
        with self.lock:
            self.input_mode = "camera"
            self.latest_result = None
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self._solution_signature = None
            self._scene_change_streak = 0
            self._unchanged_cache_run = 0
            self._last_scene_change_ratio = 1.0
            self._quality_confirmation_count = 0
            self.solution_locked = False
            self.solution_lock_reason = "camera_resumed"
            self.last_error = "已恢复实时摄像头，正在等待识别"
        self.new_frame.set()

    def reset_target_layout(self) -> None:
        """Forget only the selected tiling; keep live visual recognition."""

        with self.lock:
            if self.latest_result is not None:
                self.latest_result = deepcopy(self.latest_result)
                self.latest_result["target_layout_locked"] = False
            self.latest_motion_export = None
            self.latest_motion_json = None
            self.latest_motion_csv = None
            self.solution_locked = False
            self.solution_lock_reason = "target_layout_reset"
            self._unchanged_cache_run = 0
            self._quality_confirmation_count = 0
            self.last_error = None
        self.new_frame.set()

    @staticmethod
    def _draw_runtime_status(
        frame: np.ndarray,
        mode: str,
        source_region: str,
        error: str | None,
    ) -> None:
        scale = max(0.55, min(frame.shape[:2]) / 900.0)
        thickness = max(1, int(round(scale * 2)))
        label = f"mode={mode}  source={source_region}"
        cv2.putText(
            frame,
            label,
            (18, frame.shape[0] - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        if error:
            short_error = error if len(error) <= 92 else error[:89] + "..."
            cv2.putText(
                frame,
                short_error,
                (18, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale * 0.85,
                (30, 30, 255),
                thickness,
                cv2.LINE_AA,
            )

    def status(self) -> dict[str, Any]:
        with self.lock:
            return {
                "ok": self.latest_result is not None,
                "motion_ready": bool(
                    self.latest_result
                    and self.latest_result.get("motion_ready", False)
                ),
                "source": self.source,
                "input_mode": self.input_mode,
                "mode": self.mode,
                "recognition_level": (
                    "basic" if self.mode == "fixed" else "advanced"
                ),
                "source_region_setting": self.source_region,
                "use_color_hints": self.use_color_hints,
                "paper_color": self.paper_color,
                "piece_color": self.piece_color,
                "capture_fps": round(self.capture_fps, 2),
                "opencv_cpu_threads": cv2.getNumThreads(),
                "analysis_ms": round(self.analysis_ms, 2),
                "frame_sequence": self.frame_sequence,
                "result_sequence": self.result_sequence,
                "solution_locked": self.solution_locked,
                "solution_cache_hits": self.solution_cache_hits,
                "unchanged_cache_run": self._unchanged_cache_run,
                "scene_change_ratio": round(
                    self._last_scene_change_ratio, 5
                ),
                "geometry_lock_hits": self.geometry_lock_hits,
                "quality_confirmation_count": self._quality_confirmation_count,
                "quality_confirmation_required": self._required_quality_confirmations,
                "solution_lock_reason": self.solution_lock_reason,
                "motion_export_ready": self.latest_motion_export is not None,
                "motion_export": self.latest_motion_export,
                "uptime_s": round(time.time() - self.started_at, 1),
                "error": self.last_error,
                "result": self.latest_result,
            }

    def jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def raw_jpeg(self) -> bytes | None:
        with self.lock:
            frame = None if self.raw_frame is None else self.raw_frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        return encoded.tobytes() if ok else None

    def motion_export(self, file_type: str) -> bytes | None:
        with self.lock:
            if file_type == "json":
                return self.latest_motion_json
            if file_type == "csv":
                return self.latest_motion_csv
            raise ValueError(f"Unsupported motion export type: {file_type}")


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """Permit an immediate supervised restart on the same upper-computer port."""

    allow_reuse_address = True


class VisionHandler(BaseHTTPRequestHandler):
    server_version = "PuzzleVision/1.0"

    @property
    def runtime(self) -> LiveVision:
        return self.server.runtime  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {fmt % args}")

    def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(INDEX_HTML)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(INDEX_HTML)
        elif path == "/api/status":
            self._json(self.runtime.status())
        elif path == "/api/card-settings":
            self._json(self.runtime.card_settings_payload())
        elif path == "/api/white-settings":
            self._json(self.runtime.white_settings_payload())
        elif path in ("/api/export-motion.json", "/api/export-motion.csv"):
            self._motion_export(path.rsplit(".", 1)[1])
        elif path == "/stream.mjpg":
            self._stream()
        elif path == "/api/raw.jpg":
            payload = self.runtime.raw_jpeg()
            if payload is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        elif path == "/api/overlay.jpg":
            payload = self.runtime.jpeg()
            if payload is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _motion_export(self, file_type: str) -> None:
        payload = self.runtime.motion_export(file_type)
        if payload is None:
            self.send_error(
                HTTPStatus.NOT_FOUND, "No executable motion plan is available"
            )
            return
        content_type = (
            "application/json; charset=utf-8"
            if file_type == "json"
            else "text/csv; charset=utf-8"
        )
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="latest_motion_plan.{file_type}"',
        )
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/analyze-image":
            self._analyze_image()
            return
        if path == "/api/resume-camera":
            self.runtime.resume_camera()
            self._json({"ok": True})
            return
        if path == "/api/reset-target-layout":
            self.runtime.reset_target_layout()
            self._json({"ok": True})
            return
        if path == "/api/card-settings":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = self.runtime.update_card_settings(
                    payload.get("settings", {}),
                    bool(payload.get("reset", False)),
                )
                self._json(result)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(
                    {"ok": False, "error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                )
            return
        if path == "/api/white-settings":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                result = self.runtime.update_white_settings(
                    payload.get("settings", {}),
                    bool(payload.get("reset", False)),
                )
                self._json(result)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(
                    {"ok": False, "error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                )
            return
        if path != "/api/settings":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self.runtime.update_settings(
                str(payload.get("mode", self.runtime.mode)),
                str(payload.get("source_region", self.runtime.source_region)),
                payload.get("use_color_hints"),
                payload.get("paper_color"),
                payload.get("piece_color"),
            )
            self._json({"ok": True})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.BAD_REQUEST,
            )

    def _analyze_image(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("没有收到图片数据")
            if length > 25 * 1024 * 1024:
                raise ValueError("图片超过 25 MB 限制")
            result = self.runtime.analyze_uploaded_image(
                self.rfile.read(length)
            )
            self._json(
                {
                    "ok": True,
                    "motion_ready": bool(result.get("motion_ready")),
                    "pieces": len(result.get("pieces", [])),
                }
            )
        except (ValueError, DetectionError, SolveError, RuntimeError) as exc:
            self._json(
                {"ok": False, "error": str(exc)},
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )

    def _stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            while not self.runtime.stop_event.is_set():
                jpeg = self.runtime.jpeg()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A4 puzzle live vision web console")
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.json"))
    parser.add_argument("--source", default="usb:/dev/video0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--mode",
        choices=PuzzleVisionPipeline.MODES,
        default="fixed",
    )
    parser.add_argument(
        "--source-region",
        choices=("upper", "lower", "auto"),
        default="upper",
    )
    parser.add_argument("--analysis-interval", type=float, default=0.35)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument(
        "--use-color-hints",
        dest="use_color_hints",
        action="store_true",
    )
    color_group.add_argument(
        "--no-color-hints",
        dest="use_color_hints",
        action="store_false",
    )
    parser.set_defaults(use_color_hints=True)
    parser.add_argument("--paper-color", default="#00a8bd")
    parser.add_argument("--piece-color", default="#f4f4ee")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    runtime = LiveVision(
        config,
        args.source,
        args.mode,
        args.source_region,
        args.analysis_interval,
        args.jpeg_quality,
        args.use_color_hints,
        args.paper_color,
        args.piece_color,
        args.config,
    )
    server = ReusableThreadingHTTPServer((args.host, args.port), VisionHandler)
    server.runtime = runtime  # type: ignore[attr-defined]

    def stop_server(*_: Any) -> None:
        runtime.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    runtime.start()
    print(
        f"Puzzle vision console: http://{args.host}:{args.port} "
        f"camera={args.source} mode={args.mode} source_region={args.source_region}",
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        runtime.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
