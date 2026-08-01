#!/usr/bin/env python3
"""Live OpenCV piece detection viewer for the physical puzzle work area."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from puzzle_device.calibration.manual_calibration import PixelToGantryCalibration
from puzzle_device.planning import (
    AssemblyConfig,
    build_movement_plan,
    draw_assembly_preview,
    solve_assembly,
    target_rectangle_pixels,
)
from puzzle_device.vision.camera import open_uvc_camera
from puzzle_device.vision.case_replay import load_vision_case, save_vision_case
from puzzle_device.vision.piece_vision import (
    DetectionConfig,
    detect_piece_observations,
    draw_piece_observations,
    extract_piece_edges,
    load_detection_config,
    save_detection_config,
)
from puzzle_device.vision.stability import PieceStabilityTracker


CALIBRATION_PATHS = (
    Path("configs/local/calibration.json"),
    Path("configs/local/calibration_temporary.json"),
)
BACKGROUND_PATH = Path("data/local/empty_work_area.png")
DEFAULT_CONFIG_PATH = Path("configs/vision_detection.json")
LOCAL_CONFIG_PATH = Path("configs/local/vision_detection.json")
ROI_PATH = Path("configs/local/a4_roi.json")
LOCKED_RESULT_PATH = Path("output/locked_piece_observations.json")
ASSEMBLY_PLAN_PATH = Path("output/assembly_plan.json")
ASSEMBLY_PREVIEW_PATH = Path("output/assembly_preview.png")
REAL_CASES_PATH = Path("data/real_cases")


class PieceDetectionApp:
    """Show live segmentation without allowing any machine movement."""

    def __init__(self, root: tk.Tk, camera_index: int, rotate_180: bool = True):
        self.root = root
        self.root.title("拼图装置 - 实物碎片识别（不控制电机）")
        self.root.geometry("1480x880")
        self.root.minsize(1160, 700)
        self.camera_index = camera_index
        self.rotate_180 = rotate_180
        self.capture: cv2.VideoCapture | None = None
        self.current_frame: np.ndarray | None = None
        self.replay_active = False
        self.replay_path: Path | None = None
        self.background: np.ndarray | None = None
        self.photo_main = None
        self.photo_mask = None
        self.photo_edges = None
        self.main_display_scale = 1.0
        self.main_display_origin = (0, 0)
        self.analysis_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="piece-vision"
        )
        self.analysis_future: Future | None = None
        self.next_analysis_time = 0.0
        self.analysis_interval_s = 0.12
        self.roi: tuple[int, int, int, int] | None = None
        self.roi_select_mode = False
        self.roi_drag_start: tuple[float, float] | None = None
        self.roi_drag_current: tuple[float, float] | None = None
        self.calibration = PixelToGantryCalibration()
        self.calibration_name = "未加载"
        self.stability_tracker = PieceStabilityTracker()
        self.latest_pieces = []
        self.locked_pieces = []
        self.analysis_paused = False
        self.results_locked = False
        self.assembly_config = AssemblyConfig()
        self.assembly_plan = None
        self.assembly_document = None

        initial = self._initial_config()
        method_labels = {
            "background": "背景差分（推荐）",
            "white_hsv": "白色 HSV",
            "brightness": "亮度阈值",
        }
        self.method_values = {label: value for value, label in method_labels.items()}
        self.segmentation_method = tk.StringVar(value=method_labels[initial.segmentation_method])
        self.min_area = tk.StringVar(value=f"{initial.min_area_px:g}")
        self.max_area = tk.StringVar(value=f"{initial.max_area_px:g}")
        self.threshold = tk.StringVar(
            value="0" if initial.color_distance_threshold is None
            else str(initial.color_distance_threshold)
        )
        self.max_pieces = tk.StringVar(value=str(min(initial.max_pieces, 4)))
        self.white_s_max = tk.StringVar(value=str(initial.white_saturation_max))
        self.white_v_min = tk.StringVar(value=str(initial.white_value_min))
        self.brightness_min = tk.StringVar(value=str(initial.brightness_min))
        self.morphology_size = tk.StringVar(value=str(initial.morphology_size))
        self.blur_size = tk.StringVar(value=str(initial.gaussian_blur_size))
        self.canny_lower = tk.StringVar(value=str(initial.canny_lower))
        self.canny_upper = tk.StringVar(value=str(initial.canny_upper))
        self.epsilon_min = tk.StringVar(value=f"{initial.polygon_epsilon_min:g}")
        self.epsilon_preferred = tk.StringVar(value=f"{initial.polygon_epsilon_preferred:g}")
        self.epsilon_max = tk.StringVar(value=f"{initial.polygon_epsilon_max:g}")
        self.min_vertices = tk.StringVar(value=str(initial.min_vertices))
        self.max_vertices = tk.StringVar(value=str(initial.max_vertices))
        self.minimum_pick_clearance = tk.StringVar(value=f"{initial.minimum_pick_clearance_px:g}")
        self.status = tk.StringVar(value="正在打开相机。")
        self.background_state = tk.StringVar(value="背景：未采集（当前使用边缘颜色估计）")
        self.roi_state = tk.StringVar(value="A4 ROI：未设置，当前识别整张画面")
        self.matrix_state = tk.StringVar(value="坐标矩阵：未加载，只显示像素坐标")
        self.stability_state = tk.StringVar(value="稳定状态：等待识别")
        self.planning_state = tk.StringVar(value="拼接方案：请先稳定识别并锁定")

        self._build_ui()
        self._bind_parameter_changes()
        self._load_calibration()
        self._load_roi()
        self._load_saved_background()
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._update()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 16, "bold"))

        header = ttk.Frame(self.root, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="实时碎片识别", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="仅识别，不会发送任何电机命令", foreground="#a01d1d").pack(
            side="right", pady=(5, 0)
        )

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        image_pane = ttk.Frame(body)
        controls_pane = ttk.Frame(body, width=420)
        body.add(image_pane, weight=4)
        body.add(controls_pane, weight=2)

        self.controls_canvas = tk.Canvas(
            controls_pane, highlightthickness=0, borderwidth=0
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_pane, orient="vertical", command=self.controls_canvas.yview
        )
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_scrollbar.pack(side="right", fill="y")
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(self.controls_canvas, padding=(0, 0, 7, 0))
        self.controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw"
        )
        controls.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            ),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(
                self.controls_window, width=event.width
            ),
        )
        self.controls_pane = controls_pane
        self.root.bind_all("<MouseWheel>", self._on_controls_mousewheel, add="+")

        images = ttk.Panedwindow(image_pane, orient="vertical")
        images.pack(fill="both", expand=True)
        main_frame = ttk.LabelFrame(images, text="实时画面：轮廓 / 顶点 / 中心 / 方向 / 安全抓取点", padding=4)
        mask_frame = ttk.LabelFrame(images, text="碎片分割掩膜（白色为识别到的碎片）", padding=4)
        edge_frame = ttk.LabelFrame(images, text="外边缘调试（从掩膜提取，不含碎片内部花纹）", padding=4)
        images.add(main_frame, weight=3)
        images.add(mask_frame, weight=1)
        images.add(edge_frame, weight=1)
        self.main_canvas = tk.Canvas(main_frame, background="#202326", highlightthickness=0)
        self.main_canvas.pack(fill="both", expand=True)
        self.mask_canvas = tk.Canvas(mask_frame, background="#202326", highlightthickness=0)
        self.mask_canvas.pack(fill="both", expand=True)
        self.edge_canvas = tk.Canvas(edge_frame, background="#202326", highlightthickness=0)
        self.edge_canvas.pack(fill="both", expand=True)
        self.main_canvas.bind("<Configure>", lambda _event: self._show_images())
        self.main_canvas.bind("<ButtonPress-1>", self._on_roi_press)
        self.main_canvas.bind("<B1-Motion>", self._on_roi_drag)
        self.main_canvas.bind("<ButtonRelease-1>", self._on_roi_release)
        self.mask_canvas.bind("<Configure>", lambda _event: self._show_images())
        self.edge_canvas.bind("<Configure>", lambda _event: self._show_images())

        capture = ttk.LabelFrame(controls, text="A4 工作区与背景", padding=10)
        capture.pack(fill="x")
        roi_buttons = ttk.Frame(capture)
        roi_buttons.pack(fill="x")
        ttk.Button(roi_buttons, text="框选 A4 ROI", command=self._arm_roi_selection).pack(
            side="left", fill="x", expand=True)
        ttk.Button(roi_buttons, text="清除 ROI", command=self._clear_roi).pack(
            side="left", fill="x", expand=True, padx=(7, 0))
        ttk.Label(capture, textvariable=self.roi_state, wraplength=365,
                  foreground="#7c3f00", justify="left").pack(fill="x", pady=(6, 8))
        ttk.Button(capture, text="采集当前空桌面为背景", command=self._capture_background).pack(fill="x")
        ttk.Button(capture, text="清除背景，改用边缘颜色", command=self._clear_background).pack(
            fill="x", pady=(6, 0)
        )
        ttk.Label(capture, textvariable=self.background_state, wraplength=365,
                  foreground="#155e75", justify="left").pack(fill="x", pady=(7, 0))
        case_buttons = ttk.Frame(capture)
        case_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(case_buttons, text="保存当前案例", command=self._save_current_case).pack(
            side="left", fill="x", expand=True)
        ttk.Button(case_buttons, text="加载案例回放", command=self._load_case_replay).pack(
            side="left", fill="x", expand=True, padx=(7, 0))
        ttk.Button(capture, text="返回实时相机", command=self._return_to_live_camera).pack(
            fill="x", pady=(6, 0))

        parameter_box = ttk.LabelFrame(controls, text="识别参数", padding=7)
        parameter_box.pack(fill="x", pady=(9, 0))
        parameters = ttk.Notebook(parameter_box)
        parameters.pack(fill="x")
        segmentation_tab = ttk.Frame(parameters, padding=7)
        geometry_tab = ttk.Frame(parameters, padding=7)
        parameters.add(segmentation_tab, text="分割")
        parameters.add(geometry_tab, text="轮廓与抓取")

        ttk.Label(segmentation_tab, text="分割方法").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(segmentation_tab, textvariable=self.segmentation_method,
                     values=list(self.method_values), state="readonly", width=18).grid(
                         row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        self._parameter_row(segmentation_tab, "背景颜色差（0=自动）", self.threshold, 1)
        self._parameter_row(segmentation_tab, "白色 HSV 饱和度上限", self.white_s_max, 2)
        self._parameter_row(segmentation_tab, "白色 HSV 亮度下限", self.white_v_min, 3)
        self._parameter_row(segmentation_tab, "亮度分割下限", self.brightness_min, 4)
        self._parameter_row(segmentation_tab, "形态学核（奇数）", self.morphology_size, 5)
        self._parameter_row(segmentation_tab, "模糊核（奇数）", self.blur_size, 6)
        self._parameter_row(geometry_tab, "最小面积（像素）", self.min_area, 0)
        self._parameter_row(geometry_tab, "最大面积（像素，0=自动）", self.max_area, 1)
        self._parameter_row(geometry_tab, "最多碎片数", self.max_pieces, 2)
        self._parameter_row(geometry_tab, "Canny 下阈值", self.canny_lower, 3)
        self._parameter_row(geometry_tab, "Canny 上阈值", self.canny_upper, 4)
        self._parameter_row(geometry_tab, "角点近似最小比例", self.epsilon_min, 5)
        self._parameter_row(geometry_tab, "角点首选比例", self.epsilon_preferred, 6)
        self._parameter_row(geometry_tab, "角点近似最大比例", self.epsilon_max, 7)
        self._parameter_row(geometry_tab, "最少角点数", self.min_vertices, 8)
        self._parameter_row(geometry_tab, "最多角点数", self.max_vertices, 9)
        self._parameter_row(geometry_tab, "最小抓取余量（像素）", self.minimum_pick_clearance, 10)
        segmentation_tab.columnconfigure(1, weight=1)
        geometry_tab.columnconfigure(1, weight=1)

        parameter_buttons = ttk.Frame(parameter_box)
        parameter_buttons.pack(fill="x", pady=(6, 0))
        ttk.Button(parameter_buttons, text="恢复上次保存", command=self._restore_saved_parameters).pack(
            side="left", fill="x", expand=True)
        ttk.Button(parameter_buttons, text="保存参数", command=self._save_parameters).pack(
            side="left", fill="x", expand=True, padx=(7, 0))

        lock_box = ttk.LabelFrame(controls, text="稳定采样与结果锁定", padding=8)
        lock_box.pack(fill="x", pady=(9, 0))
        ttk.Label(lock_box, textvariable=self.stability_state, foreground="#7c3f00",
                  wraplength=365, justify="left").pack(fill="x")
        lock_buttons = ttk.Frame(lock_box)
        lock_buttons.pack(fill="x", pady=(6, 0))
        self.pause_button = ttk.Button(
            lock_buttons, text="暂停识别", command=self._toggle_analysis_pause
        )
        self.pause_button.pack(side="left", fill="x", expand=True)
        self.lock_button = ttk.Button(
            lock_buttons, text="确认并锁定", command=self._lock_stable_result,
            state="disabled",
        )
        self.lock_button.pack(side="left", fill="x", expand=True, padx=(7, 0))
        ttk.Button(lock_box, text="解除锁定，重新采样", command=self._unlock_result).pack(
            fill="x", pady=(6, 0)
        )

        planning_box = ttk.LabelFrame(controls, text="拼接与运动计划（仅计算）", padding=8)
        planning_box.pack(fill="x", pady=(9, 0))
        ttk.Label(planning_box, textvariable=self.planning_state, foreground="#7c3f00",
                  wraplength=365, justify="left").pack(fill="x")
        self.plan_button = ttk.Button(
            planning_box, text="计算拼接方案", command=self._calculate_assembly,
            state="disabled",
        )
        self.plan_button.pack(fill="x", pady=(6, 0))
        ttk.Label(
            planning_box,
            text="输出目标像素、XY 脉冲和旋转角；当前不会发送串口命令。",
            foreground="#a01d1d", wraplength=365, justify="left",
        ).pack(fill="x", pady=(5, 0))

        detected = ttk.LabelFrame(controls, text="当前识别结果", padding=10)
        detected.pack(fill="both", expand=True, pady=(9, 0))
        self.piece_list = tk.Listbox(detected, font=("Consolas", 9), height=12)
        self.piece_list.pack(fill="both", expand=True)
        ttk.Label(detected, textvariable=self.matrix_state, foreground="#155e75", wraplength=365,
                  justify="left").pack(fill="x", pady=(7, 0))

        footer = ttk.Frame(self.root, padding=(14, 6))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, foreground="#174c75").pack(side="left")

    def _on_controls_mousewheel(self, event: tk.Event) -> None:
        """Scroll the right control column only while the pointer is over it."""
        pane = self.controls_pane
        left = pane.winfo_rootx()
        top = pane.winfo_rooty()
        if not (
            left <= event.x_root < left + pane.winfo_width()
            and top <= event.y_root < top + pane.winfo_height()
        ):
            return
        self.controls_canvas.yview_scroll(-int(event.delta / 120), "units")

    @staticmethod
    def _parameter_row(parent: ttk.LabelFrame, label: str, variable: tk.StringVar,
                       row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=10).grid(
            row=row, column=1, sticky="ew", padx=(8, 0), pady=2
        )

    def _bind_parameter_changes(self) -> None:
        variables = (
            self.segmentation_method, self.min_area, self.max_area, self.threshold,
            self.white_s_max, self.white_v_min, self.brightness_min, self.max_pieces,
            self.morphology_size, self.blur_size, self.canny_lower, self.canny_upper,
            self.epsilon_min, self.epsilon_preferred, self.epsilon_max,
            self.min_vertices, self.max_vertices, self.minimum_pick_clearance,
        )
        for variable in variables:
            variable.trace_add("write", self._on_parameter_changed)

    def _on_parameter_changed(self, *_args) -> None:
        self._invalidate_assembly("参数已修改")
        if self.results_locked:
            self.results_locked = False
            self.analysis_paused = False
            self.locked_pieces = []
            self.pause_button.configure(text="暂停识别")
        self._reset_stability("参数已修改，重新稳定采样")
        self.next_analysis_time = 0.0

    def _open_camera(self) -> None:
        self.capture, camera_info = open_uvc_camera(self.camera_index)
        if self.capture is None:
            self.status.set(f"无法打开摄像头 {self.camera_index}。")
            return
        orientation = "已旋转 180°" if self.rotate_180 else "原始方向"
        details = "" if camera_info is None else camera_info.describe()
        self.status.set(
            f"相机已打开：{details}，画面{orientation}。先移开所有碎片，再采集空背景。"
        )

    def _config(self) -> DetectionConfig | None:
        try:
            threshold = int(self.threshold.get())
            config = DetectionConfig(
                segmentation_method=self.method_values[self.segmentation_method.get()],
                min_area_px=float(self.min_area.get()),
                max_area_px=float(self.max_area.get()),
                color_distance_threshold=None if threshold == 0 else threshold,
                white_saturation_max=int(self.white_s_max.get()),
                white_value_min=int(self.white_v_min.get()),
                brightness_min=int(self.brightness_min.get()),
                max_pieces=int(self.max_pieces.get()),
                morphology_size=int(self.morphology_size.get()),
                gaussian_blur_size=int(self.blur_size.get()),
                canny_lower=int(self.canny_lower.get()),
                canny_upper=int(self.canny_upper.get()),
                polygon_epsilon_min=float(self.epsilon_min.get()),
                polygon_epsilon_preferred=float(self.epsilon_preferred.get()),
                polygon_epsilon_max=float(self.epsilon_max.get()),
                min_vertices=int(self.min_vertices.get()),
                max_vertices=int(self.max_vertices.get()),
                minimum_pick_clearance_px=float(self.minimum_pick_clearance.get()),
            )
            config.validate()
        except ValueError:
            self.status.set("识别参数无效：检查数字、阈值大小关系和角点数量。")
            return None
        return config

    def _update(self) -> None:
        if self.capture is not None and not self.replay_active:
            ok, frame = self.capture.read()
            if ok:
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.current_frame = frame
        self._collect_analysis_result()
        self._queue_analysis()
        self.root.after(30, self._update)

    def _save_current_case(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("没有画面", "当前没有可以保存的图像。")
            return
        config = self._config()
        if config is None:
            messagebox.showerror("参数无效", "请先修正识别参数。")
            return
        try:
            case_path = save_vision_case(
                REAL_CASES_PATH,
                self.current_frame,
                self.background,
                config,
                self.roi,
                pieces=self.latest_pieces,
                mask=None if not hasattr(self, "mask_image") else self.mask_image,
                overlay=None if not hasattr(self, "overlay") else self.overlay,
                edges=None if not hasattr(self, "edge_image") else self.edge_image,
            )
        except (OSError, ValueError, cv2.error) as exc:
            messagebox.showerror("保存案例失败", str(exc))
            return
        self.status.set(f"当前案例已保存：{case_path}。可在没有相机时重复回放调试。")

    def _load_case_replay(self) -> None:
        REAL_CASES_PATH.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askdirectory(
            title="选择包含 case.json 的案例目录",
            initialdir=str(REAL_CASES_PATH.resolve()),
        )
        if not selected:
            return
        try:
            case = load_vision_case(Path(selected))
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            messagebox.showerror("加载案例失败", str(exc))
            return
        self.replay_active = True
        self.replay_path = case.path
        self.current_frame = case.frame
        self.background = case.background
        self.roi = case.roi
        self._apply_config(case.config)
        self.results_locked = False
        self.analysis_paused = False
        self.locked_pieces = []
        self.pause_button.configure(text="暂停识别")
        self._invalidate_assembly("已加载回放案例")
        self._reset_stability("案例已加载，重新稳定采样")
        if self.roi is None:
            self.roi_state.set("A4 ROI：案例未保存 ROI，当前识别整张画面")
        else:
            x, y, width, height = self.roi
            self.roi_state.set(f"A4 ROI：案例 x={x}, y={y}, w={width}, h={height}")
        self.background_state.set(
            "背景：案例未包含背景" if self.background is None else "背景：使用案例背景图"
        )
        self.status.set(f"正在离线回放：{case.path}。参数修改后会立即重新识别同一张原图。")
        self._detect_and_show()

    def _return_to_live_camera(self) -> None:
        if not self.replay_active:
            self.status.set("当前已经是实时相机模式。")
            return
        self.replay_active = False
        self.replay_path = None
        self.background = None
        self.roi = None
        self.background_state.set("背景：未采集（当前使用边缘颜色估计）")
        self.roi_state.set("A4 ROI：未设置，当前识别整张画面")
        self._load_saved_background()
        self._load_roi()
        self._apply_config(self._initial_config())
        self.results_locked = False
        self.analysis_paused = False
        self.locked_pieces = []
        self.pause_button.configure(text="暂停识别")
        self._invalidate_assembly("已返回实时相机")
        self._reset_stability("已返回实时相机，重新稳定采样")
        self.status.set("已返回实时相机，正在采集新画面。")
        self._detect_and_show()

    def _detect_and_show(self) -> None:
        """Request a fresh result without blocking Tk's event loop."""
        self.next_analysis_time = 0.0
        self._queue_analysis()

    def _queue_analysis(self) -> None:
        if self.current_frame is None:
            return
        if self.analysis_paused or self.results_locked:
            return
        if self.analysis_future is not None:
            return
        now = time.monotonic()
        if now < self.next_analysis_time:
            return
        self.next_analysis_time = now + self.analysis_interval_s
        config = self._config()
        if config is None:
            self._show_unprocessed_frame()
            return
        frame = self.current_frame.copy()
        background = None if self.background is None else self.background.copy()
        roi = self._source_roi()
        self.analysis_future = self.analysis_executor.submit(
            self._analyze_frame, frame, background, config, roi
        )

    @staticmethod
    def _analyze_frame(
        frame: np.ndarray,
        background: np.ndarray | None,
        config: DetectionConfig,
        roi: tuple[int, int, int, int] | None,
    ) -> tuple[list, np.ndarray, np.ndarray, np.ndarray, str | None]:
        try:
            pieces, mask = detect_piece_observations(frame, background, config, roi=roi)
            overlay = draw_piece_observations(frame, pieces)
            edges = extract_piece_edges(mask, config)
            return (
                pieces,
                overlay,
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),
                None,
            )
        except (RuntimeError, ValueError, cv2.error) as exc:
            blank = np.zeros_like(frame)
            return [], frame, blank, blank.copy(), str(exc)

    def _collect_analysis_result(self) -> None:
        if self.analysis_future is None or not self.analysis_future.done():
            return
        future = self.analysis_future
        self.analysis_future = None
        if self.analysis_paused or self.results_locked:
            return
        try:
            pieces, self.overlay, self.mask_image, self.edge_image, error = future.result()
        except Exception as exc:  # Keep the camera view alive after an unexpected worker failure.
            self.status.set(f"识别线程异常：{exc}")
            self._show_unprocessed_frame()
            return
        self._set_piece_list(pieces)
        if error is not None:
            self.latest_pieces = []
            self._reset_stability("检测异常，等待下一帧")
            self.status.set(f"检测提示：{error}；相机画面仍保持刷新。")
            self._show_images()
            return
        self.latest_pieces = pieces
        stability = self.stability_tracker.update(pieces)
        self.stability_state.set(f"稳定状态：{stability.reason}")
        self.lock_button.configure(state="normal" if stability.stable else "disabled")
        method_name = self.segmentation_method.get()
        self.status.set(
            f"{method_name}识别到 {len(pieces)} 个碎片。黄线=轮廓，红点=顶点，绿十字=中心，"
            "蓝线=PCA方向，紫色叉=安全抓取点、紫色圆=距边缘安全距离。"
        )
        self._show_images()

    def _reset_stability(self, reason: str = "等待稳定采样") -> None:
        status = self.stability_tracker.reset(reason)
        self.stability_state.set(f"稳定状态：{status.reason}")
        self.lock_button.configure(state="disabled")

    def _toggle_analysis_pause(self) -> None:
        if self.results_locked:
            self.status.set("结果已锁定；如需继续识别，请先解除锁定。")
            return
        self.analysis_paused = not self.analysis_paused
        self.pause_button.configure(text="继续识别" if self.analysis_paused else "暂停识别")
        if self.analysis_paused:
            self.status.set("识别已暂停，保留当前画面和结果。")
        else:
            self._reset_stability("已继续识别，重新稳定采样")
            self.status.set("识别已继续，正在重新采集连续稳定帧。")
            self._detect_and_show()

    def _lock_stable_result(self) -> None:
        if not self.stability_tracker.status.stable:
            messagebox.showwarning("结果尚未稳定", "请等待稳定状态显示“可确认锁定”。")
            return
        try:
            pieces = self.stability_tracker.averaged_observations()
        except RuntimeError as exc:
            messagebox.showerror("锁定失败", str(exc))
            return
        self.locked_pieces = pieces
        self.latest_pieces = pieces
        self.results_locked = True
        self.analysis_paused = True
        self.pause_button.configure(text="继续识别")
        self.lock_button.configure(state="disabled")
        self.overlay = draw_piece_observations(self.current_frame, pieces)
        self._set_piece_list(pieces)
        try:
            self._save_locked_result(pieces)
        except OSError as exc:
            self.results_locked = False
            self.analysis_paused = False
            self.locked_pieces = []
            self.pause_button.configure(text="暂停识别")
            messagebox.showerror("保存失败", f"无法保存锁定结果：{exc}")
            self._reset_stability("保存失败，请重新确认")
            return
        self.stability_state.set(
            f"稳定状态：已锁定 {len(pieces)} 块，结果已保存到 {LOCKED_RESULT_PATH}"
        )
        if self._valid_roi() is None:
            self.planning_state.set("拼接方案：必须先框选完整 A4 ROI")
            self.plan_button.configure(state="disabled")
        else:
            self.planning_state.set("拼接方案：已锁定，可点击计算")
            self.plan_button.configure(state="normal")
        self.status.set("稳定识别结果已锁定；现在可计算下半区拼接方案。")
        self._show_images()

    def _unlock_result(self) -> None:
        self._invalidate_assembly("已解除锁定")
        self.results_locked = False
        self.analysis_paused = False
        self.locked_pieces = []
        self.pause_button.configure(text="暂停识别")
        self._reset_stability("已解除锁定，重新稳定采样")
        self.status.set("已解除锁定，实时识别和稳定采样已恢复。")
        self._detect_and_show()

    def _save_locked_result(self, pieces) -> None:
        config = self._config()
        document = {
            "format": "puzzle-device.locked-piece-observations.v1",
            "created_local": datetime.now().astimezone().isoformat(),
            "camera": {
                "index": self.camera_index,
                "rotation_degrees": 180 if self.rotate_180 else 0,
                "image_width": None if self.current_frame is None else self.current_frame.shape[1],
                "image_height": None if self.current_frame is None else self.current_frame.shape[0],
            },
            "roi": None if self.roi is None else list(self.roi),
            "source_roi": None if self.roi is None else list(self._source_roi()),
            "calibration_file": self.calibration_name,
            "stability": {
                "averaged_frames": self.stability_tracker.required_frames,
                "center_tolerance_px": self.stability_tracker.center_tolerance_px,
                "angle_tolerance_deg": self.stability_tracker.angle_tolerance_deg,
                "area_tolerance_ratio": self.stability_tracker.area_tolerance_ratio,
            },
            "detection_parameters": None if config is None else config.to_dict(),
            "pieces": [],
        }
        for piece in pieces:
            values = piece.to_dict()
            center_pulse = self._to_pulse(piece.center)
            pick_pulse = self._to_pulse(piece.pick_point)
            values["center_pulse"] = None if center_pulse is None else list(center_pulse)
            values["pick_pulse"] = None if pick_pulse is None else list(pick_pulse)
            document["pieces"].append(values)
        LOCKED_RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCKED_RESULT_PATH.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _invalidate_assembly(self, reason: str) -> None:
        self.assembly_plan = None
        self.assembly_document = None
        if hasattr(self, "plan_button"):
            self.plan_button.configure(state="disabled")
        if hasattr(self, "planning_state"):
            self.planning_state.set(f"拼接方案：{reason}，需要重新锁定")

    def _calculate_assembly(self) -> None:
        roi = self._valid_roi()
        if not self.results_locked or not self.locked_pieces:
            messagebox.showwarning("尚未锁定", "请先等待稳定识别，然后点击“确认并锁定”。")
            return
        if roi is None:
            messagebox.showwarning("没有 A4 ROI", "请先框选完整的 A4 纸区域。")
            return
        if len(self.locked_pieces) > 4:
            messagebox.showerror("碎片数量错误", "正式拼接只支持 1–4 块，请调整识别参数。")
            return

        self.plan_button.configure(state="disabled")
        self.planning_state.set("拼接方案：正在计算，请稍候……")
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            assembly = solve_assembly(
                [np.asarray(piece.polygon, dtype=np.float64) for piece in self.locked_pieces],
                roi,
                self.assembly_config,
                require_upper_half=True,
            )
            document = build_movement_plan(
                self.locked_pieces,
                assembly,
                pulse_mapper=self._to_pulse,
                calibration_file=self.calibration_name,
                config=self.assembly_config,
            )
            preview = draw_assembly_preview(
                draw_piece_observations(self.current_frame, self.locked_pieces),
                self.locked_pieces,
                assembly,
                self.assembly_config,
            )
            ASSEMBLY_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
            ASSEMBLY_PLAN_PATH.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not cv2.imwrite(str(ASSEMBLY_PREVIEW_PATH), preview):
                raise OSError(f"无法保存预览图：{ASSEMBLY_PREVIEW_PATH}")
        except (OSError, RuntimeError, ValueError, cv2.error) as exc:
            self.planning_state.set("拼接方案：计算失败，请检查轮廓和 ROI")
            messagebox.showerror("拼接计算失败", str(exc))
            return
        finally:
            self.root.configure(cursor="")
            self.plan_button.configure(state="normal")

        self.assembly_plan = assembly
        self.assembly_document = document
        self.overlay = preview
        score = document["quality"]["geometry_score"]
        recovered = document["quality"]["recovered_size_mm"]
        fill_ratio = document["quality"]["rectangle_fill_ratio"]
        placement_gap = document["quality"]["placement_gap_actual_mm"]
        pulse_note = "已含目标脉冲" if self.calibration.matrix is not None else "未加载脉冲标定"
        self.planning_state.set(
            f"拼接方案：完成，几何分={score:.3f}，恢复尺寸="
            f"{recovered[0]:.1f}×{recovered[1]:.1f} mm，"
            f"矩形填充率={fill_ratio:.1%}，安全缝={placement_gap:.1f} mm，{pulse_note}"
        )
        self.status.set(
            f"方案已保存到 {ASSEMBLY_PLAN_PATH}，预览图保存到 {ASSEMBLY_PREVIEW_PATH}；"
            "未发送任何电机命令。"
        )
        self._show_images()

    def _show_unprocessed_frame(self) -> None:
        if self.current_frame is None:
            return
        self.overlay = self.current_frame.copy()
        blank = np.zeros_like(self.current_frame)
        self.mask_image = blank
        self.edge_image = blank.copy()
        self._set_piece_list([])
        self._show_images()

    def _set_piece_list(self, pieces) -> None:
        self.piece_list.delete(0, tk.END)
        config = self._config()
        minimum_clearance = 0.0 if config is None else config.minimum_pick_clearance_px
        for piece in pieces:
            center_pulse = self._to_pulse(piece.center)
            pick_pulse = self._to_pulse(piece.pick_point)
            center_text = "--" if center_pulse is None else f"({center_pulse[0]:4d},{center_pulse[1]:4d})"
            pick_text = "--" if pick_pulse is None else f"({pick_pulse[0]:4d},{pick_pulse[1]:4d})"
            safety = "OK" if piece.pick_clearance_px >= minimum_clearance else "WARN"
            self.piece_list.insert(
                tk.END,
                f"P{piece.piece_id}: px=({piece.center[0]:6.1f},{piece.center[1]:6.1f})\n"
                f"    center pulse={center_text}  area={piece.area_px:.0f}\n"
                f"    pick px=({piece.pick_point[0]:6.1f},{piece.pick_point[1]:6.1f})"
                f" -> pulse={pick_text}\n"
                f"    safe={piece.pick_clearance_px:.1f}px [{safety}]  vertices={len(piece.polygon)}"
                f"  angle={piece.pca_angle_deg:.1f} deg  conf={piece.confidence:.2f}",
            )

    def _to_pulse(self, center: tuple[float, float]) -> tuple[int, int] | None:
        if self.calibration.matrix is None:
            return None
        x, y = self.calibration.predict_pulse(*center)
        return round(x), round(y)

    def _show_images(self) -> None:
        if not all(hasattr(self, name) for name in ("overlay", "mask_image", "edge_image")):
            return
        self.photo_main, self.main_display_scale, self.main_display_origin = self._draw_to_canvas(
            self.main_canvas, self.overlay, self.photo_main
        )
        self._draw_roi_on_canvas()
        self.photo_mask, _scale, _origin = self._draw_to_canvas(
            self.mask_canvas, self.mask_image, self.photo_mask
        )
        self.photo_edges, _scale, _origin = self._draw_to_canvas(
            self.edge_canvas, self.edge_image, self.photo_edges
        )

    @staticmethod
    def _draw_to_canvas(canvas: tk.Canvas, image: np.ndarray, previous):
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 2 or height < 2:
            return previous, 1.0, (0, 0)
        scale = min(width / image.shape[1], height / image.shape[0])
        shown = cv2.resize(image, (max(1, int(image.shape[1] * scale)),
                                   max(1, int(image.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", shown)
        if not ok:
            return previous, scale, (0, 0)
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        origin = ((width - shown.shape[1]) // 2, (height - shown.shape[0]) // 2)
        canvas.delete("all")
        canvas.create_image(*origin, image=photo, anchor="nw")
        return photo, scale, origin

    def _arm_roi_selection(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("没有画面", "摄像头还没有可用画面。")
            return
        self.roi_select_mode = True
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.main_canvas.configure(cursor="crosshair")
        self.status.set("请在主画面中按住鼠标左键，从 A4 一个角拖到对角，松开后自动保存。")

    def _canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[float, float] | None:
        if self.current_frame is None or self.main_display_scale <= 0:
            return None
        origin_x, origin_y = self.main_display_origin
        x = (canvas_x - origin_x) / self.main_display_scale
        y = (canvas_y - origin_y) / self.main_display_scale
        if not (0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]):
            return None
        return x, y

    def _on_roi_press(self, event: tk.Event) -> None:
        if not self.roi_select_mode:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is not None:
            self.roi_drag_start = point
            self.roi_drag_current = point
            self._draw_roi_on_canvas()

    def _on_roi_drag(self, event: tk.Event) -> None:
        if not self.roi_select_mode or self.roi_drag_start is None:
            return
        point = self._canvas_to_image(event.x, event.y)
        if point is not None:
            self.roi_drag_current = point
            self._draw_roi_on_canvas()

    def _on_roi_release(self, event: tk.Event) -> None:
        if not self.roi_select_mode or self.roi_drag_start is None:
            return
        point = self._canvas_to_image(event.x, event.y) or self.roi_drag_current
        if point is None:
            return
        x0, y0 = self.roi_drag_start
        x1, y1 = point
        x, y = round(min(x0, x1)), round(min(y0, y1))
        width, height = round(abs(x1 - x0)), round(abs(y1 - y0))
        self.roi_select_mode = False
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.main_canvas.configure(cursor="")
        if width < 50 or height < 50:
            messagebox.showwarning("ROI 太小", "请重新框选完整的橙色 A4 区域。")
            self._draw_roi_on_canvas()
            return
        self._invalidate_assembly("A4 ROI 已更新")
        self.roi = (x, y, width, height)
        self._save_roi()
        self._reset_stability("ROI 已更新，重新稳定采样")
        self.roi_state.set(f"A4 ROI：x={x}, y={y}, w={width}, h={height}（已保存）")
        self.status.set("A4 ROI 已保存；框外区域不参与碎片识别，像素和脉冲仍使用全局坐标。")
        self._detect_and_show()

    def _draw_roi_on_canvas(self) -> None:
        self.main_canvas.delete("roi_overlay")
        rectangle = self.roi
        color = "#00a8ff"
        if self.roi_select_mode and self.roi_drag_start and self.roi_drag_current:
            x0, y0 = self.roi_drag_start
            x1, y1 = self.roi_drag_current
            rectangle = (round(min(x0, x1)), round(min(y0, y1)),
                         round(abs(x1 - x0)), round(abs(y1 - y0)))
            color = "#ffd400"
        if rectangle is None:
            return
        x, y, width, height = rectangle
        origin_x, origin_y = self.main_display_origin
        scale = self.main_display_scale
        canvas_x0 = origin_x + x * scale
        canvas_y0 = origin_y + y * scale
        canvas_x1 = origin_x + (x + width) * scale
        canvas_y1 = origin_y + (y + height) * scale
        self.main_canvas.create_rectangle(
            canvas_x0, canvas_y0, canvas_x1, canvas_y1,
            outline=color, width=3, tags="roi_overlay",
        )
        self.main_canvas.create_text(
            canvas_x0 + 6, canvas_y0 + 6, text="A4 ROI", anchor="nw",
            fill=color, font=("Sans", 11, "bold"), tags="roi_overlay",
        )
        if self.roi_select_mode:
            return
        divider_y = canvas_y0 + height * self.assembly_config.split_fraction * scale
        self.main_canvas.create_line(
            canvas_x0, divider_y, canvas_x1, divider_y,
            fill="#ffd400", width=2, dash=(8, 5), tags="roi_overlay",
        )
        self.main_canvas.create_text(
            canvas_x1 - 8, canvas_y0 + 8, text="识别区", anchor="ne",
            fill="#ffd400", font=("Sans", 11, "bold"), tags="roi_overlay",
        )
        self.main_canvas.create_text(
            canvas_x1 - 8, divider_y + 8, text="拼接区", anchor="ne",
            fill="#7dff9b", font=("Sans", 11, "bold"), tags="roi_overlay",
        )
        if self.assembly_plan is None:
            self.main_canvas.create_text(
                canvas_x0 + 8, divider_y + 8,
                text="目标尺寸：拼接后自动恢复", anchor="nw",
                fill="#7dff9b", font=("Sans", 10, "bold"), tags="roi_overlay",
            )
            return
        target_x, target_y, target_width, target_height = target_rectangle_pixels(
            rectangle, self.assembly_config, self.assembly_plan.target_rect_mm
        )
        self.main_canvas.create_rectangle(
            origin_x + target_x * scale,
            origin_y + target_y * scale,
            origin_x + (target_x + target_width) * scale,
            origin_y + (target_y + target_height) * scale,
            outline="#7dff9b", width=2, dash=(5, 4), tags="roi_overlay",
        )
        self.main_canvas.create_text(
            origin_x + target_x * scale + 5,
            origin_y + target_y * scale + 5,
            text=(
                f"目标 {self.assembly_plan.recovered_size_mm[0]:.1f}×"
                f"{self.assembly_plan.recovered_size_mm[1]:.1f} mm"
            ),
            anchor="nw", fill="#7dff9b",
            font=("Sans", 10, "bold"), tags="roi_overlay",
        )

    def _valid_roi(self) -> tuple[int, int, int, int] | None:
        if self.roi is None or self.current_frame is None:
            return None
        x, y, width, height = self.roi
        if x < 0 or y < 0 or x + width > self.current_frame.shape[1] or y + height > self.current_frame.shape[0]:
            self.roi_state.set("A4 ROI：保存范围与当前相机分辨率不符，请重新框选")
            return None
        return self.roi

    def _source_roi(self) -> tuple[int, int, int, int] | None:
        """Return only the upper half of the full A4 ROI for detection."""
        roi = self._valid_roi()
        if roi is None:
            return None
        x, y, width, height = roi
        return x, y, width, max(1, round(height * self.assembly_config.split_fraction))

    def _save_roi(self) -> None:
        if self.roi is None:
            return
        ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
        x, y, width, height = self.roi
        document = {
            "format": "puzzle-device.a4-roi.v1",
            "camera_rotation_degrees": 180 if self.rotate_180 else 0,
            "roi": {"x": x, "y": y, "width": width, "height": height},
        }
        ROI_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_roi(self) -> None:
        if not ROI_PATH.exists():
            return
        try:
            document = json.loads(ROI_PATH.read_text(encoding="utf-8"))
            expected_rotation = 180 if self.rotate_180 else 0
            if document.get("camera_rotation_degrees") != expected_rotation:
                return
            values = document["roi"]
            self.roi = tuple(int(values[key]) for key in ("x", "y", "width", "height"))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.roi = None
            return
        x, y, width, height = self.roi
        self.roi_state.set(f"A4 ROI：已加载 x={x}, y={y}, w={width}, h={height}")

    def _clear_roi(self) -> None:
        self._invalidate_assembly("A4 ROI 已清除")
        self.roi = None
        self.roi_select_mode = False
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.main_canvas.configure(cursor="")
        try:
            ROI_PATH.unlink(missing_ok=True)
        except OSError as exc:
            messagebox.showwarning("清除失败", f"无法删除已保存的 ROI：{exc}")
        self.roi_state.set("A4 ROI：未设置，当前识别整张画面")
        self._reset_stability("ROI 已清除，重新稳定采样")
        self.status.set("A4 ROI 已清除；当前恢复识别整张相机画面。")
        self._detect_and_show()

    def _capture_background(self) -> None:
        if self.current_frame is None:
            messagebox.showwarning("没有画面", "摄像头还没有可用画面。")
            return
        self._invalidate_assembly("背景已更新")
        self.background = self.current_frame.copy()
        BACKGROUND_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(BACKGROUND_PATH), self.background):
            messagebox.showerror("保存失败", f"无法保存背景图：{BACKGROUND_PATH}")
            return
        self.background_state.set(f"背景：已采集并保存到 {BACKGROUND_PATH}")
        self._reset_stability("背景已更新，重新稳定采样")
        self.status.set("背景已采集。现在把碎片放回工作区，观察白色掩膜是否完整覆盖每一块。")
        self._detect_and_show()

    def _clear_background(self) -> None:
        self._invalidate_assembly("背景已清除")
        self.background = None
        self.background_state.set("背景：未采集（当前使用边缘颜色估计）")
        self._reset_stability("背景已清除，重新稳定采样")
        self._detect_and_show()

    def _save_parameters(self) -> None:
        config = self._config()
        if config is None:
            messagebox.showerror("参数无效", "请检查当前参数后再保存。")
            return
        save_detection_config(LOCAL_CONFIG_PATH, config)
        self._reset_stability("参数已保存，重新稳定采样")
        self.status.set(f"视觉参数已保存：{LOCAL_CONFIG_PATH}。后续识别和自动拼接可读取同一文件。")

    def _restore_saved_parameters(self) -> None:
        if not LOCAL_CONFIG_PATH.exists():
            messagebox.showinfo("没有保存参数", "还没有保存过本机参数，请先点击“保存参数”。")
            return
        try:
            config = load_detection_config(LOCAL_CONFIG_PATH)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            messagebox.showerror("恢复失败", f"无法读取上次保存的参数：{exc}")
            return
        self._apply_config(config)
        self.status.set(f"已恢复上次保存的视觉参数：{LOCAL_CONFIG_PATH}。")
        self._reset_stability("参数已恢复，重新稳定采样")
        self._detect_and_show()

    def _apply_config(self, config: DetectionConfig) -> None:
        labels_by_method = {value: label for label, value in self.method_values.items()}
        self.segmentation_method.set(labels_by_method[config.segmentation_method])
        self.min_area.set(f"{config.min_area_px:g}")
        self.max_area.set(f"{config.max_area_px:g}")
        self.threshold.set(
            "0" if config.color_distance_threshold is None
            else str(config.color_distance_threshold)
        )
        self.white_s_max.set(str(config.white_saturation_max))
        self.white_v_min.set(str(config.white_value_min))
        self.brightness_min.set(str(config.brightness_min))
        self.max_pieces.set(str(min(config.max_pieces, 4)))
        self.morphology_size.set(str(config.morphology_size))
        self.blur_size.set(str(config.gaussian_blur_size))
        self.canny_lower.set(str(config.canny_lower))
        self.canny_upper.set(str(config.canny_upper))
        self.epsilon_min.set(f"{config.polygon_epsilon_min:g}")
        self.epsilon_preferred.set(f"{config.polygon_epsilon_preferred:g}")
        self.epsilon_max.set(f"{config.polygon_epsilon_max:g}")
        self.min_vertices.set(str(config.min_vertices))
        self.max_vertices.set(str(config.max_vertices))
        self.minimum_pick_clearance.set(f"{config.minimum_pick_clearance_px:g}")

    def _load_saved_background(self) -> None:
        if not BACKGROUND_PATH.exists():
            return
        background = cv2.imread(str(BACKGROUND_PATH), cv2.IMREAD_COLOR)
        if background is None:
            return
        self.background = background
        self.background_state.set(f"背景：已加载 {BACKGROUND_PATH}")

    def _load_calibration(self) -> None:
        for path in CALIBRATION_PATHS:
            if not path.exists():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                matrix = np.asarray(document["matrix_pixel_to_pulse"], dtype=np.float64)
                if matrix.shape != (3, 3):
                    raise ValueError("matrix must be 3x3")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            self.calibration.matrix = matrix
            self.calibration_name = path.name
            self.matrix_state.set(f"坐标矩阵：已加载 {path.name}；列表同时显示中心脉冲（不会发送）。")
            return

    @staticmethod
    def _initial_config() -> DetectionConfig:
        for path in (LOCAL_CONFIG_PATH, DEFAULT_CONFIG_PATH):
            if not path.exists():
                continue
            try:
                return load_detection_config(path)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        return DetectionConfig()

    def _close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        if self.analysis_future is not None:
            self.analysis_future.cancel()
        self.analysis_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--no-rotate-180", action="store_true",
                        help="use raw camera orientation instead of the calibrated 180-degree view")
    args = parser.parse_args()
    root = tk.Tk()
    PieceDetectionApp(root, args.camera, rotate_180=not args.no_rotate_180)
    root.mainloop()


if __name__ == "__main__":
    main()
