#!/usr/bin/env python3
"""Fixed-layout one-button competition GUI for the physical puzzle device."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
import json

import cv2


# ---------------------------------------------------------------------------
# platform-adaptive CJK-friendly UI font
# ---------------------------------------------------------------------------
def _ui_font(size: int, bold: bool = False) -> tuple:
    """Return a platform-appropriate font spec with CJK glyph support."""
    weight = "bold" if bold else "normal"
    if sys.platform.startswith("win"):
        return ("Microsoft YaHei UI", size, weight)
    # Linux / macOS — fall back to tkinter default (still CJK-capable on
    # most modern systems with locale-aware fontconfig).
    return ("", size, weight)

from apps.puzzle_control_gui import PLAN_PATH, PREVIEW_PATH, PuzzleControlApp
from puzzle_device.calibration.gantry_protocol import (
    STATUS_ACTION_FAILED,
    STATUS_COMMAND_REJECTED,
)
from puzzle_device.competition import (
    COMPETITION_LIMIT_SECONDS,
    FIELD_WHITE_MODE,
    PLAYING_CARD_MODE,
    PLAYING_CARD_V2_MODE,
    SELF_ASSEMBLY_MODE,
    SELF_TRANSFER_MODE,
    CompetitionMode,
    format_competition_time,
)
from puzzle_device.planning import (
    build_movement_plan,
    build_transfer_plan,
    draw_assembly_preview,
    draw_card_candidate_gallery,
    draw_transfer_preview,
    legacy_4_0_config,
    relaxed_card_config,
    solve_composite_card_assembly,
    solve_self_assembly,
    solve_textured_assembly,
)
from puzzle_device.vision.piece_vision import draw_piece_observations
from puzzle_device.vision.image_io import read_image, write_image
from puzzle_device.paths import LOCAL_CONFIG_DIR, OUTPUT_DIR


RUN_LOG_DIR = OUTPUT_DIR / "competition_runs"
ROI_PATH = LOCAL_CONFIG_DIR / "a4_roi.json"
CARD2_SOURCE_FRAME_PATH = OUTPUT_DIR / "card2_source_frame.png"
CARD_CANDIDATE_GALLERY_PATH = OUTPUT_DIR / "card_candidate_gallery.png"


class CompetitionApp(PuzzleControlApp):
    """Competition-focused shell around the verified planning/control pipeline."""

    def __init__(
        self,
        root: tk.Tk,
        camera_index: int,
        serial_port: str | None,
        rotate_180: bool = True,
    ) -> None:
        self.competition_active = False
        self.competition_mode: CompetitionMode | None = None
        self.competition_started_at: float | None = None
        self.competition_finished_elapsed: float | None = None
        self.competition_result = "idle"
        self.competition_waiting_for_serial = False
        self.ignore_controller_status_until_next_run = False
        self.last_competition_mode: CompetitionMode | None = None
        self.debug_planning_method = tk.StringVar(master=root, value="white")
        self.run_log_path: Path | None = None
        self.showing_candidate_gallery = False
        self.candidate_gallery_available = False
        self._allow_plan_loading = False
        super().__init__(root, camera_index, serial_port, rotate_180)
        self._allow_plan_loading = True
        self.tasks = []
        self.preview = None
        self.plan_state.set("方案：等待选择比赛题目")
        self.task_state.set("任务：尚未开始")
        self.status.set("确认设备状态后，按下对应题目的大按钮并同时移除摄像头遮挡。")
        if serial_port:
            self.root.after(250, self._connect_serial)

    def _build_ui(self) -> None:
        self.root.title("拼图装置 - 比赛一键控制")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.compact_layout = screen_width <= 1100 or screen_height <= 650
        if self.compact_layout:
            # Leave room for the desktop panel and window decorations on the
            # 1080x600 display used with Jetson.
            window_width = min(1000, max(900, screen_width - 24))
            window_height = min(560, max(500, screen_height - 55))
            self.root.geometry(f"{window_width}x{window_height}+0+0")
            self.root.minsize(900, 500)
        else:
            self.root.geometry("1366x768")
            self.root.minsize(1100, 650)

        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        if self.compact_layout:
            style.configure("CompetitionTitle.TLabel", font=_ui_font(13, bold=True))
            style.configure("Timer.TLabel", font=("Consolas", 21, "bold"), foreground="#0b4f6c")
            style.configure("Mode.TButton", font=_ui_font(10, bold=True), padding=(4, 6))
            style.configure("Danger.TButton", font=_ui_font(9, bold=True), padding=4)
        else:
            style.configure("CompetitionTitle.TLabel", font=_ui_font(18, bold=True))
            style.configure("Timer.TLabel", font=("Consolas", 32, "bold"), foreground="#0b4f6c")
            style.configure("Mode.TButton", font=_ui_font(15, bold=True), padding=(12, 15))
            style.configure("Danger.TButton", font=_ui_font(11, bold=True), padding=8)

        header = ttk.Frame(self.root, padding=(7, 4) if self.compact_layout else (14, 8))
        header.pack(fill="x")
        ttk.Label(
            header, text="拼图装置比赛控制", style="CompetitionTitle.TLabel"
        ).pack(side="left")
        ttk.Label(
            header,
            text="比赛页一键运行；调试页保留单步操作",
            foreground="#8a3b00",
        ).pack(side="right", pady=(3, 0) if self.compact_layout else (7, 0))

        self.notebook = ttk.Notebook(self.root)
        notebook_padding = 4 if self.compact_layout else 12
        self.notebook.pack(fill="both", expand=True, padx=notebook_padding,
                           pady=(0, 4 if self.compact_layout else 8))
        page_padding = 4 if self.compact_layout else 8
        competition_page = ttk.Frame(self.notebook, padding=page_padding)
        debug_page = ttk.Frame(self.notebook, padding=page_padding)
        self.notebook.add(competition_page, text="比赛")
        self.notebook.add(debug_page, text="调试")
        self._build_competition_page(competition_page)
        self._build_debug_page(debug_page)

        footer = ttk.Frame(self.root, padding=(7, 2) if self.compact_layout else (14, 4))
        footer.pack(fill="x")
        ttk.Label(
            footer, textvariable=self.status, foreground="#174c75",
            wraplength=900 if self.compact_layout else 1320,
        ).pack(side="left", fill="x", expand=True)

    def _configure_planning_vision_profile(self, config) -> None:
        if (
            (
                self.competition_active
                and self.competition_mode is not None
                and self.competition_mode.key == "requirement_2_1"
            )
            or (not self.competition_active and self.debug_planning_method.get() == "white")
        ):
            config.polygon_vertex_strategy = "legacy_4"

    def _build_competition_page(self, page: ttk.Frame) -> None:
        page.columnconfigure(0, weight=5)
        page.columnconfigure(1, weight=3, minsize=300 if self.compact_layout else 390)
        page.rowconfigure(0, weight=1)

        section_gap = 4 if self.compact_layout else 8
        control_padding = 4 if self.compact_layout else 8
        image_box = ttk.LabelFrame(page, text="实时相机 / 拼接预览",
                                   padding=3 if self.compact_layout else 5)
        image_box.grid(row=0, column=0, sticky="nsew",
                       padx=(0, 4 if self.compact_layout else 8))
        self.canvas = tk.Canvas(image_box, background="#202326", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_image())

        controls = ttk.Frame(page)
        controls.grid(row=0, column=1, sticky="nsew")

        self.timer_text = tk.StringVar(value="00:00.0")
        self.competition_state = tk.StringVar(value="待机：请选择题目")
        timer_box = ttk.LabelFrame(controls, text="比赛计时（最长120秒）", padding=control_padding)
        timer_box.pack(fill="x")
        ttk.Label(timer_box, textvariable=self.timer_text, style="Timer.TLabel").pack()
        self.competition_state_label = tk.Label(
            timer_box,
            textvariable=self.competition_state,
            background="#d9e7ef",
            foreground="#12384a",
            font=_ui_font(9 if self.compact_layout else 12, bold=True),
            padx=8,
            pady=4 if self.compact_layout else 7,
        )
        self.competition_state_label.pack(fill="x", pady=(2, 0) if self.compact_layout else (4, 0))

        modes = ttk.LabelFrame(controls, text="选择比赛题目", padding=control_padding)
        modes.pack(fill="x", pady=(section_gap, 0))
        modes.columnconfigure(0, weight=1, uniform="competition_mode")
        modes.columnconfigure(1, weight=1, uniform="competition_mode")
        self.mode_buttons: list[ttk.Button] = []
        mode_specs = (
            (SELF_TRANSFER_MODE, "固定4块\n直接搬运到指定区域", 0, 0, 1),
            (SELF_ASSEMBLY_MODE, "固定4块\n轮廓几何拼接", 0, 1, 1),
            (FIELD_WHITE_MODE, "自动识别1～4块\n白色碎片几何拼接", 1, 0, 2),
            (PLAYING_CARD_MODE, "自动识别1～4块\n宽松几何+图案剖面", 2, 0, 1),
            (PLAYING_CARD_V2_MODE, "自动识别1～4块\n复合边+滑动接缝", 2, 1, 1),
        )
        for mode, note, row, column, columnspan in mode_specs:
            button_text = mode.title if self.compact_layout else f"{mode.title}\n{note}"
            button = ttk.Button(
                modes,
                text=button_text,
                style="Mode.TButton",
                command=lambda selected=mode: self._start_competition(selected),
            )
            button.grid(
                row=row,
                column=column,
                columnspan=columnspan,
                sticky="nsew",
                padx=(0, 4) if columnspan == 1 and column == 0
                else ((4, 0) if columnspan == 1 else (0, 0)),
                pady=(0, 3 if self.compact_layout else 6) if row < 2 else (0, 0),
            )
            self.mode_buttons.append(button)

        state_box = ttk.LabelFrame(controls, text="当前流程", padding=control_padding)
        state_box.pack(fill="x", pady=(section_gap, 0))
        for variable in (
            self.camera_state,
            self.serial_state,
            self.controller_state,
            self.plan_state,
            self.task_state,
        ):
            ttk.Label(
                state_box, textvariable=variable,
                wraplength=285 if self.compact_layout else 365,
                justify="left",
                font=_ui_font(8 if self.compact_layout else 9),
            ).pack(fill="x", pady=0 if self.compact_layout else 1)

        task_box = ttk.LabelFrame(
            controls, text="任务进度", padding=4 if self.compact_layout else 6
        )
        task_box.pack(fill="both", expand=True, pady=(section_gap, 0))
        self.task_list = tk.Listbox(
            task_box, font=("Consolas", 8 if self.compact_layout else 9),
            height=3 if self.compact_layout else 5,
        )
        self.task_list.pack(fill="both", expand=True)

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", pady=(section_gap, 0))
        self.stop_competition_button = ttk.Button(
            buttons,
            text="完全停止" if self.compact_layout else "完全停止上位机流程（不发下位机命令）",
            style="Danger.TButton",
            command=self._stop_competition,
        )
        self.stop_competition_button.pack(side="left", fill="x", expand=True)
        ttk.Button(buttons, text="复位", command=self._reset_competition_ui).pack(
            side="left", padx=(4 if self.compact_layout else 8, 0)
        )
        ttk.Button(
            buttons,
            text="刷新" if self.compact_layout else "刷新当前状态",
            command=self._refresh_competition_state,
        ).pack(side="left", padx=(4 if self.compact_layout else 8, 0))
        self.retry_button = ttk.Button(
            controls,
            text="再次运行" if self.compact_layout else "再次运行上一题",
            command=self._retry_last_competition,
        )
        self.retry_button.pack(fill="x", pady=(3 if self.compact_layout else 6, 0))

    def _build_debug_page(self, page: ttk.Frame) -> None:
        page.columnconfigure(0, weight=3)
        page.columnconfigure(1, weight=2, minsize=320 if self.compact_layout else 410)
        page.rowconfigure(0, weight=1)

        left = ttk.Frame(page)
        left.grid(row=0, column=0, sticky="nsew",
                  padx=(0, 4 if self.compact_layout else 8))
        debug_image = ttk.LabelFrame(
            left, text="相机 / 当前方案", padding=3 if self.compact_layout else 5
        )
        debug_image.pack(fill="both", expand=True)
        self.debug_canvas = tk.Canvas(
            debug_image, background="#202326", highlightthickness=0
        )
        self.debug_canvas.pack(fill="both", expand=True)
        self.debug_canvas.bind("<Configure>", lambda _event: self._draw_image())
        self.debug_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.debug_canvas.bind("<B1-Motion>", self._roi_drag)
        self.debug_canvas.bind("<ButtonRelease-1>", self._roi_release)
        self.roi_selecting = False
        self.roi_drag_start = None
        self.roi_drag_current = None

        log_box = ttk.LabelFrame(left, text="带时间戳的运行与通信日志",
                                 padding=3 if self.compact_layout else 5)
        log_box.pack(fill="x", pady=(4 if self.compact_layout else 8, 0))
        self.log = tk.Text(
            log_box, height=4 if self.compact_layout else 8,
            wrap="word", state="disabled",
            font=("Consolas", 8 if self.compact_layout else 9),
        )
        self.log.pack(fill="x")

        right_container = ttk.Frame(page)
        right_container.grid(row=0, column=1, sticky="nsew")
        right_container.columnconfigure(0, weight=1)
        right_container.rowconfigure(0, weight=1)
        self.debug_controls_canvas = tk.Canvas(
            right_container, highlightthickness=0, borderwidth=0
        )
        debug_scrollbar = ttk.Scrollbar(
            right_container, orient="vertical", command=self.debug_controls_canvas.yview
        )
        self.debug_controls_canvas.configure(yscrollcommand=debug_scrollbar.set)
        self.debug_controls_canvas.grid(row=0, column=0, sticky="nsew")
        debug_scrollbar.grid(row=0, column=1, sticky="ns")
        right = ttk.Frame(
            self.debug_controls_canvas,
            padding=(0, 0, 3 if self.compact_layout else 6,
                     4 if self.compact_layout else 8),
        )
        self.debug_controls_window = self.debug_controls_canvas.create_window(
            (0, 0), window=right, anchor="nw"
        )
        right.bind("<Configure>", self._update_debug_scroll_region)
        self.debug_controls_canvas.bind("<Configure>", self._resize_debug_controls)
        self.debug_controls_canvas.bind("<Enter>", self._enable_debug_mousewheel)
        self.debug_controls_canvas.bind("<Leave>", self._disable_debug_mousewheel)
        self.debug_controls_canvas.bind("<Button-4>", self._scroll_debug_controls)
        self.debug_controls_canvas.bind("<Button-5>", self._scroll_debug_controls)

        serial_box = ttk.LabelFrame(right, text="设备连接", padding=8)
        serial_box.pack(fill="x")
        ttk.Label(serial_box, text="CH340端口").grid(row=0, column=0, sticky="w")
        ttk.Entry(serial_box, textvariable=self.serial_port).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Button(serial_box, text="连接/重连串口", command=self._connect_serial).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        ttk.Button(
            serial_box,
            text="刷新并自动查找 CH340",
            command=self._refresh_serial_port,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(
            serial_box,
            text="安全通信自检（不启动电机）",
            command=self._start_serial_health_check,
        ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(
            serial_box,
            text="重启上位机串口（不复位STM32）",
            command=self._restart_serial_connection,
        ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(
            serial_box,
            text="空闲和识别阶段断线会自动重连；动作阶段绝不自动重发当前块。",
            foreground="#8a3b00",
            wraplength=380,
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        serial_box.columnconfigure(1, weight=1)

        plan_box = ttk.LabelFrame(right, text="识别与方案调试（不会自动执行）", padding=8)
        plan_box.pack(fill="x", pady=(8, 0))
        ttk.Label(plan_box, text="调试拼接算法").pack(anchor="w")
        ttk.Radiobutton(
            plan_box,
            text="2（1）普通白色碎片 / Git 4.0",
            variable=self.debug_planning_method,
            value="white",
        ).pack(anchor="w", pady=(3, 0))
        ttk.Radiobutton(
            plan_box,
            text="2（2）扑克牌 / 宽松几何 + 图案剖面",
            variable=self.debug_planning_method,
            value="card",
        ).pack(anchor="w")
        ttk.Radiobutton(
            plan_box,
            text="2（2）法2 / 复合边 + 滑动接缝（实验）",
            variable=self.debug_planning_method,
            value="card2",
        ).pack(anchor="w")
        self.piece_count_buttons = []
        self.calculate_plan_button = ttk.Button(
            plan_box, text="自动判断1～4块并计算方案", command=self._debug_auto_plan
        )
        self.calculate_plan_button.pack(fill="x")
        ttk.Button(
            plan_box, text="固定4块并计算方案", command=lambda: self._debug_fixed_plan(4)
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(plan_box, text="重新加载最新方案", command=self._load_plan).pack(
            fill="x", pady=(6, 0)
        )
        self.candidate_gallery_button = ttk.Button(
            plan_box,
            text="查看前5名候选",
            command=self._toggle_candidate_gallery,
            state="disabled",
        )
        self.candidate_gallery_button.pack(fill="x", pady=(6, 0))
        direction = ttk.Frame(plan_box)
        direction.pack(fill="x", pady=(6, 0))
        ttk.Label(direction, text="舵机方向").pack(side="left")
        ttk.Radiobutton(
            direction, text="同向", variable=self.servo_direction, value=1,
            command=self._reload_tasks,
        ).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(
            direction, text="反向", variable=self.servo_direction, value=-1,
            command=self._reload_tasks,
        ).pack(side="left", padx=(8, 0))

        execution = ttk.LabelFrame(right, text="运动调试（会真实运动）", padding=8)
        execution.pack(fill="x", pady=(8, 0))
        self.next_button = ttk.Button(
            execution, text="发送当前块（有安全确认）", command=self._send_next
        )
        self.next_button.pack(fill="x")
        self.auto_button = ttk.Button(
            execution, text="自动连续执行（有安全确认）", command=self._start_auto_run
        )
        self.auto_button.pack(fill="x", pady=(6, 0))
        self.stop_auto_button = ttk.Button(
            execution, text="停止后续自动任务", command=self._stop_auto_run
        )
        self.stop_auto_button.pack(fill="x", pady=(6, 0))
        ttk.Button(
            execution, text="确认当前块已完成并回零",
            command=self._confirm_completed_manually,
        ).pack(fill="x", pady=(6, 0))
        ttk.Button(
            execution, text="解除上位机等待（不推进）",
            command=self._clear_wait_without_advancing,
        ).pack(fill="x", pady=(6, 0))

        tools = ttk.LabelFrame(right, text="专用调试工具", padding=8)
        tools.pack(fill="x", pady=(8, 0))
        ttk.Button(
            tools,
            text="采集当前空桌面背景",
            command=self._capture_empty_background,
        ).pack(fill="x")
        ttk.Button(
            tools,
            text="重新框选 A4 ROI",
            command=self._arm_roi_selection,
        ).pack(fill="x", pady=(6, 0))
        ttk.Label(
            tools,
            text="请先移开所有碎片，再点击；背景会保存到 data/local/empty_work_area.png。",
            foreground="#8a3b00",
            wraplength=380,
            justify="left",
        ).pack(fill="x", pady=(4, 0))

    def _update_debug_scroll_region(self, _event=None) -> None:
        self.debug_controls_canvas.configure(
            scrollregion=self.debug_controls_canvas.bbox("all")
        )

    def _resize_debug_controls(self, event: tk.Event) -> None:
        self.debug_controls_canvas.itemconfigure(
            self.debug_controls_window, width=max(1, event.width)
        )

    def _enable_debug_mousewheel(self, _event=None) -> None:
        self.root.bind_all("<MouseWheel>", self._scroll_debug_controls)
        self.root.bind_all("<Button-4>", self._scroll_debug_controls)
        self.root.bind_all("<Button-5>", self._scroll_debug_controls)

    def _disable_debug_mousewheel(self, _event=None) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _scroll_debug_controls(self, event: tk.Event) -> None:
        step = self._mousewheel_step(event)
        if step:
            self.debug_controls_canvas.yview_scroll(step, "units")

    @staticmethod
    def _mousewheel_step(event: tk.Event) -> int:
        """Return -1 (up) or +1 (down) across Windows (<MouseWheel>) and Linux (<Button-4/5>)."""
        delta = getattr(event, "delta", 0)
        if delta:
            return -1 if delta > 0 else 1
        if event.num == 4:
            return -1
        if event.num == 5:
            return 1
        return 0

    def _capture_empty_background(self) -> None:
        """Capture a fresh rotated empty-work-area frame for background subtraction."""
        if self.competition_active or self.planning_active or self.waiting_for_completion:
            messagebox.showwarning(
                "无法采集背景",
                "当前正在比赛或识别执行，请先停止当前流程后再采集空桌面背景。",
            )
            return
        if self.capture is None:
            messagebox.showerror("相机不可用", "当前没有打开 USB 相机，无法采集背景。")
            return
        if not messagebox.askyesno(
            "确认采集空桌面背景",
            "请确认 A4 工作区内没有任何碎片、吸头或其他遮挡物。\n\n现在采集当前画面作为背景吗？",
        ):
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            messagebox.showerror("采集失败", "相机没有返回有效画面，请检查相机连接后重试。")
            return
        if self.rotate_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        try:
            BACKGROUND_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not write_image(BACKGROUND_PATH, frame):
                raise OSError("image write returned false")
        except (OSError, cv2.error) as exc:
            messagebox.showerror("保存背景失败", f"无法保存背景图：{exc}")
            return
        self.current_camera_frame = frame.copy()
        self.preview = frame.copy()
        self.planning_background = frame.copy()
        self._draw_image()
        self._append_log(
            f"BACKGROUND CAPTURED: {BACKGROUND_PATH} / {frame.shape[1]}x{frame.shape[0]}"
        )
        self.status.set(
            f"空桌面背景已保存：{BACKGROUND_PATH}。后续背景差分识别会自动使用新背景。"
        )
        messagebox.showinfo("背景采集完成", f"已保存空桌面背景：\n{BACKGROUND_PATH}")

    def _arm_roi_selection(self) -> None:
        if self.current_camera_frame is None:
            messagebox.showwarning("没有画面", "当前还没有可用的相机画面。")
            return
        self.roi_selecting = True
        self.roi_drag_start = None
        self.roi_drag_current = None
        self.debug_canvas.configure(cursor="crosshair")
        self.status.set("请在左侧相机画面拖拽框选完整 A4，松开鼠标后自动保存。")

    def _debug_canvas_to_image(self, x: int, y: int):
        if self.current_camera_frame is None:
            return None
        scale = getattr(self, "debug_display_scale", 0.0)
        origin = getattr(self, "debug_display_origin", (0, 0))
        if scale <= 0:
            return None
        px = (x - origin[0]) / scale
        py = (y - origin[1]) / scale
        height, width = self.current_camera_frame.shape[:2]
        if not (0 <= px < width and 0 <= py < height):
            return None
        return float(px), float(py)

    def _roi_press(self, event: tk.Event) -> None:
        if not self.roi_selecting:
            return
        point = self._debug_canvas_to_image(event.x, event.y)
        if point is not None:
            self.roi_drag_start = point
            self.roi_drag_current = point
            self._draw_image()

    def _roi_drag(self, event: tk.Event) -> None:
        if not self.roi_selecting or self.roi_drag_start is None:
            return
        point = self._debug_canvas_to_image(event.x, event.y)
        if point is not None:
            self.roi_drag_current = point
            self._draw_image()

    def _roi_release(self, event: tk.Event) -> None:
        if not self.roi_selecting or self.roi_drag_start is None:
            return
        point = self._debug_canvas_to_image(event.x, event.y) or self.roi_drag_current
        self.roi_selecting = False
        self.debug_canvas.configure(cursor="")
        if point is None:
            return
        x0, y0 = self.roi_drag_start
        x1, y1 = point
        x = round(min(x0, x1))
        y = round(min(y0, y1))
        width = round(abs(x1 - x0))
        height = round(abs(y1 - y0))
        self.roi_drag_start = None
        self.roi_drag_current = None
        frame_height, frame_width = self.current_camera_frame.shape[:2]
        if width < 100 or height < 100 or x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
            messagebox.showwarning("ROI 无效", "请重新框选完整的 A4 区域。")
            self._draw_image()
            return
        try:
            ROI_PATH.parent.mkdir(parents=True, exist_ok=True)
            ROI_PATH.write_text(json.dumps({
                "format": "puzzle-device.a4-roi.v1",
                "camera_rotation_degrees": 180 if self.rotate_180 else 0,
                "roi": {"x": x, "y": y, "width": width, "height": height},
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            self.planning_roi = (x, y, width, height)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("ROI 保存失败", str(exc))
            return
        # ROI 改变后，旧方案中的像素目标点和脉冲目标点都可能失效。
        # 清掉当前内存任务，避免用户误把旧的 1（1）方案直接发送到下位机。
        self.tasks = []
        self.current_task_index = 0
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self.auto_run_enabled = False
        self._cancel_auto_continue()
        self.plan_state.set("方案：ROI 已更新，请重新识别并计算方案")
        self.task_state.set("任务：等待按题目重新计算")
        self._refresh_task_list()
        self._draw_image()
        self._append_log(f"ROI UPDATED: x={x}, y={y}, w={width}, h={height}")
        self.status.set(
            f"A4 ROI 已保存：x={x}, y={y}, w={width}, h={height}；旧方案已作废，请重新计算 1（1）。"
        )

    def _load_plan(self, show_errors: bool = True) -> None:
        if not self._allow_plan_loading:
            return
        super()._load_plan(show_errors)
        available = False
        if self.tasks and not self.competition_active:
            try:
                document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
                available = bool(document.get("candidate_gallery_available"))
            except (OSError, json.JSONDecodeError, TypeError):
                available = False
        self._set_candidate_gallery_available(available)

    def _set_candidate_gallery_available(self, available: bool) -> None:
        """Reset the preview switch whenever a plan is loaded or replaced."""
        self.showing_candidate_gallery = False
        self.candidate_gallery_available = bool(
            available and CARD_CANDIDATE_GALLERY_PATH.exists()
        )
        button = getattr(self, "candidate_gallery_button", None)
        if button is not None:
            button.configure(
                text="查看前5名候选",
                state="normal" if self.candidate_gallery_available else "disabled",
            )

    def _toggle_candidate_gallery(self) -> None:
        """Switch the debug canvas between the normal overlay and top-five gallery."""
        if not self.candidate_gallery_available:
            messagebox.showinfo("暂无候选预览", "请先用 2（2）法1或法2计算一份方案。")
            return
        show_gallery = not self.showing_candidate_gallery
        image_path = CARD_CANDIDATE_GALLERY_PATH if show_gallery else PREVIEW_PATH
        preview = read_image(image_path, cv2.IMREAD_COLOR)
        if preview is None:
            messagebox.showwarning("预览不可用", f"无法读取预览图片：{image_path}")
            return
        self.preview = preview
        self.showing_candidate_gallery = show_gallery
        self.candidate_gallery_button.configure(
            text="返回正常预览" if show_gallery else "查看前5名候选"
        )
        self._draw_image()

    def _plan_geometry_is_verified(self, document: dict) -> bool:
        quality = document.get("quality", {})
        if document.get("operation_mode") == "transfer_only":
            return quality.get("transfer_only") is True and quality.get("geometry_verified") is True
        if (
            (
                self.competition_active
                and self.competition_mode is not None
                and self.competition_mode.planning_method in ("texture", "texture_v2")
            )
            or (
                not self.competition_active
                and self.debug_planning_method.get() in ("card", "card2")
            )
        ):
            return (
                quality.get("geometry_verified") is True
                and quality.get("texture_verified") is True
            )
        return super()._plan_geometry_is_verified(document)

    def _start_competition(self, mode: CompetitionMode) -> None:
        if self.competition_active or self.waiting_for_completion or self.planning_active:
            messagebox.showwarning("任务正在进行", "请等待当前流程结束或停止后续任务。")
            return
        if not self.serial.connected:
            self._connect_serial()
        if not self.serial.connected:
            messagebox.showerror("无法开始比赛", "必须先连接下位机串口。")
            return

        self._prepare_new_competition_run()
        self.ignore_controller_status_until_next_run = False
        self.competition_active = True
        self.competition_mode = mode
        self.last_competition_mode = mode
        self.competition_started_at = time.monotonic()
        self.competition_finished_elapsed = None
        self.competition_result = "running"
        self.competition_waiting_for_serial = False
        self.timer_text.set("00:00.0")
        self._set_mode_buttons_enabled(False)
        self._set_competition_state(f"运行中：{mode.title}", "#ffe59a", "#5d3a00")
        self._start_run_log(mode)
        self._append_log(f"COMPETITION START: {mode.key}")
        if not self._begin_plan_calculation(mode.expected_piece_count):
            self._finish_competition("failed", "无法启动视觉规划")
        elif mode.planning_method == "transfer":
            self.plan_state.set("搬运：正在稳定识别4块碎片…")
            self.status.set("只识别4块并放到下半区4个固定点，不调用拼接算法。")
        elif mode.planning_method == "self_assembly":
            self.plan_state.set("自备拼图：正在稳定识别4块碎片…")
            self.status.set("优先匹配固定四块100×60模板；失败后尝试通用拼接，再自动保底搬运。")
        elif mode.planning_method == "texture":
            self.plan_state.set("扑克牌法1：正在稳定识别1～4块碎片…")
            self.status.set("先宽松枚举矩形候选，再用牌面图案剖面、长宽比和圆角软提示排序。")
        elif mode.planning_method == "texture_v2":
            self.plan_state.set("扑克牌法2：正在稳定识别1～4块碎片…")
            self.status.set("正在使用复合边、滑动接缝和整体评分算法计算扑克牌拼接方案。")

    def _prepare_new_competition_run(self) -> None:
        """Discard old run state without touching calibration or saved parameters."""
        self._cancel_accept_timeout()
        self._cancel_auto_continue()
        self._cancel_serial_health_check()
        self.status_parser.reset()
        if self.serial.connected:
            self.serial.discard_input()
        self.tasks = []
        self.current_task_index = 0
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self.auto_run_enabled = False
        self.serial_fault_during_motion = False
        self.preview = None
        self._refresh_task_list()
        self.plan_state.set("方案：正在开始新一轮")
        self.task_state.set("任务：等待识别")

    def _retry_last_competition(self) -> None:
        if self.last_competition_mode is None:
            messagebox.showinfo("没有上一题", "请先运行一次题目。")
            return
        self._start_competition(self.last_competition_mode)

    def _solve_for_control(
        self, frame, pieces, roi, calibration, calibration_name, config
    ):
        if (
            self.competition_active
            and self.competition_mode is not None
            and self.competition_mode.planning_method == "transfer"
        ):
            def to_pulse(point: tuple[float, float]) -> tuple[int, int]:
                x, y = calibration.predict_pulse(*point)
                return round(x), round(y)

            document, target_polygons = build_transfer_plan(
                pieces,
                roi,
                pulse_mapper=to_pulse,
                calibration_file=calibration_name,
                config=config,
            )
            preview = draw_transfer_preview(
                draw_piece_observations(frame, pieces), pieces, target_polygons
            )
            return "solve", document, preview
        if (
            self.competition_active
            and self.competition_mode is not None
            and self.competition_mode.planning_method == "self_assembly"
        ):
            def to_pulse(point: tuple[float, float]) -> tuple[int, int]:
                x, y = calibration.predict_pulse(*point)
                return round(x), round(y)

            try:
                assembly = solve_self_assembly(
                    [piece.polygon for piece in pieces],
                    roi,
                    config,
                    require_upper_half=True,
                )
                document = build_movement_plan(
                    pieces,
                    assembly,
                    pulse_mapper=to_pulse,
                    calibration_file=calibration_name,
                    config=config,
                )
                document["planning_method"] = "self_fixed_template_or_general"
                preview = draw_assembly_preview(
                    draw_piece_observations(frame, pieces), pieces, assembly, config
                )
                return "solve", document, preview
            except (RuntimeError, ValueError, cv2.error) as exc:
                document, target_polygons = build_transfer_plan(
                    pieces,
                    roi,
                    pulse_mapper=to_pulse,
                    calibration_file=calibration_name,
                    config=config,
                )
                document["operation_mode"] = "assembly_fallback_transfer"
                document["planning_method"] = "fallback_fixed_transfer"
                document["fallback_reason"] = str(exc)
                document["quality"]["assembly_fallback"] = True
                preview = draw_transfer_preview(
                    draw_piece_observations(frame, pieces), pieces, target_polygons
                )
                return "solve", document, preview
        if (
            (
                self.competition_active
                and self.competition_mode is not None
                and self.competition_mode.planning_method == "texture_v2"
            )
            or (
                not self.competition_active
                and self.debug_planning_method.get() == "card2"
            )
        ):
            card_config = relaxed_card_config(config)
            if not self.competition_active:
                write_image(CARD2_SOURCE_FRAME_PATH, frame)
            assembly = solve_composite_card_assembly(
                frame,
                [piece.polygon for piece in pieces],
                roi,
                card_config,
                require_upper_half=True,
            )

            def to_pulse(point: tuple[float, float]) -> tuple[int, int]:
                x, y = calibration.predict_pulse(*point)
                return round(x), round(y)

            document = build_movement_plan(
                pieces,
                assembly,
                pulse_mapper=to_pulse,
                calibration_file=calibration_name,
                config=card_config,
            )
            document["planning_method"] = "experimental_composite_card_v2"
            preview = draw_assembly_preview(
                draw_piece_observations(frame, pieces),
                pieces,
                assembly,
                card_config,
            )
            gallery_available = False
            if not self.competition_active and assembly.candidate_diagnostics:
                gallery = draw_card_candidate_gallery(frame, assembly, card_config)
                gallery_available = write_image(CARD_CANDIDATE_GALLERY_PATH, gallery)
            document["candidate_gallery_available"] = gallery_available
            return "solve", document, preview
        if (
            (
                self.competition_active
                and self.competition_mode is not None
                and self.competition_mode.planning_method == "texture"
            )
            or (
                not self.competition_active
                and self.debug_planning_method.get() == "card"
            )
        ):
            card_config = relaxed_card_config(config)
            assembly = solve_textured_assembly(
                frame,
                [piece.polygon for piece in pieces],
                roi,
                card_config,
                require_upper_half=True,
            )

            def to_pulse(point: tuple[float, float]) -> tuple[int, int]:
                x, y = calibration.predict_pulse(*point)
                return round(x), round(y)

            document = build_movement_plan(
                pieces,
                assembly,
                pulse_mapper=to_pulse,
                calibration_file=calibration_name,
                config=card_config,
            )
            document["planning_method"] = "relaxed_card_geometry_pattern_profile"
            preview = draw_assembly_preview(
                draw_piece_observations(frame, pieces),
                pieces,
                assembly,
                card_config,
            )
            gallery_available = False
            if not self.competition_active and assembly.candidate_diagnostics:
                gallery = draw_card_candidate_gallery(frame, assembly, card_config)
                gallery_available = write_image(CARD_CANDIDATE_GALLERY_PATH, gallery)
            document["candidate_gallery_available"] = gallery_available
            return "solve", document, preview
        if (
            (
                self.competition_active
                and self.competition_mode is not None
                and self.competition_mode.key == "requirement_2_1"
            )
            or (not self.competition_active and self.debug_planning_method.get() == "white")
        ):
            legacy_config = legacy_4_0_config(config)
            return super()._solve_for_control(
                frame, pieces, roi, calibration, calibration_name, legacy_config
            )
        return super()._solve_for_control(
            frame, pieces, roi, calibration, calibration_name, config
        )

    def _is_transfer_mode(self) -> bool:
        return (
            self.competition_mode is not None
            and self.competition_mode.planning_method == "transfer"
        )

    def _planning_progress_text(self, piece_count: int) -> tuple[str, str]:
        if self.competition_active and self._is_transfer_mode():
            return (
                f"搬运：{piece_count} 块已稳定，正在生成下半区目标坐标…",
                "识别已经稳定，正在分配A4下半区4个固定放置点并换算目标脉冲；不进行拼接解算。",
            )
        if (
            self.competition_active
            and self.competition_mode is not None
            and self.competition_mode.planning_method == "self_assembly"
        ):
            return (
                f"自备拼图：{piece_count} 块已稳定，正在优先匹配完整边…",
                "按原100×60、5:3规则排序；若拼接失败会自动生成4块保底搬运方案。",
            )
        if (
            self.competition_active
            and self.competition_mode is not None
            and self.competition_mode.planning_method in ("texture", "texture_v2")
        ):
            method_name = "法2" if self.competition_mode.planning_method == "texture_v2" else "法1"
            return (
                f"扑克牌{method_name}：{piece_count} 块已稳定，正在计算拼接方案…",
                (
                    "正在枚举复合边和滑动接缝，并使用整体矩形与纹理评分排序。"
                    if method_name == "法2"
                    else "正在比较宽松矩形候选的牌面图案位置、颜色变化和接缝连续性。"
                ),
            )
        return super()._planning_progress_text(piece_count)

    def _loaded_plan_state_text(self, document: dict, task_count: int) -> str:
        if document.get("operation_mode") == "assembly_fallback_transfer":
            return f"保底搬运：拼接解算失败，已生成 {task_count} 块固定区域搬运任务"
        if document.get("operation_mode") == "transfer_only":
            return (
                f"搬运：已分配 {task_count} 个固定放置点，"
                "保持每块原方向，目标脉冲校验通过"
            )
        quality = document.get("quality", {})
        if quality.get("texture_verified") is True:
            return (
                f"扑克牌：已加载 {task_count} 块，花纹接缝分 "
                f"{float(quality.get('texture_score', 0.0)):.3f}（越低越连续），"
                "几何、坐标和舵机角度校验通过"
            )
        return super()._loaded_plan_state_text(document, task_count)

    def _on_plan_ready(self, document: dict, completed_count: int) -> None:
        if not self.competition_active:
            super()._on_plan_ready(document, completed_count)
            return
        self._load_plan(show_errors=False)
        if not self.tasks:
            self._finish_competition("failed", "生成的方案没有可执行任务")
            return
        operation_mode = document.get("operation_mode")
        operation = (
            "FALLBACK_TRANSFER" if operation_mode == "assembly_fallback_transfer"
            else "TRANSFER" if operation_mode == "transfer_only"
            else "ASSEMBLY"
        )
        self._append_log(f"COMPETITION {operation} READY: {completed_count} piece(s)")
        if not self.serial.connected:
            self.competition_waiting_for_serial = True
            self._set_competition_state(
                "方案完成：等待串口恢复", "#ffd7a1", "#6b3900"
            )
            self.status.set("方案已保存并校验通过；等待串口自动恢复后再发送第一块。")
            self._schedule_serial_reconnect()
            return
        self._start_competition_execution()

    def _start_competition_execution(self) -> None:
        self.competition_waiting_for_serial = False
        if self._is_transfer_mode():
            self.status.set("搬运坐标校验通过，正在发送第一块；不进行拼接解算。")
        else:
            self.status.set("拼接方案校验通过，正在自动发送第一块；后续只在收到B1后继续。")
        if not self._begin_auto_run(confirm=False):
            self._finish_competition("failed", "自动连续执行未能启动")

    def _finish_plan_failure(self, exc: Exception) -> None:
        if self.competition_active and self._is_transfer_mode():
            self.planning_active = False
            self.planning_expected_piece_count = None
            self._set_piece_count_controls_enabled(True)
            self.calculate_plan_button.configure(text="自动判断1～4块并计算方案")
            self.plan_state.set("搬运：坐标生成失败，请检查识别、ROI和标定")
            self._append_log(f"TRANSFER ERROR: {exc}")
            messagebox.showerror("搬运坐标生成失败", str(exc))
            self._finish_competition("failed", f"搬运坐标生成失败：{exc}")
            return
        super()._finish_plan_failure(exc)
        if self.competition_active:
            self._finish_competition("failed", f"拼接方案失败：{exc}")

    def _handle_status(self, status: int) -> None:
        if (
            self.ignore_controller_status_until_next_run
            and not self.serial_health_check_pending
        ):
            self._append_log(
                f"STATUS {status:02X} IGNORED: competition was fully stopped by operator"
            )
            return
        super()._handle_status(status)
        if self.competition_active and status in (
            STATUS_COMMAND_REJECTED,
            STATUS_ACTION_FAILED,
        ):
            self._finish_competition("failed", f"下位机返回故障状态 {status:02X}")

    def _transmit_task(self, task) -> bool:
        # An explicit new motion command starts a new controller interaction;
        # stale-stop filtering is no longer needed. B1/B3 received before B0
        # are still rejected by the base protocol state machine.
        self.ignore_controller_status_until_next_run = False
        sent = super()._transmit_task(task)
        if self.competition_active and not sent:
            self._finish_competition("failed", f"P{task.piece_id} 串口发送失败")
        return sent

    def _on_all_tasks_complete(self, _was_auto_run: bool, _show_dialog: bool) -> None:
        if self.competition_active:
            self._finish_competition("success", "全部碎片完成、舵机归位且XY已回零")
            self._completion_signal()
            return
        super()._on_all_tasks_complete(_was_auto_run, _show_dialog)

    def _finish_competition(self, result: str, reason: str) -> None:
        if not self.competition_active and self.competition_result != "running":
            return
        elapsed = self._elapsed_seconds()
        self.competition_active = False
        self.competition_waiting_for_serial = False
        self.competition_finished_elapsed = elapsed
        self.competition_result = result
        self.auto_run_enabled = False
        self._cancel_auto_continue()
        if self.planning_active:
            self.planning_active = False
            self.planning_generation += 1
            if self.planning_future is not None:
                self.planning_future.cancel()
        self._set_mode_buttons_enabled(True)
        if result == "success":
            self._set_competition_state(
                f"完成：{format_competition_time(elapsed)}", "#8fe3a5", "#124f25"
            )
            self.status.set(f"比赛流程完成：{reason}。")
        else:
            self._set_competition_state(
                f"停止：{format_competition_time(elapsed)}", "#ffb0a8", "#68150d"
            )
            self.status.set(f"比赛流程停止：{reason}。当前已发送的单块动作不会被软件急停。")
        self._append_log(
            f"COMPETITION {result.upper()}: {format_competition_time(elapsed)} / {reason}"
        )
        self.run_log_path = None

    def _stop_competition(self) -> None:
        if not (
            self.competition_active
            or self.planning_active
            or self.waiting_for_completion
            or self.auto_run_enabled
            or self.tasks
        ):
            self.status.set("当前没有正在运行的比赛流程。")
            return

        elapsed = self._elapsed_seconds()
        had_in_flight_action = self.waiting_for_completion

        # Stop every PC-side continuation without transmitting any serial
        # command. The STM32 may finish an already accepted mechanical action,
        # but its subsequent B0/B1/B2/B3 frames are deliberately ignored until
        # the operator starts a new competition run.
        self.competition_active = False
        self.competition_waiting_for_serial = False
        self.competition_finished_elapsed = elapsed
        self.competition_result = "stopped"
        self.planning_active = False
        self.planning_expected_piece_count = None
        self.planning_generation += 1
        if self.planning_future is not None:
            self.planning_future.cancel()
            self.planning_future = None
            self.planning_future_generation = None
        self.planning_tracker.reset("比赛已完全停止")
        self.auto_run_enabled = False
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self._cancel_accept_timeout()
        self._cancel_auto_continue()
        self._cancel_serial_health_check()
        self.ignore_controller_status_until_next_run = True
        self.status_parser.reset()
        if self.serial.connected:
            try:
                self.serial.discard_input()
            except Exception as exc:
                self._append_log(f"STOP INPUT DISCARD FAILED: {exc}")
        self.tasks = []
        self.current_task_index = 0
        self._refresh_task_list()
        self._set_piece_count_controls_enabled(True)
        self._set_mode_buttons_enabled(True)
        self.plan_state.set("方案：已完全停止并清空")
        self.task_state.set("任务：已停止，无后续发送")
        self.controller_state.set("下位机：上位机已停止处理本轮状态")
        self._set_competition_state(
            f"已停止：{format_competition_time(elapsed)}", "#ffb0a8", "#68150d"
        )
        self._append_log(
            "COMPETITION HARD STOP (PC ONLY): all continuations cleared; "
            f"in_flight_action={had_in_flight_action}; no frame transmitted"
        )
        self.run_log_path = None
        if had_in_flight_action:
            self.status.set(
                "上位机比赛流程已完全停止，不会再处理或提示本轮下位机状态；"
                "停止前已经发出的当前机械动作仍会由下位机自行完成。"
            )
        else:
            self.status.set(
                "上位机比赛流程已完全停止：识别、计算、等待和后续发送均已清空；"
                "没有向下位机发送任何停止命令。"
            )

    def _reset_competition_ui(self) -> None:
        if self.competition_active or self.waiting_for_completion:
            messagebox.showwarning("不能复位", "当前流程或机械动作尚未结束。")
            return
        self.competition_mode = None
        self.competition_started_at = None
        self.competition_finished_elapsed = None
        self.competition_result = "idle"
        self.competition_waiting_for_serial = False
        self.timer_text.set("00:00.0")
        self._set_competition_state("待机：请选择题目", "#d9e7ef", "#12384a")
        self.tasks = []
        self.current_task_index = 0
        self.preview = None
        self._set_candidate_gallery_available(False)
        self._refresh_task_list()
        self.plan_state.set("方案：等待选择比赛题目")
        self.task_state.set("任务：尚未开始")
        self.status.set("已复位比赛界面；相机和串口连接保持不变。")

    def _refresh_competition_state(self) -> None:
        """Clear stale planning/execution state without touching hardware setup."""
        if self.competition_active or self.waiting_for_completion:
            messagebox.showwarning(
                "机械动作进行中",
                "当前动作尚未收到完成状态，不能刷新；避免丢失下位机执行状态。",
            )
            return

        # A previous debug calculation may still have a worker result queued.
        self.planning_active = False
        self.planning_expected_piece_count = None
        self.planning_generation += 1
        if self.planning_future is not None:
            self.planning_future.cancel()
            self.planning_future = None
            self.planning_future_generation = None
        self._set_piece_count_controls_enabled(True)
        self.calculate_plan_button.configure(text="一键重新识别并计算拼接方案")
        self._cancel_accept_timeout()
        self._cancel_auto_continue()
        self._cancel_serial_health_check()
        self.status_parser.reset()
        if self.serial.connected:
            self.serial.discard_input()
        self._reset_competition_ui()
        self._append_log("COMPETITION STATE REFRESHED")
        self.status.set("比赛状态已刷新；旧任务、预览、计时和串口缓存已清除，设备连接保持不变。")

    def _debug_auto_plan(self) -> None:
        if self.planning_active:
            self._cancel_plan_calculation()
        else:
            self._set_candidate_gallery_available(False)
            self._begin_plan_calculation(None)

    def _debug_fixed_plan(self, count: int) -> None:
        if self.planning_active:
            self._cancel_plan_calculation()
        else:
            self._set_candidate_gallery_available(False)
            self._begin_plan_calculation(count)

    def _set_mode_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.mode_buttons:
            button.configure(state=state)

    def _set_competition_state(self, text: str, background: str, foreground: str) -> None:
        self.competition_state.set(text)
        self.competition_state_label.configure(background=background, foreground=foreground)

    def _elapsed_seconds(self) -> float:
        if self.competition_finished_elapsed is not None:
            return self.competition_finished_elapsed
        if self.competition_started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self.competition_started_at)

    def _update(self) -> None:
        super()._update()
        elapsed = self._elapsed_seconds()
        if self.competition_started_at is not None:
            self.timer_text.set(format_competition_time(elapsed))
        if self.competition_active and elapsed >= COMPETITION_LIMIT_SECONDS:
            self.planning_active = False
            self.planning_generation += 1
            self._finish_competition("timeout", "超过题目规定的120秒")
        elif (
            self.competition_active
            and self.competition_waiting_for_serial
            and self.serial.connected
        ):
            self._append_log("COMPETITION SERIAL RECOVERED: starting execution")
            self._start_competition_execution()
        elif self.competition_active and not self.serial.connected:
            if self.serial_fault_during_motion:
                self._finish_competition("failed", "动作期间串口连接中断，状态未知")
            else:
                self._set_competition_state(
                    "通信恢复中：不会发送动作", "#ffd7a1", "#6b3900"
                )
                self._schedule_serial_reconnect()

    def _draw_image(self) -> None:
        if self.preview is None:
            return
        self._draw_image_to_canvas(self.canvas)
        self._draw_image_to_canvas(self.debug_canvas)

    def _draw_image_to_canvas(self, canvas: tk.Canvas) -> None:
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 2 or height < 2:
            return
        scale = min(width / self.preview.shape[1], height / self.preview.shape[0])
        shown_width = max(1, round(self.preview.shape[1] * scale))
        shown_height = max(1, round(self.preview.shape[0] * scale))
        origin = ((width - shown_width) / 2.0, (height - shown_height) / 2.0)
        # 鼠标框选 ROI 时需要把画布坐标反算回原始图像坐标。
        # 只记录调试画布的缩放和留白位置，不影响比赛主画布。
        if getattr(self, "debug_canvas", None) is canvas:
            self.debug_display_scale = scale
            self.debug_display_origin = origin
        shown = cv2.resize(
            self.preview,
            (shown_width, shown_height),
            interpolation=cv2.INTER_AREA,
        )
        ok, png = cv2.imencode(".png", shown)
        if not ok:
            return
        photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        if canvas is self.canvas:
            self.photo = photo
        else:
            self.debug_photo = photo
        canvas.delete("all")
        canvas.create_image(origin[0], origin[1], image=photo, anchor="nw")

        # 调试页叠加 ROI。保存的 ROI 用蓝色，当前拖拽中的 ROI 用黄色虚线。
        if getattr(self, "debug_canvas", None) is canvas:
            roi = None
            if self.roi_selecting and self.roi_drag_start and self.roi_drag_current:
                x0, y0 = self.roi_drag_start
                x1, y1 = self.roi_drag_current
                roi = (
                    min(x0, x1), min(y0, y1),
                    abs(x1 - x0), abs(y1 - y0),
                    "#ffd400", (8, 5),
                )
            elif getattr(self, "planning_roi", None):
                x, y, roi_width, roi_height = self.planning_roi
                roi = (x, y, roi_width, roi_height, "#39a9ff", None)
            if roi is not None:
                x, y, roi_width, roi_height, color, dash = roi
                coords = (
                    origin[0] + x * scale,
                    origin[1] + y * scale,
                    origin[0] + (x + roi_width) * scale,
                    origin[1] + (y + roi_height) * scale,
                )
                options = {"outline": color, "width": 2, "tags": "roi_overlay"}
                if dash:
                    options["dash"] = dash
                canvas.create_rectangle(*coords, **options)
                canvas.create_text(
                    coords[0] + 6, coords[1] + 6,
                    text="A4 ROI", anchor="nw", fill=color,
                    font=_ui_font(10, bold=True),
                    tags="roi_overlay",
                )

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {message}"
        self.log.configure(state="normal")
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")
        if self.run_log_path is not None:
            try:
                with self.run_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
            except OSError:
                pass

    def _start_run_log(self, mode: CompetitionMode) -> None:
        RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_log_path = RUN_LOG_DIR / f"{stamp}_{mode.key}.log"

    def _completion_signal(self) -> None:
        for delay in (0, 250, 500):
            self.root.after(delay, self.root.bell)

    def _release_resources(self) -> None:
        self._cancel_accept_timeout()
        self._cancel_auto_continue()
        self.planning_active = False
        if self.planning_future is not None:
            self.planning_future.cancel()
        self.planning_executor.shutdown(wait=False, cancel_futures=True)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.serial.close()

    def _close(self) -> None:
        self._release_resources()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--serial", help="CH340 serial port (COM30 on Windows, /dev/ttyUSB0 on Linux)")
    parser.add_argument(
        "--no-rotate-180", action="store_true", help="use raw camera orientation"
    )
    args = parser.parse_args()
    root = tk.Tk()
    CompetitionApp(root, args.camera, args.serial, rotate_180=not args.no_rotate_180)
    root.mainloop()


if __name__ == "__main__":
    main()
