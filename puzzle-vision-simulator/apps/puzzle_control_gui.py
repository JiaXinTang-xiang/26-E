#!/usr/bin/env python3
"""Competition control GUI for vision planning and gantry execution."""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from puzzle_device.calibration.gantry_protocol import (
    GantryStatusParser,
    OptionalSerialPort,
    STATUS_ACTION_FAILED,
    STATUS_ACTION_COMPLETE,
    STATUS_COMMAND_ACCEPTED,
    STATUS_COMMAND_REJECTED,
    STATUS_NAMES,
    build_dual_angle_pick_and_place_frame,
)
from puzzle_device.planning import build_execution_tasks
from puzzle_device.calibration.manual_calibration import PixelToGantryCalibration
from puzzle_device.planning import (
    AssemblyConfig,
    build_movement_plan,
    draw_assembly_preview,
    solve_assembly,
)
from puzzle_device.vision.camera import open_uvc_camera
from puzzle_device.vision.piece_vision import (
    DetectionConfig,
    detect_piece_observations,
    draw_piece_observations,
    load_detection_config,
)
from puzzle_device.vision.stability import PieceStabilityTracker


PLAN_PATH = Path("output/assembly_plan.json")
PREVIEW_PATH = Path("output/assembly_preview.png")
FAILED_PLAN_PATH = Path("output/assembly_plan_failed.json")
FAILED_PREVIEW_PATH = Path("output/assembly_preview_failed.png")
BACKGROUND_PATH = Path("data/local/empty_work_area.png")
DEFAULT_CONFIG_PATH = Path("configs/vision_detection.json")
LOCAL_CONFIG_PATH = Path("configs/local/vision_detection.json")
ROI_PATH = Path("configs/local/a4_roi.json")
CALIBRATION_PATHS = (
    Path("configs/local/calibration.json"),
    Path("configs/local/calibration_temporary.json"),
)
SERVO_HOME_ANGLE = 135
SUPPORTED_PIECE_COUNTS = (1, 2, 3, 4)


class PuzzleControlApp:
    """Load a verified plan and execute it one piece at a time."""

    def __init__(self, root: tk.Tk, camera_index: int, serial_port: str | None,
                 rotate_180: bool = True):
        self.root = root
        self.root.title("拼图装置 - 正式控制")
        self.root.geometry("1280x780")
        self.root.minsize(1060, 650)
        self.camera_index = camera_index
        self.rotate_180 = rotate_180
        self.capture, camera_info = open_uvc_camera(camera_index)
        self.serial = OptionalSerialPort(serial_port)
        self.status_parser = GantryStatusParser()
        self.tasks = []
        self.current_task_index = 0
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self.auto_run_enabled = False
        self.accept_timeout_job = None
        self.auto_continue_job = None
        self.tx_bytes = 0
        self.rx_bytes = 0
        self.accept_rx_start = 0
        self.preview = None
        self.photo = None
        self.current_camera_frame = None
        self.planning_active = False
        self.planning_future: Future | None = None
        self.planning_future_generation: int | None = None
        self.planning_generation = 0
        self.planning_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="control-vision"
        )
        self.planning_tracker = PieceStabilityTracker()
        self.planning_next_time = 0.0
        self.planning_config: DetectionConfig | None = None
        self.planning_background = None
        self.planning_roi = None
        self.planning_calibration: PixelToGantryCalibration | None = None
        self.planning_calibration_name = None
        self.planning_expected_piece_count: int | None = None
        self.assembly_config = AssemblyConfig()

        self.serial_port = tk.StringVar(value=serial_port or "")
        self.servo_direction = tk.IntVar(value=1)
        self.expected_piece_count = tk.IntVar(value=max(SUPPORTED_PIECE_COUNTS))
        self.camera_state = tk.StringVar(
            value="相机：未打开" if camera_info is None
            else f"相机：{camera_info.describe()} / {'旋转180°' if rotate_180 else '原始方向'}"
        )
        self.serial_state = tk.StringVar(value="串口：未连接")
        self.controller_state = tk.StringVar(value="下位机：尚未收到状态")
        self.plan_state = tk.StringVar(value="方案：尚未加载")
        self.task_state = tk.StringVar(value="任务：等待加载拼接方案")
        self.status = tk.StringVar(value="先在识别界面锁定碎片并计算拼接方案。")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._load_plan(show_errors=False)
        self._update()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 16, "bold"))

        header = ttk.Frame(self.root, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="拼图装置正式控制", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="发送前必须确认吸头和工作区安全", foreground="#a01d1d").pack(
            side="right", pady=(5, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        image_frame = ttk.LabelFrame(body, text="相机 / 拼接预览", padding=5)
        controls = ttk.Frame(body, width=390)
        body.add(image_frame, weight=4)
        body.add(controls, weight=2)
        self.canvas = tk.Canvas(image_frame, background="#202326", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._draw_image())

        states = ttk.LabelFrame(controls, text="设备状态", padding=9)
        states.pack(fill="x")
        for variable in (
            self.camera_state, self.serial_state, self.controller_state,
            self.plan_state, self.task_state,
        ):
            ttk.Label(states, textvariable=variable, wraplength=360, justify="left").pack(
                fill="x", pady=2)

        serial_box = ttk.LabelFrame(controls, text="串口", padding=9)
        serial_box.pack(fill="x", pady=(9, 0))
        ttk.Label(serial_box, text="端口").grid(row=0, column=0, sticky="w")
        ttk.Entry(serial_box, textvariable=self.serial_port, width=20).grid(
            row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(serial_box, text="连接", command=self._connect_serial).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0))
        serial_box.columnconfigure(1, weight=1)

        plan_box = ttk.LabelFrame(controls, text="拼接方案", padding=9)
        plan_box.pack(fill="x", pady=(9, 0))
        piece_count = ttk.Frame(plan_box)
        piece_count.pack(fill="x", pady=(0, 7))
        ttk.Label(piece_count, text="本轮碎片数").pack(side="left")
        self.piece_count_buttons = []
        for count in SUPPORTED_PIECE_COUNTS:
            button = ttk.Radiobutton(
                piece_count, text=str(count), variable=self.expected_piece_count, value=count
            )
            button.pack(side="left", padx=(12 if count == 1 else 8, 0))
            self.piece_count_buttons.append(button)
        self.calculate_plan_button = ttk.Button(
            plan_box, text="一键重新识别并计算拼接方案",
            command=self._start_plan_calculation,
        )
        self.calculate_plan_button.pack(fill="x")
        ttk.Button(plan_box, text="重新加载最新方案", command=self._load_plan).pack(fill="x")
        direction = ttk.Frame(plan_box)
        direction.pack(fill="x", pady=(7, 0))
        ttk.Label(direction, text="舵机旋转方向").pack(side="left")
        ttk.Radiobutton(direction, text="同向", variable=self.servo_direction, value=1,
                        command=self._reload_tasks).pack(side="left", padx=(12, 0))
        ttk.Radiobutton(direction, text="反向", variable=self.servo_direction, value=-1,
                        command=self._reload_tasks).pack(side="left", padx=(8, 0))
        ttk.Label(
            plan_box,
            text=(
                "抓取角 = 135° - 方向 × 旋转角 / 2；"
                "放置角 = 135° + 方向 × 旋转角 / 2。"
                "若实机转反，切换方向后重新加载。"
            ),
            foreground="#7c3f00", wraplength=350, justify="left",
        ).pack(fill="x", pady=(6, 0))

        execute = ttk.LabelFrame(controls, text="执行控制", padding=9)
        execute.pack(fill="x", pady=(9, 0))
        self.next_button = ttk.Button(
            execute, text="发送当前块（每次仅一块）", command=self._send_next)
        self.next_button.pack(fill="x")
        self.auto_button = ttk.Button(
            execute, text="自动连续执行全部", command=self._start_auto_run)
        self.auto_button.pack(fill="x", pady=(7, 0))
        self.stop_auto_button = ttk.Button(
            execute, text="停止自动执行（当前块不打断）", command=self._stop_auto_run)
        self.stop_auto_button.pack(fill="x", pady=(7, 0))
        ttk.Button(
            execute, text="确认当前块已完成并回零",
            command=self._confirm_completed_manually,
        ).pack(fill="x", pady=(7, 0))
        ttk.Button(
            execute, text="取消本次发送状态（不推进）",
            command=self._clear_wait_without_advancing,
        ).pack(fill="x", pady=(7, 0))
        ttk.Label(
            execute,
            text=(
                "自动模式仅在收到 B1 后发送下一块；若需中止后续任务，"
                "点击停止自动执行，当前正在执行的块不会被打断。"
            ),
            foreground="#a01d1d", wraplength=350, justify="left",
        ).pack(fill="x", pady=(6, 0))

        task_box = ttk.LabelFrame(controls, text="任务列表", padding=9)
        task_box.pack(fill="both", expand=True, pady=(9, 0))
        self.task_list = tk.Listbox(task_box, font=("Consolas", 9), height=9)
        self.task_list.pack(fill="both", expand=True)

        log_box = ttk.LabelFrame(controls, text="通信记录", padding=7)
        log_box.pack(fill="both", expand=True, pady=(9, 0))
        self.log = tk.Text(log_box, height=7, wrap="word", state="disabled",
                           font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

        footer = ttk.Frame(self.root, padding=(14, 6))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, foreground="#174c75").pack(side="left")

    def _connect_serial(self) -> None:
        try:
            self.serial.close()
            self.serial.port = self.serial_port.get().strip() or None
            self.serial.connect()
        except Exception as exc:
            messagebox.showerror("串口连接失败", str(exc))
            return
        if self.serial.connected:
            self.status_parser.reset()
            self._cancel_accept_timeout()
            self._cancel_auto_continue()
            self.auto_run_enabled = False
            self.waiting_for_accept = False
            self.tx_bytes = 0
            self.rx_bytes = 0
            self._update_serial_state()
            self.controller_state.set("下位机：等待发送命令")
            self._append_log(f"OPEN {self.serial.port} @ {self.serial.baudrate}")
        else:
            self.serial_state.set("串口：未填写端口，模拟模式")

    def _start_plan_calculation(self) -> None:
        if self.planning_active:
            self._cancel_plan_calculation()
            return
        if self.waiting_for_completion or self.auto_run_enabled:
            messagebox.showwarning("机械正在执行", "请等待当前任务完成并停止自动执行后再重新计算。")
            return
        if self.capture is None:
            messagebox.showerror("相机不可用", "没有可用的 USB 相机画面，无法重新识别。")
            return
        try:
            config = self._load_planning_config()
            roi = self._load_planning_roi()
            calibration, calibration_name = self._load_planning_calibration()
            background = cv2.imread(str(BACKGROUND_PATH), cv2.IMREAD_COLOR)
            if config.segmentation_method == "background" and background is None:
                raise ValueError(f"背景差分模式缺少背景图：{BACKGROUND_PATH}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("无法开始计算", str(exc))
            return

        expected_count = self.expected_piece_count.get()
        if expected_count not in SUPPORTED_PIECE_COUNTS:
            messagebox.showerror("碎片数量无效", "本轮碎片数只能选择 1、2、3 或 4。")
            return
        self.planning_config = config
        self.planning_roi = roi
        self.planning_calibration = calibration
        self.planning_calibration_name = calibration_name
        self.planning_background = background
        self.planning_expected_piece_count = expected_count
        self.planning_generation += 1
        self.planning_tracker.reset(f"等待稳定识别 {expected_count} 块碎片")
        self.planning_active = True
        self.planning_next_time = 0.0
        self.preview = None
        self._set_piece_count_controls_enabled(False)
        self.calculate_plan_button.configure(text="停止本次识别计算")
        self.plan_state.set(f"方案：正在稳定识别 {expected_count} 块碎片…")
        self.status.set(
            f"请保持 {expected_count} 块碎片和相机静止；稳定采样后会自动计算并加载方案。"
        )
        self._append_log(f"PLAN START: detecting {expected_count} stable pieces")

    def _cancel_plan_calculation(self) -> None:
        self.planning_active = False
        self.planning_expected_piece_count = None
        self.planning_generation += 1
        self.planning_tracker.reset("已停止")
        if self.planning_future is not None:
            self.planning_future.cancel()
        self._set_piece_count_controls_enabled(True)
        self.calculate_plan_button.configure(text="一键重新识别并计算拼接方案")
        self.plan_state.set("方案：本次重新识别已停止，可加载旧方案或重新计算")
        self.status.set("已停止本次识别计算；没有发送任何电机命令。")
        self._append_log("PLAN STOP")

    def _set_piece_count_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in self.piece_count_buttons:
            button.configure(state=state)

    @staticmethod
    def _load_planning_config() -> DetectionConfig:
        for path in (LOCAL_CONFIG_PATH, DEFAULT_CONFIG_PATH):
            if path.exists():
                return load_detection_config(path)
        return DetectionConfig()

    def _load_planning_roi(self) -> tuple[int, int, int, int]:
        document = json.loads(ROI_PATH.read_text(encoding="utf-8"))
        expected_rotation = 180 if self.rotate_180 else 0
        if document.get("camera_rotation_degrees") != expected_rotation:
            raise ValueError("A4 ROI 的相机旋转方向与当前界面不一致，请重新框选 ROI。")
        values = document["roi"]
        roi = tuple(int(values[key]) for key in ("x", "y", "width", "height"))
        if roi[2] <= 0 or roi[3] <= 0:
            raise ValueError("保存的 A4 ROI 无效，请回识别界面重新框选。")
        return roi

    @staticmethod
    def _load_planning_calibration() -> tuple[PixelToGantryCalibration, str]:
        for path in CALIBRATION_PATHS:
            if not path.exists():
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            matrix = np.asarray(document["matrix_pixel_to_pulse"], dtype=np.float64)
            if matrix.shape != (3, 3):
                continue
            calibration = PixelToGantryCalibration()
            calibration.matrix = matrix
            return calibration, path.name
        raise ValueError("未找到有效的像素到脉冲标定矩阵。")

    @staticmethod
    def _detect_for_control(
        frame: np.ndarray,
        background: np.ndarray | None,
        config: DetectionConfig,
        source_roi: tuple[int, int, int, int],
    ) -> tuple[str, list, np.ndarray, str | None]:
        try:
            pieces, _mask = detect_piece_observations(
                frame, background, config, roi=source_roi
            )
            return "detect", pieces, draw_piece_observations(frame, pieces), None
        except (RuntimeError, ValueError, cv2.error) as exc:
            return "detect", [], frame, str(exc)

    @staticmethod
    def _solve_for_control(
        frame: np.ndarray,
        pieces: list,
        roi: tuple[int, int, int, int],
        calibration: PixelToGantryCalibration,
        calibration_name: str,
        config: AssemblyConfig,
    ) -> tuple[str, dict, np.ndarray]:
        assembly = solve_assembly(
            [np.asarray(piece.polygon, dtype=np.float64) for piece in pieces],
            roi,
            config,
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
            config=config,
        )
        preview = draw_assembly_preview(
            draw_piece_observations(frame, pieces), pieces, assembly, config
        )
        return "solve", document, preview

    def _queue_plan_detection(self) -> None:
        if not self.planning_active or self.planning_future is not None:
            return
        if self.current_camera_frame is None or self.planning_roi is None:
            return
        now = time.monotonic()
        if now < self.planning_next_time:
            return
        self.planning_next_time = now + 0.12
        x, y, width, height = self.planning_roi
        source_roi = (x, y, width, max(1, round(height * self.assembly_config.split_fraction)))
        self.planning_future = self.planning_executor.submit(
            self._detect_for_control,
            self.current_camera_frame.copy(),
            None if self.planning_background is None else self.planning_background.copy(),
            self.planning_config,
            source_roi,
        )
        self.planning_future_generation = self.planning_generation

    def _collect_plan_result(self) -> None:
        if self.planning_future is None or not self.planning_future.done():
            return
        future = self.planning_future
        future_generation = self.planning_future_generation
        self.planning_future = None
        self.planning_future_generation = None
        if not self.planning_active or future_generation != self.planning_generation:
            return
        try:
            result = future.result()
        except Exception as exc:
            self._finish_plan_failure(exc)
            return

        if result[0] == "detect":
            _kind, pieces, overlay, error = result
            self.preview = overlay
            self._draw_image()
            expected_count = self.planning_expected_piece_count
            if expected_count not in SUPPORTED_PIECE_COUNTS:
                self._finish_plan_failure(ValueError("本轮碎片数量状态无效，请重新开始计算。"))
                return
            if error is not None:
                self.planning_tracker.reset("识别异常")
                self.plan_state.set(f"方案：识别提示：{error}")
                return
            if len(pieces) != expected_count:
                self.planning_tracker.reset(
                    f"当前识别到 {len(pieces)} 块，等待 {expected_count} 块"
                )
                self.plan_state.set(
                    f"方案：当前识别到 {len(pieces)} 块，本轮应为 {expected_count} 块"
                )
                return
            stability = self.planning_tracker.update(pieces)
            self.plan_state.set(f"方案：{stability.reason}")
            if not stability.stable:
                return
            averaged = self.planning_tracker.averaged_observations()
            self.plan_state.set(f"方案：{expected_count} 块已稳定，正在计算拼接方案…")
            self.status.set("识别已经稳定，正在计算轮廓匹配、目标位置和抓取/放置角度。")
            self.planning_future = self.planning_executor.submit(
                self._solve_for_control,
                self.current_camera_frame.copy(),
                averaged,
                self.planning_roi,
                self.planning_calibration,
                self.planning_calibration_name,
                self.assembly_config,
            )
            self.planning_future_generation = self.planning_generation
            return

        _kind, document, preview = result
        try:
            build_execution_tasks(
                document,
                servo_home_angle=SERVO_HOME_ANGLE,
                servo_direction=self.servo_direction.get(),
            )
        except (KeyError, TypeError, ValueError) as exc:
            diagnostic_note = ""
            try:
                FAILED_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
                FAILED_PLAN_PATH.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if cv2.imwrite(str(FAILED_PREVIEW_PATH), preview):
                    diagnostic_note = f"\n失败方案已保存：{FAILED_PLAN_PATH}"
                else:
                    diagnostic_note = f"\n失败数据已保存：{FAILED_PLAN_PATH}"
            except (OSError, TypeError, ValueError, cv2.error) as save_exc:
                self._append_log(f"PLAN DIAGNOSTIC SAVE ERROR: {save_exc}")
            self._finish_plan_failure(ValueError(f"{exc}{diagnostic_note}"))
            return
        try:
            PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
            PLAN_PATH.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not cv2.imwrite(str(PREVIEW_PATH), preview):
                raise OSError(f"无法保存拼接预览：{PREVIEW_PATH}")
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            self._finish_plan_failure(exc)
            return
        completed_count = self.planning_expected_piece_count or len(document.get("pieces", []))
        self.planning_active = False
        self.planning_expected_piece_count = None
        self._set_piece_count_controls_enabled(True)
        self.calculate_plan_button.configure(text="一键重新识别并计算拼接方案")
        self._append_log("PLAN COMPLETE: saved and loaded")
        self._load_plan(show_errors=True)
        messagebox.showinfo(
            "拼接方案完成",
            f"已重新识别 {completed_count} 块碎片、计算方案并自动加载。",
        )

    def _finish_plan_failure(self, exc: Exception) -> None:
        self.planning_active = False
        self.planning_expected_piece_count = None
        self._set_piece_count_controls_enabled(True)
        self.calculate_plan_button.configure(text="一键重新识别并计算拼接方案")
        self.plan_state.set("方案：计算失败，请检查识别轮廓、ROI 和标定")
        self.status.set("拼接方案计算失败；没有发送任何电机命令。")
        self._append_log(f"PLAN ERROR: {exc}")
        messagebox.showerror("拼接计算失败", str(exc))

    def _load_plan(self, show_errors: bool = True) -> None:
        if self.planning_active:
            if show_errors:
                messagebox.showwarning("正在计算方案", "请先等待计算完成或停止本次识别计算。")
            return
        try:
            document = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
            quality = document.get("quality", {})
            if quality.get("geometry_verified") is not True:
                raise ValueError("拼接方案未通过新版矩形几何校验，请回到识别界面重新计算。")
            self.tasks = build_execution_tasks(
                document, servo_home_angle=SERVO_HOME_ANGLE,
                servo_direction=self.servo_direction.get(),
            )
            preview = cv2.imread(str(PREVIEW_PATH), cv2.IMREAD_COLOR)
            if preview is None:
                raise ValueError(f"无法读取拼接预览：{PREVIEW_PATH}")
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self.tasks = []
            self.preview = None
            self.plan_state.set("方案：未找到可执行方案")
            self.task_state.set("任务：请先在识别界面计算拼接方案")
            self._refresh_task_list()
            if show_errors:
                messagebox.showerror("加载方案失败", str(exc))
            return
        self.preview = preview
        self.current_task_index = 0
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self.auto_run_enabled = False
        self._cancel_auto_continue()
        gap = float(document.get("quality", {}).get("placement_gap_actual_mm", 0.0))
        self.plan_state.set(
            f"方案：已加载 {len(self.tasks)} 块，安全缝 {gap:.1f} mm，"
            "坐标和舵机角度校验通过"
        )
        self._update_task_state()
        self._refresh_task_list()
        self._draw_image()
        self.status.set("方案已加载。每次只发送一块，完成回零后由人工确认推进。")

    def _reload_tasks(self) -> None:
        if self.waiting_for_completion:
            messagebox.showwarning("动作进行中", "请等待当前动作完成后再切换舵机方向。")
            return
        self._load_plan(show_errors=False)

    def _send_next(self) -> None:
        if self.planning_active:
            messagebox.showwarning("正在计算方案", "请等待方案计算完成或先停止本次识别计算。")
            return
        if self.auto_run_enabled:
            messagebox.showwarning("自动执行中", "请先停止自动执行，再使用单块发送。")
            return
        if self.waiting_for_completion:
            messagebox.showwarning(
                "当前块尚未确认完成",
                "请等待下位机执行完成并回到原点，再点击“确认当前块已完成并回零”。",
            )
            return
        if self.current_task_index >= len(self.tasks):
            messagebox.showinfo("任务完成", "当前拼接方案已经全部执行。")
            return
        if not self.serial.connected:
            messagebox.showwarning("串口未连接", "请先连接 CH340 对应的 COM 端口。")
            return
        task = self.tasks[self.current_task_index]
        if not messagebox.askyesno(
            "确认真实动作",
            f"将执行 P{task.piece_id}：\n"
            f"取 ({task.source_x}, {task.source_y})\n"
            f"放 ({task.target_x}, {task.target_y})\n"
            f"抓取角 {task.pick_angle_deg}°，放置角 {task.place_angle_deg}°"
            f"（相对旋转 {task.rotation_deg:+.1f}°）\n\n"
            "确认吸头、碎片和工作区安全后继续。",
        ):
            return
        self._transmit_task(task)

    def _start_auto_run(self) -> None:
        if self.planning_active:
            messagebox.showwarning("正在计算方案", "请等待方案计算完成或先停止本次识别计算。")
            return
        if self.waiting_for_completion:
            messagebox.showwarning("当前块正在执行", "请等待当前块完成后再启动自动执行。")
            return
        if self.current_task_index >= len(self.tasks):
            messagebox.showinfo("任务完成", "当前拼接方案已经全部执行。")
            return
        if not self.serial.connected:
            messagebox.showwarning("串口未连接", "请先连接 CH340 对应的 COM 端口。")
            return
        remaining = len(self.tasks) - self.current_task_index
        if not messagebox.askyesno(
            "确认自动连续执行",
            f"将从 P{self.tasks[self.current_task_index].piece_id} 开始，自动连续执行"
            f"剩余 {remaining} 块。\n\n"
            "每块必须收到下位机 B1（完成并回零）才会发送下一块。"
            "执行期间请保持工作区无人、碎片位置稳定。\n\n确认开始吗？",
        ):
            return
        self.auto_run_enabled = True
        self._append_log(f"AUTO START: {remaining} task(s) remaining")
        self.status.set("自动连续执行已启动：等待每块 B1 后自动发送下一块。")
        self._update_task_state()
        self._transmit_task(self.tasks[self.current_task_index])

    def _stop_auto_run(self) -> None:
        if not self.auto_run_enabled:
            self.status.set("当前未启用自动连续执行。")
            return
        self.auto_run_enabled = False
        self._cancel_auto_continue()
        self._append_log("AUTO STOP: no further task will be sent")
        self._update_task_state()
        self._refresh_task_list()
        if self.waiting_for_completion:
            self.status.set("自动执行已停止：当前块仍会完成，B1 后不会发送下一块。")
        else:
            self.status.set("自动执行已停止：后续任务需要手动发送。")

    def _transmit_task(self, task) -> None:
        try:
            frame = build_dual_angle_pick_and_place_frame(
                task.source_x, task.source_y, task.target_x, task.target_y,
                pick_angle_deg=task.pick_angle_deg,
                place_angle_deg=task.place_angle_deg,
            )
            self.serial.discard_input()
            self.status_parser.reset()
            self.serial.send(frame)
        except Exception as exc:
            messagebox.showerror("发送失败", str(exc))
            return
        self.waiting_for_completion = True
        self.waiting_for_accept = True
        self.accept_rx_start = self.rx_bytes
        self.tx_bytes += len(frame)
        self._update_serial_state()
        self.controller_state.set("下位机：命令已发送，等待 B0 接收确认")
        self._append_log(f"TX P{task.piece_id}: {frame.hex(' ').upper()}")
        self._update_task_state()
        self._refresh_task_list()
        self.status.set(
            f"已发送 P{task.piece_id}。请等待机械执行并回零，再人工确认完成。")
        self._cancel_accept_timeout()
        self.accept_timeout_job = self.root.after(1500, self._check_accept_status)

    def _send_auto_next(self) -> None:
        self.auto_continue_job = None
        if not self.auto_run_enabled or self.waiting_for_completion:
            return
        if self.current_task_index >= len(self.tasks):
            return
        self._transmit_task(self.tasks[self.current_task_index])

    def _check_accept_status(self) -> None:
        self.accept_timeout_job = None
        if self.waiting_for_completion and self.waiting_for_accept:
            if self.rx_bytes == self.accept_rx_start:
                self.controller_state.set(
                    "下位机：未收到任何回传字节，请检查 S1-2(TX1)→CH340 RXD"
                )
                self._append_log("STATUS TIMEOUT: RX 0 B after command")
            else:
                self.controller_state.set(
                    "下位机：收到字节但没有有效 B0，请确认固件和波特率 115200"
                )
                self._append_log("STATUS TIMEOUT: bytes received but no valid B0 frame")

    def _cancel_accept_timeout(self) -> None:
        if self.accept_timeout_job is not None:
            self.root.after_cancel(self.accept_timeout_job)
            self.accept_timeout_job = None

    def _cancel_auto_continue(self) -> None:
        if self.auto_continue_job is not None:
            self.root.after_cancel(self.auto_continue_job)
            self.auto_continue_job = None

    def _update_serial_state(self) -> None:
        if self.serial.connected:
            self.serial_state.set(
                f"串口：{self.serial.port} @ {self.serial.baudrate}，"
                f"TX {self.tx_bytes} B / RX {self.rx_bytes} B"
            )

    def _confirm_completed_manually(self) -> None:
        if self.auto_run_enabled:
            messagebox.showwarning("自动执行中", "自动模式会在收到 B1 后自行推进任务。")
            return
        if self.current_task_index >= len(self.tasks):
            messagebox.showinfo("任务已完成", "当前方案没有尚未完成的任务。")
            return
        task = self.tasks[self.current_task_index]
        if not self.waiting_for_completion:
            messagebox.showwarning("尚未发送", "当前块还没有从本界面发送，不能确认完成。")
            return
        if not messagebox.askyesno(
            "确认机械已安全回零",
            f"仅在肉眼确认 P{task.piece_id} 已完成取放、Z 轴抬起、XY 已回零、"
            "舵机已回到 135° 时继续。\n\n确认当前块完成并切换到下一块吗？",
        ):
            return
        self._append_log(f"MANUAL COMPLETE P{task.piece_id}: operator verified homing")
        self._complete_current_task()

    def _clear_wait_without_advancing(self) -> None:
        if not self.waiting_for_completion:
            return
        if not messagebox.askyesno(
            "解除等待",
            "这不会把当前任务标记为完成，也不会发送任何命令。"
            "状态不明确时请先断开电机电源并检查机械位置。\n\n确定解除等待吗？",
        ):
            return
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self._cancel_accept_timeout()
        self.auto_run_enabled = False
        self._cancel_auto_continue()
        self._append_log("WAIT CLEARED: task not advanced, no command sent")
        self._update_task_state()
        self._refresh_task_list()
        self.status.set("等待已解除，当前任务未推进；检查设备后再决定是否重新执行。")

    def _handle_status(self, status: int) -> None:
        self._append_log(f"STATUS {status:02X}: {STATUS_NAMES.get(status, '未知状态')}")
        if status == STATUS_COMMAND_ACCEPTED:
            self.waiting_for_accept = False
            self._cancel_accept_timeout()
            self.controller_state.set("下位机：已收到 B0，正在执行")
            self.status.set("下位机已接收命令，正在执行取放。")
            return
        if status == STATUS_ACTION_FAILED and self.waiting_for_accept:
            self._append_log("IGNORED STALE B3: current command has not received B0")
            return
        if status in (STATUS_COMMAND_REJECTED, STATUS_ACTION_FAILED):
            self.waiting_for_accept = False
            self.controller_state.set(
                f"下位机：收到 {status:02X}，{STATUS_NAMES[status]}"
            )
            self.waiting_for_completion = False
            self.waiting_for_accept = False
            self._cancel_accept_timeout()
            self.auto_run_enabled = False
            self._cancel_auto_continue()
            self._update_task_state()
            self._refresh_task_list()
            title = "下位机动作失败" if status == STATUS_ACTION_FAILED else "下位机拒绝命令"
            messagebox.showerror(title, STATUS_NAMES[status])
            return
        if status == STATUS_ACTION_COMPLETE and self.waiting_for_accept:
            self._append_log("IGNORED STALE B1: current command has not received B0")
            return
        if status == STATUS_ACTION_COMPLETE and self.waiting_for_completion:
            self.waiting_for_accept = False
            self._cancel_accept_timeout()
            self.controller_state.set("下位机：已收到 B1，动作完成并回零")
            if self.auto_run_enabled:
                self._append_log("AUTO COMPLETE: B1 received, scheduling next task")
                self._complete_current_task(show_dialog=False)
                if self.current_task_index < len(self.tasks):
                    self.auto_continue_job = self.root.after(500, self._send_auto_next)
                return
            self.status.set("收到下位机完成状态；请确认机械确已回零，再人工推进下一块。")

    def _complete_current_task(self, show_dialog: bool = True) -> None:
        self.waiting_for_completion = False
        self.waiting_for_accept = False
        self._cancel_accept_timeout()
        self.current_task_index += 1
        self._update_task_state()
        self._refresh_task_list()
        if self.current_task_index >= len(self.tasks):
            was_auto_run = self.auto_run_enabled
            self.auto_run_enabled = False
            self._cancel_auto_continue()
            self.status.set("全部碎片已完成并回零。")
            if show_dialog:
                messagebox.showinfo("拼接执行完成", "当前方案的所有碎片均已确认完成。")
            elif was_auto_run:
                messagebox.showinfo("自动拼接完成", "全部碎片均已收到 B1 并完成放置。")
        else:
            if self.auto_run_enabled:
                self.status.set("当前块已收到 B1，准备自动发送下一块。")
            else:
                self.status.set("当前块已确认完成，可以发送下一块。")

    def _update_task_state(self) -> None:
        if not self.tasks:
            self.task_state.set("任务：等待加载方案")
        elif self.current_task_index >= len(self.tasks):
            self.task_state.set(f"任务：{len(self.tasks)}/{len(self.tasks)}，全部完成")
        else:
            if self.waiting_for_completion:
                state = "自动执行中" if self.auto_run_enabled else "执行中"
            else:
                state = "自动等待发送" if self.auto_run_enabled else "等待确认"
            self.task_state.set(
                f"任务：{self.current_task_index + 1}/{len(self.tasks)}，{state}"
            )

    def _refresh_task_list(self) -> None:
        self.task_list.delete(0, tk.END)
        for index, task in enumerate(self.tasks):
            if index < self.current_task_index:
                marker = "DONE"
            elif index == self.current_task_index and self.waiting_for_completion:
                marker = "RUN "
            elif index == self.current_task_index:
                marker = "NEXT"
            else:
                marker = "WAIT"
            self.task_list.insert(
                tk.END,
                f"[{marker}] P{task.piece_id} "
                f"({task.source_x},{task.source_y})->({task.target_x},{task.target_y}) "
                f"rot={task.rotation_deg:+.1f} "
                f"pick={task.pick_angle_deg} place={task.place_angle_deg}",
            )

    def _update(self) -> None:
        # Status reception is safety-critical. Process it before any camera IO,
        # and never poll the USB camera once the static assembly preview exists.
        try:
            received = self.serial.read_available()
            if received:
                self.rx_bytes += len(received)
                self._update_serial_state()
                self._append_log(f"RX {len(received)} B: {received.hex(' ').upper()}")
                for status in self.status_parser.feed(received):
                    self._handle_status(status)
        except Exception as exc:
            self.serial_state.set("串口：读取失败，请重新连接")
            self._append_log(f"SERIAL ERROR: {exc}")
            self.status.set("CH340 读取失败；当前机械动作状态未知，请不要再次发送。")
            self.auto_run_enabled = False
            self._cancel_auto_continue()
            self.serial.close()
        try:
            if self.capture is not None and (self.preview is None or self.planning_active):
                ok, frame = self.capture.read()
                if ok:
                    if self.rotate_180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.current_camera_frame = frame
                    if not self.planning_active or self.planning_future is None:
                        self.preview = frame
                        self._draw_image()
        except cv2.error as exc:
            self._append_log(f"CAMERA ERROR: {exc}")
            self.capture.release()
            self.capture = None
        self._collect_plan_result()
        self._queue_plan_detection()
        self.root.after(30, self._update)

    def _draw_image(self) -> None:
        if self.preview is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width < 2 or height < 2:
            return
        scale = min(width / self.preview.shape[1], height / self.preview.shape[0])
        shown = cv2.resize(
            self.preview,
            (max(1, round(self.preview.shape[1] * scale)),
             max(1, round(self.preview.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        ok, png = cv2.imencode(".png", shown)
        if not ok:
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.photo, anchor="center")

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, f"{message}\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _close(self) -> None:
        self._cancel_accept_timeout()
        self._cancel_auto_continue()
        self.planning_active = False
        if self.planning_future is not None:
            self.planning_future.cancel()
        self.planning_executor.shutdown(wait=False, cancel_futures=True)
        if self.capture is not None:
            self.capture.release()
        self.serial.close()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--serial", help="CH340 serial port, e.g. COM32")
    parser.add_argument("--no-rotate-180", action="store_true",
                        help="use the raw camera orientation")
    args = parser.parse_args()
    root = tk.Tk()
    PuzzleControlApp(root, args.camera, args.serial, rotate_180=not args.no_rotate_180)
    root.mainloop()


if __name__ == "__main__":
    main()
