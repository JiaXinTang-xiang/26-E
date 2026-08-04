#!/usr/bin/env python3
"""Manual camera-to-gantry calibration GUI for the puzzle device."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from puzzle_device.calibration.gantry_protocol import (
    OptionalSerialPort,
    build_pick_and_place_frame,
)
from puzzle_device.calibration.manual_calibration import (
    CalibrationPoint,
    PixelToGantryCalibration,
)
from puzzle_device.paths import LOCAL_CONFIG_DIR
from puzzle_device.vision.camera import open_uvc_camera

MAX_GANTRY_X_PULSE = 2350
MAX_GANTRY_Y_PULSE = 1350
DRAFT_PATH = LOCAL_CONFIG_DIR / "calibration_points_draft.json"
CALIBRATION_PATH = LOCAL_CONFIG_DIR / "calibration.json"
TEMPORARY_CALIBRATION_PATH = LOCAL_CONFIG_DIR / "calibration_temporary.json"


class ManualCalibrationApp:
    """Pairs OpenCV camera pixels with manually supplied gantry pulse positions."""

    def __init__(self, root: tk.Tk, camera_index: int, serial_port: str | None,
                 rotate_180: bool = True):
        self.root = root
        self.root.title("拼图装置 - 相机到龙门架手动标定")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.compact_layout = screen_width <= 1100 or screen_height <= 650
        if self.compact_layout:
            self.root.geometry(
                f"{min(1000, max(900, screen_width - 24))}x"
                f"{min(560, max(500, screen_height - 55))}+0+0"
            )
            self.root.minsize(900, 500)
        else:
            self.root.geometry("1320x830")
            self.root.minsize(1080, 680)
        self.camera_index = camera_index
        self.rotate_180 = rotate_180
        self.capture = None
        self.current_frame: np.ndarray | None = None
        self.photo = None
        self.pending_phase: str | None = None
        self.calibration = PixelToGantryCalibration()
        self.motion_calibration = PixelToGantryCalibration()
        self.locked_task: tuple[int, int, int, int] | None = None
        self.test_source_pixel: tuple[float, float] | None = None
        self.test_destination_pixel: tuple[float, float] | None = None
        self.serial = OptionalSerialPort(serial_port)

        self.source_x = tk.StringVar(value="0")
        self.source_y = tk.StringVar(value="0")
        self.destination_x = tk.StringVar(value="0")
        self.destination_y = tk.StringVar(value="0")
        self.serial_port = tk.StringVar(value=serial_port or "")
        orientation = "已旋转 180°" if rotate_180 else "原始方向"
        self.status = tk.StringVar(
            value=f"相机画面{orientation}；填写源/目标脉冲，移动吸头后点击记录源点或目标点。"
        )
        self.result = tk.StringVar(value="尚未拟合矩阵")
        self.test_result = tk.StringVar(value="先点击取点和放点。")

        self._build_ui()
        if serial_port:
            self._connect_serial()
        self._load_draft()
        self._load_motion_calibration()
        self._open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._update_frame()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 16, "bold"))
        style.configure("Step.TButton", padding=(10, 7))

        header = ttk.Frame(self.root, padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="相机-龙门架手动标定", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="点击画面中的吸头实际中心；X/Y 单位为回零后的绝对脉冲",
                  foreground="#555").pack(side="right", pady=(5, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        viewer = ttk.Frame(body)
        controls_host = ttk.Frame(body, width=300 if self.compact_layout else 410)
        body.add(viewer, weight=4)
        body.add(controls_host, weight=2)

        self.canvas = tk.Canvas(viewer, background="#202326", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_image_click)
        self.canvas.bind("<Configure>", lambda _event: self._show_frame())

        self.controls_canvas = tk.Canvas(
            controls_host, highlightthickness=0, width=390)
        controls_scrollbar = ttk.Scrollbar(
            controls_host, orient="vertical", command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_scrollbar.pack(side="right", fill="y")
        self.controls_canvas.pack(side="left", fill="both", expand=True)

        controls = ttk.Frame(self.controls_canvas, padding=(0, 0, 6, 8))
        controls_window = self.controls_canvas.create_window(
            (0, 0), window=controls, anchor="nw")
        controls.bind(
            "<Configure>",
            lambda _event: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")),
        )
        self.controls_canvas.bind(
            "<Configure>",
            lambda event: self.controls_canvas.itemconfigure(
                controls_window, width=event.width),
        )
        self.root.bind_all("<MouseWheel>", self._on_controls_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_controls_mousewheel, add="+")
        self.root.bind_all("<Button-5>", self._on_controls_mousewheel, add="+")

        movement = ttk.LabelFrame(controls, text="一条取放标定任务", padding=12)
        movement.pack(fill="x")
        self._coordinate_row(movement, "源取棋脉冲 X", self.source_x, 0)
        self._coordinate_row(movement, "源取棋脉冲 Y", self.source_y, 1)
        self._coordinate_row(movement, "目标放棋脉冲 X", self.destination_x, 2)
        self._coordinate_row(movement, "目标放棋脉冲 Y", self.destination_y, 3)

        ttk.Button(movement, text="1. 锁定本次两组脉冲", style="Step.TButton",
                   command=self._lock_task).grid(
                       row=4, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Button(movement, text="2. 准备记录源点（再点击画面）", style="Step.TButton",
                   command=lambda: self._arm_click("source")).grid(
                       row=5, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(movement, text="3. 准备记录目标点（再点击画面）", style="Step.TButton",
                   command=lambda: self._arm_click("destination")).grid(
                       row=6, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(movement, text="发送旧取放帧（可选）", command=self._send_pick_and_place).grid(
            row=7, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(movement, text="取消等待点击", command=self._cancel_pending_click).grid(
            row=7, column=1, sticky="ew", padx=(8, 0), pady=(7, 0))
        movement.columnconfigure(1, weight=1)

        test = ttk.LabelFrame(controls, text="矩阵取放测试（真实动作）", padding=10)
        test.pack(fill="x", pady=(9, 0))
        ttk.Button(test, text="1. 点击画面选择取棋点", command=lambda: self._arm_test_click("source")).grid(
            row=0, column=0, sticky="ew")
        ttk.Button(test, text="2. 点击画面选择放棋点", command=lambda: self._arm_test_click("destination")).grid(
            row=0, column=1, sticky="ew", padx=(7, 0))
        ttk.Label(test, textvariable=self.test_result, foreground="#155e75",
                  wraplength=265 if self.compact_layout else 350,
                  justify="left").grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 4))
        ttk.Button(test, text="确定：执行取棋、放棋、回零", style="Step.TButton",
                   command=self._confirm_test_move).grid(
                       row=2, column=0, columnspan=2, sticky="ew")
        test.columnconfigure(0, weight=1)
        test.columnconfigure(1, weight=1)

        serial_frame = ttk.LabelFrame(controls, text="串口（可选）", padding=10)
        serial_frame.pack(fill="x", pady=(9, 0))
        ttk.Label(serial_frame, text="端口").grid(row=0, column=0, sticky="w")
        ttk.Entry(serial_frame, textvariable=self.serial_port, width=24).grid(
            row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Button(serial_frame, text="连接", command=self._connect_serial).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        serial_frame.columnconfigure(1, weight=1)

        serial_log = ttk.LabelFrame(controls, text="串口通信记录", padding=8)
        serial_log.pack(fill="both", pady=(9, 0))
        self.serial_log = tk.Text(serial_log, height=6, wrap="word", state="disabled",
                                  font=("Consolas", 9))
        self.serial_log.pack(fill="both", expand=True)

        points = ttk.LabelFrame(controls, text="已记录点", padding=10)
        points.pack(fill="both", expand=True, pady=(9, 0))
        self.point_list = tk.Listbox(points, height=10, font=("Consolas", 9))
        self.point_list.pack(fill="both", expand=True)
        buttons = ttk.Frame(points)
        buttons.pack(fill="x", pady=(7, 0))
        ttk.Button(buttons, text="删除选中点", command=self._delete_selected).pack(side="left")
        ttk.Button(buttons, text="拟合稳定平均矩阵", command=self._fit).pack(side="right")
        ttk.Label(points, textvariable=self.result, foreground="#155e75",
                  wraplength=265 if self.compact_layout else 350,
                  justify="left").pack(fill="x", pady=(7, 0))
        ttk.Button(points, text="保存到 configs/local/calibration.json",
                   command=self._save).pack(fill="x", pady=(7, 0))

        footer = ttk.Frame(self.root, padding=(14, 6))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, foreground="#174c75").pack(side="left")

    def _on_controls_mousewheel(self, event: tk.Event) -> str | None:
        """Scroll the controls only while the pointer is over the right panel."""
        canvas = self.controls_canvas
        pointer_x = canvas.winfo_pointerx()
        pointer_y = canvas.winfo_pointery()
        inside = (
            canvas.winfo_rootx() <= pointer_x < canvas.winfo_rootx() + canvas.winfo_width()
            and canvas.winfo_rooty() <= pointer_y < canvas.winfo_rooty() + canvas.winfo_height()
        )
        step = self._mousewheel_step(event)
        if not inside or step == 0:
            return None
        canvas.yview_scroll(step, "units")
        return "break"

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

    @staticmethod
    def _coordinate_row(parent: ttk.LabelFrame, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=14).grid(
            row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

    def _open_camera(self) -> None:
        self.capture, camera_info = open_uvc_camera(self.camera_index)
        if self.capture is None:
            self.status.set(f"无法打开摄像头 {self.camera_index}；仍可连接后重启程序。")
            return
        details = "" if camera_info is None else camera_info.describe()
        self.status.set(
            f"相机已打开：{details}。每次记录前确认吸头已停稳且处于实际抓取高度。"
        )

    def _update_frame(self) -> None:
        if self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                if self.rotate_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                self.current_frame = frame
                self._show_frame()
        received = self.serial.read_available()
        if received:
            self._append_serial_log(f"RX {len(received)} B: {received.hex(' ').upper()}")
        self.root.after(30, self._update_frame)

    def _show_frame(self) -> None:
        if self.current_frame is None or self.canvas.winfo_width() < 2:
            return
        frame = self._draw_overlay(self.current_frame)
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(canvas_w / frame.shape[1], canvas_h / frame.shape[0])
        shown_w = max(1, int(frame.shape[1] * scale))
        shown_h = max(1, int(frame.shape[0] * scale))
        resized = cv2.resize(frame, (shown_w, shown_h), interpolation=cv2.INTER_AREA)
        self._display_scale = scale
        self._display_origin = ((canvas_w - shown_w) // 2, (canvas_h - shown_h) // 2)
        ok, png = cv2.imencode(".png", resized)
        if not ok:
            return
        self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.canvas.delete("all")
        self.canvas.create_image(*self._display_origin, image=self.photo, anchor="nw")

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        for index, point in enumerate(self.calibration.points):
            color = (0, 220, 255) if point.phase == "source" else (80, 220, 80)
            center = (round(point.pixel_x), round(point.pixel_y))
            cv2.drawMarker(output, center, color, cv2.MARKER_CROSS, 16, 2)
            cv2.putText(output, str(index + 1), (center[0] + 7, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        for label, point, color in (
            ("PICK", self.test_source_pixel, (255, 180, 0)),
            ("PLACE", self.test_destination_pixel, (255, 80, 255)),
        ):
            if point is not None:
                center = (round(point[0]), round(point[1]))
                cv2.drawMarker(output, center, color, cv2.MARKER_TILTED_CROSS, 22, 2)
                cv2.putText(output, label, (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        if self.pending_phase:
            cv2.putText(output, f"CLICK {self.pending_phase.upper()} TOOL CENTER", (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
        return output

    def _parse_task_coordinates(self) -> tuple[int, int, int, int]:
        try:
            values = tuple(int(value) for value in (
                self.source_x.get(), self.source_y.get(),
                self.destination_x.get(), self.destination_y.get(),
            ))
        except ValueError as exc:
            raise ValueError("源点和目标点 X/Y 必须是整数脉冲") from exc
        if not (0 <= values[0] <= MAX_GANTRY_X_PULSE and
                0 <= values[2] <= MAX_GANTRY_X_PULSE):
            raise ValueError(f"X 脉冲必须在 0 到 {MAX_GANTRY_X_PULSE} 之间")
        if not (0 <= values[1] <= MAX_GANTRY_Y_PULSE and
                0 <= values[3] <= MAX_GANTRY_Y_PULSE):
            raise ValueError(f"Y 脉冲必须在 0 到 {MAX_GANTRY_Y_PULSE} 之间")
        return values

    def _lock_task(self) -> bool:
        """Freeze the four pulses so later text edits cannot corrupt a click."""
        try:
            self.locked_task = self._parse_task_coordinates()
        except ValueError as exc:
            messagebox.showerror("坐标无效", str(exc))
            return False
        source_x, source_y, destination_x, destination_y = self.locked_task
        self.status.set(
            "本次任务已锁定："
            f"源 ({source_x}, {source_y})，目标 ({destination_x}, {destination_y})。"
        )
        return True

    def _send_pick_and_place(self) -> None:
        if not self.serial.connected:
            messagebox.showwarning("串口未连接", "请先连接 CH340 对应的 COM 端口。")
            return
        try:
            if self.locked_task is None and not self._lock_task():
                return
            source_x, source_y, destination_x, destination_y = self.locked_task
            frame = build_pick_and_place_frame(source_x, source_y, destination_x, destination_y)
            self.serial.send(frame)
            self._append_serial_log(f"TX {len(frame)} B: {frame.hex(' ').upper()}")
            self.status.set(f"已发送取放帧：{frame.hex(' ').upper()}。请等待控制器执行。")
        except Exception as exc:
            messagebox.showerror("发送失败", str(exc))

    def _connect_serial(self) -> None:
        try:
            self.serial.close()
            self.serial.port = self.serial_port.get().strip() or None
            self.serial.connect()
            if self.serial.connected:
                self._append_serial_log(f"已打开 {self.serial.port} @ {self.serial.baudrate}")
            else:
                self._append_serial_log("未填写串口，当前为模拟发送模式")
            self.status.set(f"串口已连接：{self.serial.port} @ {self.serial.baudrate}")
        except Exception as exc:
            messagebox.showerror("串口连接失败", str(exc))

    def _append_serial_log(self, message: str) -> None:
        self.serial_log.configure(state="normal")
        self.serial_log.insert(tk.END, f"{message}\n")
        self.serial_log.see(tk.END)
        self.serial_log.configure(state="disabled")

    def _arm_click(self, phase: str) -> None:
        if self.current_frame is None:
            messagebox.showwarning("没有相机画面", "请检查摄像头后再记录标定点。")
            return
        if self.locked_task is None and not self._lock_task():
            return
        self.pending_phase = phase
        label = "源点" if phase == "source" else "目标点"
        self.status.set(f"现在左键点击画面中的吸头中心，记录{label}。点击前确保吸头完全停稳。")

    def _cancel_pending_click(self) -> None:
        self.pending_phase = None
        self.status.set("已取消等待点击；已记录的标定点仍保留在列表和草稿文件中。")

    def _arm_test_click(self, phase: str) -> None:
        if self.current_frame is None:
            messagebox.showwarning("没有相机画面", "请检查摄像头后再选择测试点。")
            return
        if self.motion_calibration.matrix is None:
            messagebox.showwarning("没有矩阵", "请先拟合并保存矩阵，或确认临时矩阵文件存在。")
            return
        self.pending_phase = f"test_{phase}"
        label = "取棋点" if phase == "source" else "放棋点"
        self.status.set(f"现在左键点击画面中的{label}。不会立即移动，两个点选好后再确认。")

    def _update_test_result(self) -> None:
        if self.test_source_pixel is None or self.test_destination_pixel is None:
            return
        try:
            source = self._pixel_to_safe_pulses(self.test_source_pixel)
            destination = self._pixel_to_safe_pulses(self.test_destination_pixel)
        except ValueError as exc:
            self.test_result.set(f"点位超出龙门架范围：{exc}")
            return
        self.test_result.set(
            f"取点像素 {self.test_source_pixel[0]:.0f},{self.test_source_pixel[1]:.0f} -> 脉冲 {source[0]},{source[1]}；"
            f"放点像素 {self.test_destination_pixel[0]:.0f},{self.test_destination_pixel[1]:.0f} -> 脉冲 {destination[0]},{destination[1]}"
        )

    def _pixel_to_safe_pulses(self, pixel: tuple[float, float]) -> tuple[int, int]:
        if self.motion_calibration.matrix is None:
            raise ValueError("尚未加载标定矩阵")
        pulse_x, pulse_y = self.motion_calibration.predict_pulse(*pixel)
        result = round(pulse_x), round(pulse_y)
        if not (0 <= result[0] <= MAX_GANTRY_X_PULSE):
            raise ValueError(f"X={result[0]}，允许 0 到 {MAX_GANTRY_X_PULSE}")
        if not (0 <= result[1] <= MAX_GANTRY_Y_PULSE):
            raise ValueError(f"Y={result[1]}，允许 0 到 {MAX_GANTRY_Y_PULSE}")
        return result

    def _confirm_test_move(self) -> None:
        if self.test_source_pixel is None or self.test_destination_pixel is None:
            messagebox.showwarning("尚未选完", "请先分别点击取棋点和放棋点。")
            return
        if not self.serial.connected:
            messagebox.showwarning("串口未连接", "请先连接 CH340 对应的 COM 端口，再执行真实取放测试。")
            return
        try:
            source = self._pixel_to_safe_pulses(self.test_source_pixel)
            destination = self._pixel_to_safe_pulses(self.test_destination_pixel)
        except ValueError as exc:
            messagebox.showerror("测试点无效", str(exc))
            return
        if not messagebox.askyesno(
            "确认执行真实取放",
            f"吸头将移动到取点脉冲 ({source[0]}, {source[1]})，再移动到放点脉冲 "
            f"({destination[0]}, {destination[1]})，最后按现有下位机逻辑回零。\n\n确认执行吗？",
        ):
            return
        try:
            frame = build_pick_and_place_frame(*source, *destination)
            self.serial.send(frame)
        except Exception as exc:
            messagebox.showerror("发送失败", str(exc))
            return
        self._append_serial_log(f"TEST TX {len(frame)} B: {frame.hex(' ').upper()}")
        self.status.set(
            f"已发送真实取放测试：取 ({source[0]}, {source[1]}) -> 放 ({destination[0]}, {destination[1]})。"
        )

    def _on_image_click(self, event: tk.Event) -> None:
        if self.current_frame is None or self.pending_phase is None:
            return
        origin_x, origin_y = self._display_origin
        x = (event.x - origin_x) / self._display_scale
        y = (event.y - origin_y) / self._display_scale
        if not (0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]):
            return
        if self.pending_phase in ("test_source", "test_destination"):
            if self.pending_phase == "test_source":
                self.test_source_pixel = (x, y)
                label = "取棋点"
            else:
                self.test_destination_pixel = (x, y)
                label = "放棋点"
            self.pending_phase = None
            self._update_test_result()
            self.status.set(f"已选择测试{label}：像素 ({x:.1f}, {y:.1f})。请选择另一点或点击确定执行。")
            return
        if self.locked_task is None:
            self.status.set("本次任务未锁定，请先锁定两组脉冲后重新点击。")
            self.pending_phase = None
            return
        source_x, source_y, destination_x, destination_y = self.locked_task
        if self.pending_phase == "source":
            pulse_x, pulse_y = source_x, source_y
            label = "源点"
        else:
            pulse_x, pulse_y = destination_x, destination_y
            label = "目标点"
        self.calibration.add_point(CalibrationPoint(x, y, pulse_x, pulse_y, self.pending_phase))
        self.pending_phase = None
        self._refresh_points()
        self._save_draft()
        next_hint = "；现在可等待并点击目标点" if label == "源点" else ""
        self.status.set(
            f"已记录{label}：像素 ({x:.1f}, {y:.1f}) -> 脉冲 ({pulse_x}, {pulse_y})。"
            f"草稿已保存{next_hint}"
        )

    def _refresh_points(self) -> None:
        self.point_list.delete(0, tk.END)
        errors = self.calibration.point_errors() if self.calibration.matrix is not None else None
        for index, point in enumerate(self.calibration.points):
            error = "" if errors is None else f" err={errors[index]:.1f}"
            self.point_list.insert(
                tk.END,
                f"{index + 1:02d} {point.phase[:3].upper()} px=({point.pixel_x:6.1f},{point.pixel_y:6.1f}) "
                f"pulse=({point.pulse_x:5.0f},{point.pulse_y:5.0f}){error}",
            )

    def _delete_selected(self) -> None:
        selected = self.point_list.curselection()
        if not selected:
            return
        self.calibration.remove_point(selected[0])
        self._refresh_points()
        self._save_draft()
        self.result.set("标定点已删除，需要重新拟合矩阵。")

    def _fit(self) -> None:
        try:
            metrics = self.calibration.fit_affine_average()
        except Exception as exc:
            messagebox.showerror("拟合失败", str(exc))
            return
        self._refresh_points()
        self.result.set(
            f"稳定平均拟合：使用 {metrics.inlier_count}/{metrics.point_count} 个点；"
            f"平均误差 {metrics.mean_error_pulse:.2f} pulse；"
            f"中位误差 {metrics.median_error_pulse:.2f} pulse；"
            f"最大误差 {metrics.max_error_pulse:.2f} pulse"
        )
        self.status.set("稳定平均矩阵已拟合，不会在标定区域边缘产生透视发散。")

    def _save(self) -> None:
        try:
            target = CALIBRATION_PATH
            self.calibration.save(target, {
                "camera_index": self.camera_index,
                "camera_rotation_degrees": 180 if self.rotate_180 else 0,
                "serial_port": self.serial.port,
                "notes": "manual clicks at suction-head centre",
                "fit_model": "affine_least_squares_average",
            })
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self.status.set(f"已保存标定矩阵：{target.resolve()}")

    def _save_draft(self) -> None:
        """Persist raw clicks too, because matrix fitting happens only after enough points."""
        DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "format": "puzzle-device.calibration-click-draft.v1",
            "camera_rotation_degrees": 180 if self.rotate_180 else 0,
            "points": [
                {
                    "pixel_x": point.pixel_x, "pixel_y": point.pixel_y,
                    "pulse_x": point.pulse_x, "pulse_y": point.pulse_y,
                    "phase": point.phase,
                }
                for point in self.calibration.points
            ],
        }
        DRAFT_PATH.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_draft(self) -> None:
        if not DRAFT_PATH.exists():
            return
        try:
            document = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
            points = [CalibrationPoint(**item) for item in document["points"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.status.set(f"无法读取标定草稿：{DRAFT_PATH}；本次将从空点集开始。")
            return
        self.calibration = PixelToGantryCalibration(points)
        self._refresh_points()
        self.status.set(f"已恢复 {len(points)} 个未拟合/已记录的标定点：{DRAFT_PATH}")

    def _load_motion_calibration(self) -> None:
        """Load the saved mapping used by the deliberate pick-and-place test."""
        for path in (CALIBRATION_PATH, TEMPORARY_CALIBRATION_PATH):
            if not path.exists():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                matrix = np.asarray(document["matrix_pixel_to_pulse"], dtype=np.float64)
                if matrix.shape != (3, 3):
                    raise ValueError("matrix must be 3x3")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self.status.set(f"无法读取测试矩阵 {path}：{exc}")
                continue
            self.motion_calibration.matrix = matrix
            mode = "正式" if path == CALIBRATION_PATH else "临时"
            self.test_result.set(f"已加载{mode}矩阵：{path.name}。先点击取点和放点。")
            return
        self.test_result.set("未找到矩阵；请先拟合并保存。")

    def _close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.serial.close()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--serial", help="optional serial port (COM5 on Windows, /dev/ttyUSB0 on Linux)")
    parser.add_argument(
        "--no-rotate-180", action="store_true",
        help="use the camera's raw orientation instead of the default 180-degree rotation",
    )
    args = parser.parse_args()
    root = tk.Tk()
    ManualCalibrationApp(root, args.camera, args.serial, rotate_180=not args.no_rotate_180)
    root.mainloop()


if __name__ == "__main__":
    main()
