#!/usr/bin/env python3
"""Competition control GUI for vision planning and gantry execution."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2

from puzzle_device.calibration.gantry_protocol import (
    GantryStatusParser,
    OptionalSerialPort,
    STATUS_ACTION_FAILED,
    STATUS_ACTION_COMPLETE,
    STATUS_COMMAND_ACCEPTED,
    STATUS_COMMAND_REJECTED,
    STATUS_NAMES,
    build_pick_and_place_frame,
)
from puzzle_device.planning import build_execution_tasks
from puzzle_device.vision.camera import open_uvc_camera


PLAN_PATH = Path("output/assembly_plan.json")
PREVIEW_PATH = Path("output/assembly_preview.png")
SERVO_HOME_ANGLE = 135


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
        self.preview = None
        self.photo = None

        self.serial_port = tk.StringVar(value=serial_port or "")
        self.servo_direction = tk.IntVar(value=1)
        self.camera_state = tk.StringVar(
            value="相机：未打开" if camera_info is None
            else f"相机：{camera_info.describe()} / {'旋转180°' if rotate_180 else '原始方向'}"
        )
        self.serial_state = tk.StringVar(value="串口：未连接")
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
        for variable in (self.camera_state, self.serial_state, self.plan_state, self.task_state):
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
            text="舵机角度 = 135° + 方向 × 拼接旋转角。若实机转反，切换方向后重新加载。",
            foreground="#7c3f00", wraplength=350, justify="left",
        ).pack(fill="x", pady=(6, 0))

        execute = ttk.LabelFrame(controls, text="执行控制", padding=9)
        execute.pack(fill="x", pady=(9, 0))
        self.next_button = ttk.Button(
            execute, text="发送当前块（每次仅一块）", command=self._send_next)
        self.next_button.pack(fill="x")
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
                "每次只发送当前一块。请等待下位机完成取放并回到原点，"
                "再点击“确认当前块已完成并回零”，然后才能发送下一块。"
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
            self.serial_state.set(f"串口：{self.serial.port} @ {self.serial.baudrate}，已连接")
            self._append_log(f"OPEN {self.serial.port} @ {self.serial.baudrate}")
        else:
            self.serial_state.set("串口：未填写端口，模拟模式")

    def _load_plan(self, show_errors: bool = True) -> None:
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
        self.plan_state.set(f"方案：已加载 {len(self.tasks)} 块，坐标和舵机角度校验通过")
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
            f"舵机 {task.servo_angle_deg}°（相对拼接旋转 {task.rotation_deg:+.1f}°）\n\n"
            "确认吸头、碎片和工作区安全后继续。",
        ):
            return
        self._transmit_task(task)

    def _transmit_task(self, task) -> None:
        try:
            frame = build_pick_and_place_frame(
                task.source_x, task.source_y, task.target_x, task.target_y,
                rotation_angle_deg=task.servo_angle_deg,
            )
            self.serial.send(frame)
        except Exception as exc:
            messagebox.showerror("发送失败", str(exc))
            return
        self.waiting_for_completion = True
        self._append_log(f"TX P{task.piece_id}: {frame.hex(' ').upper()}")
        self._update_task_state()
        self._refresh_task_list()
        self.status.set(
            f"已发送 P{task.piece_id}。请等待机械执行并回零，再人工确认完成。")

    def _confirm_completed_manually(self) -> None:
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
        self._append_log("WAIT CLEARED: task not advanced, no command sent")
        self._update_task_state()
        self._refresh_task_list()
        self.status.set("等待已解除，当前任务未推进；检查设备后再决定是否重新执行。")

    def _handle_status(self, status: int) -> None:
        self._append_log(f"STATUS {status:02X}: {STATUS_NAMES.get(status, '未知状态')}")
        if status == STATUS_COMMAND_ACCEPTED:
            self.status.set("下位机已接收命令，正在执行取放。")
            return
        if status in (STATUS_COMMAND_REJECTED, STATUS_ACTION_FAILED):
            self.waiting_for_completion = False
            self._update_task_state()
            self._refresh_task_list()
            title = "下位机动作失败" if status == STATUS_ACTION_FAILED else "下位机拒绝命令"
            messagebox.showerror(title, STATUS_NAMES[status])
            return
        if status == STATUS_ACTION_COMPLETE and self.waiting_for_completion:
            self.status.set("收到下位机完成状态；请确认机械确已回零，再人工推进下一块。")

    def _complete_current_task(self) -> None:
        self.waiting_for_completion = False
        self.current_task_index += 1
        self._update_task_state()
        self._refresh_task_list()
        if self.current_task_index >= len(self.tasks):
            self.status.set("全部碎片均已由人工确认完成并回零。")
            messagebox.showinfo("拼接执行完成", "当前方案的所有碎片均已确认完成。")
        else:
            self.status.set("当前块已确认完成，可以发送下一块。")

    def _update_task_state(self) -> None:
        if not self.tasks:
            self.task_state.set("任务：等待加载方案")
        elif self.current_task_index >= len(self.tasks):
            self.task_state.set(f"任务：{len(self.tasks)}/{len(self.tasks)}，全部完成")
        else:
            state = "执行中" if self.waiting_for_completion else "等待确认"
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
                f"rot={task.rotation_deg:+.1f} servo={task.servo_angle_deg}",
            )

    def _update(self) -> None:
        # Status reception is safety-critical. Process it before any camera IO,
        # and never poll the USB camera once the static assembly preview exists.
        try:
            received = self.serial.read_available()
            if received:
                self._append_log(f"RX {len(received)} B: {received.hex(' ').upper()}")
                for status in self.status_parser.feed(received):
                    self._handle_status(status)
        except Exception as exc:
            self.serial_state.set("串口：读取失败，请重新连接")
            self._append_log(f"SERIAL ERROR: {exc}")
            self.status.set("CH340 读取失败；当前机械动作状态未知，请不要再次发送。")
            self.serial.close()
        try:
            if self.capture is not None and self.preview is None:
                ok, frame = self.capture.read()
                if ok:
                    if self.rotate_180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    self.preview = frame
                    self._draw_image()
        except cv2.error as exc:
            self._append_log(f"CAMERA ERROR: {exc}")
            self.capture.release()
            self.capture = None
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
