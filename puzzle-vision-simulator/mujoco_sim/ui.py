#!/usr/bin/env python3
"""Integrated Tk control panel for the PIPER-L MuJoCo puzzle simulation."""

from __future__ import annotations

import math
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import mujoco
import numpy as np
import cv2

from mujoco_sim.robot_controller import ArmController
from mujoco_sim.run_sim import render_camera, settle
from mujoco_sim.scene_builder import build_scene_xml
from mujoco_sim.vision_planner import plan_from_camera_rgb


class SimulationWorker:
    def __init__(self, events: queue.Queue, speed_getter):
        self.events = events
        self.speed_getter = speed_getter

    def _emit_frames(self, model, data, scene_renderer, camera_renderer):
        scene_renderer.update_scene(data, camera="scene_view")
        scene = scene_renderer.render().copy()
        camera = render_camera(model, data, camera_renderer)
        self.events.put(("frames", scene, camera))

    def run(self, seed: int, piece_count: int):
        scene_renderer = camera_renderer = None
        try:
            self.events.put(("status", "正在生成独立随机碎片场景…"))
            model = mujoco.MjModel.from_xml_string(
                build_scene_xml(seed, piece_count))
            data = mujoco.MjData(model)
            controller = ArmController(model, data, piece_count)
            settle(model, data, 80)
            scene_renderer = mujoco.Renderer(model, height=540, width=720)
            # Keep the calibrated 900x1200 vision resolution; the Tk inset is
            # resized only after planning, never before detection.
            camera_renderer = mujoco.Renderer(model, height=1200, width=900)
            self._emit_frames(
                model, data, scene_renderer, camera_renderer)

            self.events.put(("status", "纯视觉识别：只输入摄像头 RGB 图像…"))
            camera_rgb = render_camera(model, data, camera_renderer)
            plans, _, _, matches, _ = plan_from_camera_rgb(camera_rgb)
            if len(plans) != piece_count:
                raise RuntimeError(
                    f"视觉识别数量 {len(plans)} 与设置 {piece_count} 不一致")
            self.events.put((
                "log",
                f"视觉识别成功：{len(plans)} 块，内部切割边匹配 {len(matches)} 对\n"))

            frame_counter = 0

            def advance(target, yaw, label, tolerance=.007, max_steps=1200):
                nonlocal frame_counter
                stable = 0
                self.events.put(("status", label))
                for _ in range(max_steps):
                    error = controller.step_toward(target, yaw)
                    controller.physics_step()
                    frame_counter += 1
                    if frame_counter % 10 == 0:
                        self._emit_frames(
                            model, data, scene_renderer, camera_renderer)
                        time.sleep(.004 / max(.1, self.speed_getter()))
                    if error < tolerance and controller.orientation_error < .035:
                        stable += 1
                        if stable > 12:
                            return
                    else:
                        stable = 0
                raise RuntimeError(f"机械臂未能到达目标：{label}")

            moved = set()
            colors = np.array([
                [.92, .28, .16, 1], [.20, .72, .24, 1],
                [.18, .42, .86, 1], [.72, .18, .68, 1],
            ])
            for order, plan in enumerate(plans, 1):
                prefix = f"[{order}/{len(plans)}] 碎片 P{plan.visual_id}"
                self.events.put((
                    "log",
                    f"{prefix}: 旋转 {math.degrees(plan.target_yaw):.2f}°，"
                    f"目标 {np.round(plan.target_xy * 100, 2)} cm\n"))
                above_pick = np.r_[plan.pickup_xy, .085]
                pick = np.r_[plan.pickup_xy, .023]
                above_target = np.r_[plan.target_xy, .085]
                place = np.r_[plan.target_xy, .024]
                advance(above_pick, 0, prefix + "：移动到碎片上方")
                advance(pick, 0, prefix + "：下降并开启电磁铁")
                body_id = controller.attach_nearest()
                if body_id is None or body_id in moved:
                    raise RuntimeError(prefix + " 吸附失败")
                moved.add(body_id)
                initial_rot = data.xmat[body_id].reshape(3, 3)
                initial_yaw = math.atan2(
                    initial_rot[1, 0], initial_rot[0, 0])
                model.geom_rgba[
                    model.geom(f"piece_{controller.attached_index}_geom").id
                ] = colors[plan.visual_id]
                advance(above_pick, plan.target_yaw, prefix + "：抬升并旋转")
                advance(above_target, plan.target_yaw, prefix + "：平移到目标")
                advance(place, plan.target_yaw, prefix + "：下降放置")
                controller.release(
                    plan.target_xy, initial_yaw + plan.target_yaw)
                for _ in range(20):
                    controller.physics_step()
                advance(above_target, 0, prefix + "：释放并撤离")

            controller.command_home()
            for _ in range(80):
                controller.physics_step()
            self._emit_frames(model, data, scene_renderer, camera_renderer)
            self.events.put(("status", "拼接完成：目标矩形 10 cm × 6 cm"))
            self.events.put(("done",))
        except Exception as exc:  # keep Tk main loop alive
            self.events.put(("error", str(exc)))
        finally:
            if scene_renderer is not None:
                scene_renderer.close()
            if camera_renderer is not None:
                camera_renderer.close()


class PuzzleMuJoCoUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E题拼图装置 · PIPER-L MuJoCo 全流程仿真")
        self.geometry("1500x900")
        self.minsize(1150, 700)
        self.events = queue.Queue()
        self.running = False
        self.speed_value = 1.0
        self.scene_photo = self.camera_photo = None
        self._build()
        self.after(30, self._poll)

    def _build(self):
        header = ttk.Frame(self, padding=(12, 8))
        header.pack(fill="x")
        ttk.Label(
            header, text="E题拼图装置 · PIPER-L 机械臂仿真",
            font=("", 20, "bold")).pack(side="left")
        ttk.Label(
            header, text="目标矩形 10 cm × 6 cm / 纯视觉规划",
            font=("", 12)).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        left = ttk.Frame(body)
        right = ttk.Frame(body, padding=(10, 0))
        body.add(left, weight=3)
        body.add(right, weight=2)

        self.scene_label = ttk.Label(
            left, text="点击“开始拼接”生成 MuJoCo 场景", anchor="center")
        self.scene_label.pack(fill="both", expand=True)
        inset = ttk.LabelFrame(left, text="仿真俯视摄像头 / 算法实际输入")
        inset.place(relx=.70, rely=.02, relwidth=.29, relheight=.38)
        self.camera_label = ttk.Label(inset, anchor="center")
        self.camera_label.pack(fill="both", expand=True)

        controls = ttk.LabelFrame(right, text="测试控制", padding=10)
        controls.pack(fill="x")
        ttk.Label(controls, text="随机种子").grid(
            row=0, column=0, sticky="w", pady=4)
        self.seed = tk.IntVar(value=14)
        ttk.Spinbox(
            controls, from_=0, to=999999, textvariable=self.seed,
            width=12).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="碎片数量（2～4）").grid(
            row=1, column=0, sticky="w", pady=4)
        self.pieces = tk.IntVar(value=4)
        ttk.Spinbox(
            controls, from_=2, to=4, textvariable=self.pieces,
            width=12).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(controls, text="动画速度").grid(
            row=2, column=0, sticky="w", pady=4)
        speed_row = ttk.Frame(controls)
        speed_row.grid(row=2, column=1, sticky="ew")
        self.speed = tk.DoubleVar(value=1.0)
        ttk.Scale(
            speed_row, from_=.25, to=4.0, variable=self.speed,
            command=self._speed_changed).pack(side="left", fill="x", expand=True)
        self.speed_text = ttk.Label(speed_row, text="1.00×", width=7)
        self.speed_text.pack(side="right")
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(
            controls, text="▶ 开始拼接（生成→识别→执行）",
            command=self._start)
        self.start_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(12, 3),
            ipady=8)

        status_box = ttk.LabelFrame(right, text="实时状态", padding=8)
        status_box.pack(fill="x", pady=(10, 0))
        self.status = tk.StringVar(value="就绪")
        ttk.Label(
            status_box, textvariable=self.status, wraplength=460,
            font=("", 11)).pack(fill="x")
        log_box = ttk.LabelFrame(right, text="视觉识别与运动日志", padding=5)
        log_box.pack(fill="both", expand=True, pady=(10, 0))
        self.log = tk.Text(log_box, height=20, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    def _speed_changed(self, _=None):
        self.speed_value = float(self.speed.get())
        self.speed_text.configure(text=f"{self.speed_value:.2f}×")

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _start(self):
        if self.running:
            return
        count = int(self.pieces.get())
        if not 1 <= count <= 4:
            messagebox.showerror("参数错误", "碎片数量必须为 2～4")
            return
        self.running = True
        self.start_button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        worker = SimulationWorker(
            self.events, lambda: self.speed_value)
        threading.Thread(
            target=worker.run, args=(int(self.seed.get()), count),
            daemon=True).start()

    def _show(self, label, rgb, width, height, attr):
        scale = min(width / rgb.shape[1], height / rgb.shape[0])
        size = (max(1, int(rgb.shape[1] * scale)),
                max(1, int(rgb.shape[0] * scale)))
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        header = f"P6\n{size[0]} {size[1]}\n255\n".encode("ascii")
        photo = tk.PhotoImage(data=header + resized.tobytes(), format="PPM")
        setattr(self, attr, photo)
        label.configure(image=photo, text="")

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "frames":
                    self._show(
                        self.scene_label, event[1],
                        max(500, self.scene_label.winfo_width()),
                        max(400, self.scene_label.winfo_height()),
                        "scene_photo")
                    self._show(
                        self.camera_label, event[2],
                        max(180, self.camera_label.winfo_width()),
                        max(220, self.camera_label.winfo_height()),
                        "camera_photo")
                elif event[0] == "status":
                    self.status.set(event[1])
                elif event[0] == "log":
                    self._append_log(event[1])
                elif event[0] == "error":
                    self.running = False
                    self.start_button.configure(state="normal")
                    self.status.set("运行失败：" + event[1])
                    messagebox.showerror("仿真错误", event[1])
                elif event[0] == "done":
                    self.running = False
                    self.start_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(30, self._poll)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    PuzzleMuJoCoUI().mainloop()


if __name__ == "__main__":
    main()
