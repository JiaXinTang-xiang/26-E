#!/usr/bin/env python3
"""E题拼图视觉算法的 Tk 桌面测试界面。"""

from __future__ import annotations

import math
import base64
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from puzzle_device.simulation import puzzle_sim as sim
from mujoco_sim.sync_bridge import MuJoCoBridge
from puzzle_device.paths import OUTPUT_DIR


class PuzzleGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("E题拼图装置 · 视觉算法仿真")
        root.geometry("1280x860")
        root.minsize(1050, 720)

        self.seed = tk.IntVar(value=7)
        self.piece_count = tk.IntVar(value=4)
        self.material_mode = tk.StringVar(value="纯色卡纸")
        self.cut_mode = tk.StringVar(value="中心放射（原版）")
        self.status = tk.StringVar(value="输入随机种子，然后生成场景")
        self.scene = None
        self.pieces = None
        self.transforms = None
        self.matches = None
        self.current_image = None
        self.photo = None
        self.batch_running = False
        self.animation_job = None
        self.animation_speed = tk.DoubleVar(value=1.0)
        self.mujoco = MuJoCoBridge()
        self.waiting_for_scene = False
        self.task_started_at = None

        self._build_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(25, self._poll_mujoco)
        self.generate()

    def _build_style(self):
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 17, "bold"))
        style.configure("Step.TButton", font=("Sans", 11), padding=(13, 9))
        style.configure("Status.TLabel", foreground="#174c75", font=("Sans", 10))

    def _build_layout(self):
        header = ttk.Frame(self.root, padding=(16, 12))
        header.pack(fill="x")
        ttk.Label(header, text="E题拼图装置 · 视觉算法仿真", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="目标矩形 10 cm × 6 cm / 俯视相机 / 像素坐标",
                  foreground="#666").pack(
            side="right", pady=(7, 0))

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        view_frame = ttk.Frame(body)
        side = ttk.Frame(body, width=390)
        body.add(view_frame, weight=4)
        body.add(side, weight=2)

        self.canvas = tk.Canvas(view_frame, bg="#25282b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._show(self.current_image))

        controls = ttk.LabelFrame(side, text="测试控制", padding=12)
        controls.pack(fill="x", padx=(10, 0))
        seed_row = ttk.Frame(controls)
        seed_row.pack(fill="x", pady=(0, 9))
        ttk.Label(seed_row, text="随机种子").pack(side="left")
        ttk.Spinbox(seed_row, from_=0, to=999999, textvariable=self.seed, width=12).pack(
            side="right")
        count_row = ttk.Frame(controls)
        count_row.pack(fill="x", pady=(0, 9))
        ttk.Label(count_row, text="碎片数量（2～4）").pack(side="left")
        ttk.Spinbox(count_row, from_=2, to=4, textvariable=self.piece_count,
                    width=12, state="readonly").pack(side="right")
        material_row = ttk.Frame(controls)
        material_row.pack(fill="x", pady=(0, 9))
        ttk.Label(material_row, text="卡片素材").pack(side="left")
        ttk.Combobox(
            material_row, textvariable=self.material_mode, state="readonly",
            values=("纯色卡纸", "大鬼扑克牌"), width=12).pack(side="right")
        self.material_mode.trace_add("write", lambda *_: self.generate())
        cut_row = ttk.Frame(controls)
        cut_row.pack(fill="x", pady=(0, 9))
        ttk.Label(cut_row, text="切割方式").pack(side="left")
        ttk.Combobox(
            cut_row, textvariable=self.cut_mode, state="readonly",
            values=("中心放射（原版）", "边界扇形", "平行斜向条带",
                    "等分矩形（四块2×2）", "T形分层切割",
                    "角块多边形", "凹五边形折线切割",
                    "全类型随机"),
            width=16).pack(side="right")
        self.cut_mode.trace_add("write", lambda *_: self.generate())
        speed_row = ttk.Frame(controls)
        speed_row.pack(fill="x", pady=(0, 9))
        ttk.Label(speed_row, text="同步动画速度").pack(side="left")
        self.speed_label = ttk.Label(speed_row, text="1.00×", width=7)
        self.speed_label.pack(side="right")
        self.speed_scale = ttk.Scale(
            speed_row, from_=.25, to=4.0, variable=self.animation_speed,
            command=self._speed_changed)
        self.speed_scale.pack(
            side="right", fill="x", expand=True, padx=(12, 8))
        self.root.after(0, lambda: (
            self.animation_speed.set(1.0), self._speed_changed()))

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="1 生成场景", command=self.generate,
                   style="Step.TButton").grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(buttons, text="2 识别碎片", command=self.detect,
                   style="Step.TButton").grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(buttons, text="3 还原矩形", command=self.restore,
                   style="Step.TButton").grid(row=1, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(buttons, text="▶ 自动演示", command=self.auto_demo,
                   style="Step.TButton").grid(row=1, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(buttons, text="批量测试 100 次", command=self.start_batch).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=3, pady=(8, 3))
        ttk.Button(buttons, text="导出本次结果", command=self.export).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        buttons.columnconfigure((0, 1), weight=1)

        result = ttk.LabelFrame(side, text="识别与位姿结果", padding=10)
        result.pack(fill="both", expand=True, padx=(10, 0), pady=(10, 0))
        self.tabs = ttk.Notebook(result)
        self.tabs.pack(fill="both", expand=True)

        matrix_tab = ttk.Frame(self.tabs)
        edge_tab = ttk.Frame(self.tabs)
        self.tabs.add(matrix_tab, text="旋转平移矩阵")
        self.tabs.add(edge_tab, text="切割边配对")

        self.matrix_text = tk.Text(matrix_tab, wrap="none", font=("DejaVu Sans Mono", 9),
                                   bg="#f8f8f8", relief="flat")
        matrix_scroll = ttk.Scrollbar(matrix_tab, orient="vertical",
                                      command=self.matrix_text.yview)
        self.matrix_text.configure(yscrollcommand=matrix_scroll.set)
        self.matrix_text.pack(side="left", fill="both", expand=True)
        matrix_scroll.pack(side="right", fill="y")

        self.edge_text = tk.Text(edge_tab, wrap="word", font=("DejaVu Sans Mono", 10),
                                 bg="#f8f8f8", relief="flat")
        self.edge_text.pack(fill="both", expand=True)

        footer = ttk.Frame(self.root, padding=(16, 7))
        footer.pack(fill="x")
        ttk.Label(footer, textvariable=self.status, style="Status.TLabel").pack(side="left")
        self.progress = ttk.Progressbar(footer, length=240, maximum=100)
        self.progress.pack(side="right")

    def _show(self, image):
        if image is None or self.canvas.winfo_width() < 10:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        scale = min(cw / image.shape[1], ch / image.shape[0])
        size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", resized)
        if not ok:
            return
        data = base64.b64encode(png.tobytes())
        self.photo = tk.PhotoImage(data=data)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor="center")

    def generate(self):
        try:
            if self.animation_job is not None:
                self.root.after_cancel(self.animation_job)
                self.animation_job = None
            count = self.piece_count.get()
            self.scene = None
            self.pieces = self.transforms = self.matches = None
            self.matrix_text.delete("1.0", "end")
            self.edge_text.delete("1.0", "end")
            self.waiting_for_scene = True
            self.status.set("正在同步生成视觉场景并打开独立 MuJoCo 窗口…")
            self.mujoco.send(
                "generate", self.seed.get(), count,
                "joker" if self.material_mode.get() == "大鬼扑克牌"
                else "color",
                self._cut_mode_code())
        except Exception as exc:
            messagebox.showerror("生成失败", str(exc))

    def detect(self):
        if self.scene is None:
            messagebox.showinfo("场景生成中", "请等待 MuJoCo 场景生成完成")
            return
        try:
            # Detection receives only camera pixels; the generator's polygons,
            # adjacency and true transforms are not retained by the GUI.
            solver_mode = sim.resolve_cut_mode(
                self.seed.get(), self.piece_count.get(),
                self._cut_mode_code())
            self.pieces = sim.detect_pieces(
                self.scene, preserve_concavity=(solver_mode == "concave"))
            self.current_image = sim.annotate_detection(
                self.scene, self.pieces,
                expect_card_corners=(
                    self.material_mode.get() == "大鬼扑克牌"))
            self._show(self.current_image)
            vertices = "，".join(f"P{i}: {len(p)} 个顶点" for i, p in enumerate(self.pieces))
            self.status.set(f"识别成功：{len(self.pieces)} 块碎片；{vertices}")
            self.mujoco.send("detect")
        except Exception as exc:
            messagebox.showerror("识别失败", str(exc))

    def restore(self):
        if self.pieces is None:
            self.detect()
        if self.pieces is None:
            return
        try:
            solver_mode = sim.resolve_cut_mode(
                self.seed.get(), self.piece_count.get(),
                self._cut_mode_code())
            if self.material_mode.get() == "大鬼扑克牌":
                self.transforms, self.matches = sim.solve_textured_card(
                    self.scene, self.pieces, solver_mode)
            else:
                self.transforms, self.matches = sim.solve(
                    self.pieces, solver_mode)
            self._write_results()
            self.status.set("视觉规划完成，正在同步启动 PIPER-L 拼接…")
            # Send the exact camera-only solution shown in this UI. MuJoCo
            # must execute this same piece order and must not independently
            # choose another geometrically equivalent assembly.
            self.mujoco.send(
                "stitch", self.pieces, self.transforms,
                self.scene.shape[1], self.scene.shape[0])
        except Exception as exc:
            messagebox.showerror("拼接失败", str(exc))

    def _cut_mode_code(self):
        return {
            "中心放射（原版）": "common",
            "边界扇形": "boundary_fan",
            "平行斜向条带": "strips",
            "等分矩形（四块2×2）": "equal_rectangles",
            "T形分层切割": "t_junction",
            "角块多边形": "corner",
            "凹五边形折线切割": "concave",
            "全类型随机": "mixed",
        }[self.cut_mode.get()]

    def _start_motion_animation(self):
        """Move pieces one by one using their solved SE(2) transforms."""
        self.animation_piece = 0
        self.animation_frame = 0
        self.animation_frames_per_piece = 36
        self._animation_step()

    def _render_motion_state(self, piece_index, translation_t, rotation_t=None):
        """Render the exact MuJoCo-reported piece progress in the 2-D view."""
        if rotation_t is None:
            rotation_t = translation_t
        poses = []
        for i, piece in enumerate(self.pieces):
            if i < piece_index:
                pose = self.transforms[i]
            elif i == piece_index:
                h = self.transforms[i]
                angle = math.atan2(h[1, 0], h[0, 0])
                src_center = piece.mean(axis=0)
                dst_center = sim.apply_h(src_center[None], h)[0]
                move_t = translation_t * translation_t * (
                    3.0 - 2.0 * translation_t)
                rotate_t = rotation_t * rotation_t * (
                    3.0 - 2.0 * rotation_t)
                center = src_center * (1.0 - move_t) + dst_center * move_t
                pose = sim.rigid(angle * rotate_t, 0, 0)
                mapped_center = sim.apply_h(src_center[None], pose)[0]
                pose[:2, 2] = center - mapped_center
            else:
                pose = np.eye(3)
            poses.append(pose)
        frame = sim.render_piece_poses(
            self.scene, self.pieces, poses,
            preserve_texture=(
                self.material_mode.get() == "大鬼扑克牌"))
        self.current_image = frame
        self._show(frame)

    def _speed_changed(self, _=None):
        value = float(self.animation_speed.get())
        self.speed_label.configure(text=f"{value:.2f}×")
        self.mujoco.set_speed(value)

    def _poll_mujoco(self):
        try:
            # Drain the IPC pipe first and collapse bursts of motion frames.
            # At 2x/4x playback only the newest pose matters visually; status,
            # error and completion events remain ordered and lossless.
            compacted = []
            pending_motion = None
            for event in self.mujoco.poll():
                if event[0] in {"motion", "motion_pose"}:
                    pending_motion = event
                    continue
                if pending_motion is not None:
                    compacted.append(pending_motion)
                    pending_motion = None
                compacted.append(event)
            if pending_motion is not None:
                compacted.append(pending_motion)
            for event in compacted:
                if event[0] == "generated":
                    self.waiting_for_scene = False
                    self.scene = event[1]
                    self.current_image = self.scene
                    self._show(self.scene)
                    self.status.set(
                        f"种子 {self.seed.get()}：视觉页面与 MuJoCo 已同步生成 "
                        f"{self.piece_count.get()} 块碎片；"
                        f"素材={self.material_mode.get()}；"
                        f"切割={self.cut_mode.get()}")
                elif event[0] == "detected":
                    self.status.set(
                        f"视觉识别已同步到 MuJoCo：检测到 {event[1]} 块碎片")
                elif event[0] == "stage":
                    if event[1] < 0:
                        self.status.set(f"MuJoCo：{event[2]}")
                    else:
                        self.status.set(
                            f"MuJoCo 实时拼接 P{event[1]}：{event[2]}")
                elif event[0] == "motion" and self.transforms is not None:
                    self._render_motion_state(event[1], event[2])
                elif event[0] == "motion_pose" and self.transforms is not None:
                    self._render_motion_state(
                        event[1], event[2], event[3])
                elif event[0] == "done":
                    if (self.scene is None or self.pieces is None or
                            self.transforms is None):
                        continue
                    self.current_image = sim.render_solution(
                        self.scene, self.pieces, self.transforms,
                        preserve_texture=(
                            self.material_mode.get() == "大鬼扑克牌"))
                    self._show(self.current_image)
                    if self.task_started_at is not None:
                        elapsed = time.monotonic() - self.task_started_at
                        verdict = "满足 2 分钟要求" if elapsed <= 120 else "超过 2 分钟"
                        self.status.set(
                            f"双窗口同步拼接完成：10 cm × 6 cm；"
                            f"用时 {elapsed:.1f} s（{verdict}）")
                        self.task_started_at = None
                    else:
                        self.status.set(
                            "双窗口同步拼接完成：目标矩形 10 cm × 6 cm")
                    # 仿真端用系统提示音模拟题目要求的完成声/光指示。
                    self.root.bell()
                elif event[0] == "error":
                    messagebox.showerror("MuJoCo 仿真错误", event[1])
                    self.status.set("MuJoCo 仿真失败：" + event[1])
        finally:
            self.root.after(25, self._poll_mujoco)

    def _close(self):
        self.mujoco.close()
        self.root.destroy()

    def _animation_step(self):
        count = len(self.pieces)
        if self.animation_piece >= count:
            self.animation_job = None
            self.current_image = sim.render_solution(
                self.scene, self.pieces, self.transforms,
                preserve_texture=(
                    self.material_mode.get() == "大鬼扑克牌"))
            self._show(self.current_image)
            self.status.set(
                f"拼接动画完成：{count} 块碎片已依次移动到 10 cm × 6 cm 目标矩形")
            return

        t = self.animation_frame / self.animation_frames_per_piece
        # Smooth acceleration/deceleration instead of a visually abrupt linear move.
        t = t * t * (3.0 - 2.0 * t)
        poses = []
        for i, piece in enumerate(self.pieces):
            if i < self.animation_piece:
                pose = self.transforms[i]
            elif i == self.animation_piece:
                h = self.transforms[i]
                angle = math.atan2(h[1, 0], h[0, 0])
                src_center = piece.mean(axis=0)
                dst_center = sim.apply_h(src_center[None], h)[0]
                center = src_center * (1.0 - t) + dst_center * t
                pose = sim.rigid(angle * t, 0, 0)
                mapped_center = sim.apply_h(src_center[None], pose)[0]
                pose[:2, 2] = center - mapped_center
            else:
                pose = np.eye(3)
            poses.append(pose)
        frame = sim.render_piece_poses(
            self.scene, self.pieces, poses,
            preserve_texture=(
                self.material_mode.get() == "大鬼扑克牌"))

        self.current_image = frame
        self._show(frame)
        self.status.set(
            f"实时拼接：正在移动 P{self.animation_piece} "
            f"({self.animation_frame}/{self.animation_frames_per_piece})")
        self.animation_frame += 1
        if self.animation_frame > self.animation_frames_per_piece:
            self.animation_piece += 1
            self.animation_frame = 0
        self.animation_job = self.root.after(25, self._animation_step)

    def _write_results(self):
        self.matrix_text.delete("1.0", "end")
        for i, (piece, h) in enumerate(zip(self.pieces, self.transforms)):
            angle = math.degrees(math.atan2(h[1, 0], h[0, 0]))
            center = piece.mean(axis=0)
            target_center = sim.apply_h(center[None], h)[0]
            lines = [
                f"碎片 P{i}",
                f"  当前中心: ({center[0]:.1f}, {center[1]:.1f}) px",
                f"  目标中心: ({target_center[0]:.1f}, {target_center[1]:.1f}) px",
                f"  旋转角度: {angle:.3f}°",
                f"  平移: tx={h[0, 2]:.3f}, ty={h[1, 2]:.3f}",
                "  齐次矩阵:",
                f"  [{h[0,0]: .6f} {h[0,1]: .6f} {h[0,2]: .3f}]",
                f"  [{h[1,0]: .6f} {h[1,1]: .6f} {h[1,2]: .3f}]",
                "  [ 0.000000  0.000000  1.000]",
                "",
            ]
            self.matrix_text.insert("end", "\n".join(lines))

        self.edge_text.delete("1.0", "end")
        self.edge_text.insert("end", "算法识别出的内部切割边：\n\n")
        for k, match in enumerate(self.matches, 1):
            err, i, ei, j, ej = match[:5]
            partial = tuple(match[5:]) != (0., 1., 0., 1.)
            self.edge_text.insert(
                "end", f"{k}. P{i} 的边 {ei} ↔ P{j} 的边 {ej}\n"
                       f"   类型：{'部分边 / T形' if partial else '完整边'}\n"
                       f"   匹配代价：{err * 100:.3f}%\n\n")
        self.edge_text.insert(
            "end", "边编号对应检测图中的红色顶点序号；边 i 是顶点 i 到下一个顶点。")

    def auto_demo(self):
        self.generate()
        # 现场随机摆片发生在一键启动之前；计时从视觉识别开始。
        self.task_started_at = time.monotonic()
        self.status.set("自动演示：场景已生成，即将进行视觉识别…")
        self.root.after(700, self._auto_detect)

    def _auto_detect(self):
        self.detect()
        self.status.set("自动演示：轮廓识别完成，即将执行几何拼接…")
        self.root.after(900, self.restore)

    def start_batch(self):
        if self.batch_running:
            return
        self.batch_running = True
        self.batch_index, self.batch_errors, self.batch_failures = 0, [], []
        self.progress["value"] = 0
        self.status.set("正在执行 100 组随机切割测试…")
        self.root.after(1, self._batch_step)

    def _batch_step(self):
        if self.batch_index >= 100:
            self.batch_running = False
            successful = len(self.batch_errors)
            max_err = max(self.batch_errors) if self.batch_errors else float("nan")
            avg_err = np.mean(self.batch_errors) if self.batch_errors else float("nan")
            self.status.set(
                f"批量测试完成：成功 {successful}/100，失败 {self.batch_failures}，"
                f"平均尺寸误差 {avg_err:.2f}px")
            messagebox.showinfo(
                "批量测试完成",
                f"成功：{successful}/100\n失败：{self.batch_failures}\n"
                f"平均矩形尺寸误差：{avg_err:.3f} px\n最大误差：{max_err:.3f} px")
            return
        test_seed = self.seed.get() + self.batch_index
        try:
            self.batch_errors.append(sim.run_once(
                test_seed, OUTPUT_DIR, save=False, piece_count=self.piece_count.get()))
        except Exception:
            self.batch_failures += 1
        self.batch_index += 1
        self.progress["value"] = self.batch_index
        self.status.set(f"批量测试：{self.batch_index}/100")
        self.root.after(1, self._batch_step)

    def export(self):
        if self.transforms is None:
            self.restore()
        if self.transforms is None:
            return
        folder = filedialog.askdirectory(title="选择结果输出目录")
        if not folder:
            return
        try:
            sim.run_once(
                    self.seed.get(), Path(folder), save=True, piece_count=self.piece_count.get())
            self.status.set(f"本次结果已导出到：{folder}")
            messagebox.showinfo("导出完成", f"图片与 transforms.json 已保存到：\n{folder}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))


def main():
    root = tk.Tk()
    PuzzleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
