from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from puzzle_core.config import DEFAULT_IMAGES_ROOT, DEFAULT_OUTPUTS_ROOT
from puzzle_core.generation import generate_case
from puzzle_core.pipeline import solve_image


MODE_OPTIONS = {
    "选手自备四片": "self",
    "现场纯白碎片": "field-white",
    "现场扑克牌碎片": "field-card",
}
MODE_NAMES = {value: key for key, value in MODE_OPTIONS.items()}


@dataclass(frozen=True)
class ResultBundle:
    input_path: Path
    output_dir: Path
    detected_path: Path
    solved_path: Path
    movement_steps: tuple[Path, ...]
    plan: dict[str, object]


def collect_result_bundle(
    input_path: str | Path, output_dir: str | Path
) -> ResultBundle:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    detected_path = output_dir / "detected.png"
    solved_path = output_dir / "solved.png"
    plan_path = output_dir / "movement_plan.json"
    required = (input_path, detected_path, solved_path, plan_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少结果文件：" + "，".join(missing))
    steps = tuple(sorted((output_dir / "movement_steps").glob("step_*.png")))
    if not steps:
        raise FileNotFoundError("没有找到 movement_steps/step_*.png")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return ResultBundle(
        input_path=input_path,
        output_dir=output_dir,
        detected_path=detected_path,
        solved_path=solved_path,
        movement_steps=steps,
        plan=plan,
    )


def _format_xy(value: object) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "-"
    return f"({float(value[0]):.1f}, {float(value[1]):.1f})"


class ZoomImageViewer(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._source: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._path: Path | None = None
        self._zoom = 1.0
        self._render_job: str | None = None

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(
            self,
            background="#242a31",
            highlightthickness=0,
            cursor="crosshair",
        )
        x_scroll = ttk.Scrollbar(
            self, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        y_scroll = ttk.Scrollbar(
            self, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas.configure(
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        controls = ttk.Frame(self, padding=(4, 5))
        controls.grid(row=2, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Button(
            controls,
            text="－",
            width=3,
            command=lambda: self.zoom_by(1.0 / 1.2),
        ).grid(row=0, column=0)
        self.zoom_text = tk.StringVar(value="适应窗口")
        ttk.Label(
            controls,
            textvariable=self.zoom_text,
            anchor=tk.CENTER,
        ).grid(row=0, column=1, sticky="ew")
        ttk.Button(
            controls,
            text="＋",
            width=3,
            command=lambda: self.zoom_by(1.2),
        ).grid(row=0, column=2)
        ttk.Button(
            controls,
            text="适应",
            width=6,
            command=self.fit,
        ).grid(row=0, column=3, padx=(6, 0))

        self.canvas.bind("<Configure>", self._schedule_render)
        self.canvas.bind(
            "<MouseWheel>",
            lambda event: self.zoom_by(1.2 if event.delta > 0 else 1.0 / 1.2),
        )

    def set_image(self, path: str | Path) -> None:
        self._path = Path(path)
        with Image.open(self._path) as opened:
            self._source = opened.convert("RGB")
        self._zoom = 1.0
        self._schedule_render()

    def fit(self) -> None:
        self._zoom = 1.0
        self._schedule_render()

    def zoom_by(self, factor: float) -> None:
        if self._source is None:
            return
        self._zoom = min(8.0, max(0.25, self._zoom * factor))
        self._schedule_render()

    def _schedule_render(self, _event: tk.Event | None = None) -> None:
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(40, self._render)

    def _render(self) -> None:
        self._render_job = None
        if self._source is None:
            self.canvas.delete("all")
            return
        canvas_width = max(self.canvas.winfo_width() - 12, 100)
        canvas_height = max(self.canvas.winfo_height() - 12, 100)
        image_width, image_height = self._source.size
        fit_scale = min(
            canvas_width / image_width,
            canvas_height / image_height,
        )
        scale = max(0.02, fit_scale * self._zoom)
        size = (
            max(1, int(round(image_width * scale))),
            max(1, int(round(image_height * scale))),
        )
        resized = self._source.resize(size, Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        offset_x = max(0, (canvas_width - size[0]) // 2)
        offset_y = max(0, (canvas_height - size[1]) // 2)
        self.canvas.create_image(
            offset_x,
            offset_y,
            image=self._photo,
            anchor=tk.NW,
        )
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                max(canvas_width, size[0] + offset_x),
                max(canvas_height, size[1] + offset_y),
            )
        )
        self.zoom_text.set(
            f"{scale * 100:.0f}% · {image_width}×{image_height}"
        )


class PuzzleVisualApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("电赛 E 题 · 拼图视觉与运动仿真")
        self.root.geometry("1420x900")
        self.root.minsize(1080, 700)
        self._configure_style()

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_bundle: ResultBundle | None = None
        self.step_index = 0
        self.playing = False
        self.play_job: str | None = None

        self.kind_text = tk.StringVar(value="现场扑克牌碎片")
        self.count_text = tk.StringVar(value="4")
        self.seed_text = tk.StringVar(value="20260729")
        self.status_text = tk.StringVar(value="就绪")
        self.summary_text = tk.StringVar(
            value="设置参数后点击“生成并求解”，或打开一张已有输入图。"
        )
        self.output_text = tk.StringVar(value="尚未生成结果")
        self.step_text = tk.StringVar(value="步骤 0/0")

        self._build_toolbar()
        self._build_main_area()
        self._build_statusbar()
        self._on_kind_changed()
        self.root.after(100, self._poll_messages)

    def _configure_style(self) -> None:
        self.root.configure(background="#1d2228")
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background="#1d2228", foreground="#e9eef3")
        style.configure("TFrame", background="#1d2228")
        style.configure("TLabelframe", background="#1d2228")
        style.configure(
            "TLabelframe.Label",
            background="#1d2228",
            foreground="#e9eef3",
        )
        style.configure("TLabel", background="#1d2228", foreground="#e9eef3")
        style.configure(
            "TButton",
            background="#343c45",
            foreground="#f2f5f7",
            borderwidth=1,
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#46515d"), ("disabled", "#292f35")],
            foreground=[("disabled", "#7e8790")],
        )
        style.configure(
            "Primary.TButton",
            background="#1677d2",
            foreground="#ffffff",
            padding=(14, 7),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#258be6"), ("disabled", "#31536f")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#2a3037",
            foreground="#f2f5f7",
            insertcolor="#ffffff",
        )
        style.configure(
            "TCombobox",
            fieldbackground="#2a3037",
            background="#343c45",
            foreground="#f2f5f7",
            arrowcolor="#e9eef3",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#2a3037"), ("disabled", "#252a30")],
            foreground=[("readonly", "#f2f5f7"), ("disabled", "#7e8790")],
        )
        style.configure(
            "TNotebook",
            background="#1d2228",
            borderwidth=0,
        )
        style.configure(
            "TNotebook.Tab",
            background="#2b3239",
            foreground="#d7dde2",
            padding=(13, 7),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#1677d2"), ("active", "#3b454f")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Treeview",
            background="#252b32",
            fieldbackground="#252b32",
            foreground="#e9eef3",
            rowheight=28,
        )
        style.configure(
            "Treeview.Heading",
            background="#343c45",
            foreground="#f1f4f6",
        )
        style.map(
            "Treeview",
            background=[("selected", "#196fba")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background="#258be6",
            troughcolor="#2a3037",
        )

    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(12, 10))
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="案例类型").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.kind_combo = ttk.Combobox(
            toolbar,
            textvariable=self.kind_text,
            values=list(MODE_OPTIONS),
            state="readonly",
            width=18,
        )
        self.kind_combo.grid(row=0, column=1, padx=(0, 14))
        self.kind_combo.bind("<<ComboboxSelected>>", self._on_kind_changed)

        ttk.Label(toolbar, text="碎片数量").grid(
            row=0, column=2, sticky="w", padx=(0, 6)
        )
        self.count_spin = ttk.Spinbox(
            toolbar,
            from_=1,
            to=4,
            textvariable=self.count_text,
            width=5,
            state="readonly",
        )
        self.count_spin.grid(row=0, column=3, padx=(0, 14))

        ttk.Label(toolbar, text="随机种子").grid(
            row=0, column=4, sticky="w", padx=(0, 6)
        )
        self.seed_entry = ttk.Entry(
            toolbar,
            textvariable=self.seed_text,
            width=12,
        )
        self.seed_entry.grid(row=0, column=5, padx=(0, 16))

        self.generate_button = ttk.Button(
            toolbar,
            text="生成并求解",
            style="Primary.TButton",
            command=self.start_generate,
        )
        self.generate_button.grid(row=0, column=6, padx=(0, 8))
        self.open_button = ttk.Button(
            toolbar,
            text="打开输入图",
            command=self.choose_input,
        )
        self.open_button.grid(row=0, column=7, padx=(0, 8))
        self.folder_button = ttk.Button(
            toolbar,
            text="打开输出目录",
            command=self.open_output_folder,
            state=tk.DISABLED,
        )
        self.folder_button.grid(row=0, column=8)
        toolbar.columnconfigure(9, weight=1)

    def _build_main_area(self) -> None:
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

        visual = ttk.Frame(paned)
        side = ttk.Frame(paned, width=430)
        paned.add(visual, weight=4)
        paned.add(side, weight=2)

        self.image_tabs = ttk.Notebook(visual)
        self.image_tabs.pack(fill=tk.BOTH, expand=True)

        self.input_viewer = ZoomImageViewer(self.image_tabs)
        self.detected_viewer = ZoomImageViewer(self.image_tabs)
        self.solved_viewer = ZoomImageViewer(self.image_tabs)
        self.image_tabs.add(self.input_viewer, text="输入图")
        self.image_tabs.add(self.detected_viewer, text="识别标注")
        self.image_tabs.add(self.solved_viewer, text="最终拼图")

        movement_frame = ttk.Frame(self.image_tabs)
        movement_frame.rowconfigure(0, weight=1)
        movement_frame.columnconfigure(0, weight=1)
        self.movement_viewer = ZoomImageViewer(movement_frame)
        self.movement_viewer.grid(row=0, column=0, sticky="nsew")
        navigation = ttk.Frame(movement_frame, padding=(6, 5))
        navigation.grid(row=1, column=0, sticky="ew")
        navigation.columnconfigure(2, weight=1)
        ttk.Button(
            navigation,
            text="上一帧",
            command=lambda: self.change_step(-1),
        ).grid(row=0, column=0, padx=(0, 6))
        self.play_button = ttk.Button(
            navigation,
            text="播放",
            command=self.toggle_play,
        )
        self.play_button.grid(row=0, column=1)
        ttk.Label(
            navigation,
            textvariable=self.step_text,
            anchor=tk.CENTER,
        ).grid(row=0, column=2, sticky="ew")
        ttk.Button(
            navigation,
            text="下一帧",
            command=lambda: self.change_step(1),
        ).grid(row=0, column=3)
        self.image_tabs.add(movement_frame, text="移动过程")

        side_tabs = ttk.Notebook(side)
        side_tabs.pack(fill=tk.BOTH, expand=True)

        summary_page = ttk.Frame(side_tabs, padding=12)
        summary_page.columnconfigure(0, weight=1)
        ttk.Label(
            summary_page,
            text="运行摘要",
            font=("Microsoft YaHei UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            summary_page,
            textvariable=self.summary_text,
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=390,
        ).grid(row=1, column=0, sticky="new", pady=(10, 14))
        ttk.Separator(summary_page).grid(row=2, column=0, sticky="ew")
        ttk.Label(
            summary_page,
            text="输出位置",
            font=("Microsoft YaHei UI", 11, "bold"),
        ).grid(row=3, column=0, sticky="w", pady=(14, 6))
        ttk.Label(
            summary_page,
            textvariable=self.output_text,
            justify=tk.LEFT,
            anchor=tk.NW,
            wraplength=390,
        ).grid(row=4, column=0, sticky="new")
        side_tabs.add(summary_page, text="摘要")

        pieces_page = ttk.Frame(side_tabs, padding=(6, 8))
        pieces_page.rowconfigure(0, weight=1)
        pieces_page.columnconfigure(0, weight=1)
        columns = ("id", "source", "pick", "target", "rotation")
        self.piece_table = ttk.Treeview(
            pieces_page,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "id": "编号",
            "source": "原中心/mm",
            "pick": "抓取点/mm",
            "target": "目标中心/mm",
            "rotation": "旋转/°",
        }
        widths = {
            "id": 52,
            "source": 116,
            "pick": 116,
            "target": 116,
            "rotation": 76,
        }
        for column in columns:
            self.piece_table.heading(column, text=headings[column])
            self.piece_table.column(
                column,
                width=widths[column],
                minwidth=48,
                anchor=tk.CENTER,
                stretch=column != "id",
            )
        table_scroll = ttk.Scrollbar(
            pieces_page,
            orient=tk.VERTICAL,
            command=self.piece_table.yview,
        )
        self.piece_table.configure(yscrollcommand=table_scroll.set)
        self.piece_table.grid(row=0, column=0, sticky="nsew")
        table_scroll.grid(row=0, column=1, sticky="ns")
        side_tabs.add(pieces_page, text="碎片坐标")

        json_page = ttk.Frame(side_tabs, padding=(6, 8))
        json_page.rowconfigure(0, weight=1)
        json_page.columnconfigure(0, weight=1)
        self.json_text = tk.Text(
            json_page,
            background="#252b32",
            foreground="#e9eef3",
            insertbackground="#ffffff",
            selectbackground="#196fba",
            wrap=tk.NONE,
            relief=tk.FLAT,
            font=("Consolas", 10),
        )
        json_x = ttk.Scrollbar(
            json_page, orient=tk.HORIZONTAL, command=self.json_text.xview
        )
        json_y = ttk.Scrollbar(
            json_page, orient=tk.VERTICAL, command=self.json_text.yview
        )
        self.json_text.configure(
            xscrollcommand=json_x.set,
            yscrollcommand=json_y.set,
        )
        self.json_text.grid(row=0, column=0, sticky="nsew")
        json_y.grid(row=0, column=1, sticky="ns")
        json_x.grid(row=1, column=0, sticky="ew")
        side_tabs.add(json_page, text="运动 JSON")

    def _build_statusbar(self) -> None:
        status = ttk.Frame(self.root, padding=(12, 5))
        status.pack(fill=tk.X)
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_text).grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(
            status,
            mode="indeterminate",
            length=180,
        )
        self.progress.grid(row=0, column=1, sticky="e")

    def _on_kind_changed(self, _event: tk.Event | None = None) -> None:
        if MODE_OPTIONS[self.kind_text.get()] == "self":
            self.count_text.set("4")
            self.count_spin.configure(state=tk.DISABLED)
        else:
            self.count_spin.configure(state="readonly")

    def _read_parameters(self) -> tuple[str, int, int]:
        kind = MODE_OPTIONS[self.kind_text.get()]
        try:
            count = int(self.count_text.get())
            seed = int(self.seed_text.get().strip())
        except ValueError as error:
            raise ValueError("碎片数量和随机种子必须是整数") from error
        if count not in (1, 2, 3, 4):
            raise ValueError("现场碎片数量只能为 1～4")
        if kind == "self":
            count = 4
        return kind, count, seed

    def start_generate(self) -> None:
        try:
            kind, count, seed = self._read_parameters()
        except ValueError as error:
            messagebox.showerror("参数错误", str(error), parent=self.root)
            return
        self._set_busy(True, "正在生成碎片并执行识别、拼图……")

        def job() -> None:
            try:
                case_dir = generate_case(
                    kind,
                    seed,
                    count,
                    output_root=DEFAULT_IMAGES_ROOT,
                )
                output_dir = DEFAULT_OUTPUTS_ROOT / case_dir.name
                solve_image(case_dir / "input.png", output_dir)
                bundle = collect_result_bundle(
                    case_dir / "input.png",
                    output_dir,
                )
                self.message_queue.put(("done", bundle))
            except Exception as error:
                self.message_queue.put(("error", error))

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def choose_input(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="选择 A4 拼图输入图片",
            filetypes=[
                ("图像文件", "*.png *.jpg *.jpeg *.bmp"),
                ("所有文件", "*.*"),
            ],
            initialdir=DEFAULT_IMAGES_ROOT,
        )
        if selected:
            self.solve_existing(Path(selected))

    def solve_existing(self, input_path: Path) -> None:
        if not input_path.is_file():
            messagebox.showerror(
                "文件不存在",
                str(input_path),
                parent=self.root,
            )
            return
        if input_path.name.lower() == "input.png" and input_path.parent.parent == DEFAULT_IMAGES_ROOT:
            output_name = input_path.parent.name
        else:
            output_name = (
                f"imported_{input_path.stem}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}"
            )
        output_dir = DEFAULT_OUTPUTS_ROOT / output_name
        self._set_busy(True, "正在识别已有图片并计算拼图……")

        def job() -> None:
            try:
                solve_image(input_path, output_dir)
                bundle = collect_result_bundle(input_path, output_dir)
                self.message_queue.put(("done", bundle))
            except Exception as error:
                self.message_queue.put(("error", error))

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def _poll_messages(self) -> None:
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()
                if kind == "done":
                    self._load_bundle(payload)  # type: ignore[arg-type]
                    self._set_busy(False, "处理完成")
                else:
                    self._set_busy(False, "处理失败")
                    messagebox.showerror(
                        "处理失败",
                        f"{type(payload).__name__}: {payload}",
                        parent=self.root,
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._poll_messages)

    def _load_bundle(self, bundle: ResultBundle) -> None:
        self.current_bundle = bundle
        self.input_viewer.set_image(bundle.input_path)
        self.detected_viewer.set_image(bundle.detected_path)
        self.solved_viewer.set_image(bundle.solved_path)
        self.step_index = 0
        self._show_step()
        self.image_tabs.select(1)

        plan = bundle.plan
        target = plan.get("target_rect", {})
        quality = plan.get("quality", {})
        pieces = plan.get("pieces", [])
        if not isinstance(target, dict):
            target = {}
        if not isinstance(quality, dict):
            quality = {}
        if not isinstance(pieces, list):
            pieces = []
        mode = str(plan.get("mode", "-"))
        self.summary_text.set(
            "\n".join(
                [
                    f"识别模式：{MODE_NAMES.get(mode, mode)}",
                    f"碎片数量：{len(pieces)}",
                    (
                        "目标矩形："
                        f"{float(target.get('width_mm', 0.0)):.2f} × "
                        f"{float(target.get('height_mm', 0.0)):.2f} mm"
                    ),
                    f"目标中心：{_format_xy(target.get('center_mm'))} mm",
                    f"几何评分：{float(quality.get('geometry_score', 0.0)):.4f}",
                    f"纹理评分：{float(quality.get('texture_score', 0.0)):.4f}",
                ]
            )
        )
        self.output_text.set(str(bundle.output_dir.resolve()))
        self.json_text.configure(state=tk.NORMAL)
        self.json_text.delete("1.0", tk.END)
        self.json_text.insert(
            "1.0",
            json.dumps(plan, ensure_ascii=False, indent=2),
        )
        self.json_text.configure(state=tk.DISABLED)
        for row in self.piece_table.get_children():
            self.piece_table.delete(row)
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            self.piece_table.insert(
                "",
                tk.END,
                values=(
                    piece.get("id", "-"),
                    _format_xy(piece.get("source_center_mm")),
                    _format_xy(piece.get("pick_point_mm")),
                    _format_xy(piece.get("target_center_mm")),
                    f"{float(piece.get('rotation_deg', 0.0)):.1f}",
                ),
            )
        self.folder_button.configure(state=tk.NORMAL)

    def _show_step(self) -> None:
        if self.current_bundle is None:
            return
        steps = self.current_bundle.movement_steps
        self.step_index = min(max(0, self.step_index), len(steps) - 1)
        self.movement_viewer.set_image(steps[self.step_index])
        self.step_text.set(
            f"步骤 {self.step_index}/{len(steps) - 1} · {steps[self.step_index].name}"
        )

    def change_step(self, delta: int) -> None:
        if self.current_bundle is None:
            return
        self.stop_playback()
        self.step_index += delta
        self._show_step()

    def toggle_play(self) -> None:
        if self.current_bundle is None:
            return
        if self.playing:
            self.stop_playback()
        else:
            if self.step_index >= len(self.current_bundle.movement_steps) - 1:
                self.step_index = 0
            self.playing = True
            self.play_button.configure(text="暂停")
            self._advance_playback()

    def _advance_playback(self) -> None:
        if not self.playing or self.current_bundle is None:
            return
        self._show_step()
        if self.step_index >= len(self.current_bundle.movement_steps) - 1:
            self.stop_playback()
            return
        self.step_index += 1
        self.play_job = self.root.after(750, self._advance_playback)

    def stop_playback(self) -> None:
        self.playing = False
        self.play_button.configure(text="播放")
        if self.play_job is not None:
            self.root.after_cancel(self.play_job)
            self.play_job = None

    def open_output_folder(self) -> None:
        if self.current_bundle is None:
            return
        path = self.current_bundle.output_dir.resolve()
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("输出目录", str(path), parent=self.root)
        except OSError as error:
            messagebox.showerror("无法打开目录", str(error), parent=self.root)

    def _set_busy(self, busy: bool, status: str) -> None:
        self.status_text.set(status)
        button_state = tk.DISABLED if busy else tk.NORMAL
        self.generate_button.configure(state=button_state)
        self.open_button.configure(state=button_state)
        if busy:
            self.stop_playback()
            self.progress.start(12)
        else:
            self.progress.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the E-problem puzzle visual interface."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional input image to solve immediately after launch.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = tk.Tk()
    app = PuzzleVisualApp(root)
    if args.input is not None:
        root.after(250, lambda: app.solve_existing(args.input))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
