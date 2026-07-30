#!/usr/bin/env python3
"""三子棋对弈系统 v4 - 全部题目功能"""

import os, sys, json, time, cv2, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import customtkinter as ctk
from lzjpy.chess_detection import (detect_chessboard, draw_board_overlay,
                              load_calibration, init_undistort_maps)
from lzjpy.game_engine import GameEngine
from lzjpy.serial_client import SerialClient

_yolo_detect = None
DEFAULT_PARAMS = {"canny_low": 50, "canny_high": 150, "area_min": 7000,
    "area_max": 15000, "min_rectangularity": 0.65, "yolo_conf": 0.4}
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
params = DEFAULT_PARAMS.copy()
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def cv2_to_ctk(cv_img, w=None, h=None):
    if cv_img is None: return None
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if w and h: pil = pil.resize((w, h), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)

def draw_virtual_board(pieces, highlight=-1, size=200):
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    cs = size // 3
    for i in range(1, 3):
        cv2.line(img, (i*cs, 0), (i*cs, size), (50,50,50), 2)
        cv2.line(img, (0, i*cs), (size, i*cs), (50,50,50), 2)
    if 0 <= highlight < 9:
        hr, hc = highlight // 3, highlight % 3
        cv2.rectangle(img, (hc*cs+2, hr*cs+2), ((hc+1)*cs-2, (hr+1)*cs-2), (0,200,255), 4)
    for idx in range(9):
        row, col = idx // 3, idx % 3
        cx, cy = col*cs + cs//2, row*cs + cs//2
        cv2.putText(img, str(idx + 1), (col * cs + 5, row * cs + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1,
                    cv2.LINE_AA)
        r = int(cs*0.35)
        if pieces[idx] == 1: cv2.circle(img, (cx,cy), r, (40,40,40), -1)
        elif pieces[idx] == 2:
            cv2.circle(img, (cx,cy), r, (230,230,230), -1)
            cv2.circle(img, (cx,cy), r, (50,50,50), 2)
    return img

def draw_crop_board(frame, board_box, pieces, highlight=-1, size=300):
    if board_box is None or frame is None: return None
    pts = board_box.astype(np.float32).reshape(4, 2)
    dst = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(frame, M, (size, size))
    cs = size // 3
    for i in range(1, 3):
        cv2.line(crop, (i*cs,0), (i*cs,size), (255,0,0), 1)
        cv2.line(crop, (0,i*cs), (size,i*cs), (255,0,0), 1)
    if 0 <= highlight < 9:
        hr, hc = highlight // 3, highlight % 3
        cv2.rectangle(crop, (hc*cs+3, hr*cs+3), ((hc+1)*cs-3, (hr+1)*cs-3), (0,255,255), 4)
    for idx in range(9):
        row, col = idx // 3, idx % 3
        cx, cy = col*cs + cs//2, row*cs + cs//2
        cv2.putText(crop, str(idx + 1), (col * cs + 7, row * cs + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1,
                    cv2.LINE_AA)
        r = int(cs*0.35)
        if pieces[idx] == 1: cv2.circle(crop, (cx,cy), r, (0,0,0), -1)
        elif pieces[idx] == 2:
            cv2.circle(crop, (cx,cy), r, (255,255,255), -1)
            cv2.circle(crop, (cx,cy), r, (0,0,0), 2)
    return crop

# =============================================================================
# 可点击棋盘
# =============================================================================
class ClickableBoard(ctk.CTkFrame):
    def __init__(self, master, size=300, **kw):
        super().__init__(master, **kw)
        self.board_size = size
        self.on_cell_click = None
        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self._click)

    def _click(self, ev):
        if self.on_cell_click is None: return
        cs = self.label.winfo_width() // 3
        if cs <= 0: return
        col, row = ev.x // cs, ev.y // cs
        if 0 <= row < 3 and 0 <= col < 3:
            self.on_cell_click(row * 3 + col)

    def update(self, cv_img):
        w = self.label.winfo_width()
        h = self.label.winfo_height()
        if w < 50: w = self.board_size
        if h < 50: h = self.board_size
        ctk_img = cv2_to_ctk(cv_img, w, h)
        if ctk_img: self.label.configure(image=ctk_img, text="")

# =============================================================================
# 主应用
# =============================================================================
class TicTacToeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("三子棋对弈系统")
        self.geometry("1150x720")
        self.minsize(1050, 620)
        self.engine = GameEngine(SerialClient(port='/dev/ttyUSB0'))
        self.cap = None; self.running = True
        self.use_undistort = False; self.map1 = self.map2 = None
        self.squares_center = None; self.inner_radius = 30
        self.board_box = None; self.pieces = [0]*9; self.detections = []
        self.grid_selected = []; self._init_camera(); self._load_params()
        self._build_ui(); self._start_video()

    def _init_camera(self):
        yp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calibration.yaml")
        try:
            cm, dc, _ = load_calibration(yp)
            self.map1, self.map2 = init_undistort_maps(cm, dc)
            self.use_undistort = True
        except: self.use_undistort = False
        for cid in [2, 0, 8]:
            cap = cv2.VideoCapture(cid)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap = cap; return

    def _load_params(self):
        global params
        try:
            with open(PARAMS_FILE) as f: params.update(json.load(f))
        except FileNotFoundError: pass

    def _save_params(self):
        with open(PARAMS_FILE, 'w') as f: json.dump(params, f, indent=2)

    # ==================================================================
    # UI构建
    # ==================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)  # 棋盘
        self.grid_columnconfigure(1, weight=3)  # 缩略图+预览
        self.grid_columnconfigure(2, weight=4)  # 控制面板
        self.grid_rowconfigure(0, weight=1)

        F = ctk.CTkFont; fb = F(size=14, weight="bold"); fn = F(size=13)

        # ---- 左: 裁剪棋盘(可点击) ----
        self.crop_board = ClickableBoard(self, size=300)
        self.crop_board.grid(row=0, column=0, sticky="nsew", padx=(5,2), pady=5)
        self.crop_board.on_cell_click = self._on_board_click
        ctk.CTkLabel(self, text="点击棋盘选目标格子", font=F(size=11), text_color="gray").grid(
            row=1, column=0, sticky="n", pady=(0,3))

        # ---- 中: 虚拟棋盘 + 摄像头预览 ----
        mid = ctk.CTkFrame(self)
        mid.grid(row=0, column=1, sticky="nsew", padx=2, pady=5, rowspan=2)
        mid.grid_columnconfigure(0, weight=1); mid.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(mid, text="虚拟棋盘 (点击选格子)", font=fb).pack(pady=(5,2))
        self.virtual_board = ClickableBoard(mid, size=180)
        self.virtual_board.pack(padx=5, pady=3)
        self.virtual_board.on_cell_click = self._on_board_click

        ctk.CTkLabel(mid, text="摄像头预览", font=fb).pack(pady=(10,2))
        self.cam_preview = ctk.CTkLabel(mid, text="")
        self.cam_preview.pack(padx=5, pady=3, fill="both", expand=True)

        # ---- 右: 控制面板 ----
        R = ctk.CTkFrame(self)
        R.grid(row=0, column=2, sticky="nsew", padx=(2,5), pady=5, rowspan=2)
        R.grid_columnconfigure(0, weight=1)
        r = 0; P = {'padx': 7, 'pady': 2}

        # -- 题目 --
        ctk.CTkLabel(R, text="题目选择", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        tf = ctk.CTkFrame(R, fg_color="transparent"); tf.grid(row=r, column=0, sticky="ew", padx=3)
        tasks = ["(1)单黑","(2)两黑两白","(3)旋转棋盘",
                 "(4)AI先手","(5)人先手","(6)作弊检测","(8)双人对弈"]
        for i, d in enumerate(tasks):
            t = 8 if d == "(8)双人对弈" else i + 1
            ctk.CTkButton(tf, text=d, height=30, font=F(size=12),
                          command=lambda t=t: self._on_task(t)).grid(
                row=i//2, column=i%2, padx=2, pady=2, sticky="ew")
            tf.grid_columnconfigure(i%2, weight=1)
        r += 1

        # -- 执棋 --
        ctk.CTkLabel(R, text="执棋方", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        sf = ctk.CTkFrame(R, fg_color="transparent"); sf.grid(row=r, column=0, sticky="ew", padx=3)
        self.side_var = ctk.IntVar(value=1)
        ctk.CTkRadioButton(sf, text="系统执黑", variable=self.side_var, value=1,
                           font=fn, command=lambda: self._on_side(1)).pack(side="left", padx=12)
        ctk.CTkRadioButton(sf, text="系统执白", variable=self.side_var, value=2,
                           font=fn, command=lambda: self._on_side(2)).pack(side="left", padx=12)
        r += 1

        # -- 第一步设置 (题目4) --
        ctk.CTkLabel(R, text="第一步 (题目4)", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        ff = ctk.CTkFrame(R, fg_color="transparent"); ff.grid(row=r, column=0, sticky="ew", padx=3)
        ctk.CTkLabel(ff, text="走格子:", font=fn).pack(side="left", padx=5)
        self.first_var = ctk.IntVar(value=5)
        self.first_combo = ctk.CTkOptionMenu(ff, values=[str(i) for i in range(1,10)],
                                              variable=self.first_var, width=60,
                                              command=lambda v: self.engine.set_first_move(int(v)))
        self.first_combo.pack(side="left", padx=5)
        self.first_combo.set("5")
        r += 1

        # -- 已选目标 --
        ctk.CTkLabel(R, text="已选目标格子", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        self.target_label = ctk.CTkLabel(R, text="(未选择 / 点击棋盘)",
                                         font=F(size=15, weight="bold"),
                                         fg_color="#1a2a3a", corner_radius=8, height=36)
        self.target_label.grid(row=r, column=0, sticky="ew", padx=6, pady=3); r += 1

        # -- 操作 --
        ctk.CTkLabel(R, text="操作", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        bf = ctk.CTkFrame(R, fg_color="transparent"); bf.grid(row=r, column=0, sticky="ew", padx=3)
        self.btn_confirm = ctk.CTkButton(bf, text="确认落子", height=42,
                                         font=F(size=16, weight="bold"),
                                         command=self._on_confirm)
        self.btn_confirm.pack(side="left", padx=2, fill="x", expand=True)
        ctk.CTkButton(bf, text="重检", width=55, height=42, fg_color="#555",
                      command=self._on_reset).pack(side="left", padx=2)
        ctk.CTkButton(bf, text="清空", width=55, height=42, fg_color="#555",
                      command=self._on_clear).pack(side="left", padx=2)
        ctk.CTkButton(bf, text="标定", width=55, height=42, fg_color="#2a4a2a",
                      command=self._on_calibrate).pack(side="left", padx=2)
        r += 1

        # -- 计时器 --
        self.timer_label = ctk.CTkLabel(R, text="用时: 0.0s / 15s",
                                        font=F(size=18, weight="bold"),
                                        fg_color="#2a1a1a", corner_radius=8, height=40)
        self.timer_label.grid(row=r, column=0, sticky="ew", padx=6, pady=(5,2)); r += 1

        # -- 胜负/状态 --
        self.win_label = ctk.CTkLabel(R, text="", font=F(size=18, weight="bold"),
                                      fg_color="#1a2a1a", corner_radius=10, height=50)
        self.win_label.grid(row=r, column=0, sticky="ew", padx=6, pady=3); r += 1

        # -- 回合+棋盘统计 --
        self.round_label = ctk.CTkLabel(R, text="", font=F(size=13))
        self.round_label.grid(row=r, column=0, sticky="w", padx=8, pady=2); r += 1

        # -- 机械臂调试: 手动输入脉冲坐标测试位置 --
        ctk.CTkLabel(R, text="机械臂测试 (脉冲坐标)", font=fb).grid(row=r, column=0, sticky="w", **P); r += 1
        df = ctk.CTkFrame(R, fg_color="transparent"); df.grid(row=r, column=0, sticky="ew", padx=3, pady=2); r += 1
        ctk.CTkLabel(df, text="X:", font=fn).pack(side="left", padx=3)
        self.test_x = ctk.CTkEntry(df, width=55, height=28, font=fn)
        self.test_x.pack(side="left", padx=2)
        self.test_x.insert(0, "1175")
        ctk.CTkLabel(df, text="Y:", font=fn).pack(side="left", padx=3)
        self.test_y = ctk.CTkEntry(df, width=55, height=28, font=fn)
        self.test_y.pack(side="left", padx=2)
        self.test_y.insert(0, "675")
        ctk.CTkButton(df, text="测试落子", width=72, height=28, font=F(size=12),
                      fg_color="#2a5a2a", command=self._on_test_move).pack(side="left", padx=4)
        self.test_label = ctk.CTkLabel(df, text="", font=F(size=11), text_color="gray")
        self.test_label.pack(side="left", padx=5)

        # -- 作弊弹窗区域 --
        self.cheat_frame = ctk.CTkFrame(R, fg_color="#4a2020", corner_radius=8)
        self.cheat_label = ctk.CTkLabel(self.cheat_frame, text="", font=F(size=13), text_color="#ff6666")
        self.cheat_label.pack(side="left", padx=10, pady=8)
        self.cheat_btn = ctk.CTkButton(self.cheat_frame, text="恢复棋子", height=30,
                                        fg_color="#aa3333", command=self._on_restore_cheat)
        self.cheat_btn.pack(side="right", padx=10, pady=5)
        # 默认隐藏
        r += 1

        # -- 调试 --
        self.dbg_header = ctk.CTkButton(R, text="调试参数", height=26,
                                        fg_color="transparent", text_color="gray",
                                        command=self._toggle_debug)
        self.dbg_header.grid(row=r, column=0, sticky="w", padx=6, pady=(8,2)); r += 1
        self.dbg_frame = ctk.CTkFrame(R, fg_color="transparent")
        self.dbg_visible = True
        self._build_debug(self.dbg_frame)
        self.dbg_frame.grid(row=r, column=0, sticky="ew", padx=3, pady=3)

    def _build_debug(self, P):
        P.grid_columnconfigure(1, weight=1); rr = 0
        for key, label, lo, hi in [
            ("canny_low","Canny Low",5,250),("canny_high","Canny High",10,255),
            ("area_min","Area Min",500,50000),("area_max","Area Max",1000,200000),
            ("min_rectangularity","Rect度",30,95),("yolo_conf","YOLO Conf",10,90)]:
            ctk.CTkLabel(P, text=label, font=ctk.CTkFont(size=11), width=65).grid(row=rr, column=0, sticky="w")
            cur = int(params[key]*100) if key in ("min_rectangularity","yolo_conf") else int(params[key])
            s = ctk.CTkSlider(P, from_=lo, to=hi, height=14, number_of_steps=hi-lo,
                              command=lambda v,k=key: self._on_slider(k,v))
            s.set(cur); s.grid(row=rr, column=1, sticky="ew", padx=5, pady=2)
            vl = ctk.CTkLabel(P, text=str(params[key]), width=36, font=ctk.CTkFont(size=10))
            vl.grid(row=rr, column=2, sticky="e")
            setattr(self, f"_l_{key}", vl); setattr(self, f"_s_{key}", s); rr += 1
        bf = ctk.CTkFrame(P, fg_color="transparent")
        bf.grid(row=rr, column=0, columnspan=3, sticky="ew", pady=3)
        ctk.CTkButton(bf, text="保存阈值", height=24, command=self._save_params).pack(side="left", padx=3)
        ctk.CTkButton(bf, text="恢复默认", height=24, fg_color="#555", command=self._reset_params).pack(side="left", padx=3)

    def _toggle_debug(self):
        if self.dbg_visible:
            self.dbg_frame.grid_forget(); self.dbg_header.configure(text="调试参数")
        else:
            r = self.dbg_header.grid_info()["row"] + 1
            self.dbg_frame.grid(row=r, column=0, sticky="ew", padx=3, pady=3)
            self.dbg_header.configure(text="调试参数")
        self.dbg_visible = not self.dbg_visible

    # ==================================================================
    # 事件
    # ==================================================================
    def _on_task(self, t):
        self.engine.start_task(t, self.side_var.get())
        self.grid_selected.clear()
        self.target_label.configure(text="(未选择 / 点击棋盘)")
        self.win_label.configure(text="", fg_color="#1a2a1a")
        self.cheat_frame.grid_forget()
        # 题目8: 双人对弈, 按钮文字随当前玩家变化
        if t == 8:
            self.btn_confirm.configure(text="确认落子 (黑方)")
        else:
            self.btn_confirm.configure(text="确认落子")
        self._update_status()

    def _on_side(self, s): self.engine.side = s

    def _on_board_click(self, idx):
        """点击棋盘格子切换选中。UI、视觉和游戏逻辑均使用同一编号。"""
        print(f"[CLICK] 格子={idx+1}, task={self.engine.task}")
        if self.engine.task == 8:
            # 双人对弈: 单选模式, 点击不同格子替换选中
            if self.grid_selected and self.grid_selected[0] == idx:
                self.grid_selected.clear()
            else:
                self.grid_selected = [idx]
        elif idx in self.grid_selected:
            self.grid_selected.remove(idx)
        else:
            self.grid_selected.append(idx)
        self.engine.set_target_grids([g+1 for g in self.grid_selected])
        s = ", ".join(str(g+1) for g in sorted(self.grid_selected))
        self.target_label.configure(text=s if s else "(未选择 / 点击棋盘)")

    def _on_clear(self):
        self.grid_selected.clear()
        self.target_label.configure(text="(未选择 / 点击棋盘)")
        self.engine.set_target_grids([])

    def _on_confirm(self):
        """确认落子按钮"""
        eng = self.engine

        # --- 题目(8): 双人对弈 ---
        if eng.task == 8:
            if not self.grid_selected:
                self.win_label.configure(text="请先在棋盘上选择落子位置!", fg_color="#4a3a1a")
                return
            grid_num = self.grid_selected[0] + 1
            player = "黑方" if eng.step_count % 2 == 0 else "白方"
            print(f"[UI] {player} 落子格子{grid_num}")
            result = eng.execute_human_move(grid_num)
            self.grid_selected = []
            self.target_label.configure(text="(未选择)")
            self._update_status()
            if eng.status == "done":
                self._show_winner()
                self.btn_confirm.configure(text="确认落子")
            else:
                next_player = "黑方" if eng.step_count % 2 == 0 else "白方"
                self.btn_confirm.configure(text=f"确认落子 ({next_player})")
            return

        # --- 题目(1)-(6) ---
        # (2)(3) 批量: 检查是否选了4个格子
        if eng.task in (1, 2, 3):
            if not self.grid_selected:
                self.win_label.configure(text="请先在棋盘上选择目标格子!", fg_color="#4a3a1a")
                return
            eng.status = "ai_thinking"

        eng.human_confirmed()
        self._update_status()

        if eng.status == "ai_thinking":
            self._do_ai_move()

        if eng.status == "cheat_detected":
            self._show_cheat()

        if eng.status == "done":
            self._show_winner()

    def _do_ai_move(self):
        """AI走一步(批量模式会自动循环直到完成或等待)"""
        eng = self.engine
        result = eng.ai_make_move()
        if result is None: return
        grid_num, is_done = result
        self._update_status()

        if eng.status == "batch_next":
            # 批量模式: 继续走下一步
            self._do_ai_move()
        elif eng.status == "done" or is_done:
            self._show_winner()
        # 否则等待人

    def _show_winner(self):
        w = self.engine.winner
        if self.engine.task == 8:
            if w == 0:
                self.win_label.configure(text="平局!", fg_color="#3a3a1a")
            elif w == 1:
                self.win_label.configure(text="黑方 获胜!", fg_color="#1a3a1a")
            elif w == 2:
                self.win_label.configure(text="白方 获胜!", fg_color="#3a3a1a")
            return
        if w == 0:
            self.win_label.configure(text="平局!", fg_color="#3a3a1a")
        elif w == self.engine.side:
            self.win_label.configure(text="系统(Fairy) 获胜!", fg_color="#1a3a1a")
        elif w != -1:
            self.win_label.configure(text="对手 获胜!", fg_color="#3a1a1a")

    def _show_cheat(self):
        eng = self.engine
        n = len(eng.cheated_pieces)
        g = ", ".join(str(m["row"]*3+m["col"]+1) for m in eng.cheated_pieces)
        self.cheat_label.configure(text=f"检测到作弊! {n}颗棋子被移动 (格子{g})")
        r = self.dbg_header.grid_info()["row"] + (1 if self.dbg_visible else 0)
        self.cheat_frame.grid(row=r, column=0, sticky="ew", padx=6, pady=3)

    def _on_restore_cheat(self):
        restored = self.engine.restore_all_cheated()
        self.cheat_frame.grid_forget()
        self.win_label.configure(text=f"已恢复棋子到格子: {restored}", fg_color="#2a3a1a")
        self._update_status()

    def _on_calibrate(self):
        """手动标定: 棋盘放标准位后点此按钮"""
        if self.squares_center:
            self.engine.serial.calibration.reset()
            self.engine.serial.calibration.calibrate(self.squares_center)
            self.win_label.configure(text="标定完成!", fg_color="#2a3a1a")
        else:
            self.win_label.configure(text="未检测到棋盘, 标定失败", fg_color="#3a1a1a")

    def _on_test_move(self):
        """机械臂测试: 手动输入脉冲坐标, 从黑棋区取棋放到指定位置"""
        try:
            x = int(self.test_x.get())
            y = int(self.test_y.get())
        except ValueError:
            self.test_label.configure(text="请输入有效数字")
            return
        from lzjpy.serial_client import build_frame, CMD_PLACE_PIECE, Z_DEFAULT, BLACK_PIECES
        frame = build_frame(cmd=CMD_PLACE_PIECE,
                            src_x=BLACK_PIECES[0][0], src_y=BLACK_PIECES[0][1], src_z=Z_DEFAULT,
                            dst_x=x, dst_y=y, dst_z=Z_DEFAULT)
        if self.engine.serial.enabled:
            self.engine.serial.serial.write(frame)
            self.test_label.configure(text=f"已发送 ({x},{y})")
        else:
            self.test_label.configure(text=f"模拟 ({x},{y}), 串口未连")
            print(f"[TEST] 模拟: ({x},{y})")

    def _on_reset(self):
        self.squares_center = None; self.board_box = None
        self.pieces = [0]*9; self._on_clear()

    def _on_slider(self, key, value):
        global params
        if key in ("min_rectangularity","yolo_conf"): params[key] = round(value/100.0, 2)
        else: params[key] = int(value)
        lbl = getattr(self, f"_l_{key}", None)
        if lbl: lbl.configure(text=str(params[key]))

    def _reset_params(self):
        global params
        params.update(DEFAULT_PARAMS)
        for key, val in DEFAULT_PARAMS.items():
            s = getattr(self, f"_s_{key}", None)
            l = getattr(self, f"_l_{key}", None)
            if s: s.set(int(val*100) if key in ("min_rectangularity","yolo_conf") else val)
            if l: l.configure(text=str(val))

    def _update_status(self):
        eng = self.engine
        # 计时器
        t = eng.get_elapsed()
        self.timer_label.configure(text=f"用时: {t:.1f}s / 15s",
                                   fg_color="#3a1a1a" if t > 12 else "#2a1a1a")
        # 回合
        b, w, e = eng.get_counts()
        if eng.task == 8:
            current = "黑方" if eng.step_count % 2 == 0 else "白方"
            self.round_label.configure(
                text=f"双人对弈 | 当前: {current} | 黑:{b} 白:{w}")
        else:
            sd = "黑" if eng.side == 1 else "白"
            turn_text = "系统" if eng.turn == "system" else "对手"
            self.round_label.configure(
                text=f"回合: {turn_text} | 黑:{b} 白:{w} | 系统执{sd} | 题{eng.task}")
        # 胜负
        if eng.status == "done":
            self._show_winner()

    # ==================================================================
    # 视频循环
    # ==================================================================
    def _start_video(self): self._update_video()

    def _update_video(self):
        if not self.running: return
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 标定参数对应摄像头原始的 640x480 图像，必须先去畸变。
                if self.use_undistort:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                # 摄像头相对棋盘转了 90 度且图像左右镜像；后续检测和所有
                # UI 棋盘均只使用这一个已校正的坐标系。
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                cl, ch = int(params["canny_low"]), int(params["canny_high"])
                amin, amax = int(params["area_min"]), int(params["area_max"])
                result = detect_chessboard(frame, canny_low=cl, canny_high=ch,
                                           area_min=amin, area_max=amax)
                self.squares_center, self.inner_radius, self.board_box, _, _ = result
                self.pieces = [0]*9; self.detections = []
                if self.squares_center:
                    try: self.pieces, self.detections = self._yolo_detect(frame)
                    except: pass

                    # 每帧更新脉冲坐标 (需手动点"标定"启用)
                    cal = self.engine.serial.calibration
                    if cal.is_ready():
                        cal.update_grid_positions(self.squares_center)

                # 更新棋盘状态 (不做自动检测，人通过按钮确认)
                if self.engine.status not in ("idle",):
                    self.engine.process_vision_result(self.pieces)

                # 裁剪棋盘 (pieces 按视觉顺序, 画面也按视觉显示)
                hl = self.grid_selected[-1] if self.grid_selected else -1
                ci = draw_crop_board(frame, self.board_box, self.pieces, highlight=hl, size=300)
                self.crop_board.update(ci if ci is not None else frame)

                # 虚拟棋盘: 显示引擎内部棋盘 (物理顺序)
                self.virtual_board.update(draw_virtual_board(
                    self.engine.get_board_1d(), highlight=hl, size=180))

                # 摄像头预览(等比)
                pw, ph = self.cam_preview.winfo_width(), self.cam_preview.winfo_height()
                if pw > 50 and ph > 50:
                    fh, fw = frame.shape[:2]
                    s = min((pw-4)/fw, (ph-4)/fh)
                    # 棋盘坐标基于原始 frame，先绘制再缩放，避免标记错位。
                    pv2 = draw_board_overlay(frame, self.squares_center,
                                             self.inner_radius, self.board_box, self.pieces, 0)
                    pv2 = cv2.resize(pv2, (int(fw*s), int(fh*s)))
                    ci2 = cv2_to_ctk(pv2, int(fw*s), int(fh*s))
                    if ci2: self.cam_preview.configure(image=ci2, text="")

                self._update_status()
        self.after(33, self._update_video)

    def _yolo_detect(self, frame):
        global _yolo_detect
        if _yolo_detect is None:
            from lzjpy.chess_detection import detect_pieces_yolo
            _yolo_detect = detect_pieces_yolo
        return _yolo_detect(frame, self.squares_center, self.inner_radius,
                           conf=params.get("yolo_conf", 0.4))

    def on_close(self):
        self.running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = TicTacToeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
