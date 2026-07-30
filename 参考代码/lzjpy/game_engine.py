"""
游戏引擎 — 三子棋对弈核心逻辑 v2
- 题目(1)-(3): 批量目标格子 + 依次执行
- 题目(4)-(5): 对弈流程 + 胜负判定 + 第一步设置
- 题目(6): 作弊检测 + 自动恢复
"""

import time
from lzjpy.ai import find_best_move, check_win


class GameEngine:
    def __init__(self, serial_client=None):
        self.board = [[0] * 3 for _ in range(3)]
        self.prev_board = [[0] * 3 for _ in range(3)]
        self.side = 1            # 1=系统执黑, 2=系统执白
        self.task = 4
        self.turn = "system"     # system / human
        self.winner = -1         # -1=未结束, 0=平局, 1=黑赢, 2=白赢
        self.target_grids = []   # 题目(1)(2)(3)目标格子 [1,3,5,7]
        self.batch_idx = 0       # 批量执行时的当前步
        self.first_move = 5      # 题目(4)第一步走哪格 (默认中心)
        self.step_count = 0
        self.status = "idle"
        self.serial = serial_client
        self.move_times = []     # 每步耗时记录
        self.round_start = 0     # 当前回合开始时间
        self.cheated_pieces = [] # 被移走的棋子列表
        self.turn_start_board = [[0]*3 for _ in range(3)]  # 人回合开始时的棋盘快照

    # ==================================================================
    # 重置
    # ==================================================================
    def reset(self):
        self.board = [[0] * 3 for _ in range(3)]
        self.prev_board = [[0] * 3 for _ in range(3)]
        self.winner = -1
        self.turn = "system"
        self.step_count = 0
        self.batch_idx = 0
        self.status = "idle"
        self.cheated_pieces = []
        self.move_times = []
        self.round_start = 0
        if self.serial:
            self.serial.black_idx = 0
            self.serial.white_idx = 0

    # ==================================================================
    # 题目启动
    # ==================================================================
    def start_task(self, task_num, side=1):
        self.reset()
        self.task = task_num
        self.side = side
        self.status = "playing"
        self.round_start = time.time()

        if task_num in (1, 2, 3):
            self.turn = "system"
            self.status = "batch_ready"  # 等待选格子+确认
        elif task_num == 4:
            self.turn = "system"
            self.status = "playing_first"  # 系统先走第一步
        elif task_num == 5:
            self.turn = "human"
            self.status = "waiting_human"
            self.turn_start_board = [row[:] for row in self.board]
        elif task_num == 6:
            self.turn = "human"
            self.status = "waiting_human"
            self.turn_start_board = [row[:] for row in self.board]
        elif task_num == 8:
            # 双人对弈: 黑方先行, 轮流点击格子, 机器执行落子
            self.status = "waiting_human"
            self.turn_start_board = [row[:] for row in self.board]

        return self.get_status_text()

    def set_target_grids(self, grids):
        self.target_grids = sorted(grids)
        self.batch_idx = 0

    def set_first_move(self, grid_num):
        self.first_move = grid_num

    # ==================================================================
    # 棋盘更新 + 自动检测人落子
    # ==================================================================
    def process_vision_result(self, pieces_1d):
        """视觉检测结果按 UI 的标准 1-9 顺序更新棋盘。"""
        self.prev_board = [row[:] for row in self.board]
        for i in range(3):
            for j in range(3):
                self.board[i][j] = pieces_1d[i * 3 + j]

    def check_human_moved(self):
        """
        自动检测人是否落子了.
        条件: 等待人中 + 棋盘多了一颗对方棋子.
        返回: 检测到的格子号(1-9) 或 None
        """
        if self.status != "waiting_human":
            return None

        opponent = 3 - self.side  # 系统执黑→对方白棋(2), 系统执白→对方黑棋(1)

        # 统计前后变化
        changes = []
        for i in range(3):
            for j in range(3):
                old = self.prev_board[i][j]
                new = self.board[i][j]
                if old != new:
                    changes.append((i, j, old, new))

        # 多了1颗对方棋子 → 人落子了
        new_opponent = [(r, c) for r, c, o, n in changes
                        if o == 0 and n == opponent]
        if len(new_opponent) == 1:
            r, c = new_opponent[0]
            return r * 3 + c + 1  # 返回格子号 1-9
        return None

    # ==================================================================
    # 人确认后的处理
    # ==================================================================
    def human_confirmed(self):
        """人按确认按钮 → AI走棋"""
        if self.status == "done":
            return self.get_status_text()

        self.round_start = time.time()

        # 作弊检测 (题目6)
        if self.task == 6:
            moved = self.detect_cheat()
            if moved:
                self.cheated_pieces = moved
                self.status = "cheat_detected"
                return self.get_status_text()

        # 检查人是否赢了
        self.winner = check_win(self.board)
        if self.winner != -1:
            self.status = "done"
            return self.get_status_text()

        # 检查平局
        if all(self.board[i][j] != 0 for i in range(3) for j in range(3)):
            self.winner = 0
            self.status = "done"
            return self.get_status_text()

        self.turn = "system"
        self.status = "ai_thinking"
        return self.get_status_text()

    # ==================================================================
    # AI走棋
    # ==================================================================
    def ai_make_move(self):
        """AI走一步棋, 返回 (grid_num, is_last)"""
        if self.status not in ("ai_thinking", "batch_ready"):
            return None

        # --- (1)(2)(3) 批量目标格子 ---
        if self.task in (1, 2, 3):
            if self.batch_idx < len(self.target_grids):
                pos = self.target_grids[self.batch_idx]
                row, col = (pos - 1) // 3, (pos - 1) % 3
                self.batch_idx += 1
            else:
                self.status = "done"
                return None
        else:
            # --- (4)(5) Minimax对弈 ---
            # 题目4第一步走预设位置
            if self.task == 4 and self.step_count == 0 and self.turn == "system":
                row, col = (self.first_move - 1) // 3, (self.first_move - 1) % 3
            else:
                move = find_best_move(self.board, self.side)
                if move is None:
                    self.status = "done"
                    return None
                row, col = move

        # 落子
        piece_color = 1 if self.step_count % 2 == 0 else 2
        if self.task in (4, 5):
            piece_color = self.side
        self.board[row][col] = piece_color
        self.step_count += 1
        grid_num = row * 3 + col + 1

        # 串口发送
        piece_zone = 0 if piece_color == 1 else 1
        if self.serial:
            self.serial.send_piece_command(piece_zone, grid_num)
        else:
            print(f"[ENGINE] 落子: 格子{grid_num} 颜色={piece_color}")

        elapsed = time.time() - self.round_start
        self.move_times.append(elapsed)

        # 检查胜负
        self.winner = check_win(self.board)
        if self.winner != -1:
            self.status = "done"
            is_last = True
        elif all(self.board[i][j] != 0 for i in range(3) for j in range(3)):
            self.winner = 0
            self.status = "done"
            is_last = True
        else:
            # 批量模式: 还有格子要走吗?
            if self.task in (1, 2, 3):
                if self.batch_idx < len(self.target_grids):
                    self.status = "batch_next"  # 继续下一格
                    is_last = False
                else:
                    self.status = "done"
                    is_last = True
            else:
                # 对弈: 切换回合
                self.turn = "human"
                self.status = "waiting_human"
                self.turn_start_board = [row[:] for row in self.board]
                is_last = False

        self.round_start = time.time()
        return grid_num, (is_last or self.status == "done")

    # ==================================================================
    # 题目(8): 双人对弈 — 人点格子, 机器执行
    # ==================================================================
    def execute_human_move(self, grid_num):
        """
        执行人类玩家选择的落子位置.
        黑先白后交替, 发送串口指令让机械臂执行.
        返回: (grid_num, is_done)
        """
        row, col = (grid_num - 1) // 3, (grid_num - 1) % 3

        # 黑先白后交替
        piece_color = 1 if self.step_count % 2 == 0 else 2
        self.board[row][col] = piece_color
        self.step_count += 1
        elapsed = time.time() - self.round_start
        self.move_times.append(elapsed)

        # 串口发送
        piece_zone = 0 if piece_color == 1 else 1
        if self.serial:
            self.serial.send_piece_command(piece_zone, grid_num)
        else:
            pname = "黑" if piece_color == 1 else "白"
            print(f"[ENGINE] 双人对弈: {pname}方 → 格子{grid_num}")

        # 检查胜负
        self.winner = check_win(self.board)
        if self.winner != -1:
            self.status = "done"
            is_last = True
        elif all(self.board[i][j] != 0 for i in range(3) for j in range(3)):
            self.winner = 0
            self.status = "done"
            is_last = True
        else:
            self.status = "waiting_human"  # 等下一个玩家
            self.turn_start_board = [row[:] for row in self.board]
            is_last = False

        self.round_start = time.time()
        return grid_num, (is_last or self.status == "done")

    # ==================================================================
    # 作弊检测
    # ==================================================================
    def detect_cheat(self):
        """
        检测系统棋子是否被移动. 对比人回合开始时的快照.
        返回: [{row, col, old, new}, ...] 或 []
        """
        moved = []
        for i in range(3):
            for j in range(3):
                before = self.turn_start_board[i][j]
                now = self.board[i][j]
                if before != now:
                    # 系统棋子被移走: 原来有 现在没了/变了
                    if before == self.side and now != self.side:
                        moved.append({"row": i, "col": j, "old": before, "new": now})
        return moved

    def restore_cheated_piece(self, row, col):
        """恢复被移动的棋子"""
        self.board[row][col] = self.side
        grid_num = row * 3 + col + 1
        piece_zone = 0 if self.side == 1 else 1
        if self.serial:
            self.serial.send_piece_command(piece_zone, grid_num)
        return grid_num

    def restore_all_cheated(self):
        """恢复所有被作弊移动的棋子, 返回恢复的格子列表"""
        restored = []
        for m in self.cheated_pieces:
            g = self.restore_cheated_piece(m["row"], m["col"])
            restored.append(g)
        self.cheated_pieces = []
        self.turn = "human"
        self.status = "waiting_human"
        self.turn_start_board = [row[:] for row in self.board]
        return restored

    def get_elapsed(self):
        """当前回合已用时间(秒)"""
        if self.round_start == 0:
            return 0
        return time.time() - self.round_start

    # ==================================================================
    # 状态查询
    # ==================================================================
    def get_board_1d(self):
        return [self.board[i][j] for i in range(3) for j in range(3)]

    def get_status_text(self):
        if self.status == "idle":
            return "就绪，选择题目开始"
        if self.status == "done":
            if self.winner == 0:
                return "平局!"
            if self.winner == -1:
                return "完成"
            win_name = "系统(Fairy)" if self.winner == self.side else "对手"
            return f"{win_name} 获胜!"
        if self.status == "cheat_detected":
            return f"检测到作弊! {len(self.cheated_pieces)}颗棋子被移动"
        if self.status == "ai_thinking":
            return "Fairy 思考中..."
        if self.status == "waiting_human":
            return "等待对手落子..."
        if self.status == "playing_first":
            return "系统先走第一步..."
        if self.status == "batch_ready":
            return f"题目{self.task}: 请选目标格子并确认"
        if self.status == "batch_next":
            return f"题目{self.task}: 继续下一步 ({self.batch_idx+1}/{len(self.target_grids)})"
        if self.status == "playing":
            return f"题目{self.task} 进行中..."
        return "..."

    def get_counts(self):
        b = sum(cell == 1 for row in self.board for cell in row)
        w = sum(cell == 2 for row in self.board for cell in row)
        return b, w, 9 - b - w
