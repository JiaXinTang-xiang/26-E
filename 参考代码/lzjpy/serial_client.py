"""
串口通信模块 — 匹配 STM32 17字节协议

帧格式 (17 bytes):
  [0]     0xAA        帧头
  [1]     0x0F        长度
  [2]     CMD         命令字 (0xA1 = 放棋)
  [3-4]   src_x       uint16, 高字节在前
  [5-6]   src_y       uint16
  [7-8]   src_z       uint16
  [9-10]  dst_x       uint16
  [11-12] dst_y       uint16
  [13-14] dst_z       uint16
  [15]    xor8        校验 (byte[1]~byte[14] 逐字节异或)
  [16]    0x55        帧尾
"""

import struct
import time
import cv2
import numpy as np

# =====================================================================
# 坐标映射表 (来自 STM32 StepMotor.h)
# 机械臂放棋/取棋的固定脉冲位置
# =====================================================================

# 九宫格: 格子号(1-9) → 脉冲 (X, Y), 匹配 STM32 Chessboard[9][2]
CHESS_GRID = {
    1: (780, 433),
    2: (1090, 433),
    3: (1400, 433),
    4: (780, 664),
    5: (1090, 664),
    6: (1400, 664),
    7: (780, 895),
    8: (1090, 895),
    9: (1400, 895),
}

# 棋子放置区: 所有取棋脉冲坐标 (Chessplace[10])
# 黑棋左侧[0]~[4], 白棋右侧[5]~[9], 按 Y 递增排列
BLACK_PIECES = [(230, 330), (230, 503), (230, 675), (230, 847), (230, 1020)]
WHITE_PIECES = [(2030, 330), (2030, 503), (2030, 675), (2030, 847), (2030, 1020)]

Z_DEFAULT = 0

# UI、视觉和机械臂共用左上到右下的标准格子编号。图像方向在 ui_app 中统一校正。
GRID_REMAP = {1:1, 2:4, 3:7, 4:2, 5:5, 6:8, 7:3, 8:6, 9:9}

# =====================================================================
# 协议常量
# =====================================================================
FRAME_HEADER = 0xAA
FRAME_FOOTER = 0x55
FRAME_LENGTH = 0x0F
CMD_PLACE_PIECE = 0xA1
FRAME_SIZE = 17


def _xor_checksum(data: bytes) -> int:
    """XOR 校验: 逐字节异或"""
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


def build_frame(cmd: int, src_x: int, src_y: int, src_z: int,
                dst_x: int, dst_y: int, dst_z: int) -> bytes:
    """构建 17 字节数据帧"""
    # payload: cmd(1B) + 6个坐标(12B) = 13字节 → 帧 byte[2]~byte[14]
    payload = struct.pack('>BHHHHHH',
                          cmd & 0xFF,
                          src_x & 0xFFFF, src_y & 0xFFFF, src_z & 0xFFFF,
                          dst_x & 0xFFFF, dst_y & 0xFFFF, dst_z & 0xFFFF)

    # 校验段: 长度(1B) + payload(13B) = byte[1]~byte[14]
    check_data = struct.pack('>B', FRAME_LENGTH) + payload
    checksum = _xor_checksum(check_data)

    frame = struct.pack('>BB', FRAME_HEADER, FRAME_LENGTH) + payload \
            + struct.pack('>BB', checksum, FRAME_FOOTER)

    assert len(frame) == FRAME_SIZE, f"帧长度错误: {len(frame)}"
    return frame


# =====================================================================
# 标定: 摄像头像素坐标 → 机械臂脉冲坐标
# =====================================================================
class Calibration:
    """
    标定: 棋盘放标准位, 记录 9 个格子的像素坐标, 计算 H(像素→脉冲).
    运行时: H 不变, 每帧输入当前像素, 输出实时脉冲跟随棋盘移动.
    """

    def __init__(self):
        self.H = None            # 3×3 单应矩阵 (像素 → 脉冲), 标定时算一次
        self.grid_pulses = {}    # 格子号(1-9) → (pulse_x, pulse_y) 实时更新

    def is_ready(self):
        return self.H is not None

    def calibrate(self, squares_center):
        """
        标定(做一次): 棋盘放标准位, 用此刻的 9 格像素坐标和 CHESS_GRID 算 H.
        之后 H 不变, 靠输入像素不同来跟踪棋盘移动.
        """
        src_pts = []  # 标准位像素
        dst_pts = []  # CHESS_GRID 脉冲
        for (px, py), grid_num in squares_center:
            if grid_num in CHESS_GRID:
                src_pts.append([px, py])
                dst_pts.append(list(CHESS_GRID[grid_num]))

        if len(src_pts) < 4:
            print("[CALIB] 检测到的格子不足4个, 标定失败")
            return

        self.H, _ = cv2.findHomography(
            np.array(src_pts, dtype=np.float32),
            np.array(dst_pts, dtype=np.float32))
        # 立即更新一次当前脉冲坐标
        self._transform(squares_center)
        print(f"[CALIB] 标定完成, H 已锁定, 使用 {len(src_pts)} 个点")

    def update_grid_positions(self, squares_center):
        """每帧: 当前像素 × H → 实时脉冲坐标"""
        self._transform(squares_center)

    def _transform(self, squares_center):
        if self.H is None:
            return
        for (px, py), grid_num in squares_center:
            pt = np.array([[[px, py]]], dtype=np.float32)
            result = cv2.perspectiveTransform(pt, self.H)
            self.grid_pulses[grid_num] = (int(result[0][0][0]), int(result[0][0][1]))

    def get_grid_pulse(self, grid_num):
        """获取实时脉冲坐标, 未标定则回退到 CHESS_GRID"""
        if self.is_ready() and grid_num in self.grid_pulses:
            return self.grid_pulses[grid_num]
        return CHESS_GRID.get(grid_num, (0, 0))

    def reset(self):
        self.H = None
        self.grid_pulses.clear()


# =====================================================================
# SerialClient
# =====================================================================
class SerialClient:
    """串口通信客户端"""

    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.enabled = False
        self.calibration = Calibration()
        self.black_idx = 0   # 下一个取的黑棋编号
        self.white_idx = 0   # 下一个取的白棋编号
        if self.port:
            self.connect()

    def connect(self, port=None, baudrate=None):
        """连接串口"""
        if port:
            self.port = port
        if baudrate:
            self.baudrate = baudrate

        if not self.port:
            print("[SERIAL] 未配置串口")
            return False

        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.enabled = True
            print(f"[SERIAL] {self.port} @ {self.baudrate} 已连接")
            return True
        except Exception as e:
            print(f"[SERIAL] 连接失败: {e}")
            self.enabled = False
            return False

    def disconnect(self):
        """断开串口"""
        if self.serial:
            self.serial.close()
            self.serial = None
        self.enabled = False

    def send_piece_command(self, from_pos, to_pos):
        """
        发送落子指令到机械臂. 每次取棋自动使用下一个空闲棋子位置.
        参数:
            from_pos: 棋子区 0=黑 1=白
            to_pos:   目标格子 1-9
        """
        to_pos = GRID_REMAP.get(to_pos, to_pos)  # 视觉→物理
        if from_pos == 0:
            idx = self.black_idx % len(BLACK_PIECES)
            src_x, src_y = BLACK_PIECES[idx]
            self.black_idx += 1
        else:
            idx = self.white_idx % len(WHITE_PIECES)
            src_x, src_y = WHITE_PIECES[idx]
            self.white_idx += 1

        dst_x, dst_y = self.calibration.get_grid_pulse(to_pos)

        frame = build_frame(cmd=CMD_PLACE_PIECE,
                            src_x=src_x, src_y=src_y, src_z=Z_DEFAULT,
                            dst_x=dst_x, dst_y=dst_y, dst_z=Z_DEFAULT)

        if not self.enabled or self.serial is None:
            from_zone = "黑棋区" if from_pos == 0 else "白棋区"
            print(f"[SERIAL] 模拟发送: {from_zone}#{idx}({src_x},{src_y}) → 格子{to_pos}({dst_x},{dst_y})")
            print(f"        帧数据: {frame.hex(' ').upper()}")
            return True

        try:
            self.serial.write(frame)
            from_zone = "黑棋区" if from_pos == 0 else "白棋区"
            print(f"[SERIAL] 已发送: {from_zone}#{idx}({src_x},{src_y}) → 格子{to_pos}({dst_x},{dst_y})")
            return True
        except Exception as e:
            print(f"[SERIAL] 发送失败: {e}")
            return False

    def read_line(self, timeout=0.1):
        """读取一行串口数据"""
        if not self.enabled or self.serial is None:
            return None
        try:
            self.serial.timeout = timeout
            return self.serial.readline()
        except Exception:
            return None
