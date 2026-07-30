#!/usr/bin/env python3
"""
三子棋 - 棋盘实时检测
- 相机标定: camera_calibration.yaml
- 棋盘检测: 得分制 approxPolyDP + minAreaRect 回退
- 角点排序: 极角法 + Y坐标分组二次校验
- 参数集中管理 + 键盘实时调参 + 安全窗口管理

Usage:
    python chess_detection.py               # 正常运行
    python chess_detection.py --debug       # 调试模式
    python chess_detection.py --cam 8       # 旭日派
"""

import cv2
import numpy as np
import argparse
import os
import re
import time

# =============================================================================
# 参数集中管理
# =============================================================================

# --- 图像预处理 ---
GAUSSIAN_BLUR_SIZE = 7
CANNY_LOWER = 50
CANNY_UPPER = 150

# --- 调试开关 ---
DEBUG_SHOW_EDGES = False
DEBUG_SHOW_CANDIDATES = False

# --- 棋盘检测 ---
MIN_CONTOUR_AREA = 7000
MAX_CONTOUR_AREA = 15000
ASPECT_MIN = 0.4
ASPECT_MAX = 1.8
MIN_RECTANGULARITY = 0.65
MIN_CONTOUR_PIXELS = 300
PAPER_AREA_RATIO = 0.50

# =============================================================================
# 安全窗口管理
# =============================================================================
_windows_created = {}

# --- 鼠标调试状态 ---
_mouse_info = {'x': 0, 'y': 0, 'bgr': (0, 0, 0), 'hsv': (0, 0, 0), 'clicked': None}
_current_frame = None  # 鼠标回调用的帧缓存


def mouse_callback(event, x, y, flags, param):
    """鼠标回调: 移动显示像素值, 点击打印详情"""
    global _mouse_info, _current_frame
    _mouse_info['x'], _mouse_info['y'] = x, y

    if _current_frame is not None and 0 <= x < _current_frame.shape[1] and 0 <= y < _current_frame.shape[0]:
        bgr = _current_frame[y, x]
        _mouse_info['bgr'] = tuple(int(v) for v in bgr)
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
        _mouse_info['hsv'] = tuple(int(v) for v in hsv)

    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse_info['clicked'] = (x, y)
        print(f"\n[CLICK] ({x},{y}) BGR={_mouse_info['bgr']} HSV={_mouse_info['hsv']}")

def safe_show(name, image):
    _windows_created[name] = True
    cv2.imshow(name, image)

def safe_destroy(name):
    if _windows_created.pop(name, False):
        cv2.destroyWindow(name)

def safe_destroy_all_debug():
    for name in list(_windows_created.keys()):
        if name.startswith("Debug"):
            safe_destroy(name)

# =============================================================================
# 1. 相机标定
# =============================================================================
def load_calibration(yaml_path):
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        cm = np.array(data['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
        dc = np.array(data['distortion_coefficients']['data'], dtype=np.float32)
        pm = np.array(data['projection_matrix']['data'], dtype=np.float32).reshape(3, 4)
        return cm, dc, pm[:, :3]
    except Exception:
        pass
    try:
        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        cm = fs.getNode('camera_matrix').mat()
        dc = fs.getNode('distortion_coefficients').mat()
        if cm is not None and dc is not None:
            return cm, dc.reshape(-1), None
    except Exception:
        pass
    text = open(yaml_path).read()
    nums = lambda key: [float(x) for x in
                        re.search(rf'{key}.*?data:\s*\[([^\]]+)\]', text, re.DOTALL)
                        .group(1).replace('\n', '').split(',')]
    cm_data = nums('camera_matrix')
    cm = np.array(cm_data, dtype=np.float32).reshape(3, 3)
    try:
        dc = np.array(nums('distortion_coefficients'), dtype=np.float32)
    except Exception:
        dc = np.zeros(5, dtype=np.float32)
    try:
        pm = np.array(nums('projection_matrix'), dtype=np.float32).reshape(3, 4)[:, :3]
    except Exception:
        pm = None
    return cm, dc, pm


def init_undistort_maps(camera_matrix, dist_coeffs, width=640, height=480):
    # alpha=0: 只保留有效像素，无黑边，矫正幅度最温和
    # alpha=1: 保留全部像素（含黑边），矫正幅度最强
    # 不用 P 矩阵，直接用 getOptimalNewCameraMatrix 控制 alpha
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 0, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3),
        new_camera_matrix, (width, height), cv2.CV_16SC2)
    return map1, map2

# =============================================================================
# 2. 鲁棒角点排序 —— 极角法 + Y坐标校验
# =============================================================================
def sort_corners(corners):
    """四点排序: 左上 → 右上 → 右下 → 左下 (极角法)"""
    pts = corners.reshape(4, 2).astype(np.float32)
    centroid = np.mean(pts, axis=0)

    # 计算各点相对于重心的极角 [0, 2π)
    angles = [(np.arctan2(p[1] - centroid[1], p[0] - centroid[0]) + 2 * np.pi) % (2 * np.pi)
              for p in pts]
    sorted_idx = np.argsort(angles)

    # 找最接近 225° (5π/4) 的点作为左上角起点
    target = 5 * np.pi / 4
    best_start = 0
    best_diff = float('inf')
    for i, idx in enumerate(sorted_idx):
        diff = min(abs(angles[idx] - target), 2 * np.pi - abs(angles[idx] - target))
        if diff < best_diff:
            best_diff = diff
            best_start = i

    ordered = np.array([pts[sorted_idx[(best_start + i) % 4]] for i in range(4)])

    # Y坐标分组二次校验
    y_mean = np.mean(ordered[:, 1])
    top_mask = ordered[:, 1] <= y_mean
    bottom_mask = ordered[:, 1] > y_mean

    if np.sum(top_mask) >= 2 and np.sum(bottom_mask) >= 2:
        top = ordered[top_mask][np.argsort(ordered[top_mask][:, 0])]
        bottom = ordered[bottom_mask][np.argsort(ordered[bottom_mask][:, 0])]
        if len(top) == 2 and len(bottom) == 2:
            ordered = np.array([top[0], top[1], bottom[1], bottom[0]])

    return ordered.astype(np.int32)


# =============================================================================
# 3. 得分制四边形优化 —— 参考 optimize_quadrilateral
# =============================================================================
def optimize_quadrilateral(contour):
    """
    凸包 → 多 epsilon approxPolyDP 打分 → 选最佳 4 点
    回退: minAreaRect boxPoints
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    hull_area = float(cv2.contourArea(hull))

    best_box = None
    best_score = 0

    for eps_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08]:
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) != 4:
            continue

        approx_area = float(cv2.contourArea(approx))
        if hull_area > 0:
            area_ratio = approx_area / hull_area
            if 0.8 <= area_ratio <= 1.3:
                score = 1.0 - abs(1.0 - area_ratio)
                if score > best_score:
                    best_score = score
                    best_box = approx.reshape(4, 2).astype(np.float32)

    if best_box is not None:
        return best_box

    # 回退: minAreaRect
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect)


# =============================================================================
# 4. 棋盘检测
# =============================================================================
def detect_chessboard(frame, canny_low=None, canny_high=None,
                      area_min=None, area_max=None):
    if canny_low is None:
        canny_low = CANNY_LOWER
    if canny_high is None:
        canny_high = CANNY_UPPER
    if area_min is None:
        area_min = MIN_CONTOUR_AREA
    if area_max is None:
        area_max = MAX_CONTOUR_AREA

    h_img, w_img = int(frame.shape[0]), int(frame.shape[1])
    image_area = h_img * w_img

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None or hierarchy is None:
        return None, None, None, edges, []

    all_quads = []
    for i, cnt in enumerate(contours):
        area = float(cv2.contourArea(cnt))
        if area < MIN_CONTOUR_PIXELS:
            continue

        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        bw, bh = rect[1]
        rect_area = float(bw) * float(bh)
        if rect_area < area_min or rect_area > area_max:
            continue

        aspect = float(bw) / float(bh) if float(bh) > 0 else 0.0
        if not (ASPECT_MIN < aspect < ASPECT_MAX):
            continue

        rectangularity = float(cv2.contourArea(hull)) / rect_area if rect_area > 0 else 0
        if rectangularity < MIN_RECTANGULARITY:
            continue

        # 得分制获取最优 4 点
        box = optimize_quadrilateral(cnt)

        hi = hierarchy[0][i]
        has_parent = bool(hi[3] != -1)
        has_child = bool(hi[2] != -1)

        all_quads.append({
            'idx': int(i),
            'box': box,
            'area': rect_area,
            'aspect': aspect,
            'rectangularity': rectangularity,
            'has_parent': has_parent,
            'has_child': has_child,
            'parent_idx': int(hi[3]),
        })

    candidates = [np.intp(q['box']) for q in all_quads]

    if not all_quads:
        return None, None, None, edges, candidates

    # ---- 选择: 跳过纸张，优先内部棋盘 ----
    all_quads.sort(key=lambda x: x['area'], reverse=True)
    selected = None

    for q in all_quads:
        if q['area'] > image_area * PAPER_AREA_RATIO:
            continue
        if selected is None:
            selected = q
            if q['has_child']:
                break
        elif not selected['has_child'] and q['has_child']:
            selected = q
            break

    if selected is None and len(all_quads) >= 2:
        selected = all_quads[1]
    elif selected is None and len(all_quads) >= 1:
        selected = all_quads[0]

    if selected is None:
        return None, None, None, edges, candidates

    box = np.intp(selected['box'])
    box_sorted = sort_corners(box)
    squares_center, inner_radius = compute_squares_center(box_sorted)

    return squares_center, inner_radius, np.intp(box_sorted), edges, candidates


# =============================================================================
# 5. 九宫格计算
# =============================================================================
def compute_squares_center(board_box):
    """
    双线性插值计算9宫格中心点。
    不依赖旋转角度——直接用四个角点线性插值，
    无论棋盘怎么旋转、透视畸变，格子都在正确位置。
    """
    TL, TR, BR, BL = board_box.astype(np.float32)

    # 边长估算（用于内接圆半径）
    board_width = max(np.linalg.norm(TR - TL), np.linalg.norm(BR - BL))
    board_height = max(np.linalg.norm(BL - TL), np.linalg.norm(BR - TR))
    cell_size = min(board_width, board_height) / 3
    inner_radius = int(cell_size * 0.4)

    squares_center = []
    for row in range(3):
        for col in range(3):
            # u: 沿上边方向 (TL→TR) 的插值比例
            # v: 沿左边方向 (TL→BL) 的插值比例
            u = (col + 0.5) / 3  # 0.167, 0.5, 0.833
            v = (row + 0.5) / 3

            # 双线性：先沿上下边插出水平线两端，再在竖直方向插
            top_pt = TL + (TR - TL) * u
            bottom_pt = BL + (BR - BL) * u
            center = top_pt + (bottom_pt - top_pt) * v

            idx = row * 3 + col + 1
            squares_center.append(((center[0], center[1]), idx))

    return squares_center, inner_radius


# =============================================================================
# 6. 可视化
# =============================================================================
def draw_board_overlay(frame, squares_center, inner_radius, board_box=None):
    dbg = frame.copy()

    if board_box is not None:
        cv2.drawContours(dbg, [board_box], -1, (0, 255, 0), 3)
        # 四角圆点标记
        for pt in board_box:
            cv2.circle(dbg, tuple(pt), 6, (0, 255, 0), -1)

    if squares_center:
        tl = np.array(squares_center[0][0])
        tr = np.array(squares_center[2][0])
        bl = np.array(squares_center[6][0])
        br = np.array(squares_center[8][0])

        top_edge = tr - tl
        bottom_edge = br - bl
        left_edge = bl - tl
        right_edge = br - tr

        for frac in [1/3, 2/3]:
            pt1 = tuple((tl + left_edge * frac).astype(int))
            pt2 = tuple((tr + right_edge * frac).astype(int))
            cv2.line(dbg, pt1, pt2, (255, 0, 0), 2)
            pt1 = tuple((tl + top_edge * frac).astype(int))
            pt2 = tuple((bl + bottom_edge * frac).astype(int))
            cv2.line(dbg, pt1, pt2, (255, 0, 0), 2)

        for (cx, cy), num in squares_center:
            ix, iy = int(cx), int(cy)
            cv2.circle(dbg, (ix, iy), inner_radius, (255, 0, 0), 2)
            cv2.putText(dbg, str(num), (ix - 10, iy - inner_radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.drawMarker(dbg, (ix, iy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)

    return dbg


def draw_info_panel(img, board_box, inner_radius, candidates_count,
                    canny_low, canny_high, area_min, area_max):
    """在画面右侧绘制半透明信息面板"""
    h, w = img.shape[:2]
    panel_w = 220
    overlay = img.copy()

    # 半透明黑色背景
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(img, 0.4, overlay, 0.6, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 25
    dy = 22

    def row(label, value, color=(0, 255, 255)):
        nonlocal y
        cv2.putText(img, f"{label}: {value}", (w - panel_w + 8, y), font, 0.45, color, 1)
        y += dy

    def sep():
        nonlocal y
        cv2.line(img, (w - panel_w + 8, y), (w - 8, y), (80, 80, 80), 1)
        y += 8

    # 标题
    cv2.putText(img, "DEBUG INFO", (w - panel_w + 8, y), font, 0.55, (255, 255, 255), 1)
    y += 28

    sep()
    row("Canny Low", canny_low, (0, 200, 255))
    row("Canny High", canny_high, (0, 200, 255))

    sep()
    row("Area Min", area_min, (200, 200, 0))
    row("Area Max", area_max, (200, 200, 0))

    sep()
    if board_box is not None:
        area_val = int(cv2.contourArea(board_box))
        w_val = int(np.linalg.norm(board_box[1] - board_box[0]))
        h_val = int(np.linalg.norm(board_box[3] - board_box[0]))
        aspect_val = w_val / h_val if h_val > 0 else 0
        row("Board Area", area_val, (0, 255, 0))
        row("Board W x H", f"{w_val} x {h_val}", (0, 255, 0))
        row("Board Aspect", f"{aspect_val:.2f}", (0, 255, 0))
        row("Cell Radius", inner_radius, (0, 255, 0))
    else:
        row("Board", "NOT FOUND", (0, 0, 255))

    sep()
    row("Candidates", candidates_count, (255, 200, 0))

    sep()
    # 鼠标像素信息
    mx, my = _mouse_info['x'], _mouse_info['y']
    bgr = _mouse_info['bgr']
    hsv = _mouse_info['hsv']
    cv2.putText(img, f"Mouse: ({mx},{my})", (w - panel_w + 8, y), font, 0.45, (200, 200, 200), 1)
    y += dy
    cv2.putText(img, f"BGR: ({bgr[0]},{bgr[1]},{bgr[2]})", (w - panel_w + 8, y), font, 0.45, (200, 200, 200), 1)
    y += dy
    cv2.putText(img, f"HSV: ({hsv[0]},{hsv[1]},{hsv[2]})", (w - panel_w + 8, y), font, 0.45, (200, 200, 200), 1)


def draw_candidates(frame, candidates, selected_box=None):
    dbg = frame.copy()
    for box in candidates:
        color = (0, 255, 255)
        if selected_box is not None and np.array_equal(box, selected_box):
            color = (0, 255, 0)
        cv2.drawContours(dbg, [box], -1, color, 2)
        area = int(cv2.contourArea(box))
        x, y = box[0]
        cv2.putText(dbg, f"A:{area}", (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return dbg


def print_params():
    """打印当前所有参数"""
    print(f"\n========== 当前参数 ==========")
    print(f"Canny:       {CANNY_LOWER} / {CANNY_UPPER}")
    print(f"面积范围:    {MIN_CONTOUR_AREA} - {MAX_CONTOUR_AREA}")
    print(f"宽高比范围:  {ASPECT_MIN} - {ASPECT_MAX}")
    print(f"矩形度下限:  {MIN_RECTANGULARITY}")
    print(f"纸张过滤比:  {PAPER_AREA_RATIO}")
    print(f"================================\n")


# =============================================================================
# 7. 主循环
# =============================================================================
def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    YAML_PATH = os.path.join(SCRIPT_DIR, "camera_calibration.yaml")

    parser = argparse.ArgumentParser(description="三子棋 - 棋盘实时检测")
    parser.add_argument("--cam", type=int, default=2, help="摄像头 ID")
    parser.add_argument("--calib", type=str, default=YAML_PATH, help="标定 YAML")
    parser.add_argument("--no-undistort", action="store_true", help="跳过畸变矫正")
    parser.add_argument("--debug", action="store_true", help="调试模式: 中间结果窗口")
    args = parser.parse_args()

    # ---- 标定 ----
    if not args.no_undistort and os.path.exists(args.calib):
        try:
            cm, dc, Pm = load_calibration(args.calib)
            map1, map2 = init_undistort_maps(cm, dc)
            use_undistort = True
            print(f"[OK] 已加载标定: {args.calib}")
            print(f"     内参: fx={cm[0,0]:.2f} fy={cm[1,1]:.2f} "
                  f"cx={cm[0,2]:.2f} cy={cm[1,2]:.2f}")
            print(f"     畸变: k1={dc[0]:.4f} k2={dc[1]:.4f} "
                  f"p1={dc[2]:.4f} p2={dc[3]:.4f}")
        except Exception as e:
            print(f"[WARN] 标定失败: {e}")
            use_undistort = False
    else:
        use_undistort = False

    # ---- 摄像头 ----
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        for fb in [2, 0, 8]:
            cap.open(fb)
            if cap.isOpened():
                print(f"[INFO] 使用摄像头 {fb}")
                break
        else:
            print("[ERROR] 无法打开摄像头")
            return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ---- 全局可调参数 ----
    global CANNY_LOWER, CANNY_UPPER, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA
    global ASPECT_MIN, ASPECT_MAX, MIN_RECTANGULARITY, PAPER_AREA_RATIO

    # ---- 状态 ----
    squares_center = None
    inner_radius = 30
    board_box = None
    edges_debug = None
    candidates = []
    frame_count = 0
    start_time = time.time()

    # ---- 鼠标回调 ----
    cv2.namedWindow("Chessboard Detection")
    cv2.setMouseCallback("Chessboard Detection", mouse_callback)

    # ---- 调试 ----
    debug_mode = args.debug
    show_binary = False
    show_edges = False
    show_candidates = False
    show_panel = args.debug  # 右侧信息面板

    print("\n" + "=" * 55)
    print("  三子棋 - 棋盘实时检测")
    print("  调试: B=二值化 E=边缘 C=候选框 P=面板")
    print("  调参: [/]Canny下阈 {/}Canny上阈 ,/.面积")
    print("  +/-矩形度  1=打印参数  q=退出  s=截图")
    print("  鼠标: 移动=像素值 点击=打印详情")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if use_undistort:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # ---- 每帧检测棋盘 ----
        result = detect_chessboard(frame, canny_low=CANNY_LOWER, canny_high=CANNY_UPPER,
                                   area_min=MIN_CONTOUR_AREA, area_max=MAX_CONTOUR_AREA)
        squares_center, inner_radius, board_box, edges_debug, candidates = result

        # ---- 鼠标回调帧缓存 ----
        global _current_frame
        _current_frame = frame

        # ---- 显示 ----
        overlay = draw_board_overlay(frame, squares_center, inner_radius, board_box)

        # FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # 右侧信息面板
        if show_panel:
            draw_info_panel(overlay, board_box, inner_radius, len(candidates),
                            CANNY_LOWER, CANNY_UPPER, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA)

        # 底部状态栏 (参考 red_control.py draw_status_bar)
        h, w = overlay.shape[:2]
        sep_line = 2
        bar_h = 60
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h - bar_h), (w, h - bar_h + sep_line), (80, 80, 80), -1)

        lines = [
            f"FPS:{fps:.0f} | {'Board OK' if squares_center else 'No Board'}",
            f"Canny:[{CANNY_LOWER},{CANNY_UPPER}] | Area:[{MIN_CONTOUR_AREA},{MAX_CONTOUR_AREA}] | Rect:{MIN_RECTANGULARITY:.2f}",
            f"Mouse:({_mouse_info['x']},{_mouse_info['y']}) BGR:{_mouse_info['bgr']} HSV:{_mouse_info['hsv']}"
        ]
        for i, line in enumerate(lines):
            cv2.putText(overlay, line, (10, h - bar_h + 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        safe_show("Chessboard Detection", overlay)

        # ---- 调试窗口 (独立开关) ----
        if show_edges and edges_debug is not None:
            safe_show("Debug-Edges", edges_debug)
        else:
            safe_destroy("Debug-Edges")

        if show_candidates and candidates:
            cand_img = draw_candidates(frame, candidates, board_box)
            safe_show("Debug-Candidates", cand_img)
        else:
            safe_destroy("Debug-Candidates")

        # ---- 键盘 (参考 red_control.py 交互风格) ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            stamp = cv2.getTickCount()
            cv2.imwrite(f"snapshot_{stamp}.jpg", overlay)
            print(f"[SAVE] snapshot_{stamp}.jpg")

        # --- 调试窗口开关 ---
        elif key == ord('e') or key == ord('E'):
            show_edges = not show_edges
            print(f"[TOGGLE] 边缘检测: {'ON' if show_edges else 'OFF'}")
        elif key == ord('c') or key == ord('C'):
            show_candidates = not show_candidates
            print(f"[TOGGLE] 候选框: {'ON' if show_candidates else 'OFF'}")
        elif key == ord('p') or key == ord('P'):
            show_panel = not show_panel
            print(f"[TOGGLE] 信息面板: {'ON' if show_panel else 'OFF'}")
        elif key == ord('d') or key == ord('D'):
            # 一键全开/全关
            all_on = show_edges and show_candidates and show_panel
            show_edges = show_candidates = show_panel = not all_on
            print(f"[TOGGLE] 全部调试: {'ON' if not all_on else 'OFF'}")

        # --- Canny (分开上下阈值) ---
        elif key == ord('['):
            CANNY_LOWER = max(5, CANNY_LOWER - 5)
            print(f"[ADJ] Canny下阈值: {CANNY_LOWER} (上:{CANNY_UPPER})")
        elif key == ord(']'):
            CANNY_LOWER = min(CANNY_UPPER - 5, CANNY_LOWER + 5)
            print(f"[ADJ] Canny下阈值: {CANNY_LOWER} (上:{CANNY_UPPER})")
        elif key == ord('{'):
            CANNY_UPPER = max(CANNY_LOWER + 5, CANNY_UPPER - 5)
            print(f"[ADJ] Canny上阈值: {CANNY_UPPER} (下:{CANNY_LOWER})")
        elif key == ord('}'):
            CANNY_UPPER = min(255, CANNY_UPPER + 5)
            print(f"[ADJ] Canny上阈值: {CANNY_UPPER} (下:{CANNY_LOWER})")

        # --- 面积范围 ---
        elif key == ord(','):
            MIN_CONTOUR_AREA = max(500, MIN_CONTOUR_AREA - 1000)
            print(f"[ADJ] 面积范围: [{MIN_CONTOUR_AREA}, {MAX_CONTOUR_AREA}]")
        elif key == ord('.'):
            MIN_CONTOUR_AREA = min(MAX_CONTOUR_AREA - 500, MIN_CONTOUR_AREA + 1000)
            print(f"[ADJ] 面积范围: [{MIN_CONTOUR_AREA}, {MAX_CONTOUR_AREA}]")

        # --- 矩形度 ---
        elif key == ord('+') or key == ord('='):
            MIN_RECTANGULARITY = min(0.95, MIN_RECTANGULARITY + 0.05)
            print(f"[ADJ] 矩形度下限: {MIN_RECTANGULARITY:.2f}")
        elif key == ord('-'):
            MIN_RECTANGULARITY = max(0.3, MIN_RECTANGULARITY - 0.05)
            print(f"[ADJ] 矩形度下限: {MIN_RECTANGULARITY:.2f}")

        # --- 打印参数 ---
        elif key == ord('1'):
            print_params()

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
