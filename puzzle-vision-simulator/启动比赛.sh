#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CAMERA="${CAMERA:-0}"
SERIAL="${SERIAL:-${SERIAL_PORT:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "$SERIAL" ]]; then
    for candidate in /dev/ttyCH341USB0 /dev/ttyCH340USB0 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0; do
        if [[ -e "$candidate" ]]; then
            SERIAL="$candidate"
            break
        fi
    done
fi

extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera)
            [[ $# -ge 2 ]] || { echo "--camera 缺少参数" >&2; exit 2; }
            CAMERA="$2"
            shift 2
            ;;
        --serial)
            [[ $# -ge 2 ]] || { echo "--serial 缺少参数" >&2; exit 2; }
            SERIAL="$2"
            shift 2
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "找不到 Python：$PYTHON_BIN" >&2
    exit 1
}

python_version="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "错误：当前项目需要 Python 3.10 或更高版本；检测到 $python_version。" >&2
    echo "Jetson Nano 默认 JetPack 4 常带 Python 3.6，不能直接运行本项目。请使用包含 Python 3.10+ 的系统/容器，或用 PYTHON_BIN 指向已安装的 Python 3.10+。" >&2
    exit 1
fi
echo "Python: $python_version"
echo "相机: /dev/video${CAMERA}"
echo "串口: ${SERIAL:-未检测到，请连接 CH340 后在界面刷新}"
echo "CUDA: ${PUZZLE_VISION_CUDA:-auto}（可设为 0 强制 CPU）"

if [[ -z "${DISPLAY:-}" ]]; then
    echo "错误：没有检测到 DISPLAY，Tkinter 比赛界面无法打开。" >&2
    echo "请在 Jetson 桌面终端运行；若通过 SSH，需正确配置 X11/桌面 DISPLAY。" >&2
    exit 1
fi

required_files=(
    "configs/local/a4_roi.json"
    "configs/local/calibration.json"
    "configs/local/vision_detection.json"
    "data/local/empty_work_area.png"
)
for required_file in "${required_files[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "错误：缺少比赛配置文件：$SCRIPT_DIR/$required_file" >&2
        exit 1
    fi
done

mkdir -p output
if [[ ! -w output ]]; then
    echo "错误：输出目录不可写：$SCRIPT_DIR/output" >&2
    exit 1
fi

if [[ -n "$SERIAL" && ! -r "$SERIAL" ]]; then
    echo "警告：当前用户可能无权读取串口 $SERIAL。" >&2
    echo "请执行 sudo usermod -aG dialout \"$USER\"，然后注销或重启。" >&2
fi

command_args=(
    -m apps.competition_gui
    --camera "$CAMERA"
)
if [[ -n "$SERIAL" ]]; then
    command_args+=(--serial "$SERIAL")
fi
command_args+=("${extra_args[@]}")

exec "$PYTHON_BIN" "${command_args[@]}"
