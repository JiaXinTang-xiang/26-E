#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CAMERA="${CAMERA:-0}"
SERIAL="${SERIAL:-${SERIAL_PORT:-}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

python_version="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "错误：当前项目需要 Python 3.10 或更高版本；检测到 $python_version。" >&2
    echo "Jetson Nano 默认 JetPack 4 的 Python 3.6 不能直接运行；请用 PYTHON_BIN 指向 Python 3.10+。" >&2
    exit 1
fi

if [[ -z "$SERIAL" ]]; then
    for candidate in /dev/ttyCH341USB0 /dev/ttyCH340USB0 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0; do
        if [[ -e "$candidate" ]]; then
            SERIAL="$candidate"
            break
        fi
    done
fi

command_args=(-m apps.manual_calibration_gui --camera "$CAMERA")
if [[ -n "$SERIAL" ]]; then
    command_args+=(--serial "$SERIAL")
fi
command_args+=("$@")

exec "$PYTHON_BIN" "${command_args[@]}"
