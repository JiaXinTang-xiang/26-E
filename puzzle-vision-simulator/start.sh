#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# -- defaults (override via env or args) ----------------------------------
CAMERA="${CAMERA:-0}"
SERIAL="${SERIAL:-}"
export PUZZLE_ASSEMBLY_WORKERS="${PUZZLE_ASSEMBLY_WORKERS:-2}"

# auto-detect CH341 / CH340 serial port if not explicitly set
if [[ -z "$SERIAL" ]]; then
    for candidate in /dev/ttyCH341USB0 /dev/ttyCH340USB0 /dev/ttyUSB0 /dev/ttyUSB1; do
        if [[ -e "$candidate" ]]; then
            SERIAL="$candidate"
            break
        fi
    done
fi

# allow command-line overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --camera) CAMERA="$2"; shift 2 ;;
        --serial) SERIAL="$2"; shift 2 ;;
        *) echo "usage: $0 [--camera N] [--serial /dev/ttyXXX]" >&2; exit 2 ;;
    esac
done

echo "相机:  /dev/video${CAMERA}"
echo "串口:  ${SERIAL:-（未连接）}"
echo "拼接进程: $PUZZLE_ASSEMBLY_WORKERS"
echo "----------------------------------------"

PYTHON_BIN="python3"
if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

exec "$PYTHON_BIN" -m apps.competition_gui \
    --camera "$CAMERA" \
    ${SERIAL:+--serial "$SERIAL"}
