#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CAMERA_INDEX="${CAMERA_INDEX:-0}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"

exec python3 -m apps.manual_calibration_gui \
  --camera "$CAMERA_INDEX" \
  --serial "$SERIAL_PORT" \
  "$@"
