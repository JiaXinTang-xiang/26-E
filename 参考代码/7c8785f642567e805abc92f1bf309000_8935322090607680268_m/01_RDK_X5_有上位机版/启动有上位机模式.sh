#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 可通过环境变量覆盖：
# START_MODE=fixed|unknown-white|unknown-pattern
# CAMERA_SOURCE=usb:/dev/video0
# PORT=8000
exec ./启动RDK_X5服务.sh
