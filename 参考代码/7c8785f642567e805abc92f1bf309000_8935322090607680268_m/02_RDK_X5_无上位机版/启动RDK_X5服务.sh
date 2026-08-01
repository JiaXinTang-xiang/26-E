#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
CAMERA_SOURCE="${CAMERA_SOURCE:-usb:/dev/video0}"
START_MODE="${START_MODE:-fixed}"
SOURCE_REGION="${SOURCE_REGION:-upper}"

pkill -f '[p]ython3 vision_server.py' 2>/dev/null || true
fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
sleep 1

nohup python3 vision_server.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --source "${CAMERA_SOURCE}" \
  --mode "${START_MODE}" \
  --source-region "${SOURCE_REGION}" \
  --use-color-hints \
  > vision_server.log 2>&1 &

echo "RDK X5 视觉服务已启动，PID=$!"
echo "日志：$(pwd)/vision_server.log"
echo "浏览器：http://<RDK_X5_IP>:${PORT}/"
