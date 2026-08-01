#!/usr/bin/env bash
set -euo pipefail
pkill -f '[p]ython3 vision_server.py' 2>/dev/null || true
fuser -k 8000/tcp >/dev/null 2>&1 || true
echo "RDK X5 视觉服务已停止。"
