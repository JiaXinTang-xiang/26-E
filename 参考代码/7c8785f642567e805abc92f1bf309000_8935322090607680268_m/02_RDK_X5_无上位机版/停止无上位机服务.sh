#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
pkill -f '[p]ython3 vision_server.py' 2>/dev/null || true
fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
echo "RDK X5 无上位机视觉服务已停止。"
