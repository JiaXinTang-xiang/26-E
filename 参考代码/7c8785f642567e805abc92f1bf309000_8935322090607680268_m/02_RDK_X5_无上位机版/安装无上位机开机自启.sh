#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
START_MODE="${START_MODE:-fixed}"
CAMERA_SOURCE="${CAMERA_SOURCE:-usb:/dev/video0}"
SOURCE_REGION="${SOURCE_REGION:-upper}"
PORT="${PORT:-8000}"
SERVICE_NAME="a4-puzzle-vision.service"

case "${START_MODE}" in
  fixed|unknown-white|unknown-pattern) ;;
  *)
    echo "START_MODE 必须是 fixed、unknown-white 或 unknown-pattern。" >&2
    exit 2
    ;;
esac

if [[ ! -f "${PROJECT_DIR}/vision_server.py" ]]; then
  echo "当前目录缺少 vision_server.py。" >&2
  exit 2
fi

sudo tee "/etc/systemd/system/${SERVICE_NAME}" >/dev/null <<EOF
[Unit]
Description=A4 Puzzle Vision Headless on RDK X5
After=local-fs.target
ConditionPathExists=${PROJECT_DIR}/vision_server.py

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 ${PROJECT_DIR}/vision_server.py --host 127.0.0.1 --port ${PORT} --source ${CAMERA_SOURCE} --mode ${START_MODE} --source-region ${SOURCE_REGION} --use-color-hints
Restart=on-failure
RestartSec=2
KillSignal=SIGTERM
TimeoutStopSec=8

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl status "${SERVICE_NAME}" --no-pager

echo
echo "已安装无上位机开机自启服务：${SERVICE_NAME}"
echo "状态接口：http://127.0.0.1:${PORT}/api/status"
echo "日志命令：journalctl -u ${SERVICE_NAME} -f"
