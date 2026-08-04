#!/usr/bin/env bash
# Compatibility entry point. The maintained launcher is 启动比赛.sh.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/启动比赛.sh" "$@"
