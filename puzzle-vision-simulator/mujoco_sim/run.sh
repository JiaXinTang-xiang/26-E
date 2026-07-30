#!/usr/bin/env bash
set -e
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    exec "$ROOT_DIR/.venv/bin/python" -m mujoco_sim.run_sim "$@"
fi

exec python3 -m mujoco_sim.run_sim "$@"
