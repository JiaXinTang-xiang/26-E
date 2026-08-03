"""Stable paths for running the project from any working directory.

The PC workflow was usually started from the repository directory, so the
original code used paths such as ``configs/local/...`` relative to the current
working directory.  On Jetson this is easy to break when launching from a
desktop shortcut, systemd, or another shell directory.  Keep all project data
relative to this source tree instead.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
LOCAL_CONFIG_DIR = CONFIG_DIR / "local"
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATA_DIR = DATA_DIR / "local"
OUTPUT_DIR = PROJECT_ROOT / "output"


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project tree."""
    return PROJECT_ROOT.joinpath(*parts)

