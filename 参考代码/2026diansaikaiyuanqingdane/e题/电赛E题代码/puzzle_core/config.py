from __future__ import annotations

from pathlib import Path

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
SPLIT_Y_MM = A4_HEIGHT_MM / 2.0
TARGET_CENTER_MM = (A4_WIDTH_MM / 2.0, (SPLIT_Y_MM + A4_HEIGHT_MM) / 2.0)
DEFAULT_SCALE = 5.0

# The generated bitmap is the A4 itself. Use a warm paper white so it reads as
# a real sheet against the dark GUI viewer, while pure-white pieces remain
# slightly brighter and can still be segmented in the synthetic pipeline.
BACKGROUND_BGR = (242, 244, 246)
DIVIDER_BGR = (92, 98, 104)
SELF_PIECE_BGR = (35, 205, 245)
WHITE_PIECE_BGR = (255, 255, 255)
PAPER_EDGE_BGR = (150, 155, 160)
PIECE_EDGE_BGR = (105, 110, 116)
PIECE_SHADOW_BGR = (150, 154, 158)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES_ROOT = PACKAGE_ROOT / "images"
DEFAULT_OUTPUTS_ROOT = PACKAGE_ROOT / "outputs"
