#!/usr/bin/env python3
"""Generate the figures used by 切割方式研究.md."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from puzzle_sim import CARD_H, CARD_W, random_cut  # noqa: E402


OUT = ROOT / "docs" / "media"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
COLORS = ["#ef6548", "#55c64a", "#477fd7", "#c54bb7"]
JOKER = ROOT / "mujoco_sim" / "assets" / "cards" / "big_joker.png"


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def panel(polygons, title, subtitle, width=520, height=390):
    image = Image.new("RGB", (width, height), "#101820")
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), title, fill="#ffffff", font=font(27, True))
    draw.text((24, 54), subtitle, fill="#74d8ff", font=font(18))
    margin_x, top = 50, 105
    scale = min((width - 2 * margin_x) / CARD_W, 225 / CARD_H)
    ox = (width - CARD_W * scale) / 2
    oy = top
    for index, polygon in enumerate(polygons):
        pts = [(ox + x * scale, oy + y * scale) for x, y in polygon]
        draw.polygon(pts, fill=COLORS[index % len(COLORS)],
                     outline="#ffffff", width=3)
        center = np.mean(pts, axis=0)
        label = f"P{index}"
        box = draw.textbbox((0, 0), label, font=font(20, True))
        draw.text((center[0] - (box[2] - box[0]) / 2,
                   center[1] - (box[3] - box[1]) / 2),
                  label, fill="#ffffff", font=font(20, True),
                  stroke_width=2, stroke_fill="#263238")
    draw.rectangle((ox, oy, ox + CARD_W * scale, oy + CARD_H * scale),
                   outline="#ffffff", width=3)
    return image


def build_catalog():
    categories = [
        ("中心放射", "common"),
        ("边界扇形", "boundary_fan"),
        ("平行/斜向条带", "strips"),
        ("等分矩形", "equal_rectangles"),
        ("T形分层", "t_junction"),
        ("角块多边形", "corner"),
        ("凹四/五边形", "concave"),
    ]
    notes = {
        ("common", 1): "未切割；完整10 cm×6 cm矩形",
        ("common", 2): "单直线；完整边匹配",
        ("common", 3): "单一内部公共点",
        ("common", 4): "内部公共点；闭环C4",
        ("boundary_fan", 1): "退化为未切割矩形",
        ("boundary_fan", 2): "退化为单直线切割",
        ("boundary_fan", 3): "边界公共点；路径P3",
        ("boundary_fan", 4): "边界公共点；路径P4",
        ("strips", 1): "未切割矩形",
        ("strips", 2): "两条带；无内部公共点",
        ("strips", 3): "三条带；允许不同斜率",
        ("strips", 4): "四条带；最短边≥2 cm",
        ("equal_rectangles", 1): "完整矩形",
        ("equal_rectangles", 2): "两块相同5 cm×6 cm矩形",
        ("equal_rectangles", 3): "三块相同10/3 cm×6 cm矩形",
        ("equal_rectangles", 4): "2×2；四块相同5 cm×3 cm矩形",
        ("t_junction", 1): "退化为未切割矩形",
        ("t_junction", 2): "退化为单直线切割",
        ("t_junction", 3): "单T点；长边对应两短边",
        ("t_junction", 4): "双T点；两侧分层",
        ("corner", 1): "退化为未切割矩形",
        ("corner", 2): "退化为单直线切割",
        ("corner", 3): "单角块；三角形+五边形",
        ("corner", 4): "双角块；三/五边形混合",
        ("concave", 1): "未切割；不存在合法凹口",
        ("concave", 2): "凹四/五边形+互补片",
        ("concave", 3): "凹片+两块扇形子片",
        ("concave", 4): "凹片+三块扇形子片",
    }
    cases = [
        (f"{title} · {count}片", notes[(mode, count)],
         mode, count, 100 * category_index + count)
        for category_index, (title, mode) in enumerate(categories, 1)
        for count in range(2, 5)
    ]
    panels = []
    for title, subtitle, mode, count, seed in cases:
        polygons = random_cut(np.random.default_rng(seed), count, mode)
        panels.append(panel(polygons, title, subtitle))
    cols, rows = 3, 7
    canvas = Image.new("RGB", (cols * 520, rows * 390), "#081119")
    for i, item in enumerate(panels):
        canvas.paste(item, ((i % cols) * 520, (i // cols) * 390))
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / "cut-methods-comparison.png", quality=95)


def build_matching_figure():
    image = Image.new("RGB", (1500, 620), "#081119")
    draw = ImageDraw.Draw(image)
    draw.text((42, 24), "完整边匹配与 T 形部分边匹配",
              fill="white", font=font(34, True))

    # Complete edge example.
    draw.text((70, 90), "A  完整边 ↔ 完整边", fill="#74d8ff",
              font=font(25, True))
    draw.polygon([(80, 155), (350, 125), (350, 405), (80, 375)],
                 fill=COLORS[0], outline="white", width=3)
    draw.polygon([(350, 125), (650, 180), (650, 460), (350, 405)],
                 fill=COLORS[2], outline="white", width=3)
    draw.line([(350, 125), (350, 405)], fill="#ffdf4d", width=10)
    draw.text((112, 480), "两侧端点一一对应，刚体变换唯一",
              fill="white", font=font(21))

    # Partial edge example.
    draw.text((790, 90), "B  长边 ↔ 两条短边（T 点）", fill="#74d8ff",
              font=font(25, True))
    draw.polygon([(800, 135), (1060, 135), (1060, 445), (800, 445)],
                 fill=COLORS[1], outline="white", width=3)
    draw.polygon([(1060, 135), (1405, 135), (1405, 285), (1060, 285)],
                 fill=COLORS[2], outline="white", width=3)
    draw.polygon([(1060, 285), (1405, 285), (1405, 445), (1060, 445)],
                 fill=COLORS[3], outline="white", width=3)
    draw.line([(1060, 135), (1060, 445)], fill="#ffdf4d", width=10)
    draw.ellipse((1048, 273, 1072, 297), fill="#ff3b30", outline="white")
    draw.text((1080, 252), "T 点", fill="#ff8a80", font=font(20, True))
    draw.text((820, 480), "短边枚举长边首段/末段，再用矩形全局约束判定",
              fill="white", font=font(21))
    image.save(OUT / "cut-edge-matching-comparison.png", quality=95)


def joker_panel(polygons, title, subtitle, width=520, height=390):
    """Draw actual cut boundaries over the white-background Joker texture."""
    image = Image.new("RGB", (width, height), "#101820")
    draw = ImageDraw.Draw(image)
    draw.text((24, 16), title, fill="#ffffff", font=font(27, True))
    draw.text((24, 54), subtitle, fill="#74d8ff", font=font(18))
    card = Image.open(JOKER).convert("RGB").resize(
        (400, 240), Image.Resampling.LANCZOS)
    ox, oy = 60, 108
    image.paste(card, (ox, oy))
    draw = ImageDraw.Draw(image)
    scale_x, scale_y = 400 / CARD_W, 240 / CARD_H
    for index, polygon in enumerate(polygons):
        points = [
            (ox + x * scale_x, oy + y * scale_y)
            for x, y in polygon
        ]
        # Cyan/white double stroke remains visible over both red-black artwork
        # and the white card margin.
        draw.line(points + [points[0]], fill="#ffffff", width=8, joint="curve")
        draw.line(points + [points[0]], fill="#00d9ff", width=3, joint="curve")
        center = np.mean(points, axis=0)
        radius = 18
        draw.ellipse(
            (center[0] - radius, center[1] - radius,
             center[0] + radius, center[1] + radius),
            fill=COLORS[index % len(COLORS)], outline="#ffffff", width=2)
        label = f"P{index}"
        box = draw.textbbox((0, 0), label, font=font(15, True))
        draw.text(
            (center[0] - (box[2] - box[0]) / 2,
             center[1] - (box[3] - box[1]) / 2 - 1),
            label, fill="#ffffff", font=font(15, True),
            stroke_width=1, stroke_fill="#263238")
    draw.rectangle((ox, oy, ox + 400, oy + 240),
                   outline="#ffffff", width=3)
    return image


def build_joker_catalog():
    """Nine representative topology configurations using the real Joker."""
    cases = [
        ("单直线切割", "2片 · 完整边匹配", "common", 2, 3),
        ("三片中心放射", "3片 · 内部公共点", "common", 3, 7),
        ("四片中心放射", "4片 · 闭环C4", "common", 4, 14),
        ("边界扇形", "4片 · 边界公共点", "boundary_fan", 4, 5),
        ("平行/斜向条带", "4片 · 无公共顶点", "strips", 4, 4),
        ("四块相同矩形", "4片 · 2×2等分", "equal_rectangles", 4, 2),
        ("T形分层", "3片 · 部分边匹配", "t_junction", 3, 6),
        ("角块多边形", "4片 · 三/五边形混合", "corner", 4, 16),
        ("凹片复合", "4片 · 凹四/五边形", "concave", 4, 12),
    ]
    panels = [
        joker_panel(
            random_cut(np.random.default_rng(seed), count, mode),
            title, subtitle)
        for title, subtitle, mode, count, seed in cases
    ]
    canvas = Image.new("RGB", (3 * 520, 3 * 390), "#081119")
    for index, item in enumerate(panels):
        canvas.paste(item, ((index % 3) * 520, (index // 3) * 390))
    canvas.save(OUT / "joker-cut-methods-comparison.png", quality=95)


if __name__ == "__main__":
    build_catalog()
    build_matching_figure()
    build_joker_catalog()
