#!/usr/bin/env python3
"""E题拼图装置：随机切割、随机摆放、视觉识别和矩形还原仿真。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 900, 1200
DIVIDER_Y = 575
PIXELS_PER_CM = 40.0
CARD_WIDTH_CM, CARD_HEIGHT_CM = 10.0, 6.0
CARD_W = CARD_WIDTH_CM * PIXELS_PER_CM
CARD_H = CARD_HEIGHT_CM * PIXELS_PER_CM
TARGET_Y = 790
PIECE_BGR = (245, 245, 245)
PAPER_BGR = (18, 18, 18)
CARD_ASSET = (
    Path(__file__).resolve().parents[2]
    / "mujoco_sim/assets/cards/big_joker_rounded.png")


def _clip_half_plane(poly: np.ndarray, point: np.ndarray,
                     normal: np.ndarray, keep_positive: bool) -> np.ndarray:
    """Clip a convex polygon by one side of an infinite straight cut."""
    result = []
    signed = (poly - point) @ normal
    if not keep_positive:
        signed = -signed
    for i, current in enumerate(poly):
        previous = poly[i - 1]
        current_d, previous_d = signed[i], signed[i - 1]
        current_in, previous_in = current_d >= -1e-7, previous_d >= -1e-7
        if current_in != previous_in:
            t = previous_d / (previous_d - current_d)
            result.append(previous + t * (current - previous))
        if current_in:
            result.append(current)
    return np.asarray(result, dtype=float)


def _valid_piece(poly: np.ndarray) -> bool:
    if not 3 <= len(poly) <= 5:
        return False
    lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    return (lengths.min() >= 2.0 * PIXELS_PER_CM and
            abs(cv2.contourArea(poly.astype(np.float32))) >= 6500)


def _edge_on_target_boundary(a: np.ndarray, b: np.ndarray,
                             tolerance: float = 1e-6) -> bool:
    return (
        (abs(a[0]) <= tolerance and abs(b[0]) <= tolerance)
        or (abs(a[0] - CARD_W) <= tolerance
            and abs(b[0] - CARD_W) <= tolerance)
        or (abs(a[1]) <= tolerance and abs(b[1]) <= tolerance)
        or (abs(a[1] - CARD_H) <= tolerance
            and abs(b[1] - CARD_H) <= tolerance)
    )


def validate_cut_layout(pieces: list[np.ndarray]) -> None:
    """Enforce all three field-piece geometry requirements."""
    for index, piece in enumerate(pieces):
        if not 3 <= len(piece) <= 5:
            raise RuntimeError(f"P{index} 边数为 {len(piece)}，要求3～5边")
        edge_list = edges(piece)
        lengths = [np.linalg.norm(b - a) for a, b in edge_list]
        if min(lengths) < 2.0 * PIXELS_PER_CM - 1e-6:
            raise RuntimeError(
                f"P{index} 最短边 {min(lengths) / PIXELS_PER_CM:.3f}cm，小于2cm")
        if not any(_edge_on_target_boundary(a, b) for a, b in edge_list):
            raise RuntimeError(f"P{index} 没有位于目标矩形外边上的边")


def _common_vertex_cut(rng: np.random.Generator,
                       piece_count: int) -> list[np.ndarray]:
    """Original proven radial/common-vertex layouts."""
    tl, tr = np.array([0., 0.]), np.array([CARD_W, 0.])
    br, bl = np.array([CARD_W, CARD_H]), np.array([0., CARD_H])
    if piece_count == 1:
        return [np.array([tl, tr, br, bl])]
    if piece_count == 2:
        t = np.array([rng.uniform(.25, .75) * CARD_W, 0.0])
        b = np.array([rng.uniform(.25, .75) * CARD_W, CARD_H])
        return [np.array([tl, t, b, bl]), np.array([t, tr, br, b])]
    if piece_count == 3:
        c = np.array([rng.uniform(.43, .57) * CARD_W,
                      rng.uniform(.34, .48) * CARD_H])
        t = np.array([rng.uniform(.32, .68) * CARD_W, 0.0])
        # Keep both resulting outer vertical segments at least 2 cm.
        r = np.array([CARD_W, rng.uniform(.50, .65) * CARD_H])
        l = np.array([0.0, rng.uniform(.50, .65) * CARD_H])
        return [
            np.array([t, tr, r, c]),
            np.array([r, br, bl, l, c]),
            np.array([l, tl, t, c]),
        ]
    c = np.array([rng.uniform(.38, .62) * CARD_W,
                  rng.uniform(.35, .65) * CARD_H])
    t = np.array([rng.uniform(.20, .80) * CARD_W, 0.0])
    r = np.array([CARD_W, rng.uniform(1/3, 2/3) * CARD_H])
    b = np.array([rng.uniform(.20, .80) * CARD_W, CARD_H])
    l = np.array([0.0, rng.uniform(1/3, 2/3) * CARD_H])
    return [
        np.array([tl, t, c, l]),
        np.array([t, tr, r, c]),
        np.array([c, r, br, b]),
        np.array([l, c, b, bl]),
    ]


def _parallel_non_common_cut(rng: np.random.Generator,
                             piece_count: int) -> list[np.ndarray]:
    """Parallel full-edge cuts with no shared global vertex."""
    if piece_count == 1:
        return [np.array([
            [0., 0.], [CARD_W, 0.], [CARD_W, CARD_H], [0., CARD_H]])]
    # Four horizontal strips cannot satisfy the official 2 cm minimum edge
    # inside a 6 cm height, so four-piece layouts use vertical/oblique strips.
    vertical = piece_count == 4 or bool(rng.integers(0, 2))
    span = CARD_W if vertical else CARD_H
    minimum = 2.0 * PIXELS_PER_CM
    remainder = span - piece_count * minimum

    def boundaries():
        if remainder <= 1e-8:
            widths = np.full(piece_count, minimum)
        else:
            widths = minimum + rng.dirichlet(
                np.full(piece_count, 2.5)) * remainder
        return np.r_[0.0, np.cumsum(widths)]

    first, second = boundaries(), boundaries()
    pieces = []
    for i in range(piece_count):
        if vertical:
            pieces.append(np.array([
                [first[i], 0.0], [first[i + 1], 0.0],
                [second[i + 1], CARD_H], [second[i], CARD_H]]))
        else:
            pieces.append(np.array([
                [0.0, first[i]], [CARD_W, second[i]],
                [CARD_W, second[i + 1]], [0.0, first[i + 1]]]))
    return pieces


def _equal_rectangle_cut(piece_count: int) -> list[np.ndarray]:
    """Equal rectangular tiles, including the ambiguous four-piece 2×2 case."""
    if piece_count == 4:
        mid_x, mid_y = CARD_W / 2, CARD_H / 2
        return [
            np.array([[0., 0.], [mid_x, 0.], [mid_x, mid_y], [0., mid_y]]),
            np.array([[mid_x, 0.], [CARD_W, 0.],
                      [CARD_W, mid_y], [mid_x, mid_y]]),
            np.array([[0., mid_y], [mid_x, mid_y],
                      [mid_x, CARD_H], [0., CARD_H]]),
            np.array([[mid_x, mid_y], [CARD_W, mid_y],
                      [CARD_W, CARD_H], [mid_x, CARD_H]]),
        ]
    # For 1–3 pieces, equal vertical rectangles keep every edge >=2 cm.
    bounds = np.linspace(0., CARD_W, piece_count + 1)
    return [
        np.array([[bounds[i], 0.], [bounds[i + 1], 0.],
                  [bounds[i + 1], CARD_H], [bounds[i], CARD_H]])
        for i in range(piece_count)
    ]


def _boundary_fan_cut(rng: np.random.Generator,
                      piece_count: int) -> list[np.ndarray]:
    """Rays share one point on the top boundary rather than an interior point."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    center = np.array([rng.uniform(.42, .58) * CARD_W, 0.])
    minimum = 2.0 * PIXELS_PER_CM
    remainder = CARD_W - piece_count * minimum
    widths = (np.full(piece_count, minimum) if remainder <= 1e-8 else
              minimum + rng.dirichlet(np.full(piece_count, 2.5)) * remainder)
    bottom = np.r_[0.0, np.cumsum(widths)]
    cuts = [np.array([x, CARD_H]) for x in bottom[1:-1]]
    pieces = [
        np.array([[0., 0.], center, cuts[0], [0., CARD_H]])
    ]
    for left, right in zip(cuts, cuts[1:]):
        pieces.append(np.array([center, right, left]))
    pieces.append(np.array([
        center, [CARD_W, 0.], [CARD_W, CARD_H], cuts[-1]]))
    return pieces


def _t_junction_cut(rng: np.random.Generator,
                    piece_count: int) -> list[np.ndarray]:
    """Hierarchical T cuts, including long-edge to short-edge adjacency."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    top = rng.uniform(.42, .58) * CARD_W
    bottom = rng.uniform(.42, .58) * CARD_W
    p0, p1 = np.array([top, 0.]), np.array([bottom, CARD_H])
    if piece_count == 3:
        t = rng.uniform(.34, .66)
        junction = p0 * (1 - t) + p1 * t
        right = np.array([CARD_W, rng.uniform(.34, .66) * CARD_H])
        return [
            np.array([[0., 0.], p0, p1, [0., CARD_H]]),
            np.array([p0, [CARD_W, 0.], right, junction]),
            np.array([junction, right, [CARD_W, CARD_H], p1]),
        ]
    left_t = rng.uniform(.34, .54)
    right_t = rng.uniform(.46, .66)
    left_junction = p0 * (1 - left_t) + p1 * left_t
    right_junction = p0 * (1 - right_t) + p1 * right_t
    left = np.array([0., rng.uniform(.34, .60) * CARD_H])
    right = np.array([CARD_W, rng.uniform(.40, .66) * CARD_H])
    return [
        np.array([[0., 0.], p0, left_junction, left]),
        np.array([left, left_junction, p1, [0., CARD_H]]),
        np.array([p0, [CARD_W, 0.], right, right_junction]),
        np.array([right_junction, right, [CARD_W, CARD_H], p1]),
    ]


def _corner_polygon_cut(rng: np.random.Generator,
                        piece_count: int) -> list[np.ndarray]:
    """Opposite corner triangles plus central quadrilateral/pentagon pieces."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    a = np.array([rng.uniform(.20, .28) * CARD_W, 0.])
    b = np.array([0., rng.uniform(.35, .52) * CARD_H])
    e = np.array([rng.uniform(.52, .60) * CARD_W, 0.])
    f = np.array([rng.uniform(.40, .48) * CARD_W, CARD_H])
    if piece_count == 3:
        return [
            np.array([[0., 0.], a, b]),
            np.array([a, e, f, [0., CARD_H], b]),
            np.array([e, [CARD_W, 0.], [CARD_W, CARD_H], f]),
        ]
    c = np.array([CARD_W, rng.uniform(.48, .65) * CARD_H])
    d = np.array([rng.uniform(.72, .80) * CARD_W, CARD_H])
    return [
        np.array([[0., 0.], a, b]),
        np.array([a, e, f, [0., CARD_H], b]),
        np.array([e, [CARD_W, 0.], c, d, f]),
        np.array([c, [CARD_W, CARD_H], d]),
    ]


def _concave_polyline_cut(rng: np.random.Generator,
                          piece_count: int) -> list[np.ndarray]:
    """A bent cut producing one legal concave pentagon.

    The internal boundary R→J→BL is a two-segment polyline.  Additional
    pieces fan from J to independently spaced points on the bottom edge.
    Every result has 3–5 sides, a >=2 cm outside edge and >=2 cm edges.
    """
    if piece_count == 1:
        return _common_vertex_cut(rng, 1)
    tl, tr = np.array([0., 0.]), np.array([CARD_W, 0.])
    br, bl = np.array([CARD_W, CARD_H]), np.array([0., CARD_H])
    # Starting the polyline at the top-right corner yields a concave
    # quadrilateral; starting lower on the right edge yields a pentagon.
    right = (
        np.array([CARD_W, 0.])
        if bool(rng.integers(0, 2))
        else np.array([CARD_W, rng.uniform(.35, .47) * CARD_H])
    )
    junction = np.array([
        rng.uniform(.43, .60) * CARD_W,
        # A deep, camera-resolvable re-entrant corner.  Keeping it above the
        # right boundary endpoint also avoids a near-collinear "fake" notch.
        rng.uniform(.08, .14) * CARD_H,
    ])
    concave = (
        np.array([tl, tr, junction, bl])
        if np.allclose(right, tr)
        else np.array([tl, tr, right, junction, bl])
    )
    pieces = [concave]

    lower_count = piece_count - 1
    minimum = 2.0 * PIXELS_PER_CM
    remainder = CARD_W - lower_count * minimum
    widths = (np.full(lower_count, minimum) if remainder <= 1e-8 else
              minimum + rng.dirichlet(np.full(lower_count, 3.0)) * remainder)
    bounds = np.r_[0., np.cumsum(widths)]
    # Walk intervals from right to left to preserve the boundary winding.
    for interval in range(lower_count - 1, -1, -1):
        lo, hi = bounds[interval], bounds[interval + 1]
        if interval == lower_count - 1:
            poly = [right, br, np.array([lo, CARD_H]), junction]
        elif interval == 0:
            poly = [np.array([hi, CARD_H]), bl, junction]
        else:
            poly = [
                np.array([hi, CARD_H]),
                np.array([lo, CARD_H]),
                junction,
            ]
        pieces.append(np.asarray(poly, dtype=float))
    return pieces


def random_cut(rng: np.random.Generator, piece_count: int = 4,
               cut_mode: str = "sequential") -> list[np.ndarray]:
    """Make 1-4 pieces by independent sequential cuts.

    Unlike a radial-only generator, this produces parallel cuts, T junctions,
    branch cuts and mixed layouts. No global common vertex is assumed.
    """
    if not 1 <= piece_count <= 4:
        raise ValueError("切割后的碎片数量必须在 1～4 之间")
    if piece_count == 1:
        return [np.array(
            [[0., 0.], [CARD_W, 0.], [CARD_W, CARD_H], [0., CARD_H]],
            dtype=float,
        )]
    aliases = {"sequential": "strips"}
    cut_mode = aliases.get(cut_mode, cut_mode)
    allowed = {
        "common", "boundary_fan", "strips", "equal_rectangles",
        "t_junction", "corner", "concave", "mixed"}
    if cut_mode not in allowed:
        raise ValueError(f"未知切割方式：{cut_mode}")
    if cut_mode == "mixed":
        choices = ["common", "boundary_fan", "strips", "equal_rectangles"]
        if piece_count >= 3:
            choices += ["t_junction", "corner"]
        if piece_count >= 2:
            choices += ["concave"]
        cut_mode = str(rng.choice(choices))
    if cut_mode == "common":
        pieces = _common_vertex_cut(rng, piece_count)
    elif cut_mode == "boundary_fan":
        pieces = _boundary_fan_cut(rng, piece_count)
    elif cut_mode == "strips":
        pieces = _parallel_non_common_cut(rng, piece_count)
    elif cut_mode == "equal_rectangles":
        pieces = _equal_rectangle_cut(piece_count)
    elif cut_mode == "t_junction":
        pieces = _t_junction_cut(rng, piece_count)
    elif cut_mode == "concave":
        pieces = _concave_polyline_cut(rng, piece_count)
    else:
        pieces = _corner_polygon_cut(rng, piece_count)
    validate_cut_layout(pieces)
    return pieces


def resolve_cut_mode(seed: int, piece_count: int, cut_mode: str) -> str:
    """Resolve the deterministic category used by a mixed-mode scene."""
    cut_mode = {"sequential": "strips"}.get(cut_mode, cut_mode)
    if cut_mode != "mixed":
        return cut_mode
    choices = ["common", "boundary_fan", "strips", "equal_rectangles"]
    if piece_count >= 3:
        choices += ["t_junction", "corner"]
    if piece_count >= 2:
        choices += ["concave"]
    return str(np.random.default_rng(seed).choice(choices))


def rigid(angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def apply_h(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.c_[points, np.ones(len(points))] @ h.T
    return q[:, :2] / q[:, 2, None]


def place_randomly(polys: list[np.ndarray], rng: np.random.Generator):
    placed = [None] * len(polys)
    occupancy = np.zeros((DIVIDER_Y - 25, CANVAS_W), np.uint8)
    # Place large pieces first so that 3-piece layouts do not trap the largest
    # sector in the remaining narrow gaps.
    ordered = sorted(range(len(polys)),
                     key=lambda i: abs(cv2.contourArea(polys[i].astype(np.float32))),
                     reverse=True)
    for index in ordered:
        poly = polys[index]
        centroid = poly.mean(axis=0)
        local = poly - centroid
        accepted = False
        for _ in range(2000):
            angle = rng.uniform(-math.pi, math.pi)
            rot = rigid(angle, 0, 0)
            rotated = apply_h(local, rot)
            mn, mx = rotated.min(0), rotated.max(0)
            tx = rng.uniform(25 - mn[0], CANVAS_W - 25 - mx[0])
            ty = rng.uniform(25 - mn[1], DIVIDER_Y - 35 - mx[1])
            scene_poly = rotated + [tx, ty]
            mask = np.zeros_like(occupancy)
            cv2.fillPoly(mask, [np.round(scene_poly).astype(np.int32)], 255)
            dilated = cv2.dilate(mask, np.ones((19, 19), np.uint8))
            if not np.any((dilated > 0) & (occupancy > 0)):
                cv2.fillPoly(occupancy, [np.round(scene_poly).astype(np.int32)], 255)
                placed[index] = scene_poly
                accepted = True
                break
        if not accepted:
            raise RuntimeError("无法在上半区无重叠摆放碎片，请更换随机种子")
    return placed


def _joker_card() -> np.ndarray:
    try:
        encoded = np.fromfile(CARD_ASSET, dtype=np.uint8)
    except OSError:
        encoded = np.empty(0, dtype=np.uint8)
    card = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if card is None:
        raise RuntimeError(f"无法读取大鬼扑克牌素材：{CARD_ASSET}")
    return cv2.resize(card, (int(CARD_W), int(CARD_H)),
                      interpolation=cv2.INTER_AREA)


def _rounded_card_mask() -> np.ndarray:
    """Physical 10×6 cm playing-card stock with approximately 4 mm corners."""
    width, height = int(CARD_W), int(CARD_H)
    radius = max(8, int(round(0.40 * PIXELS_PER_CM)))
    mask = np.zeros((height, width), np.uint8)
    cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1),
                  255, -1)
    cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1),
                  255, -1)
    for x, y in (
            (radius, radius),
            (width - radius - 1, radius),
            (radius, height - radius - 1),
            (width - radius - 1, height - radius - 1)):
        cv2.circle(mask, (x, y), radius, 255, -1)
    return mask


def render_scene(placed: list[np.ndarray], source: list[np.ndarray] | None = None,
                 material_mode: str = "color") -> np.ndarray:
    image = np.full((CANVAS_H, CANVAS_W, 3), PAPER_BGR, np.uint8)
    cv2.line(image, (0, DIVIDER_Y), (CANVAS_W, DIVIDER_Y), (210, 210, 210), 4)
    target_x = int((CANVAS_W - CARD_W) / 2)
    target_y = TARGET_Y
    cv2.rectangle(image, (target_x, target_y),
                  (int(target_x + CARD_W), int(target_y + CARD_H)), (190, 190, 190), 2)
    cv2.putText(image, "Target: 10 cm x 6 cm", (target_x, target_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 220, 220), 2, cv2.LINE_AA)
    joker = _joker_card() if material_mode == "joker" else None
    joker_stock = _rounded_card_mask() if joker is not None else None
    for index, poly in enumerate(placed):
        pts = np.round(poly).astype(np.int32)
        if joker is None or source is None:
            cv2.fillPoly(image, [pts], PIECE_BGR, lineType=cv2.LINE_8)
        else:
            src = source[index].astype(np.float32)
            dst = poly.astype(np.float32)
            transform = cv2.getAffineTransform(src[:3], dst[:3])
            warped = cv2.warpAffine(
                joker, transform, (CANVAS_W, CANVAS_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            polygon_mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
            cv2.fillPoly(polygon_mask, [pts], 255)
            warped_stock = cv2.warpAffine(
                joker_stock, transform, (CANVAS_W, CANVAS_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = cv2.bitwise_and(
                polygon_mask, np.where(
                    warped_stock > 96, 255, 0).astype(np.uint8))
            image[mask > 0] = warped[mask > 0]
            physical_contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(
                image, physical_contours, -1, (120, 120, 120),
                2, cv2.LINE_AA)
            continue
        # Use a darker color of the same hue so the visual border remains part
        # of the segmented foreground instead of clipping polygon corners.
        cv2.polylines(image, [pts], True, (120, 120, 120), 2, cv2.LINE_AA)
    return image


def generate_camera_frame(seed: int, piece_count: int,
                          material_mode: str = "color",
                          cut_mode: str = "sequential") -> np.ndarray:
    """Generation boundary: return pixels only, never geometry or true poses."""
    rng = np.random.default_rng(seed)
    source_polygons = random_cut(rng, piece_count, cut_mode)
    placed_polygons = place_randomly(source_polygons, rng)
    rendered = render_scene(
        placed_polygons, source_polygons, material_mode=material_mode)

    # Simulate a camera/file boundary. Geometry variables stay local to this
    # function and the vision stage receives only decoded image pixels.
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("仿真相机图像编码失败")
    camera_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if camera_frame is None:
        raise RuntimeError("仿真相机图像解码失败")
    return camera_frame


def order_clockwise(vertices: np.ndarray) -> np.ndarray:
    c = vertices.mean(axis=0)
    a = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(a)]


def _line_intersection(a: np.ndarray, b: np.ndarray,
                       c: np.ndarray, d: np.ndarray) -> np.ndarray | None:
    """Intersection of two infinite 2-D lines, or None when nearly parallel."""
    u, v = b - a, d - c
    cross = u[0] * v[1] - u[1] * v[0]
    if abs(cross) < 1e-7:
        return None
    delta = c - a
    t = (delta[0] * v[1] - delta[1] * v[0]) / cross
    return a + t * u


def _merge_rounded_corners(vertices: np.ndarray) -> np.ndarray:
    """Replace a rounded-corner chord by the virtual sharp line intersection.

    A physical playing-card corner appears after contour simplification as a
    short diagonal chord between two long, near-perpendicular straight edges.
    Official cut edges are at least 2 cm, so this sub-centimetre chord can be
    distinguished without using card artwork or generator geometry.
    """
    points = [p.astype(float) for p in vertices]
    if len(points) < 4:
        return np.asarray(points)
    bounds = np.ptp(np.asarray(points), axis=0)
    chord_limit = min(42.0, max(12.0, 0.24 * float(bounds.min())))
    changed = True
    while changed and len(points) >= 4:
        changed = False
        count = len(points)
        for i in range(count):
            # Rotate so a,b are the end of a long edge. One physical arc may
            # simplify to one, two or three consecutive short chord segments.
            rotated = points[(i - 1):] + points[:(i - 1)]
            a, b = rotated[0], rotated[1]
            previous = b - a
            lp = float(np.linalg.norm(previous))
            for short_count in range(1, min(3, count - 3) + 1):
                arc_edges = [
                    rotated[k + 1] - rotated[k]
                    for k in range(1, short_count + 1)
                ]
                arc_lengths = [float(np.linalg.norm(edge))
                               for edge in arc_edges]
                if not all(3.0 <= length <= chord_limit
                           for length in arc_lengths):
                    continue
                c = rotated[short_count + 1]
                d = rotated[short_count + 2]
                following = d - c
                ln = float(np.linalg.norm(following))
                arc_length = sum(arc_lengths)
                if lp < 1.5 * arc_length or ln < 1.5 * arc_length:
                    continue
                cosine = abs(float(
                    np.dot(previous, following) / (lp * ln)))
                if cosine > math.cos(math.radians(68.0)):
                    continue
                corner = _line_intersection(a, b, c, d)
                if corner is None:
                    continue
                if max(np.linalg.norm(corner - b),
                       np.linalg.norm(corner - c)) > 2.5 * arc_length:
                    continue
                # Replace every sampled arc vertex by one virtual intersection.
                points = (
                    rotated[:1] + [corner]
                    + rotated[short_count + 2:])
                changed = True
                break
            if changed:
                break
    return np.asarray(points, dtype=float)


def _piece_foreground_mask(image: np.ndarray) -> np.ndarray:
    """Segment white card stock while closing thin printed artwork strokes."""
    roi = image[:DIVIDER_Y]
    # The official setup uses white/printed pieces on black A4 paper. Estimate
    # the dominant work-plane brightness on every frame so MuJoCo lighting
    # gradients and real camera exposure changes do not require a fixed value.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    paper_level = float(np.median(gray))
    threshold = max(70.0, paper_level + 38.0)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    # The larger close reconnects white card stock across thin black/red print
    # strokes at a cut boundary. RETR_EXTERNAL below then ignores all remaining
    # artwork holes, so only the physical card silhouette drives geometry.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _physical_piece_contours(image: np.ndarray) -> list[np.ndarray]:
    """Return ordered measured silhouettes, including real rounded arcs."""
    roi = image[:DIVIDER_Y]
    mask = _piece_foreground_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    physical = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= 1 or y <= 1 or x + w >= roi.shape[1] - 1 or y + h >= roi.shape[0] - 1:
            # Ignore the visible table/camera-rig strips outside the A4 sheet.
            continue
        physical.append(cnt.reshape(-1, 2))
    physical.sort(key=lambda p: (p.mean(0)[1], p.mean(0)[0]))
    return physical


def _match_physical_contours(
        pieces: list[np.ndarray],
        physical: list[np.ndarray]) -> list[np.ndarray]:
    """Associate measured silhouettes with recovered polygons one-to-one.

    Both lists used to be sorted independently. Recovering a virtual rounded
    corner slightly moves a polygon's vertex mean and can swap two nearby
    fragments, causing the arc from one piece to be tested/drawn on another.
    With at most four pieces, exhaustive assignment is deterministic and tiny.
    """
    if len(pieces) != len(physical):
        return physical
    piece_centers = [piece.mean(axis=0) for piece in pieces]
    contour_centers = [contour.mean(axis=0) for contour in physical]
    best = min(
        itertools.permutations(range(len(physical))),
        key=lambda order: sum(
            np.linalg.norm(piece_centers[i] - contour_centers[j])
            for i, j in enumerate(order)))
    return [physical[j] for j in best]


def detect_pieces(image: np.ndarray, expected_count: int | None = None,
                  preserve_concavity: bool = False) -> list[np.ndarray]:
    physical = _physical_piece_contours(image)
    pieces = []
    for measured in physical:
        cnt = measured.reshape(-1, 1, 2).astype(np.int32)
        # For concave mode the external contour order must be retained: sorting
        # vertices around the centroid or taking a convex hull fills the notch.
        # The supplied Joker has a white perimeter, so texture holes remain
        # interior contours and do not alter this outer boundary.
        boundary = cnt if preserve_concavity else cv2.convexHull(cnt)
        peri = cv2.arcLength(boundary, True)
        approx = cv2.approxPolyDP(
            boundary, 0.005 * peri, True).reshape(-1, 2).astype(float)
        approx = _merge_rounded_corners(approx)
        # Ordinary sharp polygons usually simplify directly to 3–5 vertices.
        # If residual print noise survives, retain the old conservative
        # approximation as a fallback rather than inventing extra cut edges.
        if len(approx) > 5:
            approx = cv2.approxPolyDP(
                boundary, 0.025 * peri, True).reshape(-1, 2).astype(float)
            approx = _merge_rounded_corners(approx)
        if 3 <= len(approx) <= 5:
            pieces.append(approx if preserve_concavity
                          else order_clockwise(approx))
    pieces.sort(key=lambda p: (p.mean(0)[1], p.mean(0)[0]))
    if not 1 <= len(pieces) <= 4:
        raise RuntimeError(f"视觉检测到 {len(pieces)} 块碎片，赛题要求为 1～4 块")
    if expected_count is not None and len(pieces) != expected_count:
        raise RuntimeError(f"视觉检测到 {len(pieces)} 块碎片，设置数量为 {expected_count}")
    return pieces


def _rounded_vertices(piece: np.ndarray,
                      measured: np.ndarray) -> list[tuple[np.ndarray, float]]:
    contour = measured.astype(np.float32).reshape(-1, 1, 2)
    result = []
    for point in piece:
        distance = abs(cv2.pointPolygonTest(
            contour, tuple(map(float, point)), True))
        if distance >= 2.0:
            result.append((point.astype(float), distance))
    return result


def _rounded_vertices_by_piece(
        pieces: list[np.ndarray], physical: list[np.ndarray],
        expected_total: int | None = None
) -> list[list[tuple[np.ndarray, float]]]:
    """Classify card-stock rounded corners jointly across all fragments."""
    candidates = []
    for piece_index, (piece, measured) in enumerate(zip(pieces, physical)):
        contour = measured.astype(np.float32).reshape(-1, 1, 2)
        for vertex_index, point in enumerate(piece):
            distance = abs(cv2.pointPolygonTest(
                contour, tuple(map(float, point)), True))
            candidates.append((
                float(distance), piece_index, vertex_index,
                point.astype(float)))
    if expected_total is None:
        selected = [candidate for candidate in candidates
                    if candidate[0] >= 2.0]
    else:
        # One complete playing card always contributes exactly four stock
        # corners, regardless of how the interior is cut. Joint ranking avoids
        # losing a shallow/one-chord arc while rejecting small approximation
        # offsets at sharp cut vertices.
        selected = sorted(candidates, reverse=True)[:expected_total]
    result = [[] for _ in pieces]
    for distance, piece_index, _vertex_index, point in selected:
        result[piece_index].append((point, max(2.0, distance)))
    return result


def _draw_measured_arc(out: np.ndarray, measured: np.ndarray,
                       virtual_corner: np.ndarray, distance: float,
                       transform: np.ndarray | None = None) -> None:
    """Draw only the local physical arc, never artwork-distorted cut edges."""
    radius = min(38.0, max(20.0, distance * 5.0))
    points = measured.astype(float)
    near = np.linalg.norm(points - virtual_corner, axis=1) <= radius
    shown = apply_h(points, transform) if transform is not None else points
    count = len(points)
    for index in range(count):
        next_index = (index + 1) % count
        if near[index] and near[next_index]:
            cv2.line(
                out, tuple(np.round(shown[index]).astype(int)),
                tuple(np.round(shown[next_index]).astype(int)),
                (255, 210, 0), 4, cv2.LINE_AA)


def edges(poly: np.ndarray):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def align_edge(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """Rigid transform mapping src_a->dst_a and src_b->dst_b."""
    u, v = src_b - src_a, dst_b - dst_a
    angle = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    r = rigid(angle, 0, 0)
    mapped = apply_h(np.array([src_a]), r)[0]
    r[:2, 2] = dst_a - mapped
    return r


def optimize_pose_graph(pieces, matches, initial):
    """Globally distribute closed-loop edge error over all movable pieces."""
    if len(pieces) < 3:
        return initial

    def pack(poses):
        values = []
        for h in poses[1:]:
            values.extend([math.atan2(h[1, 0], h[0, 0]), h[0, 2], h[1, 2]])
        return np.asarray(values, dtype=float)

    def unpack(x):
        poses = [initial[0]]
        for k in range(len(pieces) - 1):
            theta, tx, ty = x[3 * k:3 * k + 3]
            poses.append(rigid(theta, tx, ty))
        return poses

    def residual(x):
        poses = unpack(x)
        values = []
        for match in matches:
            _, i, _ei, j, _ej = match[:5]
            ia, ib, ja, jb = match_segments(pieces, match)
            wi = apply_h(np.array([ia, ib]), poses[i])
            wj = apply_h(np.array([jb, ja]), poses[j])
            values.extend((wi - wj).ravel())
        return np.asarray(values)

    x = pack(initial)
    for _ in range(20):
        r0 = residual(x)
        jac = np.empty((len(r0), len(x)))
        for k in range(len(x)):
            step = 1e-5 if k % 3 == 0 else 1e-3
            shifted = x.copy()
            shifted[k] += step
            jac[:, k] = (residual(shifted) - r0) / step
        delta, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        x += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return unpack(x)


def candidate_matchings(pieces: list[np.ndarray]):
    all_edges = {(i, e): edge for i, p in enumerate(pieces) for e, edge in enumerate(edges(p))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        la, lb = np.linalg.norm(b - a), np.linalg.norm(d - c)
        rel = abs(la - lb) / max(la, lb)
        # Rasterized shallow vertices (especially on legal five-sided pieces)
        # can shift an endpoint by several pixels, so use a tolerant shortlist;
        # final selection is decided by closure, overlap and rectangle fill.
        if rel < 0.12:
            candidates.append((rel, i, ei, j, ej, 0., 1., 0., 1.))
        ratio = min(la, lb) / max(la, lb)
        if 0.22 <= ratio <= 0.88:
            # T junction: the shorter complete edge can occupy either end of
            # a longer collinear edge. The global rectangle score resolves
            # which endpoint is the physical junction.
            # Full-edge evidence is more reliable. Keep partial candidates
            # behind all plausible full matches, then let rectangle coverage
            # select them only when a T junction genuinely requires one.
            penalty = 0.15
            if la > lb:
                candidates.extend([
                    (penalty, i, ei, j, ej, 0., ratio, 0., 1.),
                    (penalty, i, ei, j, ej, 1. - ratio, 1., 0., 1.),
                ])
            else:
                candidates.extend([
                    (penalty, i, ei, j, ej, 0., 1., 0., ratio),
                    (penalty, i, ei, j, ej, 0., 1., 1. - ratio, 1.),
                ])
    candidates.sort()
    # Preserve ambiguity *per pair of pieces*. A global top-N shortlist is
    # biased toward repeated outer-card lengths (especially rectangular or
    # near-symmetric fragments) and can discard the only true cut edge for
    # another pair. That leaves the global scorer no correct topology to
    # choose from, regardless of how strongly overlap is penalized.
    grouped: dict[tuple[int, int], list[tuple]] = {}
    for candidate in candidates:
        pair = tuple(sorted((candidate[1], candidate[3])))
        grouped.setdefault(pair, []).append(candidate)
    shortlist = []
    for group in grouped.values():
        full = [candidate for candidate in group
                if tuple(candidate[5:]) == (0., 1., 0., 1.)]
        partial = [candidate for candidate in group
                   if tuple(candidate[5:]) != (0., 1., 0., 1.)]
        # Four-sided fragments have at most 16 full edge pairings. Retaining
        # the best eight per piece pair covers measurement noise while keeping
        # the spanning-tree search bounded.
        shortlist.extend(full[:8])
        shortlist.extend(partial[:4])
    shortlist.sort()
    return shortlist


def match_segments(pieces, match):
    """Return the two full or partial edge segments encoded by a candidate."""
    _, i, ei, j, ej, ia0, ia1, ja0, ja1 = match
    a, b = edges(pieces[i])[ei]
    c, d = edges(pieces[j])[ej]
    return (a + (b - a) * ia0, a + (b - a) * ia1,
            c + (d - c) * ja0, c + (d - c) * ja1)


def matching_sets(pieces: list[np.ndarray], cut_mode: str = "auto"):
    count = len(pieces)
    if count == 1:
        yield ()
        return
    cand = candidate_matchings(pieces)
    # Every sequential straight cut adds one adjacency edge, so a connected
    # spanning tree (N-1 matches) is sufficient. Additional junction contacts
    # are evaluated implicitly by overlap and rectangular fill quality.
    pair_count = (
        count if ((cut_mode == "common" and count >= 3)
                  or (cut_mode == "concave" and count >= 2))
        else count - 1)
    full = [m for m in cand if tuple(m[5:]) == (0., 1., 0., 1.)]
    partial = [m for m in cand if tuple(m[5:]) != (0., 1., 0., 1.)]
    if cut_mode == "t_junction" and count >= 3:
        combos = (
            tuple(base) + (part,)
            for base in itertools.combinations(full, pair_count - 1)
            for part in partial
        )
    elif cut_mode in {
            "common", "boundary_fan", "strips", "corner",
            "concave", "equal_rectangles", "sequential"}:
        combos = itertools.combinations(full, pair_count)
    else:
        combos = itertools.chain(
            itertools.combinations(full, pair_count),
            (tuple(base) + (part,)
             for base in itertools.combinations(full, pair_count - 1)
             for part in partial))
    for combo in combos:
        used, degree = set(), [0] * count
        ok = True
        graph = [set() for _ in range(count)]
        for match in combo:
            _, i, ei, j, ej = match[:5]
            if (i, ei) in used or (j, ej) in used:
                ok = False
                break
            used |= {(i, ei), (j, ej)}
            degree[i] += 1
            degree[j] += 1
            graph[i].add(j)
            graph[j].add(i)
        if not ok or any(d == 0 for d in degree):
            continue
        if cut_mode == "common" and count >= 3 and any(d != 2 for d in degree):
            continue
        seen, stack = {0}, [0]
        while stack:
            for j in graph[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == count:
            yield combo


def assemble_from_matches(pieces, matches):
    adjacency = [[] for _ in pieces]
    for match in matches:
        _, i, _ei, j, _ej = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, match, reversed_sides in adjacency[i]:
            ia, ib, ja, jb = match_segments(pieces, match)
            if reversed_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            wa, wb = apply_h(np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, wb, wa)  # cutting edges meet in reverse order
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                closure_error += np.linalg.norm(apply_h(pieces[j], proposed) -
                                                apply_h(pieces[j], transforms[j]), axis=1).mean()
    assembled = [apply_h(p, h) for p, h in zip(pieces, transforms)]
    # Quality: low overlap, compact rectangular union, and graph closure.
    allp = np.vstack(assembled)
    mn, mx = allp.min(0), allp.max(0)
    scale = 1.0
    shift = -mn + 10
    w, h = np.ceil(mx - mn + 20).astype(int)
    masks = []
    for p in assembled:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p * scale + shift).astype(np.int32)], 1)
        masks.append(m)
    total = sum(masks)
    overlap = float(np.count_nonzero(total > 1))
    union = (total > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    rect_area = rw * rh
    union_area = float(np.count_nonzero(union))
    expected_area = CARD_W * CARD_H
    fill_error = max(0.0, rect_area - union_area)
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    expected_aspect = CARD_W / CARD_H
    aspect_error = abs(math.log(max(aspect, 1e-6) / expected_aspect))
    disconnected_area = sum(cv2.contourArea(c) for c in cnts) - cv2.contourArea(cnt)
    perimeter_error = abs(cv2.arcLength(cnt, True) -
                          2.0 * (CARD_W + CARD_H))
    match_error = sum(x[0] for x in matches) * 5000
    # A correct solution must cover one 10x6 rectangle. These global terms
    # reject locally plausible equal-length outer-edge matches that otherwise
    # fold pieces over each other or create an L-shaped/oversized assembly.
    score = (
        closure_error * 8
        + overlap * 12
        + fill_error * 8
        + abs(union_area - expected_area) * 4
        + abs(rect_area - expected_area) * 3
        + aspect_error * 80000
        + disconnected_area * 20
        + perimeter_error * 25
        + match_error
    )
    return score, transforms, assembled


def solve(pieces: list[np.ndarray], cut_mode: str = "auto"):
    if cut_mode == "equal_rectangles":
        count = len(pieces)
        if count == 4:
            cell_w, cell_h = CARD_W / 2, CARD_H / 2
            slots = [(0., 0.), (cell_w, 0.),
                     (0., cell_h), (cell_w, cell_h)]
        else:
            cell_w, cell_h = CARD_W / count, CARD_H
            slots = [(i * cell_w, 0.) for i in range(count)]
        target_origin = np.array([
            (CANVAS_W - CARD_W) / 2, float(TARGET_Y)])
        transforms = []
        for piece, slot in zip(pieces, slots):
            best = None
            for a, b in edges(piece):
                vector = b - a
                angle = -math.atan2(vector[1], vector[0])
                rotation = rigid(angle, 0., 0.)
                rotated = apply_h(piece, rotation)
                low, high = rotated.min(axis=0), rotated.max(axis=0)
                size = high - low
                cost = abs(size[0] - cell_w) + abs(size[1] - cell_h)
                if best is None or cost < best[0]:
                    best = (cost, rotation, low)
            _, rotation, low = best
            destination = target_origin + np.asarray(slot)
            translation = rigid(0., *(destination - low))
            transforms.append(translation @ rotation)
        # Identical blank rectangles have no observable identity. Any bijection
        # between detected pieces and equal target cells is a correct solution.
        return transforms, ()

    best = None
    for matches in matching_sets(pieces, cut_mode):
        result = assemble_from_matches(pieces, matches)
        if best is None or result[0] < best[0]:
            best = (*result, matches)
    if best is None:
        raise RuntimeError("未找到满足边长配对与碎片邻接关系的拼接")
    _, transforms, assembled, matches = best
    # A spanning tree has no loop error to distribute: its propagated edge
    # alignment is already exact. Running the periodic angle optimizer on such
    # an underconstrained graph can jump to another 2π branch and fold one
    # otherwise correct fragment over another. Refine only true closed graphs.
    if len(matches) >= len(pieces):
        transforms = optimize_pose_graph(pieces, matches, transforms)
    assembled = [apply_h(p, h) for p, h in zip(pieces, transforms)]

    # Normalize recovered rectangle to the requested lower-half target.
    allp = np.vstack(assembled).astype(np.float32)
    center, size, angle = cv2.minAreaRect(allp)
    if size[0] < size[1]:
        angle += 90.0
    normalize = rigid(math.radians(-angle), 0, 0)
    rotated = apply_h(allp, normalize)
    mn, mx = rotated.min(0), rotated.max(0)
    if (mx - mn)[0] < (mx - mn)[1]:
        normalize = rigid(math.radians(90) - math.radians(angle), 0, 0)
        rotated = apply_h(allp, normalize)
        mn, mx = rotated.min(0), rotated.max(0)
    target_origin = np.array([(CANVAS_W - CARD_W) / 2, float(TARGET_Y)])
    translate = rigid(0, *(target_origin - mn))
    final = [translate @ normalize @ h for h in transforms]
    return final, matches


def solve_textured_card(image: np.ndarray, pieces: list[np.ndarray],
                        cut_mode: str = "auto"):
    """Rigidly register printed fragments to the known Joker card artwork.

    Geometry remains responsible for segmentation and the fallback solution.
    Printed-card mode additionally uses local visual features to resolve
    geometrically equivalent edge permutations. Only camera pixels and the
    known uncut card artwork are used; generator polygons/poses are not.
    """
    reference = cv2.imread(str(CARD_ASSET), cv2.IMREAD_COLOR)
    if reference is None:
        return solve(pieces, cut_mode)
    sift = cv2.SIFT_create(
        nfeatures=3500, contrastThreshold=.018, edgeThreshold=14)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    ref_keypoints, ref_desc = sift.detectAndCompute(reference_gray, None)
    if ref_desc is None or len(ref_keypoints) < 8:
        return solve(pieces, cut_mode)

    target_origin = np.array([
        (CANVAS_W - CARD_W) / 2, float(TARGET_Y)], dtype=np.float64)
    reference_scale = np.array([
        CARD_W / reference.shape[1], CARD_H / reference.shape[0]],
        dtype=np.float64)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    transforms = []
    for piece in pieces:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillPoly(
            mask, [np.round(piece).astype(np.int32)], 255)
        keypoints, desc = sift.detectAndCompute(gray, mask)
        if desc is None or len(keypoints) < 4:
            return solve(pieces, cut_mode)
        pairs = matcher.knnMatch(desc, ref_desc, k=2)
        good = [a for a, b in pairs if a.distance < .76 * b.distance]
        if len(good) < 4:
            return solve(pieces, cut_mode)
        source = np.float32(
            [keypoints[m.queryIdx].pt for m in good])
        reference_points = np.float32(
            [ref_keypoints[m.trainIdx].pt for m in good])
        target = (
            reference_points * reference_scale + target_origin)
        affine, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC,
            ransacReprojThreshold=2.2, maxIters=5000,
            confidence=.999, refineIters=30)
        if affine is None or inliers is None or int(inliers.sum()) < 4:
            return solve(pieces, cut_mode)
        # RANSAC estimates similarity scale as well. The real fragments are
        # rigid and MuJoCo must not stretch them, so retain only its rotation
        # and refit translation from all inlier correspondences.
        linear = affine[:, :2].astype(np.float64)
        scale = math.sqrt(
            linear[0, 0] ** 2 + linear[1, 0] ** 2)
        if not .94 <= scale <= 1.06:
            return solve(pieces, cut_mode)
        rotation = linear / scale
        keep = inliers.ravel().astype(bool)
        translation = (
            target[keep]
            - source[keep] @ rotation.T).mean(axis=0)
        transform = np.eye(3)
        transform[:2, :2] = rotation
        transform[:2, 2] = translation
        transforms.append(transform)
    return transforms, ()


def analyze_camera_frame(image: np.ndarray, cut_mode: str = "auto"):
    """Pure vision/planning boundary: input is an image, output is detected poses."""
    detected = detect_pieces(
        image, preserve_concavity=(cut_mode == "concave"))
    transforms, matches = solve(detected, cut_mode)
    return detected, transforms, matches


def annotate_detection(image, pieces, expect_card_corners=False):
    out = image.copy()
    physical = _match_physical_contours(
        pieces, _physical_piece_contours(image))
    rounded_by_piece = _rounded_vertices_by_piece(
        pieces, physical, 4 if expect_card_corners else None)
    cv2.rectangle(out, (12, 10), (592, 42), (20, 20, 20), -1)
    cv2.putText(
        out, "CYAN: measured rounded edge   YELLOW: virtual geometry",
        (22, 32), cv2.FONT_HERSHEY_SIMPLEX, .52,
        (240, 240, 240), 1, cv2.LINE_AA)
    for i, p in enumerate(pieces):
        pts = np.round(p).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 220, 255), 2)
        rounded = rounded_by_piece[i] if i < len(rounded_by_piece) else []
        for corner, distance in rounded:
            _draw_measured_arc(out, physical[i], corner, distance)
        for k, pt in enumerate(pts):
            cv2.circle(out, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(out, str(k), tuple(pt + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 160), 1, cv2.LINE_AA)
        c = np.round(p.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_piece_poses(image, pieces, transforms, preserve_texture=False):
    """Render every piece at an arbitrary pose without losing card artwork."""
    out = render_scene([])
    physical = (
        _match_physical_contours(
            pieces, _physical_piece_contours(image))
        if preserve_texture else [])
    rounded_by_piece = (
        _rounded_vertices_by_piece(pieces, physical, 4)
        if preserve_texture else [[] for _ in pieces])
    foreground = None
    if preserve_texture:
        foreground = np.zeros(image.shape[:2], np.uint8)
        foreground[:DIVIDER_Y] = _piece_foreground_mask(image)
    for i, (piece, transform) in enumerate(zip(pieces, transforms)):
        target = np.round(apply_h(piece, transform)).astype(np.int32)
        if preserve_texture:
            source_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(
                source_mask, [np.round(piece).astype(np.int32)], 255)
            measured = physical[i] if i < len(physical) else None
            rounded = (
                rounded_by_piece[i]
                if measured is not None else [])
            # Keep recovered straight cut edges immune to dark artwork. Only
            # around a verified rounded corner do measured stock pixels clip
            # the otherwise virtual sharp polygon mask.
            for corner, distance in rounded:
                radius = int(round(min(38.0, max(20.0, distance * 5.0))))
                local_disk = np.zeros_like(source_mask)
                cv2.circle(
                    local_disk, tuple(np.round(corner).astype(int)),
                    radius, 255, -1)
                area = local_disk > 0
                source_mask[area] = cv2.bitwise_and(
                    source_mask, foreground)[area]
            warped_image = cv2.warpPerspective(
                image, transform, (image.shape[1], image.shape[0]))
            warped_mask = cv2.warpPerspective(
                source_mask, transform, (image.shape[1], image.shape[0]),
                flags=cv2.INTER_NEAREST)
            out[warped_mask > 127] = warped_image[warped_mask > 127]
            for corner, distance in rounded:
                _draw_measured_arc(
                    out, measured, corner, distance, transform)
        else:
            cv2.fillPoly(out, [target], PIECE_BGR)
            cv2.polylines(
                out, [target], True, (45, 45, 45), 2, cv2.LINE_AA)
        center = np.round(target.mean(axis=0)).astype(int)
        cv2.putText(
            out, f"P{i}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
            .65, (245, 245, 245), 3, cv2.LINE_AA)
        cv2.putText(
            out, f"P{i}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX,
            .65, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def render_solution(image, pieces, transforms, preserve_texture=False):
    if preserve_texture:
        out = render_piece_poses(
            image, pieces, transforms, preserve_texture=True)
        cv2.rectangle(out, (12, 10), (650, 42), (20, 20, 20), -1)
        cv2.putText(
            out, "CYAN: detected physical rounded edges",
            (22, 32), cv2.FONT_HERSHEY_SIMPLEX, .50,
            (240, 240, 240), 1, cv2.LINE_AA)
        return out
    out = image.copy()
    colors = [PIECE_BGR] * 4
    for i, (p, h) in enumerate(zip(pieces, transforms)):
        q = np.round(apply_h(p, h)).astype(np.int32)
        if preserve_texture:
            source_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(source_mask, [np.round(p).astype(np.int32)], 255)
            warped_image = cv2.warpPerspective(
                image, h, (image.shape[1], image.shape[0]))
            warped_mask = cv2.warpPerspective(
                source_mask, h, (image.shape[1], image.shape[0]))
            out[warped_mask > 127] = warped_image[warped_mask > 127]
        else:
            cv2.fillPoly(out, [q], colors[i])
        cv2.polylines(out, [q], True, (20, 20, 20), 2, cv2.LINE_AA)
        c = np.round(q.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (20, 20, 20), 2, cv2.LINE_AA)
    return out


def make_summary(scene, detected, solution):
    scale = 0.48
    panels = []
    for title, im in [("1 INPUT", scene), ("2 DETECT", detected), ("3 RESTORE", solution)]:
        x = cv2.resize(im, None, fx=scale, fy=scale)
        cv2.rectangle(x, (0, 0), (x.shape[1], 42), (255, 255, 255), -1)
        cv2.putText(x, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .75, (20, 20, 20), 2)
        panels.append(x)
    return np.hstack(panels)


def run_once(seed: int, output: Path, save=True, piece_count: int = 4):
    scene = generate_camera_frame(seed, piece_count)
    detected_pieces, transforms, matches = analyze_camera_frame(scene)
    # This assertion belongs to simulation evaluation only. It is not supplied
    # to detection or planning and cannot influence their result.
    if len(detected_pieces) != piece_count:
        raise RuntimeError(
            f"独立视觉检测到 {len(detected_pieces)} 块，仿真评测设置为 {piece_count} 块")

    # Pixel-domain reconstruction metrics.
    restored = [apply_h(p, h) for p, h in zip(detected_pieces, transforms)]
    allp = np.vstack(restored).astype(np.float32)
    rect = cv2.minAreaRect(allp)
    rw, rh = sorted(rect[1], reverse=True)
    dimension_error = max(abs(rw - CARD_W), abs(rh - CARD_H))

    if save:
        output.mkdir(parents=True, exist_ok=True)
        detected = annotate_detection(scene, detected_pieces)
        solution = render_solution(scene, detected_pieces, transforms)
        cv2.imwrite(str(output / "scene.png"), scene)
        cv2.imwrite(str(output / "detected.png"), detected)
        cv2.imwrite(str(output / "solution.png"), solution)
        cv2.imwrite(str(output / "summary.png"), make_summary(scene, detected, solution))
        records = []
        for i, (p, h) in enumerate(zip(detected_pieces, transforms)):
            angle = math.degrees(math.atan2(h[1, 0], h[0, 0]))
            records.append({
                "piece_id": i,
                "detected_center_px": p.mean(0).round(3).tolist(),
                "rotation_deg": round(angle, 6),
                "translation_px": h[:2, 2].round(6).tolist(),
                "matrix_2x3": h[:2].round(9).tolist(),
                "matrix_3x3": h.round(9).tolist(),
            })
        data = {
            "seed": seed,
            "piece_count": piece_count,
            "coordinate_system": "image pixels: x right, y down",
            "scale_px_per_cm": PIXELS_PER_CM,
            "target_rectangle_cm": {
                "width": CARD_WIDTH_CM,
                "height": CARD_HEIGHT_CM,
            },
            "target_rectangle_px": {
                "x": (CANVAS_W - CARD_W) / 2,
                "y": TARGET_Y,
                "width": CARD_W,
                "height": CARD_H,
            },
            "matched_cut_edges": [
                [int(match[1]), int(match[2]),
                 int(match[3]), int(match[4]),
                 bool(tuple(match[5:]) != (0., 1., 0., 1.))]
                for match in matches],
            "dimension_error_px": round(float(dimension_error), 4),
            "pieces": records,
        }
        (output / "transforms.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dimension_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("output/demo"), help="输出目录")
    parser.add_argument("--pieces", type=int, choices=range(2, 5), default=4,
                        help="碎片数量，范围 2～4")
    parser.add_argument("--batch", type=int, default=0, help="批量验证次数（不保存图片）")
    args = parser.parse_args()
    if args.batch:
        errors, failures = [], []
        for seed in range(args.seed, args.seed + args.batch):
            try:
                errors.append(run_once(seed, args.output, save=False, piece_count=args.pieces))
            except Exception as exc:
                failures.append((seed, str(exc)))
        print(f"批量测试: {args.batch} 次，成功 {len(errors)}，失败 {len(failures)}")
        if errors:
            print(f"矩形尺寸最大误差: {max(errors):.3f} px，平均: {np.mean(errors):.3f} px")
        if failures:
            print("失败样例:", failures[:10])
            raise SystemExit(1)
    else:
        err = run_once(args.seed, args.output, save=True, piece_count=args.pieces)
        print(f"完成。输出目录: {args.output.resolve()}")
        print(f"还原矩形尺寸误差: {err:.3f} px")
        print(f"位姿矩阵: {(args.output / 'transforms.json').resolve()}")


if __name__ == "__main__":
    main()
