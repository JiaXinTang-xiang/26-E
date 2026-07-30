from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .config import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BACKGROUND_BGR,
    DEFAULT_SCALE,
    DIVIDER_BGR,
    PAPER_EDGE_BGR,
    PIECE_EDGE_BGR,
    PIECE_SHADOW_BGR,
    SELF_PIECE_BGR,
    SPLIT_Y_MM,
    TARGET_CENTER_MM,
    WHITE_PIECE_BGR,
)
from .geometry import (
    edge_lengths,
    fan_partition,
    polygon_area,
    polygon_centroid,
    rotation_matrix,
    self_piece_polygons,
)
from .image_io import write_image


def _mm_to_px(points: np.ndarray, scale: float) -> np.ndarray:
    return np.round(np.asarray(points, dtype=np.float64) * scale).astype(np.int32)


def _blank_a4(scale: float) -> np.ndarray:
    width_px = int(round(A4_WIDTH_MM * scale))
    height_px = int(round(A4_HEIGHT_MM * scale))
    image = np.full((height_px, width_px, 3), BACKGROUND_BGR, dtype=np.uint8)
    split_px = int(round(SPLIT_Y_MM * scale))
    thickness = max(1, int(round(2.0 * scale)))
    cv2.line(
        image,
        (0, split_px),
        (width_px - 1, split_px),
        DIVIDER_BGR,
        thickness,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        image,
        (1, 1),
        (width_px - 2, height_px - 2),
        PAPER_EDGE_BGR,
        max(2, int(round(0.55 * scale))),
        cv2.LINE_AA,
    )
    return image


def _draw_suit(
    image: np.ndarray,
    center: tuple[int, int],
    size: int,
    suit: str,
    color: tuple[int, int, int],
) -> None:
    x, y = center
    size = max(size, 6)
    if suit == "diamond":
        points = np.array(
            [[x, y - size], [x + size, y], [x, y + size], [x - size, y]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(image, points, color, cv2.LINE_AA)
    elif suit == "heart":
        radius = max(2, size // 2)
        cv2.circle(image, (x - radius, y - radius // 2), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x + radius, y - radius // 2), radius, color, -1, cv2.LINE_AA)
        points = np.array(
            [[x - size, y], [x + size, y], [x, y + size + radius]], dtype=np.int32
        )
        cv2.fillConvexPoly(image, points, color, cv2.LINE_AA)
    elif suit == "club":
        radius = max(2, size // 2)
        cv2.circle(image, (x, y - radius), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x - radius, y + radius // 2), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x + radius, y + radius // 2), radius, color, -1, cv2.LINE_AA)
        cv2.rectangle(
            image,
            (x - max(1, size // 5), y + radius // 2),
            (x + max(1, size // 5), y + size + radius),
            color,
            -1,
        )
    else:  # spade
        radius = max(2, size // 2)
        cv2.circle(image, (x - radius, y + radius // 3), radius, color, -1, cv2.LINE_AA)
        cv2.circle(image, (x + radius, y + radius // 3), radius, color, -1, cv2.LINE_AA)
        points = np.array(
            [[x - size, y], [x + size, y], [x, y - size - radius]], dtype=np.int32
        )
        cv2.fillConvexPoly(image, points, color, cv2.LINE_AA)
        cv2.rectangle(
            image,
            (x - max(1, size // 5), y + radius // 2),
            (x + max(1, size // 5), y + size + radius),
            color,
            -1,
        )


def _card_texture(
    width_mm: float,
    height_mm: float,
    scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, str]]:
    width_px = int(round(width_mm * scale))
    height_px = int(round(height_mm * scale))
    texture = np.full((height_px, width_px, 3), WHITE_PIECE_BGR, dtype=np.uint8)
    suits = ["heart", "diamond", "club", "spade"]
    ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    suit = suits[int(rng.integers(0, len(suits)))]
    rank = ranks[int(rng.integers(0, len(ranks)))]
    color = (40, 40, 210) if suit in ("heart", "diamond") else (28, 28, 28)
    inset = max(5, int(round(3.0 * scale)))
    cv2.rectangle(
        texture,
        (inset, inset),
        (width_px - inset - 1, height_px - inset - 1),
        color,
        max(2, int(round(0.8 * scale))),
        cv2.LINE_AA,
    )
    font_scale = max(0.55, min(width_px, height_px) / 150.0)
    thickness = max(1, int(round(scale * 0.45)))
    cv2.putText(
        texture,
        rank,
        (inset + 3, inset + int(11 * scale)),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    text_size = cv2.getTextSize(rank, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)[0]
    cv2.putText(
        texture,
        rank,
        (width_px - inset - text_size[0] - 3, height_px - inset - 3),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    small = max(6, int(round(3.5 * scale)))
    _draw_suit(texture, (inset + small + 2, inset + int(18 * scale)), small, suit, color)
    _draw_suit(
        texture,
        (width_px - inset - small - 2, height_px - inset - int(18 * scale)),
        small,
        suit,
        color,
    )
    large = max(14, int(round(min(width_mm, height_mm) * scale * 0.22)))
    _draw_suit(texture, (width_px // 2, height_px // 2), large, suit, color)
    if rank.isdigit():
        value = min(int(rank), 10)
        rows = max(1, int(math.ceil(value / 3.0)))
        for idx in range(value):
            row = idx // 3
            col = idx % 3
            px = int(width_px * (0.34 + 0.16 * col))
            py = int(height_px * (0.25 + 0.5 * row / max(rows - 1, 1)))
            _draw_suit(texture, (px, py), max(4, small // 2), suit, color)
    return texture, {"rank": rank, "suit": suit}


def _solid_texture(
    width_mm: float, height_mm: float, scale: float, color: tuple[int, int, int]
) -> np.ndarray:
    return np.full(
        (int(round(height_mm * scale)), int(round(width_mm * scale)), 3),
        color,
        dtype=np.uint8,
    )


def _piece_mask(
    polygon_mm: np.ndarray, width_mm: float, height_mm: float, scale: float
) -> np.ndarray:
    mask = np.zeros(
        (int(round(height_mm * scale)), int(round(width_mm * scale))), dtype=np.uint8
    )
    cv2.fillPoly(mask, [_mm_to_px(polygon_mm, scale)], 255)
    return mask


def _save_piece_rgba(
    path: Path, texture: np.ndarray, mask: np.ndarray, polygon_mm: np.ndarray, scale: float
) -> None:
    polygon_px = _mm_to_px(polygon_mm, scale)
    x, y, width, height = cv2.boundingRect(polygon_px)
    pad = max(2, int(round(scale)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(texture.shape[1], x + width + pad)
    y1 = min(texture.shape[0], y + height + pad)
    rgba = cv2.cvtColor(texture[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask[y0:y1, x0:x1]
    write_image(path, rgba)


def _scatter_pieces(
    scene: np.ndarray,
    texture: np.ndarray,
    polygons_mm: list[np.ndarray],
    width_mm: float,
    height_mm: float,
    scale: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    scene_height, scene_width = scene.shape[:2]
    source_masks = [
        _piece_mask(poly, width_mm, height_mm, scale) for poly in polygons_mm
    ]
    # Collision testing does not need the 5 px/mm render resolution. A smaller
    # occupancy grid makes backtracking seeds several times faster while the
    # final image and all reported coordinates remain full resolution.
    collision_scale = min(scale, 2.0)
    occupied = np.zeros(
        (
            int(round(A4_HEIGHT_MM * collision_scale)),
            int(round(A4_WIDTH_MM * collision_scale)),
        ),
        dtype=np.uint8,
    )
    placements: list[dict[str, object] | None] = [None] * len(polygons_mm)
    order = sorted(range(len(polygons_mm)), key=lambda i: polygon_area(polygons_mm[i]), reverse=True)
    trial_count = 0
    trial_limit = 30000
    clearance_px = max(3, int(round(2.5 * collision_scale)))
    clearance_kernel = np.ones(
        (2 * clearance_px + 1,) * 2, dtype=np.uint8
    )

    def place_next(order_index: int) -> bool:
        nonlocal trial_count
        if order_index == len(order):
            return True
        piece_index = order[order_index]
        polygon = polygons_mm[piece_index]
        center_local = polygon_centroid(polygon)
        expanded = cv2.dilate(occupied, clearance_kernel)
        for _ in range(500):
            trial_count += 1
            if trial_count > trial_limit:
                return False
            angle = float(rng.uniform(-180.0, 180.0))
            rotation = rotation_matrix(angle)
            rotated_local = (polygon - center_local) @ rotation.T
            min_xy = rotated_local.min(axis=0)
            max_xy = rotated_local.max(axis=0)
            x_low = 5.0 - min_xy[0]
            x_high = A4_WIDTH_MM - 5.0 - max_xy[0]
            y_low = 5.0 - min_xy[1]
            y_high = SPLIT_Y_MM - 6.0 - max_xy[1]
            if x_low >= x_high or y_low >= y_high:
                continue
            destination_center = np.array(
                [float(rng.uniform(x_low, x_high)), float(rng.uniform(y_low, y_high))]
            )
            destination_polygon = rotated_local + destination_center
            candidate_mask = np.zeros_like(occupied)
            cv2.fillPoly(
                candidate_mask,
                [_mm_to_px(destination_polygon, collision_scale)],
                255,
            )
            if np.any((candidate_mask > 0) & (expanded > 0)):
                continue
            occupied[candidate_mask > 0] = 255
            placements[piece_index] = {
                "angle_deg": angle,
                "source_center_mm": destination_center,
                "source_polygon_mm": destination_polygon,
                "local_center_mm": center_local,
                "rotation": rotation,
            }
            if place_next(order_index + 1):
                return True
            occupied[candidate_mask > 0] = 0
            placements[piece_index] = None
        return False

    if not place_next(0):
        raise RuntimeError("Unable to scatter pieces without overlap")

    records: list[dict[str, object]] = []
    for piece_index, placement in enumerate(placements):
        assert placement is not None
        rotation = np.asarray(placement["rotation"])
        center_local = np.asarray(placement["local_center_mm"])
        destination_center = np.asarray(placement["source_center_mm"])
        source_center_px = center_local * scale
        destination_center_px = destination_center * scale
        affine = np.zeros((2, 3), dtype=np.float64)
        affine[:, :2] = rotation
        affine[:, 2] = destination_center_px - rotation @ source_center_px
        warped_texture = cv2.warpAffine(
            texture,
            affine,
            (scene_width, scene_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=BACKGROUND_BGR,
        )
        warped_mask = cv2.warpAffine(
            source_masks[piece_index],
            affine,
            (scene_width, scene_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow_offset = max(2, int(round(0.9 * scale)))
        shadow_mask = np.zeros_like(warped_mask)
        shadow_mask[
            shadow_offset:,
            shadow_offset:,
        ] = warped_mask[
            : scene_height - shadow_offset,
            : scene_width - shadow_offset,
        ]
        shadow_only = (shadow_mask > 0) & (warped_mask == 0)
        scene[shadow_only] = PIECE_SHADOW_BGR
        scene[warped_mask > 0] = warped_texture[warped_mask > 0]
        source_polygon = np.asarray(placement["source_polygon_mm"])
        cv2.polylines(
            scene,
            [_mm_to_px(source_polygon, scale)],
            True,
            PIECE_EDGE_BGR,
            max(1, int(round(scale * 0.35))),
            cv2.LINE_AA,
        )
        records.append(
            {
                "id": piece_index + 1,
                "target_polygon_local_mm": polygons_mm[piece_index].round(6).tolist(),
                "target_center_local_mm": polygon_centroid(polygons_mm[piece_index])
                .round(6)
                .tolist(),
                "source_polygon_mm": source_polygon.round(6).tolist(),
                "source_center_mm": destination_center.round(6).tolist(),
                "source_angle_deg": round(float(placement["angle_deg"]), 6),
                "edge_lengths_mm": edge_lengths(polygons_mm[piece_index])
                .round(6)
                .tolist(),
            }
        )
    return records


def generate_case(
    kind: str,
    seed: int,
    count: int = 4,
    output_root: Path | str | None = None,
    scale: float = DEFAULT_SCALE,
    save_artifacts: bool = True,
) -> Path:
    if kind not in {"self", "field-white", "field-card"}:
        raise ValueError("kind must be self, field-white, or field-card")
    if kind == "self":
        count = 4
    elif count not in (1, 2, 3, 4):
        raise ValueError("Field piece count must be between 1 and 4")

    rng = np.random.default_rng(int(seed))
    if kind == "self":
        polygons, width_mm, height_mm = self_piece_polygons()
        texture = _solid_texture(
            width_mm, height_mm, scale, SELF_PIECE_BGR
        )
        card = None
    else:
        width_mm = float(rng.integers(90, 121))
        height_mm = float(rng.integers(50, 91))
        polygons = fan_partition(width_mm, height_mm, count, rng)
        if kind == "field-white":
            texture = _solid_texture(
                width_mm, height_mm, scale, WHITE_PIECE_BGR
            )
            card = None
        else:
            texture, card = _card_texture(width_mm, height_mm, scale, rng)

    output_root = Path(output_root) if output_root is not None else Path("images")
    case_name = f"{kind.replace('-', '_')}_seed{int(seed)}_n{count}"
    case_dir = output_root / case_name
    pieces_dir = case_dir / "pieces"
    case_dir.mkdir(parents=True, exist_ok=True)
    if save_artifacts:
        pieces_dir.mkdir(parents=True, exist_ok=True)

    masks = [
        _piece_mask(poly, width_mm, height_mm, scale) for poly in polygons
    ]
    if save_artifacts:
        for index, (polygon, mask) in enumerate(zip(polygons, masks), start=1):
            _save_piece_rgba(
                pieces_dir / f"piece_{index:02d}.png",
                texture,
                mask,
                polygon,
                scale,
            )

    scene = _blank_a4(scale)
    records = _scatter_pieces(
        scene, texture, polygons, width_mm, height_mm, scale, rng
    )
    write_image(case_dir / "input.png", scene)

    target_origin = np.array(TARGET_CENTER_MM) - np.array([width_mm, height_mm]) / 2.0
    if save_artifacts:
        reference = _blank_a4(scale)
        for polygon, mask in zip(polygons, masks):
            destination_polygon = polygon + target_origin
            x0 = int(round(target_origin[0] * scale))
            y0 = int(round(target_origin[1] * scale))
            h, w = texture.shape[:2]
            roi = reference[y0 : y0 + h, x0 : x0 + w]
            roi_mask = mask[: roi.shape[0], : roi.shape[1]]
            roi[roi_mask > 0] = texture[: roi.shape[0], : roi.shape[1]][roi_mask > 0]
            cv2.polylines(
                reference,
                [_mm_to_px(destination_polygon, scale)],
                True,
                PIECE_EDGE_BGR,
                max(1, int(round(scale * 0.3))),
                cv2.LINE_AA,
            )
        write_image(case_dir / "target_reference.png", reference)

    for record, polygon in zip(records, polygons):
        target_polygon = polygon + target_origin
        record["target_polygon_mm"] = target_polygon.round(6).tolist()
        record["target_center_mm"] = polygon_centroid(target_polygon).round(6).tolist()

    manifest = {
        "kind": kind,
        "seed": int(seed),
        "count": count,
        "scale_px_per_mm": float(scale),
        "a4_size_mm": [A4_WIDTH_MM, A4_HEIGHT_MM],
        "split_y_mm": SPLIT_Y_MM,
        "target_rect": {
            "center_mm": list(TARGET_CENTER_MM),
            "width_mm": width_mm,
            "height_mm": height_mm,
        },
        "card": card,
        "pieces": records,
    }
    (case_dir / "ground_truth.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return case_dir
