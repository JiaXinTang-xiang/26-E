from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from puzzle_vision.config import load_config
from puzzle_vision.detector import PieceObservation
from puzzle_vision.geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    polygon_intersection_area,
    rotation_matrix_row,
    safe_interior_point,
)
from puzzle_vision.solver import SolveError, solve_card


CARD_WIDTH_MM = 57.0
CARD_HEIGHT_MM = 88.0
PIXELS_PER_MM = 4.0
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
DIVIDER_Y_MM = 148.5
SUPPORTED_RANKS = (
    "A",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "J",
    "Q",
    "K",
    "BJ",
    "RJ",
)
SUITS = ("club", "diamond", "heart", "spade")


def _clip_half_plane(
    polygon: np.ndarray,
    normal: np.ndarray,
    offset: float,
    keep_positive: bool,
) -> np.ndarray:
    result: list[np.ndarray] = []
    for index, current in enumerate(polygon):
        previous = polygon[index - 1]
        current_value = float(current @ normal - offset)
        previous_value = float(previous @ normal - offset)
        current_inside = (
            current_value >= -1e-8
            if keep_positive
            else current_value <= 1e-8
        )
        previous_inside = (
            previous_value >= -1e-8
            if keep_positive
            else previous_value <= 1e-8
        )
        if current_inside != previous_inside:
            ratio = previous_value / (previous_value - current_value)
            result.append(previous + ratio * (current - previous))
        if current_inside:
            result.append(current)
    return np.asarray(result, dtype=np.float64)


def _card_outline() -> np.ndarray:
    radius = 3.2
    return np.asarray(
        [
            [radius, 0.0],
            [CARD_WIDTH_MM - radius, 0.0],
            [CARD_WIDTH_MM, radius],
            [CARD_WIDTH_MM, CARD_HEIGHT_MM - radius],
            [CARD_WIDTH_MM - radius, CARD_HEIGHT_MM],
            [radius, CARD_HEIGHT_MM],
            [0.0, CARD_HEIGHT_MM - radius],
            [0.0, radius],
        ],
        dtype=np.float64,
    )


def _random_cut_pieces(
    rng: np.random.Generator, piece_count: int
) -> list[np.ndarray]:
    pieces = [_card_outline()]
    attempts = 0
    while len(pieces) < piece_count and attempts < 100:
        attempts += 1
        piece_index = int(
            np.argmax([polygon_area(piece) for piece in pieces])
        )
        polygon = pieces[piece_index]
        angle = float(rng.uniform(0.12, math.pi - 0.12))
        normal = np.asarray([math.cos(angle), math.sin(angle)])
        projections = polygon @ normal
        fraction = float(rng.uniform(0.34, 0.66))
        offset = float(
            np.min(projections)
            + fraction * (np.max(projections) - np.min(projections))
        )
        first = _clip_half_plane(polygon, normal, offset, True)
        second = _clip_half_plane(polygon, normal, offset, False)
        if min(len(first), len(second)) < 3:
            continue
        if min(polygon_area(first), polygon_area(second)) < 240.0:
            continue
        pieces[piece_index : piece_index + 1] = [
            normalize_winding(first),
            normalize_winding(second),
        ]
    if len(pieces) != piece_count:
        raise RuntimeError("Could not generate a valid straight-line card cut")
    return pieces


def _draw_club(
    image: np.ndarray,
    center_mm: tuple[float, float],
    size_mm: float,
    colour: tuple[int, int, int] = (20, 20, 20),
) -> None:
    cx = int(round(center_mm[0] * PIXELS_PER_MM))
    cy = int(round(center_mm[1] * PIXELS_PER_MM))
    radius = max(2, int(round(size_mm * PIXELS_PER_MM * 0.23)))
    cv2.circle(image, (cx, cy - radius), radius, colour, -1)
    cv2.circle(
        image, (cx - radius, cy + radius // 2), radius, colour, -1
    )
    cv2.circle(
        image, (cx + radius, cy + radius // 2), radius, colour, -1
    )
    cv2.rectangle(
        image,
        (cx - max(1, radius // 3), cy + radius // 2),
        (cx + max(1, radius // 3), cy + radius * 2),
        colour,
        -1,
    )


def _draw_suit(
    image: np.ndarray,
    suit: str,
    center_mm: tuple[float, float],
    size_mm: float,
) -> None:
    colour = (
        (25, 25, 210)
        if suit in ("heart", "diamond")
        else (20, 20, 20)
    )
    if suit == "club":
        _draw_club(image, center_mm, size_mm, colour)
        return
    cx = int(round(center_mm[0] * PIXELS_PER_MM))
    cy = int(round(center_mm[1] * PIXELS_PER_MM))
    radius = max(2, int(round(size_mm * PIXELS_PER_MM * 0.24)))
    if suit == "diamond":
        points = np.asarray(
            [
                [cx, cy - radius * 2],
                [cx + radius * 2, cy],
                [cx, cy + radius * 2],
                [cx - radius * 2, cy],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], colour)
        return
    if suit == "heart":
        cv2.circle(image, (cx - radius, cy - radius // 2), radius, colour, -1)
        cv2.circle(image, (cx + radius, cy - radius // 2), radius, colour, -1)
        points = np.asarray(
            [
                [cx - radius * 2, cy],
                [cx + radius * 2, cy],
                [cx, cy + radius * 3],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], colour)
        return
    # Spade: inverted heart plus stem.
    points = np.asarray(
        [
            [cx, cy - radius * 3],
            [cx - radius * 2, cy],
            [cx + radius * 2, cy],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [points], colour)
    cv2.circle(image, (cx - radius, cy), radius, colour, -1)
    cv2.circle(image, (cx + radius, cy), radius, colour, -1)
    cv2.rectangle(
        image,
        (cx - max(1, radius // 3), cy),
        (cx + max(1, radius // 3), cy + radius * 2),
        colour,
        -1,
    )


def _pip_positions(rank: str) -> tuple[tuple[float, float], ...]:
    positions = (
        (17.0, 17.0),
        (40.0, 17.0),
        (17.0, 34.0),
        (40.0, 34.0),
        (28.5, 44.0),
        (17.0, 54.0),
        (40.0, 54.0),
        (17.0, 71.0),
        (40.0, 71.0),
        (28.5, 25.5),
    )
    counts = {
        "A": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "10": 10,
    }
    count = counts.get(rank, 0)
    if count == 1:
        return ((28.5, 44.0),)
    return positions[:count]


def _draw_card_face(rank: str, suit: str) -> np.ndarray:
    width = int(round(CARD_WIDTH_MM * PIXELS_PER_MM))
    height = int(round(CARD_HEIGHT_MM * PIXELS_PER_MM))
    image = np.full((height, width, 3), 246, dtype=np.uint8)
    red = suit in ("heart", "diamond") or rank == "RJ"
    colour = (25, 25, 210) if red else (20, 20, 20)
    corner_text = rank
    font_scale = 0.62 if len(corner_text) > 1 else 0.85
    cv2.putText(
        image,
        corner_text,
        (int(2.5 * PIXELS_PER_MM), int(11.0 * PIXELS_PER_MM)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        2,
        cv2.LINE_AA,
    )
    if rank not in ("BJ", "RJ"):
        _draw_suit(image, suit, (7.0, 16.0), 4.5)
    # Real playing cards repeat an inverted index at the opposite corner.
    # Render that index too, so a cut through one label does not make the rank
    # unknowable.
    corner_patch = np.full(
        (int(19 * PIXELS_PER_MM), int(17 * PIXELS_PER_MM), 3),
        246,
        dtype=np.uint8,
    )
    cv2.putText(
        corner_patch,
        corner_text,
        (int(1.5 * PIXELS_PER_MM), int(10.0 * PIXELS_PER_MM)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        colour,
        2,
        cv2.LINE_AA,
    )
    if rank not in ("BJ", "RJ"):
        _draw_suit(corner_patch, suit, (6.0, 15.0), 4.5)
    inverted = cv2.rotate(corner_patch, cv2.ROTATE_180)
    patch_height, patch_width = inverted.shape[:2]
    image[
        image.shape[0] - patch_height : image.shape[0],
        image.shape[1] - patch_width : image.shape[1],
    ] = inverted
    if rank in ("J", "Q", "K"):
        cv2.rectangle(
            image,
            (int(12 * PIXELS_PER_MM), int(20 * PIXELS_PER_MM)),
            (int(45 * PIXELS_PER_MM), int(68 * PIXELS_PER_MM)),
            (215, 205, 160),
            -1,
        )
        cv2.putText(
            image,
            rank,
            (int(18 * PIXELS_PER_MM), int(57 * PIXELS_PER_MM)),
            cv2.FONT_HERSHEY_TRIPLEX,
            2.1,
            colour,
            4,
            cv2.LINE_AA,
        )
        _draw_suit(image, suit, (28.5, 31.0), 8.0)
        _draw_suit(image, suit, (28.5, 61.0), 8.0)
    elif rank in ("BJ", "RJ"):
        cv2.putText(
            image,
            "JOKER",
            (int(8 * PIXELS_PER_MM), int(48 * PIXELS_PER_MM)),
            cv2.FONT_HERSHEY_DUPLEX,
            0.85,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            (int(28.5 * PIXELS_PER_MM), int(65 * PIXELS_PER_MM)),
            int(10 * PIXELS_PER_MM),
            colour,
            3,
        )
        cv2.line(
            image,
            (int(20 * PIXELS_PER_MM), int(72 * PIXELS_PER_MM)),
            (int(37 * PIXELS_PER_MM), int(58 * PIXELS_PER_MM)),
            colour,
            3,
        )
    else:
        for position in _pip_positions(rank):
            _draw_suit(image, suit, position, 8.0)
    return image


def _render_scrambled_case(
    pieces: list[np.ndarray],
    rng: np.random.Generator,
    rank: str,
    suit: str,
    glare_mode: str = "clear",
) -> tuple[np.ndarray, list[PieceObservation], list[dict[str, Any]]]:
    ppm = PIXELS_PER_MM
    canvas = np.full(
        (
            int(round(A4_HEIGHT_MM * ppm)),
            int(round(A4_WIDTH_MM * ppm)),
            3,
        ),
        (185, 130, 50),
        dtype=np.uint8,
    )
    card = _draw_card_face(rank, suit)
    source_outline = np.rint(_card_outline() * ppm).astype(np.int32)
    card_alpha = np.zeros(card.shape[:2], dtype=np.uint8)
    cv2.fillPoly(card_alpha, [source_outline], 255)
    observations: list[PieceObservation] = []
    truth: list[dict[str, Any]] = []
    placed_polygons: list[np.ndarray] = []
    for index, polygon in enumerate(pieces):
        source_center = polygon_centroid(polygon)
        selected: tuple[
            float, np.ndarray, np.ndarray, np.ndarray
        ] | None = None
        for _ in range(500):
            angle = float(rng.uniform(-math.pi, math.pi))
            rotation = rotation_matrix_row(angle)
            local = (polygon - source_center) @ rotation
            lower = np.min(local, axis=0)
            upper = np.max(local, axis=0)
            margin = 4.0
            x_low = margin - float(lower[0])
            x_high = A4_WIDTH_MM - margin - float(upper[0])
            y_low = margin - float(lower[1])
            y_high = DIVIDER_Y_MM - margin - float(upper[1])
            if x_low >= x_high or y_low >= y_high:
                continue
            target_center = np.asarray(
                [
                    rng.uniform(x_low, x_high),
                    rng.uniform(y_low, y_high),
                ],
                dtype=np.float64,
            )
            candidate = local + target_center
            if any(
                polygon_intersection_area(candidate, other) > 0.1
                for other in placed_polygons
            ):
                continue
            selected = (angle, rotation, target_center, candidate)
            break
        if selected is None:
            raise RuntimeError(
                "Could not place simulated fragments without clipping/overlap"
            )
        angle, rotation, target_center, scrambled = selected
        translation = target_center - source_center @ rotation
        scrambled = normalize_winding(scrambled)
        placed_polygons.append(scrambled)

        piece_alpha = np.zeros(card.shape[:2], dtype=np.uint8)
        cv2.fillPoly(
            piece_alpha,
            [np.rint(polygon * ppm).astype(np.int32)],
            255,
        )
        source = cv2.bitwise_and(card, card, mask=piece_alpha)
        affine = np.asarray(
            [
                [
                    rotation[0, 0],
                    rotation[1, 0],
                    translation[0] * ppm,
                ],
                [
                    rotation[0, 1],
                    rotation[1, 1],
                    translation[1] * ppm,
                ],
            ],
            dtype=np.float64,
        )
        warped = cv2.warpAffine(
            source,
            affine,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        warped_alpha = cv2.warpAffine(
            piece_alpha,
            affine,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
        )
        canvas[warped_alpha > 0] = warped[warped_alpha > 0]
        # Reflection can erase the printed face without erasing the physical
        # outline.  A fully saturated fragment models the difficult real
        # camera case.  In partial mode alternating fragments retain their
        # artwork, exercising per-piece pattern/outline priority rather than
        # a single global decision.
        glare_applied = glare_mode == "full" or (
            glare_mode == "partial" and index % 2 == 1
        )
        if glare_applied:
            canvas[warped_alpha > 0] = (255, 255, 255)

        lengths = edge_lengths(scrambled)
        observations.append(
            PieceObservation(
                id=f"piece_{index + 1}",
                polygon_mm=scrambled,
                contour_px=np.rint(scrambled * ppm)
                .astype(np.int32)
                .reshape(-1, 1, 2),
                centroid_mm=polygon_centroid(scrambled),
                pickup_mm=safe_interior_point(scrambled),
                area_mm2=polygon_area(polygon),
                perimeter_mm=float(np.sum(lengths)),
                edge_lengths_mm=lengths,
            )
        )
        truth.append(
            {
                "piece_id": f"piece_{index + 1}",
                "source_polygon_mm": np.round(polygon, 4).tolist(),
                "scramble_rotation_deg": round(math.degrees(angle), 4),
                "glare_applied": glare_applied,
            }
        )
    observations.sort(
        key=lambda item: (item.pickup_mm[1], item.pickup_mm[0])
    )
    for index, observation in enumerate(observations, start=1):
        observation.id = f"piece_{index}"
    cv2.line(
        canvas,
        (0, int(round(DIVIDER_Y_MM * ppm))),
        (canvas.shape[1] - 1, int(round(DIVIDER_Y_MM * ppm))),
        (0, 0, 0),
        max(2, int(round(1.0 * ppm))),
    )
    return canvas, observations, truth


def run_simulation(
    cases: int,
    seed: int,
    output_dir: Path,
    config_path: str,
    ranks: tuple[str, ...] = SUPPORTED_RANKS,
    glare_modes: tuple[str, ...] = ("clear",),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    passed = 0
    for case_index in range(cases):
        piece_count = 2 + case_index % 3
        rank = ranks[(case_index // 3) % len(ranks)]
        suit = SUITS[case_index % len(SUITS)]
        if rank == "BJ":
            suit = "spade"
        elif rank == "RJ":
            suit = "heart"
        glare_mode = glare_modes[case_index % len(glare_modes)]
        pieces = _random_cut_pieces(rng, piece_count)
        image, observations, truth = _render_scrambled_case(
            pieces, rng, rank, suit, glare_mode
        )
        unknown = dict(config["unknown"])
        unknown["target_zone_mm"] = [
            0.0,
            DIVIDER_Y_MM,
            A4_WIDTH_MM,
            A4_HEIGHT_MM,
        ]
        try:
            plan, information = solve_card(
                observations,
                unknown,
                image,
                PIXELS_PER_MM,
            )
            ok = bool(information.get("solution_accepted", False))
            error = None
        except SolveError as exc:
            plan = []
            information = {}
            ok = False
            error = str(exc)
        passed += int(ok)
        recognized = information.get("card_recognition", {}).get("rank")
        recognition_match = (
            recognized == rank
            or {str(recognized), rank} == {"6", "9"}
        )
        case = {
            "case": case_index + 1,
            "piece_count": piece_count,
            "passed": ok,
            "rank_truth": rank,
            "suit_truth": suit,
            "glare_mode": glare_mode,
            "rank_recognized": recognized,
            "recognition_match": recognition_match,
            "error": error,
            "solver": information,
            "plan": plan,
            "truth": truth,
        }
        results.append(case)
        cv2.imwrite(
            str(output_dir / f"card_sim_{case_index + 1:03d}.jpg"),
            image,
        )
        (output_dir / f"card_sim_{case_index + 1:03d}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    summary = {
        "cases": cases,
        "passed": passed,
        "pass_rate": round(passed / max(cases, 1), 6),
        "seed": seed,
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "target_orientation": "portrait",
        "supported_ranks": list(ranks),
        "recognition_correct": sum(
            int(item["recognition_match"]) for item in results
        ),
        "recognition_rate": round(
            sum(int(item["recognition_match"]) for item in results)
            / max(cases, 1),
            6,
        ),
        "glare_modes": list(glare_modes),
        "by_glare_mode": {
            glare_mode: {
                "cases": sum(
                    int(item["glare_mode"] == glare_mode)
                    for item in results
                ),
                "passed": sum(
                    int(
                        item["glare_mode"] == glare_mode
                        and item["passed"]
                    )
                    for item in results
                ),
                "geometry_only": sum(
                    int(
                        item["glare_mode"] == glare_mode
                        and item.get("solver", {})
                        .get("card_recognition", {})
                        .get("fusion_strategy")
                        == "geometry_only"
                    )
                    for item in results
                ),
                "hybrid_partial_pattern": sum(
                    int(
                        item["glare_mode"] == glare_mode
                        and item.get("solver", {})
                        .get("card_recognition", {})
                        .get("fusion_strategy")
                        == "hybrid_partial_pattern"
                    )
                    for item in results
                ),
                "geometry_plus_pattern": sum(
                    int(
                        item["glare_mode"] == glare_mode
                        and item.get("solver", {})
                        .get("card_recognition", {})
                        .get("fusion_strategy")
                        == "geometry_plus_pattern"
                    )
                    for item in results
                ),
            }
            for glare_mode in glare_modes
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate and solve straight-cut playing-card puzzles"
    )
    parser.add_argument("--cases", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", default="playing_card_simulation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument(
        "--full-ranks",
        action="store_true",
        help="Run A, 2-10, J/Q/K and both Jokers",
    )
    parser.add_argument(
        "--glare-cycle",
        action="store_true",
        help=(
            "Cycle clear, partially glare-obscured and fully "
            "glare-obscured fragments"
        ),
    )
    args = parser.parse_args()
    summary = run_simulation(
        max(1, args.cases),
        args.seed,
        Path(args.output),
        args.config,
        SUPPORTED_RANKS if args.full_ranks else ("9",),
        ("clear", "partial", "full")
        if args.glare_cycle
        else ("clear",),
    )
    print(
        json.dumps(
            {
                "cases": summary["cases"],
                "passed": summary["passed"],
                "pass_rate": summary["pass_rate"],
                "recognition_rate": summary["recognition_rate"],
                "by_glare_mode": summary["by_glare_mode"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
