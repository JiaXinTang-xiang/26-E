from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .detector import PieceObservation


_RANKS = tuple("A2345678901JQKBR")
_FONTS = (
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
)


@lru_cache(maxsize=1)
def _rank_templates() -> tuple[tuple[str, np.ndarray], ...]:
    templates: list[tuple[str, np.ndarray]] = []
    for rank in _RANKS:
        for font in _FONTS:
            canvas = np.zeros((100, 100), dtype=np.uint8)
            (width, height), _ = cv2.getTextSize(rank, font, 2.0, 4)
            cv2.putText(
                canvas,
                rank,
                ((100 - width) // 2, (100 + height) // 2),
                font,
                2.0,
                255,
                4,
                cv2.LINE_AA,
            )
            contours, _ = cv2.findContours(
                canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                templates.append(
                    (rank, max(contours, key=cv2.contourArea))
                )
    return tuple(templates)


def _component_holes(component: np.ndarray) -> int:
    contours, hierarchy = cv2.findContours(
        component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None or not contours:
        return 0
    return sum(1 for item in hierarchy[0] if int(item[3]) >= 0)


def _classify_rank(component: np.ndarray) -> tuple[str, float] | None:
    contours, _ = cv2.findContours(
        component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    scores: dict[str, float] = {}
    for rank, template in _rank_templates():
        value = float(
            cv2.matchShapes(
                contour, template, cv2.CONTOURS_MATCH_I1, 0.0
            )
        )
        scores[rank] = min(scores.get(rank, float("inf")), value)
    ordered = sorted((value, rank) for rank, value in scores.items())
    if not ordered or ordered[0][0] > 0.16:
        return None
    best_score, best_rank = ordered[0]
    # Six and nine are the same silhouette under a half turn.  Keep the
    # camera-visible glyph result; the final portrait card orientation may
    # rotate the whole completed card by 180 degrees without mirroring it.
    confidence = max(0.0, min(1.0, 1.0 - best_score / 0.20))
    return best_rank, confidence


def _rounded_corner_frames(
    polygon: np.ndarray, config: dict[str, Any]
) -> list[dict[str, np.ndarray]]:
    points = np.asarray(polygon, dtype=np.float64)
    count = len(points)
    frames: list[dict[str, np.ndarray]] = []
    minimum_chord = float(config.get("card_rounded_chord_min_mm", 2.0))
    maximum_chord = float(config.get("card_rounded_chord_max_mm", 8.0))
    maximum_error = float(
        config.get("card_rounded_corner_angle_error_deg", 15.0)
    )
    for index in range(count):
        start = points[index]
        end = points[(index + 1) % count]
        chord_length = float(np.linalg.norm(end - start))
        if not minimum_chord <= chord_length <= maximum_chord:
            continue
        incoming = start - points[(index - 1) % count]
        outgoing = points[(index + 2) % count] - end
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if min(incoming_length, outgoing_length) < maximum_chord:
            continue
        first = incoming / incoming_length
        second = outgoing / outgoing_length
        line_angle = np.degrees(
            np.arccos(np.clip(abs(float(first @ second)), 0.0, 1.0))
        )
        if abs(90.0 - float(line_angle)) > maximum_error:
            continue
        matrix = np.column_stack((first, -second))
        if abs(float(np.linalg.det(matrix))) < 1e-6:
            continue
        coefficient = np.linalg.solve(matrix, end - start)[0]
        corner = start + coefficient * first
        centroid_direction = np.mean(points, axis=0) - corner
        first_inward = (
            first if float(first @ centroid_direction) >= 0.0 else -first
        )
        second_inward = (
            second if float(second @ centroid_direction) >= 0.0 else -second
        )
        frames.append(
            {
                "corner": corner,
                "first_inward": first_inward,
                "second_inward": second_inward,
            }
        )
    return frames


@lru_cache(maxsize=4)
def _corner_label_templates(
    pixels_per_mm_key: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    ppm = pixels_per_mm_key / 100.0
    height = int(round(14.0 * ppm))
    width = int(round(17.0 * ppm))
    templates: list[tuple[str, np.ndarray]] = []
    for rank in (
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
    ):
        canvas = np.zeros((height, width), dtype=np.uint8)
        scale = 0.62 if len(rank) > 1 else 0.85
        cv2.putText(
            canvas,
            rank,
            (int(round(2.5 * ppm)), int(round(11.0 * ppm))),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            255,
            2,
            cv2.LINE_AA,
        )
        templates.append((rank, np.uint8(canvas > 80) * 255))
    return tuple(templates)


def _recognize_corner_patch(
    gray: np.ndarray,
    frame: dict[str, np.ndarray],
    pixels_per_mm: float,
) -> tuple[str, float] | None:
    ppm = float(pixels_per_mm)
    height = int(round(14.0 * ppm))
    width = int(round(17.0 * ppm))
    x_values = np.arange(width, dtype=np.float32) / ppm
    y_values = np.arange(height, dtype=np.float32) / ppm
    grid_x, grid_y = np.meshgrid(x_values, y_values)
    corner = frame["corner"]
    axes = (
        (frame["first_inward"], frame["second_inward"]),
        (frame["second_inward"], frame["first_inward"]),
    )
    best: tuple[float, str] | None = None
    for first_axis, second_axis in axes:
        source_x = (
            corner[0]
            + grid_x * first_axis[0]
            + grid_y * second_axis[0]
        ) * ppm
        source_y = (
            corner[1]
            + grid_x * first_axis[1]
            + grid_y * second_axis[1]
        ) * ppm
        patch = cv2.remap(
            gray,
            source_x.astype(np.float32),
            source_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        for oriented in (patch, cv2.rotate(patch, cv2.ROTATE_180)):
            paper_level = float(np.percentile(oriented, 75))
            ink = np.uint8(
                oriented < min(175.0, paper_level - 35.0)
            ) * 255
            if cv2.countNonZero(ink) < max(6, int(0.15 * ppm * ppm)):
                continue
            padded = cv2.copyMakeBorder(
                ink, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0
            )
            for rank, template in _corner_label_templates(
                int(round(ppm * 100.0))
            ):
                response = cv2.matchTemplate(
                    padded, template, cv2.TM_CCOEFF_NORMED
                )
                score = float(np.max(response))
                if best is None or score > best[0]:
                    best = (score, rank)
    if best is None or best[0] < 0.18:
        return None
    return best[1], max(0.0, min(1.0, best[0]))


def recognize_card_marks(
    rectified_image: np.ndarray,
    observations: list[PieceObservation],
    pixels_per_mm: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recognize printed rank marks on separated playing-card fragments.

    The implementation uses only OpenCV so it runs unchanged on the PC, RDK
    X5 and K230 Linux builds.  It deliberately does not require Tesseract.
    """

    cfg = config or {}
    gray = cv2.cvtColor(rectified_image, cv2.COLOR_BGR2GRAY)
    ppm = float(pixels_per_mm)
    minimum_area = float(cfg.get("card_mark_min_area_mm2", 0.35)) * ppm * ppm
    maximum_area = float(cfg.get("card_mark_max_area_mm2", 140.0)) * ppm * ppm
    rank_candidates: list[dict[str, Any]] = []
    piece_marks: list[dict[str, Any]] = []
    visible_piece_count = 0
    glare_limited_piece_count = 0

    for piece_index, observation in enumerate(observations):
        rounded_frames = _rounded_corner_frames(
            observation.polygon_mm, cfg
        )
        for frame in rounded_frames:
            corner_result = _recognize_corner_patch(
                gray, frame, ppm
            )
            if corner_result is None:
                continue
            corner_rank, corner_confidence = corner_result
            # A rectified corner is useful only when its whole index is
            # legible.  Glare and clipped artwork can otherwise create a
            # weak two-character match (for example a club being read as
            # "RJ") whose negative selection score would incorrectly
            # override a stronger connected-component digit candidate.
            if corner_confidence < float(
                cfg.get("card_corner_patch_min_confidence", 0.60)
            ):
                continue
            rank_candidates.append(
                {
                    "center_mm": np.round(frame["corner"], 3).tolist(),
                    "bbox_mm": None,
                    "area_mm2": 0.0,
                    "holes": 0,
                    "piece_id": observation.id,
                    "piece_index": piece_index,
                    "rank": corner_rank,
                    "confidence": round(corner_confidence, 4),
                    "corner_distance_mm": 0.0,
                    "rounded_corner_reference": True,
                    "selection_score": round(
                        -4.0 * corner_confidence, 4
                    ),
                    "recognition_method": "rectified_corner_patch",
                }
            )
        piece_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(
            piece_mask,
            [np.asarray(observation.contour_px, dtype=np.int32)],
            -1,
            255,
            thickness=cv2.FILLED,
        )
        # Remove the dark cut outline and its shadow before looking for ink.
        erosion = max(1, int(round(0.7 * ppm)))
        interior = cv2.erode(
            piece_mask,
            np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8),
        )
        values = gray[interior > 0]
        if values.size == 0:
            piece_marks.append(
                {
                    "piece_id": observation.id,
                    "marks": [],
                    "pattern_visible": False,
                    "glare_limited": True,
                    "overexposed_ratio": 1.0,
                    "contrast_std_gray": 0.0,
                    "recognition_priority": "geometry_only",
                }
            )
            glare_limited_piece_count += 1
            continue
        paper_level = float(np.median(values))
        overexposed_ratio = float(np.mean(values >= 248))
        contrast_std = float(np.std(values))
        threshold = min(
            float(cfg.get("card_black_ink_max_gray", 155.0)),
            paper_level
            - float(cfg.get("card_black_ink_contrast_gray", 48.0)),
        )
        ink = np.uint8((interior > 0) & (gray < threshold)) * 255
        ink = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            ink
        )
        marks: list[dict[str, Any]] = []
        for label in range(1, count):
            x, y, width, height, area = (
                int(value) for value in stats[label]
            )
            if not minimum_area <= area <= maximum_area:
                continue
            component = np.uint8(labels == label) * 255
            center_px = np.asarray(centroids[label], dtype=np.float64)
            mark = {
                "center_mm": np.round(center_px / ppm, 3).tolist(),
                "bbox_mm": [
                    round(x / ppm, 3),
                    round(y / ppm, 3),
                    round(width / ppm, 3),
                    round(height / ppm, 3),
                ],
                "area_mm2": round(area / (ppm * ppm), 3),
                "holes": _component_holes(component),
            }
            classified = _classify_rank(component)
            if classified is not None:
                rank, confidence = classified
                rank_component_plausible = bool(
                    area / (ppm * ppm)
                    >= float(
                        cfg.get(
                            "card_rank_component_min_area_mm2", 1.2
                        )
                    )
                    and max(width, height) / ppm
                    >= float(
                        cfg.get(
                            "card_rank_component_min_span_mm", 2.0
                        )
                    )
                )
                corner_distance_mm = float(
                    np.min(
                        np.linalg.norm(
                            np.asarray(
                                [
                                    frame["corner"]
                                    for frame in rounded_frames
                                ]
                                if rounded_frames
                                else observation.polygon_mm,
                                dtype=np.float64,
                            )
                            - center_px[None, :] / ppm,
                            axis=1,
                        )
                    )
                )
                candidate = {
                    **mark,
                    "piece_id": observation.id,
                    "piece_index": piece_index,
                    "rank": rank,
                    "confidence": round(confidence, 4),
                    "corner_distance_mm": round(corner_distance_mm, 3),
                    # Merely having a rounded corner somewhere on the same
                    # fragment does not make every dark component a corner
                    # index.  On J/Q/K cards the portrait contains many
                    # letter-like components; treating one of those as a
                    # corner anchor can force an otherwise impossible pose.
                    "rounded_corner_reference": bool(
                        rounded_frames
                        and corner_distance_mm
                        <= float(
                            cfg.get(
                                "card_component_corner_reference_max_mm",
                                14.0,
                            )
                        )
                    ),
                }
                candidate["selection_score"] = round(
                    abs(float(candidate["corner_distance_mm"]) - 9.0)
                    + 3.0 * (1.0 - confidence),
                    4,
                )
                if rank_component_plausible:
                    rank_candidates.append(candidate)
                    mark["rank_candidate"] = rank
                    mark["rank_confidence"] = round(confidence, 4)
            marks.append(mark)
        ink_ratio = float(cv2.countNonZero(ink)) / max(
            1, cv2.countNonZero(interior)
        )
        pattern_visible = bool(
            marks
            or ink_ratio
            >= float(cfg.get("card_minimum_pattern_pixel_ratio", 0.0015))
        )
        glare_limited = bool(
            overexposed_ratio
            >= float(cfg.get("card_glare_overexposed_ratio", 0.55))
            and not pattern_visible
        )
        visible_piece_count += int(pattern_visible)
        glare_limited_piece_count += int(glare_limited)
        piece_marks.append(
            {
                "piece_id": observation.id,
                "marks": marks,
                "pattern_visible": pattern_visible,
                "glare_limited": glare_limited,
                "overexposed_ratio": round(overexposed_ratio, 5),
                "contrast_std_gray": round(contrast_std, 3),
                "pattern_pixel_ratio": round(ink_ratio, 6),
                "recognition_priority": (
                    "rank_and_geometry"
                    if any("rank_candidate" in mark for mark in marks)
                    else "pattern_and_geometry"
                    if pattern_visible
                    else "geometry_only"
                ),
            }
        )

    # Multi-character corner indices are composed after single-glyph
    # recognition.  Distance is rotation invariant, so this remains valid
    # while fragments are randomly oriented.
    grouped: list[dict[str, Any]] = []
    for first_index, first in enumerate(rank_candidates):
        for second in rank_candidates[first_index + 1 :]:
            if first["piece_index"] != second["piece_index"]:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(first["center_mm"], dtype=np.float64)
                    - np.asarray(second["center_mm"], dtype=np.float64)
                )
            )
            if distance > float(cfg.get("card_grouped_rank_distance_mm", 12.0)):
                continue
            pair = {str(first["rank"]), str(second["rank"])}
            grouped_rank: str | None = None
            if pair == {"1", "0"}:
                grouped_rank = "10"
            elif pair == {"B", "J"}:
                grouped_rank = "BJ"
            elif pair == {"R", "J"}:
                grouped_rank = "RJ"
            if grouped_rank is None:
                continue
            confidence = min(
                float(first["confidence"]),
                float(second["confidence"]),
            )
            grouped.append(
                {
                    "center_mm": np.round(
                        (
                            np.asarray(first["center_mm"], dtype=np.float64)
                            + np.asarray(second["center_mm"], dtype=np.float64)
                        )
                        * 0.5,
                        3,
                    ).tolist(),
                    "bbox_mm": first["bbox_mm"],
                    "area_mm2": round(
                        float(first["area_mm2"])
                        + float(second["area_mm2"]),
                        3,
                    ),
                    "holes": int(first["holes"]) + int(second["holes"]),
                    "piece_id": first["piece_id"],
                    "piece_index": first["piece_index"],
                    "rank": grouped_rank,
                    "confidence": round(confidence, 4),
                    "corner_distance_mm": round(
                        min(
                            float(first["corner_distance_mm"]),
                            float(second["corner_distance_mm"]),
                        ),
                        3,
                    ),
                    "selection_score": round(
                        min(
                            float(first["selection_score"]),
                            float(second["selection_score"]),
                        )
                        - 1.0,
                        4,
                    ),
                    "components": [
                        str(first["rank"]),
                        str(second["rank"]),
                    ],
                    "rounded_corner_reference": bool(
                        first.get("rounded_corner_reference", False)
                        or second.get("rounded_corner_reference", False)
                    ),
                }
            )
    rank_candidates.extend(grouped)
    minimum_component_confidence = float(
        cfg.get("card_component_rank_min_confidence", 0.45)
    )
    # The auxiliary glyph alphabet contains 0/1/B/R only so that 10 and the
    # two Jokers can be composed.  Never publish those glyphs alone as a card
    # rank, and discard weak connected-component guesses instead of letting
    # reflections or suit pips report a plausible-looking J/0.
    rank_candidates = [
        item
        for item in rank_candidates
        if (
            str(item["rank"]) in {"10", "BJ", "RJ"}
            or (
                str(item["rank"])
                not in {"0", "1", "B", "R"}
                and (
                    item.get("recognition_method")
                    == "rectified_corner_patch"
                    or float(item["confidence"])
                    >= minimum_component_confidence
                )
            )
        )
    ]
    rounded_candidates = [
        item
        for item in rank_candidates
        if bool(item.get("rounded_corner_reference", False))
    ]
    if rounded_candidates:
        rank_candidates = rounded_candidates
    rank_candidates.sort(
        key=lambda item: (
            float(item["selection_score"]),
            -float(item["confidence"]),
            str(item["piece_id"]),
        )
    )
    best = rank_candidates[0] if rank_candidates else None
    visibility_fraction = visible_piece_count / max(len(observations), 1)
    if visible_piece_count == 0:
        strategy = "geometry_only"
    elif visible_piece_count == len(observations):
        strategy = "geometry_plus_pattern"
    else:
        strategy = "hybrid_partial_pattern"
    return {
        "rank_detected": best is not None,
        "rank": best["rank"] if best is not None else None,
        "rank_confidence": (
            float(best["confidence"]) if best is not None else 0.0
        ),
        "rank_piece_id": (
            str(best["piece_id"]) if best is not None else None
        ),
        "rank_piece_index": (
            int(best["piece_index"]) if best is not None else None
        ),
        "rank_rounded_corner_reference": (
            bool(best.get("rounded_corner_reference", False))
            if best is not None
            else False
        ),
        "rank_recognition_method": (
            str(best.get("recognition_method", "connected_component"))
            if best is not None
            else None
        ),
        "rank_center_mm": (
            list(best["center_mm"]) if best is not None else None
        ),
        "candidates": rank_candidates,
        "pieces": piece_marks,
        "pattern_visible_pieces": visible_piece_count,
        "glare_limited_pieces": glare_limited_piece_count,
        "pattern_visibility_fraction": round(visibility_fraction, 6),
        "fusion_strategy": strategy,
        "priority_order": [
            "non_overlap_and_outline",
            "rounded_card_corners",
            "visible_rank_suit_pattern",
            "seam_texture",
        ],
        "recognizer": "opencv_shape_rank",
        "supported_ranks": [
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
        ],
    }
