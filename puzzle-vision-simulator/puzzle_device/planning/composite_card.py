"""Experimental requirement 2(2) solver with composite/sliding seams.

This module is deliberately isolated from :mod:`assembly`.  The existing
playing-card method remains the competition default while this method is
tested from the debug page.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import itertools
import math
import multiprocessing as mp
import os
import platform
import time

import cv2
import numpy as np

from .assembly import (
    AssemblyConfig,
    AssemblyPlan,
    _align_edge,
    _apply,
    _covered_fraction,
    _edges,
    _inside_normal,
    _intersection_area,
    _match_segments,
    _prepare_texture_seam_context,
    _texture_seam_diagnostics,
    _unexplained_edge_ratio,
    global_pixels_to_a4,
    relaxed_card_config,
)


Match = tuple[float, int, int, int, int, float, float, float, float]


@dataclass(frozen=True)
class _State:
    transforms: tuple[np.ndarray | None, ...]
    matches: tuple[Match, ...]
    search_score: float


def _edge_length(edge: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.linalg.norm(edge[1] - edge[0]))


def _sliding_starts(remaining: float, samples: int = 7) -> tuple[float, ...]:
    """Return endpoint and interior positions for a short segment on a long edge."""
    if remaining <= 1e-6:
        return (0.0,)
    values = {0.0, remaining, remaining * 0.5}
    for value in np.linspace(0.0, remaining, max(2, samples)):
        values.add(float(value))
    return tuple(sorted(values))


def _composite_edge_candidates(
    polygons: list[np.ndarray], config: AssemblyConfig
) -> dict[tuple[int, int], tuple[Match, ...]]:
    """Generate full and freely sliding partial-edge hypotheses."""
    output: dict[tuple[int, int], tuple[Match, ...]] = {}
    for first, second in itertools.combinations(range(len(polygons)), 2):
        full: list[Match] = []
        partial_groups: list[list[Match]] = []
        for first_edge, edge_a in enumerate(_edges(polygons[first])):
            first_length = _edge_length(edge_a)
            for second_edge, edge_b in enumerate(_edges(polygons[second])):
                second_length = _edge_length(edge_b)
                maximum = max(first_length, second_length, 1e-6)
                relative = abs(first_length - second_length) / maximum
                ratio = min(first_length, second_length) / maximum
                if relative <= max(config.edge_relative_tolerance, 0.24):
                    full.append((
                        relative, first, first_edge, second, second_edge,
                        0.0, 1.0, 0.0, 1.0,
                    ))
                if not 0.16 <= ratio <= 0.96:
                    continue
                group: list[Match] = []
                remaining = 1.0 - ratio
                # Penalize length imbalance only softly.  A true composite
                # seam can legitimately match one long edge to two short ones.
                penalty = 0.10 + relative * 0.65
                if first_length >= second_length:
                    for start in _sliding_starts(remaining):
                        group.append((
                            penalty, first, first_edge, second, second_edge,
                            start, start + ratio, 0.0, 1.0,
                        ))
                else:
                    for start in _sliding_starts(remaining):
                        group.append((
                            penalty, first, first_edge, second, second_edge,
                            0.0, 1.0, start, start + ratio,
                        ))
                partial_groups.append(group)

        # Preserve hypotheses from every edge pair and every sliding region;
        # do not globally truncate by length error as method 1 does.
        full.sort(key=lambda match: match[0])
        selected: list[Match] = full[:24]
        for group in sorted(partial_groups, key=lambda values: values[0][0]):
            selected.extend(group)
        unique: dict[tuple, Match] = {}
        for match in selected:
            key = (
                match[1], match[2], match[3], match[4],
                *(round(value, 4) for value in match[5:]),
            )
            if key not in unique or match[0] < unique[key][0]:
                unique[key] = match
        values = sorted(unique.values(), key=lambda match: match[0])
        if values:
            output[(first, second)] = tuple(values)
    return output


def _attach_transform(
    polygons: list[np.ndarray], state: _State, match: Match
) -> tuple[int, np.ndarray] | None:
    _, first, _first_edge, second, _second_edge, *_ = match
    first_transform = state.transforms[first]
    second_transform = state.transforms[second]
    if (first_transform is None) == (second_transform is None):
        return None
    first_a, first_b, second_a, second_b = _match_segments(polygons, match)
    if first_transform is not None:
        world_a, world_b = _apply(np.asarray([first_a, first_b]), first_transform)
        return second, _align_edge(second_a, second_b, world_b, world_a)
    world_a, world_b = _apply(np.asarray([second_a, second_b]), second_transform)
    return first, _align_edge(first_a, first_b, world_b, world_a)


def _state_key(state: _State) -> tuple:
    values = []
    for index, transform in enumerate(state.transforms):
        if transform is None:
            continue
        angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        values.append((
            index,
            round(angle / 2.0),
            round(float(transform[0, 2]) / 1.5),
            round(float(transform[1, 2]) / 1.5),
        ))
    return tuple(values)


def _match_identity(match: Match) -> tuple:
    return (
        int(match[1]), int(match[2]), int(match[3]), int(match[4]),
        *(round(float(value), 4) for value in match[5:]),
    )


def _match_is_partial(match: Match) -> bool:
    return not (
        abs(match[5]) < 1e-6 and abs(match[6] - 1.0) < 1e-6
        and abs(match[7]) < 1e-6 and abs(match[8] - 1.0) < 1e-6
    )


def _state_topology_key(state: _State) -> tuple:
    """Coarse connection topology used to reserve diverse beam states."""
    piece_pairs = tuple(sorted(
        (min(int(match[1]), int(match[3])), max(int(match[1]), int(match[3])))
        for match in state.matches
    ))
    partial_count = sum(_match_is_partial(match) for match in state.matches)
    return piece_pairs, int(partial_count)


def _state_sort_key(state: _State) -> tuple:
    """Deterministic ordering independent of process completion order."""
    return (
        round(float(state.search_score), 9),
        _state_topology_key(state),
        tuple(sorted(_match_identity(match) for match in state.matches)),
        _state_key(state),
    )


def _select_stable_beam(states: list[_State], width: int) -> list[_State]:
    """Keep strong states while reserving every plausible seam topology."""
    ordered = sorted(states, key=_state_sort_key)
    if len(ordered) <= width:
        return ordered
    groups: dict[tuple, list[_State]] = {}
    for state in ordered:
        groups.setdefault(_state_topology_key(state), []).append(state)
    # Reserve representatives before filling the remaining slots globally.
    reserve_per_group = max(4, min(16, width // max(len(groups), 1)))
    selected: list[_State] = []
    selected_ids: set[int] = set()
    for key in sorted(groups):
        for state in groups[key][:reserve_per_group]:
            if len(selected) >= width:
                break
            selected.append(state)
            selected_ids.add(id(state))
    if len(selected) < width:
        for state in ordered:
            if id(state) in selected_ids:
                continue
            selected.append(state)
            if len(selected) >= width:
                break
    return sorted(selected, key=_state_sort_key)


def _partial_state_score(
    polygons: list[np.ndarray], transforms: tuple[np.ndarray | None, ...], matches: tuple[Match, ...]
) -> float:
    placed = [
        _apply(polygons[index], transform)
        for index, transform in enumerate(transforms)
        if transform is not None
    ]
    overlap = 0.0
    for first, second in itertools.combinations(placed, 2):
        first_min, first_max = first.min(axis=0), first.max(axis=0)
        second_min, second_max = second.min(axis=0), second.max(axis=0)
        if np.any(first_max <= second_min) or np.any(second_max <= first_min):
            continue
        overlap += _intersection_area(first, second)
    total_area = sum(abs(float(cv2.contourArea(p.astype(np.float32)))) for p in placed)
    points = np.vstack(placed).astype(np.float32)
    rectangle = cv2.minAreaRect(points)
    rect_area = max(float(rectangle[1][0] * rectangle[1][1]), 1.0)
    compactness = min(total_area / rect_area, 1.0)
    return (
        overlap / max(total_area, 1.0) * 6500.0
        + (1.0 - compactness) * 80.0
        + sum(float(match[0]) for match in matches) * 7.0
    )


def _search_composite_root(
    arguments: tuple[
        list[np.ndarray],
        AssemblyConfig,
        dict[tuple[int, int], tuple[Match, ...]],
        int,
        int,
    ]
) -> list[_State]:
    """Search one anchored root; top-level so Windows can pickle the worker."""
    polygons, config, candidates, root, beam_width = arguments
    count = len(polygons)
    candidates_by_piece: list[list[Match]] = [[] for _ in range(count)]
    for matches in candidates.values():
        for match in matches:
            candidates_by_piece[match[1]].append(match)
            candidates_by_piece[match[3]].append(match)
    transforms: list[np.ndarray | None] = [None] * count
    transforms[root] = np.eye(3, dtype=np.float64)
    beam = [_State(tuple(transforms), (), 0.0)]
    for _depth in range(1, count):
        next_states: dict[tuple, _State] = {}
        for state in beam:
            placed_indexes = [
                index for index, transform in enumerate(state.transforms)
                if transform is not None
            ]
            attempted: set[tuple] = set()
            for placed_index in placed_indexes:
                for match in candidates_by_piece[placed_index]:
                    match_key = (
                        match[1], match[2], match[3], match[4],
                        *(round(value, 4) for value in match[5:]),
                    )
                    if match_key in attempted:
                        continue
                    attempted.add(match_key)
                    attached = _attach_transform(polygons, state, match)
                    if attached is None:
                        continue
                    piece_index, transform = attached
                    transforms = list(state.transforms)
                    transforms[piece_index] = transform
                    transformed = tuple(transforms)
                    new_matches = state.matches + (match,)
                    score = _partial_state_score(polygons, transformed, new_matches)
                    if score >= 780.0:
                        continue
                    candidate = _State(transformed, new_matches, score)
                    key = _state_key(candidate)
                    old = next_states.get(key)
                    if old is None or score < old.search_score:
                        next_states[key] = candidate
        beam = sorted(
            next_states.values(), key=lambda state: state.search_score
        )[:beam_width]
        if not beam:
            break
    return [state for state in beam if all(t is not None for t in state.transforms)]


def _completed_state_identity(state: _State) -> tuple:
    """Identify the same physical layout found from different anchored roots.

    Relative transforms remove the arbitrary global rotation/translation of
    the root search.  Match identities remain in the key because an identical
    placement reached through a different seam topology can have a different
    texture/unexplained-edge score and must still be evaluated.
    """
    transforms = tuple(transform for transform in state.transforms if transform is not None)
    if not transforms:
        return (), ()
    inverse = np.linalg.inv(transforms[0])
    relative = []
    for transform in transforms:
        value = inverse @ transform
        angle = math.degrees(math.atan2(value[1, 0], value[0, 0]))
        relative.append((
            round(angle, 4),
            round(float(value[0, 2]), 4),
            round(float(value[1, 2]), 4),
        ))
    matches = tuple(sorted(_match_identity(match) for match in state.matches))
    return tuple(relative), matches


def _card2_timing_enabled() -> bool:
    configured = os.environ.get("PUZZLE_CARD2_TIMING", "").strip().lower()
    if configured:
        return configured not in ("0", "false", "no", "off")
    return platform.machine().lower() in ("aarch64", "arm64", "armv7l")


def _beam_layouts(
    polygons: list[np.ndarray], config: AssemblyConfig
) -> list[tuple[tuple[np.ndarray, ...], tuple[Match, ...]]]:
    count = len(polygons)
    beam_width = 150 if count >= 4 else 180
    candidates = _composite_edge_candidates(polygons, config)
    configured = os.environ.get("PUZZLE_CARD2_WORKERS", "").strip()
    if configured:
        worker_count = max(1, min(count, int(configured)))
    else:
        is_arm = platform.machine().lower() in ("aarch64", "arm64", "armv7l")
        worker_count = max(1, min(count, os.cpu_count() or 1, 2 if is_arm else 4))
    arguments = [
        (polygons, config, candidates, root, beam_width) for root in range(count)
    ]
    if worker_count == 1:
        root_results = [_search_composite_root(argument) for argument in arguments]
    else:
        # Spawn is safe when called from the GUI's planning thread on both
        # Windows and Linux/Jetson. Forking a multithreaded camera process is not.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            root_results = list(executor.map(_search_composite_root, arguments))
    states = [state for values in root_results for state in values]
    states.sort(key=lambda state: state.search_score)
    unique_states: dict[tuple, _State] = {}
    for state in states:
        key = _completed_state_identity(state)
        old = unique_states.get(key)
        if old is None or state.search_score < old.search_score:
            unique_states[key] = state
    states = sorted(unique_states.values(), key=lambda state: state.search_score)
    return [
        (tuple(transform for transform in state.transforms if transform is not None), state.matches)
        for state in states[: beam_width * 2]
    ]


def _layout_metrics(
    polygons: list[np.ndarray], transforms: tuple[np.ndarray, ...], config: AssemblyConfig
) -> tuple[float, tuple, list[np.ndarray]] | None:
    placed = [_apply(polygon, transform) for polygon, transform in zip(polygons, transforms)]
    overlap = sum(
        _intersection_area(placed[first], placed[second])
        for first, second in itertools.combinations(range(len(placed)), 2)
    )
    total_area = sum(abs(float(cv2.contourArea(p.astype(np.float32)))) for p in placed)
    points = np.vstack(placed).astype(np.float32)
    min_rect = cv2.minAreaRect(points)
    side_a, side_b = min_rect[1]
    width, height = max(side_a, side_b), min(side_a, side_b)
    rect_area = max(width * height, 1.0)
    union_area = max(0.0, total_area - overlap)
    fill = min(union_area / rect_area, 1.0)
    hull = cv2.convexHull(points)
    hull_area = max(abs(float(cv2.contourArea(hull))), 1.0)
    convexity = min(union_area / hull_area, 1.0)
    hull_rect = min(hull_area / rect_area, 1.0)
    overlap_ratio = overlap / max(total_area, 1.0)
    if overlap_ratio > 0.10 or fill < 0.55:
        return None
    unexplained = _unexplained_edge_ratio(placed, min_rect, config)
    aspect = width / max(height, 1e-6)
    aspect_error = abs(math.log(max(aspect, 1e-6) / config.card_aspect_ratio))
    # Fill/holes and unexplained boundary dominate. Card aspect remains a soft cue.
    score = (
        (1.0 - fill) * 1450.0
        + (1.0 - convexity) * 600.0
        + (1.0 - hull_rect) * 600.0
        + overlap_ratio * 7000.0
        + unexplained * 720.0
        + aspect_error * 90.0
    )
    metrics = (
        (float(width), float(height)), float(fill), float(convexity),
        float(hull_rect), float(overlap_ratio), float(unexplained),
    )
    return float(score), metrics, placed


def _layout_signature(placed: list[np.ndarray], size: tuple[float, float]) -> np.ndarray:
    centers = [polygon.mean(axis=0) for polygon in placed]
    scale = max(math.hypot(*size), 1e-6)
    return np.asarray([
        np.linalg.norm(centers[first] - centers[second]) / scale
        for first, second in itertools.combinations(range(len(centers)), 2)
    ])


def _geometry_qualified_selection_scores(evaluated: list[tuple]) -> dict[int, float]:
    """Return final selection scores for strong rectangular candidates."""
    if len(evaluated) <= 1:
        return {id(item): float(item[0]) for item in evaluated}
    best_fill = max(item[5][1] for item in evaluated)
    qualified = [
        item for item in evaluated
        if item[5][1] >= max(0.90, best_fill - 0.04)
        and item[5][2] >= 0.95
        and item[5][3] >= 0.92
        and item[5][4] <= 0.02
        and item[5][5] <= 0.22
    ]
    if len(qualified) < 2:
        return {id(item): float(item[0]) for item in evaluated}
    geometry_floor = min(item[1] for item in qualified)
    scores = {
        id(item): float(item[6]) * 1000.0
        + max(0.0, float(item[1]) - geometry_floor) * 0.15
        for item in qualified
    }
    # Geometry-ineligible layouts remain behind every qualified rectangle.
    tail_start = max(scores.values()) + 1000.0
    for tail_index, item in enumerate(
        sorted(
            (value for value in evaluated if id(value) not in scores),
            key=lambda value: value[0],
        )
    ):
        scores[id(item)] = tail_start + tail_index + float(item[0]) * 1e-3
    return scores


def _rerank_geometry_qualified_layouts(evaluated: list[tuple]) -> list[tuple]:
    """Let artwork decide only after candidates pass a strong geometry gate.

    A cut card can have two physically plausible rectangular layouts. Raw
    geometry then strongly favours the layout with the tidier measured outer
    contour even when its printed artwork is discontinuous. Within the gate,
    compress that geometry advantage and let a meaningful seam-texture
    difference decide. Weak/equal texture naturally falls back to geometry.
    """
    scores = _geometry_qualified_selection_scores(evaluated)
    return sorted(evaluated, key=lambda item: (scores[id(item)], item[0]))


def solve_composite_card_assembly(
    image: np.ndarray,
    polygons: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    require_upper_half: bool = True,
) -> AssemblyPlan:
    """Solve a card with sliding and one-long-to-many-short seam hypotheses."""
    cfg = relaxed_card_config(config)
    if not 1 <= len(polygons) <= 4:
        raise ValueError("composite card assembly expects one to four pieces")
    local = [global_pixels_to_a4(np.asarray(p, np.float64), roi, cfg) for p in polygons]
    split_y = cfg.a4_height_mm * cfg.split_fraction
    if require_upper_half and any(p.mean(axis=0)[1] >= split_y for p in local):
        raise ValueError("碎片不全在 A4 上半区")
    if len(local) == 1:
        from .assembly import solve_textured_assembly
        return solve_textured_assembly(image, polygons, roi, cfg, require_upper_half)

    search_started = time.monotonic()
    layouts = _beam_layouts(local, cfg)
    search_elapsed = time.monotonic() - search_started
    evaluation_started = time.monotonic()
    texture_context = _prepare_texture_seam_context(image, local, roi, cfg)
    texture_cache: dict[tuple, tuple[float, int]] = {}
    evaluated = []
    for transforms, matches in layouts:
        result = _layout_metrics(local, transforms, cfg)
        if result is None:
            continue
        geometry_score, metrics, placed = result
        size, fill, convexity, hull_rect, overlap, unexplained = metrics
        seam_diagnostics = _texture_seam_diagnostics(
            image,
            local,
            matches,
            roi,
            cfg,
            prepared_context=texture_context,
            cache=texture_cache,
        )
        seam_scores = tuple(score for score, _evidence in seam_diagnostics)
        seam_evidence = tuple(evidence for _score, evidence in seam_diagnostics)
        informative = [
            (score, evidence)
            for score, evidence in seam_diagnostics
            if evidence > 0
        ]
        if informative:
            texture = float(
                sum(score * evidence for score, evidence in informative)
                / sum(evidence for _score, evidence in informative)
            )
        else:
            texture = float(cfg.card_texture_neutral_score)
        # A blank/white seam is unknown, not evidence of continuity. Penalize
        # candidates that cannot verify every proposed cut with real artwork.
        missing_ratio = (
            sum(evidence == 0 for evidence in seam_evidence)
            / max(len(seam_evidence), 1)
        )
        texture += missing_ratio * 0.18
        total = (
            geometry_score
            + texture * 55.0
            + sum(match[0] for match in matches) * 5.0
        )
        evaluated.append((
            total, geometry_score, transforms, matches, placed, metrics,
            texture, seam_scores, seam_evidence,
        ))
    evaluation_elapsed = time.monotonic() - evaluation_started
    if _card2_timing_enabled():
        print(
            "[CARD2 TIMING] "
            f"pieces={len(local)} layouts={len(layouts)} valid={len(evaluated)} "
            f"unique_seams={len(texture_cache)} "
            f"search={search_elapsed:.2f}s evaluation={evaluation_elapsed:.2f}s "
            f"total={search_elapsed + evaluation_elapsed:.2f}s",
            flush=True,
        )
    if not evaluated:
        raise RuntimeError("法2未生成可用复合边拼接候选，请检查轮廓顶点")
    evaluated.sort(key=lambda item: item[0])

    # Keep representatives from different physical layouts for the debug gallery.
    diverse = []
    signatures = []
    for item in evaluated:
        signature = _layout_signature(item[4], item[5][0])
        if any(
            signature.shape == old.shape
            and float(np.sqrt(np.mean((signature - old) ** 2))) < 0.025
            and float(np.max(np.abs(signature - old))) < 0.05
            for old in signatures
        ):
            continue
        diverse.append(item)
        signatures.append(signature)
        if len(diverse) >= 12:
            break
    if not diverse:
        diverse = [evaluated[0]]
    selection_scores = _geometry_qualified_selection_scores(diverse)
    diverse = _rerank_geometry_qualified_layouts(diverse)
    diverse = diverse[:5]
    best = diverse[0]
    total, geometry_score, transforms, matches, placed, metrics, texture, seam_scores, seam_evidence = best
    size, fill, convexity, hull_rect, overlap, unexplained = metrics

    diagnostics = []
    for rank, item in enumerate(diverse, 1):
        item_total, item_geometry, _item_transforms, item_matches, item_placed, item_metrics, item_texture, item_seams, item_evidence = item
        item_size, item_fill, item_convexity, item_hull, item_overlap, item_unexplained = item_metrics
        diagnostics.append({
            "rank": rank,
            "total_score": float(item_total),
            "base_total_score": float(item_total),
            "selection_score": float(selection_scores[id(item)]),
            "family_support": 1,
            "consensus_bonus": 0.0,
            "geometry_score": float(item_geometry),
            "card_shape_penalty": 0.0,
            "structure_penalty": float(item_unexplained * 720.0),
            "texture_penalty": float(item_texture * 55.0),
            "texture_score": float(item_texture),
            "texture_seam_scores": [float(value) for value in item_seams],
            "texture_seam_evidence": [int(value) for value in item_evidence],
            "partial_match_count": sum(
                not (abs(m[5]) < 1e-6 and abs(m[6] - 1) < 1e-6
                     and abs(m[7]) < 1e-6 and abs(m[8] - 1) < 1e-6)
                for m in item_matches
            ),
            "rounded_outer_edge_match_count": 0,
            "recovered_size_mm": [float(value) for value in item_size],
            "rectangle_fill_ratio": float(item_fill),
            "union_convexity_ratio": float(item_convexity),
            "hull_rectangle_ratio": float(item_hull),
            "overlap_ratio": float(item_overlap),
            "unexplained_edge_ratio": float(item_unexplained),
            "placed_polygons_a4": [np.asarray(p).round(4).tolist() for p in item_placed],
        })

    points = np.vstack(placed).astype(np.float32)
    _center, _rect_size, angle = cv2.minAreaRect(points)
    from .assembly import _rigid
    rotation = _rigid(-math.radians(angle))
    rotated = [_apply(polygon, rotation) for polygon in placed]
    bounds = np.vstack(rotated)
    minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
    if maximum[0] - minimum[0] < maximum[1] - minimum[1]:
        rotation = _rigid(math.pi / 2.0) @ rotation
        rotated = [_apply(polygon, rotation) for polygon in placed]
        bounds = np.vstack(rotated)
        minimum, maximum = bounds.min(axis=0), bounds.max(axis=0)
    lower_height = cfg.a4_height_mm - split_y
    target_center = np.asarray([cfg.a4_width_mm / 2.0, split_y + lower_height / 2.0])
    current_center = (minimum + maximum) / 2.0
    offset = target_center - current_center
    final_transforms = tuple(_rigid(0, *offset) @ rotation @ transform for transform in transforms)
    target_rect = (
        float(target_center[0] - size[0] / 2.0),
        float(target_center[1] - size[1] / 2.0),
        float(size[0]), float(size[1]),
    )
    return AssemblyPlan(
        roi=tuple(int(value) for value in roi),
        split_y_mm=float(split_y),
        target_rect_mm=target_rect,
        transforms=final_transforms,
        matches=tuple(matches),
        score=float(total),
        recovered_size_mm=tuple(float(value) for value in size),
        rectangle_fill_ratio=float(fill),
        union_convexity_ratio=float(convexity),
        hull_rectangle_ratio=float(hull_rect),
        overlap_ratio=float(overlap),
        dimension_error_ratio=0.0,
        upper_piece_ids=tuple(range(len(polygons))),
        texture_score=float(texture),
        texture_seam_scores=tuple(float(value) for value in seam_scores),
        candidate_diagnostics=tuple(diagnostics),
    )
