"""A4-plane assembly planning for the physical puzzle work area.

The solver consumes measured polygons only. It does not use simulator state,
piece identities, or motor commands; those belong to the GUI and calibration
layers respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class AssemblyConfig:
    """Geometry tolerances and physical target dimensions."""

    a4_width_mm: float = 210.0
    a4_height_mm: float = 297.0
    split_fraction: float = 0.5
    target_width_mm: float = 100.0
    target_height_mm: float = 60.0
    edge_relative_tolerance: float = 0.12
    partial_min_ratio: float = 0.22
    partial_max_ratio: float = 0.88
    candidates_per_piece_pair: int = 16
    max_states: int = 50000
    minimum_rectangle_fill_ratio: float = 0.82
    minimum_union_convexity_ratio: float = 0.92
    minimum_hull_rectangle_ratio: float = 0.90
    maximum_overlap_ratio: float = 0.02
    placement_gap_mm: float = 5.0
    maximum_piece_offset_mm: float = 12.0
    maximum_corresponding_vertex_distance_mm: float = 20.0
    # Diagnostic only: physical size no longer rejects a rectangular result.
    maximum_dimension_error_ratio: float | None = None


@dataclass(frozen=True)
class AssemblyPlan:
    """Solved source-to-target transforms and diagnostics in A4 millimetres."""

    roi: tuple[int, int, int, int]
    split_y_mm: float
    target_rect_mm: tuple[float, float, float, float]
    transforms: tuple[np.ndarray, ...]
    matches: tuple[tuple[float, int, int, int, int, float, float, float, float], ...]
    score: float
    recovered_size_mm: tuple[float, float]
    rectangle_fill_ratio: float
    union_convexity_ratio: float
    hull_rectangle_ratio: float
    overlap_ratio: float
    dimension_error_ratio: float
    upper_piece_ids: tuple[int, ...]


def _edges(polygon: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (polygon[index], polygon[(index + 1) % len(polygon)])
        for index in range(len(polygon))
    ]


def _rigid(angle: float, tx: float = 0.0, ty: float = 0.0) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def _apply(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    values = np.c_[np.asarray(points, dtype=np.float64), np.ones(len(points))]
    return (values @ transform.T)[:, :2]


def _align_edge(source_a, source_b, target_a, target_b) -> np.ndarray:
    source_vector = np.asarray(source_b) - np.asarray(source_a)
    target_vector = np.asarray(target_b) - np.asarray(target_a)
    angle = math.atan2(target_vector[1], target_vector[0]) - math.atan2(
        source_vector[1], source_vector[0]
    )
    transform = _rigid(angle)
    mapped = _apply(np.asarray([source_a]), transform)[0]
    transform[:2, 2] = np.asarray(target_a) - mapped
    return transform


def _intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    first32 = np.asarray(first, dtype=np.float32)
    second32 = np.asarray(second, dtype=np.float32)
    if cv2.isContourConvex(first32) and cv2.isContourConvex(second32):
        area, _ = cv2.intersectConvexConvex(first32, second32)
        return float(area)
    points = np.vstack([first32, second32])
    minimum = np.floor(points.min(axis=0) - 1).astype(int)
    maximum = np.ceil(points.max(axis=0) + 1).astype(int)
    scale = 2.0
    shape = np.maximum(np.ceil((maximum - minimum) * scale).astype(int) + 3, 3)
    masks = []
    for polygon in (first32, second32):
        mask = np.zeros((shape[1], shape[0]), dtype=np.uint8)
        pixels = np.round((polygon - minimum) * scale).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 255)
        masks.append(mask)
    return float(np.count_nonzero((masks[0] > 0) & (masks[1] > 0))) / scale**2


def _match_segments(polygons, match):
    _, first, first_edge, second, second_edge, first_start, first_end, second_start, second_end = match
    first_a, first_b = _edges(polygons[first])[first_edge]
    second_a, second_b = _edges(polygons[second])[second_edge]
    return (
        first_a + (first_b - first_a) * first_start,
        first_a + (first_b - first_a) * first_end,
        second_a + (second_b - second_a) * second_start,
        second_a + (second_b - second_a) * second_end,
    )


def _edge_candidates(polygons, config: AssemblyConfig):
    candidates_by_pair = {}
    for first, second in itertools.combinations(range(len(polygons)), 2):
        full = []
        partial = []
        for first_edge, (first_a, first_b) in enumerate(_edges(polygons[first])):
            first_length = float(np.linalg.norm(first_b - first_a))
            for second_edge, (second_a, second_b) in enumerate(_edges(polygons[second])):
                second_length = float(np.linalg.norm(second_b - second_a))
                relative = abs(first_length - second_length) / max(first_length, second_length, 1e-6)
                if relative <= config.edge_relative_tolerance:
                    full.append((relative, first, first_edge, second, second_edge, 0.0, 1.0, 0.0, 1.0))
                ratio = min(first_length, second_length) / max(first_length, second_length, 1e-6)
                if config.partial_min_ratio <= ratio <= config.partial_max_ratio:
                    penalty = 0.15 + relative
                    if first_length >= second_length:
                        partial.extend([
                            (penalty, first, first_edge, second, second_edge, 0.0, ratio, 0.0, 1.0),
                            (penalty, first, first_edge, second, second_edge, 1.0 - ratio, 1.0, 0.0, 1.0),
                        ])
                    else:
                        partial.extend([
                            (penalty, first, first_edge, second, second_edge, 0.0, 1.0, 0.0, ratio),
                            (penalty, first, first_edge, second, second_edge, 0.0, 1.0, 1.0 - ratio, 1.0),
                        ])
        full.sort()
        partial.sort()
        full_limit = min(len(full), max(1, config.candidates_per_piece_pair // 4))
        partial_limit = max(0, config.candidates_per_piece_pair - full_limit)

        # Keep candidates across the whole length-ratio range. The correct
        # physical cut can have a lower ratio than several plausible but wrong
        # edges, so taking only the numerically smallest penalties is unsafe.
        partial_pairs = [partial[index:index + 2] for index in range(0, len(partial), 2)]
        pair_limit = partial_limit // 2
        if len(partial_pairs) <= pair_limit:
            selected_pairs = partial_pairs
        elif pair_limit > 1:
            indexes = sorted({
                round(index * (len(partial_pairs) - 1) / (pair_limit - 1))
                for index in range(pair_limit)
            })
            selected_pairs = [partial_pairs[index] for index in indexes]
        elif pair_limit == 1:
            selected_pairs = [partial_pairs[len(partial_pairs) // 2]]
        else:
            selected_pairs = []
        selected = full[:full_limit] + [
            candidate for pair in selected_pairs for candidate in pair
        ]
        selected.sort()
        if selected:
            candidates_by_pair[(first, second)] = selected
    return candidates_by_pair


def _matching_sets(polygons, config: AssemblyConfig):
    count = len(polygons)
    if count == 1:
        yield ()
        return
    candidates_by_pair = _edge_candidates(polygons, config)
    pair_count = count - 1
    for piece_pairs in itertools.combinations(candidates_by_pair, pair_count):
        pair_graph = [set() for _ in polygons]
        for first, second in piece_pairs:
            pair_graph[first].add(second)
            pair_graph[second].add(first)
        seen, stack = {0}, [0]
        while stack:
            for neighbour in pair_graph[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        if len(seen) != count:
            continue
        for combo in itertools.product(*(candidates_by_pair[pair] for pair in piece_pairs)):
            occupied_segments = {}
            graph = [set() for _ in polygons]
            valid = True
            for match in combo:
                _, first, first_edge, second, second_edge, first_start, first_end, second_start, second_end = match
                for key, interval in (
                    ((first, first_edge), (first_start, first_end)),
                    ((second, second_edge), (second_start, second_end)),
                ):
                    old_intervals = occupied_segments.setdefault(key, [])
                    if any(
                        min(interval[1], old[1]) - max(interval[0], old[0]) > 0.05
                        for old in old_intervals
                    ):
                        valid = False
                        break
                    old_intervals.append(interval)
                if not valid:
                    break
                graph[first].add(second)
                graph[second].add(first)
            if not valid or any(not neighbours for neighbours in graph):
                continue
            if valid:
                yield combo


def _assemble(polygons, matches, config: AssemblyConfig):
    adjacency = [[] for _ in polygons]
    for match in matches:
        _, first, _first_edge, second, _second_edge, *_ = match
        adjacency[first].append((second, match, False))
        adjacency[second].append((first, match, True))
    transforms: list[np.ndarray | None] = [None] * len(polygons)
    transforms[0] = np.eye(3)
    closure_error = 0.0
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbour, match, reversed_sides in adjacency[current]:
            first_a, first_b, second_a, second_b = _match_segments(polygons, match)
            if reversed_sides:
                first_a, first_b, second_a, second_b = second_a, second_b, first_a, first_b
            world_a, world_b = _apply(np.asarray([first_a, first_b]), transforms[current])
            proposed = _align_edge(second_a, second_b, world_b, world_a)
            if transforms[neighbour] is None:
                transforms[neighbour] = proposed
                stack.append(neighbour)
            else:
                proposed_points = _apply(polygons[neighbour], proposed)
                current_points = _apply(polygons[neighbour], transforms[neighbour])
                closure_error += float(np.linalg.norm(proposed_points - current_points, axis=1).mean())
    if any(transform is None for transform in transforms):
        return None
    placed = [_apply(polygon, transform) for polygon, transform in zip(polygons, transforms)]
    overlap = sum(
        _intersection_area(placed[first], placed[second])
        for first, second in itertools.combinations(range(len(placed)), 2)
    )
    points = np.vstack(placed).astype(np.float32)
    min_rect = cv2.minAreaRect(points)
    side_a, side_b = min_rect[1]
    width, height = max(side_a, side_b), min(side_a, side_b)
    total_area = sum(abs(float(cv2.contourArea(p.astype(np.float32)))) for p in placed)
    rect_area = max(width * height, 1.0)
    union_area = max(0.0, total_area - overlap)
    rectangle_fill_ratio = min(union_area / rect_area, 1.0)
    hull = cv2.convexHull(points)
    hull_area = max(abs(float(cv2.contourArea(hull))), 1.0)
    union_convexity_ratio = min(union_area / hull_area, 1.0)
    hull_rectangle_ratio = min(hull_area / rect_area, 1.0)
    overlap_ratio = overlap / max(total_area, 1.0)
    dimension_error = (
        abs(width - config.target_width_mm) / config.target_width_mm
        + abs(height - config.target_height_mm) / config.target_height_mm
    )
    aspect = width / max(height, 1e-6)
    expected_aspect = config.target_width_mm / config.target_height_mm
    aspect_error = abs(math.log(max(aspect, 1e-6) / expected_aspect))

    # Absolute millimetres are diagnostic only. The 5:3 aspect remains useful
    # because a 200x30 strip is rectangular but is not the required card shape.
    score = (
        closure_error * 4.0
        + overlap_ratio * 5000.0
        + (1.0 - rectangle_fill_ratio) * 1000.0
        + (1.0 - union_convexity_ratio) * 500.0
        + (1.0 - hull_rectangle_ratio) * 500.0
        + aspect_error * 500.0
    )
    score += sum(float(match[0]) for match in matches) * 10.0
    return (
        score,
        transforms,
        placed,
        (float(width), float(height)),
        float(rectangle_fill_ratio),
        float(union_convexity_ratio),
        float(hull_rectangle_ratio),
        float(overlap_ratio),
        float(dimension_error),
    )


def solve_assembly(
    polygons: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    require_upper_half: bool = True,
) -> AssemblyPlan:
    """Solve measured polygons and place the recovered rectangle in ROI's lower half."""
    cfg = config or AssemblyConfig()
    if not 1 <= len(polygons) <= 4:
        raise ValueError("assembly expects one to four pieces")
    roi_x, roi_y, roi_width, roi_height = (int(value) for value in roi)
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI must be a positive rectangle")
    local_polygons = [
        global_pixels_to_a4(np.asarray(p, dtype=np.float64), roi, cfg)
        for p in polygons
    ]
    split_y = cfg.a4_height_mm * cfg.split_fraction
    if require_upper_half:
        lower = [index for index, polygon in enumerate(local_polygons) if polygon.mean(axis=0)[1] >= split_y]
        if lower:
            raise ValueError(
                "碎片不全在 A4 上半区：" + ", ".join(f"P{index}" for index in lower)
            )
    best = None
    states = 0
    for matches in _matching_sets(local_polygons, cfg):
        states += 1
        if states > cfg.max_states:
            break
        result = _assemble(local_polygons, matches, cfg)
        if result is not None and (best is None or result[0] < best[0]):
            best = (*result, matches)
    if best is None:
        raise RuntimeError("未找到满足边长配对、闭合和矩形填充条件的拼接方案")
    (
        score,
        transforms,
        placed,
        recovered_size,
        rectangle_fill_ratio,
        union_convexity_ratio,
        hull_rectangle_ratio,
        overlap_ratio,
        dimension_error,
        matches,
    ) = best

    if (
        rectangle_fill_ratio < cfg.minimum_rectangle_fill_ratio
        or union_convexity_ratio < cfg.minimum_union_convexity_ratio
        or hull_rectangle_ratio < cfg.minimum_hull_rectangle_ratio
        or overlap_ratio > cfg.maximum_overlap_ratio
    ):
        raise RuntimeError(
            "未找到可靠矩形拼接：最佳候选尺寸 "
            f"{recovered_size[0]:.1f}×{recovered_size[1]:.1f} mm，"
            f"矩形填充率 {rectangle_fill_ratio:.1%}，"
            f"轮廓凸度 {union_convexity_ratio:.1%}，"
            f"外框矩形度 {hull_rectangle_ratio:.1%}，"
            f"重叠率 {overlap_ratio:.1%}。"
            "绝对尺寸不参与淘汰，请检查轮廓顶点后重新锁定。"
        )

    # Normalize the recovered assembly to an axis-aligned target rectangle.
    all_points = np.vstack(placed).astype(np.float32)
    center, size, angle = cv2.minAreaRect(all_points)
    rotation = _rigid(-math.radians(angle))
    rotated = [_apply(polygon, rotation) for polygon in placed]
    bounds = np.vstack(rotated)
    minimum = bounds.min(axis=0)
    maximum = bounds.max(axis=0)
    if maximum[0] - minimum[0] < maximum[1] - minimum[1]:
        rotation = _rigid(math.pi / 2.0) @ rotation
        rotated = [_apply(polygon, rotation) for polygon in placed]
        bounds = np.vstack(rotated)
        minimum = bounds.min(axis=0)
        maximum = bounds.max(axis=0)

    target_width = cfg.target_width_mm
    target_height = cfg.target_height_mm
    lower_height = cfg.a4_height_mm - split_y
    if target_width > cfg.a4_width_mm or target_height > lower_height:
        raise ValueError("目标矩形尺寸超出 A4 下半区")
    target_center = np.array([
        cfg.a4_width_mm / 2.0,
        split_y + lower_height / 2.0,
    ])
    current_center = (minimum + maximum) / 2.0
    # Keep the recovered scale: only translate/rotate. The target rectangle is
    # a validation target and defines the lower-half destination center.
    offset = target_center - current_center
    final_transforms = tuple(_rigid(0, offset[0], offset[1]) @ rotation @ transform
                             for transform in transforms)
    target_rect = (
        target_center[0] - target_width / 2.0,
        target_center[1] - target_height / 2.0,
        target_width,
        target_height,
    )
    return AssemblyPlan(
        roi=tuple(int(value) for value in roi),
        split_y_mm=float(split_y),
        target_rect_mm=tuple(float(value) for value in target_rect),
        transforms=final_transforms,
        matches=tuple(matches),
        score=float(score),
        recovered_size_mm=recovered_size,
        rectangle_fill_ratio=float(rectangle_fill_ratio),
        union_convexity_ratio=float(union_convexity_ratio),
        hull_rectangle_ratio=float(hull_rectangle_ratio),
        overlap_ratio=float(overlap_ratio),
        dimension_error_ratio=float(dimension_error),
        upper_piece_ids=tuple(range(len(polygons))),
    )


def global_pixels_to_a4(
    points: np.ndarray,
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
) -> np.ndarray:
    """Map global camera pixels into the axis-aligned A4 millimetre plane."""
    cfg = config or AssemblyConfig()
    roi_x, roi_y, roi_width, roi_height = roi
    values = np.asarray(points, dtype=np.float64) - [roi_x, roi_y]
    return values * [cfg.a4_width_mm / roi_width, cfg.a4_height_mm / roi_height]


def a4_to_global_pixels(
    points: np.ndarray,
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
) -> np.ndarray:
    """Map A4 millimetres back into global camera pixels."""
    cfg = config or AssemblyConfig()
    roi_x, roi_y, roi_width, roi_height = roi
    values = np.asarray(points, dtype=np.float64)
    return values * [roi_width / cfg.a4_width_mm, roi_height / cfg.a4_height_mm] + [roi_x, roi_y]


def transform_global_points(
    points: np.ndarray,
    plan: AssemblyPlan,
    transform: np.ndarray,
    config: AssemblyConfig | None = None,
) -> np.ndarray:
    """Apply an A4-plane rigid transform to global camera pixel points."""
    a4_points = global_pixels_to_a4(points, plan.roi, config)
    return a4_to_global_pixels(_apply(a4_points, transform), plan.roi, config)
