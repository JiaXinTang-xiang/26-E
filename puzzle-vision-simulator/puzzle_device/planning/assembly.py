"""A4-plane assembly planning for the physical puzzle work area.

The solver consumes measured polygons only. It does not use simulator state,
piece identities, or motor commands; those belong to the GUI and calibration
layers respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math

import cv2
import numpy as np

from .template_assembly import load_self_piece_template, solve_fixed_template


@dataclass(frozen=True)
class AssemblyConfig:
    """Geometry tolerances and physical target dimensions."""

    a4_width_mm: float = 210.0
    a4_height_mm: float = 297.0
    split_fraction: float = 0.5
    minimum_target_width_mm: float = 90.0
    maximum_target_width_mm: float = 120.0
    minimum_target_height_mm: float = 50.0
    maximum_target_height_mm: float = 90.0
    target_size_tolerance_mm: float = 3.0
    candidate_size_margin_mm: float = 8.0
    # Legacy nominal size for callers that request a rectangle before solving.
    target_width_mm: float = 100.0
    target_height_mm: float = 60.0
    # Requirement 1(2) uses the same lower-half rectangle, turned end-for-end.
    # This swaps the four fixed target positions without changing the ROI or
    # pixel-to-pulse calibration.
    self_template_target_rotation_deg: float = 180.0
    # Keep the fixed self-template away from the X1-side mechanical edge.
    self_template_target_offset_x_mm: float = -8.0
    edge_relative_tolerance: float = 0.12
    # A two-piece puzzle has one complete shared cut edge. Search every
    # plausible complete-edge pair instead of dropping the true seam because
    # several unrelated outer edges happen to have closer measured lengths.
    two_piece_edge_relative_tolerance: float = 0.30
    partial_min_ratio: float = 0.22
    partial_max_ratio: float = 0.88
    candidates_per_piece_pair: int = 16
    # Every length-compatible edge pair must also form a physically plausible
    # seam after its rigid placement is proposed.
    seam_max_direction_cosine: float = -0.985
    seam_max_collinearity_error_mm: float = 1.0
    seam_min_contact_overlap_ratio: float = 0.92
    seam_max_endpoint_error_mm: float = 1.5
    seam_max_inside_normal_dot: float = -0.20
    max_states: int = 50000
    minimum_rectangle_fill_ratio: float = 0.80
    minimum_union_convexity_ratio: float = 0.90
    minimum_hull_rectangle_ratio: float = 0.87
    maximum_overlap_ratio: float = 0.03
    # Soft ranking terms. Hard geometry gates remain unchanged; these terms
    # only break ties between otherwise reliable candidates.
    dimension_score_weight: float = 180.0
    edge_match_score_weight: float = 15.0
    unexplained_edge_score_weight: float = 700.0
    boundary_line_tolerance_mm: float = 1.5
    structure_candidate_limit: int = 240
    texture_geometry_window_score: float = 80.0
    texture_score_weight: float = 150.0
    placement_gap_mm: float = 5.0
    maximum_piece_offset_mm: float = 12.0
    maximum_corresponding_vertex_distance_mm: float = 20.0
    # Kept for compatibility with older callers; range validation is authoritative.
    maximum_dimension_error_ratio: float | None = None
    # Selects the ranking/filter profile without duplicating the solver.
    solver_profile: str = "current"


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
    texture_score: float | None = None
    texture_seam_scores: tuple[float, ...] = ()
    placement_offset_directions: tuple[tuple[float, float], ...] = ()
    placement_reference_polygons: tuple[np.ndarray, ...] = ()
    enforce_corresponding_vertex_limit: bool = True


def legacy_4_0_config(config: AssemblyConfig | None = None) -> AssemblyConfig:
    """Return the 4.0 white-piece ranking/filter profile."""
    values = dict((config or AssemblyConfig()).__dict__)
    values["solver_profile"] = "legacy_4_0"
    # Keep the original node-4 geometry thresholds.  The profile switch is
    # intended to restore the old search/ranking path, not to silently change
    # the task quality criteria.
    return AssemblyConfig(**values)


def _range_error_ratio(value: float, minimum: float, maximum: float) -> float:
    if value < minimum:
        return (minimum - value) / max(minimum, 1e-6)
    if value > maximum:
        return (value - maximum) / max(maximum, 1e-6)
    return 0.0


def _target_size_error_ratio(
    size: tuple[float, float], config: AssemblyConfig
) -> float:
    width, height = size
    return (
        _range_error_ratio(
            width, config.minimum_target_width_mm, config.maximum_target_width_mm
        )
        + _range_error_ratio(
            height, config.minimum_target_height_mm, config.maximum_target_height_mm
        )
    )


def target_size_is_accepted(
    size: tuple[float, float], config: AssemblyConfig | None = None
) -> bool:
    """Return whether recovered long/short sides meet the task range."""
    cfg = config or AssemblyConfig()
    width, height = size
    tolerance = cfg.target_size_tolerance_mm
    return (
        cfg.minimum_target_width_mm - tolerance
        <= width
        <= cfg.maximum_target_width_mm + tolerance
        and cfg.minimum_target_height_mm - tolerance
        <= height
        <= cfg.maximum_target_height_mm + tolerance
    )


def _candidate_size_is_plausible(
    size: tuple[float, float], config: AssemblyConfig
) -> bool:
    """Keep camera-scale error soft while rejecting impossible target sizes."""
    width, height = size
    margin = config.candidate_size_margin_mm
    return (
        config.minimum_target_width_mm - margin
        <= width
        <= config.maximum_target_width_mm + margin
        and config.minimum_target_height_mm - margin
        <= height
        <= config.maximum_target_height_mm + margin
    )


def _geometry_is_reliable(result: tuple, config: AssemblyConfig) -> bool:
    return (
        result[4] >= config.minimum_rectangle_fill_ratio
        and result[5] >= config.minimum_union_convexity_ratio
        and result[6] >= config.minimum_hull_rectangle_ratio
        and result[7] <= config.maximum_overlap_ratio
    )


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


def _collinear_interval_on_edge(
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    other_start: np.ndarray,
    other_end: np.ndarray,
    tolerance: float,
) -> tuple[float, float] | None:
    """Return the covered fraction of one edge by a collinear segment."""
    vector = edge_end - edge_start
    other_vector = other_end - other_start
    length = float(np.linalg.norm(vector))
    other_length = float(np.linalg.norm(other_vector))
    if length <= 1e-9 or other_length <= 1e-9:
        return None
    direction = vector / length
    other_direction = other_vector / other_length
    if abs(float(direction @ other_direction)) < 0.985:
        return None
    if max(
        _point_to_line_distance(other_start, edge_start, edge_end),
        _point_to_line_distance(other_end, edge_start, edge_end),
    ) > tolerance:
        return None
    projected = sorted((
        float((other_start - edge_start) @ direction) / length,
        float((other_end - edge_start) @ direction) / length,
    ))
    start = max(0.0, projected[0])
    end = min(1.0, projected[1])
    if end - start <= 1e-6:
        return None
    return start, end


def _covered_fraction(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1e-6:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return min(1.0, sum(end - start for start, end in merged))


def _unexplained_edge_ratio(
    placed: list[np.ndarray], min_rect: tuple, config: AssemblyConfig
) -> float:
    """Measure edges that are neither target boundary nor a real piece seam."""
    rectangle = cv2.boxPoints(min_rect).astype(np.float64)
    rectangle_edges = _edges(rectangle)
    total_length = 0.0
    unexplained_length = 0.0
    tolerance = config.boundary_line_tolerance_mm

    for piece_index, polygon in enumerate(placed):
        for edge_start, edge_end in _edges(polygon):
            edge_length = float(np.linalg.norm(edge_end - edge_start))
            if edge_length <= 1e-9:
                continue
            total_length += edge_length
            intervals: list[tuple[float, float]] = []
            for boundary_start, boundary_end in rectangle_edges:
                interval = _collinear_interval_on_edge(
                    edge_start, edge_end, boundary_start, boundary_end, tolerance
                )
                if interval is not None:
                    intervals.append(interval)

            first_normal = _inside_normal(polygon, edge_start, edge_end)
            for other_index, other_polygon in enumerate(placed):
                if other_index == piece_index:
                    continue
                for other_start, other_end in _edges(other_polygon):
                    interval = _collinear_interval_on_edge(
                        edge_start, edge_end, other_start, other_end, tolerance
                    )
                    if interval is None:
                        continue
                    other_normal = _inside_normal(
                        other_polygon, other_start, other_end
                    )
                    if float(first_normal @ other_normal) <= config.seam_max_inside_normal_dot:
                        intervals.append(interval)
            unexplained_length += edge_length * (1.0 - _covered_fraction(intervals))

    return unexplained_length / max(total_length, 1e-9)


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
    # Node 4 used the same edge-candidate generation for every piece count.
    # The newer exhaustive two-piece/full-edge branch remains available only
    # to the current profile.
    two_piece_mode = len(polygons) == 2 and config.solver_profile != "legacy_4_0"
    full_edge_tolerance = (
        config.two_piece_edge_relative_tolerance
        if two_piece_mode else config.edge_relative_tolerance
    )
    for first, second in itertools.combinations(range(len(polygons)), 2):
        full = []
        partial = []
        for first_edge, (first_a, first_b) in enumerate(_edges(polygons[first])):
            first_length = float(np.linalg.norm(first_b - first_a))
            for second_edge, (second_a, second_b) in enumerate(_edges(polygons[second])):
                second_length = float(np.linalg.norm(second_b - second_a))
                relative = abs(first_length - second_length) / max(first_length, second_length, 1e-6)
                if relative <= full_edge_tolerance:
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
        if two_piece_mode:
            # For two fragments the shared cut is necessarily a full edge.
            # There are at most 25 pairs, so exhaustive comparison is cheap.
            full_limit = len(full)
            partial_limit = 0
        else:
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
    if (
        config.solver_profile != "legacy_4_0"
        and not _matched_seams_are_valid(polygons, transforms, placed, matches, config)
    ):
        return None
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
    recovered_size = (float(width), float(height))
    dimension_error = _target_size_error_ratio(recovered_size, config)

    if config.solver_profile == "legacy_4_0":
        score = (
            closure_error * 4.0
            + overlap_ratio * 5000.0
            + (1.0 - rectangle_fill_ratio) * 1000.0
            + (1.0 - union_convexity_ratio) * 500.0
            + (1.0 - hull_rectangle_ratio) * 500.0
        )
        score += sum(float(match[0]) for match in matches) * 10.0
    else:
        # Keep the hard geometry terms dominant, then use measured size and
        # edge agreement as soft tie-breakers.
        score = (
            closure_error * 4.0
            + overlap_ratio * 5000.0
            + (1.0 - rectangle_fill_ratio) * 1000.0
            + (1.0 - union_convexity_ratio) * 500.0
            + (1.0 - hull_rectangle_ratio) * 500.0
        )
        score += dimension_error * config.dimension_score_weight
        score += sum(float(match[0]) for match in matches) * config.edge_match_score_weight
    return (
        score,
        transforms,
        placed,
        recovered_size,
        float(rectangle_fill_ratio),
        float(union_convexity_ratio),
        float(hull_rectangle_ratio),
        float(overlap_ratio),
        float(dimension_error),
    )


def _point_to_line_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return math.inf
    offset = np.asarray(point, dtype=np.float64) - start
    cross = direction[0] * offset[1] - direction[1] * offset[0]
    return abs(float(cross)) / length


def _matched_seams_are_valid(
    polygons: list[np.ndarray],
    transforms: list[np.ndarray],
    placed: list[np.ndarray],
    matches: tuple,
    config: AssemblyConfig,
) -> bool:
    """Reject length-only candidates that do not make a real shared seam."""
    for match in matches:
        _, first, _first_edge, second, _second_edge, *_ = match
        first_a, first_b, second_a, second_b = _match_segments(polygons, match)
        first_a, first_b = _apply(np.asarray([first_a, first_b]), transforms[first])
        second_a, second_b = _apply(np.asarray([second_a, second_b]), transforms[second])
        first_vector = first_b - first_a
        second_vector = second_b - second_a
        first_length = float(np.linalg.norm(first_vector))
        second_length = float(np.linalg.norm(second_vector))
        if first_length <= 1e-9 or second_length <= 1e-9:
            return False
        direction_cosine = float(first_vector @ second_vector / (first_length * second_length))
        if direction_cosine > config.seam_max_direction_cosine:
            return False
        line_error = max(
            _point_to_line_distance(second_a, first_a, first_b),
            _point_to_line_distance(second_b, first_a, first_b),
        )
        if line_error > config.seam_max_collinearity_error_mm:
            return False
        direction = first_vector / first_length
        projected_second = np.array([
            float((second_a - first_a) @ direction),
            float((second_b - first_a) @ direction),
        ])
        overlap = max(
            0.0,
            min(first_length, float(projected_second.max()))
            - max(0.0, float(projected_second.min())),
        )
        if overlap / min(first_length, second_length) < config.seam_min_contact_overlap_ratio:
            return False
        endpoint_error = max(
            float(np.linalg.norm(first_a - second_b)),
            float(np.linalg.norm(first_b - second_a)),
        )
        if endpoint_error > config.seam_max_endpoint_error_mm:
            return False

        first_normal = _inside_normal(placed[first], first_a, first_b)
        second_normal = _inside_normal(placed[second], second_a, second_b)
        if float(first_normal @ second_normal) > config.seam_max_inside_normal_dot:
            return False
    return True


def _inside_normal(
    polygon: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-9)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    midpoint = (np.asarray(start) + np.asarray(end)) * 0.5
    if cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.float32),
        tuple((midpoint + normal * 0.8).astype(float)),
        False,
    ) < 0:
        normal = -normal
    return normal


def _sample_bilinear(image: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    x, y = (float(value) for value in point)
    if x < 0.0 or y < 0.0 or x >= image.shape[1] - 1 or y >= image.shape[0] - 1:
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    dx, dy = x - x0, y - y0
    top = image[y0, x0] * (1.0 - dx) + image[y0, x0 + 1] * dx
    bottom = image[y0 + 1, x0] * (1.0 - dx) + image[y0 + 1, x0 + 1] * dx
    return top * (1.0 - dy) + bottom * dy


def _texture_seam_scores(
    image: np.ndarray,
    polygons_a4: list[np.ndarray],
    matches: tuple,
    roi: tuple[int, int, int, int],
    config: AssemblyConfig,
) -> tuple[float, ...]:
    """Measure printed-pattern discontinuity across every proposed cut seam."""
    if image is None or image.ndim != 3:
        raise ValueError("扑克牌花纹匹配需要有效的彩色相机画面")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    scale = np.array(
        [roi[2] / config.a4_width_mm, roi[3] / config.a4_height_mm],
        dtype=np.float64,
    )
    origin = np.asarray(roi[:2], dtype=np.float64)
    seam_scores: list[float] = []
    for match in matches:
        _, first, _first_edge, second, _second_edge, *_ = match
        first_start, first_end, second_start, second_end = _match_segments(
            polygons_a4, match
        )
        first_normal = _inside_normal(
            polygons_a4[first], first_start, first_end
        )
        second_normal = _inside_normal(
            polygons_a4[second], second_start, second_end
        )
        differences: list[float] = []
        # Skip the vertices, where contour approximation and rounded card corners
        # are least reliable. Matching direction is reversed on the second edge.
        for fraction in np.linspace(0.08, 0.92, 17):
            first_edge_point = first_start * (1.0 - fraction) + first_end * fraction
            second_edge_point = second_end * (1.0 - fraction) + second_start * fraction
            for distance_mm in (1.0, 2.0, 3.2):
                first_point = first_edge_point + first_normal * distance_mm
                second_point = second_edge_point + second_normal * distance_mm
                first_color = _sample_bilinear(lab, origin + first_point * scale)
                second_color = _sample_bilinear(lab, origin + second_point * scale)
                if first_color is None or second_color is None:
                    continue
                # L carries most printed detail; a/b retain red and black suit cues.
                delta = np.abs(first_color - second_color)
                differences.append(
                    float(0.60 * delta[0] + 0.20 * delta[1] + 0.20 * delta[2])
                    / 255.0
                )
        seam_scores.append(float(np.mean(differences)) if differences else 1.0)
    return tuple(seam_scores)


def _choose_candidate(
    candidates: list[tuple],
    polygons_a4: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig,
    texture_image: np.ndarray | None,
    prefer_fixed_card_shape: bool,
) -> tuple[tuple, float | None, tuple[float, ...]]:
    def partial_match_count(candidate: tuple) -> int:
        return sum(
            not (
                abs(match[5]) < 1e-6
                and abs(match[6] - 1.0) < 1e-6
                and abs(match[7]) < 1e-6
                and abs(match[8] - 1.0) < 1e-6
            )
            for match in candidate[-1]
        )

    def fixed_card_penalty(candidate: tuple) -> float:
        if not prefer_fixed_card_shape:
            return 0.0
        width, height = candidate[3]
        aspect = width / max(height, 1e-6)
        expected_aspect = config.target_width_mm / config.target_height_mm
        dimension_error = (
            abs(width - config.target_width_mm) / config.target_width_mm
            + abs(height - config.target_height_mm) / config.target_height_mm
        )
        aspect_error = abs(math.log(max(aspect, 1e-6) / expected_aspect))
        return 450.0 * dimension_error + 650.0 * aspect_error

    # Full cut edges are much less ambiguous. Keep partial-edge matching only as
    # a fallback for real T-junction cuts where no reliable all-full layout exists.
    full_edge_candidates = [
        candidate for candidate in candidates if partial_match_count(candidate) == 0
    ]
    if full_edge_candidates:
        candidates = full_edge_candidates

    if texture_image is None or len(polygons_a4) == 1:
        return min(
            candidates,
            key=lambda candidate: candidate[0] + fixed_card_penalty(candidate),
        ), (
            0.0 if texture_image is not None else None
        ), ()

    # Printed white margins can make unrelated outer edges look deceptively
    # continuous. Let texture resolve only candidates that are already close
    # to the best geometric layout.
    candidates = sorted(candidates, key=lambda candidate: candidate[0])[:80]
    geometry_floor = candidates[0][0]
    candidates = [
        candidate for candidate in candidates
        if candidate[0] <= geometry_floor + config.texture_geometry_window_score
    ]
    evaluated = []
    for candidate in candidates:
        matches = candidate[-1]
        seam_scores = _texture_seam_scores(
            texture_image, polygons_a4, matches, roi, config
        )
        texture_score = float(np.mean(seam_scores)) if seam_scores else 1.0
        evaluated.append((candidate[0]
                          + texture_score * config.texture_score_weight
                          + fixed_card_penalty(candidate),
                          candidate, texture_score, seam_scores))
    _, candidate, texture_score, seam_scores = min(
        evaluated, key=lambda item: item[0]
    )
    return candidate, texture_score, seam_scores


def solve_assembly(
    polygons: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    require_upper_half: bool = True,
    texture_image: np.ndarray | None = None,
    prefer_fixed_card_shape: bool = False,
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
    best_in_range = None
    best_any_size = None
    reliable_candidates = []
    states = 0
    for matches in _matching_sets(local_polygons, cfg):
        states += 1
        if states > cfg.max_states:
            break
        result = _assemble(local_polygons, matches, cfg)
        if result is None:
            continue
        candidate = (*result, matches)
        if best_any_size is None or result[0] < best_any_size[0]:
            best_any_size = candidate
        size_ok = (
            target_size_is_accepted(result[3], cfg)
            if cfg.solver_profile == "legacy_4_0"
            else _candidate_size_is_plausible(result[3], cfg)
        )
        if not size_ok:
            continue
        if best_in_range is None or result[0] < best_in_range[0]:
            best_in_range = candidate
        if _geometry_is_reliable(result, cfg) and (
            best is None or result[0] < best[0]
        ):
            best = candidate
        if _geometry_is_reliable(result, cfg):
            reliable_candidates.append(candidate)
    if best_any_size is None:
        raise RuntimeError("未找到满足边长配对、闭合和矩形填充条件的拼接方案")
    if best_in_range is None:
        recovered_size = best_any_size[3]
        raise RuntimeError(
            "未找到题目尺寸范围内的矩形拼接：最佳几何候选尺寸 "
            f"{recovered_size[0]:.1f}×{recovered_size[1]:.1f} mm；"
            f"题目要求长边 {cfg.minimum_target_width_mm:.0f}–"
            f"{cfg.maximum_target_width_mm:.0f} mm、短边 "
            f"{cfg.minimum_target_height_mm:.0f}–"
            f"{cfg.maximum_target_height_mm:.0f} mm，视觉测量容差 "
            f"±{(cfg.target_size_tolerance_mm if cfg.solver_profile == 'legacy_4_0' else cfg.candidate_size_margin_mm):.1f} mm。请检查 A4 ROI 和轮廓顶点。"
        )
    if best is None:
        recovered_size = best_in_range[3]
        range_message = (
            "尺寸已通过题目范围校验，请检查轮廓顶点后重新锁定。"
            if cfg.solver_profile == "legacy_4_0"
            else "尺寸已通过宽松物理范围校验，请检查轮廓顶点后重新锁定。"
        )
        raise RuntimeError(
            "未找到可靠矩形拼接：最佳尺寸合法候选为 "
            f"{recovered_size[0]:.1f}×{recovered_size[1]:.1f} mm，"
            f"矩形填充率 {best_in_range[4]:.1%}，"
            f"轮廓凸度 {best_in_range[5]:.1%}，"
            f"外框矩形度 {best_in_range[6]:.1%}，"
            f"重叠率 {best_in_range[7]:.1%}。"
            + range_message
        )
    # The structural edge audit is more expensive than the rigid solve. Apply
    # it only to the strongest reliable candidates before final ranking.
    if reliable_candidates and cfg.solver_profile != "legacy_4_0":
        ordered = sorted(reliable_candidates, key=lambda candidate: candidate[0])
        audited = []
        for candidate in ordered[: max(0, cfg.structure_candidate_limit)]:
            placed = candidate[2]
            points = np.vstack(placed).astype(np.float32)
            min_rect = cv2.minAreaRect(points)
            unexplained = _unexplained_edge_ratio(placed, min_rect, cfg)
            audited.append((
                candidate[0] + unexplained * cfg.unexplained_edge_score_weight,
                *candidate[1:],
            ))
        if audited:
            reliable_candidates = audited
    best, texture_score, texture_seam_scores = _choose_candidate(
        reliable_candidates, local_polygons, tuple(int(value) for value in roi),
        cfg, texture_image, prefer_fixed_card_shape,
    )
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

    target_width, target_height = recovered_size
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
        texture_score=texture_score,
        texture_seam_scores=texture_seam_scores,
    )


def solve_textured_assembly(
    image: np.ndarray,
    polygons: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    require_upper_half: bool = True,
) -> AssemblyPlan:
    """Solve a printed-card puzzle using geometry plus seam continuity."""
    return solve_assembly(
        polygons,
        roi,
        config,
        require_upper_half=require_upper_half,
        texture_image=image,
    )


def solve_self_assembly(
    polygons: list[np.ndarray],
    roi: tuple[int, int, int, int],
    config: AssemblyConfig | None = None,
    require_upper_half: bool = True,
) -> AssemblyPlan:
    """Solve the self-prepared four pieces by fixed template, then general fallback."""
    cfg = config or AssemblyConfig()
    if len(polygons) != 4:
        raise ValueError("自备拼图要求恰好识别到4块碎片")
    local_polygons = [
        global_pixels_to_a4(np.asarray(p, dtype=np.float64), roi, cfg)
        for p in polygons
    ]
    split_y = cfg.a4_height_mm * cfg.split_fraction
    if require_upper_half:
        lower = [
            index for index, polygon in enumerate(local_polygons)
            if polygon.mean(axis=0)[1] >= split_y
        ]
        if lower:
            raise ValueError(
                "碎片不全在 A4 上半区：" + ", ".join(f"P{index}" for index in lower)
            )

    try:
        template = load_self_piece_template()
        result = solve_fixed_template(local_polygons, template)
        target_width, target_height = template.target_size_mm
        lower_height = cfg.a4_height_mm - split_y
        if target_width > cfg.a4_width_mm or target_height > lower_height:
            raise ValueError("固定模板目标尺寸超出 A4 下半区")
        target_center = np.array([
            cfg.a4_width_mm / 2.0 + cfg.self_template_target_offset_x_mm,
            split_y + lower_height / 2.0,
        ])
        target_origin = target_center - [target_width / 2.0, target_height / 2.0]
        template_center = np.array([target_width / 2.0, target_height / 2.0])
        target_rotation = math.radians(cfg.self_template_target_rotation_deg)
        # Result transforms end in template coordinates. Rotate that complete
        # shape about its own centre before placing it in the lower A4 region.
        target_pose = (
            _rigid(0.0, float(target_center[0]), float(target_center[1]))
            @ _rigid(target_rotation)
            @ _rigid(0.0, float(-template_center[0]), float(-template_center[1]))
        )
        transforms = tuple(target_pose @ transform for transform in result.transforms)
        placed = [
            _apply(polygon, transform)
            for polygon, transform in zip(local_polygons, transforms)
        ]
        overlap = sum(
            _intersection_area(placed[first], placed[second])
            for first, second in itertools.combinations(range(4), 2)
        )
        total_area = sum(
            abs(float(cv2.contourArea(polygon.astype(np.float32))))
            for polygon in placed
        )
        rect_area = max(target_width * target_height, 1.0)
        union_area = max(0.0, total_area - overlap)
        points = np.vstack(placed).astype(np.float32)
        hull_area = max(abs(float(cv2.contourArea(cv2.convexHull(points)))), 1.0)
        rectangle_fill_ratio = min(union_area / rect_area, 1.0)
        union_convexity_ratio = min(union_area / hull_area, 1.0)
        hull_rectangle_ratio = min(hull_area / rect_area, 1.0)
        overlap_ratio = overlap / max(total_area, 1.0)
        if rectangle_fill_ratio < cfg.minimum_rectangle_fill_ratio:
            raise RuntimeError(
                f"固定模板矩形填充率过低：{rectangle_fill_ratio:.1%}"
            )
        if overlap_ratio > cfg.maximum_overlap_ratio:
            raise RuntimeError(
                f"固定模板重叠率过高：{overlap_ratio:.1%}"
            )

        # Template-neighbour graph. Edge indices/fractions are diagnostic only;
        # movement planning uses the piece pairs to create the safety gap.
        template_pairs = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
        observed_by_template = {
            template_index: observed_index
            for observed_index, template_index in enumerate(result.assignment)
        }
        matches = tuple(
            (
                float(result.piece_errors[observed_by_template[first]]
                      + result.piece_errors[observed_by_template[second]]),
                observed_by_template[first], 0,
                observed_by_template[second], 0,
                0.0, 1.0, 0.0, 1.0,
            )
            for first, second in template_pairs
        )
        template_directions = tuple(
            tuple((_rigid(target_rotation)[:2, :2] @ np.asarray(direction)).tolist())
            for direction in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 0.0), (1.0, 1.0))
        )
        placement_directions = [None] * 4
        placement_references = [None] * 4
        for observed_index, template_index in enumerate(result.assignment):
            placement_directions[observed_index] = template_directions[template_index]
            placement_references[observed_index] = _apply(
                template.pieces[template_index], target_pose
            )
        return AssemblyPlan(
            roi=tuple(int(value) for value in roi),
            split_y_mm=float(split_y),
            target_rect_mm=(
                float(target_origin[0]), float(target_origin[1]),
                float(target_width), float(target_height),
            ),
            transforms=transforms,
            matches=matches,
            score=float(result.score),
            recovered_size_mm=(float(target_width), float(target_height)),
            rectangle_fill_ratio=float(rectangle_fill_ratio),
            union_convexity_ratio=float(union_convexity_ratio),
            hull_rectangle_ratio=float(hull_rectangle_ratio),
            overlap_ratio=float(overlap_ratio),
            dimension_error_ratio=0.0,
            upper_piece_ids=tuple(range(4)),
            placement_offset_directions=tuple(placement_directions),
            placement_reference_polygons=tuple(placement_references),
            enforce_corresponding_vertex_limit=False,
        )
    except (OSError, KeyError, json.JSONDecodeError, RuntimeError, ValueError):
        return solve_assembly(
            polygons,
            roi,
            cfg,
            require_upper_half=require_upper_half,
            prefer_fixed_card_shape=True,
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
