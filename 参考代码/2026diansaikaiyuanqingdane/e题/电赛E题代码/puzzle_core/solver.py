from __future__ import annotations

import math
import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from .config import TARGET_CENTER_MM
from .geometry import (
    edge_lengths,
    normalize_angle,
    polygon_area,
    rigid_edge_alignment,
    rotation_matrix,
    self_piece_polygons,
)
from .models import DetectedPiece, Placement, SolvedLayout


@dataclass
class _Candidate:
    placements: dict[int, Placement]
    geometry_score: float
    width_mm: float
    height_mm: float
    texture_score: float = 0.0


def _convex_vertices(points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return hull.reshape(-1, 2).astype(np.float64)


def _best_rigid_correspondence(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    source = _convex_vertices(source)
    target = _convex_vertices(target)
    if len(source) != len(target):
        return 1e9, np.eye(2), np.zeros(2)
    best = (1e9, np.eye(2), np.zeros(2))
    for reverse in (False, True):
        ordered = target[::-1] if reverse else target
        for shift in range(len(ordered)):
            destination = np.roll(ordered, shift, axis=0)
            source_center = source.mean(axis=0)
            target_center = destination.mean(axis=0)
            src = source - source_center
            dst = destination - target_center
            u, _, vt = np.linalg.svd(src.T @ dst)
            rotation = vt.T @ u.T
            if np.linalg.det(rotation) < 0:
                vt[-1, :] *= -1
                rotation = vt.T @ u.T
            translation = target_center - rotation @ source_center
            predicted = source @ rotation.T + translation
            error = float(
                np.sqrt(np.mean(np.sum((predicted - destination) ** 2, axis=1)))
            )
            if error < best[0]:
                best = (error, rotation, translation)
    return best


def _solve_self_template(
    pieces: list[DetectedPiece],
) -> _Candidate:
    if len(pieces) != 4:
        raise RuntimeError("The self-prepared puzzle requires exactly four pieces")
    templates, _, _ = self_piece_polygons()
    pair_results: dict[tuple[int, int], tuple[float, np.ndarray, np.ndarray]] = {}
    for piece_index, piece in enumerate(pieces):
        for template_index, template in enumerate(templates):
            pair_results[(piece_index, template_index)] = _best_rigid_correspondence(
                piece.local_polygon_mm, template
            )
    best_cost = float("inf")
    best_placements: dict[int, Placement] | None = None
    for assignment in itertools.permutations(range(4)):
        cost = sum(
            pair_results[(piece_index, assignment[piece_index])][0]
            for piece_index in range(4)
        )
        if cost >= best_cost:
            continue
        placements: dict[int, Placement] = {}
        for piece_index, template_index in enumerate(assignment):
            _, rotation, translation = pair_results[(piece_index, template_index)]
            placements[piece_index] = Placement(piece_index, rotation, translation)
        best_cost = cost
        best_placements = placements
    if best_placements is None or best_cost > 8.0:
        raise RuntimeError("Unable to match the four detected pieces to the fixed template")
    score, width, height = _layout_score(pieces, best_placements, "self")
    return _Candidate(best_placements, score, width, height)


def _edge_points(polygon: np.ndarray, edge_index: int) -> tuple[np.ndarray, np.ndarray]:
    return polygon[edge_index], polygon[(edge_index + 1) % len(polygon)]


def _intersection_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = np.asarray(poly_a, dtype=np.float32)
    b = np.asarray(poly_b, dtype=np.float32)
    if cv2.isContourConvex(a) and cv2.isContourConvex(b):
        area, _ = cv2.intersectConvexConvex(a, b)
        return float(area)
    all_points = np.vstack([a, b]).astype(np.float64)
    min_xy = np.floor(all_points.min(axis=0) - 1.0)
    max_xy = np.ceil(all_points.max(axis=0) + 1.0)
    scale = 4.0
    size = np.maximum(np.ceil((max_xy - min_xy) * scale).astype(int) + 1, 2)
    masks = []
    for polygon in (a, b):
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        pixels = np.round((polygon - min_xy) * scale).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 255)
        mask = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
        masks.append(mask)
    return float(np.count_nonzero((masks[0] > 0) & (masks[1] > 0))) / (
        scale * scale
    )


def _layout_polygons(
    pieces: list[DetectedPiece], placements: dict[int, Placement]
) -> dict[int, np.ndarray]:
    return {
        index: placement.transform(pieces[index].local_polygon_mm)
        for index, placement in placements.items()
    }


def _layout_score(
    pieces: list[DetectedPiece],
    placements: dict[int, Placement],
    mode: str,
) -> tuple[float, float, float]:
    polygons = _layout_polygons(pieces, placements)
    overlap = 0.0
    keys = sorted(polygons)
    for a_pos, key_a in enumerate(keys):
        for key_b in keys[a_pos + 1 :]:
            overlap += _intersection_area(polygons[key_a], polygons[key_b])
    all_points = np.vstack([polygons[key] for key in keys]).astype(np.float32)
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(all_points)
    width, height = sorted((float(side_a), float(side_b)), reverse=True)
    rectangle_area = max(width * height, 1e-6)
    pieces_area = sum(polygon_area(poly) for poly in polygons.values())
    fill_error = abs(rectangle_area - pieces_area) / rectangle_area
    overlap_ratio = overlap / max(pieces_area, 1e-6)
    if mode == "self":
        dimension_penalty = abs(width - 100.0) / 100.0 + abs(height - 60.0) / 60.0
    else:
        width_error = max(0.0, 90.0 - width, width - 120.0) / 30.0
        height_error = max(0.0, 50.0 - height, height - 90.0) / 40.0
        dimension_penalty = width_error + height_error
    score = 100.0 * fill_error + 300.0 * overlap_ratio + 20.0 * dimension_penalty
    return score, width, height


def _edge_match_allowed(length_a: float, length_b: float) -> bool:
    return abs(length_a - length_b) <= max(2.5, 0.055 * min(length_a, length_b))


def _search_candidates(
    pieces: list[DetectedPiece], mode: str, max_states: int = 40000
) -> list[_Candidate]:
    if len(pieces) == 1:
        placement = Placement(0, np.eye(2), np.zeros(2))
        score, width, height = _layout_score(pieces, {0: placement}, mode)
        return [_Candidate({0: placement}, score, width, height)]

    edge_sizes = [edge_lengths(piece.local_polygon_mm) for piece in pieces]
    candidates: list[_Candidate] = []
    state_counter = 0
    identity = Placement(0, np.eye(2), np.zeros(2))

    def recurse(
        placements: dict[int, Placement],
        used_edges: set[tuple[int, int]],
    ) -> None:
        nonlocal state_counter
        if state_counter >= max_states:
            return
        state_counter += 1
        if len(placements) == len(pieces):
            score, width, height = _layout_score(pieces, placements, mode)
            if mode == "self":
                dimension_ok = 92.0 <= width <= 108.0 and 52.0 <= height <= 68.0
            else:
                dimension_ok = 82.0 <= width <= 128.0 and 42.0 <= height <= 98.0
            if dimension_ok and score < 60.0:
                candidates.append(
                    _Candidate(dict(placements), score, width, height)
                )
            return

        placed_polygons = _layout_polygons(pieces, placements)
        unplaced = [idx for idx in range(len(pieces)) if idx not in placements]
        for new_index in unplaced:
            new_polygon = pieces[new_index].local_polygon_mm
            for placed_index, placed_polygon in placed_polygons.items():
                for placed_edge in range(len(placed_polygon)):
                    if (placed_index, placed_edge) in used_edges:
                        continue
                    p0, p1 = _edge_points(placed_polygon, placed_edge)
                    length_p = float(np.linalg.norm(p1 - p0))
                    for new_edge in range(len(new_polygon)):
                        if (new_index, new_edge) in used_edges:
                            continue
                        length_q = float(edge_sizes[new_index][new_edge])
                        if not _edge_match_allowed(length_p, length_q):
                            continue
                        q0, q1 = _edge_points(new_polygon, new_edge)
                        rotation, translation = rigid_edge_alignment(q0, q1, p1, p0)
                        placement = Placement(new_index, rotation, translation)
                        transformed = placement.transform(new_polygon)
                        invalid = False
                        for existing in placed_polygons.values():
                            if _intersection_area(transformed, existing) > 80.0:
                                invalid = True
                                break
                        if invalid:
                            continue
                        placements[new_index] = placement
                        used_edges.add((placed_index, placed_edge))
                        used_edges.add((new_index, new_edge))
                        recurse(placements, used_edges)
                        used_edges.remove((placed_index, placed_edge))
                        used_edges.remove((new_index, new_edge))
                        del placements[new_index]

    recurse({0: identity}, set())
    if not candidates:
        raise RuntimeError(
            "No rectangular assembly was found from the detected polygon edges"
        )
    candidates.sort(key=lambda candidate: candidate.geometry_score)
    return candidates[:80]


def _inside_normal(polygon: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = b - a
    direction /= max(np.linalg.norm(direction), 1e-9)
    normal = np.array([-direction[1], direction[0]])
    midpoint = (a + b) * 0.5
    test = midpoint + normal * 0.8
    if cv2.pointPolygonTest(
        polygon.astype(np.float32), (float(test[0]), float(test[1])), False
    ) < 0:
        normal = -normal
    return normal


def _bilinear_sample(image: np.ndarray, point_px: np.ndarray) -> np.ndarray:
    x, y = float(point_px[0]), float(point_px[1])
    if x < 0 or y < 0 or x >= image.shape[1] - 1 or y >= image.shape[0] - 1:
        return np.array([0.0, 0.0, 0.0])
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    dx, dy = x - x0, y - y0
    top = image[y0, x0].astype(np.float64) * (1.0 - dx) + image[
        y0, x0 + 1
    ].astype(np.float64) * dx
    bottom = image[y0 + 1, x0].astype(np.float64) * (1.0 - dx) + image[
        y0 + 1, x0 + 1
    ].astype(np.float64) * dx
    return top * (1.0 - dy) + bottom * dy


def _piece_color_at(
    piece: DetectedPiece,
    placement: Placement,
    world_point: np.ndarray,
    scale: float,
) -> np.ndarray:
    local = placement.rotation.T @ (world_point - placement.translation)
    source_mm = local + piece.center_mm
    return _bilinear_sample(piece.source_image, source_mm * scale)


def _texture_score(
    pieces: list[DetectedPiece],
    placements: dict[int, Placement],
    scale: float,
) -> float:
    polygons = _layout_polygons(pieces, placements)
    differences: list[float] = []
    indices = sorted(polygons)
    for pos, index_a in enumerate(indices):
        poly_a = polygons[index_a]
        for index_b in indices[pos + 1 :]:
            poly_b = polygons[index_b]
            for edge_a in range(len(poly_a)):
                a0, a1 = _edge_points(poly_a, edge_a)
                for edge_b in range(len(poly_b)):
                    b0, b1 = _edge_points(poly_b, edge_b)
                    if not _edge_match_allowed(
                        float(np.linalg.norm(a1 - a0)),
                        float(np.linalg.norm(b1 - b0)),
                    ):
                        continue
                    endpoint_error = np.linalg.norm(a0 - b1) + np.linalg.norm(a1 - b0)
                    if endpoint_error > 3.0:
                        continue
                    normal_a = _inside_normal(poly_a, a0, a1)
                    normal_b = _inside_normal(poly_b, b0, b1)
                    for fraction in np.linspace(0.08, 0.92, 13):
                        seam = a0 * (1.0 - fraction) + a1 * fraction
                        for delta in (0.35, 0.8, 1.4):
                            color_a = _piece_color_at(
                                pieces[index_a],
                                placements[index_a],
                                seam + normal_a * delta,
                                scale,
                            )
                            color_b = _piece_color_at(
                                pieces[index_b],
                                placements[index_b],
                                seam + normal_b * delta,
                                scale,
                            )
                            differences.append(
                                float(np.mean(np.abs(color_a - color_b))) / 255.0
                            )
    if not differences:
        return 1.0
    return float(np.mean(differences))


def _align_candidate(
    pieces: list[DetectedPiece],
    candidate: _Candidate,
    mode: str,
) -> SolvedLayout:
    polygons = _layout_polygons(pieces, candidate.placements)
    all_points = np.vstack(list(polygons.values())).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(all_points)).astype(np.float64)
    edges = np.roll(box, -1, axis=0) - box
    lengths = np.linalg.norm(edges, axis=1)
    longest = edges[int(np.argmax(lengths))]
    angle = math.degrees(math.atan2(longest[1], longest[0]))
    global_rotation = rotation_matrix(-angle)
    rotated = {
        index: polygon @ global_rotation.T for index, polygon in polygons.items()
    }
    all_rotated = np.vstack(list(rotated.values()))
    min_xy = all_rotated.min(axis=0)
    max_xy = all_rotated.max(axis=0)
    if (max_xy[0] - min_xy[0]) < (max_xy[1] - min_xy[1]):
        extra = rotation_matrix(90.0)
        global_rotation = extra @ global_rotation
        rotated = {
            index: polygon @ extra.T for index, polygon in rotated.items()
        }
        all_rotated = np.vstack(list(rotated.values()))
        min_xy = all_rotated.min(axis=0)
        max_xy = all_rotated.max(axis=0)
    current_center = (min_xy + max_xy) * 0.5
    offset = np.asarray(TARGET_CENTER_MM) - current_center

    aligned_polygons: dict[int, np.ndarray] = {}
    final_rotations: dict[int, np.ndarray] = {}
    final_translations: dict[int, np.ndarray] = {}
    for index, placement in candidate.placements.items():
        final_rotation = global_rotation @ placement.rotation
        final_translation = global_rotation @ placement.translation + offset
        final_rotations[index] = final_rotation
        final_translations[index] = final_translation
        aligned_polygons[index] = (
            pieces[index].local_polygon_mm @ final_rotation.T + final_translation
        )
    all_aligned = np.vstack(list(aligned_polygons.values()))
    size = all_aligned.max(axis=0) - all_aligned.min(axis=0)
    return SolvedLayout(
        placements=candidate.placements,
        aligned_polygons_mm=aligned_polygons,
        final_rotations=final_rotations,
        final_translations=final_translations,
        width_mm=float(size[0]),
        height_mm=float(size[1]),
        geometry_score=float(candidate.geometry_score),
        texture_score=float(candidate.texture_score),
        mode=mode,
    )


def solve_layout(
    pieces: list[DetectedPiece],
    mode: str,
    scale: float,
) -> SolvedLayout:
    if mode == "self":
        candidates = [_solve_self_template(pieces)]
    else:
        candidates = _search_candidates(pieces, mode)
    if mode == "field-card":
        for candidate in candidates[:40]:
            candidate.texture_score = _texture_score(
                pieces, candidate.placements, scale
            )
        candidates[:40] = sorted(
            candidates[:40],
            key=lambda candidate: candidate.geometry_score
            + 4.0 * candidate.texture_score,
        )
    return _align_candidate(pieces, candidates[0], mode)


def placement_rotation_deg(rotation: np.ndarray) -> float:
    return normalize_angle(
        math.degrees(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    )
