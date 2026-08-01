from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .card_recognition import recognize_card_marks
from .detector import PieceObservation
from .geometry import (
    best_cyclic_alignment,
    compose_transforms,
    edge_alignment_transforms,
    edge_lengths,
    invert_transform,
    is_proper_rotation,
    min_area_rectangle,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    polygon_intersection_area,
    polygons_overlap,
    rotation_matrix_row,
    safe_interior_point,
    transform_angle_deg,
    transform_points,
    wrap_angle_deg,
)


class SolveError(RuntimeError):
    pass


def _polygon_intersection_area(
    first: np.ndarray,
    second: np.ndarray,
    pixels_per_mm: float = 2.0,
) -> float:
    """Backward-compatible wrapper used by solver diagnostics."""

    return polygon_intersection_area(
        first,
        second,
        pixels_per_mm=pixels_per_mm,
    )


def _validate_fixed_template(
    templates: list[dict[str, Any]], fixed_cfg: dict[str, Any]
) -> dict[str, Any]:
    target_size = np.asarray(
        fixed_cfg.get("target_size_mm", [100.0, 60.0]),
        dtype=np.float64,
    )
    if target_size.shape != (2,) or np.any(target_size <= 0):
        raise SolveError("Fixed target_size_mm must be [width, height]")
    tolerance = 1.0
    total_area = 0.0
    polygons = [item["polygon"] for item in templates]
    for polygon in polygons:
        if (
            np.min(polygon[:, 0]) < -tolerance
            or np.min(polygon[:, 1]) < -tolerance
            or np.max(polygon[:, 0]) > target_size[0] + tolerance
            or np.max(polygon[:, 1]) > target_size[1] + tolerance
        ):
            raise SolveError("A fixed template piece lies outside the target rectangle")
        total_area += polygon_area(polygon)

    maximum_overlap = 0.0
    for first, second in itertools.combinations(polygons, 2):
        if cv2.isContourConvex(first.astype(np.float32)) and cv2.isContourConvex(
            second.astype(np.float32)
        ):
            overlap = float(
                cv2.intersectConvexConvex(
                    first.astype(np.float32), second.astype(np.float32)
                )[0]
            )
            maximum_overlap = max(maximum_overlap, overlap)
    rectangle_area = float(target_size[0] * target_size[1])
    coverage_ratio = total_area / max(rectangle_area, 1e-9)
    corners = np.asarray(
        [
            [0.0, 0.0],
            [target_size[0], 0.0],
            target_size,
            [0.0, target_size[1]],
        ],
        dtype=np.float32,
    )
    corners_covered = all(
        any(
            cv2.pointPolygonTest(
                polygon.astype(np.float32).reshape(-1, 1, 2),
                tuple(map(float, corner)),
                False,
            )
            >= 0
            for polygon in polygons
        )
        for corner in corners
    )
    if (
        abs(total_area - rectangle_area) > tolerance
        or maximum_overlap > tolerance
        or not corners_covered
    ):
        raise SolveError(
            "Fixed template does not completely tile its target rectangle "
            f"(coverage={coverage_ratio:.4f}, overlap={maximum_overlap:.2f}, "
            f"corners_covered={corners_covered})"
        )
    return {
        "target_size_mm": np.round(target_size, 3).tolist(),
        "target_coverage_ratio": round(coverage_ratio, 6),
        "target_corners_covered": corners_covered,
        "maximum_target_overlap_mm2": round(maximum_overlap, 6),
    }


def _sample_polygon(polygon: np.ndarray, count: int = 80) -> np.ndarray:
    p = normalize_winding(polygon)
    lengths = edge_lengths(p)
    perimeter = float(np.sum(lengths))
    positions = np.linspace(0.0, perimeter, count, endpoint=False)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    result = []
    for distance in positions:
        edge = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(p) - 1)
        fraction = (distance - cumulative[edge]) / max(lengths[edge], 1e-9)
        result.append(p[edge] + fraction * (p[(edge + 1) % len(p)] - p[edge]))
    return np.asarray(result, dtype=np.float64)


def _shape_alignment(
    template: np.ndarray,
    observed: np.ndarray,
    sample_count: int = 80,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(template) == len(observed):
        r, t, error, _ = best_cyclic_alignment(template, observed)
        return r, t, error

    src = _sample_polygon(template, count=sample_count)
    dst = _sample_polygon(observed, count=sample_count)
    # Sampling is uniform by perimeter; best_cyclic_alignment already checks
    # every possible starting position.
    r, t, error, _ = best_cyclic_alignment(src, dst)
    return r, t, error


def solve_fixed(
    observations: list[PieceObservation], fixed_cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_templates = [
        {
            "id": item["id"],
            "polygon": normalize_winding(
                np.asarray(item["vertices_mm"], dtype=np.float64)
            ),
        }
        for item in fixed_cfg["pieces"]
    ]
    if len(observations) != len(base_templates):
        raise SolveError(
            f"Fixed mode requires {len(base_templates)} pieces, "
            f"detected {len(observations)}"
        )
    coverage_info = _validate_fixed_template(base_templates, fixed_cfg)
    target_size = np.asarray(coverage_info["target_size_mm"], dtype=np.float64)
    reflected_templates = [
        {
            "id": item["id"],
            "polygon": normalize_winding(
                np.column_stack(
                    [
                        target_size[0] - item["polygon"][:, 0],
                        item["polygon"][:, 1],
                    ]
                )
            ),
        }
        for item in base_templates
    ]
    _validate_fixed_template(reflected_templates, fixed_cfg)

    layout_results: list[dict[str, Any]] = []
    for layout_name, templates in (
        ("figure2", base_templates),
        ("figure2_reflected", reflected_templates),
    ):
        alignments: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray, float]
        ] = {}
        costs = np.full(
            (len(observations), len(templates)), 1e6, dtype=np.float64
        )
        for obs_index, observation in enumerate(observations):
            for template_index, template in enumerate(templates):
                r, t, error = _shape_alignment(
                    template["polygon"], observation.polygon_mm
                )
                area_ratio = observation.area_mm2 / max(
                    polygon_area(template["polygon"]), 1e-6
                )
                area_penalty = abs(
                    math.log(max(area_ratio, 1e-6))
                ) * 8.0
                vertex_penalty = abs(
                    len(observation.polygon_mm)
                    - len(template["polygon"])
                ) * 1.5
                costs[obs_index, template_index] = (
                    error + area_penalty + vertex_penalty
                )
                alignments[(obs_index, template_index)] = (r, t, error)

        best_assignment: tuple[int, ...] | None = None
        best_cost = float("inf")
        for assignment in itertools.permutations(range(len(templates))):
            cost = sum(
                costs[index, assignment[index]]
                for index in range(len(observations))
            )
            if cost < best_cost:
                best_cost = float(cost)
                best_assignment = assignment
        assert best_assignment is not None
        layout_results.append(
            {
                "name": layout_name,
                "templates": templates,
                "alignments": alignments,
                "assignment": best_assignment,
                "cost": best_cost,
            }
        )

    selected = min(layout_results, key=lambda item: item["cost"])
    templates = selected["templates"]
    alignments = selected["alignments"]
    best_assignment = selected["assignment"]
    best_cost = float(selected["cost"])

    target_origin = np.asarray(fixed_cfg["target_origin_mm"], dtype=np.float64)
    plan: list[dict[str, Any]] = []
    errors = []
    for obs_index, template_index in enumerate(best_assignment):
        observation = observations[obs_index]
        template = templates[template_index]
        current_r, current_t, match_error = alignments[
            (obs_index, template_index)
        ]
        inverse_r, inverse_t = invert_transform(current_r, current_t)
        if not (
            is_proper_rotation(current_r)
            and is_proper_rotation(inverse_r)
        ):
            raise SolveError("Mirrored fixed-piece transforms are forbidden")
        target_polygon = template["polygon"] + target_origin
        measured_target_polygon = transform_points(
            observation.polygon_mm,
            inverse_r,
            inverse_t + target_origin,
        )
        target_pick = transform_points(
            observation.pickup_mm[None, :], inverse_r, inverse_t + target_origin
        )[0]
        rotate_deg = wrap_angle_deg(-transform_angle_deg(current_r))
        errors.append(match_error)
        plan.append(
            {
                "piece_id": observation.id,
                "template_id": template["id"],
                "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                "place_mm": np.round(target_pick, 3).tolist(),
                "rotate_deg": round(rotate_deg, 3),
                "mirrored": False,
                "target_polygon_mm": np.round(target_polygon, 3).tolist(),
                "measured_target_polygon_mm": np.round(
                    measured_target_polygon, 3
                ).tolist(),
                "match_error_mm": round(match_error, 3),
            }
        )

    maximum_error = max(errors, default=0.0)
    if maximum_error > float(fixed_cfg["max_match_error_mm"]):
        raise SolveError(
            f"Fixed-piece match error is too large ({maximum_error:.2f} mm). "
            "Check segmentation or update the fixed template vertices."
        )
    return plan, {
        "assignment_cost": round(best_cost, 4),
        "target_layout_variant": selected["name"],
        "layout_candidate_costs": {
            item["name"]: round(float(item["cost"]), 4)
            for item in layout_results
        },
        "max_match_error_mm": round(maximum_error, 4),
        "match_limit_mm": float(fixed_cfg["max_match_error_mm"]),
        **coverage_info,
        "target_origin_mm": np.round(target_origin, 3).tolist(),
        "solution_accepted": True,
        "solution_quality": "fixed_template_match",
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "piece_reflection_used": False,
        "target_non_overlapping": True,
        "maximum_target_overlap_mm2": coverage_info[
            "maximum_target_overlap_mm2"
        ],
    }


def solve_taught(
    observations: list[PieceObservation],
    layout: dict[str, Any],
    unknown_cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match observations to a previously demonstrated rectangle.

    The match uses proper rigid rotations only.  A taught layout is a fast
    path for a known set of two to four physical pieces and also prevents an
    unchanged scene from jumping between several geometrically similar
    autonomous solutions.
    """

    base_templates = [
        {
            "id": item["id"],
            "polygon": normalize_winding(
                np.asarray(item["vertices_mm"], dtype=np.float64)
            ),
        }
        for item in layout.get("pieces", [])
    ]
    if len(observations) != len(base_templates) or not base_templates:
        raise SolveError("Taught layout piece count does not match")

    target_size = np.asarray(
        layout.get("target_size_mm"), dtype=np.float64
    )
    if target_size.shape != (2,) or np.any(target_size <= 0):
        raise SolveError("Taught target_size_mm must be [width, height]")

    reflected_templates = [
        {
            "id": item["id"],
            # This is a global mirror of the demonstrated *layout*.  Matching
            # and moving every observed piece below still uses only a proper
            # rotation and translation.  It handles the common case where all
            # pieces were turned over together between demonstration and run.
            "polygon": normalize_winding(
                np.column_stack(
                    [
                        target_size[0] - item["polygon"][:, 0],
                        item["polygon"][:, 1],
                    ]
                )
            ),
        }
        for item in base_templates
    ]

    variant_results: list[dict[str, Any]] = []
    for variant_name, variant_templates in (
        ("demonstrated", base_templates),
        ("globally_reflected", reflected_templates),
    ):
        variant_alignments: dict[
            tuple[int, int],
            tuple[np.ndarray, np.ndarray, float],
        ] = {}
        variant_costs = np.full(
            (len(observations), len(variant_templates)),
            1e6,
            dtype=np.float64,
        )
        for obs_index, observation in enumerate(observations):
            for template_index, template in enumerate(
                variant_templates
            ):
                r, t, error = _shape_alignment(
                    template["polygon"],
                    observation.polygon_mm,
                    sample_count=36,
                )
                if not is_proper_rotation(r):
                    continue
                area_ratio = observation.area_mm2 / max(
                    polygon_area(template["polygon"]), 1e-6
                )
                variant_costs[obs_index, template_index] = (
                    error
                    + abs(
                        math.log(max(area_ratio, 1e-6))
                    )
                    * 8.0
                    + abs(
                        len(observation.polygon_mm)
                        - len(template["polygon"])
                    )
                    * 1.5
                )
                variant_alignments[
                    (obs_index, template_index)
                ] = (r, t, error)
        variant_assignment = min(
            itertools.permutations(
                range(len(variant_templates))
            ),
            key=lambda value: sum(
                variant_costs[index, value[index]]
                for index in range(len(observations))
            ),
        )
        variant_results.append(
            {
                "name": variant_name,
                "templates": variant_templates,
                "alignments": variant_alignments,
                "costs": variant_costs,
                "assignment": variant_assignment,
                "total_cost": float(
                    sum(
                        variant_costs[index][
                            variant_assignment[index]
                        ]
                        for index in range(len(observations))
                    )
                ),
            }
        )
    selected_variant = min(
        variant_results,
        key=lambda item: item["total_cost"],
    )
    layout_variant = str(selected_variant["name"])
    templates = selected_variant["templates"]
    alignments = selected_variant["alignments"]
    costs = selected_variant["costs"]
    assignment = selected_variant["assignment"]
    match_errors = [
        alignments[(index, assignment[index])][2]
        for index in range(len(observations))
    ]
    maximum_error = max(match_errors, default=float("inf"))
    error_limit = float(layout.get("max_match_error_mm", 12.0))
    if maximum_error > error_limit:
        raise SolveError(
            f"Taught-layout match error is too large "
            f"({maximum_error:.2f} mm > {error_limit:.2f} mm)"
        )

    # A demonstration is often made with several millimetres of clearance so
    # the operator can show the intended adjacency without manually achieving
    # a perfect final fit.  Such a capture is a topology/orientation hint, not
    # an already valid motion target.  When its nominal bounding rectangle is
    # not sufficiently filled, run the normal rigid, no-reflection solver but
    # rank edge dockings by their agreement with the demonstrated relative
    # poses.  The returned layout must still pass the same independent
    # rectangle, overlap, and fill checks as a fully autonomous solution.
    demonstrated_fill = sum(
        polygon_area(item["polygon"]) for item in templates
    ) / max(float(np.prod(target_size)), 1e-6)
    if demonstrated_fill < float(
        unknown_cfg.get("minimum_accepted_fill_ratio", 0.9)
    ):
        pose_hints: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        for obs_index, template_index in enumerate(assignment):
            current_r, current_t, _ = alignments[
                (obs_index, template_index)
            ]
            pose_hints[obs_index] = invert_transform(
                current_r, current_t
            )
        guided_config = dict(unknown_cfg)
        guided_config["pose_hint_weight"] = float(
            unknown_cfg.get("pose_hint_weight", 1.8)
        )
        guided_config["minimum_partial_remainder_mm"] = float(
            unknown_cfg.get(
                "guided_minimum_partial_remainder_mm", 6.0
            )
        )
        guided_config["max_search_seconds"] = float(
            unknown_cfg.get(
                "guided_search_seconds",
                unknown_cfg.get("max_search_seconds", 2.5),
            )
        )
        guided_config["fallback_search_seconds"] = float(
            unknown_cfg.get(
                "guided_fallback_search_seconds",
                unknown_cfg.get("fallback_search_seconds", 2.5),
            )
        )
        plan, info = UnknownPuzzleSolver(
            observations,
            guided_config,
            pose_hints=pose_hints,
        ).solve()
        info["solver_method"] = "taught_guided_rigid_search"
        info["taught_layout_name"] = layout.get(
            "name", "unnamed"
        )
        info["taught_layout_variant"] = layout_variant
        info["taught_layout_variant_costs"] = {
            str(item["name"]): round(
                float(item["total_cost"]), 4
            )
            for item in variant_results
        }
        info["taught_assignment_cost"] = round(
            float(
                sum(
                    costs[index, assignment[index]]
                    for index in range(len(observations))
                )
            ),
            4,
        )
        info["taught_max_match_error_mm"] = round(
            maximum_error, 4
        )
        info["taught_demonstrated_fill_ratio"] = round(
            demonstrated_fill, 6
        )
        return plan, info

    zone = np.asarray(unknown_cfg["target_zone_mm"], dtype=np.float64)
    zone_center = np.asarray(
        [(zone[0] + zone[2]) * 0.5, (zone[1] + zone[3]) * 0.5],
        dtype=np.float64,
    )
    target_origin = zone_center - target_size * 0.5
    plan: list[dict[str, Any]] = []
    target_polygons: list[np.ndarray] = []
    for obs_index, template_index in enumerate(assignment):
        observation = observations[obs_index]
        template = templates[template_index]
        current_r, current_t, match_error = alignments[
            (obs_index, template_index)
        ]
        inverse_r, inverse_t = invert_transform(current_r, current_t)
        if not (
            is_proper_rotation(current_r)
            and is_proper_rotation(inverse_r)
        ):
            raise SolveError("Mirrored taught-piece transforms are forbidden")
        target_polygon = template["polygon"] + target_origin
        measured_target = transform_points(
            observation.polygon_mm,
            inverse_r,
            inverse_t + target_origin,
        )
        place = transform_points(
            observation.pickup_mm[None, :],
            inverse_r,
            inverse_t + target_origin,
        )[0]
        target_polygons.append(target_polygon)
        plan.append(
            {
                "piece_id": observation.id,
                "template_id": template["id"],
                "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                "place_mm": np.round(place, 3).tolist(),
                "rotate_deg": round(
                    wrap_angle_deg(-transform_angle_deg(current_r)), 3
                ),
                "mirrored": False,
                "target_polygon_mm": np.round(
                    target_polygon, 3
                ).tolist(),
                "measured_target_polygon_mm": np.round(
                    measured_target, 3
                ).tolist(),
                "match_error_mm": round(match_error, 3),
            }
        )

    maximum_overlap = 0.0
    target_non_overlapping = True
    for first, second in itertools.combinations(target_polygons, 2):
        overlap = _polygon_intersection_area(first, second)
        maximum_overlap = max(maximum_overlap, overlap)
        if polygons_overlap(
            first,
            second,
            float(unknown_cfg["overlap_tolerance_mm"]),
        ):
            target_non_overlapping = False

    pieces_area = sum(item.area_mm2 for item in observations)
    fill_ratio = pieces_area / max(float(np.prod(target_size)), 1e-6)
    dimension_ok = (
        float(unknown_cfg["min_width_mm"]) <= target_size[0]
        <= float(unknown_cfg["max_width_mm"])
        and float(unknown_cfg["min_height_mm"]) <= target_size[1]
        <= float(unknown_cfg["max_height_mm"])
    )
    fill_ok = (
        float(unknown_cfg.get("minimum_accepted_fill_ratio", 0.9))
        <= fill_ratio
        <= float(unknown_cfg.get("maximum_candidate_fill_ratio", 1.08))
    )
    solution_accepted = dimension_ok and target_non_overlapping and fill_ok
    return plan, {
        "solver_method": "taught_layout",
        "taught_layout_name": layout.get("name", "unnamed"),
        "taught_layout_variant": layout_variant,
        "taught_layout_variant_costs": {
            str(item["name"]): round(
                float(item["total_cost"]), 4
            )
            for item in variant_results
        },
        "assignment_cost": round(
            float(sum(costs[index, assignment[index]] for index in range(len(observations)))),
            4,
        ),
        "max_match_error_mm": round(maximum_error, 4),
        "match_limit_mm": error_limit,
        "target_size_mm": np.round(target_size, 3).tolist(),
        "target_origin_mm": np.round(target_origin, 3).tolist(),
        "fill_ratio": round(fill_ratio, 6),
        "geometry_score": round(maximum_error, 5),
        "solution_accepted": solution_accepted,
        "solution_quality": "taught_template_match",
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "piece_reflection_used": False,
        "target_non_overlapping": target_non_overlapping,
        "maximum_target_overlap_mm2": round(maximum_overlap, 5),
        "target_outline_denoised": True,
    }


def _card_shape_ambiguity(
    observations: list[PieceObservation],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Estimate whether geometry alone has several plausible seam pairings."""

    tolerance = float(config.get("edge_length_tolerance_mm", 4.0))
    relative = float(config.get("edge_length_relative_tolerance", 0.10))
    minimum = float(config.get("minimum_solver_edge_mm", 8.0))
    descriptors: list[tuple[int, int, float]] = []
    for piece_index, observation in enumerate(observations):
        for edge_index, length in enumerate(
            edge_lengths(observation.polygon_mm)
        ):
            value = float(length)
            if value >= minimum:
                descriptors.append((piece_index, edge_index, value))
    ambiguous_edges = 0
    maximum_alternatives = 0
    for piece_index, _, length in descriptors:
        alternatives = sum(
            1
            for other_piece, _, other_length in descriptors
            if other_piece != piece_index
            and abs(length - other_length)
            <= max(tolerance, relative * min(length, other_length))
        )
        maximum_alternatives = max(maximum_alternatives, alternatives)
        ambiguous_edges += int(alternatives >= 2)
    threshold = int(config.get("card_pattern_ambiguity_edge_count", 2))
    return {
        "ambiguous": bool(ambiguous_edges >= threshold),
        "ambiguous_edge_count": ambiguous_edges,
        "maximum_edge_alternatives": maximum_alternatives,
        "threshold": threshold,
    }


def solve_card(
    observations: list[PieceObservation],
    unknown_cfg: dict[str, Any],
    rectified_image: np.ndarray | None = None,
    pixels_per_mm: float = 4.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assemble four rounded-corner playing-card fragments.

    Every fragment owns one of the original card corners.  Candidate motions
    therefore anchor one fragment to each target corner and use only proper
    rotations.  Straight boundary support ranks the candidates; full polygon
    gap and overlap select the final result.  No reflection or scaling is
    introduced.
    """

    if not 2 <= len(observations) <= 4:
        raise SolveError("Playing-card mode requires two to four pieces")

    # A card cut does not necessarily leave exactly one rounded corner on
    # every fragment: one fragment can own two original corners while another
    # owns none.  The old one-piece/one-corner permutation therefore rejected
    # legal cuts.  Reuse the general rigid edge solver, but mark rounded-card
    # boundary edges as unavailable for seam docking and constrain the result
    # to a portrait playing-card rectangle.
    card_cfg = dict(unknown_cfg)
    card_cfg["card_mode"] = True
    card_cfg["target_orientation"] = "portrait"
    card_cfg["min_width_mm"] = float(
        unknown_cfg.get("card_minimum_long_side_mm", 78.0)
    )
    card_cfg["max_width_mm"] = float(
        unknown_cfg.get("card_maximum_long_side_mm", 100.0)
    )
    card_cfg["min_height_mm"] = float(
        unknown_cfg.get("card_minimum_short_side_mm", 45.0)
    )
    card_cfg["max_height_mm"] = float(
        unknown_cfg.get("card_maximum_short_side_mm", 68.0)
    )
    card_cfg["max_search_seconds"] = float(
        unknown_cfg.get("card_search_seconds", 6.0)
    )
    card_cfg["exact_search_seconds"] = float(
        unknown_cfg.get("card_exact_search_seconds", 1.5)
    )
    card_cfg["fallback_search_seconds"] = float(
        unknown_cfg.get("card_fallback_search_seconds", 2.0)
    )
    # Standard playing-card dimensions are a much stronger bound than the
    # broad generic field-rectangle limits.  Applying the tight bound during
    # partial search prevents the slow board from spending its deadline on
    # plausible-looking 64 x 94 mm layouts that can never be the original
    # 57 x 88 mm card.
    card_cfg["search_dimension_slack_mm"] = float(
        unknown_cfg.get("card_search_dimension_slack_mm", 1.5)
    )
    card_cfg["max_pair_options_exact"] = int(
        unknown_cfg.get("card_max_pair_options_exact", 36)
    )
    card_cfg["max_pair_options_partial"] = int(
        unknown_cfg.get("card_max_pair_options_partial", 36)
    )
    # The formal task requires genuinely non-overlapping target fragments.
    # Disable display-oriented boundary snapping for cards and make pairwise
    # polygon intersection a hard acceptance condition.
    card_cfg["boundary_snap_tolerance_mm"] = 0.0
    card_cfg["maximum_accepted_pair_overlap_mm2"] = float(
        unknown_cfg.get("card_maximum_pair_overlap_mm2", 0.5)
    )
    card_cfg["overlap_tolerance_mm"] = float(
        unknown_cfg.get("card_overlap_tolerance_mm", 0.15)
    )
    card_cfg["minimum_partial_remainder_mm"] = float(
        unknown_cfg.get("card_minimum_partial_remainder_mm", 3.0)
    )
    recognition: dict[str, Any] = {
        "rank_detected": False,
        "rank": None,
        "rank_confidence": 0.0,
    }
    ambiguity = _card_shape_ambiguity(observations, unknown_cfg)
    pattern_assist = bool(
        rectified_image is not None and ambiguity["ambiguous"]
    )
    recognition["pattern_assist_enabled"] = pattern_assist
    recognition["shape_ambiguity"] = ambiguity
    if pattern_assist and rectified_image is not None:
        recognition = recognize_card_marks(
            rectified_image,
            observations,
            pixels_per_mm,
            unknown_cfg,
        )
        recognition["pattern_assist_enabled"] = True
        recognition["shape_ambiguity"] = ambiguity
        if recognition.get("rank_detected", False):
            card_cfg["card_rank_piece_index"] = int(
                recognition["rank_piece_index"]
            )
            card_cfg["card_rank_center_mm"] = list(
                recognition["rank_center_mm"]
            )
        visibility = float(
            recognition.get("pattern_visibility_fraction", 0.0)
        )
        confidence = float(recognition.get("rank_confidence", 0.0))
        card_cfg["texture_weight"] = float(
            unknown_cfg.get("card_texture_weight", 0.3)
        ) * visibility
        card_cfg["card_rank_score_weight"] = float(
            unknown_cfg.get("card_rank_score_weight", 0.2)
        ) * confidence
    attempts: list[
        tuple[list[dict[str, Any]], dict[str, Any], bool]
    ] = []
    errors: list[str] = []
    # A reliably read corner index provides a much stronger pose anchor than
    # an arbitrary equal-length edge.  Try fixed portrait-card dimensions
    # inferred from total area first.  Each trial remains a proper rigid
    # transform and must still satisfy the non-overlap hard condition.
    rank_piece_index = recognition.get("rank_piece_index")
    if (
        recognition.get("rank_detected", False)
        and rank_piece_index is not None
        and float(recognition.get("rank_confidence", 0.0))
        >= float(
            unknown_cfg.get(
                "card_anchor_min_rank_confidence", 0.60
            )
        )
        and bool(
            recognition.get(
                "rank_rounded_corner_reference", False
            )
        )
    ):
        measured_area = sum(item.area_mm2 for item in observations)
        aspect = float(
            unknown_cfg.get("card_aspect_ratio", 88.0 / 57.0)
        )
        anchor_complete = False
        anchored_started = time.perf_counter()
        anchored_attempts = 0
        anchored_maximum_attempts = int(
            unknown_cfg.get("card_anchored_max_attempts", 2)
        )
        anchored_total_seconds = float(
            unknown_cfg.get("card_anchored_total_seconds", 1.2)
        )
        for expected_fill in unknown_cfg.get(
            "card_expected_fill_ratios", [0.98, 0.94, 0.90]
        ):
            target_area = measured_area / max(
                float(expected_fill), 1e-6
            )
            width = math.sqrt(target_area / aspect)
            height = width * aspect
            anchor_options = _card_rank_anchor_options(
                observations[int(rank_piece_index)].polygon_mm,
                width,
                height,
                unknown_cfg,
            )
            for anchor_rotation, anchor_translation, _ in anchor_options:
                if (
                    anchored_attempts >= anchored_maximum_attempts
                    or time.perf_counter() - anchored_started
                    >= anchored_total_seconds
                ):
                    break
                anchored_attempts += 1
                attempt_cfg = dict(card_cfg)
                attempt_cfg["card_target_size_mm"] = [width, height]
                attempt_cfg["card_rounded_chord_max_mm"] = 0.0
                attempt_cfg["max_search_seconds"] = float(
                    unknown_cfg.get(
                        "card_anchored_search_seconds", 1.4
                    )
                )
                attempt_cfg["exact_search_seconds"] = float(
                    unknown_cfg.get(
                        "card_anchored_exact_seconds", 0.35
                    )
                )
                try:
                    attempt_plan, attempt_information = (
                        UnknownPuzzleSolver(
                            observations,
                            attempt_cfg,
                            rectified_image=rectified_image,
                            pixels_per_mm=pixels_per_mm,
                            use_texture=pattern_assist,
                            initial_transforms={
                                int(rank_piece_index): (
                                    anchor_rotation,
                                    anchor_translation,
                                )
                            },
                        ).solve()
                    )
                    attempt_information[
                        "card_rank_corner_anchor_used"
                    ] = True
                    attempt_information[
                        "card_expected_fill_ratio"
                    ] = float(expected_fill)
                    attempts.append(
                        (attempt_plan, attempt_information, False)
                    )
                    if attempt_information.get(
                        "solution_accepted", False
                    ):
                        anchor_complete = True
                        break
                except SolveError as exc:
                    errors.append(str(exc))
            if anchor_complete:
                break
            if (
                anchored_attempts >= anchored_maximum_attempts
                or time.perf_counter() - anchored_started
                >= anchored_total_seconds
            ):
                break

    strict_already_accepted = any(
        item[1].get("solution_accepted", False) for item in attempts
    )
    # A hand-cut bevel can imitate a rounded card corner.  For patterned cards
    # an accepted geometry-only strict pass is not enough evidence to suppress
    # the unexcluded pass: doing so was the main cause of stable but visibly
    # wrong J/Q/K layouts.  Keep the second pass short when a usable strict
    # result already exists, then let texture and geometry choose globally.
    search_variants = (
        (True, False)
        if pattern_assist
        else (() if strict_already_accepted else (True, False))
    )
    for exclude_rounded_edges in search_variants:
        attempt_cfg = dict(card_cfg)
        if not exclude_rounded_edges:
            # A short hand-cut bevel can occasionally imitate a rounded card
            # chord.  Retry without edge exclusion only when the strict pass
            # did not yield an accepted, non-overlapping rectangle.
            attempt_cfg["card_rounded_chord_max_mm"] = 0.0
            if any(
                item[1].get("solution_accepted", False)
                for item in attempts
            ):
                attempt_cfg["max_search_seconds"] = min(
                    float(attempt_cfg["max_search_seconds"]),
                    float(
                        unknown_cfg.get(
                            "card_pattern_unexcluded_search_seconds",
                            0.65,
                        )
                    ),
                )
                attempt_cfg["exact_search_seconds"] = min(
                    float(attempt_cfg["exact_search_seconds"]),
                    float(
                        unknown_cfg.get(
                            "card_pattern_unexcluded_exact_seconds",
                            0.22,
                        )
                    ),
                )
        try:
            attempt_plan, attempt_information = UnknownPuzzleSolver(
                observations,
                attempt_cfg,
                rectified_image=rectified_image,
                pixels_per_mm=pixels_per_mm,
                use_texture=pattern_assist,
            ).solve()
            attempts.append(
                (
                    attempt_plan,
                    attempt_information,
                    exclude_rounded_edges,
                )
            )
            if attempt_information.get("solution_accepted", False):
                break
        except SolveError as exc:
            errors.append(str(exc))
    if not attempts:
        raise SolveError(
            errors[-1]
            if errors
            else "No non-overlapping playing-card assembly was found"
        )
    plan, information, excluded_edges = min(
        attempts,
        key=lambda item: (
            not bool(item[1].get("solution_accepted", False)),
            float(item[1].get("total_score", float("inf"))),
            float(item[1].get("geometry_score", float("inf"))),
        ),
    )
    information["solver_method"] = "portrait_card_rigid_search"
    information["card_aspect_ratio"] = round(
        float(unknown_cfg.get("card_aspect_ratio", 88.0 / 57.0)), 6
    )
    information["card_rounded_boundary_edges_excluded"] = excluded_edges
    if errors:
        information["card_solver_attempt_errors"] = errors
    information["card_recognition"] = recognition
    return plan, information

    # Legacy one-corner-per-piece search retained below for reference.  The
    # general card search above is the active path.
    if len(observations) != 4:
        raise SolveError("Playing-card mode requires exactly four pieces")
    aspect = float(unknown_cfg.get("card_aspect_ratio", 88.0 / 57.0))
    if aspect <= 1.0:
        raise SolveError("card_aspect_ratio must be greater than 1")
    expected_fills = [
        float(value)
        for value in unknown_cfg.get(
            "card_expected_fill_ratios", [0.90, 0.94, 0.98]
        )
        if 0.75 <= float(value) <= 1.02
    ]
    if not expected_fills:
        expected_fills = [0.94]
    outside_tolerance = float(
        unknown_cfg.get("card_boundary_tolerance_mm", 4.0)
    )
    option_limit = int(unknown_cfg.get("card_corner_option_limit", 7))
    polygons = [
        np.asarray(item.polygon_mm, dtype=np.float64)
        for item in observations
    ]
    measured_area = sum(item.area_mm2 for item in observations)
    polygon_areas = [polygon_area(polygon) for polygon in polygons]
    local_corners = (
        (True, True),
        (False, True),
        (False, False),
        (True, False),
    )

    best: dict[str, Any] | None = None
    tested = 0
    started = time.perf_counter()
    maximum_seconds = float(
        unknown_cfg.get("card_search_seconds", 4.0)
    )

    for expected_fill in expected_fills:
        target_area = measured_area / max(expected_fill, 1e-6)
        width = math.sqrt(target_area / aspect)
        height = width * aspect
        size = np.asarray([width, height], dtype=np.float64)
        corner_points = np.asarray(
            [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            dtype=np.float64,
        )
        options: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for piece_index, polygon in enumerate(polygons):
            for corner_index, (left, top) in enumerate(local_corners):
                generated: dict[int, dict[str, Any]] = {}
                for edge_index in range(len(polygon)):
                    vector = (
                        polygon[(edge_index + 1) % len(polygon)]
                        - polygon[edge_index]
                    )
                    if float(np.linalg.norm(vector)) < 2.0:
                        continue
                    source_angle = math.atan2(vector[1], vector[0])
                    for target_angle in (0.0, math.pi / 2.0):
                        r = rotation_matrix_row(target_angle - source_angle)
                        rotated = polygon @ r
                        shift = np.asarray(
                            [
                                -float(np.min(rotated[:, 0]))
                                if left
                                else width - float(np.max(rotated[:, 0])),
                                -float(np.min(rotated[:, 1]))
                                if top
                                else height - float(np.max(rotated[:, 1])),
                            ],
                            dtype=np.float64,
                        )
                        placed = rotated + shift
                        outside = (
                            max(0.0, -float(np.min(placed[:, 0])))
                            + max(0.0, float(np.max(placed[:, 0])) - width)
                            + max(0.0, -float(np.min(placed[:, 1])))
                            + max(0.0, float(np.max(placed[:, 1])) - height)
                        )
                        if outside > outside_tolerance:
                            continue
                        x_side = 0.0 if left else width
                        y_side = 0.0 if top else height
                        support = 0.0
                        for index in range(len(placed)):
                            first = placed[index]
                            second = placed[(index + 1) % len(placed)]
                            edge_length = float(
                                np.linalg.norm(second - first)
                            )
                            if (
                                max(
                                    abs(float(first[0] - x_side)),
                                    abs(float(second[0] - x_side)),
                                )
                                <= outside_tolerance
                                or max(
                                    abs(float(first[1] - y_side)),
                                    abs(float(second[1] - y_side)),
                                )
                                <= outside_tolerance
                            ):
                                support += edge_length
                        corner_distance = float(
                            np.min(
                                np.linalg.norm(
                                    placed - corner_points[corner_index],
                                    axis=1,
                                )
                            )
                        )
                        option_score = (
                            12.0 * outside
                            + 0.10 * corner_distance
                            - 0.035 * support
                        )
                        angle_key = int(
                            round(wrap_angle_deg(transform_angle_deg(r)))
                        )
                        candidate = {
                            "r": r,
                            "t": shift,
                            "polygon": placed,
                            "score": option_score,
                        }
                        previous = generated.get(angle_key)
                        if (
                            previous is None
                            or option_score < previous["score"]
                        ):
                            generated[angle_key] = candidate
                options[(piece_index, corner_index)] = sorted(
                    generated.values(),
                    key=lambda item: item["score"],
                )[:option_limit]

        for assignment in itertools.permutations(range(4)):
            option_groups = [
                options.get((piece_index, assignment[piece_index]), [])
                for piece_index in range(4)
            ]
            if any(not group for group in option_groups):
                continue
            for selected in itertools.product(*option_groups):
                tested += 1
                if (
                    tested % 128 == 0
                    and time.perf_counter() - started >= maximum_seconds
                ):
                    break
                target_polygons = [
                    np.asarray(item["polygon"], dtype=np.float64)
                    for item in selected
                ]
                overlap_area = 0.0
                rejected = False
                for first, second in itertools.combinations(
                    target_polygons, 2
                ):
                    overlap_area += _polygon_intersection_area(
                        first, second, pixels_per_mm=1.0
                    )
                    if overlap_area > target_area * 0.10:
                        rejected = True
                        break
                if rejected:
                    continue
                union_area = max(
                    0.0, sum(polygon_areas) - overlap_area
                )
                gap_area = max(0.0, target_area - union_area)
                gap_ratio = gap_area / max(target_area, 1e-6)
                overlap_ratio = overlap_area / max(target_area, 1e-6)
                geometry_score = 100.0 * (
                    gap_ratio + 2.0 * overlap_ratio
                )
                total_score = geometry_score + 0.04 * sum(
                    float(item["score"]) for item in selected
                )
                if best is None or total_score < best["total_score"]:
                    best = {
                        "selected": selected,
                        "size": size,
                        "target_area": target_area,
                        "gap_area": gap_area,
                        "overlap_area": overlap_area,
                        "gap_ratio": gap_ratio,
                        "overlap_ratio": overlap_ratio,
                        "geometry_score": geometry_score,
                        "total_score": total_score,
                        "expected_fill": expected_fill,
                    }
            if time.perf_counter() - started >= maximum_seconds:
                break
        if time.perf_counter() - started >= maximum_seconds:
            break

    if best is None:
        raise SolveError("No rounded-corner playing-card assembly was found")
    zone = np.asarray(unknown_cfg["target_zone_mm"], dtype=np.float64)
    zone_center = np.asarray(
        [(zone[0] + zone[2]) * 0.5, (zone[1] + zone[3]) * 0.5],
        dtype=np.float64,
    )
    size = np.asarray(best["size"], dtype=np.float64)
    target_origin = zone_center - size * 0.5
    plan: list[dict[str, Any]] = []
    target_polygons = []
    for piece_index, observation in enumerate(observations):
        selected = best["selected"][piece_index]
        r = np.asarray(selected["r"], dtype=np.float64)
        t = np.asarray(selected["t"], dtype=np.float64) + target_origin
        if not is_proper_rotation(r):
            raise SolveError("Mirrored playing-card transforms are forbidden")
        polygon = transform_points(observation.polygon_mm, r, t)
        place = transform_points(observation.pickup_mm[None, :], r, t)[0]
        target_polygons.append(polygon)
        plan.append(
            {
                "piece_id": observation.id,
                "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                "place_mm": np.round(place, 3).tolist(),
                "rotate_deg": round(
                    wrap_angle_deg(transform_angle_deg(r)), 3
                ),
                "mirrored": False,
                "target_polygon_mm": np.round(polygon, 3).tolist(),
                "measured_target_polygon_mm": np.round(
                    polygon, 3
                ).tolist(),
                "boundary_snap_mm": [0.0, 0.0],
                "boundary_snap_requested_mm": [0.0, 0.0],
            }
        )

    maximum_overlap = max(
        (
            _polygon_intersection_area(first, second)
            for first, second in itertools.combinations(
                target_polygons, 2
            )
        ),
        default=0.0,
    )
    maximum_score = float(
        unknown_cfg.get("card_maximum_geometry_score", 16.0)
    )
    maximum_overlap_ratio = float(
        unknown_cfg.get("card_maximum_overlap_ratio", 0.025)
    )
    accepted = (
        float(best["geometry_score"]) <= maximum_score
        and float(best["overlap_ratio"]) <= maximum_overlap_ratio
    )
    return plan, {
        "solver_method": "rounded_card_corner_search",
        "search_nodes": tested,
        "search_timed_out": (
            time.perf_counter() - started >= maximum_seconds
        ),
        "candidate_count": tested,
        "geometry_score": round(float(best["geometry_score"]), 5),
        "assembly_gap_mm2": round(float(best["gap_area"]), 5),
        "assembly_overlap_mm2": round(
            float(best["overlap_area"]), 5
        ),
        "assembly_gap_ratio": round(float(best["gap_ratio"]), 6),
        "assembly_overlap_ratio": round(
            float(best["overlap_ratio"]), 6
        ),
        "total_score": round(float(best["total_score"]), 5),
        "target_size_mm": np.round(size, 3).tolist(),
        "target_origin_mm": np.round(target_origin, 3).tolist(),
        "target_orientation": "portrait",
        "fill_ratio": round(measured_area / float(np.prod(size)), 6),
        "card_aspect_ratio": round(aspect, 6),
        "card_expected_fill_ratio": round(
            float(best["expected_fill"]), 6
        ),
        "solution_accepted": accepted,
        "solution_quality": "rounded_card" if accepted else "rejected",
        "motion_model": "rotation_and_translation_only",
        "mirror_allowed": False,
        "piece_reflection_used": False,
        "target_non_overlapping": (
            float(best["overlap_ratio"]) <= maximum_overlap_ratio
        ),
        "target_strictly_non_overlapping": maximum_overlap <= float(
            unknown_cfg.get("maximum_accepted_pair_overlap_mm2", 100.0)
        ),
        "maximum_target_overlap_mm2": round(maximum_overlap, 5),
        "target_outline_denoised": True,
    }


@dataclass
class AssemblyCandidate:
    transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    matches: list[tuple[int, int, int, int]]
    geometry_score: float
    long_side_mm: float
    short_side_mm: float
    fill_ratio: float
    box: np.ndarray
    theta: float
    overlap_area_mm2: float = 0.0
    gap_area_mm2: float = 0.0
    overlap_ratio: float = 0.0
    gap_ratio: float = 0.0
    texture_score: float = 0.0
    card_symmetry_score: float = 0.0
    card_rank_score: float = 0.0
    card_rank_corner_distance_mm: float | None = None
    total_score: float = 0.0


def _card_rounded_corner_frames(
    polygon: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Locate short chords that approximate an original rounded card corner.

    A rounded corner is observed as a short chord between two long, nearly
    perpendicular card-boundary edges.  The two adjacent long edges are outer
    card edges and must never be paired as cut seams.
    """

    points = np.asarray(polygon, dtype=np.float64)
    count = len(points)
    if count < 4:
        return []
    minimum_chord = float(config.get("card_rounded_chord_min_mm", 2.0))
    maximum_chord = float(config.get("card_rounded_chord_max_mm", 8.0))
    maximum_error = float(
        config.get("card_rounded_corner_angle_error_deg", 25.0)
    )
    frames: list[dict[str, Any]] = []
    for chord_index in range(count):
        chord_start = points[chord_index]
        chord_end = points[(chord_index + 1) % count]
        chord_length = float(np.linalg.norm(chord_end - chord_start))
        if not minimum_chord <= chord_length <= maximum_chord:
            continue
        incoming = chord_start - points[(chord_index - 1) % count]
        outgoing = points[(chord_index + 2) % count] - chord_end
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if min(incoming_length, outgoing_length) < maximum_chord:
            continue
        first = incoming / incoming_length
        second = outgoing / outgoing_length
        line_angle = math.degrees(
            math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
        )
        angle_error = abs(90.0 - line_angle)
        if angle_error > maximum_error:
            continue

        # Intersection of the two infinite boundary rays gives the virtual
        # sharp rectangle corner.  It is more stable than anchoring a noisy
        # rounded chord endpoint.
        matrix = np.column_stack((first, -second))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-6:
            continue
        coefficients = np.linalg.solve(matrix, chord_end - chord_start)
        virtual_corner = chord_start + coefficients[0] * first
        frames.append(
            {
                "chord_edge": chord_index,
                "outer_edges": {
                    (chord_index - 1) % count,
                    (chord_index + 1) % count,
                },
                "virtual_corner": virtual_corner,
                "incoming_direction": first,
                "outgoing_direction": second,
                "angle_error_deg": angle_error,
                "score": angle_error + 0.15 * chord_length,
            }
        )
    frames.sort(key=lambda item: float(item["score"]))
    return frames


def _card_rank_anchor_options(
    polygon: np.ndarray,
    width: float,
    height: float,
    config: dict[str, Any],
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    frames = _card_rounded_corner_frames(polygon, config)
    if not frames:
        return []
    target_corners = (
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([width, height], dtype=np.float64),
    )
    tolerance = float(config.get("card_boundary_tolerance_mm", 4.0))
    candidates: list[tuple[np.ndarray, np.ndarray, float]] = []
    for frame in frames:
        source_corner = np.asarray(
            frame["virtual_corner"], dtype=np.float64
        )
        for target_corner in target_corners:
            expected_inward = (
                np.asarray([1.0, 1.0], dtype=np.float64)
                if np.allclose(target_corner, 0.0)
                else np.asarray([-1.0, -1.0], dtype=np.float64)
            )
            for source_direction in (
                np.asarray(frame["incoming_direction"], dtype=np.float64),
                np.asarray(frame["outgoing_direction"], dtype=np.float64),
            ):
                source_angle = math.atan2(
                    float(source_direction[1]),
                    float(source_direction[0]),
                )
                for target_angle in (
                    0.0,
                    math.pi / 2.0,
                    math.pi,
                    -math.pi / 2.0,
                ):
                    rotation = rotation_matrix_row(
                        target_angle - source_angle
                    )
                    translation = target_corner - source_corner @ rotation
                    placed = transform_points(
                        polygon, rotation, translation
                    )
                    centroid_direction = (
                        polygon_centroid(placed) - target_corner
                    )
                    if np.any(
                        centroid_direction * expected_inward <= 0.0
                    ):
                        continue
                    outside = (
                        max(0.0, -float(np.min(placed[:, 0])))
                        + max(0.0, float(np.max(placed[:, 0])) - width)
                        + max(0.0, -float(np.min(placed[:, 1])))
                        + max(0.0, float(np.max(placed[:, 1])) - height)
                    )
                    if outside > tolerance:
                        continue
                    score = (
                        10.0 * outside
                        + float(frame["score"])
                    )
                    candidates.append((rotation, translation, score))
    unique: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]
    ] = {}
    for rotation, translation, score in candidates:
        key = (
            int(round(transform_angle_deg(rotation) * 10.0)),
            int(round(float(translation[0]) * 10.0)),
            int(round(float(translation[1]) * 10.0)),
        )
        previous = unique.get(key)
        if previous is None or score < previous[2]:
            unique[key] = (rotation, translation, score)
    return sorted(unique.values(), key=lambda item: item[2])


class UnknownPuzzleSolver:
    def __init__(
        self,
        observations: list[PieceObservation],
        config: dict[str, Any],
        rectified_image: np.ndarray | None = None,
        pixels_per_mm: float = 4.0,
        use_texture: bool = False,
        pose_hints: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] | None = None,
        initial_transforms: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] | None = None,
    ):
        self.observations = observations
        self.polygons = [item.polygon_mm for item in observations]
        self.centroids = [
            polygon_centroid(polygon) for polygon in self.polygons
        ]
        self.config = config
        self.image = rectified_image
        self.ppm = float(pixels_per_mm)
        self.use_texture = use_texture and rectified_image is not None
        self.lab_image = (
            cv2.cvtColor(rectified_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            if self.use_texture and rectified_image is not None
            else None
        )
        self.piece_paper_lab = (
            self._estimate_piece_paper_lab()
            if self.lab_image is not None
            else [
                np.asarray([230.0, 128.0, 128.0], dtype=np.float32)
                for _ in self.polygons
            ]
        )
        self.nodes = 0
        self.candidates: list[AssemblyCandidate] = []
        self.visited: set[tuple[Any, ...]] = set()
        self.identity = np.eye(2, dtype=np.float64)
        self.zero = np.zeros(2, dtype=np.float64)
        self.allow_partial_edges = False
        self.deadline = float("inf")
        # A wall-clock deadline alone makes the result depend on CPU load:
        # the same contours can visit a different number of deterministic DFS
        # nodes while the camera/UI is busy.  Generic straight-edged pieces
        # therefore get a small per-stage node floor before the deadline is
        # allowed to stop them.  Card mode keeps its latency-first behaviour.
        self.timeout_node_floor = 0
        self.timed_out = False
        self.card_mode = bool(config.get("card_mode", False))
        rank_piece = config.get("card_rank_piece_index")
        rank_center = config.get("card_rank_center_mm")
        self.card_rank_piece_index = (
            int(rank_piece) if rank_piece is not None else None
        )
        self.card_rank_center_mm = (
            np.asarray(rank_center, dtype=np.float64)
            if rank_center is not None
            else None
        )
        self.initial_transforms = {
            int(index): (
                np.asarray(transform[0], dtype=np.float64),
                np.asarray(transform[1], dtype=np.float64),
            )
            for index, transform in (initial_transforms or {}).items()
        }
        configured_card_size = config.get("card_target_size_mm")
        self.card_target_size = (
            np.asarray(configured_card_size, dtype=np.float64)
            if configured_card_size is not None
            else None
        )
        self.card_corner_frames = (
            [
                _card_rounded_corner_frames(polygon, config)
                for polygon in self.polygons
            ]
            if self.card_mode
            else [[] for _ in self.polygons]
        )
        self.excluded_solver_edges = [
            {
                edge
                for frame in frames
                for edge in frame["outer_edges"]
            }
            for frames in self.card_corner_frames
        ]
        self.pose_hints = pose_hints or {}
        self.hint_centroids = {
            index: transform_points(
                self.centroids[index][None, :],
                transform[0],
                transform[1],
            )[0]
            for index, transform in self.pose_hints.items()
        }
        self.pair_options = self._precompute_pair_options()

    def _estimate_piece_paper_lab(self) -> list[np.ndarray]:
        """Estimate each fragment's white-card colour under local lighting."""

        assert self.lab_image is not None
        references: list[np.ndarray] = []
        height, width = self.lab_image.shape[:2]
        for polygon in self.polygons:
            contour = np.round(polygon * self.ppm).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            erosion = max(1, int(round(0.8 * self.ppm)))
            mask = cv2.erode(
                mask,
                np.ones((2 * erosion + 1, 2 * erosion + 1), np.uint8),
            )
            values = self.lab_image[mask > 0]
            if values.size == 0:
                references.append(
                    np.asarray([230.0, 128.0, 128.0], dtype=np.float32)
                )
                continue
            # Printed artwork is predominantly darker than the card stock.
            # Use the brighter half so a large J/Q/K portrait cannot drag the
            # local white reference down, while still following uneven camera
            # exposure from fragment to fragment.
            cutoff = float(np.percentile(values[:, 0], 55.0))
            paper_values = values[values[:, 0] >= cutoff]
            if len(paper_values) < 8:
                paper_values = values
            references.append(
                np.median(paper_values, axis=0).astype(np.float32)
            )
        return references

    def _pattern_mask(
        self, lab: np.ndarray, piece_index: int
    ) -> np.ndarray:
        reference = self.piece_paper_lab[piece_index]
        lightness_drop = reference[0] - lab[:, 0]
        chroma_delta = np.linalg.norm(
            lab[:, 1:] - reference[None, 1:], axis=1
        )
        return (
            lightness_drop
            > float(
                self.config.get(
                    "card_pattern_lightness_drop_lab", 24.0
                )
            )
        ) | (
            chroma_delta
            > float(
                self.config.get(
                    "card_pattern_chroma_delta_lab", 16.0
                )
            )
        )

    def _pattern_features(
        self, lab: np.ndarray, piece_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exposure-normalised artwork strength and colour features.

        Every fragment has its own local white-card reference.  Subtracting
        that reference makes a club under a lamp comparable with the same club
        in a darker part of the A4 sheet.  The continuous strength also avoids
        the unstable all-or-nothing result of the old binary ink mask.
        """

        reference = self.piece_paper_lab[piece_index]
        light_scale = max(
            1.0,
            float(
                self.config.get(
                    "card_pattern_lightness_drop_lab", 24.0
                )
            ),
        )
        chroma_scale = max(
            1.0,
            float(
                self.config.get(
                    "card_pattern_chroma_delta_lab", 16.0
                )
            ),
        )
        darkness = np.clip(
            (reference[0] - lab[:, 0]) / light_scale,
            0.0,
            4.0,
        )
        chroma_vector = np.clip(
            (lab[:, 1:] - reference[None, 1:]) / chroma_scale,
            -4.0,
            4.0,
        )
        chroma_strength = np.linalg.norm(chroma_vector, axis=1)
        strength = np.maximum(darkness, chroma_strength)
        features = np.column_stack(
            (darkness, chroma_vector)
        ).astype(np.float32)
        return strength.astype(np.float32), features

    def _pattern_profile_continuity(
        self,
        first_lab: np.ndarray,
        first_index: int,
        second_lab: np.ndarray,
        second_index: int,
    ) -> tuple[float, int, int]:
        """Score one pair of opposite, inward edge profiles.

        Geometry-identical fragments are disambiguated by the order and colour
        of printed strokes along the proposed seam.  A small longitudinal
        shift is tested because contour simplification can move the sampled
        start point by one or two pixels.  If neither side contains visible
        artwork the result is neutral, preserving the required outline-only
        fallback for glare.
        """

        first_strength, first_features = self._pattern_features(
            first_lab, first_index
        )
        second_strength, second_features = self._pattern_features(
            second_lab, second_index
        )
        count = min(len(first_strength), len(second_strength))
        if count < 6:
            return 0.0, 0, 0
        first_strength = first_strength[:count]
        second_strength = second_strength[:count]
        first_features = first_features[:count]
        second_features = second_features[:count]
        maximum_shift = max(
            0,
            min(
                int(self.config.get("card_pattern_shift_samples", 2)),
                max(0, count // 6),
            ),
        )
        minimum_ratio = float(
            self.config.get(
                "card_pattern_min_informative_ratio", 0.035
            )
        )
        profile_weight = float(
            self.config.get("card_pattern_profile_weight", 1.25)
        )
        transition_weight = float(
            self.config.get("card_pattern_transition_weight", 0.65)
        )
        one_sided_penalty = float(
            self.config.get("card_pattern_one_sided_penalty", 7.0)
        )
        match_bonus = float(
            self.config.get("card_pattern_match_bonus", 2.5)
        )
        best: tuple[float, int, int] | None = None
        for shift in range(-maximum_shift, maximum_shift + 1):
            if shift < 0:
                first_slice = slice(-shift, count)
                second_slice = slice(0, count + shift)
            elif shift > 0:
                first_slice = slice(0, count - shift)
                second_slice = slice(shift, count)
            else:
                first_slice = slice(0, count)
                second_slice = slice(0, count)
            first_s = first_strength[first_slice]
            second_s = second_strength[second_slice]
            first_f = first_features[first_slice]
            second_f = second_features[second_slice]
            if len(first_s) < 6:
                continue
            first_visible = first_s >= 0.78
            second_visible = second_s >= 0.78
            informative = first_visible | second_visible
            informative_count = int(np.count_nonzero(informative))
            if informative_count < max(
                2, int(np.ceil(minimum_ratio * len(first_s)))
            ):
                continue
            matched = first_visible & second_visible
            matched_count = int(np.count_nonzero(matched))
            one_sided_ratio = float(
                np.mean(np.logical_xor(first_visible, second_visible)[informative])
            )
            matched_ratio = matched_count / max(informative_count, 1)
            soft_delta = float(
                np.mean(
                    np.minimum(
                        np.abs(first_s[informative] - second_s[informative]),
                        4.0,
                    )
                )
            )
            feature_delta = float(
                np.mean(
                    np.minimum(
                        np.linalg.norm(
                            first_f[informative] - second_f[informative],
                            axis=1,
                        ),
                        5.0,
                    )
                )
            )
            if len(first_s) >= 3:
                transitions = (
                    informative[:-1]
                    | informative[1:]
                )
                transition_delta = (
                    float(
                        np.mean(
                            np.minimum(
                                np.abs(
                                    np.diff(first_s)
                                    - np.diff(second_s)
                                )[transitions],
                                4.0,
                            )
                        )
                    )
                    if np.any(transitions)
                    else 0.0
                )
            else:
                transition_delta = 0.0
            score = (
                one_sided_penalty * one_sided_ratio
                + profile_weight * (0.55 * soft_delta + 0.45 * feature_delta)
                + transition_weight * transition_delta
                - match_bonus * matched_ratio
            )
            option = (float(score), informative_count, matched_count)
            if best is None or option[0] < best[0]:
                best = option
        return best if best is not None else (0.0, 0, 0)

    def _edge_pattern_score(
        self,
        first_index: int,
        first_edge_index: int,
        second_index: int,
        second_edge_index: int,
    ) -> float:
        """Fast source-image evidence for one proposed edge docking.

        Similar hand-cut fragments often have several almost equal sloping
        edges.  Geometry alone then orders the DFS arbitrarily and the correct
        arrangement may never reach the final-candidate beam.  Score the two
        original edge strips before search: corresponding samples run in
        opposite directions because a legal rigid docking reverses one edge.
        Blank/blank is neutral, matched printed structure is rewarded, and a
        one-sided J/Q/K stroke is penalised.
        """

        if (
            not self.card_mode
            or not self.use_texture
            or self.lab_image is None
        ):
            return 0.0
        first_polygon = self.polygons[first_index]
        second_polygon = self.polygons[second_index]
        first_start = first_polygon[first_edge_index]
        first_end = first_polygon[(first_edge_index + 1) % len(first_polygon)]
        second_start = second_polygon[second_edge_index]
        second_end = second_polygon[
            (second_edge_index + 1) % len(second_polygon)
        ]
        first_length = float(np.linalg.norm(first_end - first_start))
        second_length = float(np.linalg.norm(second_end - second_start))
        overlap = min(first_length, second_length)
        if overlap < 3.0:
            return 0.0
        sample_count = max(10, min(36, int(overlap / 1.2)))
        fractions = np.linspace(
            0.08, 0.92, sample_count, dtype=np.float64
        )[:, None]
        first_edge_points = (
            first_start + fractions * (first_end - first_start)
        )
        # Docked polygon edges have opposite directions.
        second_edge_points = (
            second_end + fractions * (second_start - second_end)
        )

        def inward_normal(
            polygon: np.ndarray, start: np.ndarray, end: np.ndarray
        ) -> np.ndarray:
            direction = end - start
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            normal = np.asarray(
                [-direction[1], direction[0]], dtype=np.float64
            )
            midpoint = (start + end) * 0.5
            if float(
                (polygon_centroid(polygon) - midpoint) @ normal
            ) < 0.0:
                normal *= -1.0
            return normal

        first_normal = inward_normal(
            first_polygon, first_start, first_end
        )
        second_normal = inward_normal(
            second_polygon, second_start, second_end
        )
        costs: list[tuple[float, int]] = []
        matched_count = 0
        informative_count = 0
        for offset in self.config.get(
            "card_pair_pattern_offsets_mm", [0.45, 1.4, 2.8]
        ):
            offset = max(0.2, float(offset))
            first_lab = self._sample_lab(
                first_edge_points + first_normal * offset
            )
            second_lab = self._sample_lab(
                second_edge_points + second_normal * offset
            )
            profile_score, informative, matched = (
                self._pattern_profile_continuity(
                    first_lab,
                    first_index,
                    second_lab,
                    second_index,
                )
            )
            if informative <= 0:
                continue
            informative_count += informative
            matched_count += matched
            costs.append((profile_score, informative))
        if not costs:
            return 0.0
        weighted_cost = sum(
            score * evidence for score, evidence in costs
        ) / max(sum(evidence for _, evidence in costs), 1)
        # A seam supported at several inward offsets is more reliable than one
        # accidental dark pixel.  Keep the bonus bounded so geometry remains
        # the primary legality test.
        evidence_bonus = min(
            1.5, 2.5 * matched_count / max(informative_count, 1)
        )
        return float(weighted_cost - evidence_bonus)

    def _precompute_pair_options(
        self,
    ) -> dict[
        tuple[int, int, bool],
        list[
            tuple[
                float,
                int,
                int,
                np.ndarray,
                np.ndarray,
            ]
        ],
    ]:
        """Precompute every rigid edge docking once.

        The previous depth-first search rebuilt the same edge lengths,
        rotations and translations at every node.  With at most four
        five-sided pieces, all pairwise docking options are small enough to
        cache and later compose with the placed piece transform.
        """

        result: dict[
            tuple[int, int, bool],
            list[
                tuple[
                    float,
                    int,
                    int,
                    np.ndarray,
                    np.ndarray,
                ]
            ],
        ] = {}
        lengths = [edge_lengths(polygon) for polygon in self.polygons]
        tolerance = float(self.config["edge_length_tolerance_mm"])
        relative = float(self.config["edge_length_relative_tolerance"])
        # With only two or three pieces there are fewer independent seams to
        # average out contour/perspective error.  Permit a modest additional
        # edge-length error, while the final filled-rectangle and overlap
        # checks remain unchanged.
        if not self.card_mode and len(self.polygons) <= 3:
            tolerance = max(
                tolerance,
                float(self.config.get("sparse_edge_length_tolerance_mm", 5.5)),
            )
            relative = max(
                relative,
                float(
                    self.config.get(
                        "sparse_edge_length_relative_tolerance", 0.14
                    )
                ),
            )
        minimum_solver_edge = float(
            self.config.get("minimum_solver_edge_mm", 8.0)
        )
        internal_partial_fractions = (
            None
            if self.card_mode
            else tuple(
                float(value)
                for value in self.config.get(
                    "partial_alignment_fractions", [0.25, 0.5, 0.75]
                )
            )
        )
        for new_index, new_polygon in enumerate(self.polygons):
            for placed_index, placed_polygon in enumerate(self.polygons):
                if new_index == placed_index:
                    continue
                exact: list[
                    tuple[
                        float,
                        int,
                        int,
                        np.ndarray,
                        np.ndarray,
                    ]
                ] = []
                partial: list[
                    tuple[
                        float,
                        int,
                        int,
                        np.ndarray,
                        np.ndarray,
                    ]
                ] = []
                for new_edge_index in range(len(new_polygon)):
                    if (
                        self.card_mode
                        and new_edge_index
                        in self.excluded_solver_edges[new_index]
                    ):
                        continue
                    source_edge = np.asarray(
                        [
                            new_polygon[new_edge_index],
                            new_polygon[
                                (new_edge_index + 1) % len(new_polygon)
                            ],
                        ],
                        dtype=np.float64,
                    )
                    for placed_edge_index in range(len(placed_polygon)):
                        if (
                            self.card_mode
                            and placed_edge_index
                            in self.excluded_solver_edges[placed_index]
                        ):
                            continue
                        destination_edge = np.asarray(
                            [
                                placed_polygon[placed_edge_index],
                                placed_polygon[
                                    (placed_edge_index + 1)
                                    % len(placed_polygon)
                                ],
                            ],
                            dtype=np.float64,
                        )
                        new_length = float(
                            lengths[new_index][new_edge_index]
                        )
                        placed_length = float(
                            lengths[placed_index][placed_edge_index]
                        )
                        if (
                            new_length < minimum_solver_edge
                            or placed_length < minimum_solver_edge
                        ):
                            continue
                        length_error = abs(new_length - placed_length)
                        allowed = max(
                            tolerance,
                            relative * min(new_length, placed_length),
                        )
                        options = edge_alignment_transforms(
                            source_edge,
                            destination_edge,
                            tolerance,
                            relative,
                            float(
                                self.config.get(
                                    "minimum_partial_remainder_mm",
                                    18.0,
                                )
                            ),
                            internal_partial_fractions,
                        )
                        for r, t, overlap in options:
                            pattern_score = self._edge_pattern_score(
                                new_index,
                                new_edge_index,
                                placed_index,
                                placed_edge_index,
                            )
                            # A long edge is not necessarily an internal seam:
                            # in many legal cuts it is an outer rectangle side.
                            # The old ``-overlap`` term therefore sent DFS down
                            # large numbers of attractive-looking outer-edge
                            # branches before it reached the true, shorter cut
                            # edge.  Keep a small overlap preference but make
                            # measured edge-length agreement the primary local
                            # cue.  Completed layouts are still accepted only
                            # by the global rectangle/union checks below.
                            item = (
                                -float(
                                    self.config.get(
                                        "edge_overlap_order_weight", 0.08
                                    )
                                )
                                * overlap
                                + float(
                                    self.config.get(
                                        "edge_length_error_order_weight", 0.50
                                    )
                                )
                                * length_error
                                + float(
                                    self.config.get(
                                        "card_pair_pattern_weight", 0.9
                                    )
                                )
                                * pattern_score,
                                new_edge_index,
                                placed_edge_index,
                                r,
                                t,
                            )
                            partial.append(item)
                            if length_error <= allowed:
                                exact.append(item)
                exact_limit = int(
                    self.config.get("max_pair_options_exact", 18)
                )
                partial_limit = int(
                    self.config.get("max_pair_options_partial", 28)
                )
                result[(new_index, placed_index, False)] = sorted(
                    exact, key=lambda item: item[0]
                )[:exact_limit]
                result[(new_index, placed_index, True)] = sorted(
                    partial, key=lambda item: item[0]
                )[:partial_limit]
        return result

    def _state_key(
        self, transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    ) -> tuple[Any, ...]:
        key: list[Any] = []
        for index in sorted(transforms):
            r, t = transforms[index]
            centroid = transform_points(
                self.centroids[index][None, :], r, t
            )[0]
            key.extend(
                [
                    index,
                    round(transform_angle_deg(r) / 2.0),
                    round(float(centroid[0])),
                    round(float(centroid[1])),
                ]
            )
        return tuple(key)

    def _placed_polygon(
        self, index: int, transforms: dict[int, tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        r, t = transforms[index]
        return transform_points(self.polygons[index], r, t)

    def _valid_new_polygon(
        self,
        polygon: np.ndarray,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        # During search, independently detected corners can make two truly
        # matching seams overlap by a thin noisy sliver.  Keep those candidates
        # and let the completed-layout union score decide; the final motion
        # plan still uses the stricter overlap_tolerance_mm.
        tolerance = float(
            self.config.get(
                "search_overlap_tolerance_mm",
                self.config["overlap_tolerance_mm"],
            )
        )
        for index in transforms:
            if polygons_overlap(
                polygon, self._placed_polygon(index, transforms), tolerance
            ):
                return False
        if self.card_target_size is not None:
            tolerance = float(
                self.config.get("card_boundary_tolerance_mm", 4.0)
            )
            outside = (
                max(0.0, -float(np.min(polygon[:, 0])))
                + max(
                    0.0,
                    float(np.max(polygon[:, 0]))
                    - float(self.card_target_size[0]),
                )
                + max(0.0, -float(np.min(polygon[:, 1])))
                + max(
                    0.0,
                    float(np.max(polygon[:, 1]))
                    - float(self.card_target_size[1]),
                )
            )
            if outside > tolerance:
                return False
        all_polygons = [
            self._placed_polygon(index, transforms) for index in transforms
        ] + [polygon]
        _, long_side, short_side, _ = min_area_rectangle(all_polygons)
        slack = float(self.config.get("search_dimension_slack_mm", 8.0))
        if (
            long_side > float(self.config["max_width_mm"]) + slack
            or short_side > float(self.config["max_height_mm"]) + slack
        ):
            return False
        # The minimum enclosing rectangle cannot become smaller after adding
        # more pieces.  Reject a sparse partial layout that could never reach
        # the configured minimum fill ratio.
        total_area = sum(item.area_mm2 for item in self.observations)
        search_fill = float(
            self.config.get("search_minimum_fill_ratio", 0.72)
        )
        return (
            long_side * short_side
            <= total_area / max(search_fill, 1e-6)
        )

    def _evaluate(
        self,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
        matches: list[tuple[int, int, int, int]],
    ) -> None:
        polygons = [
            self._placed_polygon(index, transforms)
            for index in range(len(self.polygons))
        ]
        box, long_side, short_side, theta = min_area_rectangle(polygons)
        if self.card_target_size is not None:
            card_width = float(self.card_target_size[0])
            card_height = float(self.card_target_size[1])
            long_side = max(card_width, card_height)
            short_side = min(card_width, card_height)
            rectangle_area = max(card_width * card_height, 1e-6)
        else:
            rectangle_area = max(long_side * short_side, 1e-6)
        pieces_area = sum(item.area_mm2 for item in self.observations)
        fill_ratio = pieces_area / rectangle_area
        if fill_ratio > float(
            self.config.get("maximum_candidate_fill_ratio", 1.08)
        ):
            return

        # Area alone cannot distinguish a true tiling from a layout with one
        # large overlap and an equally large empty corner.  Measure every
        # pairwise intersection and score the actual union against the
        # enclosing rectangle.  With at most four pieces this global check is
        # cheap, and it prevents a locally attractive edge match from winning
        # before the remaining pieces have formed a rectangle.
        overlap_area = sum(
            _polygon_intersection_area(
                first,
                second,
                pixels_per_mm=float(
                    self.config.get(
                        "candidate_overlap_pixels_per_mm", 1.0
                    )
                ),
            )
            for first, second in itertools.combinations(polygons, 2)
        )
        estimated_union_area = max(0.0, pieces_area - overlap_area)
        gap_area = max(0.0, rectangle_area - estimated_union_area)
        overfill_area = max(0.0, estimated_union_area - rectangle_area)
        overlap_ratio = overlap_area / rectangle_area
        gap_ratio = gap_area / rectangle_area

        min_w = float(self.config["min_width_mm"])
        max_w = float(self.config["max_width_mm"])
        min_h = float(self.config["min_height_mm"])
        max_h = float(self.config["max_height_mm"])
        dimension_penalty = (
            max(0.0, min_w - long_side)
            + max(0.0, long_side - max_w)
            + max(0.0, min_h - short_side)
            + max(0.0, short_side - max_h)
        )
        if dimension_penalty > 12.0:
            return
        # A missing region counts once; an overlap counts twice because the
        # same paper area both covers the wrong location and leaves a hole
        # elsewhere.  This score is zero only for a non-overlapping, full
        # rectangular tiling of a legal target size.
        layout_error = (
            gap_area + overfill_area + 2.0 * overlap_area
        ) / rectangle_area
        score = 100.0 * layout_error + 1.5 * dimension_penalty
        if self.card_target_size is not None:
            all_points = np.vstack(polygons)
            extent_error = (
                abs(float(np.min(all_points[:, 0])))
                + abs(float(np.min(all_points[:, 1])))
                + abs(
                    float(np.max(all_points[:, 0]))
                    - float(self.card_target_size[0])
                )
                + abs(
                    float(np.max(all_points[:, 1]))
                    - float(self.card_target_size[1])
                )
            )
            if extent_error > float(
                self.config.get("card_maximum_extent_error_mm", 12.0)
            ):
                return
            score += float(
                self.config.get("card_extent_score_weight", 2.0)
            ) * extent_error
        if self.card_mode:
            expected_aspect = float(
                self.config.get("card_aspect_ratio", 88.0 / 57.0)
            )
            aspect_error = abs(
                long_side / max(short_side, 1e-6) - expected_aspect
            )
            if aspect_error > float(
                self.config.get("card_maximum_aspect_error", 0.22)
            ):
                return
            score += float(
                self.config.get("card_aspect_score_weight", 65.0)
            ) * aspect_error
        rank_score = 0.0
        rank_corner_distance: float | None = None
        if (
            self.card_mode
            and self.card_rank_piece_index is not None
            and self.card_rank_center_mm is not None
            and self.card_rank_piece_index in transforms
        ):
            rank_r, rank_t = transforms[self.card_rank_piece_index]
            rank_point = transform_points(
                self.card_rank_center_mm[None, :], rank_r, rank_t
            )[0]
            rank_corner_distance = float(
                np.min(np.linalg.norm(box - rank_point, axis=1))
            )
            expected_inset = float(
                self.config.get("card_rank_corner_distance_mm", 10.5)
            )
            free_tolerance = float(
                self.config.get("card_rank_corner_tolerance_mm", 5.0)
            )
            rank_score = max(
                0.0,
                abs(rank_corner_distance - expected_inset)
                - free_tolerance,
            )
        candidate = AssemblyCandidate(
            transforms={
                index: (value[0].copy(), value[1].copy())
                for index, value in transforms.items()
            },
            matches=list(matches),
            geometry_score=score,
            long_side_mm=long_side,
            short_side_mm=short_side,
            fill_ratio=fill_ratio,
            box=box,
            theta=theta,
            overlap_area_mm2=overlap_area,
            gap_area_mm2=gap_area,
            overlap_ratio=overlap_ratio,
            gap_ratio=gap_ratio,
            card_rank_score=rank_score,
            card_rank_corner_distance_mm=rank_corner_distance,
        )
        self.candidates.append(candidate)
        self.candidates.sort(key=lambda item: item.geometry_score)
        maximum = int(self.config["max_final_candidates"])
        if len(self.candidates) > maximum:
            self.candidates = self.candidates[:maximum]

    def _search(
        self,
        transforms: dict[int, tuple[np.ndarray, np.ndarray]],
        matches: list[tuple[int, int, int, int]],
    ) -> None:
        if self.candidates:
            best = self.candidates[0]
            if (
                best.geometry_score
                <= float(
                    self.config.get("early_stop_geometry_score", 2.5)
                )
                and best.overlap_ratio
                <= float(
                    self.config.get("early_stop_overlap_ratio", 0.003)
                )
                and best.fill_ratio
                >= float(
                    self.config.get("early_stop_fill_ratio", 0.97)
                )
            ):
                return
        if (
            time.perf_counter() >= self.deadline
            and self.nodes >= self.timeout_node_floor
        ):
            self.timed_out = True
            return
        if self.nodes >= int(self.config["max_search_nodes"]):
            return
        self.nodes += 1
        state_key = self._state_key(transforms)
        if state_key in self.visited:
            return
        self.visited.add(state_key)
        if len(transforms) == len(self.polygons):
            self._evaluate(transforms, matches)
            return

        unplaced = [index for index in range(len(self.polygons)) if index not in transforms]
        generated: list[
            tuple[float, int, int, int, np.ndarray, np.ndarray, int]
        ] = []
        for new_index in unplaced:
            for placed_index, placed_transform in transforms.items():
                for (
                    option_score,
                    new_edge_index,
                    placed_edge_index,
                    local_r,
                    local_t,
                ) in self.pair_options.get(
                    (
                        new_index,
                        placed_index,
                        self.allow_partial_edges,
                    ),
                    [],
                ):
                    r, t = compose_transforms(
                        local_r,
                        local_t,
                        placed_transform[0],
                        placed_transform[1],
                    )
                    generated.append(
                        (
                            option_score,
                            new_index,
                            new_edge_index,
                            placed_edge_index,
                            r,
                            t,
                            placed_index,
                        )
                    )

        if not self.pose_hints:
            if self.card_mode or not bool(
                self.config.get("autonomous_compactness_ordering", True)
            ):
                generated.sort(key=lambda item: item[0])
            else:
                # Rank autonomous branches by the density of the partial
                # assembly as well as by edge agreement.  A correct seam tends
                # to add paper area without greatly enlarging its enclosing
                # rectangle, whereas docking two unrelated outer sides creates
                # a large empty wedge.  This only changes deterministic search
                # order; it does not prune a geometrically legal transform.
                placed_polygons = [
                    self._placed_polygon(index, transforms)
                    for index in transforms
                ]
                placed_area = sum(
                    self.observations[index].area_mm2
                    for index in transforms
                )
                total_piece_area = sum(
                    item.area_mm2 for item in self.observations
                )
                compactness_weight = float(
                    self.config.get(
                        "autonomous_compactness_order_weight", 0.022
                    )
                )
                final_box_weight = float(
                    self.config.get(
                        "autonomous_final_box_order_weight", 0.045
                    )
                )
                dimension_weight = float(
                    self.config.get(
                        "autonomous_dimension_order_weight", 3.0
                    )
                )
                ranked: list[
                    tuple[
                        float,
                        tuple[
                            float,
                            int,
                            int,
                            int,
                            np.ndarray,
                            np.ndarray,
                            int,
                        ],
                    ]
                ] = []
                for item in generated:
                    (
                        option_score,
                        new_index,
                        _,
                        _,
                        r,
                        t,
                        _,
                    ) = item
                    transformed = transform_points(
                        self.polygons[new_index], r, t
                    )
                    _, long_side, short_side, _ = min_area_rectangle(
                        placed_polygons + [transformed]
                    )
                    box_area = max(long_side * short_side, 1e-6)
                    assembly_area = (
                        placed_area
                        + self.observations[new_index].area_mm2
                    )
                    quick_score = (
                        option_score
                        + compactness_weight
                        * max(0.0, box_area - assembly_area)
                        + final_box_weight
                        * max(0.0, box_area - total_piece_area)
                        + dimension_weight
                        * (
                            max(
                                0.0,
                                long_side
                                - float(self.config["max_width_mm"]),
                            )
                            + max(
                                0.0,
                                short_side
                                - float(self.config["max_height_mm"]),
                            )
                        )
                    )
                    ranked.append((quick_score, item))
                generated = [
                    item
                    for _, item in sorted(
                        ranked, key=lambda value: value[0]
                    )
                ]
        else:
            # A demonstrated layout provides stronger information than edge
            # length alone. Rank rigid dockings by their relative-pose error
            # and partial-assembly compactness.
            placed_polygons = [
                self._placed_polygon(index, transforms)
                for index in transforms
            ]
            placed_area = sum(
                self.observations[index].area_mm2
                for index in transforms
            )
            total_piece_area = sum(
                item.area_mm2 for item in self.observations
            )
            ranked = []
            for item in generated:
                (
                    option_score,
                    new_index,
                    _,
                    _,
                    r,
                    t,
                    placed_index,
                ) = item
                transformed = transform_points(
                    self.polygons[new_index], r, t
                )
                _, long_side, short_side, _ = min_area_rectangle(
                    placed_polygons + [transformed]
                )
                box_area = max(long_side * short_side, 1e-6)
                assembly_area = (
                    placed_area
                    + self.observations[new_index].area_mm2
                )
                quick_score = (
                    option_score
                    + 0.018
                    * max(0.0, box_area - assembly_area)
                    + 0.055
                    * max(0.0, box_area - total_piece_area)
                    + 4.0
                    * (
                        max(
                            0.0,
                            long_side
                            - float(self.config["max_width_mm"]),
                        )
                        + max(
                            0.0,
                            short_side
                            - float(self.config["max_height_mm"]),
                        )
                    )
                )
                placed_r, placed_t = transforms[placed_index]
                actual_new = transform_points(
                    self.centroids[new_index][None, :],
                    r,
                    t,
                )[0]
                actual_placed = transform_points(
                    self.centroids[placed_index][None, :],
                    placed_r,
                    placed_t,
                )[0]
                hint_placed_r, _ = self.pose_hints[
                    placed_index
                ]
                actual_relative = (
                    actual_new - actual_placed
                ) @ placed_r.T
                hint_relative = (
                    self.hint_centroids[new_index]
                    - self.hint_centroids[placed_index]
                ) @ hint_placed_r.T
                angle_error = abs(
                    wrap_angle_deg(
                        transform_angle_deg(r)
                        - transform_angle_deg(placed_r)
                        - transform_angle_deg(
                            self.pose_hints[new_index][0]
                        )
                        + transform_angle_deg(hint_placed_r)
                    )
                )
                hint_deviation = float(
                    np.linalg.norm(
                        actual_relative - hint_relative
                    )
                ) + 0.18 * angle_error
                quick_score += float(
                    self.config.get("pose_hint_weight", 0.0)
                ) * hint_deviation
                ranked.append((quick_score, item))
            generated = [
                item
                for _, item in sorted(
                    ranked, key=lambda value: value[0]
                )
            ]
            # The demonstrated relative poses make hundreds of distant edge
            # dockings provably unhelpful as first choices.  Keep a generous
            # deterministic beam at each DFS node; the final geometry checks
            # remain unchanged.
            generated = generated[
                : int(
                    self.config.get(
                        "pose_hint_branch_limit", 36
                    )
                )
            ]
        local_seen: set[tuple[Any, ...]] = set()
        for (
            _,
            new_index,
            new_edge_index,
            placed_edge_index,
            r,
            t,
            placed_index,
        ) in generated:
            centroid = transform_points(
                self.centroids[new_index][None, :], r, t
            )[0]
            placement_key = (
                new_index,
                round(transform_angle_deg(r)),
                round(float(centroid[0])),
                round(float(centroid[1])),
            )
            if placement_key in local_seen:
                continue
            local_seen.add(placement_key)
            transformed_polygon = transform_points(
                self.polygons[new_index], r, t
            )
            if not self._valid_new_polygon(transformed_polygon, transforms):
                continue
            next_transforms = dict(transforms)
            next_transforms[new_index] = (r, t)
            next_matches = matches + [
                (
                    placed_index,
                    placed_edge_index,
                    new_index,
                    new_edge_index,
                )
            ]
            self._search(next_transforms, next_matches)
            if self.nodes >= int(self.config["max_search_nodes"]):
                break

    def _sample_lab(self, points_mm: np.ndarray) -> np.ndarray:
        assert self.lab_image is not None
        x = (points_mm[:, 0] * self.ppm).astype(np.float32).reshape(-1, 1)
        y = (points_mm[:, 1] * self.ppm).astype(np.float32).reshape(-1, 1)
        return cv2.remap(
            self.lab_image,
            x,
            y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1, 3)

    def _best_anchor(self, allow_partial: bool) -> int:
        """Choose the piece that creates the fewest first-level branches.

        Every complete assembly can be expressed relative to any piece, so a
        more distinctive anchor cannot remove a valid solution.  It does avoid
        spending the real-time budget on a large piece whose several similar
        outer edges all attract accidental matches.
        """

        counts: list[tuple[int, int]] = []
        for anchor in range(len(self.polygons)):
            branch_count = sum(
                len(
                    self.pair_options.get(
                        (new_index, anchor, allow_partial), []
                    )
                )
                for new_index in range(len(self.polygons))
                if new_index != anchor
            )
            counts.append((branch_count, anchor))
        return min(counts)[1]

    def _texture_score(self, candidate: AssemblyCandidate) -> float:
        if not self.use_texture or self.image is None:
            return 0.0
        profile_costs: list[tuple[float, int]] = []
        informative_samples = 0
        matched_pattern_samples = 0
        offsets = self.config.get("card_seam_sample_offsets_mm")
        if not isinstance(offsets, (list, tuple)) or not offsets:
            offsets = [0.45, float(self.config["seam_sample_offset_mm"])]
        offsets = [max(0.2, float(value)) for value in offsets]
        for first, first_edge, second, second_edge in candidate.matches:
            first_r, first_t = candidate.transforms[first]
            second_r, second_t = candidate.transforms[second]
            first_poly = transform_points(self.polygons[first], first_r, first_t)
            second_poly = transform_points(self.polygons[second], second_r, second_t)
            a = first_poly[first_edge]
            b = first_poly[(first_edge + 1) % len(first_poly)]
            c = second_poly[second_edge]
            d = second_poly[(second_edge + 1) % len(second_poly)]
            direction = b - a
            length = float(np.linalg.norm(direction))
            if length < 2.0:
                continue
            unit = direction / length
            second_projection = [(float((p - a) @ unit)) for p in (c, d)]
            lower = max(1.0, min(second_projection))
            upper = min(length - 1.0, max(second_projection))
            if upper - lower < 2.0:
                continue
            count = max(12, int((upper - lower) / 1.25))
            seam = a + np.linspace(lower, upper, count)[:, None] * unit
            normal = np.array([-unit[1], unit[0]], dtype=np.float64)
            first_centroid = polygon_centroid(first_poly)
            second_centroid = polygon_centroid(second_poly)
            if np.dot(first_centroid - np.mean(seam, axis=0), normal) < 0:
                first_normal = -normal
            else:
                first_normal = normal
            if np.dot(second_centroid - np.mean(seam, axis=0), normal) < 0:
                second_normal = -normal
            else:
                second_normal = normal
            first_inv_r, first_inv_t = invert_transform(first_r, first_t)
            second_inv_r, second_inv_t = invert_transform(second_r, second_t)
            for offset in offsets:
                first_source = transform_points(
                    seam + first_normal * offset,
                    first_inv_r,
                    first_inv_t,
                )
                second_source = transform_points(
                    seam + second_normal * offset,
                    second_inv_r,
                    second_inv_t,
                )
                first_lab = self._sample_lab(first_source)
                second_lab = self._sample_lab(second_source)
                profile_score, informative, matched = (
                    self._pattern_profile_continuity(
                        first_lab,
                        first,
                        second_lab,
                        second,
                    )
                )
                if informative > 0:
                    informative_samples += informative
                    matched_pattern_samples += matched
                    profile_costs.append((profile_score, informative))
        if informative_samples == 0:
            # No visible pattern intersects any proposed seam.  This is
            # deliberately neutral: blank-to-blank must not beat a candidate
            # supported by continuous J/Q/K artwork.
            return float(
                self.config.get("card_texture_neutral_score", 50.0)
            )
        profile_cost = sum(
            cost * evidence for cost, evidence in profile_costs
        ) / max(
            sum(evidence for _, evidence in profile_costs),
            1,
        )
        evidence_bonus = min(
            4.0,
            8.0 * matched_pattern_samples / max(informative_samples, 1),
        )
        return max(0.0, 18.0 + profile_cost - evidence_bonus)

    def _card_rotational_symmetry_score(
        self, candidate: AssemblyCandidate
    ) -> float:
        """Compare the assembled artwork with its 180-degree counterpart.

        Ordinary number cards and double-headed J/Q/K artwork are designed
        around half-turn symmetry.  This global evidence resolves layouts in
        which every candidate seam lies mostly on blank paper.  Geometry-only
        and detected-joker paths leave this score disabled.
        """

        if (
            not self.card_mode
            or not self.use_texture
            or self.lab_image is None
            or not bool(
                self.config.get(
                    "card_rotational_symmetry_enabled", True
                )
            )
        ):
            return 0.0
        box = np.asarray(candidate.box, dtype=np.float64)
        if box.shape != (4, 2):
            return 0.0
        first_axis = box[1] - box[0]
        second_axis = box[3] - box[0]
        if (
            np.linalg.norm(first_axis) < 1e-6
            or np.linalg.norm(second_axis) < 1e-6
        ):
            return 0.0
        rows = int(self.config.get("card_symmetry_grid_rows", 34))
        columns = int(self.config.get("card_symmetry_grid_columns", 22))
        u, v = np.meshgrid(
            np.linspace(0.04, 0.96, max(8, columns)),
            np.linspace(0.04, 0.96, max(12, rows)),
        )
        points = (
            box[0]
            + u.reshape(-1, 1) * first_axis
            + v.reshape(-1, 1) * second_axis
        )
        opposite = (
            box[0]
            + (1.0 - u.reshape(-1, 1)) * first_axis
            + (1.0 - v.reshape(-1, 1)) * second_axis
        )

        placed_polygons = {
            index: transform_points(
                self.polygons[index], transform[0], transform[1]
            )
            for index, transform in candidate.transforms.items()
        }

        def assembled_lab(
            query: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = np.zeros((len(query), 3), dtype=np.float32)
            valid = np.zeros(len(query), dtype=bool)
            owners = np.full(len(query), -1, dtype=np.int16)
            for index, polygon in placed_polygons.items():
                contour = polygon.astype(np.float32).reshape(-1, 1, 2)
                selected = np.asarray(
                    [
                        not valid[position]
                        and cv2.pointPolygonTest(
                            contour,
                            (
                                float(point[0]),
                                float(point[1]),
                            ),
                            False,
                        )
                        >= 0.0
                        for position, point in enumerate(query)
                    ],
                    dtype=bool,
                )
                if not np.any(selected):
                    continue
                rotation, translation = candidate.transforms[index]
                inverse_rotation, inverse_translation = invert_transform(
                    rotation, translation
                )
                source = transform_points(
                    query[selected],
                    inverse_rotation,
                    inverse_translation,
                )
                values[selected] = self._sample_lab(source)
                valid[selected] = True
                owners[selected] = index
            return values, valid, owners

        first_lab, first_valid, first_owners = assembled_lab(points)
        second_lab, second_valid, second_owners = assembled_lab(opposite)
        valid = first_valid & second_valid
        if np.count_nonzero(valid) < 24:
            return 0.0
        first_lab = first_lab[valid]
        second_lab = second_lab[valid]
        first_owners = first_owners[valid]
        second_owners = second_owners[valid]
        first_pattern = np.zeros(len(first_lab), dtype=bool)
        second_pattern = np.zeros(len(second_lab), dtype=bool)
        for piece_index in candidate.transforms:
            first_selected = first_owners == piece_index
            second_selected = second_owners == piece_index
            if np.any(first_selected):
                first_pattern[first_selected] = self._pattern_mask(
                    first_lab[first_selected], piece_index
                )
            if np.any(second_selected):
                second_pattern[second_selected] = self._pattern_mask(
                    second_lab[second_selected], piece_index
                )
        informative = first_pattern | second_pattern
        if np.count_nonzero(informative) < int(
            self.config.get("card_symmetry_minimum_pattern_samples", 12)
        ):
            return 0.0
        mismatch = np.logical_xor(first_pattern, second_pattern)
        colour_delta = np.linalg.norm(
            first_lab - second_lab, axis=1
        ) / 12.0
        matched = first_pattern & second_pattern
        costs = (
            10.0 * mismatch[informative].astype(np.float32)
            + np.minimum(colour_delta[informative], 8.0)
            - 2.0 * matched[informative].astype(np.float32)
        )
        return max(0.0, float(np.mean(costs)))

    def solve(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not 1 <= len(self.polygons) <= 4:
            raise SolveError("Unknown mode supports one to four pieces")
        search_started = time.perf_counter()
        total_budget = float(self.config.get("max_search_seconds", 8.0))
        exact_budget = min(
            total_budget,
            float(self.config.get("exact_search_seconds", 1.5)),
        )
        self.deadline = search_started + exact_budget
        self.timeout_node_floor = (
            0
            if self.card_mode
            else int(self.config.get("exact_search_node_floor", 40))
        )
        # Most manufactured puzzles have one-to-one internal edges. Search that
        # small space first. If it cannot form a convincing rectangle, enable
        # partial long-edge matches for T-junction layouts.
        self.allow_partial_edges = False
        exact_anchor = (
            min(self.initial_transforms)
            if self.initial_transforms
            else self._best_anchor(False)
        )
        self._search(
            (
                dict(self.initial_transforms)
                if self.initial_transforms
                else {exact_anchor: (self.identity, self.zero)}
            ),
            [],
        )
        exact_nodes = self.nodes
        exact_timed_out = self.timed_out
        best_exact = (
            min((item.geometry_score for item in self.candidates), default=float("inf"))
        )
        partial_searched = False
        partial_timed_out = False
        if not self.candidates or best_exact > float(
            self.config.get("exact_search_accept_score", 15.0)
        ):
            partial_searched = True
            self.allow_partial_edges = True
            self.visited.clear()
            self.nodes = 0
            self.timed_out = False
            self.deadline = search_started + total_budget
            self.timeout_node_floor = (
                0
                if self.card_mode
                else int(
                    self.config.get("partial_search_node_floor", 80)
                )
            )
            # If exact docking already produced a near-valid rectangle, keep
            # the same root and only add T-junction options.  Changing the root
            # here can send the time-limited search into a completely
            # different, much less useful branch order.
            partial_anchor = (
                exact_anchor
                if self.initial_transforms
                or (
                    self.candidates
                    and best_exact
                    <= float(
                        self.config.get(
                            "partial_reuse_exact_anchor_score", 16.0
                        )
                    )
                )
                else self._best_anchor(True)
            )
            self._search(
                (
                    dict(self.initial_transforms)
                    if self.initial_transforms
                    else {partial_anchor: (self.identity, self.zero)}
                ),
                [],
            )
            partial_timed_out = self.timed_out
        else:
            partial_anchor = None
        self.nodes += exact_nodes
        self.timed_out = partial_timed_out if partial_searched else exact_timed_out

        # A DFS rooted at one piece is geometrically complete but its ordering
        # matters under a hard real-time deadline.  Retry from piece 1 only
        # when the first pass has not produced an acceptable rectangle.  This
        # keeps easy frames fast and gives ambiguous shapes a different search
        # ordering without introducing reflections or random results.
        minimum_fill = float(
            self.config.get("minimum_accepted_fill_ratio", 0.90)
        )
        maximum_score = float(
            self.config.get("maximum_accepted_geometry_score", 16.0)
        )
        provisional_ok = any(
            item.fill_ratio >= minimum_fill
            and item.geometry_score <= maximum_score
            for item in self.candidates
        )
        fallback_anchors: list[int] = []
        fallback_timed_out = False
        primary_nodes = self.nodes
        used_anchors = {exact_anchor}
        if partial_anchor is not None:
            used_anchors.add(partial_anchor)
        need_pattern_alternatives = bool(
            self.card_mode
            and self.use_texture
            and len(self.candidates)
            < int(
                self.config.get(
                    "card_minimum_pattern_candidates", 3
                )
            )
        )
        if (
            (not provisional_ok or need_pattern_alternatives)
            and not self.initial_transforms
        ):
            remaining_anchors = [
                index
                for index in range(len(self.polygons))
                if index not in used_anchors
            ]
            # Under a deadline, DFS ordering depends on the root piece.  Spend
            # the fallback budget across every untried anchor rather than only
            # piece 1.  The order is deterministic and therefore an unchanged
            # source scene still produces the same target layout.
            fallback_budget = float(
                self.config.get(
                    (
                        "card_pattern_alternative_search_seconds"
                        if need_pattern_alternatives
                        else "fallback_search_seconds"
                    ),
                    0.70 if need_pattern_alternatives else 2.5,
                )
            )
            fallback_started = time.perf_counter()
            fallback_nodes = 0
            for position, fallback_anchor in enumerate(
                remaining_anchors
            ):
                remaining_time = (
                    fallback_budget
                    - (time.perf_counter() - fallback_started)
                )
                remaining_count = len(remaining_anchors) - position
                if remaining_time <= 0.05:
                    fallback_timed_out = True
                    break
                fallback_anchors.append(fallback_anchor)
                self.allow_partial_edges = True
                self.visited.clear()
                self.nodes = 0
                self.timed_out = False
                self.timeout_node_floor = (
                    0
                    if self.card_mode
                    else int(
                        self.config.get(
                            "fallback_search_node_floor", 140
                        )
                    )
                )
                self.deadline = (
                    time.perf_counter()
                    + remaining_time / max(1, remaining_count)
                )
                self._search(
                    {
                        fallback_anchor: (
                            self.identity,
                            self.zero,
                        )
                    },
                    [],
                )
                fallback_nodes += self.nodes
                fallback_timed_out = (
                    fallback_timed_out or self.timed_out
                )
                provisional_ok = any(
                    item.fill_ratio >= minimum_fill
                    and item.geometry_score <= maximum_score
                    for item in self.candidates
                )
                enough_pattern_candidates = (
                    not need_pattern_alternatives
                    or len(self.candidates)
                    >= int(
                        self.config.get(
                            "card_minimum_pattern_candidates", 3
                        )
                    )
                )
                if provisional_ok and enough_pattern_candidates:
                    break
            self.nodes = primary_nodes + fallback_nodes
            self.timed_out = (
                fallback_timed_out
                or partial_timed_out
                or exact_timed_out
            )
        if not self.candidates:
            raise SolveError(
                "No rectangular assembly was found. Check polygon vertices, "
                "edge-length calibration, or increase solver tolerances."
            )
        for candidate in self.candidates:
            candidate.texture_score = self._texture_score(candidate)
            candidate.card_symmetry_score = (
                self._card_rotational_symmetry_score(candidate)
            )
            candidate.total_score = candidate.geometry_score + float(
                self.config["texture_weight"]
            ) * candidate.texture_score + float(
                self.config.get("card_symmetry_score_weight", 0.65)
            ) * candidate.card_symmetry_score + float(
                self.config.get("card_rank_score_weight", 0.2)
            ) * candidate.card_rank_score
        self.candidates.sort(key=lambda item: item.total_score)
        best = self.candidates[0]

        # Rotate the long side horizontal, then centre the rectangle in the
        # configured lower A4 target zone.
        global_r = rotation_matrix_row(-best.theta)
        assembled_polygons = [
            transform_points(
                self.polygons[index],
                best.transforms[index][0],
                best.transforms[index][1],
            )
            for index in range(len(self.polygons))
        ]
        rotated = [polygon @ global_r for polygon in assembled_polygons]
        all_points = np.vstack(rotated)
        lower = np.min(all_points, axis=0)
        upper = np.max(all_points, axis=0)
        size = upper - lower
        target_orientation = str(
            self.config.get("target_orientation", "landscape")
        ).lower()
        rotate_to_portrait = (
            target_orientation == "portrait"
            and size[0] >= size[1]
        )
        rotate_to_landscape = (
            target_orientation != "portrait"
            and size[0] < size[1]
        )
        if rotate_to_portrait or rotate_to_landscape:
            extra = rotation_matrix_row(-math.pi / 2)
            global_r = global_r @ extra
            rotated = [polygon @ extra for polygon in rotated]
            all_points = np.vstack(rotated)
            lower = np.min(all_points, axis=0)
            upper = np.max(all_points, axis=0)
            size = upper - lower

        zone = np.asarray(self.config["target_zone_mm"], dtype=np.float64)
        zone_center = np.array(
            [(zone[0] + zone[2]) * 0.5, (zone[1] + zone[3]) * 0.5],
            dtype=np.float64,
        )
        target_lower = zone_center - size * 0.5
        global_t = target_lower - lower
        target_center = target_lower + size * 0.5
        clearance = float(self.config.get("placement_clearance_mm", 0.0))

        plan: list[dict[str, Any]] = []
        for index, observation in enumerate(self.observations):
            local_r, local_t = best.transforms[index]
            target_r, target_t = compose_transforms(
                local_r, local_t, global_r, global_t
            )
            if not is_proper_rotation(target_r):
                raise SolveError("Mirrored autonomous-piece transforms are forbidden")
            target_polygon = transform_points(
                observation.polygon_mm, target_r, target_t
            )
            if clearance > 0:
                direction = polygon_centroid(target_polygon) - target_center
                norm = float(np.linalg.norm(direction))
                if norm > 1e-6:
                    shift = direction / norm * clearance
                    target_t = target_t + shift
                    target_polygon = target_polygon + shift
            place = transform_points(
                observation.pickup_mm[None, :], target_r, target_t
            )[0]
            plan.append(
                {
                    "piece_id": observation.id,
                    "pick_mm": np.round(observation.pickup_mm, 3).tolist(),
                    "place_mm": np.round(place, 3).tolist(),
                    "rotate_deg": round(
                        wrap_angle_deg(transform_angle_deg(target_r)), 3
                    ),
                    "mirrored": False,
                    "target_polygon_mm": np.round(target_polygon, 3).tolist(),
                }
            )

        # The rectangle and every piece must share exactly the same A4-paper
        # coordinate offset.  Small contour/shadow errors otherwise leave a
        # visible outer-edge gap even though the matched seam is correct.
        # Snap only a piece extremum already close to an outer side; this adds
        # translation only and can never mirror or scale a piece.
        display_snap_tolerance = float(
            self.config.get("boundary_snap_tolerance_mm", 0.0)
        )
        # Moving individual pieces to make the four outer lines look exact can
        # destroy an already valid seam and create a large physical overlap.
        # Keep physical snapping opt-in and independent from the harmless
        # display-only vertex straightening performed below.
        snap_tolerance = float(
            self.config.get("motion_boundary_snap_tolerance_mm", 0.0)
        )
        target_upper = target_lower + size
        if snap_tolerance > 0.0:
            for item_index, item in enumerate(plan):
                polygon = np.asarray(
                    item["target_polygon_mm"], dtype=np.float64
                )
                shift = np.zeros(2, dtype=np.float64)
                for axis in range(2):
                    options: list[float] = []
                    lower_delta = float(
                        target_lower[axis] - np.min(polygon[:, axis])
                    )
                    upper_delta = float(
                        target_upper[axis] - np.max(polygon[:, axis])
                    )
                    if abs(lower_delta) <= snap_tolerance:
                        options.append(lower_delta)
                    if abs(upper_delta) <= snap_tolerance:
                        options.append(upper_delta)
                    if options:
                        shift[axis] = min(options, key=abs)
                requested_shift = shift.copy()
                if np.any(np.abs(shift) > 1e-9):
                    # Do not close an outer-edge gap by creating a real
                    # inter-piece overlap.  Reduce the correction until it is
                    # compatible with every other measured outline.
                    for fraction in (1.0, 0.75, 0.5, 0.25, 0.0):
                        candidate_shift = requested_shift * fraction
                        candidate = polygon + candidate_shift
                        conflicts = False
                        for other_index, other in enumerate(plan):
                            if other_index == item_index:
                                continue
                            other_polygon = np.asarray(
                                other["target_polygon_mm"], dtype=np.float64
                            )
                            if polygons_overlap(
                                candidate,
                                other_polygon,
                                float(self.config["overlap_tolerance_mm"]),
                            ):
                                conflicts = True
                                break
                        if not conflicts:
                            shift = candidate_shift
                            break
                if np.any(np.abs(shift) > 1e-9):
                    polygon += shift
                    place = np.asarray(item["place_mm"], dtype=np.float64) + shift
                    item["place_mm"] = np.round(place, 3).tolist()
                    item["target_polygon_mm"] = np.round(
                        polygon, 3
                    ).tolist()
                item["boundary_snap_mm"] = np.round(shift, 3).tolist()
                item["boundary_snap_requested_mm"] = np.round(
                    requested_shift, 3
                ).tolist()

        # Camera shadows and hand-cut fuzz can make two correctly docked
        # measured contours overlap by a thin sliver.  The task explicitly
        # forbids overlap, so card mode applies a final translation-only
        # relief.  It never rotates, scales or mirrors a fragment.  A tiny
        # visible gap is preferable to an unsafe robot command that stacks
        # paper.
        overlap_relief = [
            np.zeros(2, dtype=np.float64) for _ in plan
        ]
        if self.card_mode and len(plan) > 1:
            accepted_pair_overlap = float(
                self.config.get(
                    "maximum_accepted_pair_overlap_mm2", 0.5
                )
            )
            step = float(
                self.config.get("card_overlap_relief_step_mm", 0.15)
            )
            maximum_shift = float(
                self.config.get("card_overlap_relief_max_shift_mm", 2.0)
            )
            maximum_iterations = int(
                self.config.get("card_overlap_relief_iterations", 40)
            )

            def relief_cost(
                polygons: list[np.ndarray],
            ) -> float:
                overlap_cost = sum(
                    max(
                        0.0,
                        _polygon_intersection_area(first, second)
                        - accepted_pair_overlap,
                    )
                    for first, second in itertools.combinations(
                        polygons, 2
                    )
                )
                outside_cost = sum(
                    max(0.0, target_lower[0] - float(np.min(poly[:, 0])))
                    + max(0.0, float(np.max(poly[:, 0])) - target_upper[0])
                    + max(0.0, target_lower[1] - float(np.min(poly[:, 1])))
                    + max(0.0, float(np.max(poly[:, 1])) - target_upper[1])
                    for poly in polygons
                )
                # The target bounds are estimated from shadowed camera
                # contours.  Treat them as a soft guide in card mode: forcing
                # every noisy vertex inside that estimate can leave a thin
                # physical overlap at a seam.  A sub-millimetre outward
                # relief/gap is safer for the robot and does not alter the
                # fragment rotation, scale or handedness.
                outside_weight = float(
                    self.config.get(
                        "card_overlap_relief_outside_weight", 0.5
                    )
                )
                return overlap_cost + outside_weight * outside_cost

            # First use an exact separating-translation jump.  Repeating
            # 0.1 mm trial moves is unnecessarily expensive on an RDK and can
            # stop at a local minimum when three fragments meet.  Candidate
            # directions come from the pair-centre vector and every polygon
            # edge normal; a short binary search finds the smallest rigid
            # translation that removes that pair's measured sliver.  No
            # rotation, scale or reflection is introduced.
            direct_iterations = int(
                self.config.get(
                    "card_overlap_relief_direct_iterations", 4
                )
            )
            direct_binary_steps = int(
                self.config.get(
                    "card_overlap_relief_binary_steps", 6
                )
            )

            def available_distance(
                displacement: np.ndarray, direction: np.ndarray
            ) -> float:
                projection = float(displacement @ direction)
                remainder = (
                    projection * projection
                    + maximum_shift * maximum_shift
                    - float(displacement @ displacement)
                )
                if remainder <= 0.0:
                    return 0.0
                return max(0.0, -projection + math.sqrt(remainder))

            for _ in range(direct_iterations):
                polygons = [
                    np.asarray(
                        item["target_polygon_mm"], dtype=np.float64
                    )
                    for item in plan
                ]
                overlapping_pairs = [
                    (
                        _polygon_intersection_area(
                            polygons[first_index],
                            polygons[second_index],
                        ),
                        first_index,
                        second_index,
                    )
                    for first_index, second_index in itertools.combinations(
                        range(len(plan)), 2
                    )
                ]
                area, first_index, second_index = max(
                    overlapping_pairs, default=(0.0, 0, 0)
                )
                if area <= accepted_pair_overlap:
                    break
                center_delta = (
                    polygon_centroid(polygons[first_index])
                    - polygon_centroid(polygons[second_index])
                )
                directions: list[np.ndarray] = []
                edge_directions: list[np.ndarray] = []
                norm = float(np.linalg.norm(center_delta))
                if norm > 1e-8:
                    directions.append(center_delta / norm)
                for polygon in (
                    polygons[first_index],
                    polygons[second_index],
                ):
                    for edge_index in range(len(polygon)):
                        edge = (
                            polygon[(edge_index + 1) % len(polygon)]
                            - polygon[edge_index]
                        )
                        normal = np.asarray(
                            [-edge[1], edge[0]], dtype=np.float64
                        )
                        normal_norm = float(np.linalg.norm(normal))
                        if normal_norm <= 1e-8:
                            continue
                        normal /= normal_norm
                        if float(normal @ center_delta) < 0.0:
                            normal *= -1.0
                        if not any(
                            abs(float(normal @ existing)) > 0.999
                            for existing in directions
                        ):
                            directions.append(normal)
                            edge_directions.append(normal)
                first_contour = polygons[first_index].astype(
                    np.float32
                ).reshape(-1, 1, 2)
                second_contour = polygons[second_index].astype(
                    np.float32
                ).reshape(-1, 1, 2)
                if (
                    edge_directions
                    and cv2.isContourConvex(first_contour)
                    and cv2.isContourConvex(second_contour)
                ):
                    # For two convex polygons the minimum-translation
                    # separating axis is one of their edge normals.  Testing
                    # only that axis replaces dozens of equivalent RDK-side
                    # binary searches without changing the result.
                    def projected_overlap(axis: np.ndarray) -> float:
                        first_projection = (
                            polygons[first_index] @ axis
                        )
                        second_projection = (
                            polygons[second_index] @ axis
                        )
                        return float(
                            min(
                                np.max(first_projection),
                                np.max(second_projection),
                            )
                            - max(
                                np.min(first_projection),
                                np.min(second_projection),
                            )
                        )

                    directions = [
                        min(edge_directions, key=projected_overlap)
                    ]

                current_cost = relief_cost(polygons)
                direct_move: tuple[
                    int, np.ndarray, int | None, np.ndarray | None
                ] | None = None
                direct_cost = current_cost
                for direction in directions:
                    first_available = available_distance(
                        overlap_relief[first_index], direction
                    )
                    second_available = available_distance(
                        overlap_relief[second_index], -direction
                    )
                    variants = (
                        (
                            first_index,
                            direction,
                            first_available,
                            None,
                            None,
                        ),
                        (
                            second_index,
                            -direction,
                            second_available,
                            None,
                            None,
                        ),
                        (
                            first_index,
                            direction * 0.5,
                            2.0
                            * min(first_available, second_available),
                            second_index,
                            -direction * 0.5,
                        ),
                    )
                    for (
                        move_index,
                        move_direction,
                        maximum_distance,
                        other_index,
                        other_direction,
                    ) in variants:
                        if maximum_distance <= 1e-6:
                            continue

                        def pair_area(distance: float) -> float:
                            first = polygons[first_index].copy()
                            second = polygons[second_index].copy()
                            if move_index == first_index:
                                first += move_direction * distance
                            else:
                                second += move_direction * distance
                            if (
                                other_index == first_index
                                and other_direction is not None
                            ):
                                first += other_direction * distance
                            elif (
                                other_index == second_index
                                and other_direction is not None
                            ):
                                second += other_direction * distance
                            return _polygon_intersection_area(first, second)

                        if (
                            pair_area(maximum_distance)
                            > accepted_pair_overlap
                        ):
                            continue
                        lower_distance = 0.0
                        upper_distance = maximum_distance
                        for _ in range(direct_binary_steps):
                            middle = (
                                lower_distance + upper_distance
                            ) * 0.5
                            if (
                                pair_area(middle)
                                > accepted_pair_overlap
                            ):
                                lower_distance = middle
                            else:
                                upper_distance = middle
                        distance = upper_distance + 0.02
                        move = move_direction * distance
                        other_move = (
                            other_direction * distance
                            if other_direction is not None
                            else None
                        )
                        candidate_polygons = [
                            polygon.copy() for polygon in polygons
                        ]
                        candidate_polygons[move_index] += move
                        if (
                            other_index is not None
                            and other_move is not None
                        ):
                            candidate_polygons[other_index] += other_move
                        cost = relief_cost(candidate_polygons)
                        if cost + 1e-8 < direct_cost:
                            direct_cost = cost
                            direct_move = (
                                move_index,
                                move,
                                other_index,
                                other_move,
                            )
                if direct_move is None:
                    break
                move_index, move, other_index, other_move = direct_move
                move_targets = [(move_index, move)]
                if other_index is not None and other_move is not None:
                    move_targets.append((other_index, other_move))
                for item_index, displacement in move_targets:
                    polygon = np.asarray(
                        plan[item_index]["target_polygon_mm"],
                        dtype=np.float64,
                    )
                    place = np.asarray(
                        plan[item_index]["place_mm"], dtype=np.float64
                    )
                    plan[item_index]["target_polygon_mm"] = np.round(
                        polygon + displacement, 3
                    ).tolist()
                    plan[item_index]["place_mm"] = np.round(
                        place + displacement, 3
                    ).tolist()
                    overlap_relief[item_index] += displacement

            for _ in range(maximum_iterations):
                polygons = [
                    np.asarray(
                        item["target_polygon_mm"], dtype=np.float64
                    )
                    for item in plan
                ]
                current_cost = relief_cost(polygons)
                if current_cost <= 1e-6:
                    break
                best_move: tuple[
                    int, np.ndarray, int | None, np.ndarray | None
                ] | None = None
                best_cost = current_cost
                for first_index, second_index in itertools.combinations(
                    range(len(plan)), 2
                ):
                    area = _polygon_intersection_area(
                        polygons[first_index], polygons[second_index]
                    )
                    if area <= accepted_pair_overlap:
                        continue
                    directions: list[np.ndarray] = []
                    center_delta = (
                        polygon_centroid(polygons[first_index])
                        - polygon_centroid(polygons[second_index])
                    )
                    norm = float(np.linalg.norm(center_delta))
                    if norm > 1e-6:
                        directions.append(center_delta / norm)
                    for angle_index in range(16):
                        angle = angle_index * math.pi / 8.0
                        directions.append(
                            np.asarray(
                                [math.cos(angle), math.sin(angle)],
                                dtype=np.float64,
                            )
                        )
                    for direction in directions:
                        variants = (
                            (
                                first_index,
                                direction * step,
                                None,
                                None,
                            ),
                            (
                                second_index,
                                -direction * step,
                                None,
                                None,
                            ),
                            (
                                first_index,
                                direction * step * 0.5,
                                second_index,
                                -direction * step * 0.5,
                            ),
                        )
                        for (
                            move_index,
                            move,
                            other_index,
                            other_move,
                        ) in variants:
                            if (
                                np.linalg.norm(
                                    overlap_relief[move_index] + move
                                )
                                > maximum_shift
                            ):
                                continue
                            if (
                                other_index is not None
                                and other_move is not None
                                and np.linalg.norm(
                                    overlap_relief[other_index]
                                    + other_move
                                )
                                > maximum_shift
                            ):
                                continue
                            candidate_polygons = [
                                polygon.copy() for polygon in polygons
                            ]
                            candidate_polygons[move_index] += move
                            if (
                                other_index is not None
                                and other_move is not None
                            ):
                                candidate_polygons[other_index] += other_move
                            cost = relief_cost(candidate_polygons)
                            if cost + 1e-8 < best_cost:
                                best_cost = cost
                                best_move = (
                                    move_index,
                                    move,
                                    other_index,
                                    other_move,
                                )
                if best_move is None:
                    break
                move_index, move, other_index, other_move = best_move
                move_targets = [(move_index, move)]
                if other_index is not None and other_move is not None:
                    move_targets.append((other_index, other_move))
                for item_index, displacement in move_targets:
                    polygon = np.asarray(
                        plan[item_index]["target_polygon_mm"],
                        dtype=np.float64,
                    )
                    polygon += displacement
                    place = np.asarray(
                        plan[item_index]["place_mm"], dtype=np.float64
                    )
                    place += displacement
                    plan[item_index]["target_polygon_mm"] = np.round(
                        polygon, 3
                    ).tolist()
                    plan[item_index]["place_mm"] = np.round(
                        place, 3
                    ).tolist()
                    overlap_relief[item_index] += displacement
            for item, displacement in zip(plan, overlap_relief):
                item["overlap_relief_mm"] = np.round(
                    displacement, 3
                ).tolist()

        maximum_target_overlap = 0.0
        target_strictly_non_overlapping = True
        target_polygons = [
            np.asarray(item["target_polygon_mm"], dtype=np.float32)
            for item in plan
        ]
        overlap_tolerance = float(self.config["overlap_tolerance_mm"])
        for first, second in itertools.combinations(target_polygons, 2):
            overlap_area = _polygon_intersection_area(first, second)
            maximum_target_overlap = max(maximum_target_overlap, overlap_area)
            if polygons_overlap(first, second, overlap_tolerance):
                target_strictly_non_overlapping = False
        # The overlap raster includes the shared seam itself (roughly half a
        # square millimetre per millimetre of seam at 2 px/mm), plus thin
        # segmentation slivers.  Accept only a bounded contour uncertainty;
        # the global gap/overlap score still rejects genuinely stacked pieces.
        accepted_overlap_area = float(
            self.config.get("maximum_accepted_pair_overlap_mm2", 80.0)
        )
        target_non_overlapping = (
            maximum_target_overlap <= accepted_overlap_area
        )
        minimum_fill = float(
            self.config.get("minimum_accepted_fill_ratio", 0.90)
        )
        maximum_score = float(
            self.config.get("maximum_accepted_geometry_score", 12.0)
        )
        solution_accepted = (
            best.fill_ratio >= minimum_fill
            and best.geometry_score <= maximum_score
            and target_non_overlapping
        )
        # Both advanced tasks define legal rectangle dimensions.  Apply a
        # small, explicit measurement tolerance at the hard gate: perspective
        # correction, edge shadows and polygon simplification can make a
        # physical 120 mm side measure about 120.6 mm.  This is not geometric
        # scaling and does not change any rigid pick/place transform; it only
        # prevents a near-perfect legal assembly from being rejected by
        # sub-millimetre camera error.
        dimension_tolerance = max(
            0.0,
            float(
                self.config.get(
                    "dimension_measurement_tolerance_mm",
                    0.0,
                )
            ),
        )
        dimension_ok = bool(
            float(self.config["min_width_mm"]) - dimension_tolerance
            <= best.long_side_mm
            <= float(self.config["max_width_mm"]) + dimension_tolerance
            and float(self.config["min_height_mm"]) - dimension_tolerance
            <= best.short_side_mm
            <= float(self.config["max_height_mm"]) + dimension_tolerance
        )
        solution_accepted = bool(solution_accepted and dimension_ok)
        # Preserve the rigid-motion result for control/verification, while
        # straightening only those displayed vertices that are within the
        # contour tolerance of an already established rectangle side.  This
        # compensates edge shadows and lens/perspective residuals; it does not
        # change pick/place coordinates, scale, chirality, or rotation.
        for item in plan:
            measured = np.asarray(
                item["target_polygon_mm"], dtype=np.float64
            )
            display = measured.copy()
            for vertex in display:
                if abs(vertex[0] - target_lower[0]) <= display_snap_tolerance:
                    vertex[0] = target_lower[0]
                elif abs(vertex[0] - target_upper[0]) <= display_snap_tolerance:
                    vertex[0] = target_upper[0]
                if abs(vertex[1] - target_lower[1]) <= display_snap_tolerance:
                    vertex[1] = target_lower[1]
                elif abs(vertex[1] - target_upper[1]) <= display_snap_tolerance:
                    vertex[1] = target_upper[1]
            item["measured_target_polygon_mm"] = np.round(
                measured, 3
            ).tolist()
            item["target_polygon_mm"] = np.round(display, 3).tolist()
        return plan, {
            "search_nodes": self.nodes,
            "exact_anchor_piece": self.observations[exact_anchor].id,
            "partial_anchor_piece": (
                self.observations[partial_anchor].id
                if partial_anchor is not None
                else None
            ),
            "fallback_anchor_piece": (
                self.observations[fallback_anchors[0]].id
                if fallback_anchors
                else None
            ),
            "fallback_anchor_pieces": [
                self.observations[index].id
                for index in fallback_anchors
            ],
            "fallback_search_timed_out": fallback_timed_out,
            "search_timed_out": self.timed_out,
            "partial_edge_search": partial_searched,
            "exact_stage_timed_out": exact_timed_out,
            "candidate_count": len(self.candidates),
            "geometry_score": round(best.geometry_score, 5),
            "assembly_gap_mm2": round(best.gap_area_mm2, 5),
            "assembly_overlap_mm2": round(best.overlap_area_mm2, 5),
            "assembly_gap_ratio": round(best.gap_ratio, 6),
            "assembly_overlap_ratio": round(best.overlap_ratio, 6),
            "texture_score": round(best.texture_score, 5),
            "card_symmetry_score": round(
                best.card_symmetry_score, 5
            ),
            "card_rank_score": round(best.card_rank_score, 5),
            "card_rank_corner_distance_mm": (
                round(best.card_rank_corner_distance_mm, 3)
                if best.card_rank_corner_distance_mm is not None
                else None
            ),
            "total_score": round(best.total_score, 5),
            "target_size_mm": np.round(size, 3).tolist(),
            "target_origin_mm": np.round(target_lower, 3).tolist(),
            "target_orientation": target_orientation,
            "fill_ratio": round(best.fill_ratio, 6),
            "target_dimensions_legal": dimension_ok,
            "dimension_measurement_tolerance_mm": round(
                dimension_tolerance, 3
            ),
            "solution_accepted": solution_accepted,
            "motion_model": "rotation_and_translation_only",
            "mirror_allowed": False,
            "solution_quality": (
                "high"
                if solution_accepted
                and best.fill_ratio >= 0.94
                and best.geometry_score <= 8.0
                else "usable"
                if solution_accepted
                else "rejected"
            ),
            "placement_clearance_mm": clearance,
            "boundary_snap_tolerance_mm": display_snap_tolerance,
            "motion_boundary_snap_tolerance_mm": snap_tolerance,
            "target_outline_denoised": True,
            "target_non_overlapping": target_non_overlapping,
            "target_strictly_non_overlapping": (
                target_strictly_non_overlapping
            ),
            "maximum_accepted_pair_overlap_mm2": accepted_overlap_area,
            "maximum_target_overlap_mm2": round(maximum_target_overlap, 5),
        }


def solve_unknown(
    observations: list[PieceObservation],
    unknown_cfg: dict[str, Any],
    rectified_image: np.ndarray | None,
    pixels_per_mm: float,
    use_texture: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Every supplied piece belongs to the same finished rectangle, so their
    # measured area must be compatible with the smallest legal rectangle.
    # This inexpensive check is intentionally performed before the search.
    # Apart from saving several seconds on bad masks, it prevents thin UI
    # annotation strokes (for example, a photo of our already annotated
    # monitor output) from being reported as a puzzle-solver failure.
    total_piece_area = float(sum(item.area_mm2 for item in observations))
    minimum_legal_area = float(
        unknown_cfg["min_width_mm"]
        * unknown_cfg["min_height_mm"]
    )
    minimum_input_ratio = float(
        unknown_cfg.get("minimum_input_total_area_ratio", 0.50)
    )
    minimum_input_area = minimum_legal_area * minimum_input_ratio
    if total_piece_area < minimum_input_area:
        raise SolveError(
            "Input contours are not the physical puzzle pieces: detected "
            f"total area {total_piece_area:.0f} mm^2 is below the "
            f"{minimum_input_area:.0f} mm^2 plausibility limit "
            f"(legal rectangle minimum {minimum_legal_area:.0f} mm^2). "
            "A photo/screenshot containing the upper-computer's coloured "
            "overlay was probably supplied, so annotation lines were "
            "segmented as pieces. Use the raw camera image (/api/raw.jpg) "
            "without boxes, labels or centroids."
        )
    return UnknownPuzzleSolver(
        observations,
        unknown_cfg,
        rectified_image=rectified_image,
        pixels_per_mm=pixels_per_mm,
        use_texture=use_texture,
    ).solve()
