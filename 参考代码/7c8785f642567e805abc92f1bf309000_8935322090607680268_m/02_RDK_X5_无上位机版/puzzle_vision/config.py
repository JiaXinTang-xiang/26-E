from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


def _fixed_piece_geometry() -> list[dict[str, Any]]:
    """Return a reproducible four-piece layout matching the dimensions in Fig. 2.

    Coordinates are millimetres, relative to the 100 mm x 60 mm target rectangle.
    Q is placed 20 mm from B at 45 degrees.  R lies on Q-D and is 30 mm from D.
    If the team's physical cut uses a different Q direction, edit only these
    template vertices or use a taught template file.
    """

    a = [0.0, 20.0]
    b = [20.0, 0.0]
    c = [0.0, 30.0]
    d = [100.0, 60.0]
    q = [20.0 + 20.0 / math.sqrt(2.0), 20.0 / math.sqrt(2.0)]
    qd = math.dist(q, d)
    r = [
        d[0] + (q[0] - d[0]) * 30.0 / qd,
        d[1] + (q[1] - d[1]) * 30.0 / qd,
    ]

    return [
        {"id": "fixed_1", "vertices_mm": [[0.0, 0.0], b, q, a]},
        {"id": "fixed_2", "vertices_mm": [b, [100.0, 0.0], d, q]},
        {"id": "fixed_3", "vertices_mm": [a, q, r, c]},
        {"id": "fixed_4", "vertices_mm": [c, r, d, [0.0, 60.0]]},
    ]


DEFAULT_CONFIG: dict[str, Any] = {
    "paper": {
        "width_mm": 210.0,
        "height_mm": 297.0,
        "divider_y_mm": 148.5,
        "divider_search_half_range_mm": 12.0,
        "cached_divider_max_offset_mm": 4.0,
        "divider_max_width_mm": 5.0,
        "divider_min_contrast_lab": 12.0,
        "divider_broad_search_half_range_mm": 38.0,
        "divider_broad_min_contrast_lab": 16.2,
        "divider_horizontal_support_quantile": 0.34,
        "divider_min_continuous_width_ratio": 0.55,
        "divider_max_horizontal_gap_ratio": 0.035,
        "dark_paper_lightness_threshold_lab": 92.0,
        "color_bgr": None,
        "color_hint_weight": 0.8,
        "detection_max_dimension": 960,
        "fallback_detection_max_dimension": 1600,
        "minimum_image_area_ratio": 0.045,
        "maximum_image_area_ratio": 0.78,
        "minimum_frame_margin_ratio": 0.008,
        "maximum_colour_aspect_error_ratio": 0.14,
        "edge_refine_enabled": True,
        "edge_refine_search_ratio": 0.055,
        "edge_refine_max_search_px": 72,
        "edge_refine_probe_ratio": 0.0045,
        "edge_refine_min_lab_step": 5.0,
        "edge_refine_groups": 9,
        "edge_refine_max_angle_deg": 4.5,
        "edge_refine_max_aspect_error_ratio": 0.20,
        "pixels_per_mm": 4.0,
        "corner_override_px": None,
        "rotation_quadrants": 0,
    },
    "camera": {
        "source": "auto",
        "width": 1920,
        "height": 1080,
        "fps": 30,
        "warmup_frames": 8,
    },
    "segmentation": {
        "lab_distance_threshold": 24.0,
        "lab_chroma_threshold": 10.0,
        "lab_lightness_threshold": 16.0,
        "piece_color_bgr": None,
        "piece_color_tolerance_lab": 32.0,
        "piece_color_tolerance_chroma": 12.0,
        "background_difference_threshold": 20.0,
        "min_area_mm2": 100.0,
        "max_area_mm2": 6500.0,
        "max_pieces": 4,
        "required_pieces": 4,
        "paper_margin_mm": 3.0,
        "divider_margin_mm": 4.0,
        "reject_roi_border_components": True,
        "roi_border_component_min_contact_mm": 8.0,
        "roi_border_piece_colour_tolerance_lab": 18.0,
        "roi_border_no_reference_contact_mm": 18.0,
        "morph_open_mm": 0.6,
        "morph_close_mm": 2.2,
        "polygon_epsilon_mm": 1.0,
        "max_polygon_epsilon_mm": 5.0,
        "minimum_detected_edge_mm": 8.0,
        "minimum_corner_turn_deg": 30.0,
        "collinear_short_edge_mm": 18.0,
        "assume_convex_pieces": False,
        "expected_min_vertices": 3,
        "expected_max_vertices": 5,
        "advanced_expected_max_vertices": 5,
        "card_expected_max_vertices": 8,
        "card_minimum_detected_edge_mm": 3.0,
        "card_minimum_corner_turn_deg": 12.0,
        "card_collinear_short_edge_mm": 6.0,
        "card_expected_total_area_mm2": 5016.0,
        "card_expected_total_area_min_ratio": 0.68,
        "card_expected_total_area_max_ratio": 1.25,
    },
    "fixed": {
        "target_origin_mm": [55.0, 192.75],
        "target_size_mm": [100.0, 60.0],
        "pieces": _fixed_piece_geometry(),
        "max_match_error_mm": 25.0,
    },
    "unknown": {
        "use_taught_layout": True,
        "taught_layout_path": "taught_layout.json",
        "target_orientation": "landscape",
        "card_aspect_ratio": 88.0 / 57.0,
        "card_expected_fill_ratios": [0.98, 0.94, 0.90],
        "card_boundary_tolerance_mm": 4.0,
        "card_corner_option_limit": 7,
        "card_search_seconds": 2.2,
        "card_exact_search_seconds": 0.55,
        "card_fallback_search_seconds": 0.7,
        "card_max_pair_options_exact": 36,
        "card_max_pair_options_partial": 36,
        "card_maximum_geometry_score": 16.0,
        "card_maximum_overlap_ratio": 0.025,
        "card_minimum_long_side_mm": 78.0,
        "card_maximum_long_side_mm": 100.0,
        "card_minimum_short_side_mm": 45.0,
        "card_maximum_short_side_mm": 68.0,
        "card_rounded_chord_min_mm": 2.0,
        "card_rounded_chord_max_mm": 8.0,
        "card_rounded_corner_angle_error_deg": 15.0,
        "card_maximum_aspect_error": 0.22,
        "card_aspect_score_weight": 65.0,
        "card_mark_min_area_mm2": 0.35,
        "card_mark_max_area_mm2": 140.0,
        "card_black_ink_max_gray": 155.0,
        "card_black_ink_contrast_gray": 48.0,
        "card_rank_corner_distance_mm": 10.5,
        "card_rank_corner_tolerance_mm": 5.0,
        "card_rank_score_weight": 0.2,
        # Card-only artwork scores.  Keep these separate from the generic
        # unknown-white texture weight so changing playing-card parameters in
        # the upper-computer UI cannot slow down or perturb white fragments.
        "card_texture_weight": 0.30,
        "card_pair_pattern_weight": 0.90,
        "card_pattern_profile_weight": 1.25,
        "card_pattern_transition_weight": 0.65,
        "card_pattern_one_sided_penalty": 7.0,
        "card_pattern_match_bonus": 2.5,
        "card_pattern_shift_samples": 2,
        "card_pattern_min_informative_ratio": 0.035,
        "card_texture_neutral_score": 50.0,
        "card_minimum_pattern_candidates": 3,
        "card_pattern_lightness_drop_lab": 24.0,
        "card_pattern_chroma_delta_lab": 16.0,
        "card_symmetry_score_weight": 0.65,
        "card_grouped_rank_distance_mm": 12.0,
        "card_corner_patch_min_confidence": 0.35,
        "card_component_rank_min_confidence": 0.55,
        "card_maximum_pair_overlap_mm2": 0.5,
        "card_overlap_tolerance_mm": 0.15,
        "card_minimum_partial_remainder_mm": 3.0,
        "card_glare_overexposed_ratio": 0.55,
        "card_minimum_pattern_pixel_ratio": 0.0015,
        "card_anchor_min_rank_confidence": 0.60,
        "card_anchored_search_seconds": 0.75,
        "card_anchored_exact_seconds": 0.25,
        "card_anchored_max_attempts": 2,
        "card_anchored_total_seconds": 1.2,
        "card_maximum_extent_error_mm": 12.0,
        "card_extent_score_weight": 2.0,
        "card_overlap_relief_step_mm": 0.10,
        "card_overlap_relief_max_shift_mm": 4.0,
        "card_overlap_relief_direct_iterations": 4,
        "card_overlap_relief_binary_steps": 6,
        "card_overlap_relief_iterations": 4,
        "card_overlap_relief_outside_weight": 0.5,
        "target_zone_mm": [0.0, 148.5, 210.0, 297.0],
        "min_width_mm": 90.0,
        "max_width_mm": 120.0,
        "min_height_mm": 50.0,
        "max_height_mm": 90.0,
        # Contours are measured after A4 perspective correction.  Residual
        # lens distortion, edge shadows and polygon simplification can move a
        # legal target side by well under 1% (for example 120.62 mm instead of
        # 120 mm).  Keep the formal dimensions as the reported limits, but
        # allow this bounded camera-measurement uncertainty at the hard
        # actuator-export gate.
        "dimension_measurement_tolerance_mm": 2.0,
        # All detected pieces together form the whole target rectangle.
        # Reject implausibly tiny masks before launching the combinatorial
        # search; these are normally UI annotation strokes or paper texture.
        "minimum_input_total_area_ratio": 0.50,
        "edge_length_tolerance_mm": 4.5,
        "edge_length_relative_tolerance": 0.12,
        "edge_overlap_order_weight": 0.08,
        "edge_length_error_order_weight": 0.50,
        "autonomous_compactness_ordering": True,
        "autonomous_compactness_order_weight": 0.022,
        "autonomous_final_box_order_weight": 0.045,
        "autonomous_dimension_order_weight": 3.0,
        "sparse_edge_length_tolerance_mm": 5.5,
        "sparse_edge_length_relative_tolerance": 0.14,
        "minimum_solver_edge_mm": 8.0,
        "minimum_partial_remainder_mm": 6.0,
        "partial_alignment_fractions": [],
        "guided_minimum_partial_remainder_mm": 6.0,
        "overlap_tolerance_mm": 3.0,
        "search_overlap_tolerance_mm": 8.0,
        "maximum_accepted_pair_overlap_mm2": 100.0,
        "search_dimension_slack_mm": 8.0,
        "search_minimum_fill_ratio": 0.72,
        "max_search_nodes": 18000,
        "max_pair_options_exact": 64,
        "max_pair_options_partial": 64,
        "max_search_seconds": 1.2,
        "fallback_search_seconds": 1.0,
        "exact_search_node_floor": 40,
        "partial_search_node_floor": 80,
        "fallback_search_node_floor": 140,
        "guided_search_seconds": 2.5,
        "guided_fallback_search_seconds": 2.5,
            "pose_hint_weight": 1.8,
            "pose_hint_branch_limit": 36,
        "exact_search_seconds": 0.35,
        "candidate_overlap_pixels_per_mm": 2.0,
        "early_stop_geometry_score": 8.0,
        "early_stop_overlap_ratio": 0.006,
        "early_stop_fill_ratio": 0.94,
        "maximum_candidate_fill_ratio": 1.08,
        "max_final_candidates": 80,
        "exact_search_accept_score": 8.0,
        "partial_reuse_exact_anchor_score": 16.0,
        "texture_weight": 0.30,
        "seam_sample_offset_mm": 1.5,
        "placement_clearance_mm": 0.0,
        "boundary_snap_tolerance_mm": 6.0,
        "motion_boundary_snap_tolerance_mm": 0.0,
        "minimum_accepted_fill_ratio": 0.90,
        "maximum_accepted_geometry_score": 16.0,
    },
    # Upper-computer overrides for ordinary white pieces.  These values are
    # overlaid only in unknown-white mode by vision_server.py; keeping them in
    # a separate section prevents lighting/area tuning from changing the
    # fixed-template or playing-card pipelines.
    "white_tuning": {},
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            config = deep_update(config, json.load(handle))
    return config


def save_default_config(path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, ensure_ascii=False, indent=2)
