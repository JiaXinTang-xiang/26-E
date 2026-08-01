from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import DetectionError, PaperView, detect_pieces, rectify_paper
from .solver import (
    SolveError,
    solve_card,
    solve_fixed,
    solve_taught,
    solve_unknown,
)


class PuzzleVisionPipeline:
    MODES = ("fixed", "unknown-white", "unknown-pattern")

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._cached_paper_corners: np.ndarray | None = None
        self.taught_layout: dict[str, Any] | None = None
        taught_path = config.get("unknown", {}).get("taught_layout_path")
        if taught_path:
            path = Path(taught_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[1] / path
            try:
                with path.open("r", encoding="utf-8") as handle:
                    self.taught_layout = json.load(handle)
            except (OSError, ValueError, TypeError):
                self.taught_layout = None

    def rectify(self, frame: np.ndarray) -> PaperView:
        if self._cached_paper_corners is not None:
            try:
                cached_paper = rectify_paper(
                    frame,
                    self.config["paper"],
                    cached_corners=self._cached_paper_corners,
                )
                expected_divider = float(
                    self.config["paper"]["divider_y_mm"]
                )
                maximum_cached_offset = float(
                    self.config["paper"].get(
                        "cached_divider_max_offset_mm", 4.0
                    )
                )
                if (
                    abs(cached_paper.divider_y_mm - expected_divider)
                    <= maximum_cached_offset
                ):
                    return cached_paper
                # A stale homography may still contain the divider inside the
                # broad detection window while shifting an A4 edge onto the
                # table.  That table strip then looks like a false piece.
                # Re-detect the outer A4 whenever the divider no longer maps
                # close enough to its known paper coordinate.
                self._cached_paper_corners = None
            except DetectionError:
                # The A4 or camera moved enough that the cached homography no
                # longer places the divider near the expected middle.  Drop
                # the cache and perform one complete detection.
                self._cached_paper_corners = None
        paper = rectify_paper(frame, self.config["paper"])
        self._cached_paper_corners = paper.corners_px.copy()
        return paper

    def invalidate_paper_cache(self) -> None:
        """Force the next frame to redetect the A4 outer boundary."""

        self._cached_paper_corners = None

    def analyze(
        self,
        frame: np.ndarray,
        mode: str,
        background_rectified: np.ndarray | None = None,
        source_region: str = "upper",
        allow_unsolved: bool = False,
        progress_callback: Any | None = None,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        if mode not in self.MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        started = time.perf_counter()
        paper = self.rectify(frame)
        rectified_at = time.perf_counter()
        segmentation_config = deepcopy(self.config["segmentation"])
        # The four Figure-2 pieces are convex.  Advanced mode must retain
        # concave three-to-five-edge pieces supplied by the test site.
        segmentation_config["assume_convex_pieces"] = mode == "fixed"
        # The formal field pieces have at most five structural sides.
        # Unknown-white keeps one tolerance vertex for a wrinkled hand cut,
        # while card mode below deliberately removes that tolerance because
        # printed marks must not become false cut edges.
        if mode != "fixed":
            segmentation_config["expected_max_vertices"] = int(
                segmentation_config.get(
                    "advanced_expected_max_vertices", 5
                )
            )
        if mode == "unknown-pattern":
            # A rounded original card corner is represented by one short
            # chord, not by several curve-following segments.  Therefore it
            # still consumes only one of the task's at-most-five structural
            # edges.  Enforcing this prior suppresses false vertices caused by
            # ranks/suits, glare, lamination and a rough cut boundary.
            segmentation_config["expected_max_vertices"] = int(
                segmentation_config.get(
                    "card_expected_max_vertices", 5
                )
            )
            segmentation_config["minimum_detected_edge_mm"] = float(
                segmentation_config.get(
                    "card_minimum_detected_edge_mm", 3.0
                )
            )
            segmentation_config["minimum_corner_turn_deg"] = float(
                segmentation_config.get(
                    "card_minimum_corner_turn_deg", 12.0
                )
            )
            segmentation_config["collinear_short_edge_mm"] = float(
                segmentation_config.get(
                    "card_collinear_short_edge_mm", 6.0
                )
            )
            segmentation_config["expected_total_area_mm2"] = float(
                segmentation_config.get(
                    "card_expected_total_area_mm2", 57.0 * 88.0
                )
            )
            segmentation_config["expected_total_area_min_ratio"] = float(
                segmentation_config.get(
                    "card_expected_total_area_min_ratio", 0.68
                )
            )
            segmentation_config["expected_total_area_max_ratio"] = float(
                segmentation_config.get(
                    "card_expected_total_area_max_ratio", 1.25
                )
            )
        # Figure 2 always has four pieces.  The field-provided advanced task
        # explicitly allows one to four pieces.
        segmentation_config["required_pieces"] = 4 if mode == "fixed" else 0
        observations, mask, detected_region, segmentation_mode = detect_pieces(
            paper,
            self.config["paper"],
            segmentation_config,
            background_rectified,
            source_region,
        )
        detected_at = time.perf_counter()
        if progress_callback is not None:
            # Publish a lightweight geometry-only frame before the potentially
            # expensive arrangement search.  The callback is deliberately
            # best-effort: a UI/socket failure must never abort vision.
            progress_result: dict[str, Any] = {
                "ok": True,
                "motion_ready": False,
                "recognition_candidate_ready": False,
                "mode": mode,
                "source_region": detected_region,
                "destination_region": (
                    "lower" if detected_region == "upper" else "upper"
                ),
                "segmentation_mode": segmentation_mode,
                "paper": {
                    "size_mm": [paper.width_mm, paper.height_mm],
                    "corners_in_camera_px": np.round(
                        paper.corners_px, 2
                    ).tolist(),
                    "homography_camera_to_paper_px": np.round(
                        paper.homography, 8
                    ).tolist(),
                    "pixels_per_mm": paper.pixels_per_mm,
                    "divider": {
                        "detected_y_mm": round(paper.divider_y_mm, 3),
                        "width_mm": round(paper.divider_width_mm, 3),
                        "contrast_lab": round(
                            paper.divider_contrast_lab, 3
                        ),
                    },
                },
                "pieces": [
                    observation.to_dict() for observation in observations
                ],
                "plan": [],
                "solver": {
                    "target_size_mm": [
                        float(
                            self.config["unknown"].get(
                                "default_target_width_mm", 100.0
                            )
                        ),
                        float(
                            self.config["unknown"].get(
                                "default_target_height_mm", 60.0
                            )
                        ),
                    ],
                    "solution_accepted": False,
                    "solution_quality": "detecting",
                },
            }
            try:
                progress_callback(progress_result)
            except Exception:
                pass
        try:
            if mode == "fixed":
                fixed_config = deepcopy(self.config["fixed"])
                if detected_region == "lower":
                    vertices = [
                        np.asarray(piece["vertices_mm"], dtype=np.float64)
                        for piece in fixed_config["pieces"]
                    ]
                    points = np.vstack(vertices)
                    target_height = float(
                        np.max(points[:, 1]) - np.min(points[:, 1])
                    )
                    fixed_config["target_origin_mm"][1] = (
                        paper.height_mm
                        - float(fixed_config["target_origin_mm"][1])
                        - target_height
                    )
                plan, solver_info = solve_fixed(observations, fixed_config)
            else:
                unknown_config = deepcopy(self.config["unknown"])
                # Keep the two advanced paths independent.  Ordinary white
                # fragments use the original landscape target convention;
                # only the patterned/playing-card path is forced portrait.
                # solve_card() also enforces portrait internally, but setting
                # it here keeps its generic fallback consistent.
                unknown_config["target_orientation"] = (
                    "portrait"
                    if mode == "unknown-pattern"
                    else "landscape"
                )
                if detected_region == "lower":
                    unknown_config["target_zone_mm"] = [
                        0.0,
                        0.0,
                        paper.width_mm,
                        paper.divider_y_mm,
                    ]
                else:
                    unknown_config["target_zone_mm"] = [
                        0.0,
                        paper.divider_y_mm,
                        paper.width_mm,
                        paper.height_mm,
                    ]
                taught_error: str | None = None
                card_error: str | None = None
                card_attempt_info: dict[str, Any] | None = None
                card_solved = False
                if (
                    mode == "unknown-pattern"
                    and 2 <= len(observations) <= 4
                ):
                    try:
                        plan, solver_info = solve_card(
                            observations,
                            unknown_config,
                            paper.image,
                            paper.pixels_per_mm,
                        )
                        card_attempt_info = solver_info
                        if not solver_info.get(
                            "solution_accepted", False
                        ):
                            # Keep the rejected card result for diagnostics,
                            # but do not launch the unrelated generic fragment
                            # solver.  Pattern mode is the playing-card path;
                            # a second full solver pass only delays the UI and
                            # can propose a non-card-shaped rectangle.
                            solver_info.setdefault(
                                "solve_error",
                                "Rounded-card search did not form a "
                                "sufficiently filled rectangle",
                            )
                        card_solved = True
                    except SolveError as exc:
                        card_error = str(exc)
                        plan = []
                        solver_info = {
                            "solution_accepted": False,
                            "solution_quality": "rejected",
                            "solve_error": card_error,
                            "motion_model": (
                                "rotation_and_translation_only"
                            ),
                            "mirror_allowed": False,
                            "piece_reflection_used": False,
                            "target_non_overlapping": False,
                        }
                        card_solved = True
                if (
                    not card_solved
                    and bool(unknown_config.get("use_taught_layout", False))
                    and self.taught_layout is not None
                    and int(self.taught_layout.get("piece_count", 0))
                    == len(observations)
                ):
                    try:
                        plan, solver_info = solve_taught(
                            observations,
                            self.taught_layout,
                            unknown_config,
                        )
                        if not solver_info.get(
                            "solution_accepted", False
                        ):
                            raise SolveError(
                                "Taught layout does not form a filled, "
                                "non-overlapping legal rectangle"
                            )
                    except SolveError as exc:
                        taught_error = str(exc)
                        plan, solver_info = solve_unknown(
                            observations,
                            unknown_config,
                            paper.image,
                            paper.pixels_per_mm,
                            use_texture=mode == "unknown-pattern",
                        )
                elif not card_solved:
                    plan, solver_info = solve_unknown(
                        observations,
                        unknown_config,
                        paper.image,
                        paper.pixels_per_mm,
                        use_texture=mode == "unknown-pattern",
                    )
                if taught_error is not None:
                    solver_info["taught_layout_fallback_reason"] = taught_error
                if card_error is not None:
                    solver_info["card_solver_fallback_reason"] = card_error
                if (
                    card_attempt_info is not None
                    and solver_info is not card_attempt_info
                ):
                    solver_info["card_solver_attempt"] = card_attempt_info
                    card_recognition = card_attempt_info.get(
                        "card_recognition"
                    )
                    if isinstance(card_recognition, dict):
                        solver_info["card_recognition"] = card_recognition
        except SolveError as exc:
            if not allow_unsolved:
                raise
            plan = []
            solver_info = {
                "solution_accepted": False,
                "solution_quality": "rejected",
                "solve_error": str(exc),
                "motion_model": "rotation_and_translation_only",
                "mirror_allowed": False,
                "piece_reflection_used": False,
                "target_non_overlapping": False,
            }
        solved_at = time.perf_counter()
        result: dict[str, Any] = {
            "ok": True,
            "motion_ready": bool(solver_info.get("solution_accepted", True)),
            "mode": mode,
            "source_region": detected_region,
            "destination_region": "lower" if detected_region == "upper" else "upper",
            "segmentation_mode": segmentation_mode,
            "coordinate_frame": {
                "origin": "A4 top-left",
                "units": "mm",
                "x_positive": "right",
                "y_positive": "down",
                "rotation_positive": "clockwise",
                "piece_motion": "rotation_and_translation_only",
                "mirror_allowed": False,
            },
            "paper": {
                "size_mm": [paper.width_mm, paper.height_mm],
                "corners_in_camera_px": np.round(paper.corners_px, 2).tolist(),
                "homography_camera_to_paper_px": np.round(
                    paper.homography, 8
                ).tolist(),
                "pixels_per_mm": paper.pixels_per_mm,
                "divider": {
                    "detected_y_mm": round(paper.divider_y_mm, 3),
                    "width_mm": round(paper.divider_width_mm, 3),
                    "contrast_lab": round(paper.divider_contrast_lab, 3),
                },
            },
            "pieces": [observation.to_dict() for observation in observations],
            "plan": plan,
            "solver": solver_info,
            "timing_ms": {
                "rectify": round((rectified_at - started) * 1000.0, 2),
                "detect": round((detected_at - rectified_at) * 1000.0, 2),
                "solve": round((solved_at - detected_at) * 1000.0, 2),
                "total": round((solved_at - started) * 1000.0, 2),
            },
        }
        debug = self.draw_debug(paper, observations, plan, mode)
        card_recognition = solver_info.get("card_recognition")
        if mode == "unknown-pattern" and isinstance(
            card_recognition, dict
        ):
            result["card_recognition"] = card_recognition
            if card_recognition.get("rank_detected", False):
                cv2.putText(
                    debug,
                    (
                        f"card rank={card_recognition.get('rank')} "
                        f"confidence={float(card_recognition.get('rank_confidence', 0.0)):.2f}"
                    ),
                    (16, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        return result, debug, mask, paper.image

    def draw_debug(
        self,
        paper: PaperView,
        observations: list[Any],
        plan: list[dict[str, Any]],
        mode: str,
    ) -> np.ndarray:
        debug = paper.image.copy()
        ppm = float(self.config["paper"]["pixels_per_mm"])
        divider = int(round(paper.divider_y_mm * ppm))
        cv2.line(debug, (0, divider), (debug.shape[1] - 1, divider), (0, 0, 255), 2)
        palette = [
            (0, 255, 0),
            (255, 180, 0),
            (255, 0, 255),
            (0, 200, 255),
        ]
        by_id = {item["piece_id"]: item for item in plan}
        for index, observation in enumerate(observations):
            colour = palette[index % len(palette)]
            polygon_px = np.rint(observation.polygon_mm * ppm).astype(np.int32)
            cv2.polylines(debug, [polygon_px], True, colour, 3, cv2.LINE_AA)
            pickup = tuple(np.rint(observation.pickup_mm * ppm).astype(int))
            cv2.drawMarker(
                debug, pickup, colour, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA
            )
            cv2.putText(
                debug,
                observation.id,
                (pickup[0] + 7, pickup[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                colour,
                2,
                cv2.LINE_AA,
            )
            target = by_id.get(observation.id)
            if target:
                target_px = np.rint(
                    np.asarray(target["target_polygon_mm"]) * ppm
                ).astype(np.int32)
                cv2.polylines(
                    debug, [target_px], True, colour, 3, cv2.LINE_AA
                )
                place = tuple(
                    np.rint(np.asarray(target["place_mm"]) * ppm).astype(int)
                )
                cv2.circle(debug, place, 6, colour, -1, cv2.LINE_AA)
                cv2.putText(
                    debug,
                    f"{target['rotate_deg']:+.1f}deg",
                    (place[0] + 8, place[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                    cv2.LINE_AA,
                )
        cv2.putText(
            debug,
            mode,
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return debug

    @staticmethod
    def draw_camera_overlay(
        frame: np.ndarray,
        result: dict[str, Any],
    ) -> np.ndarray:
        """Project millimetre-space detections and plans onto the raw camera view."""

        overlay = frame.copy()
        paper = result["paper"]
        ppm = float(paper["pixels_per_mm"])
        camera_to_paper = np.asarray(
            paper["homography_camera_to_paper_px"], dtype=np.float64
        )
        paper_to_camera = np.linalg.inv(camera_to_paper)
        line_width = max(2, int(round(max(frame.shape[:2]) / 600.0)))

        def project(points_mm: Any) -> np.ndarray:
            points = np.asarray(points_mm, dtype=np.float32).reshape(-1, 1, 2)
            points *= ppm
            camera = cv2.perspectiveTransform(
                points, paper_to_camera.astype(np.float64)
            )
            return np.rint(camera.reshape(-1, 2)).astype(np.int32)

        corners = np.rint(
            np.asarray(paper["corners_in_camera_px"], dtype=np.float64)
        ).astype(np.int32)
        cv2.polylines(
            overlay, [corners], True, (40, 240, 40), line_width + 1, cv2.LINE_AA
        )

        divider_y = float(paper["divider"]["detected_y_mm"])
        divider = project([[0.0, divider_y], [float(paper["size_mm"][0]), divider_y]])
        cv2.line(
            overlay,
            tuple(divider[0]),
            tuple(divider[1]),
            (0, 190, 255),
            line_width + 1,
            cv2.LINE_AA,
        )

        target_layer = overlay.copy()
        plan_by_id = {item["piece_id"]: item for item in result["plan"]}
        motion_ready = bool(result.get("motion_ready", False))

        target_size = result.get("solver", {}).get("target_size_mm", [100.0, 60.0])
        target_origin = result.get("solver", {}).get("target_origin_mm")
        if motion_ready and target_origin is not None:
            rectangle_lower = np.asarray(target_origin, dtype=np.float64)
            rectangle_upper = rectangle_lower + np.asarray(
                target_size, dtype=np.float64
            )
        elif motion_ready and result["plan"]:
            planned_points = np.vstack(
                [
                    np.asarray(item["target_polygon_mm"], dtype=np.float64)
                    for item in result["plan"]
                ]
            )
            rectangle_lower = np.min(planned_points, axis=0)
            rectangle_upper = np.max(planned_points, axis=0)
        else:
            destination_upper = result["destination_region"] == "upper"
            zone_top = 0.0 if destination_upper else divider_y
            zone_bottom = divider_y if destination_upper else float(paper["size_mm"][1])
            centre = np.asarray(
                [
                    float(paper["size_mm"][0]) * 0.5,
                    (zone_top + zone_bottom) * 0.5,
                ]
            )
            half_size = np.asarray(target_size, dtype=np.float64) * 0.5
            rectangle_lower = centre - half_size
            rectangle_upper = centre + half_size
        target_rectangle = project(
            [
                rectangle_lower,
                [rectangle_upper[0], rectangle_lower[1]],
                rectangle_upper,
                [rectangle_lower[0], rectangle_upper[1]],
            ]
        )
        if motion_ready:
            # The target is one rectangle in the A4 coordinate frame.  Draw
            # that complete target first, then overlay the four measured seam
            # outlines.  Segmentation noise can no longer look like a missing
            # outer corner or an unsynchronised right-hand gap.
            cv2.fillPoly(
                target_layer,
                [target_rectangle],
                (180, 40, 210),
                cv2.LINE_AA,
            )
            overlay = cv2.addWeighted(target_layer, 0.24, overlay, 0.76, 0.0)
        rectangle_colour = (230, 60, 230) if motion_ready else (30, 30, 255)
        cv2.polylines(
            overlay,
            [target_rectangle],
            True,
            rectangle_colour,
            line_width + 1,
            cv2.LINE_AA,
        )
        rectangle_label = (
            f"TARGET RECT {float(target_size[0]):.1f}x"
            f"{float(target_size[1]):.1f} mm "
            f"{'READY' if motion_ready else 'NOT READY'}"
        )
        label_anchor = target_rectangle[int(np.argmin(target_rectangle[:, 1]))]
        cv2.putText(
            overlay,
            rectangle_label,
            (int(label_anchor[0]), max(18, int(label_anchor[1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            rectangle_colour,
            line_width,
            cv2.LINE_AA,
        )

        pickup_colour = (40, 255, 255)
        piece_colours = [
            (255, 190, 35),
            (70, 230, 90),
            (220, 80, 255),
            (20, 150, 255),
        ]
        for piece_index, piece in enumerate(result["pieces"]):
            piece_id = piece["piece_id"]
            piece_colour = piece_colours[piece_index % len(piece_colours)]
            polygon = project(piece["polygon_mm"])
            centroid = project([piece["centroid_mm"]])[0]
            pickup = project([piece["pickup_mm"]])[0]
            cv2.polylines(
                overlay,
                [polygon],
                True,
                piece_colour,
                line_width,
                cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                tuple(centroid),
                6 + line_width,
                piece_colour,
                -1,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                overlay,
                tuple(pickup),
                pickup_colour,
                cv2.MARKER_CROSS,
                16 + 2 * line_width,
                line_width,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                f"{piece_id} C",
                (int(centroid[0]) + 8, int(centroid[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                piece_colour,
                line_width,
                cv2.LINE_AA,
            )

            target = plan_by_id.get(piece_id) if motion_ready else None
            if not target:
                continue
            target_polygon = project(target["target_polygon_mm"])
            place = project([target["place_mm"]])[0]
            piece_layer = overlay.copy()
            cv2.fillPoly(
                piece_layer,
                [target_polygon],
                piece_colour,
                cv2.LINE_AA,
            )
            overlay = cv2.addWeighted(
                piece_layer, 0.38, overlay, 0.62, 0.0
            )
            cv2.polylines(
                overlay,
                [target_polygon],
                True,
                piece_colour,
                line_width,
                cv2.LINE_AA,
            )
            cv2.drawMarker(
                overlay,
                tuple(place),
                piece_colour,
                cv2.MARKER_TILTED_CROSS,
                18 + 2 * line_width,
                line_width,
                cv2.LINE_AA,
            )
            angle = np.deg2rad(float(target["rotate_deg"]))
            direction_mm = np.asarray(target["place_mm"], dtype=np.float64) + np.array(
                [15.0 * np.cos(angle), 15.0 * np.sin(angle)]
            )
            arrow_end = project([direction_mm])[0]
            cv2.arrowedLine(
                overlay,
                tuple(place),
                tuple(arrow_end),
                piece_colour,
                line_width,
                cv2.LINE_AA,
                tipLength=0.25,
            )
            cv2.putText(
                overlay,
                f"{piece_id} {float(target['rotate_deg']):+.1f} deg",
                (int(place[0]) + 8, int(place[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                piece_colour,
                line_width,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            (
                f"A4 210x297 mm | divider {divider_y:.1f} mm | "
                f"{result['source_region']} -> {result['destination_region']} | "
                f"{'MOTION READY' if motion_ready else 'PLAN REJECTED'}"
            ),
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            line_width + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "green=A4  orange=divider  per-piece colour=source+target",
            (18, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
        return overlay
