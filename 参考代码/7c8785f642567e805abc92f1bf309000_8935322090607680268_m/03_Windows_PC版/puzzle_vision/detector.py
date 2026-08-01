from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    safe_interior_point,
)


class DetectionError(RuntimeError):
    pass


@dataclass
class PaperView:
    image: np.ndarray
    homography: np.ndarray
    corners_px: np.ndarray
    pixels_per_mm: float
    width_mm: float
    height_mm: float
    divider_y_mm: float
    divider_width_mm: float
    divider_contrast_lab: float

    def pixels_to_mm(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) / self.pixels_per_mm


@dataclass
class PieceObservation:
    id: str
    polygon_mm: np.ndarray
    contour_px: np.ndarray
    centroid_mm: np.ndarray
    pickup_mm: np.ndarray
    area_mm2: float
    perimeter_mm: float
    edge_lengths_mm: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "piece_id": self.id,
            "polygon_mm": np.round(self.polygon_mm, 3).tolist(),
            "centroid_mm": np.round(self.centroid_mm, 3).tolist(),
            "pickup_mm": np.round(self.pickup_mm, 3).tolist(),
            "area_mm2": round(self.area_mm2, 3),
            "perimeter_mm": round(self.perimeter_mm, 3),
            "edge_lengths_mm": np.round(self.edge_lengths_mm, 3).tolist(),
            "vertices": int(len(self.polygon_mm)),
        }


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    result[0] = pts[np.argmin(sums)]
    result[2] = pts[np.argmax(sums)]
    result[1] = pts[np.argmin(diffs)]
    result[3] = pts[np.argmax(diffs)]
    return result


def _rotate_corner_order(corners: np.ndarray, quadrants: int) -> np.ndarray:
    q = quadrants % 4
    if q == 0:
        return corners
    return np.roll(corners, -q, axis=0)


def _quad_boundary_colour_contrast(
    lab_image: np.ndarray,
    quad: np.ndarray,
) -> float:
    """Measure the local Lab colour step across a candidate sheet boundary."""

    height, width = lab_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    radius = max(2, int(round(min(height, width) * 0.008)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    inside = cv2.subtract(mask, cv2.erode(mask, kernel))
    outside = cv2.subtract(cv2.dilate(mask, kernel), mask)
    inside_samples = lab_image[inside > 0]
    outside_samples = lab_image[outside > 0]
    if len(inside_samples) < 40 or len(outside_samples) < 40:
        return 0.0
    inside_colour = np.median(inside_samples, axis=0)
    outside_colour = np.median(outside_samples, axis=0)
    return float(np.linalg.norm(inside_colour - outside_colour))


def _line_intersection(
    first_point: np.ndarray,
    first_direction: np.ndarray,
    second_point: np.ndarray,
    second_direction: np.ndarray,
) -> np.ndarray | None:
    matrix = np.column_stack((first_direction, -second_direction))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    parameters = np.linalg.solve(matrix, second_point - first_point)
    return first_point + parameters[0] * first_direction


def _sample_lab_along_points(
    lab_image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    map_x = points[:, 0].astype(np.float32).reshape(1, -1)
    map_y = points[:, 1].astype(np.float32).reshape(1, -1)
    sampled = cv2.remap(
        lab_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1, 3).astype(np.float32)


def _refine_a4_edges(
    image: np.ndarray,
    coarse_quad: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray:
    """Snap a coarse A4 quad to the real paper/table colour transitions.

    The mandatory divider is an excellent coarse locator, but its hand-drawn
    endpoints commonly stop a few pixels before the paper edges.  Extending
    that line with the A4 ratio therefore leaves narrow table/paper strips in
    the rectified view.  Refine each outer edge independently at the original
    image resolution and intersect the four fitted lines.
    """

    if not bool(paper_cfg.get("edge_refine_enabled", True)):
        return order_quad(coarse_quad)

    height, width = image.shape[:2]
    minimum_dimension = float(min(height, width))
    search = float(
        np.clip(
            minimum_dimension
            * float(paper_cfg.get("edge_refine_search_ratio", 0.055)),
            10.0,
            float(paper_cfg.get("edge_refine_max_search_px", 72.0)),
        )
    )
    probe = float(
        np.clip(
            minimum_dimension
            * float(paper_cfg.get("edge_refine_probe_ratio", 0.0045)),
            2.0,
            6.0,
        )
    )
    minimum_step = float(
        paper_cfg.get("edge_refine_min_lab_step", 5.0)
    )
    group_count = int(
        np.clip(paper_cfg.get("edge_refine_groups", 9), 5, 15)
    )
    quad = order_quad(np.asarray(coarse_quad, dtype=np.float32))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = cv2.GaussianBlur(lab, (5, 5), 0)
    offsets = np.linspace(
        -search, search, max(31, int(round(2.0 * search)) + 1),
        dtype=np.float32,
    )
    fitted_lines: list[tuple[np.ndarray, np.ndarray]] = []
    accepted_edges = 0

    for edge_index in range(4):
        start = quad[edge_index].astype(np.float64)
        end = quad[(edge_index + 1) % 4].astype(np.float64)
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length < 12.0:
            return quad
        tangent = vector / length
        # order_quad returns TL, TR, BR, BL. In image coordinates this makes
        # [dy, -dx] point outside on every edge.
        outward = np.asarray([tangent[1], -tangent[0]], dtype=np.float64)
        sample_count = int(np.clip(round(length / 4.0), 64, 220))
        fractions = np.linspace(0.055, 0.945, sample_count)
        centres = (
            start[None, :]
            + fractions[:, None] * vector[None, :]
        )

        # One remap per side of an edge is markedly faster than issuing a
        # remap for every offset (important on RDK X5 and desktop live video).
        boundaries = (
            centres[None, :, :]
            + offsets[:, None, None] * outward[None, None, :]
        )
        outside_points = (
            boundaries + probe * outward[None, None, :]
        ).reshape(-1, 2)
        inside_points = (
            boundaries - probe * outward[None, None, :]
        ).reshape(-1, 2)
        outside_lab = _sample_lab_along_points(
            lab, outside_points
        ).reshape(len(offsets), sample_count, 3)
        inside_lab = _sample_lab_along_points(
            lab, inside_points
        ).reshape(len(offsets), sample_count, 3)
        differences = np.linalg.norm(
            outside_lab - inside_lab, axis=2
        ).astype(np.float32)

        # A genuine outer edge produces a colour step along almost its whole
        # length.  The lower quantile deliberately rejects isolated shadows,
        # printed marks and the divider crossing an edge.
        aggregate = (
            np.percentile(differences, 32.0, axis=1)
            + 0.35 * np.median(differences, axis=1)
        )
        best_index = int(np.argmax(aggregate))
        best_offset = float(offsets[best_index])
        best_score = float(aggregate[best_index])
        coarse_index = int(np.argmin(np.abs(offsets)))
        coarse_score = float(aggregate[coarse_index])
        if best_score < minimum_step or (
            abs(best_offset) > 0.72 * search
            and best_score < max(minimum_step * 1.8, coarse_score * 1.18)
        ):
            fitted_lines.append((start, tangent))
            continue

        # Permit a small change in edge angle by locating the colour step in
        # several longitudinal groups, then fitting a line through those
        # points.  A penalty keeps local selections on the same physical edge
        # instead of jumping to a nearby piece or shadow.
        edge_points: list[np.ndarray] = []
        for group_indices in np.array_split(
            np.arange(sample_count), group_count
        ):
            if len(group_indices) < 2:
                continue
            group_scores = np.percentile(
                differences[:, group_indices], 38.0, axis=1
            )
            group_scores -= (
                0.045 * np.abs(offsets - best_offset)
            )
            local_index = int(np.argmax(group_scores))
            local_score = float(
                np.percentile(
                    differences[local_index, group_indices], 38.0
                )
            )
            if local_score < 0.72 * minimum_step:
                continue
            fraction = float(np.median(fractions[group_indices]))
            edge_points.append(
                start
                + fraction * vector
                + float(offsets[local_index]) * outward
            )

        if len(edge_points) >= 5:
            points = np.asarray(edge_points, dtype=np.float32)
            fit = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01)
            direction = np.asarray(
                [float(fit[0]), float(fit[1])], dtype=np.float64
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            if float(np.dot(direction, tangent)) < 0:
                direction = -direction
            maximum_angle = float(
                paper_cfg.get("edge_refine_max_angle_deg", 4.5)
            )
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(direction, tangent)), -1.0, 1.0
                        )
                    )
                )
            )
            if angle <= maximum_angle:
                line_point = np.asarray(
                    [float(fit[2]), float(fit[3])], dtype=np.float64
                )
            else:
                line_point = start + best_offset * outward
                direction = tangent
        else:
            line_point = start + best_offset * outward
            direction = tangent
        fitted_lines.append((line_point, direction))
        accepted_edges += 1

    if accepted_edges < 2:
        return quad

    refined_points: list[np.ndarray] = []
    for corner_index in range(4):
        previous_line = fitted_lines[(corner_index - 1) % 4]
        current_line = fitted_lines[corner_index]
        intersection = _line_intersection(
            previous_line[0],
            previous_line[1],
            current_line[0],
            current_line[1],
        )
        if intersection is None:
            return quad
        refined_points.append(intersection)
    refined = order_quad(np.asarray(refined_points, dtype=np.float32))

    allowance = max(6.0, 0.025 * minimum_dimension)
    if (
        np.min(refined[:, 0]) < -allowance
        or np.max(refined[:, 0]) > width + allowance
        or np.min(refined[:, 1]) < -allowance
        or np.max(refined[:, 1]) > height + allowance
        or not cv2.isContourConvex(np.rint(refined).astype(np.int32))
    ):
        return quad
    coarse_area = abs(float(cv2.contourArea(quad)))
    refined_area = abs(float(cv2.contourArea(refined)))
    area_ratio = refined_area / max(coarse_area, 1.0)
    if not 0.76 <= area_ratio <= 1.30:
        return quad

    horizontal = 0.5 * (
        np.linalg.norm(refined[1] - refined[0])
        + np.linalg.norm(refined[2] - refined[3])
    )
    vertical = 0.5 * (
        np.linalg.norm(refined[3] - refined[0])
        + np.linalg.norm(refined[2] - refined[1])
    )
    observed_aspect = max(horizontal, vertical) / max(
        min(horizontal, vertical), 1.0
    )
    expected_aspect = float(paper_cfg["height_mm"]) / float(
        paper_cfg["width_mm"]
    )
    aspect_error = abs(observed_aspect - expected_aspect) / expected_aspect
    if aspect_error > float(
        paper_cfg.get("edge_refine_max_aspect_error_ratio", 0.20)
    ):
        return quad

    # Keep the mandatory divider as an independent safeguard against snapping
    # onto a similarly coloured rectangle elsewhere in the scene.
    if not _quick_candidate_has_divider(image, refined, paper_cfg):
        return quad
    return refined.astype(np.float32)


def _finalise_a4_corners(
    frame: np.ndarray,
    corners: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray:
    corners = _refine_a4_edges(frame, corners, paper_cfg)
    top = np.linalg.norm(corners[1] - corners[0])
    left = np.linalg.norm(corners[3] - corners[0])
    if top > left:
        corners = np.roll(corners, -1, axis=0)
    return _rotate_corner_order(
        corners.astype(np.float32), paper_cfg["rotation_quadrants"]
    )


def _find_a4_from_divider_line(
    image: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Recover the complete A4 from its mandatory mid-sheet divider.

    With a near-normal camera, the divider endpoints give the sheet width and
    the known A4 aspect ratio gives the two outer halves.  This remains stable
    when a dark corner of coloured paper blends into the table.
    """

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(16, 0.45 * median))
    high = int(min(230, max(low + 28, 1.35 * median)))
    edges = cv2.Canny(gray, low, high)
    minimum_dimension = min(height, width)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 360.0,
        threshold=max(24, int(minimum_dimension * 0.03)),
        minLineLength=max(90, int(minimum_dimension * 0.17)),
        maxLineGap=max(28, int(minimum_dimension * 0.08)),
    )
    if lines is None:
        return None

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    aspect = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    image_area = float(width * height)
    maximum_area_ratio = float(
        paper_cfg.get("maximum_image_area_ratio", 0.78)
    )
    candidates: list[tuple[float, np.ndarray]] = []
    for raw in lines[:, 0]:
        first = np.asarray(raw[:2], dtype=np.float64)
        second = np.asarray(raw[2:], dtype=np.float64)
        if first[0] > second[0]:
            first, second = second, first
        vector = second - first
        length = float(np.linalg.norm(vector))
        if length <= 1.0 or abs(vector[1]) > 0.16 * abs(vector[0]):
            continue
        tangent = vector / length
        downward = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        if downward[1] < 0:
            downward = -downward
        half_height = 0.5 * length * aspect
        quad = np.asarray(
            [
                first - downward * half_height,
                second - downward * half_height,
                second + downward * half_height,
                first + downward * half_height,
            ],
            dtype=np.float32,
        )
        allowance = max(3.0, minimum_dimension * 0.008)
        if (
            np.min(quad[:, 0]) < -allowance
            or np.max(quad[:, 0]) > width + allowance
            or np.min(quad[:, 1]) < -allowance
            or np.max(quad[:, 1]) > height + allowance
        ):
            continue
        area_ratio = abs(cv2.contourArea(quad)) / image_area
        if (
            area_ratio < float(
                paper_cfg.get("minimum_image_area_ratio", 0.045)
            )
            or area_ratio > maximum_area_ratio
        ):
            continue
        boundary_contrast = _quad_boundary_colour_contrast(lab, quad)
        if boundary_contrast < 5.0:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
        inset = max(3, int(round(minimum_dimension * 0.018)))
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
            ),
        )
        # Exclude the divider itself before measuring paper uniformity.
        centre_y = int(round(0.25 * np.sum(quad[:, 1])))
        divider_margin = max(2, int(round(minimum_dimension * 0.012)))
        mask[
            max(0, centre_y - divider_margin) :
            min(height, centre_y + divider_margin + 1)
        ] = 0
        samples = lab[mask > 0]
        if len(samples) < 300:
            continue
        paper_colour = np.median(samples, axis=0)
        dispersion = float(
            np.median(np.linalg.norm(samples - paper_colour, axis=1))
        )
        centre = np.mean(quad, axis=0)
        centre_error = float(
            np.linalg.norm(
                centre - np.asarray([width * 0.5, height * 0.5])
            )
            / max(math.hypot(width, height), 1.0)
        )
        score = (
            0.9 * area_ratio
            + 0.025 * min(boundary_contrast, 70.0)
            - 0.014 * min(dispersion, 60.0)
            - 0.12 * centre_error
        )
        candidates.append((score, quad))

    for _, candidate in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        if _quick_candidate_has_divider(image, candidate, paper_cfg):
            return candidate
    return None


def _find_a4_from_central_colour(
    image: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Find the large, nearly uniform sheet that occupies the image centre.

    In the competition setup the A4 is always present, approximately centred,
    and its lower half contains no pieces.  Sampling several central/lower
    patches therefore gives a much stronger and faster prior than requiring a
    perfect grayscale border.  This works for white, blue, red, or other paper
    colours and bridges the black divider before contour extraction.
    """

    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample_colours: list[np.ndarray] = []
    paper_colour = paper_cfg.get("color_bgr")
    if paper_colour is not None:
        pixel = np.asarray(paper_colour, dtype=np.uint8).reshape(1, 1, 3)
        sample_colours.append(
            cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        )

    patch_half_w = max(8, int(round(width * 0.055)))
    patch_half_h = max(8, int(round(height * 0.045)))
    for x_fraction, y_fraction in (
        (0.50, 0.62),
        (0.50, 0.52),
        (0.42, 0.62),
        (0.58, 0.62),
        (0.50, 0.40),
    ):
        cx = int(round(width * x_fraction))
        cy = int(round(height * y_fraction))
        patch = lab[
            max(0, cy - patch_half_h) : min(height, cy + patch_half_h + 1),
            max(0, cx - patch_half_w) : min(width, cx + patch_half_w + 1),
        ]
        if patch.size:
            sample_colours.append(np.median(patch.reshape(-1, 3), axis=0))

    image_area = float(width * height)
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    minimum_area_ratio = float(paper_cfg.get("minimum_image_area_ratio", 0.045))
    open_size = max(3, int(round(min(height, width) * 0.005)) | 1)
    divider_bridge_x = max(5, int(round(min(height, width) * 0.010)) | 1)
    divider_bridge_y = max(9, int(round(min(height, width) * 0.018)) | 1)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    bridge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (divider_bridge_x, divider_bridge_y)
    )
    candidates: list[tuple[float, np.ndarray]] = []
    seen_colours: set[tuple[int, int, int]] = set()
    for colour in sample_colours:
        colour_key = tuple(np.rint(colour / 4.0).astype(int).tolist())
        if colour_key in seen_colours:
            continue
        seen_colours.add(colour_key)
        distance_modes: list[tuple[str, np.ndarray, tuple[float, ...]]] = [
            ("lab", np.linalg.norm(lab - colour, axis=2), (22.0, 32.0, 44.0))
        ]
        chroma_strength = float(
            np.linalg.norm(colour[1:] - np.asarray([128.0, 128.0]))
        )
        # Black and dark-grey paper is easily split into several Lab clusters
        # by vignetting and specular reflections.  Its useful distinction from
        # the table is primarily lightness, not a/b chroma.  Sweep an L-only
        # mask as an additional candidate source; the A4 aspect ratio and the
        # mandatory divider still validate the resulting quadrilateral, so a
        # broad shadow on the table is not accepted as paper.
        if chroma_strength < 14.0 or float(colour[0]) < float(
            paper_cfg.get("dark_paper_lightness_threshold_lab", 92.0)
        ):
            distance_modes.insert(
                0,
                (
                    "lightness",
                    np.abs(lab[:, :, 0] - float(colour[0])),
                    (8.0, 14.0, 22.0, 34.0, 48.0),
                ),
            )
        if chroma_strength >= 7.0:
            distance_modes.insert(
                0,
                (
                    "chroma",
                    np.linalg.norm(lab[:, :, 1:] - colour[1:], axis=2),
                    (8.0, 12.0, 18.0, 26.0),
                ),
            )
        for distance_mode, distance, tolerances in distance_modes:
            for tolerance in tolerances:
                mask = np.where(distance <= tolerance, 255, 0).astype(np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, bridge_kernel, iterations=2
                )
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for contour in contours:
                    area = abs(cv2.contourArea(contour))
                    if area < image_area * minimum_area_ratio:
                        continue
                    hull = cv2.convexHull(contour)
                    hull_perimeter = cv2.arcLength(hull, True)
                    quad: np.ndarray | None = None
                    for epsilon in (
                        0.008,
                        0.012,
                        0.018,
                        0.025,
                        0.035,
                        0.05,
                        0.075,
                    ):
                        approx = cv2.approxPolyDP(
                            hull, epsilon * hull_perimeter, True
                        )
                        if len(approx) == 4 and cv2.isContourConvex(approx):
                            quad = order_quad(approx.reshape(4, 2))
                            break
                    if quad is None:
                        continue
                    # A competition camera can be mounted tightly enough that
                    # one A4 corner lies on the image boundary.  Do not reject
                    # such a colour candidate here: the mandatory divider
                    # validation below is a stronger guard against selecting
                    # the table or only half of the sheet.
                    if (
                        np.min(quad[:, 0]) < -2.0
                        or np.max(quad[:, 0]) > width + 2.0
                        or np.min(quad[:, 1]) < -2.0
                        or np.max(quad[:, 1]) > height + 2.0
                    ):
                        continue
                    # One perspective corner may touch the video boundary,
                    # but an entire candidate edge on the frame boundary is
                    # normally a thresholded table/background region.
                    boundary_counts = (
                        int(np.count_nonzero(quad[:, 0] <= 1.0)),
                        int(
                            np.count_nonzero(
                                quad[:, 0] >= width - 2.0
                            )
                        ),
                        int(np.count_nonzero(quad[:, 1] <= 1.0)),
                        int(
                            np.count_nonzero(
                                quad[:, 1] >= height - 2.0
                            )
                        ),
                    )
                    if max(boundary_counts) >= 2:
                        continue
                    horizontal = 0.5 * (
                        np.linalg.norm(quad[1] - quad[0])
                        + np.linalg.norm(quad[2] - quad[3])
                    )
                    vertical = 0.5 * (
                        np.linalg.norm(quad[3] - quad[0])
                        + np.linalg.norm(quad[2] - quad[1])
                    )
                    if min(horizontal, vertical) < min(height, width) * 0.12:
                        continue
                    ratio = max(horizontal, vertical) / max(
                        min(horizontal, vertical), 1.0
                    )
                    aspect_error = abs(ratio - target_ratio) / target_ratio
                    if aspect_error > float(
                        paper_cfg.get(
                            "maximum_colour_aspect_error_ratio", 0.22
                        )
                    ):
                        continue
                    # A colour threshold can occasionally return the entire
                    # video frame.  A 16:9 camera frame is not an A4 sheet,
                    # and accepting it lets a floor seam masquerade as the
                    # required divider.
                    if (
                        np.min(quad[:, 0]) <= 1.0
                        and np.max(quad[:, 0]) >= width - 2.0
                        and np.min(quad[:, 1]) <= 1.0
                        and np.max(quad[:, 1]) >= height - 2.0
                    ):
                        continue
                    quad_area = abs(cv2.contourArea(quad))
                    if quad_area / image_area > float(
                        paper_cfg.get(
                            "maximum_image_area_ratio", 0.78
                        )
                    ):
                        continue
                    rectangularity = area / max(quad_area, 1.0)
                    centre = np.mean(quad, axis=0)
                    centre_error = float(
                        np.linalg.norm(
                            centre
                            - np.asarray([width * 0.5, height * 0.5])
                        )
                        / max(math.hypot(width, height), 1.0)
                    )
                    score = (
                        3.0 * quad_area / image_area
                        + 0.35 * min(rectangularity, 1.0)
                        - 0.8 * aspect_error
                        - 0.25 * centre_error
                        - 0.002 * tolerance
                        + (0.08 if distance_mode == "chroma" else 0.0)
                        + (0.05 if distance_mode == "lightness" else 0.0)
                    )
                    candidates.append((score, quad))
    if not candidates:
        return None
    # Threshold sweeps often rediscover the same quadrilateral.  Deduplicate
    # those candidates, then strongly reward the task guarantee that the A4
    # colour is visibly different from its immediate surroundings.
    unique: dict[tuple[int, ...], tuple[float, np.ndarray]] = {}
    for score, candidate in candidates:
        key = tuple(np.rint(candidate.reshape(-1) / 6.0).astype(int).tolist())
        previous = unique.get(key)
        if previous is None or score > previous[0]:
            unique[key] = (score, candidate)
    candidates = []
    for score, candidate in unique.values():
        boundary_contrast = _quad_boundary_colour_contrast(lab, candidate)
        candidates.append(
            (
                score + 0.018 * min(boundary_contrast, 70.0),
                candidate,
            )
        )
    # The most uniform region can include a similarly coloured laptop or
    # background object.  Try candidates in score order and retain the first
    # one that also contains the task's divider at the expected A4 position.
    for _, candidate in sorted(candidates, key=lambda item: item[0], reverse=True):
        candidate = candidate.copy()
        top = np.linalg.norm(candidate[1] - candidate[0])
        left = np.linalg.norm(candidate[3] - candidate[0])
        if top > left:
            candidate = np.roll(candidate, -1, axis=0)
        if _quick_candidate_has_divider(image, candidate, paper_cfg):
            return candidate
    return None


def _quick_candidate_has_divider(
    image: np.ndarray,
    corners: np.ndarray,
    paper_cfg: dict[str, Any],
) -> bool:
    """Reject a colour-region quadrilateral that cannot contain the task sheet."""

    width = 210
    height = 297
    destination = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        np.asarray(corners, dtype=np.float32), destination
    )
    preview = cv2.warpPerspective(image, homography, (width, height))
    try:
        find_divider(preview, paper_cfg, 1.0)
        return True
    except DetectionError:
        return False


def _find_a4_from_long_lines(
    image: np.ndarray,
    edges: np.ndarray,
    paper_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Recover an A4 quadrilateral when clutter interrupts one outer edge.

    Two long side lines plus either the top or bottom line are sufficient
    because the A4 aspect ratio is known.  This is especially useful when a
    laptop or mechanism sits close to one paper edge.
    """

    height, width = image.shape[:2]
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(45, int(min(height, width) * 0.04)),
        minLineLength=max(120, int(min(height, width) * 0.18)),
        maxLineGap=max(30, int(min(height, width) * 0.045)),
    )
    if lines is None:
        return None

    vertical: list[tuple[float, float, float, float, float]] = []
    horizontal: list[tuple[float, float, float]] = []
    middle_y = height * 0.5
    middle_x = width * 0.5
    for raw in lines[:, 0]:
        x1, y1, x2, y2 = map(float, raw)
        dx, dy = x2 - x1, y2 - y1
        length = float(math.hypot(dx, dy))
        if abs(dy) >= 2.0 * abs(dx) and abs(dy) > 1.0:
            slope = dx / dy
            intercept = x1 - slope * y1
            x_middle = intercept + slope * middle_y
            vertical.append((x_middle, length, slope, intercept, min(y1, y2)))
        elif abs(dx) >= 2.0 * abs(dy) and abs(dx) > 1.0:
            slope = dy / dx
            intercept = y1 - slope * x1
            y_middle = intercept + slope * middle_x
            horizontal.append((y_middle, length, slope))
    if len(vertical) < 2 or not horizontal:
        return None

    def deduplicate(
        values: list[tuple[Any, ...]], coordinate_index: int
    ) -> list[tuple[Any, ...]]:
        selected: dict[int, tuple[Any, ...]] = {}
        for item in values:
            key = int(round(float(item[coordinate_index]) / 18.0))
            if key not in selected or float(item[1]) > float(selected[key][1]):
                selected[key] = item
        return list(selected.values())

    vertical = deduplicate(vertical, 0)
    horizontal = deduplicate(horizontal, 0)
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour_hint_lab: np.ndarray | None = None
    if paper_cfg.get("color_bgr") is not None:
        pixel = np.asarray(paper_cfg["color_bgr"], dtype=np.uint8).reshape(1, 1, 3)
        colour_hint_lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB).reshape(3)

    candidates: list[tuple[float, np.ndarray]] = []
    for left, right in itertools.combinations(vertical, 2):
        if left[0] > right[0]:
            left, right = right, left
        # The camera is mounted approximately perpendicular to the table, so
        # A4 side edges remain near vertical and nearly parallel.
        if (
            abs(left[2]) > 0.065
            or abs(right[2]) > 0.065
            or abs(left[2] - right[2]) > 0.055
        ):
            continue
        paper_width = right[0] - left[0]
        # A 16:9 camera can include the complete work area around the A4, so
        # the sheet may occupy only about one quarter of the frame width.
        if paper_width < width * 0.18:
            continue
        paper_height = paper_width * target_ratio
        side_evidence = min(1.0, (left[1] + right[1]) / max(height, 1.0))
        for y_line, line_length, _ in horizontal:
            for top, bottom in (
                (y_line, y_line + paper_height),
                (y_line - paper_height, y_line),
            ):
                if top < -5 or bottom > height + 5:
                    continue

                def x_at(item: tuple[float, ...], y: float) -> float:
                    return item[3] + item[2] * y

                quad = np.asarray(
                    [
                        [x_at(left, top), top],
                        [x_at(right, top), top],
                        [x_at(right, bottom), bottom],
                        [x_at(left, bottom), bottom],
                    ],
                    dtype=np.float32,
                )
                frame_margin = max(4.0, min(height, width) * 0.008)
                if (
                    np.min(quad[:, 0]) < frame_margin
                    or np.max(quad[:, 0]) > width - frame_margin
                    or top < frame_margin
                    or bottom > height - frame_margin
                ):
                    continue
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
                inset = max(5, int(round(min(height, width) * 0.015)))
                mask = cv2.erode(
                    mask,
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
                    ),
                )
                samples = lab[mask > 0]
                if len(samples) < 1000:
                    continue
                median = np.median(samples, axis=0)
                dispersion = float(
                    np.median(np.linalg.norm(samples - median, axis=1))
                )
                colour_error = (
                    float(np.linalg.norm(median - colour_hint_lab))
                    if colour_hint_lab is not None
                    else 0.0
                )
                line_evidence = min(1.0, line_length / max(paper_width, 1.0))
                area_score = cv2.contourArea(quad) / float(width * height)
                score = (
                    area_score
                    + 0.18 * side_evidence
                    + 0.12 * line_evidence
                    - 0.018 * dispersion
                    - 0.004 * colour_error
                )
                candidates.append((score, quad))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_a4_corners(frame: np.ndarray, paper_cfg: dict[str, Any]) -> np.ndarray:
    override = paper_cfg.get("corner_override_px")
    if override is not None:
        corners = np.asarray(override, dtype=np.float32)
        if corners.shape != (4, 2):
            raise DetectionError("paper.corner_override_px must contain four [x,y] points")
        return _rotate_corner_order(order_quad(corners), paper_cfg["rotation_quadrants"])

    height, width = frame.shape[:2]
    maximum_dimension = float(paper_cfg.get("detection_max_dimension", 960))
    scale = min(1.0, maximum_dimension / max(height, width))
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    divider_candidate = _find_a4_from_divider_line(small, paper_cfg)
    # A horizontal wood seam can look like the mandatory divider after the
    # camera is rotated 90 degrees.  In that case the divider-derived quad is
    # usually a small false A4, while the colour boundary finds the complete
    # sheet.  Prefer a substantially larger colour-consistent quad whenever the
    # divider candidate occupies only a small part of the frame.
    divider_area_ratio = (
        abs(cv2.contourArea(divider_candidate))
        / float(max(small.shape[0] * small.shape[1], 1))
        if divider_candidate is not None
        else 0.0
    )
    colour_candidate = None
    if divider_candidate is None or divider_area_ratio < 0.20:
        colour_candidate = _find_a4_from_central_colour(small, paper_cfg)
    if divider_candidate is not None and (
        colour_candidate is None
        or abs(cv2.contourArea(divider_candidate))
        >= 0.70 * abs(cv2.contourArea(colour_candidate))
    ):
        corners = divider_candidate / scale
        return _finalise_a4_corners(frame, corners, paper_cfg)
    if colour_candidate is not None:
        corners = colour_candidate / scale
        return _finalise_a4_corners(frame, corners, paper_cfg)

    fallback_dimension = float(
        paper_cfg.get("fallback_detection_max_dimension", 1600)
    )
    fallback_scale = min(1.0, fallback_dimension / max(height, width))
    if fallback_scale > scale + 1e-3:
        scale = fallback_scale
        small = cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(20, 0.55 * median))
    high = int(min(240, max(low + 30, 1.45 * median)))
    edges = cv2.Canny(gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(small.shape[0] * small.shape[1])
    target_ratio = float(paper_cfg["height_mm"]) / float(paper_cfg["width_mm"])
    paper_colour = paper_cfg.get("color_bgr")
    colour_hint_lab: np.ndarray | None = None
    small_lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    if paper_colour is not None:
        colour_pixel = np.asarray(paper_colour, dtype=np.uint8).reshape(1, 1, 3)
        colour_hint_lab = cv2.cvtColor(
            colour_pixel, cv2.COLOR_BGR2LAB
        ).reshape(3).astype(np.float32)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = abs(cv2.contourArea(contour))
        if area < image_area * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_quad(approx.reshape(4, 2))
        horizontal = 0.5 * (
            np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])
        )
        vertical = 0.5 * (
            np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])
        )
        if min(horizontal, vertical) < 1:
            continue
        ratio = max(horizontal, vertical) / min(horizontal, vertical)
        aspect_error = abs(ratio - target_ratio) / target_ratio
        if aspect_error > 0.24:
            continue
        rectangularity = area / max(horizontal * vertical, 1.0)
        score = area / image_area - 0.8 * aspect_error + 0.1 * rectangularity
        boundary_contrast = _quad_boundary_colour_contrast(
            small_lab, quad
        )
        score += 0.018 * min(boundary_contrast, 70.0)
        if colour_hint_lab is not None:
            colour_mask = np.zeros(small.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(
                colour_mask, np.rint(quad).astype(np.int32), 255
            )
            inset = max(3, int(round(min(small.shape[:2]) * 0.01)))
            colour_mask = cv2.erode(
                colour_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)
                ),
            )
            samples = small_lab[colour_mask > 0]
            if len(samples):
                observed_colour = np.median(samples, axis=0)
                colour_error = float(
                    np.linalg.norm(observed_colour - colour_hint_lab)
                )
                score -= float(paper_cfg.get("color_hint_weight", 0.8)) * min(
                    colour_error / 100.0, 1.5
                )
        candidates.append((score, quad))

    corners = None
    if candidates:
        # A half sheet split by the black task divider can itself look almost
        # exactly like an A-series rectangle.  Area/aspect scoring alone may
        # therefore select only the upper half, especially when four bright
        # pieces make that region easy to contour.  The real task sheet must
        # also contain the divider near its middle, so validate candidates in
        # score order before accepting one.
        for _, candidate in sorted(
            candidates, key=lambda item: item[0], reverse=True
        ):
            candidate_for_check = candidate.copy()
            candidate_top = np.linalg.norm(
                candidate_for_check[1] - candidate_for_check[0]
            )
            candidate_left = np.linalg.norm(
                candidate_for_check[3] - candidate_for_check[0]
            )
            if candidate_top > candidate_left:
                candidate_for_check = np.roll(
                    candidate_for_check, -1, axis=0
                )
            if _quick_candidate_has_divider(
                small, candidate_for_check, paper_cfg
            ):
                corners = candidate_for_check
                break
    if corners is None:
        corners = _find_a4_from_long_lines(small, edges, paper_cfg)
        if corners is None:
            raise DetectionError(
                "A4 sheet was not found. Improve border contrast or set "
                "paper.corner_override_px in config.json."
            )
    corners = corners / scale

    return _finalise_a4_corners(frame, corners, paper_cfg)


def rectify_paper(
    frame: np.ndarray,
    paper_cfg: dict[str, Any],
    cached_corners: np.ndarray | None = None,
) -> PaperView:
    corners = (
        np.asarray(cached_corners, dtype=np.float32).copy()
        if cached_corners is not None
        else find_a4_corners(frame, paper_cfg)
    )
    ppm = float(paper_cfg["pixels_per_mm"])
    out_w = int(round(float(paper_cfg["width_mm"]) * ppm))
    out_h = int(round(float(paper_cfg["height_mm"]) * ppm))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(corners, destination)
    image = cv2.warpPerspective(frame, homography, (out_w, out_h))
    divider_y_mm, divider_width_mm, divider_contrast = find_divider(
        image, paper_cfg, ppm
    )
    return PaperView(
        image=image,
        homography=homography,
        corners_px=corners,
        pixels_per_mm=ppm,
        width_mm=float(paper_cfg["width_mm"]),
        height_mm=float(paper_cfg["height_mm"]),
        divider_y_mm=divider_y_mm,
        divider_width_mm=divider_width_mm,
        divider_contrast_lab=divider_contrast,
    )


def find_divider(
    rectified: np.ndarray,
    paper_cfg: dict[str, Any],
    pixels_per_mm: float,
) -> tuple[float, float, float]:
    """Locate and validate the horizontal upper/lower divider on the A4 sheet.

    The competition sheet has one obvious solid line near half height.  Median
    Lab colour per row makes the test insensitive to the A4/table colours and
    suppresses local dirt, wrinkles, and pieces that cover only part of a row.
    """

    ppm = float(pixels_per_mm)
    expected = int(round(float(paper_cfg["divider_y_mm"]) * ppm))
    half_range = max(
        1,
        int(round(float(paper_cfg["divider_search_half_range_mm"]) * ppm)),
    )
    maximum_width = max(
        1,
        int(round(float(paper_cfg["divider_max_width_mm"]) * ppm)),
    )
    minimum_contrast = float(paper_cfg["divider_min_contrast_lab"])
    x_margin = max(2, int(round(8.0 * ppm)))
    if rectified.shape[1] <= 2 * x_margin + 20:
        raise DetectionError("Rectified A4 image is too narrow to validate divider")

    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB).astype(np.float32)
    strip = lab[:, x_margin:-x_margin]
    row_colour = np.median(strip, axis=1)
    gap = maximum_width + max(2, int(round(1.0 * ppm)))
    flank = max(3, int(round(3.0 * ppm)))
    support_quantile = float(
        paper_cfg.get("divider_horizontal_support_quantile", 0.34)
    )
    support_quantile = min(max(support_quantile, 0.15), 0.49)

    def score_window(start: int, stop: int) -> np.ndarray:
        """Score both dark and bright full-width divider strokes.

        The previous global row median was reliable for a black line on a
        bright sheet, but a silver/white marker on black paper can contain
        narrow reflections and exposure gradients.  Per-column local
        baselines plus two-sided quantiles preserve a line that covers most of
        the sheet while rejecting a fragment or glare patch that covers only a
        small part of a row.
        """

        scores = np.zeros(max(0, stop - start), dtype=np.float32)
        lightness = strip[:, :, 0]
        for offset, y in enumerate(range(start, stop)):
            upper_rows = strip[max(0, y - gap - flank) : y - gap]
            lower_rows = strip[
                y + gap + 1 : min(len(strip), y + gap + flank + 1)
            ]
            if len(upper_rows) == 0 or len(lower_rows) == 0:
                continue
            baseline_colour = np.median(
                np.vstack(
                    (
                        np.median(upper_rows, axis=1),
                        np.median(lower_rows, axis=1),
                    )
                ),
                axis=0,
            )
            median_score = float(
                np.linalg.norm(row_colour[y] - baseline_colour)
            )

            upper_l = lightness[max(0, y - gap - flank) : y - gap]
            lower_l = lightness[
                y + gap + 1 : min(len(lightness), y + gap + flank + 1)
            ]
            baseline_l = np.median(
                np.concatenate((upper_l, lower_l), axis=0), axis=0
            )
            delta_l = lightness[y] - baseline_l
            bright_score = float(
                np.quantile(delta_l, support_quantile)
            )
            dark_score = float(
                -np.quantile(delta_l, 1.0 - support_quantile)
            )
            supported_score = max(0.0, bright_score, dark_score)
            scores[offset] = max(median_score, supported_score)
        return scores

    start = max(maximum_width + 2, expected - half_range)
    stop = min(len(row_colour) - maximum_width - 2, expected + half_range + 1)
    if stop <= start:
        raise DetectionError("Divider search window is outside the rectified A4")

    scores = score_window(start, stop)
    peak_offset = int(np.argmax(scores))
    peak_score = float(scores[peak_offset])
    minimum_continuous_ratio = float(
        paper_cfg.get("divider_min_continuous_width_ratio", 0.55)
    )
    minimum_continuous_ratio = min(
        max(minimum_continuous_ratio, 0.30), 0.90
    )

    def row_has_wide_stroke(y: int, score: float) -> bool:
        upper_l = strip[
            max(0, y - gap - flank) : y - gap, :, 0
        ]
        lower_l = strip[
            y + gap + 1 : min(len(strip), y + gap + flank + 1), :, 0
        ]
        if len(upper_l) == 0 or len(lower_l) == 0:
            return False
        baseline_l = np.median(
            np.concatenate((upper_l, lower_l), axis=0), axis=0
        )
        delta_l = strip[y, :, 0] - baseline_l
        stroke_threshold = max(
            5.0,
            min(
                max(minimum_contrast * 0.65, score * 0.42),
                max(minimum_contrast, score * 0.70),
            ),
        )
        gap_close = max(
            3,
            int(
                round(
                    len(delta_l)
                    * float(
                        paper_cfg.get(
                            "divider_max_horizontal_gap_ratio", 0.035
                        )
                    )
                )
            ),
        )
        kernel = np.ones((1, gap_close), dtype=np.uint8)
        for signed in (delta_l >= stroke_threshold, delta_l <= -stroke_threshold):
            closed = cv2.morphologyEx(
                signed.astype(np.uint8).reshape(1, -1),
                cv2.MORPH_CLOSE,
                kernel,
            ).reshape(-1)
            padded = np.pad(closed, (1, 1), mode="constant")
            changes = np.diff(padded.astype(np.int8))
            run_starts = np.flatnonzero(changes == 1)
            run_ends = np.flatnonzero(changes == -1)
            longest = (
                int(np.max(run_ends - run_starts))
                if len(run_starts)
                else 0
            )
            if longest / max(len(closed), 1) >= minimum_continuous_ratio:
                return True
        return False

    def select_wide_peak(
        window_scores: np.ndarray,
        window_start: int,
        required_score: float,
    ) -> tuple[int, float] | None:
        for candidate_offset in np.argsort(window_scores)[::-1]:
            candidate_score = float(window_scores[candidate_offset])
            if candidate_score < required_score:
                break
            if row_has_wide_stroke(
                window_start + int(candidate_offset), candidate_score
            ):
                return int(candidate_offset), candidate_score
        return None

    selected = select_wide_peak(scores, start, minimum_contrast)
    accepted = selected is not None
    diagnostic_peak = peak_score
    if selected is not None:
        peak_offset, peak_score = selected

    # A dark A4 is often detected from a slightly incomplete outer contour:
    # vignetting can move the true mid-line just beyond the narrow nominal
    # window even though the line itself is obvious.  Search a wider central
    # band only when it contains a substantially stronger, broad line.  This
    # is not a threshold relaxation and therefore does not turn floor seams
    # into valid dividers.
    if not accepted:
        broad_half_range = max(
            half_range,
            int(
                round(
                    float(
                        paper_cfg.get(
                            "divider_broad_search_half_range_mm", 38.0
                        )
                    )
                    * ppm
                )
            ),
        )
        broad_start = max(maximum_width + 2, expected - broad_half_range)
        broad_stop = min(
            len(row_colour) - maximum_width - 2,
            expected + broad_half_range + 1,
        )
        broad_scores = score_window(broad_start, broad_stop)
        broad_peak = float(np.max(broad_scores))
        diagnostic_peak = max(diagnostic_peak, broad_peak)
        broad_minimum = max(
            minimum_contrast,
            float(
                paper_cfg.get(
                    "divider_broad_min_contrast_lab",
                    minimum_contrast * 1.35,
                )
            ),
        )
        broad_selected = select_wide_peak(
            broad_scores, broad_start, broad_minimum
        )
        if broad_selected is not None:
            start, stop = broad_start, broad_stop
            scores = broad_scores
            peak_offset, peak_score = broad_selected
            accepted = True

    if not accepted:
        raise DetectionError(
            "A4 sheet was found, but the required horizontal divider was not. "
            f"No full-width bright/dark stroke passed validation "
            f"(peak supported Lab contrast {diagnostic_peak:.1f})."
        )

    band_threshold = max(minimum_contrast, peak_score * 0.45)
    left = peak_offset
    right = peak_offset
    while left > 0 and scores[left - 1] >= band_threshold:
        left -= 1
    while right + 1 < len(scores) and scores[right + 1] >= band_threshold:
        right += 1
    detected_width_px = right - left + 1
    # Marker strokes, perspective blur and automatic exposure can make a
    # valid 5 mm divider measure slightly wider.  Width is diagnostic only;
    # the task setup guarantees this is the divider, so do not reject an
    # otherwise strong mid-sheet line.

    band_scores = scores[left : right + 1].astype(np.float64)
    band_rows = np.arange(start + left, start + right + 1, dtype=np.float64)
    centre_px = float(np.average(band_rows, weights=band_scores))
    return (
        centre_px / ppm,
        detected_width_px / ppm,
        peak_score,
    )


def _dominant_lab_background(
    image: np.ndarray,
    valid_roi: np.ndarray,
    lab_image: np.ndarray | None = None,
) -> np.ndarray:
    lab = (
        np.asarray(lab_image, dtype=np.float32)
        if lab_image is not None
        else cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    )
    samples = lab[valid_roi > 0]
    if len(samples) < 100:
        raise DetectionError("Not enough background pixels to estimate A4 colour")
    # All pieces together occupy less than half of the source half-sheet, so
    # the per-channel median is guaranteed to represent paper.  A quantised
    # histogram mode is not safe under overexposure: four uniformly white
    # pieces may fall into one bin while a lighting gradient spreads the paper
    # across several bins.
    return np.median(samples, axis=0).astype(np.float32)


def _source_roi_mask(
    paper: PaperView,
    segmentation_cfg: dict[str, Any],
    source_region: str,
) -> np.ndarray:
    image = paper.image
    ppm = paper.pixels_per_mm
    divider_px = int(round(paper.divider_y_mm * ppm))
    margin = int(round(float(segmentation_cfg["paper_margin_mm"]) * ppm))
    divider_margin = int(
        round(float(segmentation_cfg["divider_margin_mm"]) * ppm)
    )
    roi = np.zeros(image.shape[:2], dtype=np.uint8)
    if source_region == "upper":
        top = margin
        bottom = divider_px - divider_margin
    elif source_region == "lower":
        top = divider_px + divider_margin
        bottom = image.shape[0] - margin - 1
    else:
        raise ValueError(f"Unsupported source region: {source_region}")
    cv2.rectangle(
        roi,
        (margin, top),
        (image.shape[1] - margin - 1, bottom),
        255,
        -1,
    )
    return roi


def _remove_roi_border_components(
    mask: np.ndarray,
    roi: np.ndarray,
    pixels_per_mm: float,
    segmentation_cfg: dict[str, Any],
    lab_image: np.ndarray | None = None,
) -> np.ndarray:
    """Remove warped table strips without dropping a real edge-near piece.

    A long contact with the ROI is suspicious, but is not sufficient on its
    own: a randomly placed real fragment may touch the A4 edge.  When one or
    more independent foreground components exist, use their median Lab colour
    as the same-colour piece reference and retain matching border components.
    """

    if not bool(
        segmentation_cfg.get("reject_roi_border_components", True)
    ):
        return mask
    binary = np.where((mask > 0) & (roi > 0), 255, 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    if count <= 1:
        return binary
    inner = cv2.erode(roi, np.ones((3, 3), dtype=np.uint8))
    boundary = cv2.subtract(roi, inner) > 0
    boundary_labels = labels[boundary]
    contact_threshold = max(
        3,
        int(
            round(
                float(
                    segmentation_cfg.get(
                        "roi_border_component_min_contact_mm", 8.0
                    )
                )
                * float(pixels_per_mm)
            )
        ),
    )
    touching_labels, touching_counts = np.unique(
        boundary_labels[boundary_labels > 0], return_counts=True
    )
    touching = {
        int(label): int(contacts)
        for label, contacts in zip(touching_labels, touching_counts)
        if int(contacts) >= contact_threshold
    }
    if not touching:
        return binary

    areas = np.bincount(labels.reshape(-1), minlength=count)
    minimum_area = float(
        segmentation_cfg.get("min_area_mm2", 25.0)
    ) * float(pixels_per_mm) ** 2
    maximum_area = float(
        segmentation_cfg.get("max_area_mm2", 10000.0)
    ) * float(pixels_per_mm) ** 2
    reference_colours: list[np.ndarray] = []
    if lab_image is not None:
        lab = np.asarray(lab_image, dtype=np.float32)
        for label in range(1, count):
            if label in touching:
                continue
            if not minimum_area <= float(areas[label]) <= maximum_area:
                continue
            samples = lab[labels == label]
            if len(samples) >= 20:
                reference_colours.append(
                    np.median(samples, axis=0).astype(np.float32)
                )

    colour_tolerance = float(
        segmentation_cfg.get(
            "roi_border_piece_colour_tolerance_lab", 18.0
        )
    )
    no_reference_contact = max(
        contact_threshold,
        int(
            round(
                float(
                    segmentation_cfg.get(
                        "roi_border_no_reference_contact_mm", 18.0
                    )
                )
                * float(pixels_per_mm)
            )
        ),
    )
    cleaned = binary.copy()
    for label, contacts in touching.items():
        preserve_as_piece = False
        if (
            reference_colours
            and minimum_area <= float(areas[label]) <= maximum_area
            and lab_image is not None
        ):
            observed = np.median(
                np.asarray(lab_image, dtype=np.float32)[labels == label],
                axis=0,
            )
            preserve_as_piece = (
                min(
                    float(np.linalg.norm(observed - reference))
                    for reference in reference_colours
                )
                <= colour_tolerance
            )
        elif contacts < no_reference_contact:
            # With no interior colour reference, only remove an unmistakably
            # long border strip; keep a short edge contact to avoid a miss.
            preserve_as_piece = True
        if not preserve_as_piece:
            cleaned[labels == label] = 0
    return cleaned


def foreground_mask(
    paper: PaperView,
    paper_cfg: dict[str, Any],
    segmentation_cfg: dict[str, Any],
    background_rectified: np.ndarray | None = None,
    source_region: str = "upper",
    colour_mode: str = "combined",
    current_lab: np.ndarray | None = None,
    estimated_background_lab: np.ndarray | None = None,
) -> np.ndarray:
    image = paper.image
    ppm = paper.pixels_per_mm
    roi = _source_roi_mask(
        paper, segmentation_cfg, source_region
    )

    current_lab = (
        np.asarray(current_lab, dtype=np.float32)
        if current_lab is not None
        else cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    )
    piece_colour = segmentation_cfg.get("piece_color_bgr")
    piece_distance: np.ndarray | None = None
    piece_chroma_distance: np.ndarray | None = None
    hint_relative_foreground: np.ndarray | None = None
    if piece_colour is not None:
        colour_pixel = np.asarray(piece_colour, dtype=np.uint8).reshape(1, 1, 3)
        piece_lab = cv2.cvtColor(
            colour_pixel, cv2.COLOR_BGR2LAB
        ).reshape(3).astype(np.float32)
        piece_distance = np.linalg.norm(current_lab - piece_lab, axis=2)
        piece_chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - piece_lab[1:], axis=2
        )
    if background_rectified is not None:
        if background_rectified.shape[:2] != image.shape[:2]:
            background_rectified = cv2.resize(
                background_rectified,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        reference_lab = cv2.cvtColor(
            background_rectified, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
        distance = np.linalg.norm(current_lab - reference_lab, axis=2)
        chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - reference_lab[:, :, 1:], axis=2
        )
        lightness_delta = current_lab[:, :, 0] - reference_lab[:, :, 0]
        background = np.median(
            reference_lab[roi > 0], axis=0
        ).astype(np.float32)
        threshold = float(
            segmentation_cfg["background_difference_threshold"]
        )
    else:
        sampling_roi = roi.copy()
        # Exclude a thin band near the divider and retain a broad region so
        # the A4 colour remains the modal colour even with four pieces.
        background = (
            np.asarray(estimated_background_lab, dtype=np.float32)
            if estimated_background_lab is not None
            else _dominant_lab_background(
                image, sampling_roi, current_lab
            )
        )
        distance = np.linalg.norm(current_lab - background, axis=2)
        chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - background[1:], axis=2
        )
        lightness_delta = current_lab[:, :, 0] - background[0]
        threshold = float(segmentation_cfg["lab_distance_threshold"])

    # Convert the two user-selected colours into a relative colour offset.
    # Applying that offset to the observed paper colour compensates for camera
    # exposure and white-balance changes, so the colour controls remain useful
    # in dim or warm lighting instead of requiring camera-exact RGB values.
    paper_colour = paper_cfg.get("color_bgr")
    if piece_colour is not None and paper_colour is not None:
        paper_pixel = np.asarray(
            paper_colour, dtype=np.uint8
        ).reshape(1, 1, 3)
        expected_paper_lab = cv2.cvtColor(
            paper_pixel, cv2.COLOR_BGR2LAB
        ).reshape(3).astype(np.float32)
        colour_delta = piece_lab - expected_paper_lab
        colour_delta = colour_delta.copy()
        colour_delta[0] *= 0.70
        adapted_piece_lab = np.clip(
            background + colour_delta, 0.0, 255.0
        )
        adapted_distance = np.linalg.norm(
            current_lab - adapted_piece_lab, axis=2
        )
        adapted_chroma_distance = np.linalg.norm(
            current_lab[:, :, 1:] - adapted_piece_lab[1:], axis=2
        )
        piece_distance = np.minimum(
            piece_distance, adapted_distance
        )
        piece_chroma_distance = np.minimum(
            piece_chroma_distance, adapted_chroma_distance
        )
        relative_masks: list[np.ndarray] = []
        light_threshold = float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
        if colour_delta[0] >= 8.0:
            relative_masks.append(lightness_delta >= light_threshold)
        elif colour_delta[0] <= -8.0:
            relative_masks.append(lightness_delta <= -light_threshold)
        if float(np.linalg.norm(colour_delta[1:])) >= 8.0:
            relative_masks.append(
                adapted_chroma_distance
                <= max(
                    18.0,
                    float(
                        segmentation_cfg.get(
                            "piece_color_tolerance_chroma", 12.0
                        )
                    ),
                )
            )
        if relative_masks:
            hint_relative_foreground = np.logical_or.reduce(
                relative_masks
            )

    chroma_threshold = float(
        segmentation_cfg.get("lab_chroma_threshold", float("inf"))
    )
    if colour_mode == "hint":
        if piece_distance is None or piece_chroma_distance is None:
            foreground = np.zeros(image.shape[:2], dtype=bool)
        else:
            # Camera exposure changes L much more than a/b chroma.  A chroma
            # hint therefore remains useful for the same coloured paper under
            # different heights and lights.  Retain a broad lightness guard so
            # neutral hints such as white do not select every grey object.
            full_match = piece_distance <= float(
                segmentation_cfg.get("piece_color_tolerance_lab", 32.0)
            )
            chroma_match = piece_chroma_distance <= float(
                segmentation_cfg.get("piece_color_tolerance_chroma", 12.0)
            )
            foreground = full_match | (
                chroma_match
                & (
                    np.abs(current_lab[:, :, 0] - piece_lab[0])
                    <= 100.0
                )
            )
            if hint_relative_foreground is not None:
                foreground |= hint_relative_foreground
    elif colour_mode == "full":
        foreground = distance >= threshold
    elif colour_mode == "light":
        foreground = lightness_delta >= float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
    elif colour_mode == "dark":
        foreground = lightness_delta <= -float(
            segmentation_cfg.get("lab_lightness_threshold", 16.0)
        )
    elif colour_mode == "chroma":
        foreground = chroma_distance >= chroma_threshold
    elif colour_mode == "combined":
        foreground = (distance >= threshold) | (
            chroma_distance >= chroma_threshold
        )
    else:
        raise ValueError(f"Unsupported colour segmentation mode: {colour_mode}")
    mask = (foreground & (roi > 0)).astype(np.uint8) * 255
    open_size = max(
        1, int(round(float(segmentation_cfg["morph_open_mm"]) * ppm))
    )
    close_size = max(
        1, int(round(float(segmentation_cfg["morph_close_mm"]) * ppm))
    )
    if open_size % 2 == 0:
        open_size += 1
    if close_size % 2 == 0:
        close_size += 1
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_size, open_size)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask


def _approximate_polygon(
    contour: np.ndarray,
    ppm: float,
    segmentation_cfg: dict[str, Any],
) -> np.ndarray:
    def remove_nearly_collinear(points: np.ndarray) -> np.ndarray:
        cleaned = points.copy()
        minimum_turn = np.deg2rad(
            float(segmentation_cfg.get("minimum_corner_turn_deg", 0.0))
        )
        maximum_short_edge = float(
            segmentation_cfg.get("collinear_short_edge_mm", float("inf"))
        )
        while len(cleaned) > minimum and minimum_turn > 0:
            turns: list[float] = []
            for index in range(len(cleaned)):
                incoming = cleaned[index] - cleaned[(index - 1) % len(cleaned)]
                outgoing = cleaned[(index + 1) % len(cleaned)] - cleaned[index]
                if min(
                    float(np.linalg.norm(incoming)),
                    float(np.linalg.norm(outgoing)),
                ) / ppm > maximum_short_edge:
                    turns.append(float("inf"))
                    continue
                denominator = max(
                    float(np.linalg.norm(incoming) * np.linalg.norm(outgoing)),
                    1e-9,
                )
                cosine = float(
                    np.clip(np.dot(incoming, outgoing) / denominator, -1.0, 1.0)
                )
                turns.append(float(np.arccos(cosine)))
            index = int(np.argmin(turns))
            if turns[index] >= minimum_turn:
                break
            cleaned = np.delete(cleaned, index, axis=0)
        return cleaned

    def remove_short_edges(points: np.ndarray) -> np.ndarray:
        cleaned = points.copy()
        while len(cleaned) > minimum:
            lengths = np.linalg.norm(
                np.roll(cleaned, -1, axis=0) - cleaned, axis=1
            ) / ppm
            edge_index = int(np.argmin(lengths))
            if float(lengths[edge_index]) >= minimum_edge:
                break
            first = edge_index
            second = (edge_index + 1) % len(cleaned)

            def corner_area(index: int) -> float:
                previous = cleaned[(index - 1) % len(cleaned)]
                current = cleaned[index]
                following = cleaned[(index + 1) % len(cleaned)]
                return abs(float(np.cross(current - previous, following - current)))

            remove_index = (
                first if corner_area(first) <= corner_area(second) else second
            )
            cleaned = np.delete(cleaned, remove_index, axis=0)
        return cleaned

    minimum = int(segmentation_cfg["expected_min_vertices"])
    maximum = int(segmentation_cfg["expected_max_vertices"])
    epsilon = float(segmentation_cfg["polygon_epsilon_mm"])
    epsilon_max = float(segmentation_cfg["max_polygon_epsilon_mm"])
    minimum_edge = float(segmentation_cfg.get("minimum_detected_edge_mm", 0.0))
    approximation_contour = (
        cv2.convexHull(contour)
        if bool(segmentation_cfg.get("assume_convex_pieces", False))
        else contour
    )
    while epsilon <= epsilon_max + 1e-6:
        approx = cv2.approxPolyDP(
            approximation_contour, epsilon * ppm, True
        ).reshape(-1, 2)
        if minimum <= len(approx) <= maximum:
            approx = remove_nearly_collinear(approx)
            approx = remove_short_edges(approx)
        lengths_mm = (
            np.linalg.norm(np.roll(approx, -1, axis=0) - approx, axis=1) / ppm
            if len(approx) >= 2
            else np.array([], dtype=np.float64)
        )
        if (
            minimum <= len(approx) <= maximum
            and len(lengths_mm)
            and float(np.min(lengths_mm)) >= minimum_edge
        ):
            return approx
        epsilon += 0.25
    raise DetectionError(
        f"Contour cannot be approximated as a {minimum}-to-{maximum}-vertex polygon"
    )


def detect_pieces(
    paper: PaperView,
    paper_cfg: dict[str, Any],
    segmentation_cfg: dict[str, Any],
    background_rectified: np.ndarray | None = None,
    source_region: str = "upper",
) -> tuple[list[PieceObservation], np.ndarray, str, str]:
    if source_region not in ("upper", "lower", "auto"):
        raise ValueError(f"Unsupported source region: {source_region}")
    regions = ("upper", "lower") if source_region == "auto" else (source_region,)
    required = int(segmentation_cfg.get("required_pieces", 0))
    candidates: list[
        tuple[list[PieceObservation], np.ndarray, str, str, float, float]
    ] = []
    current_lab = cv2.cvtColor(
        paper.image, cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    for region in regions:
        estimated_background = None
        if background_rectified is None:
            estimated_background = _dominant_lab_background(
                paper.image,
                _source_roi_mask(paper, segmentation_cfg, region),
                current_lab,
            )
        colour_modes = (
            ("hint", "light", "dark", "chroma", "full", "combined")
            if segmentation_cfg.get("piece_color_bgr") is not None
            else ("light", "dark", "chroma", "full", "combined")
        )
        for colour_mode in colour_modes:
            mask = foreground_mask(
                paper,
                paper_cfg,
                segmentation_cfg,
                background_rectified,
                region,
                colour_mode,
                current_lab=current_lab,
                estimated_background_lab=estimated_background,
            )
            mask = _remove_roi_border_components(
                mask,
                _source_roi_mask(paper, segmentation_cfg, region),
                paper.pixels_per_mm,
                segmentation_cfg,
                current_lab,
            )
            observations = _observations_from_mask(
                paper, segmentation_cfg, mask
            )
            straightness_error = sum(
                abs(polygon_area(item.polygon_mm) - item.area_mm2)
                / max(item.area_mm2, 1.0)
                for item in observations
            )
            paper_margin = float(segmentation_cfg["paper_margin_mm"])
            divider_margin = float(
                segmentation_cfg["divider_margin_mm"]
            )
            if region == "upper":
                bounds = (
                    paper_margin,
                    paper_margin,
                    paper.width_mm - paper_margin,
                    paper.divider_y_mm - divider_margin,
                )
            else:
                bounds = (
                    paper_margin,
                    paper.divider_y_mm + divider_margin,
                    paper.width_mm - paper_margin,
                    paper.height_mm - paper_margin,
                )
            # A contour touching the segmentation ROI is normally a bright
            # paper edge/illumination stripe, not a freely placed fragment.
            # Penalise it instead of blindly preferring a four-component mask.
            boundary_penalty = 0.0
            clearance = 1.0
            for observation in observations:
                polygon = observation.polygon_mm
                distances = (
                    float(np.min(polygon[:, 0]) - bounds[0]),
                    float(np.min(polygon[:, 1]) - bounds[1]),
                    float(bounds[2] - np.max(polygon[:, 0])),
                    float(bounds[3] - np.max(polygon[:, 1])),
                )
                boundary_penalty += sum(
                    max(0.0, clearance - distance)
                    for distance in distances
                )
            # Inspect the mask itself as well as its simplified polygons.
            # A bright strip just outside a slightly imperfect A4 homography
            # can merge with a real piece and then simplify to vertices a few
            # millimetres inside the ROI.  The component still touches the ROI
            # boundary in the binary mask, which is a reliable way to reject
            # this false "fourth piece" while retaining nearby free pieces.
            band = max(2, int(round(0.5 * paper.pixels_per_mm)))
            left_px = max(
                0, int(round(bounds[0] * paper.pixels_per_mm))
            )
            top_px = max(
                0, int(round(bounds[1] * paper.pixels_per_mm))
            )
            right_px = min(
                mask.shape[1] - 1,
                int(round(bounds[2] * paper.pixels_per_mm)),
            )
            bottom_px = min(
                mask.shape[0] - 1,
                int(round(bounds[3] * paper.pixels_per_mm)),
            )
            boundary_pixels = (
                cv2.countNonZero(
                    mask[
                        top_px : min(bottom_px + 1, top_px + band),
                        left_px : right_px + 1,
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        max(top_px, bottom_px - band + 1) : bottom_px + 1,
                        left_px : right_px + 1,
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        top_px : bottom_px + 1,
                        left_px : min(right_px + 1, left_px + band),
                    ]
                )
                + cv2.countNonZero(
                    mask[
                        top_px : bottom_px + 1,
                        max(left_px, right_px - band + 1) : right_px + 1,
                    ]
                )
            )
            boundary_penalty += min(
                20.0,
                float(boundary_pixels)
                / max(1.0, 2.0 * paper.pixels_per_mm),
            )
            candidates.append(
                (
                    observations,
                    mask,
                    region,
                    colour_mode,
                    straightness_error,
                    boundary_penalty,
                )
            )
            # Four valid straight-edged components are the maximum allowed by
            # the task.  Once a clean mask reaches that maximum, evaluating
            # the remaining colour modes can only duplicate the same answer.
            # This removes most of the repeated morphology/contour work on
            # RDK/K230 while preserving the full fallback sweep for 2/3-piece
            # scenes or an incomplete colour hint.
            maximum_pieces = int(
                segmentation_cfg.get("max_pieces", 4)
            )
            expected_hint_area = float(
                segmentation_cfg.get("expected_total_area_mm2", 0.0)
            )
            hint_area_ratio = (
                sum(item.area_mm2 for item in observations)
                / expected_hint_area
                if expected_hint_area > 0.0
                else 1.0
            )
            structural_card_hint = (
                expected_hint_area > 0.0
                and int(
                    segmentation_cfg.get("expected_max_vertices", 99)
                )
                <= 5
                and 0.90 <= hint_area_ratio <= 1.10
            )
            if (
                colour_mode == "hint"
                and
                (expected_hint_area <= 0.0 or structural_card_hint)
                and
                len(observations) == maximum_pieces
                and (not required or len(observations) == required)
                and straightness_error
                <= (
                    0.16
                    if structural_card_hint
                    else 0.22
                )
                and boundary_penalty <= 1.0
            ):
                return observations, mask, region, colour_mode

    expected_total_area = float(
        segmentation_cfg.get("expected_total_area_mm2", 0.0)
    )
    expected_min_ratio = float(
        segmentation_cfg.get("expected_total_area_min_ratio", 0.0)
    )
    expected_max_ratio = float(
        segmentation_cfg.get(
            "expected_total_area_max_ratio", float("inf")
        )
    )

    def candidate_area_quality(
        candidate: tuple[
            list[PieceObservation], np.ndarray, str, str, float, float
        ],
    ) -> tuple[bool, float, float]:
        if expected_total_area <= 0.0:
            return True, 0.0, 0.0
        total_area = sum(item.area_mm2 for item in candidate[0])
        ratio = total_area / expected_total_area
        error = abs(ratio - 1.0)
        # Masks produced from different colour cues naturally differ by a
        # narrow anti-aliased boundary band.  Once their total areas are all
        # within that band, exact area is weaker evidence than the task prior
        # that every fragment has three-to-five straight structural edges.
        # Treat close areas as tied and prefer the cleaner polygon fit.  This
        # prevents printed suits touching a cut edge from winning solely
        # because their bloated mask happens to be 1% nearer the nominal card
        # area.
        tie_ratio = float(
            segmentation_cfg.get("card_candidate_area_tie_ratio", 0.04)
        )
        return (
            expected_min_ratio <= ratio <= expected_max_ratio,
            -max(0.0, error - tie_ratio),
            -error,
        )

    observations, mask, detected_region, selected_mode, _, _ = max(
        candidates,
        key=lambda item: (
            not required or len(item[0]) == required,
            bool(item[0]),
            # In playing-card mode the total physical card area is known.
            # Evaluate that evidence before component count: printed ranks and
            # suits can otherwise masquerade as four tiny "pieces" and beat a
            # correct two/three-piece white-card mask.
            (
                candidate_area_quality(item)[0]
                if expected_total_area > 0.0
                else True
            ),
            (
                candidate_area_quality(item)[1]
                if expected_total_area > 0.0
                else 0.0
            ),
            len(item[0]) if not required else 0,
            -item[5],
            item[3] == "hint",
            -item[4],
            (
                candidate_area_quality(item)[2]
                if expected_total_area > 0.0
                else 0.0
            ),
            item[3] in ("light", "dark"),
            item[3] == "chroma",
            item[2] == "upper",
        ),
    )
    if not observations:
        requested = "either half" if source_region == "auto" else source_region
        raise DetectionError(
            f"No puzzle pieces were detected in the {requested} of the A4 sheet."
        )
    if required and len(observations) != required:
        raise DetectionError(
            f"Exactly {required} separate straight-edged pieces are required, "
            f"but {len(observations)} valid pieces were detected."
        )
    return observations, mask, detected_region, selected_mode


def _observations_from_mask(
    paper: PaperView,
    segmentation_cfg: dict[str, Any],
    mask: np.ndarray,
) -> list[PieceObservation]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ppm = paper.pixels_per_mm
    minimum_area = float(segmentation_cfg["min_area_mm2"]) * ppm * ppm
    maximum_area = float(segmentation_cfg["max_area_mm2"]) * ppm * ppm
    contours = [
        c
        for c in contours
        if minimum_area <= abs(cv2.contourArea(c)) <= maximum_area
    ]
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    observations: list[PieceObservation] = []
    for contour in contours:
        if len(observations) >= int(segmentation_cfg["max_pieces"]):
            break
        try:
            approx_px = _approximate_polygon(contour, ppm, segmentation_cfg)
        except DetectionError:
            continue
        polygon_mm = normalize_winding(approx_px.astype(np.float64) / ppm)
        # Preserve the measured silhouette area for rectangle-fill scoring.
        # Approximation vertices are used for edge matching, but using their
        # area would under-count wrinkled or shadowed paper corners.
        area = abs(float(cv2.contourArea(contour))) / (ppm * ppm)
        centroid = polygon_centroid(polygon_mm)
        pickup = safe_interior_point(polygon_mm)
        lengths = edge_lengths(polygon_mm)
        observations.append(
            PieceObservation(
                id=f"piece_{len(observations) + 1}",
                polygon_mm=polygon_mm,
                contour_px=contour,
                centroid_mm=centroid,
                pickup_mm=pickup,
                area_mm2=area,
                perimeter_mm=float(np.sum(lengths)),
                edge_lengths_mm=lengths,
            )
        )
    # Stable IDs should not depend on contour area: order by pickup y then x.
    observations.sort(key=lambda item: (item.pickup_mm[1], item.pickup_mm[0]))
    for index, observation in enumerate(observations, start=1):
        observation.id = f"piece_{index}"
    return observations
