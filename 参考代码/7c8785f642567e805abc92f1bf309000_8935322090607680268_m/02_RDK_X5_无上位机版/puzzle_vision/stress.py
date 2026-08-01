from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

import numpy as np

from .detector import PieceObservation
from .geometry import (
    edge_lengths,
    normalize_winding,
    polygon_area,
    polygon_centroid,
    rotation_matrix_row,
    safe_interior_point,
    transform_points,
)
from .solver import SolveError, solve_unknown


def _valid_edges(polygons: list[np.ndarray], minimum_edge_mm: float) -> bool:
    return all(
        float(np.min(edge_lengths(polygon))) >= minimum_edge_mm
        for polygon in polygons
    )


def random_rectangle_quadrilaterals(
    rng: np.random.Generator,
    minimum_edge_mm: float = 20.0,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Cut one legal target rectangle into four random quadrilaterals.

    A random interior junction is connected to one random point on every
    rectangle side.  The four resulting polygons tile the rectangle exactly
    and every piece has one or two known outer-boundary edges.
    """

    for _ in range(5000):
        width = float(rng.uniform(92.0, 119.0))
        height = float(rng.uniform(55.0, 89.0))
        top = np.array([rng.uniform(21.0, width - 21.0), 0.0])
        right = np.array([width, rng.uniform(21.0, height - 21.0)])
        bottom = np.array([rng.uniform(21.0, width - 21.0), height])
        left = np.array([0.0, rng.uniform(21.0, height - 21.0)])
        centre = np.array(
            [
                rng.uniform(0.34 * width, 0.66 * width),
                rng.uniform(0.34 * height, 0.66 * height),
            ]
        )
        polygons = [
            np.array([[0.0, 0.0], top, centre, left]),
            np.array([top, [width, 0.0], right, centre]),
            np.array([centre, right, [width, height], bottom]),
            np.array([left, centre, bottom, [0.0, height]]),
        ]
        polygons = [normalize_winding(polygon) for polygon in polygons]
        if _valid_edges(polygons, minimum_edge_mm):
            return polygons, (width, height)
    raise RuntimeError("Cannot generate a legal four-quadrilateral rectangle")


def random_rectangle_strip_quadrilaterals(
    rng: np.random.Generator,
    piece_count: int,
    minimum_edge_mm: float = 20.0,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Tile a rectangle into two or three irregular quadrilateral strips."""

    if piece_count not in (2, 3):
        raise ValueError("strip generator supports two or three pieces")
    for _ in range(1000):
        minimum_width = piece_count * minimum_edge_mm + 5.0
        width = float(rng.uniform(max(92.0, minimum_width), 119.0))
        height = float(rng.uniform(55.0, 89.0))
        remaining = width - piece_count * minimum_edge_mm
        top_segments = (
            minimum_edge_mm
            + remaining * rng.dirichlet(np.ones(piece_count))
        )
        bottom_segments = (
            minimum_edge_mm
            + remaining * rng.dirichlet(np.ones(piece_count))
        )
        top_x = np.concatenate([[0.0], np.cumsum(top_segments)])
        bottom_x = np.concatenate([[0.0], np.cumsum(bottom_segments)])
        top_x[-1] = width
        bottom_x[-1] = width
        polygons = [
            normalize_winding(
                np.asarray(
                    [
                        [top_x[index], 0.0],
                        [top_x[index + 1], 0.0],
                        [bottom_x[index + 1], height],
                        [bottom_x[index], height],
                    ],
                    dtype=np.float64,
                )
            )
            for index in range(piece_count)
        ]
        if _valid_edges(polygons, minimum_edge_mm):
            return polygons, (width, height)
    raise RuntimeError("Cannot generate legal quadrilateral strips")


def random_rectangle_tiling(
    rng: np.random.Generator,
    piece_count: int,
    minimum_edge_mm: float = 20.0,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    if piece_count == 4:
        return random_rectangle_quadrilaterals(
            rng,
            minimum_edge_mm=minimum_edge_mm,
        )
    return random_rectangle_strip_quadrilaterals(
        rng,
        piece_count=piece_count,
        minimum_edge_mm=minimum_edge_mm,
    )


def scramble_polygons(
    templates: list[np.ndarray],
    rng: np.random.Generator,
    corner_noise_mm: float,
) -> list[PieceObservation]:
    """Apply proper rotations/translations and independent camera-like noise."""

    observations: list[PieceObservation] = []
    centres = (
        np.array([35.0, 35.0]),
        np.array([115.0, 35.0]),
        np.array([35.0, 105.0]),
        np.array([135.0, 105.0]),
    )
    for index, template in enumerate(templates):
        local = template - polygon_centroid(template)
        angle = float(rng.uniform(-math.pi, math.pi))
        polygon = transform_points(
            local,
            rotation_matrix_row(angle),
            centres[index] + rng.normal(0.0, 3.0, 2),
        )
        if corner_noise_mm > 0:
            polygon = polygon + rng.normal(
                0.0, corner_noise_mm, polygon.shape
            )
        polygon = normalize_winding(polygon)
        lengths = edge_lengths(polygon)
        true_area = polygon_area(template)
        observations.append(
            PieceObservation(
                id=f"piece_{index + 1}",
                polygon_mm=polygon,
                contour_px=np.rint(polygon * 4.0)
                .astype(np.int32)
                .reshape(-1, 1, 2),
                centroid_mm=polygon_centroid(polygon),
                pickup_mm=safe_interior_point(polygon),
                area_mm2=true_area,
                perimeter_mm=float(np.sum(lengths)),
                edge_lengths_mm=lengths,
            )
        )
    return observations


def run_solver_stress_test(
    config: dict[str, Any],
    cases: int = 50,
    seed: int = 20260729,
    corner_noise_mm: float = 0.8,
    piece_counts: tuple[int, ...] = (4,),
) -> dict[str, Any]:
    counts = tuple(
        value for value in piece_counts if value in (2, 3, 4)
    )
    if not counts:
        raise ValueError("piece_counts must contain 2, 3, or 4")
    rng = np.random.default_rng(seed)
    unknown = deepcopy(config["unknown"])
    records: list[dict[str, Any]] = []
    success = 0
    accepted = 0
    mirrored = 0
    for case_index in range(cases):
        piece_count = counts[case_index % len(counts)]
        templates, expected_size = random_rectangle_tiling(
            rng,
            piece_count=piece_count,
        )
        observations = scramble_polygons(templates, rng, corner_noise_mm)
        started = time.perf_counter()
        try:
            plan, info = solve_unknown(
                observations,
                unknown,
                rectified_image=None,
                pixels_per_mm=4.0,
                use_texture=False,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            success += 1
            is_accepted = bool(info.get("solution_accepted"))
            accepted += int(is_accepted)
            reflected = any(bool(item.get("mirrored")) for item in plan)
            mirrored += int(reflected)
            records.append(
                {
                    "case": case_index,
                    "piece_count": piece_count,
                    "ok": True,
                    "accepted": is_accepted,
                    "expected_size_mm": np.round(expected_size, 3).tolist(),
                    "target_size_mm": info.get("target_size_mm"),
                    "fill_ratio": info.get("fill_ratio"),
                    "geometry_score": info.get("geometry_score"),
                    "assembly_gap_mm2": info.get("assembly_gap_mm2"),
                    "assembly_overlap_mm2": info.get(
                        "assembly_overlap_mm2"
                    ),
                    "assembly_gap_ratio": info.get("assembly_gap_ratio"),
                    "assembly_overlap_ratio": info.get(
                        "assembly_overlap_ratio"
                    ),
                    "maximum_target_overlap_mm2": info.get(
                        "maximum_target_overlap_mm2"
                    ),
                    "search_nodes": info.get("search_nodes"),
                    "search_timed_out": info.get("search_timed_out"),
                    "elapsed_ms": round(elapsed_ms, 2),
                    "mirrored": reflected,
                }
            )
        except SolveError as exc:
            records.append(
                {
                    "case": case_index,
                    "piece_count": piece_count,
                    "ok": False,
                    "accepted": False,
                    "expected_size_mm": np.round(expected_size, 3).tolist(),
                    "elapsed_ms": round(
                        (time.perf_counter() - started) * 1000.0, 2
                    ),
                    "error": str(exc),
                }
            )
    elapsed = [float(item["elapsed_ms"]) for item in records]
    return {
        "ok": accepted == cases and mirrored == 0,
        "cases": cases,
        "seed": seed,
        "corner_noise_mm": corner_noise_mm,
        "piece_counts": list(counts),
        "solver_return_rate": round(success / max(cases, 1), 4),
        "accepted_rate": round(accepted / max(cases, 1), 4),
        "mirrored_solutions": mirrored,
        "mean_elapsed_ms": round(float(np.mean(elapsed)), 2),
        "maximum_elapsed_ms": round(float(np.max(elapsed)), 2),
        "failures": [item for item in records if not item["accepted"]],
        "records": records,
    }
