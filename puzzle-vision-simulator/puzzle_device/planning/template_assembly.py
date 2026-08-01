"""Fixed four-piece template matching for the self-prepared puzzle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
import json
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "configs/self_piece_template.json"


@dataclass(frozen=True)
class SelfPieceTemplate:
    target_size_mm: tuple[float, float]
    pieces: tuple[np.ndarray, ...]
    maximum_total_normalized_error: float
    maximum_piece_normalized_error: float


@dataclass(frozen=True)
class TemplateAssemblyResult:
    transforms: tuple[np.ndarray, ...]
    assignment: tuple[int, ...]
    piece_errors: tuple[float, ...]
    score: float


def load_self_piece_template(path: Path = DEFAULT_TEMPLATE_PATH) -> SelfPieceTemplate:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("format") != "puzzle-device.self-piece-template.v1":
        raise ValueError("unsupported self-piece template format")
    target_size = tuple(float(value) for value in document["target_size_mm"])
    pieces = tuple(
        np.asarray(item["vertices_mm"], dtype=np.float64)
        for item in document["pieces"]
    )
    if len(target_size) != 2 or len(pieces) != 4:
        raise ValueError("self-piece template requires a target size and four pieces")
    if any(len(piece) < 3 for piece in pieces):
        raise ValueError("every self-piece template polygon needs at least three vertices")
    return SelfPieceTemplate(
        target_size_mm=target_size,
        pieces=pieces,
        maximum_total_normalized_error=float(
            document["maximum_total_normalized_error"]
        ),
        maximum_piece_normalized_error=float(
            document["maximum_piece_normalized_error"]
        ),
    )


def _ensure_counterclockwise(polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    if cv2.contourArea(polygon.astype(np.float32), oriented=True) < 0:
        return polygon[::-1].copy()
    return polygon.copy()


def _candidate_vertex_sets(observed: np.ndarray, count: int):
    observed = _ensure_counterclockwise(observed)
    if len(observed) < count:
        return
    for indices in combinations(range(len(observed)), count):
        yield observed[list(indices)]


def _fit_similarity(template: np.ndarray, observed: np.ndarray):
    template_center = template.mean(axis=0)
    observed_center = observed.mean(axis=0)
    source = template - template_center
    target = observed - observed_center
    u, singular_values, vt = np.linalg.svd(source.T @ target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        return None
    denominator = float(np.sum(source * source))
    if denominator <= 1e-9:
        return None
    scale = float(np.sum(singular_values) / denominator)
    translation = observed_center - scale * (rotation @ template_center)
    predicted = (scale * (rotation @ template.T)).T + translation
    rmse = float(np.sqrt(np.mean(np.sum((predicted - observed) ** 2, axis=1))))
    return scale, rotation, translation, rmse, template_center, observed_center


def _fit_observation_to_template(observed: np.ndarray, template: np.ndarray):
    template = _ensure_counterclockwise(template)
    best = None
    for vertices in _candidate_vertex_sets(observed, len(template)):
        for shift in range(len(vertices)):
            ordered = np.roll(vertices, -shift, axis=0)
            fit = _fit_similarity(template, ordered)
            if fit is None:
                continue
            scale, rotation, translation, rmse, template_center, observed_center = fit
            # A4 pixel-to-mm scale and contour erosion introduce small size error,
            # but a gross scale mismatch indicates a wrong piece assignment.
            scale_penalty = abs(math.log(max(scale, 1e-9)))
            normalized_rmse = rmse / max(math.sqrt(abs(float(
                cv2.contourArea(ordered.astype(np.float32))
            ))), 1.0)
            error = normalized_rmse + 0.30 * scale_penalty
            if best is None or error < best[0]:
                best = (
                    error, scale, rotation, translation,
                    template_center, observed_center,
                )
    return best


def _source_to_target_rigid(
    rotation: np.ndarray,
    template_center: np.ndarray,
    observed_center: np.ndarray,
) -> np.ndarray:
    # Similarity was fitted as observed = scale * R * template + t. Invert it,
    # but intentionally omit scale so the gantry only rotates and translates.
    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation.T
    transform[:2, 2] = template_center - rotation.T @ observed_center
    return transform


def solve_fixed_template(
    polygons_a4: list[np.ndarray],
    template: SelfPieceTemplate | None = None,
) -> TemplateAssemblyResult:
    """Match four observed polygons to four fixed shapes without reflection."""
    if len(polygons_a4) != 4:
        raise ValueError("固定四块模板要求恰好识别到4块碎片")
    cfg = template or load_self_piece_template()
    pair_results = {}
    for observed_index, observed in enumerate(polygons_a4):
        for template_index, target in enumerate(cfg.pieces):
            pair_results[(observed_index, template_index)] = (
                _fit_observation_to_template(observed, target)
            )

    best = None
    for assignment in permutations(range(4)):
        fits = [pair_results[(index, assignment[index])] for index in range(4)]
        if any(fit is None for fit in fits):
            continue
        errors = tuple(float(fit[0]) for fit in fits)
        maximum_error = max(errors)
        total_error = sum(errors)
        score = total_error + 0.40 * maximum_error
        if best is None or score < best[0]:
            best = (score, assignment, fits, errors)
    if best is None:
        raise RuntimeError("固定四块模板无法与当前轮廓建立对应关系")
    score, assignment, fits, errors = best
    if sum(errors) > cfg.maximum_total_normalized_error:
        raise RuntimeError(
            f"固定四块模板总误差过大：{sum(errors):.3f} > "
            f"{cfg.maximum_total_normalized_error:.3f}"
        )
    if max(errors) > cfg.maximum_piece_normalized_error:
        raise RuntimeError(
            f"固定四块模板单块误差过大：{max(errors):.3f} > "
            f"{cfg.maximum_piece_normalized_error:.3f}"
        )
    transforms = tuple(
        _source_to_target_rigid(fit[2], fit[4], fit[5]) for fit in fits
    )
    return TemplateAssemblyResult(
        transforms=transforms,
        assignment=tuple(int(value) for value in assignment),
        piece_errors=errors,
        score=float(score),
    )
