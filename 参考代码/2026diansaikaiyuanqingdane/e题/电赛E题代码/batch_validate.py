from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from puzzle_core.config import DEFAULT_OUTPUTS_ROOT, SPLIT_Y_MM
from puzzle_core.generation import generate_case
from puzzle_core.geometry import edge_lengths, longest_edge_angle, polygon_area
from puzzle_core.image_io import read_image
from puzzle_core.solver import _intersection_area, solve_layout
from puzzle_core.vision import classify_mode, detect_pieces


def _angle_error_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 90.0) % 180.0 - 90.0)


def _best_center_assignment(
    detected_centers: list[np.ndarray],
    true_centers: list[np.ndarray],
) -> tuple[tuple[int, ...], list[float]]:
    best_permutation: tuple[int, ...] | None = None
    best_errors: list[float] = []
    best_total = float("inf")
    for permutation in itertools.permutations(range(len(true_centers))):
        errors = [
            float(np.linalg.norm(detected_centers[index] - true_centers[target]))
            for index, target in enumerate(permutation)
        ]
        if sum(errors) < best_total:
            best_total = sum(errors)
            best_permutation = permutation
            best_errors = errors
    if best_permutation is None:
        raise RuntimeError("Cannot assign detected centers")
    return best_permutation, best_errors


def _has_outer_edge(polygon: np.ndarray, width: float, height: float) -> bool:
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        if (
            (abs(a[0]) < 1e-6 and abs(b[0]) < 1e-6)
            or (abs(a[0] - width) < 1e-6 and abs(b[0] - width) < 1e-6)
            or (abs(a[1]) < 1e-6 and abs(b[1]) < 1e-6)
            or (abs(a[1] - height) < 1e-6 and abs(b[1] - height) < 1e-6)
        ):
            return True
    return False


def _layout_overlap(polygons: list[np.ndarray]) -> float:
    overlap = 0.0
    for position, polygon_a in enumerate(polygons):
        for polygon_b in polygons[position + 1 :]:
            overlap += _intersection_area(polygon_a, polygon_b)
    return overlap


def validate_case(
    kind: str,
    count: int,
    seed: int,
    work_root: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    case_dir = generate_case(
        kind,
        seed,
        count,
        work_root,
        save_artifacts=False,
    )
    manifest = json.loads(
        (case_dir / "ground_truth.json").read_text(encoding="utf-8")
    )
    image = read_image(case_dir / "input.png")
    scale = image.shape[1] / 210.0
    pieces, _ = detect_pieces(image, scale)
    mode = classify_mode(image, pieces)
    layout = solve_layout(pieces, mode, scale)

    target = manifest["target_rect"]
    target_width = float(target["width_mm"])
    target_height = float(target["height_mm"])
    true_centers = [
        np.asarray(record["source_center_mm"], dtype=np.float64)
        for record in manifest["pieces"]
    ]
    assignment, center_errors = _best_center_assignment(
        [piece.center_mm for piece in pieces], true_centers
    )
    angle_errors = []
    for detected_index, true_index in enumerate(assignment):
        true_polygon = np.asarray(
            manifest["pieces"][true_index]["source_polygon_mm"],
            dtype=np.float64,
        )
        angle_errors.append(
            _angle_error_deg(
                pieces[detected_index].angle_deg,
                longest_edge_angle(true_polygon),
            )
        )

    constraints_ok = True
    generated_area = 0.0
    for record in manifest["pieces"]:
        polygon = np.asarray(record["target_polygon_local_mm"], dtype=np.float64)
        generated_area += polygon_area(polygon)
        constraints_ok &= len(polygon) <= 5
        constraints_ok &= float(np.min(edge_lengths(polygon))) >= 19.999
        constraints_ok &= _has_outer_edge(polygon, target_width, target_height)

    source_polygons = [
        np.asarray(record["source_polygon_mm"], dtype=np.float64)
        for record in manifest["pieces"]
    ]
    source_overlap = _layout_overlap(source_polygons)
    source_bounds_ok = all(
        float(polygon[:, 0].min()) >= 4.8
        and float(polygon[:, 0].max()) <= 205.2
        and float(polygon[:, 1].min()) >= 4.8
        and float(polygon[:, 1].max()) <= SPLIT_Y_MM - 5.8
        for polygon in source_polygons
    )

    final_polygons = list(layout.aligned_polygons_mm.values())
    final_overlap = _layout_overlap(final_polygons)
    final_area = sum(polygon_area(polygon) for polygon in final_polygons)
    final_rect_area = max(layout.width_mm * layout.height_mm, 1e-9)
    area_error_ratio = abs(final_rect_area - final_area) / final_rect_area
    lower_half_ok = all(
        float(polygon[:, 1].min()) > SPLIT_Y_MM for polygon in final_polygons
    )
    width_error = abs(layout.width_mm - target_width)
    height_error = abs(layout.height_mm - target_height)
    elapsed = time.perf_counter() - started

    checks = {
        "piece_count": len(pieces) == count,
        "mode": mode == kind,
        "generation_constraints": bool(constraints_ok),
        "generated_area": abs(generated_area - target_width * target_height) < 1e-3,
        "source_non_overlap": source_overlap < 1e-5,
        "source_bounds": source_bounds_ok,
        "center_error": max(center_errors, default=0.0) <= 2.0,
        "angle_error": max(angle_errors, default=0.0) <= 2.0,
        "target_dimensions": width_error <= 4.0 and height_error <= 4.0,
        "target_lower_half": lower_half_ok,
        "target_overlap": final_overlap / max(final_area, 1e-9) <= 0.01,
        "target_area": area_error_ratio <= 0.03,
        "runtime": elapsed <= 5.0,
    }
    return {
        "kind": kind,
        "count": count,
        "seed": seed,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "runtime_s": round(elapsed, 6),
            "center_error_max_mm": round(max(center_errors, default=0.0), 6),
            "angle_error_max_deg": round(max(angle_errors, default=0.0), 6),
            "width_error_mm": round(width_error, 6),
            "height_error_mm": round(height_error, 6),
            "source_overlap_mm2": round(source_overlap, 6),
            "target_overlap_mm2": round(final_overlap, 6),
            "target_area_error_ratio": round(area_error_ratio, 8),
            "geometry_score": round(float(layout.geometry_score), 6),
            "texture_score": round(float(layout.texture_score), 6),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch-validate generated E-problem field puzzles."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=20,
        help="Number of consecutive seeds for every kind/count combination.",
    )
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT / "verification_report.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be at least 1")
    started = time.perf_counter()
    cases: list[dict[str, object]] = []
    base_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(
        prefix="batch_",
        dir=base_dir,
        ignore_cleanup_errors=True,
    ) as temp:
        work_root = Path(temp)
        for kind in ("field-white", "field-card"):
            for count in (1, 2, 3, 4):
                for seed in range(args.start_seed, args.start_seed + args.seeds):
                    try:
                        cases.append(
                            validate_case(kind, count, seed, work_root)
                        )
                    except Exception as error:  # report the case and continue
                        cases.append(
                            {
                                "kind": kind,
                                "count": count,
                                "seed": seed,
                                "passed": False,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )

    failures = [case for case in cases if not case["passed"]]
    metric_rows = [case["metrics"] for case in cases if "metrics" in case]
    summary = {
        "passed": not failures,
        "case_count": len(cases),
        "failure_count": len(failures),
        "elapsed_s": round(time.perf_counter() - started, 6),
        "max_runtime_s": round(
            max((float(row["runtime_s"]) for row in metric_rows), default=0.0),
            6,
        ),
        "max_center_error_mm": round(
            max(
                (float(row["center_error_max_mm"]) for row in metric_rows),
                default=0.0,
            ),
            6,
        ),
        "max_angle_error_deg": round(
            max(
                (float(row["angle_error_max_deg"]) for row in metric_rows),
                default=0.0,
            ),
            6,
        ),
        "max_width_error_mm": round(
            max(
                (float(row["width_error_mm"]) for row in metric_rows),
                default=0.0,
            ),
            6,
        ),
        "max_height_error_mm": round(
            max(
                (float(row["height_error_mm"]) for row in metric_rows),
                default=0.0,
            ),
            6,
        ),
    }
    report = {
        "configuration": {
            "kinds": ["field-white", "field-card"],
            "counts": [1, 2, 3, 4],
            "seeds_per_combination": args.seeds,
            "start_seed": args.start_seed,
        },
        "summary": summary,
        "failures": failures,
        "cases": cases,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.report.resolve())
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
