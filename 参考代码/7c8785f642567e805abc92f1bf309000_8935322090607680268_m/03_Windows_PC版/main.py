from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

from puzzle_vision.camera import CameraError, capture_frame
from puzzle_vision.config import load_config, save_default_config
from puzzle_vision.detector import DetectionError
from puzzle_vision.pipeline import PuzzleVisionPipeline
from puzzle_vision.solver import SolveError
from puzzle_vision.stress import run_solver_stress_test
from puzzle_vision.synthetic import run_self_test


PROJECT_DIR = Path(__file__).resolve().parent
cv2.setUseOptimized(True)
cv2.setNumThreads(max(2, os.cpu_count() or 4))


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(output))


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RDK X5 vision-only solver for the A4 puzzle device"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_DIR / "config.json"),
        help="JSON configuration path",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init-config", help="write a fresh default configuration"
    )
    initialize.add_argument("--output", default=str(PROJECT_DIR / "config.json"))

    capture = subparsers.add_parser("capture", help="capture one raw camera frame")
    capture.add_argument("--source", default=None)
    capture.add_argument("--output", default="capture.jpg")

    background = subparsers.add_parser(
        "background", help="capture and rectify an empty A4 background"
    )
    background.add_argument("--source", default=None)
    background.add_argument("--output", default="background_rectified.png")
    background.add_argument("--debug", default="background_camera.jpg")

    analyze = subparsers.add_parser(
        "analyze", help="detect pieces and compute the placement plan"
    )
    analyze.add_argument("--source", default=None)
    analyze.add_argument(
        "--mode",
        choices=PuzzleVisionPipeline.MODES,
        default="fixed",
    )
    analyze.add_argument(
        "--source-region",
        choices=("upper", "lower", "auto"),
        default="upper",
        help="piece half; competition default is upper",
    )
    analyze.add_argument("--background", default=None)
    analyze.add_argument("--result", default="result.json")
    analyze.add_argument("--debug", default="debug.jpg")
    analyze.add_argument(
        "--overlay",
        default=None,
        help="optional raw-camera image with A4, pieces, centres and target poses",
    )
    analyze.add_argument("--mask", default="mask.png")
    analyze.add_argument("--rectified", default=None)

    self_test = subparsers.add_parser(
        "self-test", help="run fixed and unknown solvers on a synthetic A4 scene"
    )
    self_test.add_argument("--output-dir", default="self_test_output")

    stress_test = subparsers.add_parser(
        "stress-test",
        help="solve random legal rectangles cut into four quadrilaterals",
    )
    stress_test.add_argument("--cases", type=int, default=50)
    stress_test.add_argument("--seed", type=int, default=20260729)
    stress_test.add_argument("--corner-noise-mm", type=float, default=0.8)
    stress_test.add_argument(
        "--piece-counts",
        default="4",
        help="comma-separated legal piece counts, for example 2,3,4",
    )
    stress_test.add_argument("--result", default="stress_test_result.json")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "init-config":
        save_default_config(args.output)
        print(json.dumps({"ok": True, "config": str(Path(args.output).resolve())}))
        return 0

    config_path = Path(args.config)
    config = load_config(config_path if config_path.exists() else None)
    if args.command == "self-test":
        summary = run_self_test(config, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "stress-test":
        try:
            piece_counts = tuple(
                int(value.strip())
                for value in args.piece_counts.split(",")
                if value.strip()
            )
        except ValueError as exc:
            raise SystemExit(
                "--piece-counts must be a comma-separated list of 2, 3, 4"
            ) from exc
        summary = run_solver_stress_test(
            config,
            cases=max(1, args.cases),
            seed=args.seed,
            corner_noise_mm=max(0.0, args.corner_noise_mm),
            piece_counts=piece_counts,
        )
        write_json(args.result, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["ok"] else 3

    source = args.source or str(config["camera"]["source"])
    frame = capture_frame(source, config["camera"])
    if args.command == "capture":
        write_image(args.output, frame)
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(Path(args.output).resolve()),
                    "shape": list(frame.shape),
                }
            )
        )
        return 0

    pipeline = PuzzleVisionPipeline(config)
    if args.command == "background":
        paper = pipeline.rectify(frame)
        write_image(args.output, paper.image)
        write_image(args.debug, frame)
        print(
            json.dumps(
                {
                    "ok": True,
                    "background": str(Path(args.output).resolve()),
                    "camera_frame": str(Path(args.debug).resolve()),
                    "paper_corners_px": np.round(paper.corners_px, 2).tolist(),
                    "scale_source": "A4 210x297 mm",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    background_image = read_image(args.background) if args.background else None
    result, debug, mask, rectified = pipeline.analyze(
        frame, args.mode, background_image, args.source_region
    )
    write_json(args.result, result)
    write_image(args.debug, debug)
    if args.overlay:
        write_image(
            args.overlay,
            pipeline.draw_camera_overlay(frame, result),
        )
    write_image(args.mask, mask)
    if args.rectified:
        write_image(args.rectified, rectified)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CameraError, DetectionError, SolveError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)
