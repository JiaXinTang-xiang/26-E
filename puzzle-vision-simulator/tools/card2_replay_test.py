#!/usr/bin/env python3
"""Replay the latest saved piece polygons through experimental card method 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from puzzle_device.planning import (
    AssemblyConfig,
    draw_card_candidate_gallery,
    solve_composite_card_assembly,
)
from puzzle_device.vision.image_io import read_image, write_image
from puzzle_device.planning.assembly import (
    _apply,
    a4_to_global_pixels,
    global_pixels_to_a4,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="output/assembly_plan.json")
    parser.add_argument("--output", default="output/card2_candidate_gallery.png")
    parser.add_argument("--image", default="output/card2_source_frame.png")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.workers is not None:
        os.environ["PUZZLE_CARD2_WORKERS"] = str(max(1, args.workers))

    document = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    roi = tuple(int(value) for value in document["full_a4_roi_px"])
    config = AssemblyConfig()
    polygons = []
    for piece in sorted(document["pieces"], key=lambda item: item["piece_id"]):
        if "source_polygon_px" in piece:
            polygons.append(np.asarray(piece["source_polygon_px"], dtype=np.float64))
            continue
        # Backward compatibility for plans saved before source polygons were
        # recorded: remove execution gap, then invert the rigid transform.
        target_a4 = global_pixels_to_a4(
            np.asarray(piece["target_polygon_px"], dtype=np.float64), roi, config
        )
        target_a4 -= np.asarray(piece.get("target_offset_a4_mm", [0.0, 0.0]))
        source_a4 = _apply(
            target_a4, np.linalg.inv(np.asarray(piece["transform_a4_mm"], dtype=np.float64))
        )
        polygons.append(a4_to_global_pixels(source_a4, roi, config))
    image = read_image(args.image)
    if image is None:
        print(f"WARN: {args.image} not found; texture ranking uses a blank image")
        image = np.full((720, 1280, 3), 230, np.uint8)
    started = time.monotonic()
    plan = solve_composite_card_assembly(
        image, polygons, roi, config, require_upper_half=False
    )
    elapsed = time.monotonic() - started
    gallery = draw_card_candidate_gallery(image, plan, AssemblyConfig())
    if not write_image(args.output, gallery):
        raise OSError(f"cannot write {args.output}")
    print(
        f"PASS: {elapsed:.2f}s, size={plan.recovered_size_mm[0]:.1f}x"
        f"{plan.recovered_size_mm[1]:.1f} mm, fill={plan.rectangle_fill_ratio:.1%}"
    )
    for item in plan.candidate_diagnostics:
        print(
            f"#{item['rank']} total={item['total_score']:.1f} "
            f"size={item['recovered_size_mm'][0]:.1f}x"
            f"{item['recovered_size_mm'][1]:.1f} "
            f"fill={item['rectangle_fill_ratio']:.1%} "
            f"overlap={item['overlap_ratio']:.1%} "
            f"unexplained={item['unexplained_edge_ratio']:.1%}"
        )
    print(f"gallery: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
