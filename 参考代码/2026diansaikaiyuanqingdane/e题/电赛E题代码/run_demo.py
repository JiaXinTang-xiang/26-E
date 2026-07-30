from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzle_core.config import DEFAULT_IMAGES_ROOT, DEFAULT_OUTPUTS_ROOT
from puzzle_core.generation import generate_case
from puzzle_core.pipeline import solve_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and solve the three default E-problem demonstrations."
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--field-count", type=int, default=4, choices=(1, 2, 3, 4))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    results: list[dict[str, object]] = []
    for kind, count in (
        ("self", 4),
        ("field-white", args.field_count),
        ("field-card", args.field_count),
    ):
        case_dir = generate_case(
            kind,
            args.seed,
            count,
            output_root=DEFAULT_IMAGES_ROOT,
        )
        output_dir = DEFAULT_OUTPUTS_ROOT / case_dir.name
        plan = solve_image(case_dir / "input.png", output_dir)
        results.append(
            {
                "case": case_dir.name,
                "mode": plan["mode"],
                "detected_count": plan["detected_count"],
                "target_rect": plan["target_rect"],
                "output": str(output_dir.resolve()),
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
