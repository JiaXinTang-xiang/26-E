from __future__ import annotations

import argparse
import json
from pathlib import Path

from puzzle_core.pipeline import solve_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect, assemble, and simulate moving E-problem puzzle pieces."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = solve_image(args.input, args.output)
    summary = {
        "mode": plan["mode"],
        "detected_count": plan["detected_count"],
        "target_rect": plan["target_rect"],
        "quality": plan["quality"],
        "output": plan["output"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
