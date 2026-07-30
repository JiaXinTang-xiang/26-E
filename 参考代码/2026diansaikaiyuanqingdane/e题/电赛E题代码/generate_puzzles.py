from __future__ import annotations

import argparse
from pathlib import Path

from puzzle_core.config import DEFAULT_IMAGES_ROOT
from puzzle_core.generation import generate_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic A4 puzzle cases for the E problem."
    )
    parser.add_argument(
        "--kind",
        required=True,
        choices=("self", "field-white", "field-card"),
        help="Puzzle appearance and geometry mode.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Field piece count, 1-4. Ignored for --kind self.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_IMAGES_ROOT,
        help="Directory that receives generated case folders.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    case_dir = generate_case(
        kind=args.kind,
        seed=args.seed,
        count=args.count,
        output_root=args.output_root,
    )
    print(case_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
