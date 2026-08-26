"""Summarize milestone fronts for any supported multi-objective baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sosrl.multiobjective_calibration import run_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--max-budget", type=int)
    args = parser.parse_args()
    summary = run_calibration(
        args.input_dir,
        args.output_dir,
        algorithm=args.algorithm,
        max_budget=args.max_budget,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

