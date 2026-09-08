"""Aggregate the three-seed GP+BDQN pairwise interaction study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sosrl.workflows.tuning_statistics import (
    paired_difference_ci,
    select_aggregate_checkpoint,
    summarize_rows,
)


ARMS = ("additive", "pairwise_joint", "pairwise_residual")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _model_step(model: str) -> int:
    if model == "B0":
        return 0
    if not model.startswith("B"):
        raise ValueError(f"unexpected validation model: {model}")
    label = model[1:]
    return int(label[:-1]) * 1000 if label.endswith("k") else int(label)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_rows(root: Path, arm: str) -> tuple[list[dict], list[int]]:
    rows = []
    seeds = []
    for cell in sorted((root / arm).glob("seed_*")):
        seed = int(cell.name.split("_")[-1])
        seeds.append(seed)
        for row in _read_csv(cell / "validation" / "all_results.csv"):
            rows.append(
                {
                    **row,
                    "repeat": seed,
                    "seed": seed,
                    "split": "validation",
                    "target_environment_steps": _model_step(row["model"]),
                }
            )
    if not rows:
        raise FileNotFoundError(f"no validation results for {arm}")
    return rows, seeds


def _selected_checkpoints(
    root: Path,
    arm: str,
    seeds: list[int],
    step: int,
) -> list[dict[str, str | int]]:
    records = []
    for seed in seeds:
        if step == 0:
            manifest = json.loads(
                (root / arm / f"seed_{seed}" / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            path = Path(manifest["inputs"]["b0_scheduler"]["path"])
        else:
            path = root / arm / f"seed_{seed}" / "training" / f"checkpoint_{step // 1000}k.pt"
        records.append({"seed": seed, "path": str(path), "sha256": _sha256(path)})
    return records


def summarize(root: Path, *, samples: int) -> dict:
    rows_by_arm = {}
    selections = {}
    selected_rows = {}
    for index, arm in enumerate(ARMS):
        rows, seeds = _arm_rows(root, arm)
        rows_by_arm[arm] = rows
        selection = select_aggregate_checkpoint(
            rows,
            samples=samples,
            seed=20260908 + index * 100,
        )
        step = int(selection["selected_step"])
        chosen = [
            row for row in rows if int(row["target_environment_steps"]) == step
        ]
        selected_rows[arm] = chosen
        selections[arm] = {
            **selection,
            "aggregate_metrics": summarize_rows(
                chosen,
                samples=samples,
                seed=20261908 + index * 100,
            ),
            "selected_checkpoints": _selected_checkpoints(
                root,
                arm,
                seeds,
                step,
            ),
        }

    comparisons = {}
    for index, candidate in enumerate(ARMS[1:]):
        baseline = selected_rows["additive"]
        candidate_rows = selected_rows[candidate]
        comparisons[f"{candidate}_minus_additive"] = {
            "makespan": paired_difference_ci(
                baseline,
                candidate_rows,
                "makespan",
                both_success=True,
                samples=samples,
                seed=20262908 + index * 100,
            ),
            "final_cost": paired_difference_ci(
                baseline,
                candidate_rows,
                "final_net_cost",
                both_success=True,
                samples=samples,
                seed=20262909 + index * 100,
            ),
        }
    return {
        "performance_estimate": "three-seed aggregate mean with hierarchical 95% CI",
        "selection_rule": (
            "minimum aggregate failure count, then minimum mean successful "
            "makespan, then earliest step whose 95% CI overlaps the best"
        ),
        "arms": selections,
        "paired_comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    root = args.output_root.resolve()
    result = summarize(root, samples=int(args.samples))
    output = root / "aggregate_validation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
