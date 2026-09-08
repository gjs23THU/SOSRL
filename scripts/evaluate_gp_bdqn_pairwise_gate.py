"""Evaluate selected additive and residual-interaction BDQNs on small gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from sosrl.workflows.gp_architecture import (
    load_scenario_manifest,
    save_scenario_manifest,
)
from sosrl.workflows.gp_bdqn_tuning import _evaluate_stack_repeats
from sosrl.workflows.tuning_statistics import paired_difference_ci, summarize_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _small_gate(
    source: Path,
    destination: Path,
    *,
    size: int,
    split: str,
) -> list[dict]:
    if destination.is_file():
        scenarios = load_scenario_manifest(destination)["scenarios"]
        if len(scenarios) != size:
            raise ValueError(f"existing {destination} has the wrong size")
        return scenarios
    manifest = load_scenario_manifest(source)
    grouped = defaultdict(list)
    for scenario in manifest["scenarios"]:
        grouped[str(scenario["category"])].append(scenario)
    if size % len(grouped):
        raise ValueError("small gate size must be divisible by the category count")
    per_category = size // len(grouped)
    selected = []
    for category in sorted(grouped):
        candidates = sorted(grouped[category], key=lambda row: row["scenario_hash"])
        if len(candidates) < per_category:
            raise ValueError(f"not enough {category} scenarios")
        selected.extend(candidates[:per_category])
    save_scenario_manifest(
        destination,
        split=split,
        seed=20260908,
        scenarios=selected,
    )
    return selected


def _repeat_parent(parent_rows: list[dict], candidate_rows: list[dict]) -> list[dict]:
    by_scenario = {str(row["scenario_hash"]): row for row in parent_rows}
    repeated = []
    for candidate in candidate_rows:
        row = dict(by_scenario[str(candidate["scenario_hash"])])
        row["repeat"] = candidate["repeat"]
        repeated.append(row)
    return repeated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--gp-policy", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--iid-manifest", type=Path, required=True)
    parser.add_argument("--ood-manifest", type=Path, required=True)
    parser.add_argument("--iid-size", type=int, default=128)
    parser.add_argument("--ood-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.output_root.resolve()
    gate_root = root / "gate_small"
    gate_root.mkdir(parents=True, exist_ok=True)
    sources = {
        "iid": (args.iid_manifest.resolve(), int(args.iid_size)),
        "ood": (args.ood_manifest.resolve(), int(args.ood_size)),
    }
    validation = json.loads(
        (root / "aggregate_validation.json").read_text(encoding="utf-8")
    )
    checkpoints = {
        arm: [Path(item["path"]) for item in validation["arms"][arm]["selected_checkpoints"]]
        for arm in ("additive", "pairwise_residual")
    }
    result = {
        "design": {
            "iid_size": int(args.iid_size),
            "ood_size": int(args.ood_size),
            "selection": "validation only; small gates do not reselect checkpoints",
            "performance_estimate": "three-seed mean and hierarchical 95% CI",
        },
        "splits": {},
    }
    for split_index, (split, (source, size)) in enumerate(sources.items()):
        manifest_path = gate_root / f"{split}_{size}.json"
        scenarios = _small_gate(
            source,
            manifest_path,
            size=size,
            split=f"gate_{split}_small",
        )
        rows_by_model = {
            "parent_b2": _evaluate_stack_repeats(
                gp_policies=[args.gp_policy],
                bdqn_checkpoints=[args.parent_checkpoint],
                scenarios=scenarios,
                device=args.device,
                model="parent_b2",
                split=split,
            )
        }
        for arm in ("additive", "pairwise_residual"):
            rows_by_model[arm] = _evaluate_stack_repeats(
                gp_policies=[args.gp_policy],
                bdqn_checkpoints=checkpoints[arm],
                scenarios=scenarios,
                device=args.device,
                model=arm,
                split=split,
            )
        all_rows = [row for rows in rows_by_model.values() for row in rows]
        _write_csv(gate_root / f"{split}_results.csv", all_rows)
        summaries = {
            model: summarize_rows(
                rows,
                samples=int(args.samples),
                seed=20263908 + split_index * 100 + index * 10,
            )
            for index, (model, rows) in enumerate(rows_by_model.items())
        }
        comparisons = {}
        for index, arm in enumerate(("additive", "pairwise_residual")):
            parent = _repeat_parent(rows_by_model["parent_b2"], rows_by_model[arm])
            comparisons[f"{arm}_minus_parent"] = {
                "makespan": paired_difference_ci(
                    parent,
                    rows_by_model[arm],
                    "makespan",
                    both_success=True,
                    samples=int(args.samples),
                    seed=20264908 + split_index * 100 + index * 10,
                ),
                "final_cost": paired_difference_ci(
                    parent,
                    rows_by_model[arm],
                    "final_net_cost",
                    both_success=True,
                    samples=int(args.samples),
                    seed=20264909 + split_index * 100 + index * 10,
                ),
            }
        comparisons["pairwise_residual_minus_additive"] = {
            "makespan": paired_difference_ci(
                rows_by_model["additive"],
                rows_by_model["pairwise_residual"],
                "makespan",
                both_success=True,
                samples=int(args.samples),
                seed=20265908 + split_index * 100,
            ),
            "final_cost": paired_difference_ci(
                rows_by_model["additive"],
                rows_by_model["pairwise_residual"],
                "final_net_cost",
                both_success=True,
                samples=int(args.samples),
                seed=20265909 + split_index * 100,
            ),
        }
        result["splits"][split] = {
            "manifest": str(manifest_path),
            "summaries": summaries,
            "paired_comparisons": comparisons,
        }
    output = gate_root / "summary.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
