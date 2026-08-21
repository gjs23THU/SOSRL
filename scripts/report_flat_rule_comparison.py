"""Aggregate paired HRL/flat-rule evaluation results across training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODELS = ("hrl", "flat_rule_dqn")
METRICS = (
    "success_rate",
    "mean_success_makespan",
    "mean_net_cost",
    "mean_peak_net_cost",
    "mean_architecture_changes",
    "budget_violation_rate",
    "ever_budget_violation_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    return parser.parse_args()


def evaluation_dir(runs_root: Path, seed: int) -> Path:
    return (
        runs_root
        / f"flat_rules_budgetmatched_seed{seed}"
        / "evaluation_flat_rules_128"
    )


def verify_paired_hashes(results: pd.DataFrame, seed: int) -> None:
    hashes = {}
    for model in MODELS:
        model_rows = results.loc[results["model"] == model]
        if model_rows.empty:
            raise ValueError(f"seed {seed} is missing evaluation rows for {model}")
        hashes[model] = model_rows["scenario_hash"].astype(str).tolist()
    if hashes[MODELS[0]] != hashes[MODELS[1]]:
        raise ValueError(f"seed {seed} HRL/flat scenario hashes do not match")


def load_seed(runs_root: Path, seed: int):
    directory = evaluation_dir(runs_root, seed)
    summary_path = directory / "summary.csv"
    paired_path = directory / "paired_comparisons.csv"
    results_path = directory / "results.csv"
    for path in (summary_path, paired_path, results_path):
        if not path.exists():
            raise FileNotFoundError(path)

    results = pd.read_csv(results_path)
    verify_paired_hashes(results, seed)
    summary = pd.read_csv(summary_path)
    summary = summary.loc[summary["model"].isin(MODELS)].copy()
    if set(summary["model"]) != set(MODELS):
        raise ValueError(f"seed {seed} summary does not contain both target models")
    summary.insert(0, "seed", seed)

    paired = pd.read_csv(paired_path)
    paired = paired.loc[paired["candidate_model"] == "flat_rule_dqn"].copy()
    if len(paired) != 1:
        raise ValueError(f"seed {seed} must contain one flat-vs-HRL comparison")
    paired.insert(0, "seed", seed)
    return summary, paired


def aggregate_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, data in per_seed.groupby("model", sort=False):
        row = {"model": model, "training_seeds": int(data["seed"].nunique())}
        for metric in METRICS:
            numeric = pd.to_numeric(data[metric], errors="coerce")
            row[f"{metric}_mean"] = numeric.mean()
            row[f"{metric}_sample_std"] = numeric.std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_paired(per_seed: pd.DataFrame) -> pd.DataFrame:
    mean_fields = (
        "mean_candidate_minus_reference_makespan",
        "median_candidate_minus_reference_makespan",
        "mean_candidate_minus_reference_net_cost",
    )
    sum_fields = (
        "paired_scenarios",
        "common_success_count",
        "reference_only_success_count",
        "candidate_only_success_count",
        "candidate_faster_count",
        "reference_faster_count",
        "makespan_tie_count",
    )
    row = {"training_seeds": int(per_seed["seed"].nunique())}
    for field in mean_fields:
        numeric = pd.to_numeric(per_seed[field], errors="coerce")
        row[f"{field}_mean_across_seeds"] = numeric.mean()
        row[f"{field}_sample_std_across_seeds"] = numeric.std(ddof=1)
    for field in sum_fields:
        row[f"{field}_total"] = pd.to_numeric(
            per_seed[field], errors="raise"
        ).sum()
    return pd.DataFrame([row])


def save_figure(per_seed: pd.DataFrame, output_dir: Path) -> None:
    display_metrics = (
        ("success_rate", "Success rate"),
        ("mean_success_makespan", "Successful makespan"),
        ("mean_net_cost", "Final net cost"),
        ("budget_violation_rate", "Final budget violation rate"),
    )
    colors = {"hrl": "#1F5A94", "flat_rule_dqn": "#C26A1B"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (metric, title) in zip(axes.flat, display_metrics, strict=True):
        series = [
            pd.to_numeric(
                per_seed.loc[per_seed["model"] == model, metric],
                errors="coerce",
            ).dropna()
            for model in MODELS
        ]
        box = axis.boxplot(series, tick_labels=("HRL", "Flat-DQN"), patch_artist=True)
        for patch, model in zip(box["boxes"], MODELS, strict=True):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.35)
        for seed in sorted(per_seed["seed"].unique()):
            seed_rows = per_seed.loc[per_seed["seed"] == seed].set_index("model")
            if all(model in seed_rows.index for model in MODELS):
                values = [float(seed_rows.loc[model, metric]) for model in MODELS]
                axis.plot((1, 2), values, color="#6B7280", alpha=0.45, linewidth=1)
        axis.set_title(title, loc="left", fontweight="semibold")
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(
        "Two-level HRL vs 24-action flat rule DQN",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in (".png", ".svg"):
        fig.savefig(
            output_dir / f"hrl_vs_flat_rules_multiseed{suffix}",
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seeds = args.seeds or [1, 2, 3, 4, 5, 6]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate seeds are not allowed")
    summaries = []
    comparisons = []
    for seed in seeds:
        summary, paired = load_seed(args.runs_root, seed)
        summaries.append(summary)
        comparisons.append(paired)

    per_seed_summary = pd.concat(summaries, ignore_index=True)
    per_seed_paired = pd.concat(comparisons, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_summary.to_csv(args.output_dir / "per_seed_summary.csv", index=False)
    per_seed_paired.to_csv(args.output_dir / "per_seed_paired.csv", index=False)
    aggregate_summary(per_seed_summary).to_csv(
        args.output_dir / "multiseed_summary.csv",
        index=False,
    )
    aggregate_paired(per_seed_paired).to_csv(
        args.output_dir / "multiseed_paired.csv",
        index=False,
    )
    save_figure(per_seed_summary, args.output_dir)
    (args.output_dir / "report_manifest.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "models": list(MODELS),
                "paired_evaluation": True,
                "hidden_dim": 128,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
