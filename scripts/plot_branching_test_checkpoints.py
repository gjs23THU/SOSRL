"""Plot paired unseen-test performance at saved additive-BDQN checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#1F5A94"
ORANGE = "#C26A1B"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", nargs=3, required=True,
                        metavar=("LABEL", "HISTORY", "RESULTS"))
    parser.add_argument("--hrl-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def read_primary_results(path: Path, model: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "model", "scenario_hash", "success", "makespan", "scheduler_reward"
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    selected = data.loc[data["model"] == model].copy()
    if selected.empty:
        raise ValueError(f"{path} contains no rows for model={model!r}")
    selected["success"] = selected["success"].astype(str).str.lower().isin(
        ["true", "1"]
    )
    return selected.sort_values("scenario_hash").reset_index(drop=True)


def build_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    checkpoint_results: list[pd.DataFrame] = []
    reference_hashes: set[str] | None = None
    for checkpoint_index, (label, history_path, results_path) in enumerate(
        args.checkpoint
    ):
        history = pd.read_csv(Path(history_path))
        if "total_env_steps" not in history.columns or history.empty:
            raise ValueError(f"{history_path} has no completed training steps")
        results = read_primary_results(Path(results_path), "architecture_branching")
        hashes = set(results["scenario_hash"])
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ValueError("BDQN checkpoints do not use identical test scenarios")
        successful_makespan = results.loc[results["success"], "makespan"].to_numpy()
        reward = results["scheduler_reward"].to_numpy()
        makespan_ci = bootstrap_interval(
            successful_makespan,
            samples=args.bootstrap_samples,
            seed=args.seed + checkpoint_index,
        )
        reward_ci = bootstrap_interval(
            reward,
            samples=args.bootstrap_samples,
            seed=args.seed + 100 + checkpoint_index,
        )
        rows.append(
            {
                "checkpoint": label,
                "training_steps": int(history["total_env_steps"].iloc[-1]),
                "test_episodes": len(results),
                "success_rate": float(results["success"].mean()),
                "mean_success_makespan": float(successful_makespan.mean()),
                "makespan_ci_low": makespan_ci[0],
                "makespan_ci_high": makespan_ci[1],
                "mean_scheduler_reward": float(reward.mean()),
                "reward_ci_low": reward_ci[0],
                "reward_ci_high": reward_ci[1],
            }
        )
        checkpoint_results.append(results)

    hrl = read_primary_results(args.hrl_results, "hrl")
    if set(hrl["scenario_hash"]) != reference_hashes:
        raise ValueError("HRL baseline does not use the same paired test scenarios")
    return (
        pd.DataFrame(rows).sort_values("training_steps").reset_index(drop=True),
        checkpoint_results,
        hrl,
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors="#4B5563")
    axis.set_xlabel("Completed lower-level assignment steps")


def plot_metric(
    metrics: pd.DataFrame,
    hrl: pd.DataFrame,
    *,
    value_column: str,
    low_column: str,
    high_column: str,
    hrl_values: np.ndarray,
    hrl_ci: tuple[float, float],
    title: str,
    ylabel: str,
    output_stem: Path,
    subtitle_extra: str,
) -> None:
    fig, axis = plt.subplots(figsize=(10.8, 6.2))
    x = metrics["training_steps"].to_numpy()
    y = metrics[value_column].to_numpy()
    low = metrics[low_column].to_numpy()
    high = metrics[high_column].to_numpy()
    yerr = np.vstack([y - low, high - y])

    hrl_mean = float(np.mean(hrl_values))
    axis.axhspan(hrl_ci[0], hrl_ci[1], color=BLUE, alpha=0.09, linewidth=0)
    axis.axhline(
        hrl_mean,
        color=BLUE,
        linestyle="--",
        linewidth=2.0,
        label="Existing HRL (same Architecture DQN)",
    )
    axis.errorbar(
        x,
        y,
        yerr=yerr,
        color=ORANGE,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.8,
        markersize=7,
        linewidth=2.2,
        capsize=4,
        label="Additive BDQN checkpoints",
        zorder=4,
    )

    span = max(float(y.max() - y.min()), 1.0)
    for row in metrics.itertuples(index=False):
        value = float(getattr(row, value_column))
        value_text = f"{value:.3f}" if "reward" in value_column else f"{value:.1f}"
        annotation = value_text
        if value_column == "mean_success_makespan":
            annotation += f"\nsuccess {row.success_rate:.0%}"
        axis.annotate(
            annotation,
            (row.training_steps, value),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=ORANGE,
            fontsize=9,
            fontweight="semibold",
        )
    axis.annotate(
        f"HRL {hrl_mean:.3f}" if "reward" in value_column else f"HRL {hrl_mean:.1f}",
        (126_000, hrl_mean),
        xytext=(-2, 6),
        textcoords="offset points",
        ha="right",
        color=BLUE,
        fontsize=9,
        fontweight="semibold",
    )

    axis.set_xlim(0, 128_000)
    axis.set_xticks(x, [f"{step / 1000:.0f}k\n({step:,})" for step in x])
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK)
    axis.legend(frameon=False, loc="best")
    style_axis(axis)
    fig.text(
        0.125,
        0.965,
        (
            "Mean and 95% bootstrap CI on the same 100 unseen paired scenarios. "
            "Three saved checkpoints only; connecting segments are visual guides.\n"
            + subtitle_extra
        ),
        color=MUTED,
        fontsize=8.5,
        linespacing=1.45,
        va="top",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.865))
    for suffix in (".png", ".svg"):
        fig.savefig(
            output_stem.with_suffix(suffix),
            dpi=240 if suffix == ".png" else None,
            bbox_inches="tight",
            pad_inches=0.14,
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")
    metrics, _checkpoint_results, hrl = build_metrics(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    successful_hrl_makespan = hrl.loc[hrl["success"], "makespan"].to_numpy()
    hrl_makespan_ci = bootstrap_interval(
        successful_hrl_makespan,
        samples=args.bootstrap_samples,
        seed=args.seed + 200,
    )
    hrl_reward = hrl["scheduler_reward"].to_numpy()
    hrl_reward_ci = bootstrap_interval(
        hrl_reward,
        samples=args.bootstrap_samples,
        seed=args.seed + 201,
    )
    metrics.assign(
        hrl_mean_success_makespan=float(successful_hrl_makespan.mean()),
        hrl_makespan_ci_low=hrl_makespan_ci[0],
        hrl_makespan_ci_high=hrl_makespan_ci[1],
        hrl_mean_scheduler_reward=float(hrl_reward.mean()),
        hrl_reward_ci_low=hrl_reward_ci[0],
        hrl_reward_ci_high=hrl_reward_ci[1],
    ).to_csv(args.output_dir / "test_checkpoint_metrics.csv", index=False)

    plot_metric(
        metrics,
        hrl,
        value_column="mean_success_makespan",
        low_column="makespan_ci_low",
        high_column="makespan_ci_high",
        hrl_values=successful_hrl_makespan,
        hrl_ci=hrl_makespan_ci,
        title="Unseen-test makespan at saved BDQN checkpoints",
        ylabel="Mean makespan among successful missions",
        output_stem=args.output_dir / "test_makespan_vs_training_steps",
        subtitle_extra=(
            "Makespan excludes failed missions; success rate is printed at each marker. "
            "The focused y-scale compares checkpoint means."
        ),
    )
    plot_metric(
        metrics,
        hrl,
        value_column="mean_scheduler_reward",
        low_column="reward_ci_low",
        high_column="reward_ci_high",
        hrl_values=hrl_reward,
        hrl_ci=hrl_reward_ci,
        title="Unseen-test scheduler reward at saved BDQN checkpoints",
        ylabel="Mean comparable scheduler reward",
        output_stem=args.output_dir / "test_reward_vs_training_steps",
        subtitle_extra=(
            "Reward includes success/dead-end terminal terms and all test episodes; "
            "higher is better."
        ),
    )


if __name__ == "__main__":
    main()
