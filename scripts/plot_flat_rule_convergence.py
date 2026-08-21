"""Plot convergence diagnostics for the 24-action flat-rule DQN."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#1F5A94"
ORANGE = "#C26A1B"
RAW_BLUE = "#A7B8CA"
RAW_ORANGE = "#DFC4A3"
INK = "#1F2937"
GRID = "#E5E7EB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    if mapped.isna().any():
        raise ValueError("boolean history column contains an unknown value")
    return mapped


def prepare_history(path: Path, window: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "episode",
        "joint_reward",
        "joint_loss",
        "success",
        "makespan",
        "net_cost",
        "architecture_changes",
        "budget_violation",
        "cumulative_environment_steps",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    if len(data) < 8:
        raise ValueError("at least eight episodes are required for a trend chart")

    data["episode_number"] = pd.to_numeric(data["episode"]) + 1
    data["success_rate"] = boolean_series(data["success"])
    data["final_over_budget_rate"] = boolean_series(data["budget_violation"])
    mean_columns = (
        "joint_reward",
        "makespan",
        "net_cost",
        "architecture_changes",
        "success_rate",
        "final_over_budget_rate",
    )
    for column in mean_columns:
        numeric = pd.to_numeric(data[column], errors="coerce")
        data[f"{column}_smooth"] = numeric.rolling(
            window,
            min_periods=max(2, window // 5),
        ).mean()
    data["joint_loss"] = pd.to_numeric(data["joint_loss"], errors="coerce")
    data["joint_loss_smooth"] = data["joint_loss"].rolling(
        window,
        min_periods=max(2, window // 5),
    ).median()
    return data


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#6B7280")
    axis.spines["bottom"].set_color("#6B7280")
    axis.tick_params(colors="#4B5563")


def plot_metric(
    axis: plt.Axes,
    data: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    window: int,
    *,
    color: str = BLUE,
    raw_color: str = RAW_BLUE,
) -> None:
    x = data["episode_number"]
    axis.plot(
        x,
        data[column],
        color=raw_color,
        alpha=0.18,
        linewidth=0.55,
        rasterized=True,
    )
    axis.plot(
        x,
        data[f"{column}_smooth"],
        color=color,
        linewidth=2.15,
        label=f"{window}-episode moving average",
    )
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold")
    axis.set_ylabel(ylabel)
    style_axis(axis)


def plot_rates(
    axis: plt.Axes,
    data: pd.DataFrame,
    window: int,
) -> None:
    x = data["episode_number"]
    axis.plot(
        x,
        data["success_rate_smooth"],
        color=BLUE,
        linewidth=2.15,
        label="Success rate",
    )
    axis.plot(
        x,
        data["final_over_budget_rate_smooth"],
        color=ORANGE,
        linestyle="--",
        linewidth=2.0,
        label="Final over-budget rate",
    )
    axis.set_title(
        "(d) Rolling success and final budget compliance",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    axis.set_ylabel(f"{window}-episode rate")
    axis.set_ylim(-0.03, 1.03)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.set_yticklabels(
        [f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)]
    )
    axis.legend(frameon=False, loc="center right", fontsize=8)
    style_axis(axis)


def build_figure(
    data: pd.DataFrame,
    *,
    window: int,
    budget: float,
    seed: int,
) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    plot_metric(
        axes[0, 0],
        data,
        "joint_reward",
        "(a) Joint reward (higher is better)",
        "Episode reward",
        window,
    )
    plot_metric(
        axes[0, 1],
        data,
        "makespan",
        "(b) Mission makespan (lower is better)",
        "Makespan",
        window,
    )
    plot_metric(
        axes[1, 0],
        data,
        "net_cost",
        "(c) Final mission net cost (lower is better)",
        "Net cost",
        window,
        color=ORANGE,
        raw_color=RAW_ORANGE,
    )
    axes[1, 0].axhline(
        budget,
        color=INK,
        linestyle=":",
        linewidth=1.6,
        label=f"Budget = {budget:,.0f}",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    plot_rates(axes[1, 1], data, window)

    loss_data = data.loc[data["joint_loss"] > 0]
    axes[2, 0].plot(
        loss_data["episode_number"],
        loss_data["joint_loss"],
        color=RAW_BLUE,
        alpha=0.16,
        linewidth=0.5,
        rasterized=True,
    )
    axes[2, 0].plot(
        data["episode_number"],
        data["joint_loss_smooth"],
        color=BLUE,
        linewidth=2.15,
        label=f"{window}-episode moving median",
    )
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_title(
        "(e) DQN TD loss (log scale)",
        loc="left",
        fontsize=11,
        fontweight="semibold",
    )
    axes[2, 0].set_ylabel("TD loss")
    axes[2, 0].legend(frameon=False, fontsize=8)
    style_axis(axes[2, 0])

    plot_metric(
        axes[2, 1],
        data,
        "architecture_changes",
        "(f) Architecture changes per mission",
        "Change count",
        window,
        color=ORANGE,
        raw_color=RAW_ORANGE,
    )
    for axis in axes[2, :]:
        axis.set_xlabel("Training episode")

    actual_steps = int(data["cumulative_environment_steps"].iloc[-1])
    fig.suptitle(
        "24-action Flat-DQN training convergence",
        fontsize=16,
        fontweight="bold",
        y=0.987,
    )
    fig.text(
        0.5,
        0.958,
        (
            f"Seed {seed} | {len(data):,} complete episodes | "
            f"{actual_steps:,} environment steps | solid lines are "
            f"{window}-episode summaries; faint lines are raw episodes"
        ),
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.938,
        "Single training run; smoothing shows local trend, not a confidence interval.",
        ha="center",
        color="#6B7280",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def main() -> None:
    args = parse_args()
    if args.window <= 1:
        raise ValueError("--window must be greater than one")
    data = prepare_history(args.history, args.window)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "flat_rule_convergence_smoothed.csv", index=False)
    figure = build_figure(
        data,
        window=args.window,
        budget=args.budget,
        seed=args.seed,
    )
    for suffix in (".png", ".svg"):
        figure.savefig(
            args.output_dir / f"flat_rule_convergence_seed{args.seed}{suffix}",
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
