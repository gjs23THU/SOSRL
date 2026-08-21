"""Plot equal-step training curves for rule Scheduler DQN and additive BDQN."""

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
    parser.add_argument("--scheduler-history", type=Path, required=True)
    parser.add_argument("--branching-history", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4)
    return parser.parse_args()


def boolean_series(values: pd.Series) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    result = normalized.map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    if result.isna().any():
        raise ValueError("history contains an unknown boolean value")
    return result


def prepare_scheduler(path: Path, window: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "assigned_ops",
        "reward",
        "success",
        "dead_end",
        "makespan",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"scheduler history is missing: {sorted(missing)}")
    success = boolean_series(data["success"])
    dead_end = boolean_series(data["dead_end"])
    data["training_steps"] = pd.to_numeric(
        data["assigned_ops"], errors="raise"
    ).cumsum()
    # Legacy history stores only the telescoping base reward. Add the same
    # terminal terms used by the branching workflow for an honest comparison.
    data["comparable_reward"] = (
        pd.to_numeric(data["reward"], errors="raise")
        + success
        - 2.0 * dead_end
    )
    data["successful_makespan"] = pd.to_numeric(
        data["makespan"], errors="raise"
    ).where(success > 0)
    data["method"] = "Rule-DQN Scheduler"
    return add_smoothing(data, window)


def prepare_branching(path: Path, window: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "total_env_steps",
        "scheduler_reward",
        "success",
        "makespan",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"branching history is missing: {sorted(missing)}")
    success = boolean_series(data["success"])
    data["training_steps"] = pd.to_numeric(
        data["total_env_steps"], errors="raise"
    )
    data["comparable_reward"] = pd.to_numeric(
        data["scheduler_reward"], errors="raise"
    )
    data["successful_makespan"] = pd.to_numeric(
        data["makespan"], errors="raise"
    ).where(success > 0)
    data["method"] = "Additive BDQN"
    return add_smoothing(data, window)


def add_smoothing(data: pd.DataFrame, window: int) -> pd.DataFrame:
    minimum = max(5, window // 5)
    for column in ("comparable_reward", "successful_makespan"):
        data[f"{column}_smooth"] = data[column].rolling(
            window,
            min_periods=minimum,
        ).mean()
    return data


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.tick_params(colors="#4B5563")
    axis.set_xlabel("Cumulative assignment steps")


def plot_curve(
    scheduler: pd.DataFrame,
    branching: pd.DataFrame,
    *,
    column: str,
    ylabel: str,
    title: str,
    output_stem: Path,
    window: int,
    seed: int,
    budget_steps: int,
) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 6.5))
    specifications = (
        (scheduler, "Rule-DQN Scheduler", BLUE, "--", "o"),
        (branching, "Additive BDQN", ORANGE, "-", "s"),
    )
    for data, label, color, linestyle, marker in specifications:
        raw = data.dropna(subset=[column])
        axis.scatter(
            raw["training_steps"],
            raw[column],
            color=color,
            alpha=0.08,
            s=7,
            marker=marker,
            linewidths=0,
            rasterized=True,
        )
        smooth_column = f"{column}_smooth"
        smooth = data.dropna(subset=[smooth_column])
        axis.plot(
            smooth["training_steps"],
            smooth[smooth_column],
            color=color,
            linestyle=linestyle,
            linewidth=2.35,
            label=label,
        )
        endpoint = float(smooth[smooth_column].iloc[-1])
        endpoint_x = float(smooth["training_steps"].iloc[-1])
        axis.scatter(
            [endpoint_x],
            [endpoint],
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=5,
        )
        axis.annotate(
            f"{endpoint:.3f}" if column == "comparable_reward" else f"{endpoint:.1f}",
            (endpoint_x, endpoint),
            xytext=(-6, 9 if label.startswith("Additive") else -15),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=9,
            fontweight="semibold",
        )

    scheduler_steps = int(scheduler["training_steps"].iloc[-1])
    branching_steps = int(branching["training_steps"].iloc[-1])
    axis.axvline(
        budget_steps,
        color=INK,
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
    )
    axis.text(
        budget_steps,
        0.54 if column == "comparable_reward" else 0.02,
        f" matched training budget\n {budget_steps:,}",
        transform=axis.get_xaxis_transform(),
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8,
    )
    axis.set_xlim(
        0,
        max(scheduler_steps, branching_steps, budget_steps) * 1.035,
    )
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK)
    axis.legend(frameon=False, loc="best")
    style_axis(axis)
    fig.text(
        0.125,
        0.98,
        (
            f"Seed {seed} | x-axis is cumulative assignment steps | "
            f"{window}-episode moving mean; faint markers are raw episodes\n"
            "Single runs. Rule-DQN uses one static shared mission; BDQN uses "
            "100 adaptive missions with the frozen Architecture Provider.\n"
            f"Matched budget: {budget_steps:,} steps; completed episodes end at "
            f"{scheduler_steps:,} (Rule-DQN) and {branching_steps:,} (BDQN)."
        ),
        color=MUTED,
        fontsize=8.5,
        linespacing=1.45,
        va="top",
    )
    if column == "comparable_reward":
        zoom = axis.inset_axes([0.39, 0.13, 0.48, 0.39])
        for data, _label, color, linestyle, marker in specifications:
            raw = data.dropna(subset=[column])
            zoom.scatter(
                raw["training_steps"],
                raw[column],
                color=color,
                alpha=0.06,
                s=5,
                marker=marker,
                linewidths=0,
                rasterized=True,
            )
            smooth_column = f"{column}_smooth"
            smooth = data.dropna(subset=[smooth_column])
            zoom.plot(
                smooth["training_steps"],
                smooth[smooth_column],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
            )
        zoom.set_xlim(axis.get_xlim())
        zoom.set_ylim(0.4, 0.9)
        zoom.set_title("Zoom: normal training band", loc="left", fontsize=8)
        zoom.grid(axis="y", color=GRID, linewidth=0.65)
        zoom.tick_params(labelsize=7, colors="#4B5563")
        for spine in zoom.spines.values():
            spine.set_color("#9CA3AF")
    fig.tight_layout(rect=(0, 0, 1, 0.835))
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
    if args.window <= 1:
        raise ValueError("--window must be greater than one")
    scheduler = prepare_scheduler(args.scheduler_history, args.window)
    branching = prepare_branching(args.branching_history, args.window)
    if min(len(scheduler), len(branching)) < 8:
        raise ValueError("each history needs at least eight episodes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    budget_steps = 120_000
    pd.concat(
        [scheduler, branching],
        ignore_index=True,
        sort=False,
    ).to_csv(args.output_dir / "equal_step_training_curves.csv", index=False)
    plot_curve(
        scheduler,
        branching,
        column="successful_makespan",
        ylabel="Successful mission makespan",
        title="Successful mission makespan over training steps",
        output_stem=args.output_dir / "makespan_vs_training_steps",
        window=args.window,
        seed=args.seed,
        budget_steps=budget_steps,
    )
    plot_curve(
        scheduler,
        branching,
        column="comparable_reward",
        ylabel="Comparable scheduler reward",
        title="Scheduler reward over training steps",
        output_stem=args.output_dir / "reward_vs_training_steps",
        window=args.window,
        seed=args.seed,
        budget_steps=budget_steps,
    )


if __name__ == "__main__":
    main()
