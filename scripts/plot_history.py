"""Plot a scheduler or architecture training history without hard-coded paths."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoothed-csv", type=Path)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--title", default="SOSRL training history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    data = pd.read_csv(args.history)
    if "episode" not in data or "makespan" not in data:
        raise ValueError("history must contain episode and makespan columns")

    data["episode_number"] = data["episode"].astype(int) + 1
    data["makespan_smooth"] = data["makespan"].rolling(
        args.window,
        min_periods=1,
    ).mean()
    has_cost = "net_cost" in data.columns
    if has_cost:
        data["net_cost_smooth"] = data["net_cost"].rolling(
            args.window,
            min_periods=1,
        ).mean()

    columns = ["episode_number", "makespan", "makespan_smooth"]
    for optional in ("success", "epsilon", "loss"):
        if optional in data.columns:
            columns.append(optional)
    if has_cost:
        columns.extend(["net_cost", "net_cost_smooth"])
    csv_path = args.smoothed_csv or args.output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data[columns].to_csv(csv_path, index=False)

    row_count = 2 if has_cost else 1
    fig, axes = plt.subplots(
        row_count,
        1,
        figsize=(14, 8 if has_cost else 6),
        sharex=has_cost,
        squeeze=False,
    )
    x = data["episode_number"]
    makespan_axis = axes[0, 0]
    makespan_axis.plot(x, data["makespan"], color="#93A4B8", alpha=0.28)
    makespan_axis.plot(
        x,
        data["makespan_smooth"],
        color="#1F5A94",
        linewidth=2.1,
        label=f"{args.window}-episode moving average",
    )
    makespan_axis.set_ylabel("Makespan")
    makespan_axis.legend(frameon=False)

    if has_cost:
        cost_axis = axes[1, 0]
        cost_axis.plot(x, data["net_cost"], color="#D8B27C", alpha=0.30)
        cost_axis.plot(
            x,
            data["net_cost_smooth"],
            color="#C26A1B",
            linewidth=2.1,
            label=f"{args.window}-episode moving average",
        )
        cost_axis.axhline(
            args.budget,
            color="#374151",
            linestyle="--",
            label=f"Budget = {args.budget:g}",
        )
        cost_axis.set_ylabel("Net cost")
        cost_axis.legend(frameon=False)

    for axis in axes[:, 0]:
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1, 0].set_xlabel("Episode")
    fig.suptitle(args.title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
