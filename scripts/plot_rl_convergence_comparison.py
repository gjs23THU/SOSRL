"""Plot paper-ready SOSRL convergence curves and export summary tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"SIG": "#1F5A94", "MIG": "#C26A1B"}
LINESTYLES = {"SIG": "-", "MIG": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sig-scheduler-history", type=Path, required=True)
    parser.add_argument("--mig-scheduler-history", type=Path, required=True)
    parser.add_argument("--sig-architecture-history", type=Path, required=True)
    parser.add_argument("--mig-architecture-history", type=Path, required=True)
    parser.add_argument("--scheduler-summary", type=Path, required=True)
    parser.add_argument("--sig-architecture-summary", type=Path, required=True)
    parser.add_argument("--mig-architecture-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--budget", type=float, default=8000.0)
    return parser.parse_args()


def prepare_history(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["episode_number"] = pd.to_numeric(data["episode"]) + 1
    for column in ("success", "budget_violation"):
        if column in data:
            data[column] = data[column].astype(str).str.lower().map(
                {"true": 1.0, "false": 0.0}
            )
    return data


def rolling(data: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(data, errors="coerce")
    mean = numeric.rolling(window, min_periods=max(2, window // 5)).mean()
    std = numeric.rolling(window, min_periods=max(2, window // 5)).std()
    return mean, std


def style_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(colors="#4B5563")


def plot_series(
    axis: plt.Axes,
    histories: dict[str, pd.DataFrame],
    column: str,
    window: int,
    ylabel: str,
    title: str,
    *,
    show_raw: bool = True,
    show_band: bool = True,
    percent: bool = False,
) -> None:
    for label, data in histories.items():
        key = label.split()[0]
        values = pd.to_numeric(data[column], errors="coerce")
        if show_raw:
            axis.plot(
                data["episode_number"],
                values,
                color=COLORS[key],
                alpha=0.09,
                linewidth=0.65,
            )
        mean, std = rolling(values, window)
        axis.plot(
            data["episode_number"],
            mean,
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=2.1,
            label=label,
        )
        if show_band:
            lower = (mean - std).to_numpy(dtype=float)
            upper = (mean + std).to_numpy(dtype=float)
            x = data["episode_number"].to_numpy(dtype=float)
            axis.fill_between(x, lower, upper, color=COLORS[key], alpha=0.10)
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold")
    axis.set_ylabel(ylabel)
    if percent:
        axis.set_ylim(-0.03, 1.03)
        axis.set_yticks(np.linspace(0.0, 1.0, 6))
        axis.set_yticklabels([f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)])
    style_axis(axis)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_scheduler_convergence(
    histories: dict[str, pd.DataFrame], output_dir: Path, window: int
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    plot_series(
        axes[0, 0], histories, "reward", window, "Episode reward",
        "(a) Scheduler reward (higher is better)",
    )
    plot_series(
        axes[0, 1], histories, "makespan", window, "Makespan",
        "(b) Training makespan (lower is better)",
    )
    plot_series(
        axes[1, 0], histories, "loss", window, "TD loss (log scale)",
        "(c) Scheduler DQN loss", show_raw=False,
    )
    axes[1, 0].set_yscale("log")
    plot_series(
        axes[1, 1], histories, "epsilon", window, "Epsilon",
        "(d) Exploration schedule", show_raw=False, show_band=False,
    )
    axes[1, 1].set_ylim(0.0, 1.03)
    for axis in axes[1, :]:
        axis.set_xlabel("Training episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Scheduler DQN convergence: SIG vs MIG",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        f"{window}-episode moving average; shading is local rolling +/-1 SD, not a multi-seed confidence interval",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, output_dir / "scheduler_convergence_sig_mig")


def plot_architecture_convergence(
    histories: dict[str, pd.DataFrame],
    output_dir: Path,
    window: int,
    budget: float,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    plot_series(
        axes[0, 0], histories, "architecture_reward", window,
        "Architecture reward", "(a) Architecture reward (higher is better)",
    )
    plot_series(
        axes[0, 1], histories, "makespan", window,
        "Makespan", "(b) Mission makespan (lower is better)",
    )
    plot_series(
        axes[1, 0], histories, "net_cost", window,
        "Final net cost", "(c) Mission net cost (lower is better)",
    )
    axes[1, 0].axhline(
        budget,
        color="#374151",
        linestyle=":",
        linewidth=1.7,
        label=f"Budget = {budget:,.0f}",
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    plot_series(
        axes[1, 1], histories, "architecture_changes", window,
        "Changes per mission", "(d) Architecture change count (lower is better)",
    )
    plot_series(
        axes[2, 0], histories, "architecture_loss", window,
        "TD loss (log scale)", "(e) Architecture DQN loss", show_raw=False,
    )
    axes[2, 0].set_yscale("log")
    for label, data in histories.items():
        key = label.split()[0]
        success, _ = rolling(data["success"], window)
        violation, _ = rolling(data["budget_violation"], window)
        axes[2, 1].plot(
            data["episode_number"], success,
            color=COLORS[key], linestyle=LINESTYLES[key], linewidth=2.1,
            label=f"{label}: success",
        )
        axes[2, 1].plot(
            data["episode_number"], violation,
            color=COLORS[key], linestyle=":", linewidth=1.7,
            label=f"{label}: final over budget",
        )
    axes[2, 1].set_title(
        "(f) Rolling success and final-over-budget rates",
        loc="left", fontsize=11, fontweight="semibold",
    )
    axes[2, 1].set_ylabel("Rolling rate")
    axes[2, 1].set_ylim(-0.03, 1.03)
    axes[2, 1].set_yticks(np.linspace(0.0, 1.0, 6))
    axes[2, 1].set_yticklabels(
        [f"{value:.0%}" for value in np.linspace(0.0, 1.0, 6)]
    )
    axes[2, 1].legend(frameon=False, fontsize=8, ncol=2)
    style_axis(axes[2, 1])
    for axis in axes[2, :]:
        axis.set_xlabel("Training episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "Architecture DQN convergence with SIG and MIG schedulers",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.948,
        f"1,000 training episodes; {window}-episode moving average; shading is local rolling +/-1 SD",
        ha="center",
        color="#4B5563",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, output_dir / "architecture_convergence_sig_mig")


def interval_summary(
    name: str, data: pd.DataFrame, columns: list[str]
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"method": name}
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        row[f"{column}_first100"] = values.iloc[:100].mean()
        row[f"{column}_last100"] = values.iloc[-100:].mean()
    return row


def export_tables(
    scheduler_histories: dict[str, pd.DataFrame],
    architecture_histories: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    training_rows = []
    for name, data in scheduler_histories.items():
        training_rows.append(
            interval_summary(name, data, ["reward", "makespan", "loss", "success"])
        )
    for name, data in architecture_histories.items():
        training_rows.append(
            interval_summary(
                name,
                data,
                [
                    "architecture_reward",
                    "makespan",
                    "net_cost",
                    "architecture_changes",
                    "architecture_loss",
                    "success",
                    "budget_violation",
                ],
            )
        )
    pd.DataFrame(training_rows).to_csv(
        args.output_dir / "training_convergence_summary.csv", index=False
    )

    scheduler_summary = pd.read_csv(args.scheduler_summary).set_index("model")
    sig_arch = pd.read_csv(args.sig_architecture_summary).set_index("model").loc["hrl"]
    mig_arch = pd.read_csv(args.mig_architecture_summary).set_index("model").loc["hrl"]
    evaluation_rows = [
        {
            "method": "SIG Scheduler",
            "evaluation_scope": "scheduler-only test pool",
            "episodes": scheduler_summary.loc["SIG", "evaluation_scenarios"],
            "success_rate": scheduler_summary.loc["SIG", "success_rate"],
            "mean_success_makespan": scheduler_summary.loc["SIG", "mean_success_makespan"],
        },
        {
            "method": "MIG Scheduler",
            "evaluation_scope": "scheduler-only test pool",
            "episodes": scheduler_summary.loc["MIG", "evaluation_scenarios"],
            "success_rate": scheduler_summary.loc["MIG", "success_rate"],
            "mean_success_makespan": scheduler_summary.loc["MIG", "mean_success_makespan"],
        },
        {
            "method": "SIG + Arch1000",
            "evaluation_scope": "defective-architecture HRL test pool",
            "episodes": sig_arch["episodes"],
            "success_rate": sig_arch["success_rate"],
            "mean_success_makespan": sig_arch["mean_success_makespan"],
            "mean_final_net_cost": sig_arch["mean_net_cost"],
            "mean_peak_net_cost": sig_arch["mean_peak_net_cost"],
            "mean_architecture_changes": sig_arch["mean_architecture_changes"],
            "final_over_budget_rate": sig_arch["budget_violation_rate"],
            "ever_over_budget_rate": sig_arch["ever_budget_violation_rate"],
        },
        {
            "method": "MIG + Arch1000",
            "evaluation_scope": "defective-architecture HRL test pool",
            "episodes": mig_arch["episodes"],
            "success_rate": mig_arch["success_rate"],
            "mean_success_makespan": mig_arch["mean_success_makespan"],
            "mean_final_net_cost": mig_arch["mean_net_cost"],
            "mean_peak_net_cost": mig_arch["mean_peak_net_cost"],
            "mean_architecture_changes": mig_arch["mean_architecture_changes"],
            "final_over_budget_rate": mig_arch["budget_violation_rate"],
            "ever_over_budget_rate": mig_arch["ever_budget_violation_rate"],
        },
    ]
    pd.DataFrame(evaluation_rows).to_csv(
        args.output_dir / "evaluation_key_results.csv", index=False
    )


def main() -> None:
    args = parse_args()
    if args.window <= 1:
        raise ValueError("--window must be greater than one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_histories = {
        "SIG Scheduler": prepare_history(args.sig_scheduler_history),
        "MIG Scheduler": prepare_history(args.mig_scheduler_history),
    }
    architecture_histories = {
        "SIG + Arch": prepare_history(args.sig_architecture_history),
        "MIG + Arch": prepare_history(args.mig_architecture_history),
    }
    plot_scheduler_convergence(scheduler_histories, args.output_dir, args.window)
    plot_architecture_convergence(
        architecture_histories, args.output_dir, args.window, args.budget
    )
    export_tables(scheduler_histories, architecture_histories, args)


if __name__ == "__main__":
    main()
