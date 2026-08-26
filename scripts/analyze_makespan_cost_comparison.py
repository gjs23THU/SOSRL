"""Compare representative and Pareto-extreme makespan/cost levels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from compare_mopso_nsga2_calibration import (
    CATEGORY_LABELS,
    MOPSO_ROOT,
    NSGA_ROOT,
    load_inputs,
    read_front,
    scenario_directories,
)


OUTPUT = Path("runs/metaheuristic_budget_comparison_20260826")
BUDGETS = [300, 400]
COLORS = {"NSGA-II": "#2563EB", "MOPSO-CD": "#E4572E"}


def configure_font() -> None:
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.exists():
            fm.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = fm.FontProperties(
                fname=candidate
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def scenario_label(row: pd.Series) -> str:
    return (
        f"{str(row['split'])[0].upper()}{int(row['scenario_idx'])} · "
        f"{CATEGORY_LABELS[str(row['category'])]}"
    )


def representative_rows(nsga: pd.DataFrame, mopso: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario_idx",
        "scenario_hash",
        "split",
        "category",
        "evaluation_budget_per_run",
    ]
    values = ["gp_aligned_makespan", "gp_aligned_effective_cost"]
    frame = nsga[nsga["evaluation_budget_per_run"].isin(BUDGETS)][
        keys + values
    ].merge(
        mopso[mopso["evaluation_budget_per_run"].isin(BUDGETS)][keys + values],
        on=keys,
        suffixes=("_nsga", "_mopso"),
        validate="one_to_one",
    )
    frame["scenario_label"] = frame.apply(scenario_label, axis=1)
    frame["makespan_relative_delta_mopso_vs_nsga"] = (
        frame["gp_aligned_makespan_mopso"]
        / frame["gp_aligned_makespan_nsga"]
        - 1
    )
    frame["cost_relative_delta_mopso_vs_nsga"] = (
        frame["gp_aligned_effective_cost_mopso"]
        / frame["gp_aligned_effective_cost_nsga"]
        - 1
    )
    return frame


def pareto_extremes(
    nsga_summary: dict,
    mopso_summary: dict,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for algorithm, summary in (
        ("NSGA-II", nsga_summary),
        ("MOPSO-CD", mopso_summary),
    ):
        directories = scenario_directories(summary)
        for scenario_hash, scenario_dir in directories.items():
            for budget in BUDGETS:
                front = read_front(scenario_dir, budget)
                min_makespan_point = min(front, key=lambda point: (point[0], point[1]))
                min_cost_point = min(front, key=lambda point: (point[1], point[0]))
                rows.append(
                    {
                        "algorithm": algorithm,
                        "scenario_hash": scenario_hash,
                        "evaluation_budget_per_run": budget,
                        "front_size": len(front),
                        "minimum_makespan": min_makespan_point[0],
                        "cost_at_minimum_makespan": min_makespan_point[1],
                        "minimum_effective_cost": min_cost_point[1],
                        "makespan_at_minimum_cost": min_cost_point[0],
                    }
                )
    return pd.DataFrame(rows)


def build_summary(
    representatives: pd.DataFrame,
    extremes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    detailed: dict[str, object] = {}
    for budget in BUDGETS:
        selected = representatives[
            representatives["evaluation_budget_per_run"] == budget
        ]
        budget_detail: dict[str, object] = {}
        for algorithm, suffix in (("NSGA-II", "nsga"), ("MOPSO-CD", "mopso")):
            algorithm_extremes = extremes[
                (extremes["evaluation_budget_per_run"] == budget)
                & (extremes["algorithm"] == algorithm)
            ]
            row = {
                "evaluation_budget_per_run": budget,
                "algorithm": algorithm,
                "scenario_count": len(selected),
                "median_representative_makespan": float(
                    selected[f"gp_aligned_makespan_{suffix}"].median()
                ),
                "mean_representative_makespan": float(
                    selected[f"gp_aligned_makespan_{suffix}"].mean()
                ),
                "median_representative_effective_cost": float(
                    selected[f"gp_aligned_effective_cost_{suffix}"].median()
                ),
                "mean_representative_effective_cost": float(
                    selected[f"gp_aligned_effective_cost_{suffix}"].mean()
                ),
                "median_pareto_minimum_makespan": float(
                    algorithm_extremes["minimum_makespan"].median()
                ),
                "median_cost_at_pareto_minimum_makespan": float(
                    algorithm_extremes["cost_at_minimum_makespan"].median()
                ),
                "median_pareto_minimum_effective_cost": float(
                    algorithm_extremes["minimum_effective_cost"].median()
                ),
                "median_makespan_at_pareto_minimum_cost": float(
                    algorithm_extremes["makespan_at_minimum_cost"].median()
                ),
            }
            rows.append(row)
            budget_detail[algorithm] = row
        budget_detail["paired_comparison_mopso_vs_nsga"] = {
            "median_makespan_relative_delta": float(
                selected["makespan_relative_delta_mopso_vs_nsga"].median()
            ),
            "median_cost_relative_delta": float(
                selected["cost_relative_delta_mopso_vs_nsga"].median()
            ),
            "mopso_makespan_wins": int(
                (selected["makespan_relative_delta_mopso_vs_nsga"] < 0).sum()
            ),
            "nsga2_makespan_wins": int(
                (selected["makespan_relative_delta_mopso_vs_nsga"] > 0).sum()
            ),
            "mopso_cost_wins": int(
                (selected["cost_relative_delta_mopso_vs_nsga"] < 0).sum()
            ),
            "nsga2_cost_wins": int(
                (selected["cost_relative_delta_mopso_vs_nsga"] > 0).sum()
            ),
            "mopso_both_wins": int(
                (
                    (selected["makespan_relative_delta_mopso_vs_nsga"] < 0)
                    & (selected["cost_relative_delta_mopso_vs_nsga"] < 0)
                ).sum()
            ),
            "nsga2_both_wins": int(
                (
                    (selected["makespan_relative_delta_mopso_vs_nsga"] > 0)
                    & (selected["cost_relative_delta_mopso_vs_nsga"] > 0)
                ).sum()
            ),
        }
        detailed[str(budget)] = budget_detail
    payload = {
        "scope": {
            "scenarios": 8,
            "budgets": BUDGETS,
            "representative": "GP-aligned selected solution",
            "makespan_unit": "scheduling time unit",
            "cost_unit": "effective-cost unit",
        },
        "budgets": detailed,
    }
    return pd.DataFrame(rows), payload


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.set_facecolor("#FBFCFE")
    axis.grid(axis=grid_axis, color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB3C2")
    axis.tick_params(colors="#475569", labelsize=9)


def plot_scatter(axis: plt.Axes, frame: pd.DataFrame, budget: int) -> None:
    selected = frame[frame["evaluation_budget_per_run"] == budget]
    x_values = selected["makespan_relative_delta_mopso_vs_nsga"] * 100
    y_values = selected["cost_relative_delta_mopso_vs_nsga"] * 100
    axis.axhline(0, color="#64748B", linewidth=1)
    axis.axvline(0, color="#64748B", linewidth=1)
    axis.scatter(
        x_values,
        y_values,
        s=58,
        color="#E4572E",
        edgecolors="white",
        linewidth=0.8,
        zorder=3,
    )
    for x_value, y_value, label in zip(
        x_values, y_values, selected["scenario_label"]
    ):
        axis.annotate(
            label,
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#334155",
        )
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axis.set_xlabel("MOPSO 相对 NSGA-II 的工期差异", color="#475569")
    axis.set_ylabel("MOPSO 相对 NSGA-II 的有效成本差异", color="#475569")
    axis.set_title(
        f"{budget} 评价档逐场景工期—成本变化",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
    )
    axis.text(
        0.02,
        0.03,
        "左下：MOPSO 两项均优",
        transform=axis.transAxes,
        fontsize=8,
        color="#64748B",
    )
    style_axis(axis, grid_axis="both")


def plot_report(
    representatives: pd.DataFrame,
    level_summary: pd.DataFrame,
    payload: dict[str, object],
) -> None:
    configure_font()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160)
    figure.patch.set_facecolor("#F4F7FB")
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.085,
        top=0.79,
        wspace=0.24,
        hspace=0.4,
    )
    figure.suptitle(
        "MOPSO 与 NSGA-II 的工期—成本水平",
        x=0.075,
        y=0.955,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#101828",
    )
    figure.text(
        0.075,
        0.913,
        "同一8个正式场景；上排为逐场景成对差异，下排为GP对齐代表解的跨场景中位数",
        ha="left",
        fontsize=11.5,
        color="#475569",
    )
    detail_300 = payload["budgets"]["300"]["paired_comparison_mopso_vs_nsga"]
    detail_400 = payload["budgets"]["400"]["paired_comparison_mopso_vs_nsga"]
    figure.text(
        0.075,
        0.855,
        (
            "成对中位变化  |  300评价：工期 "
            f"{100 * detail_300['median_makespan_relative_delta']:+.1f}%、成本 "
            f"{100 * detail_300['median_cost_relative_delta']:+.1f}%  |  "
            "400评价：工期 "
            f"{100 * detail_400['median_makespan_relative_delta']:+.1f}%、成本 "
            f"{100 * detail_400['median_cost_relative_delta']:+.1f}%"
        ),
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#7C2D12",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FDBA74",
            "linewidth": 1.2,
        },
    )

    plot_scatter(axes[0, 0], representatives, 300)
    plot_scatter(axes[0, 1], representatives, 400)

    width = 0.34
    positions = np.arange(len(BUDGETS))
    axis = axes[1, 0]
    for offset, algorithm in ((-width / 2, "NSGA-II"), (width / 2, "MOPSO-CD")):
        selected = level_summary[level_summary["algorithm"] == algorithm]
        bars = axis.bar(
            positions + offset,
            selected["median_representative_makespan"],
            width,
            color=COLORS[algorithm],
            label=algorithm,
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axis.set_xticks(positions, ["300评价", "400评价"])
    axis.set_ylim(0, 650)
    axis.set_ylabel("工期（调度时间单位）", color="#475569")
    axis.set_title(
        "GP对齐代表解的中位工期",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(axis)

    axis = axes[1, 1]
    for offset, algorithm in ((-width / 2, "NSGA-II"), (width / 2, "MOPSO-CD")):
        selected = level_summary[level_summary["algorithm"] == algorithm]
        bars = axis.bar(
            positions + offset,
            selected["median_representative_effective_cost"],
            width,
            color=COLORS[algorithm],
            label=algorithm,
        )
        axis.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axis.set_xticks(positions, ["300评价", "400评价"])
    axis.set_ylim(0, 10_000)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.set_ylabel("effective_cost（成本单位）", color="#475569")
    axis.set_title(
        "GP对齐代表解的中位有效成本",
        loc="left",
        fontsize=12.5,
        fontweight="bold",
    )
    axis.legend(frameon=False, ncol=2, loc="upper left")
    style_axis(axis)

    figure.text(
        0.075,
        0.025,
        "说明：代表解由共同的GP对齐规则选取；负差异表示MOPSO更低。跨场景中位数用于描述水平，逐场景散点用于保留配对关系。",
        fontsize=9.2,
        color="#64748B",
    )
    figure.savefig(OUTPUT / "makespan_cost_comparison.png", facecolor=figure.get_facecolor())
    figure.savefig(OUTPUT / "makespan_cost_comparison.svg", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nsga, mopso, nsga_summary, mopso_summary = load_inputs()
    representatives = representative_rows(nsga, mopso)
    extremes = pareto_extremes(nsga_summary, mopso_summary)
    level_summary, payload = build_summary(representatives, extremes)
    representatives.to_csv(OUTPUT / "makespan_cost_scenario_comparison.csv", index=False)
    extremes.to_csv(OUTPUT / "pareto_extreme_comparison.csv", index=False)
    level_summary.to_csv(OUTPUT / "makespan_cost_level_summary.csv", index=False)
    (OUTPUT / "makespan_cost_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_report(representatives, level_summary, payload)
    print((OUTPUT / "makespan_cost_summary.json").resolve())
    print((OUTPUT / "makespan_cost_comparison.png").resolve())


if __name__ == "__main__":
    main()
