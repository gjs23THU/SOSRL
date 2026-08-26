"""Create a compact static preview of the MOPSO budget-calibration report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path("runs/mopso_budget_calibration_20260825")
OUTPUT = ROOT / "visual_report"
RECOMMENDED_BUDGET = 400

COLORS = {
    "总体": "#2563EB",
    "B split": "#C2410C",
    "G split": "#15803D",
    "threshold": "#7C3AED",
    "recommended": "#D97706",
    "hv": "#2563EB",
    "j": "#E4572E",
}


def configure_font() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            fm.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = fm.FontProperties(
                fname=candidate
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def label_for(row: pd.Series) -> str:
    if row["group_type"] == "all":
        return "总体"
    if row["group_type"] == "split":
        return "B split" if str(row["group_value"]).startswith("b_") else "G split"
    return {
        "capacity_tight": "容量紧张",
        "feasible_suboptimal": "可行但非最优",
        "missing_capability": "缺少能力",
        "redundant_overbudget": "冗余/超预算",
    }.get(str(row["group_value"]), str(row["group_value"]))


def percent_axis(axis: plt.Axes) -> None:
    def format_percent(value: float, _: float) -> str:
        label = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{label}%"

    axis.yaxis.set_major_formatter(FuncFormatter(format_percent))


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor("#FBFCFE")
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB3C2")
    axis.tick_params(colors="#475569", labelsize=9)
    axis.title.set_color("#172033")


def plot_curves(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for label in ("总体", "B split", "G split"):
        subset = frame[frame["label"] == label].sort_values("evaluation_budget_per_run")
        axis.plot(
            subset["evaluation_budget_per_run"],
            subset[metric] * 100,
            color=COLORS[label],
            linewidth=2.4,
            marker="o",
            markersize=5,
            label=label,
        )
    axis.axhline(
        1.0,
        color=COLORS["threshold"],
        linestyle=(0, (5, 4)),
        linewidth=1.6,
        label="1% 门槛",
    )
    axis.axvline(
        RECOMMENDED_BUDGET,
        color=COLORS["recommended"],
        linestyle=(0, (2, 3)),
        linewidth=1.6,
    )
    axis.text(
        RECOMMENDED_BUDGET + 6,
        axis.get_ylim()[1] * 0.88,
        "推荐 400",
        color=COLORS["recommended"],
        fontsize=9,
        fontweight="bold",
    )
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel("每次独立运行的评价预算", color="#475569")
    axis.set_ylabel(ylabel, color="#475569")
    axis.set_xticks([50, 100, 150, 200, 300, 400, 500])
    percent_axis(axis)
    style_axis(axis)


def main() -> None:
    configure_font()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    curve = pd.read_csv(ROOT / "budget_curve.csv")
    curve["label"] = curve.apply(label_for, axis=1)

    all_splits = curve[curve["group_type"].isin(["all", "split"])].copy()
    at_400 = curve[
        (curve["group_type"] == "category")
        & (curve["evaluation_budget_per_run"] == RECOMMENDED_BUDGET)
    ].copy()
    at_400["category"] = at_400.apply(label_for, axis=1)

    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160)
    figure.patch.set_facecolor("#F4F7FB")
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.085,
        top=0.79,
        wspace=0.24,
        hspace=0.38,
    )

    figure.suptitle(
        "MOPSO 小预算校准结果",
        x=0.075,
        y=0.955,
        ha="left",
        fontsize=24,
        fontweight="bold",
        color="#101828",
    )
    figure.text(
        0.075,
        0.912,
        "8 个正式场景 · 24 次独立运行 · 12,000 次评价 · 无失败或重试",
        ha="left",
        fontsize=11.5,
        color="#475569",
    )
    figure.text(
        0.075,
        0.855,
        "推荐：400 次评价/运行",
        ha="left",
        fontsize=16,
        fontweight="bold",
        color="#7C2D12",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FDBA74",
            "linewidth": 1.2,
        },
    )
    figure.text(
        0.29,
        0.856,
        "成功率 100%   |   总体 HV 损失 0.0498%   |   J 后悔 0%",
        ha="left",
        fontsize=12,
        color="#334155",
    )

    plot_curves(
        axes[0, 0],
        all_splits,
        "median_relative_hv_loss_to_max",
        "A. 超体积质量损失随预算收敛",
        "相对 500 评价档的 HV 损失",
    )
    axes[0, 0].set_ylim(0, 3.0)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=9, loc="upper right")

    plot_curves(
        axes[0, 1],
        all_splits,
        "median_relative_j_regret_to_max",
        "B. GP 对齐 J 后悔值决定最小预算",
        "相对 500 评价档的 J 后悔值",
    )
    axes[0, 1].set_ylim(0, 25)
    axes[0, 1].annotate(
        "300 评价时 B split = 1.613%\n仍高于 1% 门槛",
        xy=(300, 1.613),
        xytext=(215, 8.2),
        arrowprops={"arrowstyle": "->", "color": COLORS["B split"], "lw": 1.2},
        fontsize=9,
        color="#7C2D12",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF7ED", "ec": "#FED7AA"},
    )

    axis = axes[1, 0]
    categories = ["容量紧张", "可行但非最优", "缺少能力", "冗余/超预算"]
    ordered = at_400.set_index("category").reindex(categories)
    x_positions = np.arange(len(categories))
    width = 0.36
    hv_values = ordered["median_relative_hv_loss_to_max"].to_numpy() * 100
    j_values = ordered["median_relative_j_regret_to_max"].to_numpy() * 100
    axis.bar(
        x_positions - width / 2,
        hv_values,
        width,
        color=COLORS["hv"],
        label="HV 损失",
    )
    axis.bar(
        x_positions + width / 2,
        j_values,
        width,
        color=COLORS["j"],
        label="J 后悔",
    )
    axis.axhline(1.0, color=COLORS["threshold"], linestyle=(0, (5, 4)), linewidth=1.6)
    axis.set_xticks(x_positions, categories)
    axis.set_ylim(0, 1.9)
    axis.set_ylabel("相对损失/后悔值", color="#475569")
    axis.set_title(
        "C. 400 评价档的场景类别诊断",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axis.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    percent_axis(axis)
    style_axis(axis)
    for position, value in enumerate(j_values):
        if value > 1:
            axis.text(
                position + width / 2,
                value + 0.06,
                f"{value:.3f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#9A3412",
                fontweight="bold",
            )
    axis.text(
        0.0,
        -0.29,
        "每类仅 2 个场景，作为风险提示而非硬门槛；若类别也设硬门槛，需 500 评价。",
        transform=axis.transAxes,
        fontsize=9,
        color="#64748B",
    )

    axis = axes[1, 1]
    for label in ("总体", "B split", "G split"):
        subset = all_splits[all_splits["label"] == label].sort_values(
            "evaluation_budget_per_run"
        )
        axis.plot(
            subset["evaluation_budget_per_run"],
            subset["mean_pareto_front_size"],
            color=COLORS[label],
            linewidth=2.4,
            marker="o",
            markersize=5,
            label=label,
        )
    axis.axvline(
        RECOMMENDED_BUDGET,
        color=COLORS["recommended"],
        linestyle=(0, (2, 3)),
        linewidth=1.6,
    )
    axis.set_title(
        "D. 平均 Pareto 前沿规模",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    axis.set_xlabel("每次独立运行的评价预算", color="#475569")
    axis.set_ylabel("平均前沿解数量", color="#475569")
    axis.set_xticks([50, 100, 150, 200, 300, 400, 500])
    axis.set_ylim(0, 20)
    axis.legend(frameon=False, ncol=3, fontsize=9, loc="upper left")
    style_axis(axis)

    figure.text(
        0.075,
        0.025,
        "判定口径：成功率不低于 500 评价档，且中位 HV 损失与中位 J 后悔值均不超过 1%。",
        fontsize=9.5,
        color="#64748B",
    )

    png_path = OUTPUT / "mopso_calibration_overview.png"
    svg_path = OUTPUT / "mopso_calibration_overview.svg"
    figure.savefig(png_path, facecolor=figure.get_facecolor())
    figure.savefig(svg_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(png_path.resolve())
    print(svg_path.resolve())


if __name__ == "__main__":
    main()
