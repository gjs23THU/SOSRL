from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "runs" / "round1_ss_tuning" / "comparisons"

SOURCES = {
    "Fixed (V1)": ROOT
    / "runs"
    / "round1_formal"
    / "bdqn"
    / "convergence"
    / "fixed"
    / "seed_1"
    / "validation"
    / "checkpoint_summary.csv",
    "Arch (V1)": ROOT
    / "runs"
    / "round1_formal"
    / "bdqn"
    / "convergence"
    / "arch"
    / "seed_1"
    / "validation"
    / "checkpoint_summary.csv",
    "G0 (V1)": ROOT
    / "runs"
    / "round1_formal"
    / "bdqn"
    / "convergence"
    / "g0"
    / "seed_1"
    / "validation"
    / "checkpoint_summary.csv",
    "SS (LR decay)": ROOT
    / "runs"
    / "round1_ss_tuning"
    / "lr_decay_0p9975_to320k"
    / "seed_1"
    / "validation"
    / "checkpoint_summary.csv",
}

COLORS = {
    "Fixed (V1)": "#6B7280",
    "Arch (V1)": "#3568A8",
    "G0 (V1)": "#D97706",
    "SS (LR decay)": "#B83B78",
}

MARKERS = {
    "Fixed (V1)": "o",
    "Arch (V1)": "s",
    "G0 (V1)": "^",
    "SS (LR decay)": "D",
}


def load_overall_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["category"] == "all"]
    return [
        {
            "steps_k": float(row["target_environment_steps"]) / 1000.0,
            "failure_rate": float(row["failure_rate"]),
            "mean_j": float(row["mean_j"]),
            "makespan": float(row["mean_success_makespan"]),
        }
        for row in rows
    ]


def cumulative_min(values: list[float]) -> list[float]:
    result: list[float] = []
    best = float("inf")
    for value in values:
        best = min(best, value)
        result.append(best)
    return result


def export_chart_data(data: dict[str, list[dict[str, float]]]) -> Path:
    output = OUTPUT_DIR / "ss_vs_v1_seed1_chart_data.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "provider",
            "steps_k",
            "failure_rate",
            "best_failure_rate",
            "mean_j",
            "best_mean_j",
            "makespan",
            "best_makespan",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for provider, rows in data.items():
            failure = [row["failure_rate"] for row in rows]
            mean_j = [row["mean_j"] for row in rows]
            makespan = [row["makespan"] for row in rows]
            for row, best_failure, best_j, best_makespan in zip(
                rows,
                cumulative_min(failure),
                cumulative_min(mean_j),
                cumulative_min(makespan),
                strict=True,
            ):
                writer.writerow(
                    {
                        "provider": provider,
                        **row,
                        "best_failure_rate": best_failure,
                        "best_mean_j": best_j,
                        "best_makespan": best_makespan,
                    }
                )
    return output


def main() -> None:
    for provider, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {provider} summary: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {provider: load_overall_rows(path) for provider, path in SOURCES.items()}
    export_chart_data(data)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": "#8A9199",
            "axes.labelcolor": "#30343B",
            "xtick.color": "#505761",
            "ytick.color": "#505761",
            "text.color": "#22262C",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("#FBFCFE")

    panels = [
        ("failure_rate", "失败率", "失败率（越低越好）"),
        ("mean_j", "Failure-aware mean J", "Mean J（越低越好）"),
        ("makespan", "成功任务平均 Makespan", "Makespan（越低越好）"),
    ]

    for axis, (field, title, ylabel) in zip(axes, panels, strict=True):
        axis.set_facecolor("#FBFCFE")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.8, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_title(title, loc="left", fontsize=13, fontweight="semibold", pad=8)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.axvline(200, color="#89919B", linewidth=1.0, linestyle=":", zorder=0)
        axis.axvspan(200, 320, color="#E9EDF3", alpha=0.32, zorder=0)

        for provider, rows in data.items():
            x = [row["steps_k"] for row in rows]
            raw = [row[field] for row in rows]
            best = cumulative_min(raw)
            axis.plot(
                x,
                raw,
                color=COLORS[provider],
                linewidth=1.05,
                linestyle="--",
                alpha=0.30,
                zorder=1,
            )
            axis.plot(
                x,
                best,
                color=COLORS[provider],
                linewidth=2.5,
                marker=MARKERS[provider],
                markersize=5.5,
                markerfacecolor="#FBFCFE",
                markeredgewidth=1.5,
                zorder=3,
            )

        if field == "failure_rate":
            axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            axis.set_ylim(-0.035, 1.02)

    axes[0].text(
        205,
        0.965,
        "仅 SS 延长训练",
        color="#626A74",
        fontsize=9,
        va="top",
    )
    axes[-1].set_xlabel("Environment steps（k）", fontsize=11)
    axes[-1].set_xlim(-5, 325)
    axes[-1].set_xticks(range(0, 321, 40))

    provider_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[provider],
            marker=MARKERS[provider],
            markerfacecolor="#FBFCFE",
            markeredgewidth=1.5,
            linewidth=2.5,
            label=provider,
        )
        for provider in SOURCES
    ]
    style_handles = [
        Line2D([0], [0], color="#343A40", linewidth=2.5, label="截至该步最佳"),
        Line2D(
            [0],
            [0],
            color="#7D858E",
            linewidth=1.1,
            linestyle="--",
            alpha=0.55,
            label="原始 checkpoint",
        ),
    ]
    fig.legend(
        handles=provider_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
        fontsize=9.5,
        handlelength=2.5,
    )

    fig.suptitle(
        "SS 与 V1 上层 Provider 的 BDQN 验证收敛对比",
        fontsize=18,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.949,
        "Seed 1 · B-validation 256场景 · 粗线为累计最佳，浅虚线为原始值 · Test-v2未使用",
        ha="center",
        fontsize=10.5,
        color="#5A626C",
    )
    fig.text(
        0.01,
        0.012,
        "数据源：round1_formal V1（Fixed / Arch / G0）与 SS lr_decay=0.9975 连续320k实验。",
        fontsize=8.5,
        color="#68717C",
    )
    fig.subplots_adjust(top=0.885, bottom=0.075, left=0.095, right=0.975, hspace=0.27)

    png_path = OUTPUT_DIR / "ss_vs_v1_seed1_convergence.png"
    svg_path = OUTPUT_DIR / "ss_vs_v1_seed1_convergence.svg"
    fig.savefig(png_path, dpi=190, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(png_path)
    print(svg_path)


if __name__ == "__main__":
    main()
