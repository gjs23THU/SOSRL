"""Plot round-one mean convergence curves for completed seeds 1--4.

The primary figure keeps Arch and G0 as ordinary seed means.  For Fixed it
shows both the ordinary mean (light dashed line) and the requested
best-so-far envelope.  The envelope is computed within each seed first and
then averaged, so a lucky checkpoint from one seed is not substituted into
another seed's trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


PROVIDERS = ("fixed", "arch", "g0")
PROVIDER_LABELS = {"fixed": "Fixed", "arch": "Arch", "g0": "G0"}
COLORS = {"fixed": "#D9911B", "arch": "#2B6CB0", "g0": "#C44569"}
MARKERS = {"fixed": "o", "arch": "s", "g0": "^"}
CHECKPOINTS = (0, 20_000, 40_000, 60_000, 80_000, 120_000, 160_000, 200_000)

METRICS = {
    "failure_rate": {
        "title": "失败率",
        "ylabel": "Failure rate",
        "direction": "min",
        "percent": True,
        "source": "validation",
    },
    "mean_j": {
        "title": "Failure-aware mean J",
        "ylabel": "Mean J",
        "direction": "min",
        "percent": False,
        "source": "validation",
    },
    "mean_success_makespan": {
        "title": "成功场景 Makespan",
        "ylabel": "Makespan",
        "direction": "min",
        "percent": False,
        "source": "validation",
    },
    "scheduler_loss": {
        "title": "BDQN TD Loss",
        "ylabel": "Mean TD loss (log scale)",
        "direction": "min",
        "percent": False,
        "source": "training",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path(r"E:\LocalProject\SOSRL\runs\round1_formal"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to STUDY_DIR/analysis/seed1_4_mean_curves.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--loss-window-steps",
        type=int,
        default=20_000,
        help="Trailing environment-step window used for checkpoint TD loss.",
    )
    return parser.parse_args()


def configure_style() -> None:
    available = {item.name for item in font_manager.fontManager.ttflist}
    preferred = [
        name
        for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans")
        if name in available
    ]
    plt.rcParams.update(
        {
            "font.family": preferred or ["DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#374151",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "text.color": "#1F2937",
        }
    )


def read_validation(study_dir: Path, seeds: list[int]) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for provider in PROVIDERS:
        for seed in seeds:
            path = (
                study_dir
                / "bdqn"
                / "convergence"
                / provider
                / f"seed_{seed}"
                / "validation"
                / "checkpoint_summary.csv"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame = frame.loc[frame["category"].eq("all")].copy()
            frame["provider"] = provider
            frame["seed"] = seed
            frame["step"] = pd.to_numeric(
                frame["target_environment_steps"], errors="raise"
            ).astype(int)
            for metric in (
                "failure_rate",
                "mean_j",
                "mean_success_makespan",
            ):
                frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
                frame.loc[~np.isfinite(frame[metric]), metric] = np.nan
            records.append(
                frame[
                    [
                        "provider",
                        "seed",
                        "step",
                        "failure_rate",
                        "mean_j",
                        "mean_success_makespan",
                    ]
                ]
            )
    result = pd.concat(records, ignore_index=True)
    expected = set(CHECKPOINTS)
    for (provider, seed), group in result.groupby(["provider", "seed"]):
        actual = set(group["step"].astype(int))
        if actual != expected:
            raise ValueError(
                f"Unexpected checkpoints for {provider} seed {seed}: {sorted(actual)}"
            )
    return result


def read_checkpoint_losses(
    study_dir: Path, seeds: list[int], window_steps: int
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for provider in PROVIDERS:
        for seed in seeds:
            path = (
                study_dir
                / "bdqn"
                / "convergence"
                / provider
                / f"seed_{seed}"
                / "training_history.csv"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            history = pd.read_csv(
                path,
                usecols=["total_env_steps", "scheduler_loss"],
                low_memory=False,
            )
            history["total_env_steps"] = pd.to_numeric(
                history["total_env_steps"], errors="coerce"
            )
            history["scheduler_loss"] = pd.to_numeric(
                history["scheduler_loss"], errors="coerce"
            )
            for checkpoint in CHECKPOINTS:
                if checkpoint == 0:
                    value = np.nan
                else:
                    lower = max(0, checkpoint - window_steps)
                    sample = history.loc[
                        history["total_env_steps"].gt(lower)
                        & history["total_env_steps"].le(checkpoint),
                        "scheduler_loss",
                    ].dropna()
                    value = float(sample.mean()) if not sample.empty else np.nan
                rows.append(
                    {
                        "provider": provider,
                        "seed": seed,
                        "step": checkpoint,
                        "scheduler_loss": value,
                    }
                )
    return pd.DataFrame(rows)


def add_best_so_far(data: pd.DataFrame) -> pd.DataFrame:
    output = data.sort_values(["provider", "seed", "step"]).copy()
    for metric, spec in METRICS.items():
        transformed: list[pd.Series] = []
        for _, group in output.groupby(["provider", "seed"], sort=False):
            values = group[metric].copy()
            if spec["direction"] == "min":
                best = values.cummin(skipna=True)
            else:
                best = values.cummax(skipna=True)
            best.index = group.index
            transformed.append(best)
        output[f"{metric}_best_so_far"] = pd.concat(transformed).sort_index()
    return output


def summarize(data: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for provider in PROVIDERS:
        for step in CHECKPOINTS:
            sample = data.loc[
                data["provider"].eq(provider) & data["step"].eq(step)
            ]
            for metric in METRICS:
                for mode, column in (
                    ("raw", metric),
                    ("best_so_far", f"{metric}_best_so_far"),
                ):
                    values = pd.to_numeric(sample[column], errors="coerce").dropna()
                    complete_seed_mean = not (
                        metric == "mean_success_makespan" and len(values) < len(seeds)
                    )
                    records.append(
                        {
                            "provider": provider,
                            "step": step,
                            "metric": metric,
                            "mode": mode,
                            "mean": (
                                float(values.mean())
                                if not values.empty and complete_seed_mean
                                else np.nan
                            ),
                            "sd": (
                                float(values.std(ddof=1))
                                if len(values) > 1 and complete_seed_mean
                                else np.nan
                            ),
                            "n_seeds": int(len(values)),
                            "requested_seeds": len(seeds),
                        }
                    )
    return pd.DataFrame(records)


def style_axis(axis: plt.Axes, metric: str) -> None:
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xticks(CHECKPOINTS)
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: "0" if value == 0 else f"{int(value / 1000)}k")
    )
    if METRICS[metric]["percent"]:
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0%}"))
        axis.set_ylim(0, 1.02)
    if metric == "scheduler_loss":
        axis.set_yscale("log")


def draw_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    *,
    fixed_envelope: bool,
    show_title: bool = True,
) -> None:
    spec = METRICS[metric]
    for provider in PROVIDERS:
        mode = "best_so_far" if provider == "fixed" and fixed_envelope else "raw"
        subset = summary.loc[
            summary["provider"].eq(provider)
            & summary["metric"].eq(metric)
            & summary["mode"].eq(mode)
        ].sort_values("step")
        x = subset["step"].to_numpy(dtype=float)
        mean = subset["mean"].to_numpy(dtype=float)
        sd = subset["sd"].fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(mean)
        label = PROVIDER_LABELS[provider]
        if provider == "fixed" and fixed_envelope:
            label = "Fixed（前缀最优）"
            raw = summary.loc[
                summary["provider"].eq("fixed")
                & summary["metric"].eq(metric)
                & summary["mode"].eq("raw")
            ].sort_values("step")
            raw_mean = raw["mean"].to_numpy(dtype=float)
            raw_valid = np.isfinite(raw_mean)
            axis.plot(
                x[raw_valid],
                raw_mean[raw_valid],
                color=COLORS[provider],
                linestyle="--",
                linewidth=1.4,
                alpha=0.38,
                label="Fixed（原始均值）",
            )
        axis.plot(
            x[valid],
            mean[valid],
            color=COLORS[provider],
            marker=MARKERS[provider],
            markersize=5.0,
            markerfacecolor="white" if provider != "g0" else COLORS[provider],
            markeredgewidth=1.2,
            linewidth=2.25,
            label=label,
        )
        lower = mean - sd
        upper = mean + sd
        if spec["percent"]:
            lower = np.clip(lower, 0.0, 1.0)
            upper = np.clip(upper, 0.0, 1.0)
        band_valid = valid & np.isfinite(lower) & np.isfinite(upper)
        if metric == "scheduler_loss":
            band_valid &= lower > 0
        axis.fill_between(
            x[band_valid],
            lower[band_valid],
            upper[band_valid],
            color=COLORS[provider],
            alpha=0.10,
            linewidth=0,
        )
    if show_title:
        axis.set_title(spec["title"], loc="left", fontsize=12, fontweight="semibold")
    axis.set_xlabel("Environment steps")
    axis.set_ylabel(spec["ylabel"])
    style_axis(axis, metric)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_combined(
    summary: pd.DataFrame,
    output_dir: Path,
    *,
    fixed_envelope: bool,
    seeds: list[int],
    loss_window_steps: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0))
    for axis, metric in zip(axes.flat, METRICS, strict=True):
        draw_metric(axis, summary, metric, fixed_envelope=fixed_envelope)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.923),
        ncol=len(unique),
        frameon=False,
        fontsize=10,
    )
    title = "Round 1 BDQN 收敛曲线：seed 1–4 平均"
    if fixed_envelope:
        title += "（Fixed 前缀最优）"
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.99)
    subtitle = (
        f"B-validation；seeds={','.join(map(str, seeds))}；阴影为跨 seed ±1 SD。"
        f"TD loss 为前 {loss_window_steps // 1000}k steps 窗口均值。"
    )
    if fixed_envelope:
        subtitle += " Fixed 对越低越好的指标逐 seed 取截至当前 checkpoint 的最小值。"
    fig.text(0.5, 0.952, subtitle, ha="center", color="#4B5563", fontsize=9.5)
    fig.text(
        0.5,
        0.018,
        "注：makespan 仅对成功场景定义；不足4个seed具有有限值的早期 checkpoint 不绘制。",
        ha="center",
        color="#6B7280",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.035, 0.05, 0.985, 0.91), h_pad=2.3, w_pad=2.0)
    suffix = "fixed_best_so_far" if fixed_envelope else "raw"
    save_figure(fig, output_dir / f"round1_seed1_4_mean_curves_{suffix}")


def plot_individuals(
    summary: pd.DataFrame,
    output_dir: Path,
    seeds: list[int],
) -> None:
    for metric, spec in METRICS.items():
        fig, axis = plt.subplots(figsize=(10.8, 6.6))
        draw_metric(axis, summary, metric, fixed_envelope=True, show_title=False)
        axis.set_title(
            f"{spec['title']}：seed 1–4 平均",
            loc="left",
            fontsize=15,
            fontweight="bold",
        )
        axis.text(
            0.0,
            1.015,
            "Fixed 同时显示原始均值与前缀最优；阴影为跨 seed ±1 SD",
            transform=axis.transAxes,
            color="#4B5563",
            fontsize=9.5,
        )
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(
            unique.values(),
            unique.keys(),
            frameon=False,
            ncol=2,
            loc="best",
            fontsize=9,
        )
        if metric == "mean_success_makespan":
            fig.text(
                0.5,
                0.012,
                "makespan 仅对成功场景定义；不足4个seed具有有限值的早期 checkpoint 不绘制。",
                ha="center",
                color="#6B7280",
                fontsize=8.5,
            )
        fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.96))
        save_figure(fig, output_dir / f"round1_seed1_4_{metric}")


def main() -> None:
    args = parse_args()
    study_dir = args.study_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else study_dir / "analysis" / "seed1_4_mean_curves"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    validation = read_validation(study_dir, args.seeds)
    losses = read_checkpoint_losses(study_dir, args.seeds, args.loss_window_steps)
    data = validation.merge(losses, on=["provider", "seed", "step"], how="left")
    data = add_best_so_far(data)
    summary = summarize(data, args.seeds)

    plot_combined(
        summary,
        output_dir,
        fixed_envelope=False,
        seeds=args.seeds,
        loss_window_steps=args.loss_window_steps,
    )
    plot_combined(
        summary,
        output_dir,
        fixed_envelope=True,
        seeds=args.seeds,
        loss_window_steps=args.loss_window_steps,
    )
    plot_individuals(summary, output_dir, args.seeds)

    data.to_csv(output_dir / "round1_seed1_4_seed_level_plot_data.csv", index=False)
    summary.to_csv(output_dir / "round1_seed1_4_summary_plot_data.csv", index=False)
    manifest = {
        "study_dir": str(study_dir),
        "seeds": args.seeds,
        "providers": list(PROVIDERS),
        "checkpoints": list(CHECKPOINTS),
        "loss_window_steps": args.loss_window_steps,
        "fixed_transform": (
            "Within each seed, use the minimum value observed through checkpoint n "
            "for all four lower-is-better metrics; then aggregate across seeds."
        ),
        "makespan_note": (
            "Mean success makespan is undefined for a seed/checkpoint with no successes; "
            "a checkpoint is plotted only when all requested seeds have finite values."
        ),
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
