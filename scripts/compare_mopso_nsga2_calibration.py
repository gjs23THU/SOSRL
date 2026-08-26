"""Compare MOPSO and NSGA-II on their shared formal calibration scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


NSGA_ROOT = Path("runs/nsga2_budget_calibration_20260825/summary_final")
MOPSO_ROOT = Path("runs/mopso_budget_calibration_20260825")
OUTPUT = Path("runs/metaheuristic_budget_comparison_20260826")
COMMON_BUDGETS = [50, 100, 150, 200, 300, 400]
TOLERANCE = 1e-12

ALGORITHM_COLORS = {
    "NSGA-II": "#2563EB",
    "MOPSO-CD": "#E4572E",
    "tie": "#94A3B8",
}

CATEGORY_LABELS = {
    "capacity_tight": "容量紧张",
    "feasible_suboptimal": "可行但非最优",
    "missing_capability": "缺少能力",
    "redundant_overbudget": "冗余/超预算",
}


def hypervolume_2d(
    points: list[tuple[float, float]],
    reference: tuple[float, float] = (1.1, 1.1),
) -> float:
    """Match the shared calibration module's two-objective HV calculation."""
    unique = sorted(set(points))
    front = [
        point
        for point in unique
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in unique
        )
    ]
    area = 0.0
    previous_y = float(reference[1])
    for x_value, y_value in sorted(front):
        if x_value >= reference[0] or y_value >= previous_y:
            continue
        area += (reference[0] - x_value) * (previous_y - y_value)
        previous_y = y_value
    return float(area)


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    nsga = pd.read_csv(NSGA_ROOT / "scenario_budget_metrics.csv")
    mopso = pd.read_csv(MOPSO_ROOT / "scenario_budget_metrics.csv")
    nsga_summary = json.loads(
        (NSGA_ROOT / "calibration_summary.json").read_text(encoding="utf-8")
    )
    mopso_summary = json.loads(
        (MOPSO_ROOT / "calibration_summary.json").read_text(encoding="utf-8")
    )
    if set(nsga["scenario_hash"]) != set(mopso["scenario_hash"]):
        raise ValueError("NSGA-II and MOPSO scenario sets differ")
    return nsga, mopso, nsga_summary, mopso_summary


def actual_evaluation_count(summary: dict) -> int:
    total = 0
    for input_dir in summary["input_dirs"]:
        manifests = list(Path(input_dir).glob("scenario_*/run_manifest.json"))
        if len(manifests) != 1:
            raise ValueError(f"Expected one run manifest in {input_dir}, found {len(manifests)}")
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        total += sum(int(value) for value in manifest["run_evaluations"])
    return total


def scenario_directories(summary: dict) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for input_dir in summary["input_dirs"]:
        for manifest_path in Path(input_dir).glob("scenario_*/run_manifest.json"):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            output[str(manifest["scenario_hash"])] = manifest_path.parent
    return output


def read_front(scenario_dir: Path, budget: int) -> list[tuple[float, float]]:
    path = (
        scenario_dir
        / "milestones"
        / f"eval_{budget:06d}"
        / "pareto_front.csv"
    )
    front = pd.read_csv(path)
    return list(
        zip(
            front["makespan"].astype(float),
            front["effective_cost"].astype(float),
        )
    )


def jointly_normalized_hypervolumes(
    nsga_summary: dict,
    mopso_summary: dict,
) -> pd.DataFrame:
    nsga_dirs = scenario_directories(nsga_summary)
    mopso_dirs = scenario_directories(mopso_summary)
    if set(nsga_dirs) != set(mopso_dirs):
        raise ValueError("NSGA-II and MOPSO scenario directories differ")
    rows: list[dict[str, object]] = []
    for scenario_hash in sorted(nsga_dirs):
        fronts: dict[tuple[str, int], list[tuple[float, float]]] = {}
        for algorithm, directories in (
            ("nsga", nsga_dirs),
            ("mopso", mopso_dirs),
        ):
            for budget in COMMON_BUDGETS:
                fronts[(algorithm, budget)] = read_front(
                    directories[scenario_hash], budget
                )
        all_points = [point for front in fronts.values() for point in front]
        ideal = tuple(min(point[index] for point in all_points) for index in range(2))
        nadir = tuple(max(point[index] for point in all_points) for index in range(2))
        ranges = tuple(max(nadir[index] - ideal[index], 1e-12) for index in range(2))
        for budget in COMMON_BUDGETS:
            row: dict[str, object] = {
                "scenario_hash": scenario_hash,
                "evaluation_budget_per_run": budget,
            }
            for algorithm in ("nsga", "mopso"):
                normalized = [
                    (
                        (point[0] - ideal[0]) / ranges[0],
                        (point[1] - ideal[1]) / ranges[1],
                    )
                    for point in fronts[(algorithm, budget)]
                ]
                row[f"normalized_hypervolume_{algorithm}"] = hypervolume_2d(
                    normalized
                )
            rows.append(row)
    return pd.DataFrame(rows)


def common_terminal_recommendation(
    metrics: pd.DataFrame,
    summary: dict,
) -> dict[str, object]:
    """Reapply the 1% gates with a shared 400-evaluation terminal budget."""
    directories = scenario_directories(summary)
    scenario_rows: list[dict[str, object]] = []
    for scenario_hash, scenario_dir in directories.items():
        fronts = {
            budget: read_front(scenario_dir, budget) for budget in COMMON_BUDGETS
        }
        all_points = [point for front in fronts.values() for point in front]
        ideal = tuple(min(point[index] for point in all_points) for index in range(2))
        nadir = tuple(max(point[index] for point in all_points) for index in range(2))
        ranges = tuple(max(nadir[index] - ideal[index], 1e-12) for index in range(2))
        hypervolumes = {}
        for budget, front in fronts.items():
            normalized = [
                (
                    (point[0] - ideal[0]) / ranges[0],
                    (point[1] - ideal[1]) / ranges[1],
                )
                for point in front
            ]
            hypervolumes[budget] = hypervolume_2d(normalized)
        final_hv = hypervolumes[400]
        source = metrics[metrics["scenario_hash"] == scenario_hash].set_index(
            "evaluation_budget_per_run"
        )
        final_j = float(source.loc[400, "gp_aligned_j"])
        for budget in COMMON_BUDGETS:
            row = source.loc[budget]
            j_value = float(row["gp_aligned_j"])
            scenario_rows.append(
                {
                    "scenario_hash": scenario_hash,
                    "split": str(row["split"]),
                    "budget": budget,
                    "success": bool(row["success"]),
                    "hv_loss": max(
                        0.0,
                        1.0 - hypervolumes[budget] / max(final_hv, 1e-12),
                    ),
                    "j_regret": (j_value - final_j) / max(abs(final_j), 1e-12),
                }
            )
    frame = pd.DataFrame(scenario_rows)
    groups = {
        "all": frame,
        "b_validation": frame[frame["split"] == "b_validation"],
        "g_validation": frame[frame["split"] == "g_validation"],
    }
    eligible_by_group: dict[str, list[int]] = {}
    for name, group in groups.items():
        eligible_by_group[name] = []
        for budget in COMMON_BUDGETS:
            rows = group[group["budget"] == budget]
            if (
                float(rows["success"].mean()) >= 1.0
                and float(rows["hv_loss"].median()) <= 0.01
                and float(rows["j_regret"].median()) <= 0.01
            ):
                eligible_by_group[name].append(budget)
    recommendation = next(
        (
            budget
            for budget in COMMON_BUDGETS
            if all(budget in values for values in eligible_by_group.values())
        ),
        None,
    )
    return {
        "terminal_budget": 400,
        "recommended_minimum_evaluations": recommendation,
        "eligible_budgets_by_group": eligible_by_group,
    }


def merge_common(
    nsga: pd.DataFrame,
    mopso: pd.DataFrame,
    joint_hv: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "scenario_idx",
        "scenario_hash",
        "split",
        "category",
        "evaluation_budget_per_run",
    ]
    columns = keys + [
        "success",
        "pareto_front_size",
        "gp_aligned_j",
        "gp_aligned_makespan",
        "gp_aligned_effective_cost",
    ]
    merged = nsga[nsga["evaluation_budget_per_run"].isin(COMMON_BUDGETS)][
        columns
    ].merge(
        mopso[mopso["evaluation_budget_per_run"].isin(COMMON_BUDGETS)][columns],
        on=keys,
        suffixes=("_nsga", "_mopso"),
        validate="one_to_one",
    )
    expected_rows = len(COMMON_BUDGETS) * 8
    if len(merged) != expected_rows:
        raise ValueError(f"Expected {expected_rows} shared rows, found {len(merged)}")
    merged = merged.merge(
        joint_hv,
        on=["scenario_hash", "evaluation_budget_per_run"],
        validate="many_to_one",
    )
    if len(merged) != expected_rows:
        raise ValueError("Joint-hypervolume merge changed the shared row count")
    return merged


def winner_counts(values: pd.Series, higher_is_better: bool) -> dict[str, int]:
    if higher_is_better:
        nsga_wins = int((values < -TOLERANCE).sum())
        mopso_wins = int((values > TOLERANCE).sum())
    else:
        nsga_wins = int((values > TOLERANCE).sum())
        mopso_wins = int((values < -TOLERANCE).sum())
    return {
        "NSGA-II": nsga_wins,
        "MOPSO-CD": mopso_wins,
        "tie": int(len(values) - nsga_wins - mopso_wins),
    }


def build_budget_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for budget, frame in merged.groupby("evaluation_budget_per_run", sort=True):
        hv_delta = (
            frame["normalized_hypervolume_mopso"]
            - frame["normalized_hypervolume_nsga"]
        )
        j_delta = frame["gp_aligned_j_mopso"] - frame["gp_aligned_j_nsga"]
        hv_wins = winner_counts(hv_delta, higher_is_better=True)
        j_wins = winner_counts(j_delta, higher_is_better=False)
        for algorithm, suffix in (("NSGA-II", "nsga"), ("MOPSO-CD", "mopso")):
            rows.append(
                {
                    "evaluation_budget_per_run": int(budget),
                    "algorithm": algorithm,
                    "scenario_count": int(len(frame)),
                    "success_rate": float(frame[f"success_{suffix}"].mean()),
                    "mean_normalized_hypervolume": float(
                        frame[f"normalized_hypervolume_{suffix}"].mean()
                    ),
                    "median_gp_aligned_j": float(
                        frame[f"gp_aligned_j_{suffix}"].median()
                    ),
                    "mean_pareto_front_size": float(
                        frame[f"pareto_front_size_{suffix}"].mean()
                    ),
                    "hv_scenario_wins": hv_wins[algorithm],
                    "j_scenario_wins": j_wins[algorithm],
                    "hv_ties": hv_wins["tie"],
                    "j_ties": j_wins["tie"],
                }
            )
    return pd.DataFrame(rows)


def build_scenario_comparison(merged: pd.DataFrame) -> pd.DataFrame:
    frame = merged.copy()
    frame["scenario_label"] = frame.apply(
        lambda row: (
            f"{str(row['split'])[0].upper()}{int(row['scenario_idx'])} · "
            f"{CATEGORY_LABELS[str(row['category'])]}"
        ),
        axis=1,
    )
    frame["hv_delta_mopso_minus_nsga"] = (
        frame["normalized_hypervolume_mopso"]
        - frame["normalized_hypervolume_nsga"]
    )
    frame["hv_relative_delta_vs_nsga"] = (
        frame["hv_delta_mopso_minus_nsga"]
        / frame["normalized_hypervolume_nsga"].abs()
    )
    frame["j_delta_mopso_minus_nsga"] = (
        frame["gp_aligned_j_mopso"] - frame["gp_aligned_j_nsga"]
    )
    return frame


def add_axis_style(axis: plt.Axes) -> None:
    axis.set_facecolor("#FBFCFE")
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#AAB3C2")
    axis.tick_params(colors="#475569", labelsize=9)


def plot_report(
    budget_summary: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    nsga_summary: dict,
    mopso_summary: dict,
    nsga_common_terminal: dict,
    mopso_common_terminal: dict,
) -> None:
    configure_font()
    figure, axes = plt.subplots(2, 2, figsize=(16, 10.5), dpi=160)
    figure.patch.set_facecolor("#F4F7FB")
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.09,
        top=0.79,
        wspace=0.23,
        hspace=0.38,
    )

    figure.suptitle(
        "MOPSO 与 NSGA-II 小预算结果对比",
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
        "共同口径：相同 8 个正式场景、3 个种子、50–400 评价档；HV 越高越好，J 越低越好",
        ha="left",
        fontsize=11.5,
        color="#475569",
    )
    figure.text(
        0.075,
        0.855,
        (
            "共同 400 终点重校准：NSGA-II "
            f"{nsga_common_terminal['recommended_minimum_evaluations']}   |   "
            "MOPSO "
            f"{mopso_common_terminal['recommended_minimum_evaluations']}"
        ),
        ha="left",
        fontsize=15,
        fontweight="bold",
        color="#1E3A8A",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#EFF6FF",
            "edgecolor": "#93C5FD",
            "linewidth": 1.2,
        },
    )

    axis = axes[0, 0]
    for algorithm in ("NSGA-II", "MOPSO-CD"):
        subset = budget_summary[budget_summary["algorithm"] == algorithm]
        axis.plot(
            subset["evaluation_budget_per_run"],
            subset["mean_normalized_hypervolume"],
            color=ALGORITHM_COLORS[algorithm],
            linewidth=2.6,
            marker="o" if algorithm == "NSGA-II" else "s",
            markersize=5.5,
            label=algorithm,
        )
    axis.set_title(
        "A. 平均归一化超体积（共同尺度）",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xlabel("每次独立运行的评价预算", color="#475569")
    axis.set_ylabel("平均 normalized HV", color="#475569")
    axis.set_xticks(COMMON_BUDGETS)
    axis.legend(frameon=False, loc="lower right")
    add_axis_style(axis)

    axis = axes[0, 1]
    for algorithm in ("NSGA-II", "MOPSO-CD"):
        subset = budget_summary[budget_summary["algorithm"] == algorithm]
        axis.plot(
            subset["evaluation_budget_per_run"],
            subset["median_gp_aligned_j"],
            color=ALGORITHM_COLORS[algorithm],
            linewidth=2.6,
            marker="o" if algorithm == "NSGA-II" else "s",
            markersize=5.5,
            label=algorithm,
        )
    axis.set_title("B. 中位 GP 对齐 J", loc="left", fontsize=13, fontweight="bold")
    axis.set_xlabel("每次独立运行的评价预算", color="#475569")
    axis.set_ylabel("中位 J（越低越好）", color="#475569")
    axis.set_xticks(COMMON_BUDGETS)
    axis.legend(frameon=False, loc="upper right")
    add_axis_style(axis)

    axis = axes[1, 0]
    metrics = ["HV @300", "J @300", "HV @400", "J @400"]
    nsga_wins: list[int] = []
    mopso_wins: list[int] = []
    ties: list[int] = []
    for budget in (300, 400):
        rows = budget_summary[budget_summary["evaluation_budget_per_run"] == budget]
        nsga_row = rows[rows["algorithm"] == "NSGA-II"].iloc[0]
        mopso_row = rows[rows["algorithm"] == "MOPSO-CD"].iloc[0]
        nsga_wins.extend(
            [int(nsga_row["hv_scenario_wins"]), int(nsga_row["j_scenario_wins"])]
        )
        mopso_wins.extend(
            [int(mopso_row["hv_scenario_wins"]), int(mopso_row["j_scenario_wins"])]
        )
        ties.extend([int(nsga_row["hv_ties"]), int(nsga_row["j_ties"])])
    positions = np.arange(len(metrics))
    axis.barh(positions, nsga_wins, color=ALGORITHM_COLORS["NSGA-II"], label="NSGA-II 胜")
    axis.barh(
        positions,
        mopso_wins,
        left=nsga_wins,
        color=ALGORITHM_COLORS["MOPSO-CD"],
        label="MOPSO 胜",
    )
    axis.barh(
        positions,
        ties,
        left=np.array(nsga_wins) + np.array(mopso_wins),
        color=ALGORITHM_COLORS["tie"],
        label="持平",
    )
    for position, values in enumerate(zip(nsga_wins, mopso_wins, ties)):
        offset = 0
        for value in values:
            if value:
                axis.text(
                    offset + value / 2,
                    position,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > 1 else "#172033",
                    fontsize=9,
                    fontweight="bold",
                )
            offset += value
    axis.set_yticks(positions, metrics)
    axis.invert_yaxis()
    axis.set_xlim(0, 8)
    axis.set_xlabel("8 个场景中的胜出数量", color="#475569")
    axis.set_title("C. 同预算逐场景胜负", loc="left", fontsize=13, fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="lower right", fontsize=9)
    add_axis_style(axis)

    axis = axes[1, 1]
    at_400 = scenario_comparison[
        scenario_comparison["evaluation_budget_per_run"] == 400
    ].sort_values("hv_relative_delta_vs_nsga")
    values = at_400["hv_relative_delta_vs_nsga"].to_numpy() * 100
    colors = np.where(values >= 0, ALGORITHM_COLORS["MOPSO-CD"], ALGORITHM_COLORS["NSGA-II"])
    axis.barh(at_400["scenario_label"], values, color=colors)
    axis.axvline(0, color="#475569", linewidth=1)
    for position, value in enumerate(values):
        axis.text(
            value + (0.12 if value >= 0 else -0.12),
            position,
            f"{value:+.2f}%",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=8.5,
            color="#334155",
        )
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    axis.set_xlabel("MOPSO 相对 NSGA-II 的 HV 差异", color="#475569")
    axis.set_title("D. 400 评价档逐场景 HV 差异", loc="left", fontsize=13, fontweight="bold")
    add_axis_style(axis)
    axis.grid(axis="x", color="#D8DEE9", linewidth=0.8, alpha=0.75)
    axis.grid(axis="y", visible=False)

    nsga_runtime = float(nsga_summary["runtime_and_retries"]["run_wall_seconds"])
    mopso_runtime = float(mopso_summary["runtime_and_retries"]["run_wall_seconds"])
    nsga_evaluations = actual_evaluation_count(nsga_summary)
    mopso_evaluations = actual_evaluation_count(mopso_summary)
    figure.text(
        0.075,
        0.027,
        (
            "运行观测：两者成功率均为 100%；NSGA-II 约 "
            f"{nsga_runtime / nsga_evaluations:.3f} 秒/评价，MOPSO 约 "
            f"{mopso_runtime / mopso_evaluations:.3f} 秒/评价。"
            "该耗时未做同机同步基准，仅用于本轮实验参考。"
        ),
        fontsize=9.3,
        color="#64748B",
    )

    figure.savefig(OUTPUT / "mopso_vs_nsga2_overview.png", facecolor=figure.get_facecolor())
    figure.savefig(OUTPUT / "mopso_vs_nsga2_overview.svg", facecolor=figure.get_facecolor())
    plt.close(figure)


def write_summary(
    budget_summary: pd.DataFrame,
    scenario_comparison: pd.DataFrame,
    nsga_summary: dict,
    mopso_summary: dict,
    nsga_common_terminal: dict,
    mopso_common_terminal: dict,
) -> None:
    def algorithm_row(budget: int, algorithm: str) -> pd.Series:
        return budget_summary[
            (budget_summary["evaluation_budget_per_run"] == budget)
            & (budget_summary["algorithm"] == algorithm)
        ].iloc[0]

    nsga_300 = algorithm_row(300, "NSGA-II")
    mopso_300 = algorithm_row(300, "MOPSO-CD")
    nsga_400 = algorithm_row(400, "NSGA-II")
    mopso_400 = algorithm_row(400, "MOPSO-CD")
    at_400 = scenario_comparison[
        scenario_comparison["evaluation_budget_per_run"] == 400
    ]
    nsga_runtime = float(nsga_summary["runtime_and_retries"]["run_wall_seconds"])
    mopso_runtime = float(mopso_summary["runtime_and_retries"]["run_wall_seconds"])
    nsga_evaluations = actual_evaluation_count(nsga_summary)
    mopso_evaluations = actual_evaluation_count(mopso_summary)
    payload = {
        "comparison_scope": {
            "scenario_count": 8,
            "independent_runs_per_scenario": 3,
            "common_budgets": COMMON_BUDGETS,
            "success_rate_nsga2": float(nsga_400["success_rate"]),
            "success_rate_mopso": float(mopso_400["success_rate"]),
            "hypervolume_normalization": (
                "Per scenario, jointly normalized over both algorithms and all "
                "shared 50-400 evaluation fronts; reference point=(1.1,1.1)."
            ),
        },
        "calibration_recommendation": {
            "original_nsga2": {
                "terminal_budget": int(nsga_summary["max_budget_filter"]),
                "recommended_minimum_evaluations": int(
                    nsga_summary["recommended_minimum_evaluations"]
                ),
            },
            "original_mopso": {
                "terminal_budget": int(mopso_summary["max_budget_filter"]),
                "recommended_minimum_evaluations": int(
                    mopso_summary["recommended_minimum_evaluations"]
                ),
            },
            "common_terminal_400": {
                "nsga2": nsga_common_terminal,
                "mopso": mopso_common_terminal,
            },
        },
        "same_budget_300": {
            "mean_hv_nsga2": float(nsga_300["mean_normalized_hypervolume"]),
            "mean_hv_mopso": float(mopso_300["mean_normalized_hypervolume"]),
            "median_j_nsga2": float(nsga_300["median_gp_aligned_j"]),
            "median_j_mopso": float(mopso_300["median_gp_aligned_j"]),
            "hv_scenario_wins_nsga2": int(nsga_300["hv_scenario_wins"]),
            "hv_scenario_wins_mopso": int(mopso_300["hv_scenario_wins"]),
            "j_scenario_wins_nsga2": int(nsga_300["j_scenario_wins"]),
            "j_scenario_wins_mopso": int(mopso_300["j_scenario_wins"]),
            "j_ties": int(nsga_300["j_ties"]),
        },
        "same_budget_400": {
            "mean_hv_nsga2": float(nsga_400["mean_normalized_hypervolume"]),
            "mean_hv_mopso": float(mopso_400["mean_normalized_hypervolume"]),
            "mean_hv_relative_delta_mopso_vs_nsga2": float(
                at_400["normalized_hypervolume_mopso"].mean()
                / at_400["normalized_hypervolume_nsga"].mean()
                - 1
            ),
            "median_j_nsga2": float(nsga_400["median_gp_aligned_j"]),
            "median_j_mopso": float(mopso_400["median_gp_aligned_j"]),
            "hv_scenario_wins_nsga2": int(nsga_400["hv_scenario_wins"]),
            "hv_scenario_wins_mopso": int(mopso_400["hv_scenario_wins"]),
            "j_scenario_wins_nsga2": int(nsga_400["j_scenario_wins"]),
            "j_scenario_wins_mopso": int(mopso_400["j_scenario_wins"]),
            "j_ties": int(nsga_400["j_ties"]),
        },
        "at_original_recommended_budget": {
            "nsga2_budget": int(nsga_summary["recommended_minimum_evaluations"]),
            "mopso_budget": int(mopso_summary["recommended_minimum_evaluations"]),
            "mean_hv_nsga2": float(nsga_300["mean_normalized_hypervolume"]),
            "mean_hv_mopso": float(mopso_400["mean_normalized_hypervolume"]),
            "mean_hv_relative_delta_mopso_vs_nsga2": float(
                mopso_400["mean_normalized_hypervolume"]
                / nsga_300["mean_normalized_hypervolume"]
                - 1
            ),
            "median_j_nsga2": float(nsga_300["median_gp_aligned_j"]),
            "median_j_mopso": float(mopso_400["median_gp_aligned_j"]),
            "median_j_relative_delta_mopso_vs_nsga2": float(
                mopso_400["median_gp_aligned_j"]
                / nsga_300["median_gp_aligned_j"]
                - 1
            ),
            "budget_increase_mopso_vs_nsga2": float(
                mopso_summary["recommended_minimum_evaluations"]
                / nsga_summary["recommended_minimum_evaluations"]
                - 1
            ),
        },
        "observed_runtime": {
            "nsga2_actual_evaluations": nsga_evaluations,
            "mopso_actual_evaluations": mopso_evaluations,
            "nsga2_seconds_per_evaluation": nsga_runtime / nsga_evaluations,
            "mopso_seconds_per_evaluation": mopso_runtime / mopso_evaluations,
            "mopso_relative_seconds_per_evaluation_vs_nsga2": (
                (mopso_runtime / mopso_evaluations)
                / (nsga_runtime / nsga_evaluations)
                - 1
            ),
            "caveat": "Historical observed runtime, not a synchronized machine benchmark.",
        },
        "metric_direction": {
            "normalized_hypervolume": "higher_is_better",
            "gp_aligned_j": "lower_is_better",
        },
    }
    (OUTPUT / "comparison_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    nsga, mopso, nsga_summary, mopso_summary = load_inputs()
    joint_hv = jointly_normalized_hypervolumes(nsga_summary, mopso_summary)
    merged = merge_common(nsga, mopso, joint_hv)
    budget_summary = build_budget_summary(merged)
    scenario_comparison = build_scenario_comparison(merged)
    nsga_common_terminal = common_terminal_recommendation(nsga, nsga_summary)
    mopso_common_terminal = common_terminal_recommendation(mopso, mopso_summary)
    budget_summary.to_csv(OUTPUT / "budget_comparison.csv", index=False)
    scenario_comparison.to_csv(OUTPUT / "scenario_comparison.csv", index=False)
    write_summary(
        budget_summary,
        scenario_comparison,
        nsga_summary,
        mopso_summary,
        nsga_common_terminal,
        mopso_common_terminal,
    )
    plot_report(
        budget_summary,
        scenario_comparison,
        nsga_summary,
        mopso_summary,
        nsga_common_terminal,
        mopso_common_terminal,
    )
    print((OUTPUT / "comparison_summary.json").resolve())
    print((OUTPUT / "mopso_vs_nsga2_overview.png").resolve())


if __name__ == "__main__":
    main()
