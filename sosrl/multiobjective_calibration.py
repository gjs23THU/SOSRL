"""Shared milestone-budget calibration for multi-objective baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, median
from typing import Iterable, Sequence


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nondominated(
    points: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    return [
        point
        for point in unique
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other in unique
        )
    ]


def hypervolume_2d(
    points: Iterable[tuple[float, float]],
    reference: tuple[float, float] = (1.1, 1.1),
) -> float:
    front = sorted(nondominated(points))
    area = 0.0
    previous_y = float(reference[1])
    for x_value, y_value in front:
        if x_value >= reference[0] or y_value >= previous_y:
            continue
        area += (reference[0] - x_value) * (previous_y - y_value)
        previous_y = y_value
    return float(area)


def normalized_fronts(
    fronts: dict[int, list[tuple[float, float]]],
) -> dict[int, list[tuple[float, float]]]:
    all_points = [point for front in fronts.values() for point in front]
    if not all_points:
        return {budget: [] for budget in fronts}
    ideal = tuple(min(point[idx] for point in all_points) for idx in range(2))
    nadir = tuple(max(point[idx] for point in all_points) for idx in range(2))
    ranges = tuple(max(nadir[idx] - ideal[idx], 1e-12) for idx in range(2))
    return {
        budget: [
            (
                (point[0] - ideal[0]) / ranges[0],
                (point[1] - ideal[1]) / ranges[1],
            )
            for point in front
        ]
        for budget, front in fronts.items()
    }


def scenario_metrics(
    scenario_dir: Path,
    max_budget: int | None = None,
) -> list[dict[str, object]]:
    summary_rows = [
        row
        for row in read_csv(scenario_dir / "milestone_summary.csv")
        if max_budget is None
        or int(row["evaluation_budget_per_run"]) <= max_budget
    ]
    if not summary_rows:
        return []
    fronts: dict[int, list[tuple[float, float]]] = {}
    for row in summary_rows:
        budget = int(row["evaluation_budget_per_run"])
        front_path = (
            scenario_dir
            / "milestones"
            / f"eval_{budget:06d}"
            / "pareto_front.csv"
        )
        fronts[budget] = [
            (float(item["makespan"]), float(item["effective_cost"]))
            for item in read_csv(front_path)
        ]
    normalized = normalized_fronts(fronts)
    hypervolumes = {
        budget: hypervolume_2d(front) for budget, front in normalized.items()
    }
    final_budget = max(fronts)
    final_hv = hypervolumes[final_budget]
    final_row = next(
        row
        for row in summary_rows
        if int(row["evaluation_budget_per_run"]) == final_budget
    )
    final_j = (
        None
        if not final_row["gp_aligned_j"]
        else float(final_row["gp_aligned_j"])
    )
    output = []
    for row in summary_rows:
        budget = int(row["evaluation_budget_per_run"])
        gp_j = None if not row["gp_aligned_j"] else float(row["gp_aligned_j"])
        output.append(
            {
                "scenario_idx": int(row["scenario_idx"]),
                "scenario_hash": row["scenario_hash"],
                "split": row["split"],
                "category": row["category"],
                "evaluation_budget_per_run": budget,
                "independent_runs": int(row["independent_runs"]),
                "total_evaluations": int(row["total_evaluations"]),
                "success": row["success"].lower() == "true",
                "pareto_front_size": int(row["pareto_front_size"]),
                "normalized_hypervolume": hypervolumes[budget],
                "relative_hv_loss_to_max": (
                    None
                    if final_hv <= 1e-12
                    else max(0.0, 1.0 - hypervolumes[budget] / final_hv)
                ),
                "gp_aligned_j": gp_j,
                "relative_j_regret_to_max": (
                    None
                    if gp_j is None or final_j is None
                    else (gp_j - final_j) / max(abs(final_j), 1e-12)
                ),
                "gp_aligned_makespan": (
                    None
                    if not row["gp_aligned_makespan"]
                    else float(row["gp_aligned_makespan"])
                ),
                "gp_aligned_effective_cost": (
                    None
                    if not row["gp_aligned_effective_cost"]
                    else float(row["gp_aligned_effective_cost"])
                ),
            }
        )
    return output


def aggregate_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: list[tuple[str, str, list[dict[str, object]]]] = [
        ("all", "all", rows)
    ]
    for field in ("split", "category"):
        for value in sorted({str(row[field]) for row in rows}):
            groups.append(
                (
                    field,
                    value,
                    [row for row in rows if str(row[field]) == value],
                )
            )
    output = []
    for group_type, group_value, group_rows in groups:
        budgets = sorted(
            {int(row["evaluation_budget_per_run"]) for row in group_rows}
        )
        for budget in budgets:
            selected = [
                row
                for row in group_rows
                if int(row["evaluation_budget_per_run"]) == budget
            ]
            successes = [row for row in selected if bool(row["success"])]
            hv_values = [float(row["normalized_hypervolume"]) for row in selected]
            hv_losses = [
                float(row["relative_hv_loss_to_max"])
                for row in selected
                if row["relative_hv_loss_to_max"] is not None
            ]
            j_values = [
                float(row["gp_aligned_j"])
                for row in successes
                if row["gp_aligned_j"] is not None
            ]
            j_regrets = [
                float(row["relative_j_regret_to_max"])
                for row in successes
                if row["relative_j_regret_to_max"] is not None
            ]
            output.append(
                {
                    "group_type": group_type,
                    "group_value": group_value,
                    "evaluation_budget_per_run": budget,
                    "scenario_count": len(selected),
                    "success_rate": len(successes) / max(len(selected), 1),
                    "mean_normalized_hypervolume": fmean(hv_values),
                    "median_normalized_hypervolume": median(hv_values),
                    "median_relative_hv_loss_to_max": (
                        None if not hv_losses else median(hv_losses)
                    ),
                    "median_gp_aligned_j": (
                        None if not j_values else median(j_values)
                    ),
                    "median_relative_j_regret_to_max": (
                        None if not j_regrets else median(j_regrets)
                    ),
                    "mean_pareto_front_size": fmean(
                        float(row["pareto_front_size"]) for row in selected
                    ),
                }
            )
    return output


def discover_scenario_dirs(input_dirs: Sequence[Path]) -> list[Path]:
    scenario_dirs = []
    for input_dir in input_dirs:
        scenario_dirs.extend(
            path
            for path in input_dir.rglob("scenario_*")
            if path.is_dir()
            and (path / "run_manifest.json").is_file()
            and (path / "milestone_summary.csv").is_file()
        )
    return sorted(set(scenario_dirs))


def _group_row(
    curve: Sequence[dict[str, object]],
    group_type: str,
    group_value: str,
    budget: int,
) -> dict[str, object]:
    return next(
        row
        for row in curve
        if row["group_type"] == group_type
        and row["group_value"] == group_value
        and int(row["evaluation_budget_per_run"]) == budget
    )


def _runtime_status(input_dirs: Sequence[Path]) -> dict[str, object]:
    manifests = []
    statuses = []
    for input_dir in input_dirs:
        for path in input_dir.rglob("scenario_*/run_manifest.json"):
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        status_path = input_dir / "shard_status.json"
        if status_path.is_file():
            statuses.append(json.loads(status_path.read_text(encoding="utf-8")))
    return {
        "scenario_wall_seconds": sum(
            float(item.get("wall_seconds", 0.0)) for item in manifests
        ),
        "run_wall_seconds": sum(
            sum(float(value) for value in item.get("run_wall_seconds", []))
            for item in manifests
        ),
        "failed_shards": sum(
            1 for item in statuses if item.get("status") != "complete"
        ),
        "retried_shards": sum(
            1 for item in statuses if int(item.get("attempts", 1)) > 1
        ),
        "skipped_complete_shards": sum(
            1 for item in statuses if bool(item.get("skipped_complete"))
        ),
    }


def run_calibration(
    input_dirs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    algorithm: str,
    max_budget: int | None = None,
) -> dict[str, object]:
    roots = [Path(path) for path in input_dirs]
    scenario_dirs = discover_scenario_dirs(roots)
    rows = [
        row
        for scenario_dir in scenario_dirs
        for row in scenario_metrics(scenario_dir, max_budget=max_budget)
    ]
    if not rows:
        raise ValueError("no milestone scenario results found")
    curve = aggregate_rows(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "scenario_budget_metrics.csv", rows)
    write_csv(output_dir / "budget_curve.csv", curve)

    all_rows = [
        row
        for row in curve
        if row["group_type"] == "all" and row["group_value"] == "all"
    ]
    split_values = sorted(
        {
            str(row["group_value"])
            for row in curve
            if row["group_type"] == "split"
        }
    )
    required_groups = [("all", "all"), *[("split", v) for v in split_values]]
    final_budget = max(int(row["evaluation_budget_per_run"]) for row in all_rows)
    eligible = []
    stability: dict[str, list[int]] = {}
    for group_type, group_value in required_groups:
        group_key = f"{group_type}:{group_value}"
        final = _group_row(curve, group_type, group_value, final_budget)
        stability[group_key] = []
        for row in curve:
            if row["group_type"] != group_type or row["group_value"] != group_value:
                continue
            if (
                float(row["success_rate"]) >= float(final["success_rate"])
                and row["median_relative_hv_loss_to_max"] is not None
                and float(row["median_relative_hv_loss_to_max"]) <= 0.01
                and row["median_relative_j_regret_to_max"] is not None
                and float(row["median_relative_j_regret_to_max"]) <= 0.01
            ):
                stability[group_key].append(
                    int(row["evaluation_budget_per_run"])
                )
    for budget in sorted(int(row["evaluation_budget_per_run"]) for row in all_rows):
        if all(budget in stability[key] for key in stability):
            eligible.append(budget)
    recommendation = None if not eligible else eligible[0]
    runtime = _runtime_status(roots)
    summary = {
        "algorithm": algorithm,
        "input_dirs": [str(path.resolve()) for path in roots],
        "scenario_count": len({str(row["scenario_hash"]) for row in rows}),
        "max_budget_filter": max_budget,
        "milestones": sorted(
            {int(row["evaluation_budget_per_run"]) for row in rows}
        ),
        "recommended_minimum_evaluations": recommendation,
        "pilot_recommended_budget": recommendation,
        "selection_thresholds": {
            "required_groups": [
                {"group_type": group_type, "group_value": group_value}
                for group_type, group_value in required_groups
            ],
            "success_rate_not_below_max_budget": True,
            "median_relative_hv_loss_max": 0.01,
            "median_relative_j_regret_max": 0.01,
        },
        "eligible_budgets_by_group": stability,
        "runtime_and_retries": runtime,
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    recommendation_text = (
        "没有预算同时满足全部门槛"
        if recommendation is None
        else f"每次独立运行 {recommendation} 次评价"
    )
    report_budget = final_budget if recommendation is None else recommendation
    diagnostic_rows = sorted(
        (
            row
            for row in curve
            if int(row["evaluation_budget_per_run"]) == report_budget
        ),
        key=lambda row: (
            {"all": 0, "split": 1, "category": 2}.get(
                str(row["group_type"]), 9
            ),
            str(row["group_value"]),
        ),
    )
    table_lines = [
        "| 分组 | 样本数 | 成功率 | 中位 HV 损失 | 中位 J 后悔 | 1% 双指标 |",
        "| --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in diagnostic_rows:
        group_type = str(row["group_type"])
        group_value = str(row["group_value"])
        final = _group_row(curve, group_type, group_value, final_budget)
        hv_loss = row["median_relative_hv_loss_to_max"]
        j_regret = row["median_relative_j_regret_to_max"]
        passes = bool(
            float(row["success_rate"]) >= float(final["success_rate"])
            and hv_loss is not None
            and float(hv_loss) <= 0.01
            and j_regret is not None
            and float(j_regret) <= 0.01
        )
        table_lines.append(
            "| "
            f"{group_type}:{group_value} | {row['scenario_count']} | "
            f"{100.0 * float(row['success_rate']):.1f}% | "
            f"{100.0 * float(hv_loss or 0.0):.3f}% | "
            f"{100.0 * float(j_regret or 0.0):.3f}% | "
            f"{'是' if passes else '否'} |"
        )
    diagnostic_table = "\n".join(table_lines)
    report = (
        f"# {algorithm} 小预算校准结果\n\n"
        f"推荐最小预算：**{recommendation_text}**。该结论只用于小预算实验，"
        "不会修改正式 fast 配置。\n\n"
        f"- 场景数：{summary['scenario_count']}\n"
        f"- 里程碑：{', '.join(map(str, summary['milestones']))}\n"
        f"- 场景累计耗时：{runtime['scenario_wall_seconds']:.1f} 秒\n"
        f"- 独立运行累计耗时：{runtime['run_wall_seconds']:.1f} 秒\n"
        f"- 失败 shard：{runtime['failed_shards']}\n"
        f"- 发生重试的 shard：{runtime['retried_shards']}\n"
        f"- 断点跳过的完整 shard：{runtime['skipped_complete_shards']}\n\n"
        "门槛要求总体及 B/G split 的成功率不低于 500 评价档，且中位"
        "相对超体积损失与中位 GP 对齐 J 后悔值均不超过 1%。各类别与"
        "分组曲线详见 `budget_curve.csv`。类别行用于稳定性诊断，不作为"
        "推荐预算的硬门槛。\n\n"
        f"## {report_budget} 评价档分组稳定性\n\n"
        f"{diagnostic_table}\n"
    )
    (output_dir / "calibration_report.md").write_text(report, encoding="utf-8")
    return summary
