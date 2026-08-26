"""Build the canonical artifact payload for the MOPSO calibration report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path("runs/mopso_budget_calibration_20260825")
OUTPUT = ROOT / "visual_report"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_label(group_type: str, group_value: str) -> str:
    if group_type == "all":
        return "总体"
    if group_type == "split":
        return "B split" if group_value.startswith("b_") else "G split"
    labels = {
        "capacity_tight": "容量紧张",
        "feasible_suboptimal": "可行但非最优",
        "missing_capability": "缺少能力",
        "redundant_overbudget": "冗余/超预算",
    }
    return labels.get(group_value, group_value)


def main() -> None:
    curve = read_csv(ROOT / "budget_curve.csv")
    summary = read_json(ROOT / "calibration_summary.json")
    generated_at = datetime.now(timezone.utc).isoformat()

    headline_rows = [
        {
            "recommended_budget": int(
                summary["recommended_minimum_evaluations"]
            ),
            "success_rate": 1.0,
            "independent_runs": 24,
            "failed_shards": int(
                summary["runtime_and_retries"]["failed_shards"]
            ),
            "total_evaluations": 12_000,
            "wall_minutes": float(
                summary["runtime_and_retries"]["scenario_wall_seconds"]
            )
            / 60.0,
        }
    ]

    quality_rows = []
    regret_rows = []
    front_size_rows = []
    for row in curve:
        if row["group_type"] not in {"all", "split"}:
            continue
        label = group_label(row["group_type"], row["group_value"])
        budget = int(row["evaluation_budget_per_run"])
        quality_rows.append(
            {
                "budget": budget,
                "series": label,
                "loss": float(row["median_relative_hv_loss_to_max"]),
                "line_style": "solid",
                "scenario_count": int(row["scenario_count"]),
            }
        )
        regret_rows.append(
            {
                "budget": budget,
                "series": label,
                "regret": float(row["median_relative_j_regret_to_max"]),
                "line_style": "solid",
                "scenario_count": int(row["scenario_count"]),
            }
        )
        front_size_rows.append(
            {
                "budget": budget,
                "series": label,
                "front_size": float(row["mean_pareto_front_size"]),
                "line_style": "solid",
                "scenario_count": int(row["scenario_count"]),
            }
        )
    for budget in (50, 100, 150, 200, 300, 400, 500):
        quality_rows.append(
            {
                "budget": budget,
                "series": "1% 门槛",
                "loss": 0.01,
                "line_style": "dashed",
                "scenario_count": 8,
            }
        )
        regret_rows.append(
            {
                "budget": budget,
                "series": "1% 门槛",
                "regret": 0.01,
                "line_style": "dashed",
                "scenario_count": 8,
            }
        )

    category_rows = []
    table_rows = []
    for row in curve:
        if int(row["evaluation_budget_per_run"]) != 400:
            continue
        label = group_label(row["group_type"], row["group_value"])
        hv_loss = float(row["median_relative_hv_loss_to_max"])
        j_regret = float(row["median_relative_j_regret_to_max"])
        if row["group_type"] == "category":
            category_rows.extend(
                [
                    {
                        "category": label,
                        "metric": "HV 损失",
                        "value": hv_loss,
                        "scenario_count": int(row["scenario_count"]),
                    },
                    {
                        "category": label,
                        "metric": "J 后悔",
                        "value": j_regret,
                        "scenario_count": int(row["scenario_count"]),
                    },
                ]
            )
        final_row = next(
            item
            for item in curve
            if item["group_type"] == row["group_type"]
            and item["group_value"] == row["group_value"]
            and int(item["evaluation_budget_per_run"]) == 500
        )
        passes = bool(
            float(row["success_rate"]) >= float(final_row["success_rate"])
            and hv_loss <= 0.01
            and j_regret <= 0.01
        )
        order = {"all": 0, "split": 1, "category": 2}[row["group_type"]]
        table_rows.append(
            {
                "group": label,
                "group_type": row["group_type"],
                "scenario_count": int(row["scenario_count"]),
                "success_rate": float(row["success_rate"]),
                "hv_loss": hv_loss,
                "j_regret": j_regret,
                "passes": "是" if passes else "否",
                "sort_order": order * 10 + len(table_rows),
            }
        )

    sources = [
        {
            "id": "budget_curve",
            "label": "MOPSO 预算分组曲线",
            "path": "runs/mopso_budget_calibration_20260825/budget_curve.csv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "读取算法中立汇总器生成的 MOPSO 分组预算曲线。",
                "sql": (
                    "SELECT * FROM read_csv_auto("
                    "'runs/mopso_budget_calibration_20260825/budget_curve.csv', "
                    "header = true);"
                ),
                "tables_used": [
                    "runs/mopso_budget_calibration_20260825/budget_curve.csv"
                ],
                "filters": ["evaluation_budget_per_run in (50,100,150,200,300,400,500)"],
                "metric_definitions": [
                    "success_rate = successful scenarios / scenarios in group",
                    "relative_hv_loss_to_max = max(0, 1 - HV_budget / HV_500)",
                    "relative_j_regret_to_max = (J_budget - J_500) / abs(J_500)",
                ],
            },
        },
        {
            "id": "scenario_metrics",
            "label": "MOPSO 场景—预算指标",
            "path": (
                "runs/mopso_budget_calibration_20260825/"
                "scenario_budget_metrics.csv"
            ),
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "读取 8 个正式场景在七个评价预算下的指标。",
                "sql": (
                    "SELECT * FROM read_csv_auto("
                    "'runs/mopso_budget_calibration_20260825/"
                    "scenario_budget_metrics.csv', header = true);"
                ),
                "tables_used": [
                    "runs/mopso_budget_calibration_20260825/scenario_budget_metrics.csv"
                ],
                "filters": ["8 formal scenarios", "3 independent seeds per scenario"],
                "metric_definitions": [
                    "independent_runs = 3 per scenario and evaluation milestone"
                ],
            },
        },
        {
            "id": "calibration_summary",
            "label": "MOPSO 校准汇总",
            "path": (
                "runs/mopso_budget_calibration_20260825/"
                "calibration_summary.json"
            ),
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "读取 MOPSO 校准推荐值、耗时和失败重试统计。",
                "sql": (
                    "SELECT * FROM read_json_auto("
                    "'runs/mopso_budget_calibration_20260825/"
                    "calibration_summary.json');"
                ),
                "tables_used": [
                    "runs/mopso_budget_calibration_20260825/calibration_summary.json"
                ],
                "filters": ["max_budget_filter = 500"],
                "metric_definitions": [
                    "recommended_minimum_evaluations = smallest budget passing all and B/G split gates"
                ],
            },
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "MOPSO 小预算校准图表报告",
        "description": "8 个正式场景、24 次独立运行的小预算评估。",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "recommended_budget",
                "description": "满足总体与 B/G split 全部门槛的最小预算。",
                "dataset": "headline",
                "sourceId": "calibration_summary",
                "metrics": [
                    {
                        "label": "推荐评价预算",
                        "field": "recommended_budget",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "success_rate",
                "description": "400 评价档的总体成功率。",
                "dataset": "headline",
                "sourceId": "budget_curve",
                "metrics": [
                    {
                        "label": "总体成功率",
                        "field": "success_rate",
                        "format": "percent",
                    }
                ],
            },
            {
                "id": "runs",
                "description": "8 个场景，每场景 3 个独立种子。",
                "dataset": "headline",
                "sourceId": "scenario_metrics",
                "metrics": [
                    {
                        "label": "独立运行次数",
                        "field": "independent_runs",
                        "format": "number",
                    }
                ],
            },
            {
                "id": "failures",
                "description": "校准 shard 失败数；全部首尝试完成。",
                "dataset": "headline",
                "sourceId": "calibration_summary",
                "metrics": [
                    {
                        "label": "失败 shard",
                        "field": "failed_shards",
                        "format": "number",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "hv_loss_curve",
                "title": "相对 500 评价档的超体积损失",
                "subtitle": "总体与 B/G split；虚线为 1% 门槛。",
                "type": "line",
                "dataset": "quality_curve",
                "sourceId": "budget_curve",
                "encodings": {
                    "x": {
                        "field": "budget",
                        "type": "ordinal",
                        "label": "每次独立运行评价数",
                    },
                    "y": {
                        "field": "loss",
                        "type": "quantitative",
                        "label": "中位相对 HV 损失",
                        "format": "percent",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "分组",
                    },
                },
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "j_regret_curve",
                "title": "相对 500 评价档的 GP 对齐 J 后悔值",
                "subtitle": "300 评价时 B split 仍为 1.613%，400 时降至 0%。",
                "type": "line",
                "dataset": "regret_curve",
                "sourceId": "budget_curve",
                "encodings": {
                    "x": {
                        "field": "budget",
                        "type": "ordinal",
                        "label": "每次独立运行评价数",
                    },
                    "y": {
                        "field": "regret",
                        "type": "quantitative",
                        "label": "中位相对 J 后悔值",
                        "format": "percent",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "分组",
                    },
                },
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "category_stability",
                "title": "400 评价档的场景类别稳定性",
                "subtitle": "每类 2 个场景；类别结果用于诊断，不作为硬门槛。",
                "type": "bar",
                "dataset": "category_stability",
                "sourceId": "budget_curve",
                "encodings": {
                    "x": {
                        "field": "category",
                        "type": "nominal",
                        "label": "场景类别",
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "label": "中位相对损失/后悔值",
                        "format": "percent",
                    },
                    "color": {
                        "field": "metric",
                        "type": "nominal",
                        "label": "指标",
                    },
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
                "valueFormat": "percent",
                "layout": "full",
            },
            {
                "id": "front_size_curve",
                "title": "平均 Pareto 前沿规模",
                "subtitle": "400 后前沿规模仍波动，说明更多评价主要改善质量而非单调增加数量。",
                "type": "line",
                "dataset": "front_size_curve",
                "sourceId": "budget_curve",
                "encodings": {
                    "x": {
                        "field": "budget",
                        "type": "ordinal",
                        "label": "每次独立运行评价数",
                    },
                    "y": {
                        "field": "front_size",
                        "type": "quantitative",
                        "label": "平均前沿规模",
                        "format": "number",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "分组",
                    },
                },
                "valueFormat": "number",
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "budget_400_detail",
                "title": "400 评价档分组明细",
                "subtitle": "成功率、HV 损失、J 后悔值及 1% 双指标判断。",
                "dataset": "budget_400_detail",
                "sourceId": "budget_curve",
                "defaultSort": {"field": "sort_order", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "group", "label": "分组", "type": "text"},
                    {
                        "field": "scenario_count",
                        "label": "场景数",
                        "format": "number",
                    },
                    {
                        "field": "success_rate",
                        "label": "成功率",
                        "format": "percent",
                    },
                    {
                        "field": "hv_loss",
                        "label": "中位 HV 损失",
                        "format": "percent",
                    },
                    {
                        "field": "j_regret",
                        "label": "中位 J 后悔",
                        "format": "percent",
                    },
                    {"field": "passes", "label": "1% 双指标", "type": "text"},
                    {"field": "sort_order", "label": "顺序", "format": "number"},
                ],
            }
        ],
        "sources": sources,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# MOPSO 小预算校准图表报告",
            },
            {
                "id": "executive_summary",
                "type": "markdown",
                "body": (
                    "## Executive Summary\n\n"
                    "- **推荐每次独立运行使用 400 次评价。** 这是总体、B split 和 G split "
                    "同时满足成功率不低于 500 档、HV 损失不超过 1%、J 后悔值不超过 1% 的最小预算。\n"
                    "- **300 次评价仍偏紧。** 总体与 G split 已达标，但 B split 的中位 J 后悔值为 1.613%，因此不能作为统一推荐。\n"
                    "- **执行稳定。** 8 个场景、24 次独立运行全部成功，12,000 次评价无失败或重试；校准总耗时约 68.3 分钟。\n"
                    "- **类别结果存在小样本波动。** 400 评价时容量紧张与冗余/超预算类别的 J 后悔值略高于 1%，但每类仅 2 个场景，作为风险提示而非硬门槛。"
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "recommended_budget",
                    "success_rate",
                    "runs",
                    "failures",
                ],
            },
            {
                "id": "quality_heading",
                "type": "markdown",
                "body": (
                    "## 质量损失在 400 评价前已基本收敛\n\n"
                    "超体积衡量 Pareto 前沿整体覆盖质量。100 评价后总体中位损失已低于 1%，但 400 评价才同时为 B/G split 留出稳定余量。**这意味着继续增加到 500 的边际质量收益很小。**"
                ),
            },
            {"id": "quality_chart", "type": "chart", "chartId": "hv_loss_curve"},
            {
                "id": "regret_heading",
                "type": "markdown",
                "body": (
                    "## B split 决定了统一推荐预算\n\n"
                    "GP 对齐 J 同时考虑归一化工期与有效成本。300 评价时总体和 G split 的中位后悔值已为 0%，但 B split 仍为 1.613%；到 400 时三个硬分组均为 0%。**因此 400 是质量与计算开销之间的最小稳健选择。**"
                ),
            },
            {"id": "regret_chart", "type": "chart", "chartId": "j_regret_curve"},
            {
                "id": "category_heading",
                "type": "markdown",
                "body": (
                    "## 类别诊断显示两个需要持续监控的角落\n\n"
                    "400 评价下，容量紧张与冗余/超预算类别的中位 J 后悔值分别为 1.613% 和 1.063%。由于每类只有 2 个场景，不宜据此否定 400 的总体推荐；但正式实验应报告类别切分，并将 500 作为这两类的敏感性对照。"
                ),
            },
            {"id": "category_chart", "type": "chart", "chartId": "category_stability"},
            {
                "id": "front_heading",
                "type": "markdown",
                "body": (
                    "## 前沿数量不是推荐预算的决定因素\n\n"
                    "Pareto 前沿规模并不随评价数单调增长，因为新解会支配旧解并改变去重后的前沿。**预算判断应以成功率、超体积和 J 后悔值为主，而不是以解的数量为主。**"
                ),
            },
            {"id": "front_chart", "type": "chart", "chartId": "front_size_curve"},
            {"id": "detail_table", "type": "table", "tableId": "budget_400_detail"},
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## 建议下一步\n\n"
                    "1. 小预算对照实验统一采用 **400 评价×3 种子**，保持与本次校准相同的随机种子策略。\n"
                    "2. 正式高质量 MOPSO 基线继续使用 `fast=50/5000/3`，不要用校准值替换正式默认配置。\n"
                    "3. 对容量紧张和冗余/超预算场景增加 400 vs 500 的类别敏感性结果，避免小样本掩盖尾部风险。"
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further Questions\n\n"
                    "- 当每类场景扩展到至少 10 个时，400 评价的类别结论是否仍稳定？\n"
                    "- 与 NSGA-II 使用相同 400 评价预算时，哪一种算法的 HV、J 和运行时间更优？\n"
                    "- 若独立种子从 3 个增加到 5 个，推荐预算是否发生变化？"
                ),
            },
            {
                "id": "caveats",
                "type": "markdown",
                "body": (
                    "## Caveats and Assumptions\n\n"
                    "推荐规则的硬门槛应用于总体与 B/G split；场景类别仅作诊断。HV 在每个场景内按 50–500 评价期间观察到的理想点与最差点归一化，因此适合比较预算收敛，不宜直接解释为跨问题的绝对性能。校准结论只适用于当前 8 个正式场景、50 粒子和 3 个种子的设置。"
                ),
            },
        ],
    }

    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "quality_curve": quality_rows,
                "regret_curve": regret_rows,
                "category_stability": category_rows,
                "front_size_curve": front_size_rows,
                "budget_400_detail": table_rows,
            },
        },
        "sources": sources,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "artifact.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(path.resolve())


if __name__ == "__main__":
    main()
