"""Summarize anytime NSGA-II milestone fronts across scenario directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, median
from typing import Iterable

from sosrl.multiobjective_calibration import run_calibration


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nondominated(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    unique = sorted(set(points))
    return [
        point
        for index, point in enumerate(unique)
        if not any(
            other[0] <= point[0]
            and other[1] <= point[1]
            and other != point
            for other_index, other in enumerate(unique)
            if other_index != index
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
    max_budget = max(fronts)
    final_hv = hypervolumes[max_budget]
    final_row = next(
        row
        for row in summary_rows
        if int(row["evaluation_budget_per_run"]) == max_budget
    )
    final_j = (
        None if not final_row["gp_aligned_j"] else float(final_row["gp_aligned_j"])
    )
    metrics = []
    for row in summary_rows:
        budget = int(row["evaluation_budget_per_run"])
        gp_j = None if not row["gp_aligned_j"] else float(row["gp_aligned_j"])
        metrics.append(
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
    return metrics


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: list[tuple[str, str, list[dict[str, object]]]] = [("all", "all", rows)]
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
        for budget in sorted(
            {int(row["evaluation_budget_per_run"]) for row in group_rows}
        ):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-budget", type=int)
    args = parser.parse_args()

    summary = run_calibration(
        args.input_dir,
        args.output_dir,
        algorithm="NSGA-II",
        max_budget=args.max_budget,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return

    scenario_dirs = []
    for input_dir in args.input_dir:
        scenario_dirs.extend(
            sorted(path for path in input_dir.glob("scenario_*") if path.is_dir())
        )
    rows = [
        row
        for scenario_dir in scenario_dirs
        for row in scenario_metrics(scenario_dir, max_budget=args.max_budget)
    ]
    if not rows:
        raise SystemExit("no milestone scenario results found")
    curve = aggregate_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "scenario_budget_metrics.csv", rows)
    write_csv(args.output_dir / "budget_curve.csv", curve)

    all_rows = [
        row
        for row in curve
        if row["group_type"] == "all" and row["group_value"] == "all"
    ]
    split_rows = [row for row in curve if row["group_type"] == "split"]
    final_budget = max(int(row["evaluation_budget_per_run"]) for row in all_rows)
    required_groups = [
        ("all", "all"),
        *[("split", value) for value in sorted({row["group_value"] for row in split_rows})],
    ]

    def group_row(group_type: str, group_value: str, budget: int):
        return next(
            row
            for row in curve
            if row["group_type"] == group_type
            and row["group_value"] == group_value
            and int(row["evaluation_budget_per_run"]) == budget
        )

    eligible_budgets = []
    for budget in sorted(int(row["evaluation_budget_per_run"]) for row in all_rows):
        passes = True
        for group_type, group_value in required_groups:
            candidate = group_row(group_type, group_value, budget)
            final = group_row(group_type, group_value, final_budget)
            passes = bool(
                passes
                and float(candidate["success_rate"]) >= float(final["success_rate"])
                and candidate["median_relative_hv_loss_to_max"] is not None
                and float(candidate["median_relative_hv_loss_to_max"]) <= 0.01
                and candidate["median_relative_j_regret_to_max"] is not None
                and float(candidate["median_relative_j_regret_to_max"]) <= 0.01
            )
        if passes:
            eligible_budgets.append(budget)
    recommendation = None if not eligible_budgets else eligible_budgets[0]
    summary = {
        "input_dirs": [str(path.resolve()) for path in args.input_dir],
        "scenario_count": len({str(row["scenario_hash"]) for row in rows}),
        "max_budget_filter": args.max_budget,
        "milestones": sorted(
            {int(row["evaluation_budget_per_run"]) for row in rows}
        ),
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
    }
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
