"""Paired statistics and Pareto selection for GP/BDQN tuning.

The helpers in this module deliberately operate on plain row dictionaries so
long-running experiment orchestration can be resumed and independently audited
from its CSV artifacts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _as_float(row: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    value = row.get(field, default)
    if value in (None, "", "None"):
        return float(default)
    return float(value)


def _as_int(row: Mapping[str, Any], field: str, default: int = 0) -> int:
    return int(round(_as_float(row, field, default)))


def _repeat_value(row: Mapping[str, Any]) -> str:
    for field in ("repeat", "seed", "run_seed", "replicate"):
        if field in row:
            return str(row[field])
    return "0"


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        _repeat_value(row),
        str(row.get("split", row.get("evaluation_split", "all"))),
        str(row["scenario_hash"]),
    )


def hierarchical_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, float | int | list[float]]:
    """Mean and hierarchical bootstrap CI over repeats and scenarios."""

    usable = [row for row in rows if row.get(field) not in (None, "", "None")]
    if not usable:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "n": 0}
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in usable:
        grouped[_repeat_value(row)].append(_as_float(row, field))
    repeat_ids = sorted(grouped)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        selected_repeats = rng.choice(repeat_ids, size=len(repeat_ids), replace=True)
        repeat_means = []
        for repeat_id in selected_repeats:
            values = np.asarray(grouped[str(repeat_id)], dtype=np.float64)
            repeat_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        estimates[index] = float(np.mean(repeat_means))
    point = float(np.mean([np.mean(grouped[item]) for item in repeat_ids]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean": point,
        "ci95": [float(low), float(high)],
        "n": len(usable),
        "repeat_means": [float(np.mean(grouped[item])) for item in repeat_ids],
    }


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty result set.")
    successful = [row for row in rows if _as_bool(row.get("success", False))]
    makespan = hierarchical_mean_ci(
        successful, "makespan", samples=samples, seed=seed
    )
    final_cost = hierarchical_mean_ci(
        successful, "final_net_cost", samples=samples, seed=seed + 1
    )
    peak_cost = hierarchical_mean_ci(
        successful, "peak_net_cost", samples=samples, seed=seed + 2
    )
    j = hierarchical_mean_ci(
        rows, "failure_aware_j", samples=samples, seed=seed + 3
    )
    binary_rows = [
        {
            **dict(row),
            "failure_value": float(not _as_bool(row.get("success", False))),
            "budget_violation_value": float(
                _as_bool(row.get("ever_over_budget", False))
            ),
        }
        for row in rows
    ]
    failure = hierarchical_mean_ci(
        binary_rows, "failure_value", samples=samples, seed=seed + 4
    )
    budget = hierarchical_mean_ci(
        binary_rows, "budget_violation_value", samples=samples, seed=seed + 5
    )
    architecture = hierarchical_mean_ci(
        rows, "architecture_changes", samples=samples, seed=seed + 6
    )
    repeats = sorted({_repeat_value(row) for row in rows})
    repeat_points = []
    for repeat in repeats:
        repeat_rows = [row for row in rows if _repeat_value(row) == repeat]
        repeat_success = [
            row for row in repeat_rows if _as_bool(row.get("success", False))
        ]
        repeat_points.append(
            {
                "repeat": repeat,
                "episodes": len(repeat_rows),
                "failure_rate": 1.0 - len(repeat_success) / len(repeat_rows),
                "mean_success_makespan": (
                    float(np.mean([_as_float(row, "makespan") for row in repeat_success]))
                    if repeat_success
                    else float("nan")
                ),
                "mean_final_cost": (
                    float(
                        np.mean(
                            [_as_float(row, "final_net_cost") for row in repeat_success]
                        )
                    )
                    if repeat_success
                    else float("nan")
                ),
                "mean_peak_cost": (
                    float(
                        np.mean(
                            [_as_float(row, "peak_net_cost") for row in repeat_success]
                        )
                    )
                    if repeat_success
                    else float("nan")
                ),
                "mean_j": float(
                    np.mean([_as_float(row, "failure_aware_j") for row in repeat_rows])
                ),
                "budget_violation_rate": sum(
                    _as_bool(row.get("ever_over_budget", False))
                    for row in repeat_rows
                )
                / len(repeat_rows),
                "mean_architecture_changes": float(
                    np.mean(
                        [_as_float(row, "architecture_changes") for row in repeat_rows]
                    )
                ),
            }
        )
    return {
        "episodes": len(rows),
        "repeats": repeats,
        "failure_count": len(rows) - len(successful),
        "failure_rate": 1.0 - len(successful) / len(rows),
        "failure_rate_ci95": failure["ci95"],
        "mean_success_makespan": makespan["mean"],
        "mean_success_makespan_ci95": makespan["ci95"],
        "mean_final_cost": final_cost["mean"],
        "mean_final_cost_ci95": final_cost["ci95"],
        "mean_peak_cost": peak_cost["mean"],
        "mean_peak_cost_ci95": peak_cost["ci95"],
        "mean_j": j["mean"],
        "mean_j_ci95": j["ci95"],
        "budget_violation_rate": sum(
            _as_bool(row.get("ever_over_budget", False)) for row in rows
        )
        / len(rows),
        "budget_violation_rate_ci95": budget["ci95"],
        "mean_architecture_changes": architecture["mean"],
        "mean_architecture_changes_ci95": architecture["ci95"],
        "invalid_action_count": sum(
            _as_int(row, "invalid_action_count") for row in rows
        ),
        "provider_invariant_violations": sum(
            _as_int(row, "provider_invariant_violations") for row in rows
        ),
        "architecture_changes": sum(
            _as_int(row, "architecture_changes") for row in rows
        ),
        "repeat_points": repeat_points,
    }


def paired_difference_ci(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    both_success: bool = False,
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, Any]:
    """Paired candidate-minus-baseline difference with repeat clustering."""

    baseline = {_row_key(row): row for row in baseline_rows}
    candidate = {_row_key(row): row for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("paired result sets do not share repeat/split/scenario keys.")
    deltas: dict[str, list[float]] = defaultdict(list)
    for key in sorted(baseline):
        left = baseline[key]
        right = candidate[key]
        if both_success and not (
            _as_bool(left.get("success", False))
            and _as_bool(right.get("success", False))
        ):
            continue
        deltas[key[0]].append(_as_float(right, field) - _as_float(left, field))
    if not deltas or any(not values for values in deltas.values()):
        return {
            "mean_difference": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "n": 0,
        }
    repeat_ids = sorted(deltas)
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        chosen = rng.choice(repeat_ids, size=len(repeat_ids), replace=True)
        means = []
        for repeat_id in chosen:
            values = np.asarray(deltas[str(repeat_id)], dtype=np.float64)
            means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
        estimates[index] = float(np.mean(means))
    point = float(np.mean([np.mean(deltas[item]) for item in repeat_ids]))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_difference": point,
        "ci95": [float(low), float(high)],
        "n": sum(len(values) for values in deltas.values()),
        "repeat_mean_differences": [
            float(np.mean(deltas[item])) for item in repeat_ids
        ],
    }


def select_aggregate_checkpoint(
    rows: Sequence[Mapping[str, Any]],
    *,
    step_field: str = "target_environment_steps",
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, Any]:
    """Select one step from aggregate repeats without selecting a seed."""

    by_step: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[_as_int(row, step_field)].append(row)
    summaries = []
    for index, (step, step_rows) in enumerate(sorted(by_step.items())):
        summary = summarize_rows(step_rows, samples=samples, seed=seed + index * 10)
        summary["target_environment_steps"] = int(step)
        summaries.append(summary)
    minimum_failure = min(int(item["failure_count"]) for item in summaries)
    safe = [item for item in summaries if int(item["failure_count"]) == minimum_failure]
    mean_best = min(safe, key=lambda item: float(item["mean_success_makespan"]))
    best_low, best_high = map(float, mean_best["mean_success_makespan_ci95"])
    overlapping = []
    for item in safe:
        low, high = map(float, item["mean_success_makespan_ci95"])
        if low <= best_high and best_low <= high:
            overlapping.append(item)
    winner = min(overlapping, key=lambda item: int(item["target_environment_steps"]))
    return {
        "selection_order": [
            "minimum_aggregate_failure_count",
            "minimum_mean_success_makespan",
            "earliest_step_when_ci95_overlaps_best",
        ],
        "selected_step": int(winner["target_environment_steps"]),
        "selected_metrics": winner,
        "checkpoint_summaries": summaries,
    }


def decide_rule_lr_package(
    r0_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    r1_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    minimum_relative_improvement: float = 0.01,
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, Any]:
    """Choose R1 only when its preregistered evidence is conclusive."""

    required = {"iid", "ood"}
    if set(r0_by_split) != required or set(r1_by_split) != required:
        raise ValueError("rule learning-rate confirmation requires iid and ood splits.")
    split_summaries = {}
    pooled_r0 = []
    pooled_r1 = []
    failure_no_worse = True
    invariants_safe = True
    for index, split in enumerate(("iid", "ood")):
        r0 = [dict(row, split=split) for row in r0_by_split[split]]
        r1 = [dict(row, split=split) for row in r1_by_split[split]]
        left = summarize_rows(r0, samples=samples, seed=seed + index * 20)
        right = summarize_rows(r1, samples=samples, seed=seed + index * 20 + 10)
        failure_no_worse &= int(right["failure_count"]) <= int(left["failure_count"])
        invariants_safe &= (
            int(right["invalid_action_count"]) == 0
            and int(right["provider_invariant_violations"]) == 0
        )
        split_summaries[split] = {"r0": left, "r1": right}
        pooled_r0.extend(r0)
        pooled_r1.extend(r1)
    paired = paired_difference_ci(
        pooled_r0,
        pooled_r1,
        "makespan",
        both_success=True,
        samples=samples,
        seed=seed + 100,
    )
    r0_success = [row for row in pooled_r0 if _as_bool(row.get("success", False))]
    r1_success = [row for row in pooled_r1 if _as_bool(row.get("success", False))]
    r0_mean = float(np.mean([_as_float(row, "makespan") for row in r0_success]))
    r1_mean = float(np.mean([_as_float(row, "makespan") for row in r1_success]))
    relative_improvement = (r0_mean - r1_mean) / max(abs(r0_mean), 1e-12)
    r1_wins = (
        failure_no_worse
        and invariants_safe
        and relative_improvement >= float(minimum_relative_improvement)
        and float(paired["ci95"][1]) < 0.0
    )
    return {
        "winner": "R1" if r1_wins else "R0",
        "r1_requires_prefix_retrain": bool(r1_wins),
        "failure_no_worse_on_each_split": bool(failure_no_worse),
        "invariants_safe": bool(invariants_safe),
        "relative_makespan_improvement": float(relative_improvement),
        "paired_makespan_difference": paired,
        "minimum_relative_improvement": float(minimum_relative_improvement),
        "split_summaries": split_summaries,
    }


def _budget_difference_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        converted.append(
            {
                **dict(row),
                "budget_violation_value": float(
                    _as_bool(row.get("ever_over_budget", False))
                ),
            }
        )
    return converted


def robust_pareto_selection(
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline: str,
    metadata: Mapping[str, Mapping[str, Any]] | None = None,
    budget_guard: float = 0.02,
    samples: int = 5000,
    seed: int = 20261000,
) -> dict[str, Any]:
    """Safety-filter candidates, build a confidence-aware front, choose knee."""

    if baseline not in rows_by_candidate:
        raise ValueError("Pareto baseline is missing.")
    metadata = metadata or {}
    summaries = {
        name: summarize_rows(rows, samples=samples, seed=seed + index * 100)
        for index, (name, rows) in enumerate(sorted(rows_by_candidate.items()))
    }
    baseline_rows = rows_by_candidate[baseline]
    baseline_summary = summaries[baseline]
    safety = {}
    eligible = []
    for index, (name, rows) in enumerate(sorted(rows_by_candidate.items())):
        budget = paired_difference_ci(
            _budget_difference_rows(baseline_rows),
            _budget_difference_rows(rows),
            "budget_violation_value",
            samples=samples,
            seed=seed + 1000 + index,
        )
        summary = summaries[name]
        accepted = (
            int(summary["failure_count"]) <= int(baseline_summary["failure_count"])
            and int(summary["invalid_action_count"]) == 0
            and int(summary["provider_invariant_violations"]) == 0
            and float(budget["ci95"][1]) <= float(budget_guard)
        )
        safety[name] = {
            "accepted": bool(accepted),
            "budget_violation_difference": budget,
        }
        if accepted:
            eligible.append(name)

    comparisons = {}
    dominated = set()
    for left_index, left in enumerate(eligible):
        for right in eligible[left_index + 1 :]:
            pair_key = f"{left}__vs__{right}"
            right_minus_left_m = paired_difference_ci(
                rows_by_candidate[left],
                rows_by_candidate[right],
                "makespan",
                both_success=True,
                samples=samples,
                seed=seed + 2000 + len(comparisons) * 2,
            )
            right_minus_left_c = paired_difference_ci(
                rows_by_candidate[left],
                rows_by_candidate[right],
                "final_net_cost",
                both_success=True,
                samples=samples,
                seed=seed + 2001 + len(comparisons) * 2,
            )
            comparisons[pair_key] = {
                "right_minus_left_makespan": right_minus_left_m,
                "right_minus_left_final_cost": right_minus_left_c,
            }
            m_low, m_high = map(float, right_minus_left_m["ci95"])
            c_low, c_high = map(float, right_minus_left_c["ci95"])
            right_dominates = m_high <= 0.0 and c_high <= 0.0 and (
                m_high < 0.0 or c_high < 0.0
            )
            left_dominates = m_low >= 0.0 and c_low >= 0.0 and (
                m_low > 0.0 or c_low > 0.0
            )
            if right_dominates:
                dominated.add(left)
            if left_dominates:
                dominated.add(right)

    front = sorted(name for name in eligible if name not in dominated)
    if not front:
        raise RuntimeError("no safe Pareto candidate remains.")
    baseline_makespan = float(baseline_summary["mean_success_makespan"])
    baseline_cost = float(baseline_summary["mean_final_cost"])
    makespans = np.asarray(
        [
            float(summaries[name]["mean_success_makespan"])
            / max(abs(baseline_makespan), 1e-12)
            for name in front
        ]
    )
    costs = np.asarray(
        [
            float(summaries[name]["mean_final_cost"])
            / max(abs(baseline_cost), 1e-12)
            for name in front
        ]
    )
    ideal = np.asarray([float(np.min(makespans)), float(np.min(costs))])
    distances = np.sqrt((makespans - ideal[0]) ** 2 + (costs - ideal[1]) ** 2)

    def knee_key(index: int) -> tuple[float, float, float, int, int, str]:
        name = front[index]
        meta = metadata.get(name, {})
        return (
            float(distances[index]),
            float(makespans[index]),
            float(costs[index]),
            int(meta.get("gp_nodes", 10**9)),
            int(meta.get("bdqn_step", 10**9)),
            name,
        )

    knee_index = min(range(len(front)), key=knee_key)
    return {
        "baseline": baseline,
        "objective_axes": ["both_success_makespan", "final_net_cost"],
        "summaries": summaries,
        "safety": safety,
        "pairwise_comparisons": comparisons,
        "pareto_front": front,
        "knee": front[knee_index],
        "knee_distance": float(distances[knee_index]),
        "knee_normalization": {
            "baseline": baseline,
            "baseline_makespan": baseline_makespan,
            "baseline_final_cost": baseline_cost,
            "ideal_point": ideal.tolist(),
            "candidate_points": {
                name: [float(makespans[index]), float(costs[index])]
                for index, name in enumerate(front)
            },
        },
        "metadata": {name: dict(metadata.get(name, {})) for name in rows_by_candidate},
    }
