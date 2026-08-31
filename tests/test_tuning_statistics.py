import pytest

from sosrl.workflows.tuning_statistics import (
    decide_rule_lr_package,
    paired_difference_ci,
    robust_pareto_selection,
    select_aggregate_checkpoint,
    summarize_rows,
)


def _rows(
    name,
    *,
    makespan,
    final_cost=3000.0,
    failure=False,
    budget=False,
    repeats=(1, 2, 3),
):
    rows = []
    for repeat in repeats:
        for scenario in range(8):
            rows.append(
                {
                    "model": name,
                    "seed": repeat,
                    "scenario_hash": f"scenario-{scenario}",
                    "success": not failure,
                    "makespan": (
                        float(makespan) + repeat * 0.01 + (scenario - 3.5) * 0.2
                    ),
                    "final_net_cost": float(final_cost),
                    "peak_net_cost": float(final_cost) + 100.0,
                    "failure_aware_j": float(makespan) / 100.0,
                    "ever_over_budget": bool(budget),
                    "invalid_action_count": 0,
                    "provider_invariant_violations": 0,
                    "architecture_changes": 1,
                }
            )
    return rows


def test_statistics_report_all_repeat_means_and_paired_ci():
    baseline = _rows("baseline", makespan=100.0)
    candidate = _rows("candidate", makespan=95.0)

    summary = summarize_rows(candidate, samples=200, seed=1)
    paired = paired_difference_ci(
        baseline,
        candidate,
        "makespan",
        both_success=True,
        samples=200,
        seed=2,
    )

    assert summary["repeats"] == ["1", "2", "3"]
    assert len(summary["repeat_points"]) == 3
    assert len(summary["mean_success_makespan_ci95"]) == 2
    assert len(summary["mean_architecture_changes_ci95"]) == 2
    assert paired["mean_difference"] == pytest.approx(-5.0)
    assert paired["ci95"][1] < 0.0


def test_aggregate_checkpoint_uses_all_seeds_and_prefers_earlier_ci_tie():
    rows = []
    for step, makespan in ((0, 101.0), (5000, 100.0), (10000, 99.9)):
        for row in _rows(f"step-{step}", makespan=makespan):
            row["target_environment_steps"] = step
            rows.append(row)

    selection = select_aggregate_checkpoint(rows, samples=200, seed=3)

    assert selection["selected_step"] == 5000
    assert selection["selected_metrics"]["repeats"] == ["1", "2", "3"]


def test_rule_r1_requires_failure_safety_ci_and_one_percent_gain():
    r0 = {
        "iid": _rows("r0", makespan=100.0),
        "ood": _rows("r0", makespan=110.0),
    }
    r1 = {
        "iid": _rows("r1", makespan=97.0),
        "ood": _rows("r1", makespan=106.0),
    }

    decision = decide_rule_lr_package(r0, r1, samples=300, seed=4)
    assert decision["winner"] == "R1"
    assert decision["r1_requires_prefix_retrain"]

    tied = decide_rule_lr_package(
        r0,
        {
            "iid": _rows("r1", makespan=99.5),
            "ood": _rows("r1", makespan=109.5),
        },
        samples=300,
        seed=5,
    )
    assert tied["winner"] == "R0"


def test_confidence_aware_pareto_front_and_knee_are_deterministic():
    candidates = {
        "S0": _rows("S0", makespan=100.0, final_cost=3000.0),
        "fast": _rows("fast", makespan=90.0, final_cost=3200.0),
        "cheap": _rows("cheap", makespan=110.0, final_cost=2700.0),
        "dominated": _rows("dominated", makespan=120.0, final_cost=3300.0),
    }

    selection = robust_pareto_selection(
        candidates,
        baseline="S0",
        metadata={
            "S0": {"gp_nodes": 10, "bdqn_step": 0},
            "fast": {"gp_nodes": 8, "bdqn_step": 20000},
            "cheap": {"gp_nodes": 7, "bdqn_step": 10000},
        },
        samples=300,
        seed=6,
    )

    assert "dominated" not in selection["pareto_front"]
    assert set(selection["pareto_front"]) == {"S0", "fast", "cheap"}
    assert selection["knee"] in selection["pareto_front"]


def test_unsafe_candidate_is_removed_before_pareto_selection():
    baseline = _rows("S0", makespan=100.0)
    unsafe = _rows("unsafe", makespan=80.0)
    unsafe[0]["invalid_action_count"] = 1

    selection = robust_pareto_selection(
        {"S0": baseline, "unsafe": unsafe},
        baseline="S0",
        samples=200,
        seed=7,
    )

    assert not selection["safety"]["unsafe"]["accepted"]
    assert selection["pareto_front"] == ["S0"]
