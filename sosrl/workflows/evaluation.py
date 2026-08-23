"""Paired evaluation and summary utilities."""

import hashlib
import json
from statistics import mean, median

from .. import domain as syn
from .. import environment as env
from ..rl.agent import DQNAgent
from . import hierarchical, scheduler


EVAL_CONFIG_FIELDS = (
    "selected_system_num",
    "min_system_num",
    "max_system_num",
    "cost_limit",
)


def load_scheduler_agents(model_specs, device):
    agents = {}
    checkpoint_metadata = {}
    for label, path in model_specs:
        if label in agents:
            raise ValueError(f"Duplicate model label: {label}")
        agent, checkpoint = DQNAgent.load_checkpoint(
            path,
            device=device,
            load_optimizer=False,
        )
        agents[label] = agent
        checkpoint_metadata[label] = {
            "path": str(path.resolve()),
            "training_state": checkpoint.get("training_state", {}),
            "config": checkpoint["config"],
        }
    return agents, checkpoint_metadata


def validate_eval_configs(agents):
    labels = list(agents)
    if not labels:
        raise ValueError("at least one scheduler checkpoint is required")
    reference = agents[labels[0]].config
    mismatches = []
    for label in labels[1:]:
        config = agents[label].config
        for field in EVAL_CONFIG_FIELDS:
            expected = getattr(reference, field)
            actual = getattr(config, field)
            if actual != expected:
                mismatches.append(
                    f"{label}.{field}={actual!r}, expected {expected!r}"
                )
    if mismatches:
        raise ValueError(
            "Checkpoints do not share the same scenario constraints: "
            + "; ".join(mismatches)
        )
    return reference


def mission_payload(mission):
    return [
        {
            "task_idx": int(task.index),
            "release_time": int(task.release_time),
            "due_time": int(task.due_time),
            "operations": [
                {
                    "op_idx": int(operation.index),
                    "func_type": int(operation.func_type),
                    "duration": int(operation.duration),
                    "release_time": int(operation.release_time),
                }
                for operation in task.operations
            ],
        }
        for task in mission
    ]


def scenario_payload(
    scenario_idx,
    architecture,
    mission,
    *,
    category=None,
    budget=None,
    refund_rate=None,
    split=None,
    static_feasible_architecture=None,
):
    payload = {
        "scenario_idx": int(scenario_idx),
        "architecture_system_indices": sorted(
            int(system.index) for system in architecture
        ),
        "architecture_cost": float(sum(system.cost for system in architecture)),
        "mission": mission_payload(mission),
    }
    optional = {
        "category": category,
        "budget": None if budget is None else float(budget),
        "refund_rate": None if refund_rate is None else float(refund_rate),
        "split": split,
        "static_feasible_system_indices": (
            None
            if static_feasible_architecture is None
            else sorted(
                int(system.index) for system in static_feasible_architecture
            )
        ),
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["scenario_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def mission_from_payload(payload):
    """Reconstruct a complete mission from its JSON-safe representation."""
    mission = []
    for task_item in payload:
        task_idx = int(task_item["task_idx"])
        operations = [
            syn.Operation(
                index=int(operation["op_idx"]),
                name=f"Op_{task_idx}_{int(operation['op_idx'])}",
                func_type=int(operation["func_type"]),
                duration=int(operation["duration"]),
                release_time=int(operation["release_time"]),
            )
            for operation in task_item["operations"]
        ]
        mission.append(
            syn.Task(
                index=task_idx,
                name=f"Task_{task_idx}",
                operations=operations,
                release_time=int(task_item["release_time"]),
                due_time=int(task_item["due_time"]),
            )
        )
    return mission


def scenario_from_payload(payload):
    architecture = tuple(
        env.FULL_SOS[int(index)]
        for index in payload["architecture_system_indices"]
    )
    return architecture, mission_from_payload(payload["mission"])


def static_scenario_from_payload(payload):
    """Reconstruct the registered feasible static architecture and mission."""

    indices = payload.get("static_feasible_system_indices")
    if indices is None:
        raise ValueError("scenario does not register a static feasible architecture.")
    architecture = tuple(env.FULL_SOS[int(index)] for index in indices)
    return architecture, mission_from_payload(payload["mission"])


def verify_scenario_payload(payload):
    candidate = dict(payload)
    expected = candidate.pop("scenario_hash", None)
    canonical = json.dumps(
        candidate,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError("scenario payload hash mismatch.")
    return actual


def build_independent_pool(config, episodes, eval_seed):
    scheduler.set_seed(eval_seed)
    return scheduler.ScenarioPool(
        size=episodes,
        selected_system_num=config.selected_system_num,
        min_system_num=config.min_system_num,
        max_system_num=config.max_system_num,
        cost_limit=config.cost_limit,
        shared_mission=False,
        mission=None,
    )


def attach_scenario_metadata(label, results, scenario_manifest):
    rows = []
    for result, scenario in zip(results, scenario_manifest, strict=True):
        row = {
            "model": label,
            "scenario_hash": scenario["scenario_hash"],
            "architecture_size": len(
                scenario["architecture_system_indices"]
            ),
            "architecture_cost": scenario["architecture_cost"],
        }
        row.update(result)
        rows.append(row)
    return rows


def summarize_scheduler(label, rows):
    successful = [row for row in rows if row["success"]]
    return {
        "model": label,
        "evaluation_scenarios": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / max(len(rows), 1),
        "mean_success_makespan": (
            mean(row["makespan"] for row in successful)
            if successful
            else None
        ),
        "median_success_makespan": (
            median(row["makespan"] for row in successful)
            if successful
            else None
        ),
        "mean_reward": mean(row["reward"] for row in rows),
        "mean_assigned_ops": mean(row["assigned_ops"] for row in rows),
    }


def paired_rows(rows_by_model):
    labels = list(rows_by_model)
    rows = []
    for left_idx, left_label in enumerate(labels):
        for right_label in labels[left_idx + 1 :]:
            differences = []
            left_only_success = 0
            right_only_success = 0
            for left, right in zip(
                rows_by_model[left_label],
                rows_by_model[right_label],
                strict=True,
            ):
                if left["scenario_hash"] != right["scenario_hash"]:
                    raise ValueError("Models were not evaluated on matching scenarios.")
                if left["success"] and right["success"]:
                    differences.append(left["makespan"] - right["makespan"])
                elif left["success"]:
                    left_only_success += 1
                elif right["success"]:
                    right_only_success += 1
            rows.append(
                {
                    "left_model": left_label,
                    "right_model": right_label,
                    "common_success_count": len(differences),
                    "left_only_success_count": left_only_success,
                    "right_only_success_count": right_only_success,
                    "mean_left_minus_right_makespan": (
                        mean(differences) if differences else None
                    ),
                    "left_faster_count": sum(value < 0 for value in differences),
                    "right_faster_count": sum(value > 0 for value in differences),
                    "tie_count": sum(value == 0 for value in differences),
                }
            )
    return rows


def paired_adaptive_comparisons(
    rows,
    *,
    reference_label="hrl",
    candidate_labels=None,
):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    if reference_label not in grouped:
        raise ValueError(f"reference model is missing: {reference_label}")
    labels = (
        list(candidate_labels)
        if candidate_labels is not None
        else [label for label in grouped if label != reference_label]
    )
    reference_rows = grouped[reference_label]
    comparisons = []
    for label in labels:
        if label not in grouped:
            raise ValueError(f"candidate model is missing: {label}")
        candidate_rows = grouped[label]
        if len(reference_rows) != len(candidate_rows):
            raise ValueError("paired adaptive models have different scenario counts.")

        makespan_differences = []
        cost_differences = []
        reference_only_success = 0
        candidate_only_success = 0
        for reference, candidate in zip(
            reference_rows,
            candidate_rows,
            strict=True,
        ):
            if reference["scenario_hash"] != candidate["scenario_hash"]:
                raise ValueError("Models were not evaluated on matching scenarios.")
            cost_differences.append(candidate["net_cost"] - reference["net_cost"])
            if reference["success"] and candidate["success"]:
                makespan_differences.append(
                    candidate["makespan"] - reference["makespan"]
                )
            elif reference["success"]:
                reference_only_success += 1
            elif candidate["success"]:
                candidate_only_success += 1

        comparisons.append(
            {
                "reference_model": reference_label,
                "candidate_model": label,
                "paired_scenarios": len(reference_rows),
                "common_success_count": len(makespan_differences),
                "reference_only_success_count": reference_only_success,
                "candidate_only_success_count": candidate_only_success,
                "mean_candidate_minus_reference_makespan": (
                    mean(makespan_differences) if makespan_differences else None
                ),
                "median_candidate_minus_reference_makespan": (
                    median(makespan_differences) if makespan_differences else None
                ),
                "candidate_faster_count": sum(
                    difference < 0 for difference in makespan_differences
                ),
                "reference_faster_count": sum(
                    difference > 0 for difference in makespan_differences
                ),
                "makespan_tie_count": sum(
                    difference == 0 for difference in makespan_differences
                ),
                "mean_candidate_minus_reference_net_cost": mean(cost_differences),
            }
        )
    return comparisons


def compare_schedulers(agents, episodes: int, eval_seed: int):
    config = validate_eval_configs(agents)
    pool = build_independent_pool(config, episodes, eval_seed)
    manifest = [scenario_payload(*pool.get(index)) for index in range(episodes)]
    rows_by_model = {}
    for label, agent in agents.items():
        results = scheduler.evaluate_dqn(
            agent,
            pool,
            episodes=episodes,
            collect_schedule=False,
        )
        rows_by_model[label] = attach_scenario_metadata(
            label,
            results,
            manifest,
        )
    summary = [
        summarize_scheduler(label, rows)
        for label, rows in rows_by_model.items()
    ]
    return rows_by_model, summary, paired_rows(rows_by_model), manifest


def summarize_hrl(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    summary = []
    for label, model_rows in grouped.items():
        successful = [row for row in model_rows if row["success"]]
        model_summary = {
            "model": label,
            "episodes": len(model_rows),
            "success_rate": len(successful) / max(len(model_rows), 1),
            "mean_success_makespan": (
                mean(row["makespan"] for row in successful)
                if successful
                else None
            ),
            "mean_net_cost": mean(row["net_cost"] for row in model_rows),
            "mean_peak_net_cost": mean(
                row.get("peak_net_cost", row["net_cost"])
                for row in model_rows
            ),
            "mean_architecture_changes": mean(
                row["architecture_changes"] for row in model_rows
            ),
            "budget_violation_rate": mean(
                float(row["budget_violation"]) for row in model_rows
            ),
            "ever_budget_violation_rate": mean(
                float(row.get("ever_over_budget", row["budget_violation"]))
                for row in model_rows
            ),
        }
        for field in (
            "policy_parameter_count",
            "mean_architecture_inference_ms",
            "mean_scheduler_inference_ms",
            "peak_gpu_memory_mb",
        ):
            values = [row[field] for row in model_rows if field in row]
            if values:
                model_summary[field] = mean(values)
        summary.append(model_summary)
    return summary


evaluate_dqn = scheduler.evaluate_dqn
evaluate_hrl = hierarchical.evaluate_hrl
evaluate_static_scheduler = hierarchical.evaluate_static_scheduler
evaluate_architecture_baseline = hierarchical.evaluate_architecture_baseline
evaluate_flat_intdqn = hierarchical.evaluate_flat_intdqn
