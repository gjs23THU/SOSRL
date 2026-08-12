"""Paired evaluation and summary utilities."""

import hashlib
import json
from statistics import mean, median

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


def scenario_payload(scenario_idx, architecture, mission):
    payload = {
        "scenario_idx": int(scenario_idx),
        "architecture_system_indices": sorted(
            int(system.index) for system in architecture
        ),
        "architecture_cost": float(sum(system.cost for system in architecture)),
        "mission": mission_payload(mission),
    }
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
        summary.append(
            {
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
        )
    return summary


evaluate_dqn = scheduler.evaluate_dqn
evaluate_hrl = hierarchical.evaluate_hrl
evaluate_static_scheduler = hierarchical.evaluate_static_scheduler
evaluate_architecture_baseline = hierarchical.evaluate_architecture_baseline
evaluate_flat_intdqn = hierarchical.evaluate_flat_intdqn
