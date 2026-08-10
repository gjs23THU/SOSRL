import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median

import torch

import dqn


DEFAULT_MODELS = (
    "SIG=runs/SIG1000_standard_seed4/model.pt",
    "MIG=runs/MIG1000/model.pt",
    "MEG=runs/MEG1000/model.pt",
)
EVAL_CONFIG_FIELDS = (
    "selected_system_num",
    "min_system_num",
    "max_system_num",
    "cost_limit",
)


def parse_model_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use LABEL=CHECKPOINT syntax")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError("model must use LABEL=CHECKPOINT syntax")
    return label, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SIG, MIG, and MEG checkpoints on one shared test pool "
            "that is generated independently from every training pool."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        dest="models",
        help="LABEL=CHECKPOINT; repeat for each model.",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=20260724)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "SIG_MIG_MEG_independent_eval",
    )
    args = parser.parse_args()
    if args.models is None:
        args.models = [parse_model_spec(spec) for spec in DEFAULT_MODELS]
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    return args


def load_agents(model_specs, device):
    agents = {}
    checkpoint_metadata = {}
    for label, path in model_specs:
        if label in agents:
            raise ValueError(f"Duplicate model label: {label}")
        agent, checkpoint = dqn.DQNAgent.load_checkpoint(
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
    payload["scenario_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def build_independent_pool(config, episodes, eval_seed):
    dqn.set_seed(eval_seed)
    return dqn.ScenarioPool(
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


def summarize_model(label, rows):
    successful = [row for row in rows if row["success"]]
    return {
        "model": label,
        "evaluation_scenarios": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows),
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
            left_rows = rows_by_model[left_label]
            right_rows = rows_by_model[right_label]
            common_success_differences = []
            left_only_success = 0
            right_only_success = 0
            for left, right in zip(left_rows, right_rows, strict=True):
                if left["scenario_hash"] != right["scenario_hash"]:
                    raise ValueError("Models were not evaluated on matching scenarios.")
                if left["success"] and right["success"]:
                    common_success_differences.append(
                        left["makespan"] - right["makespan"]
                    )
                elif left["success"]:
                    left_only_success += 1
                elif right["success"]:
                    right_only_success += 1

            rows.append(
                {
                    "left_model": left_label,
                    "right_model": right_label,
                    "common_success_count": len(common_success_differences),
                    "left_only_success_count": left_only_success,
                    "right_only_success_count": right_only_success,
                    "mean_left_minus_right_makespan": (
                        mean(common_success_differences)
                        if common_success_differences
                        else None
                    ),
                    "left_faster_count": sum(
                        difference < 0
                        for difference in common_success_differences
                    ),
                    "right_faster_count": sum(
                        difference > 0
                        for difference in common_success_differences
                    ),
                    "tie_count": sum(
                        difference == 0
                        for difference in common_success_differences
                    ),
                }
            )
    return rows


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    agents, checkpoint_metadata = load_agents(args.models, args.device)
    eval_config = validate_eval_configs(agents)
    scenario_pool = build_independent_pool(
        eval_config,
        args.eval_episodes,
        args.eval_seed,
    )
    scenario_manifest = [
        scenario_payload(*scenario_pool.get(index))
        for index in range(args.eval_episodes)
    ]

    rows_by_model = {}
    for label, agent in agents.items():
        print(f"evaluating {label} on {args.eval_episodes} independent scenarios...")
        results = dqn.evaluate_dqn(
            agent,
            scenario_pool,
            episodes=args.eval_episodes,
            collect_schedule=False,
        )
        rows_by_model[label] = attach_scenario_metadata(
            label,
            results,
            scenario_manifest,
        )

    summary = [
        summarize_model(label, rows)
        for label, rows in rows_by_model.items()
    ]
    comparisons = paired_rows(rows_by_model)
    all_results = [
        row
        for model_rows in rows_by_model.values()
        for row in model_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "results.csv", all_results)
    save_csv(args.output_dir / "summary.csv", summary)
    save_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    with (args.output_dir / "scenarios.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(scenario_manifest, file, ensure_ascii=False, indent=2)
    with (args.output_dir / "evaluation_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "eval_seed": args.eval_seed,
                "eval_episodes": args.eval_episodes,
                "shared_test_pool": True,
                "independent_from_training": True,
                "varied_missions": True,
                "varied_architectures": True,
                "device": args.device,
                "checkpoints": checkpoint_metadata,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))
    print(f"independent evaluation saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
