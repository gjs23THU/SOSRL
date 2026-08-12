import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

import torch

import archrule
import dqn
import hrldqn
import rule


def add_common_arguments(parser):
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--scenario-pool-size", type=int, default=50)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--refund-rate", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--architecture-lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-finetune-lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "hrl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-policy hierarchical architecture/scheduling DQN."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scheduler = subparsers.add_parser("pretrain-scheduler")
    add_common_arguments(scheduler)

    architecture = subparsers.add_parser("train-architecture")
    add_common_arguments(architecture)
    architecture.add_argument(
        "--scheduler-checkpoint",
        type=Path,
        required=True,
    )

    finetune = subparsers.add_parser("finetune")
    add_common_arguments(finetune)
    finetune.add_argument("--scheduler-checkpoint", type=Path, required=True)
    finetune.add_argument("--architecture-checkpoint", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--intdqn-checkpoint", type=Path)
    evaluate.add_argument("--eval-episodes", type=int, default=100)
    evaluate.add_argument("--eval-seed", type=int, default=20260724)
    evaluate.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "hrl_evaluation",
    )
    return parser.parse_args()


def hrl_config(args):
    return hrldqn.HRLConfig(
        episodes=args.episodes,
        scenario_pool_size=args.scenario_pool_size,
        budget=args.budget,
        refund_rate=args.refund_rate,
        gamma=args.gamma,
        n_step=args.n_step,
        architecture_lr=args.architecture_lr,
        scheduler_finetune_lr=args.scheduler_finetune_lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=args.device,
    )


def scheduler_config(config):
    return dqn.DQNConfig(
        episodes=config.episodes,
        scenario_pool_size=config.scenario_pool_size,
        rule_set="standard",
        cost_limit=config.budget,
        gamma=config.gamma,
        lr=config.architecture_lr,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        min_buffer_size=config.min_buffer_size,
        target_update_interval=config.target_update_interval,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
        device=config.device,
    )


def adaptive_pool(config, seed=None, size=None):
    dqn.set_seed(config.seed if seed is None else int(seed))
    return hrldqn.AdaptiveScenarioPool(
        config.scenario_pool_size if size is None else int(size),
        config,
    )


def validate_scheduler(scheduler_agent):
    if scheduler_agent.obs_dim != 25:
        raise ValueError("scheduler checkpoint must use the 25-dimensional state.")
    if scheduler_agent.action_dim != rule.Rule.RULE_NUM:
        raise ValueError("scheduler checkpoint must contain the four standard rules.")


def save_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def print_last(history):
    if history:
        print(json.dumps(history[-1], ensure_ascii=False, indent=2))


def pretrain_scheduler(args):
    config = hrl_config(args)
    dqn.set_seed(config.seed)
    sched_config = scheduler_config(config)
    pool = dqn.ScenarioPool(
        size=config.scenario_pool_size,
        cost_limit=config.budget,
        shared_mission=False,
    )
    agent, history = dqn.train_dqn(sched_config, pool)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent.save_checkpoint(
        args.output_dir / "scheduler.pt",
        training_state={"stage": "pretrain-scheduler", "episodes": len(history)},
    )
    save_csv(args.output_dir / "scheduler_history.csv", history)
    save_json(args.output_dir / "scheduler_config.json", asdict(sched_config))
    print_last(history)


def train_architecture(args):
    config = hrl_config(args)
    scheduler_agent, _ = dqn.DQNAgent.load_checkpoint(
        args.scheduler_checkpoint,
        device=config.device,
        load_optimizer=False,
    )
    validate_scheduler(scheduler_agent)
    pool = adaptive_pool(config)
    architecture_agent, history = hrldqn.train_architecture(
        config,
        scheduler_agent,
        pool,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_agent.save_checkpoint(
        args.output_dir / "architecture.pt",
        training_state={"stage": "train-architecture", "episodes": len(history)},
    )
    hrldqn.save_combined_checkpoint(
        args.output_dir / "hrl.pt",
        architecture_agent,
        scheduler_agent,
        training_state={"stage": "train-architecture", "episodes": len(history)},
    )
    save_csv(args.output_dir / "architecture_history.csv", history)
    save_json(args.output_dir / "hrl_config.json", asdict(config))
    print_last(history)


def finetune(args):
    config = hrl_config(args)
    architecture_agent, _ = hrldqn.ArchitectureDQNAgent.load_checkpoint(
        args.architecture_checkpoint,
        device=config.device,
    )
    scheduler_agent, _ = dqn.DQNAgent.load_checkpoint(
        args.scheduler_checkpoint,
        device=config.device,
    )
    validate_scheduler(scheduler_agent)
    pool = adaptive_pool(config)
    history = hrldqn.finetune(
        config,
        architecture_agent,
        scheduler_agent,
        pool,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_agent.save_checkpoint(
        args.output_dir / "architecture.pt",
        training_state={"stage": "finetune", "episodes": len(history)},
    )
    scheduler_agent.save_checkpoint(
        args.output_dir / "scheduler.pt",
        training_state={"stage": "finetune", "episodes": len(history)},
    )
    hrldqn.save_combined_checkpoint(
        args.output_dir / "hrl.pt",
        architecture_agent,
        scheduler_agent,
        training_state={"stage": "finetune", "episodes": len(history)},
    )
    save_csv(args.output_dir / "finetune_history.csv", history)
    save_json(args.output_dir / "hrl_config.json", asdict(config))
    print_last(history)


def summarize(rows):
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
                "mean_architecture_changes": mean(
                    row["architecture_changes"] for row in model_rows
                ),
                "budget_violation_rate": mean(
                    float(row["budget_violation"]) for row in model_rows
                ),
            }
        )
    return summary


def evaluate(args):
    architecture_agent, scheduler_agent, checkpoint = (
        hrldqn.load_combined_checkpoint(
            args.checkpoint,
            device=args.device,
            load_optimizer=False,
        )
    )
    validate_scheduler(scheduler_agent)
    config = architecture_agent.config
    pool = adaptive_pool(config, seed=args.eval_seed, size=args.eval_episodes)
    scenarios = list(pool.scenarios)

    hrl_rows = hrldqn.evaluate_hrl(
        architecture_agent,
        scheduler_agent,
        scenarios,
        budget=config.budget,
        refund_rate=config.refund_rate,
    )
    for row in hrl_rows:
        row["model"] = "hrl"
    rows = list(hrl_rows)
    rows.extend(
        hrldqn.evaluate_static_scheduler(
            scheduler_agent,
            scenarios,
            label="static_initial",
            budget=config.budget,
        )
    )
    rows.extend(
        hrldqn.evaluate_static_scheduler(
            scheduler_agent,
            scenarios,
            label="full_system_upper_bound",
            full_systems=True,
            budget=config.budget,
        )
    )
    rows.extend(
        hrldqn.evaluate_architecture_baseline(
            hrldqn.FixedArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            label="fixed_architecture_rules",
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
    )
    if args.intdqn_checkpoint is not None:
        import intdqn

        flat_agent, _ = intdqn.IntDQNAgent.load_checkpoint(
            args.intdqn_checkpoint,
            device=args.device,
            load_optimizer=False,
        )
        rows.extend(
            hrldqn.evaluate_flat_intdqn(
                flat_agent,
                scenarios,
                budget=config.budget,
            )
        )
    dqn.set_seed(args.eval_seed)
    rows.extend(
        hrldqn.evaluate_architecture_baseline(
            hrldqn.RandomArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            label="random_architecture_rules",
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
    )

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "results.csv", rows)
    save_csv(args.output_dir / "summary.csv", summary)
    save_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "paired_scenarios": True,
            "unseen_missions": True,
            "checkpoint": str(args.checkpoint.resolve()),
            "training_state": checkpoint.get("training_state", {}),
            "architecture_actions": list(archrule.ArchitectureRule.RULE_NAMES),
            "scheduler_actions": list(rule.Rule.RULE_NAMES),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    if args.command == "pretrain-scheduler":
        pretrain_scheduler(args)
    elif args.command == "train-architecture":
        train_architecture(args)
    elif args.command == "finetune":
        finetune(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
