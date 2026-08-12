"""Unified command-line interface for training and evaluating SOSRL."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch

from . import domain
from .baselines import flat
from .rl.agent import ArchitectureDQNAgent, DQNAgent, IntDQNAgent
from .rl.checkpoint import load_combined_checkpoint, save_combined_checkpoint
from .rl.config import DQNConfig, HRLConfig, IntDQNConfig
from .rules.architecture import ArchitectureRule
from .rules.scheduling import Rule
from .workflows import evaluation, hierarchical, scheduler


DEFAULT_SCHEDULER_MODELS = (
    "SIG=runs/SIG1000_standard_seed4/model.pt",
    "MIG=runs/MIG1000/model.pt",
    "MEG=runs/MEG1000/model.pt",
)


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def parse_selected_system_num(value: str | None):
    if value is None or value.lower() == "none":
        return None
    if "," in value:
        low, high = value.split(",", 1)
        return int(low), int(high)
    return int(value)


def parse_model_spec(value: str):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "model must use LABEL=CHECKPOINT syntax"
        )
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "model must use LABEL=CHECKPOINT syntax"
        )
    return label, Path(path)


def save_csv(path: Path, rows) -> None:
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


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def add_device(parser) -> None:
    parser.add_argument("--device", default="auto")


def add_scheduler_arguments(parser) -> None:
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario-pool-size", type=int, default=20)
    parser.add_argument(
        "--scenario-order",
        choices=["random", "sequential"],
        default="random",
    )
    parser.add_argument(
        "--rule-set",
        choices=["standard", "huang"],
        default="standard",
    )
    parser.add_argument("--shared-mission", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=20260724)
    parser.add_argument("--selected-system-num", default="none")
    parser.add_argument("--min-system-num", type=int, default=3)
    parser.add_argument("--max-system-num", type=int, default=22)
    parser.add_argument("--cost-limit", type=float, default=8000.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "scheduler",
    )
    add_device(parser)


def add_hrl_arguments(parser) -> None:
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
        "--output-dir",
        type=Path,
        default=Path("runs") / "hrl",
    )
    add_device(parser)


def add_flat_arguments(parser) -> None:
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--fixed-mission", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "intdqn",
    )
    add_device(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sosrl",
        description="Two-policy hierarchical architecture/scheduling DQN.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_scheduler = subparsers.add_parser("train-scheduler")
    add_scheduler_arguments(train_scheduler)
    train_scheduler.set_defaults(handler=handle_train_scheduler)

    train_architecture = subparsers.add_parser("train-architecture")
    add_hrl_arguments(train_architecture)
    train_architecture.add_argument(
        "--scheduler-checkpoint",
        type=Path,
        required=True,
    )
    train_architecture.set_defaults(handler=handle_train_architecture)

    finetune = subparsers.add_parser("finetune")
    add_hrl_arguments(finetune)
    finetune.add_argument("--scheduler-checkpoint", type=Path, required=True)
    finetune.add_argument("--architecture-checkpoint", type=Path, required=True)
    finetune.set_defaults(handler=handle_finetune)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--intdqn-checkpoint", type=Path)
    evaluate.add_argument("--eval-episodes", type=int, default=100)
    evaluate.add_argument("--eval-seed", type=int, default=20260724)
    evaluate.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "hrl_evaluation",
    )
    add_device(evaluate)
    evaluate.set_defaults(handler=handle_evaluate)

    train_flat = subparsers.add_parser("train-flat")
    add_flat_arguments(train_flat)
    train_flat.set_defaults(handler=handle_train_flat)

    compare = subparsers.add_parser("compare-schedulers")
    compare.add_argument(
        "--model",
        action="append",
        type=parse_model_spec,
        dest="models",
    )
    compare.add_argument("--eval-episodes", type=int, default=100)
    compare.add_argument("--eval-seed", type=int, default=20260724)
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "SIG_MIG_MEG_independent_eval",
    )
    add_device(compare)
    compare.set_defaults(handler=handle_compare_schedulers)
    return parser


def scheduler_config(args) -> DQNConfig:
    return DQNConfig(
        episodes=args.episodes,
        scenario_pool_size=args.scenario_pool_size,
        scenario_order=args.scenario_order,
        shared_mission=args.shared_mission,
        rule_set=args.rule_set,
        selected_system_num=parse_selected_system_num(args.selected_system_num),
        min_system_num=args.min_system_num,
        max_system_num=args.max_system_num,
        cost_limit=args.cost_limit,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=resolve_device(args.device),
    )


def hrl_config(args) -> HRLConfig:
    return HRLConfig(
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
        device=resolve_device(args.device),
    )


def validate_scheduler(agent) -> None:
    if agent.obs_dim != 25:
        raise ValueError("scheduler checkpoint must use the 25-dimensional state.")
    if agent.action_dim != Rule.RULE_NUM:
        raise ValueError("scheduler checkpoint must contain four standard rules.")


def adaptive_pool(config, seed=None, size=None):
    scheduler.set_seed(config.seed if seed is None else int(seed))
    return hierarchical.AdaptiveScenarioPool(
        config.scenario_pool_size if size is None else int(size),
        config,
    )


def handle_train_scheduler(args) -> None:
    config = scheduler_config(args)
    scheduler.set_seed(config.seed)
    train_pool = scheduler.ScenarioPool(
        size=config.scenario_pool_size,
        selected_system_num=config.selected_system_num,
        min_system_num=config.min_system_num,
        max_system_num=config.max_system_num,
        cost_limit=config.cost_limit,
        shared_mission=config.shared_mission,
    )
    agent, history = scheduler.train_dqn(config, train_pool)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = agent.save_checkpoint(
        args.output_dir / "scheduler.pt",
        training_state={
            "stage": "train-scheduler",
            "episodes": len(history),
            "epsilon": history[-1]["epsilon"] if history else config.epsilon_start,
        },
    )

    scheduler.set_seed(args.eval_seed)
    eval_pool = scheduler.ScenarioPool(
        size=args.eval_episodes,
        selected_system_num=config.selected_system_num,
        min_system_num=config.min_system_num,
        max_system_num=config.max_system_num,
        cost_limit=config.cost_limit,
        shared_mission=False,
    )
    results = scheduler.evaluate_dqn(
        agent,
        eval_pool,
        episodes=args.eval_episodes,
        collect_schedule=True,
    )
    result_rows = [
        {key: value for key, value in row.items() if key != "schedule"}
        for row in results
    ]
    schedules = [item for row in results for item in row.get("schedule", [])]
    save_csv(args.output_dir / "train_history.csv", history)
    save_csv(args.output_dir / "eval_results.csv", result_rows)
    save_csv(args.output_dir / "eval_schedule.csv", schedules)
    save_json(args.output_dir / "config.json", asdict(config))
    print_json({"checkpoint": str(checkpoint_path), "evaluation": result_rows})


def handle_train_architecture(args) -> None:
    config = hrl_config(args)
    scheduler_agent, _ = DQNAgent.load_checkpoint(
        args.scheduler_checkpoint,
        device=config.device,
        load_optimizer=False,
    )
    validate_scheduler(scheduler_agent)
    pool = adaptive_pool(config)
    architecture_agent, history = hierarchical.train_architecture(
        config,
        scheduler_agent,
        pool,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture_agent.save_checkpoint(
        args.output_dir / "architecture.pt",
        training_state={"stage": "train-architecture", "episodes": len(history)},
    )
    save_combined_checkpoint(
        args.output_dir / "hrl.pt",
        architecture_agent,
        scheduler_agent,
        training_state={"stage": "train-architecture", "episodes": len(history)},
    )
    save_csv(args.output_dir / "architecture_history.csv", history)
    save_json(args.output_dir / "hrl_config.json", asdict(config))
    print_json(history[-1] if history else {})


def handle_finetune(args) -> None:
    config = hrl_config(args)
    architecture_agent, _ = ArchitectureDQNAgent.load_checkpoint(
        args.architecture_checkpoint,
        device=config.device,
    )
    scheduler_agent, _ = DQNAgent.load_checkpoint(
        args.scheduler_checkpoint,
        device=config.device,
    )
    validate_scheduler(scheduler_agent)
    history = hierarchical.finetune(
        config,
        architecture_agent,
        scheduler_agent,
        adaptive_pool(config),
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
    save_combined_checkpoint(
        args.output_dir / "hrl.pt",
        architecture_agent,
        scheduler_agent,
        training_state={"stage": "finetune", "episodes": len(history)},
    )
    save_csv(args.output_dir / "finetune_history.csv", history)
    save_json(args.output_dir / "hrl_config.json", asdict(config))
    print_json(history[-1] if history else {})


def handle_evaluate(args) -> None:
    device = resolve_device(args.device)
    architecture_agent, scheduler_agent, checkpoint = load_combined_checkpoint(
        args.checkpoint,
        device=device,
        load_optimizer=False,
    )
    validate_scheduler(scheduler_agent)
    config = architecture_agent.config
    scenarios = list(
        adaptive_pool(
            config,
            seed=args.eval_seed,
            size=args.eval_episodes,
        ).scenarios
    )
    hrl_rows = hierarchical.evaluate_hrl(
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
        hierarchical.evaluate_static_scheduler(
            scheduler_agent,
            scenarios,
            label="static_initial",
            budget=config.budget,
        )
    )
    rows.extend(
        hierarchical.evaluate_static_scheduler(
            scheduler_agent,
            scenarios,
            label="full_system_upper_bound",
            full_systems=True,
            budget=config.budget,
        )
    )
    rows.extend(
        hierarchical.evaluate_architecture_baseline(
            hierarchical.FixedArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            label="fixed_architecture_rules",
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
    )
    if args.intdqn_checkpoint is not None:
        flat_agent, _ = IntDQNAgent.load_checkpoint(
            args.intdqn_checkpoint,
            device=device,
            load_optimizer=False,
        )
        rows.extend(
            hierarchical.evaluate_flat_intdqn(
                flat_agent,
                scenarios,
                budget=config.budget,
            )
        )
    scheduler.set_seed(args.eval_seed)
    rows.extend(
        hierarchical.evaluate_architecture_baseline(
            hierarchical.RandomArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            label="random_architecture_rules",
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
    )
    summary = evaluation.summarize_hrl(rows)
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
            "architecture_actions": list(ArchitectureRule.RULE_NAMES),
            "scheduler_actions": list(Rule.RULE_NAMES),
        },
    )
    print_json(summary)


def handle_train_flat(args) -> None:
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 100000
    config = IntDQNConfig(
        episodes=args.episodes,
        fixed_mission=args.fixed_mission,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=resolve_device(args.device),
        log_interval=args.log_interval,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent, history = flat.train_intdqn(config)
    checkpoint_path = agent.save_checkpoint(
        args.output_dir / "flat.pt",
        training_state={
            "stage": "train-flat",
            "episodes": len(history),
            "final_epsilon": (
                history[-1]["epsilon"] if history else config.epsilon_start
            ),
        },
    )
    results = flat.evaluate_intdqn(
        agent,
        episodes=args.eval_episodes,
        eval_seed=eval_seed,
        collect_schedule=True,
    )
    result_rows = [
        {key: value for key, value in row.items() if key != "schedule"}
        for row in results
    ]
    schedules = [item for row in results for item in row.get("schedule", [])]
    save_csv(args.output_dir / "train_history.csv", history)
    save_csv(args.output_dir / "eval_results.csv", result_rows)
    save_csv(args.output_dir / "eval_schedule.csv", schedules)
    save_json(
        args.output_dir / "config.json",
        {
            "training": asdict(config),
            "evaluation": {"episodes": args.eval_episodes, "seed": eval_seed},
            "environment": domain.CONFIG,
        },
    )
    print_json({"checkpoint": str(checkpoint_path), "evaluation": result_rows})


def handle_compare_schedulers(args) -> None:
    model_specs = args.models
    if model_specs is None:
        model_specs = [parse_model_spec(value) for value in DEFAULT_SCHEDULER_MODELS]
    device = resolve_device(args.device)
    agents, metadata = evaluation.load_scheduler_agents(model_specs, device)
    rows_by_model, summary, comparisons, manifest = evaluation.compare_schedulers(
        agents,
        args.eval_episodes,
        args.eval_seed,
    )
    all_rows = [row for rows in rows_by_model.values() for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "results.csv", all_rows)
    save_csv(args.output_dir / "summary.csv", summary)
    save_csv(args.output_dir / "paired_comparisons.csv", comparisons)
    save_json(args.output_dir / "scenarios.json", manifest)
    save_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "shared_test_pool": True,
            "independent_from_training": True,
            "device": device,
            "checkpoints": metadata,
        },
    )
    print_json({"summary": summary, "paired_comparisons": comparisons})


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "eval_episodes", 1) <= 0:
        parser.error("--eval-episodes must be positive")
    args.handler(args)
