"""Unified command-line interface for training and evaluating SOSRL."""

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from . import domain
from .baselines import flat, flat_rules
from .rl.agent import (
    ArchitectureDQNAgent,
    DQNAgent,
    FlatRuleDQNAgent,
    IntDQNAgent,
)
from .rl.branching import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_FEATURE_DIM,
    GLOBAL_FEATURE_NAMES,
    SYSTEM_FEATURE_DIM,
    SYSTEM_FEATURE_NAMES,
    TASK_FEATURE_DIM,
    TASK_FEATURE_NAMES,
    BranchingDQNAgent,
)
from .rl.checkpoint import load_combined_checkpoint, save_combined_checkpoint
from .rl.config import BranchingDQNConfig, DQNConfig, HRLConfig, IntDQNConfig
from .gp.config import GPArchitectureConfig
from .rules.architecture import ArchitectureRule
from .rules.scheduling import Rule
from .workflows import branching, evaluation, hierarchical, scheduler


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


def add_flat_rule_arguments(parser) -> None:
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--scenario-pool-size", type=int, default=100)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--refund-rate", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
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
        default=Path("runs") / "flat_rules",
    )
    add_device(parser)


def add_branching_arguments(parser) -> None:
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--scenario-pool-size", type=int, default=100)
    parser.add_argument("--max-env-steps", type=int, default=240000)
    parser.add_argument("--budget", type=float, default=8000.0)
    parser.add_argument("--refund-rate", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "branching_scheduler",
    )
    add_device(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sosrl",
        description="SOSRL architecture and scheduling training/evaluation workflows.",
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
    evaluate.add_argument(
        "--flat-rule-model",
        action="append",
        type=parse_model_spec,
        dest="flat_rule_models",
    )
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

    train_flat_rules = subparsers.add_parser("train-flat-rules")
    add_flat_rule_arguments(train_flat_rules)
    train_flat_rules.set_defaults(handler=handle_train_flat_rules)

    train_branching = subparsers.add_parser("train-branching-scheduler")
    add_branching_arguments(train_branching)
    train_branching.add_argument(
        "--architecture-checkpoint",
        type=Path,
        required=True,
    )
    train_branching.set_defaults(handler=handle_train_branching_scheduler)

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

    generate_gp = subparsers.add_parser("generate-gp-scenarios")
    generate_gp.add_argument("--base-seed", type=int, default=20260820)
    generate_gp.add_argument("--train-size", type=int, default=256)
    generate_gp.add_argument("--validation-size", type=int, default=128)
    generate_gp.add_argument("--test-size", type=int, default=500)
    generate_gp.add_argument("--ood-size", type=int, default=200)
    generate_gp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "gp_scenarios",
    )
    generate_gp.set_defaults(handler=handle_generate_gp_scenarios)

    train_gp = subparsers.add_parser("train-gp-architecture")
    train_gp.add_argument("--scheduler-checkpoint", type=Path, required=True)
    train_gp.add_argument("--scenario-dir", type=Path, required=True)
    train_gp.add_argument(
        "--feature-set",
        choices=["system", "system_demand", "system_delta", "op_context"],
        default="system_delta",
    )
    train_gp.add_argument("--population-size", type=int, default=200)
    train_gp.add_argument("--generations", type=int, default=80)
    train_gp.add_argument("--runs", type=int, default=10)
    train_gp.add_argument("--train-batch-size", type=int, default=16)
    train_gp.add_argument("--anchor-size", type=int, default=64)
    train_gp.add_argument("--anchor-interval", type=int, default=10)
    train_gp.add_argument("--anchor-top-k", type=int, default=10)
    train_gp.add_argument("--workers", type=int, default=1)
    train_gp.add_argument("--base-seed", type=int, default=20260820)
    train_gp.add_argument("--resume-state", type=Path)
    train_gp.add_argument("--device", default="cpu")
    train_gp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "gp_architecture",
    )
    train_gp.set_defaults(handler=handle_train_gp_architecture)

    evaluate_gp = subparsers.add_parser("evaluate-gp-stack")
    evaluate_gp.add_argument("--gp-policy", type=Path, required=True)
    evaluate_gp.add_argument("--scheduler-checkpoint", type=Path, required=True)
    evaluate_gp.add_argument("--scenario-manifest", type=Path, required=True)
    evaluate_gp.add_argument("--manual-architecture-checkpoint", type=Path)
    evaluate_gp.add_argument(
        "--baselines",
        nargs="+",
        choices=["fixed", "random_concrete", "manual6_dqn", "gp"],
        default=["fixed", "random_concrete", "gp"],
    )
    evaluate_gp.add_argument("--collect-schedule", action="store_true")
    evaluate_gp.add_argument("--device", default="cpu")
    evaluate_gp.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "gp_architecture" / "evaluation",
    )
    evaluate_gp.set_defaults(handler=handle_evaluate_gp_stack)

    finetune_branching_gp = subparsers.add_parser(
        "finetune-branching-with-gp"
    )
    finetune_branching_gp.add_argument(
        "--scheduler-checkpoint", type=Path, required=True
    )
    finetune_branching_gp.add_argument("--gp-policy", type=Path, required=True)
    finetune_branching_gp.add_argument("--scenario-dir", type=Path, required=True)
    finetune_branching_gp.add_argument("--output-dir", type=Path, required=True)
    finetune_branching_gp.add_argument("--extra-env-steps", type=int, default=40000)
    finetune_branching_gp.add_argument(
        "--checkpoint-interval-steps", type=int, default=10000
    )
    finetune_branching_gp.add_argument("--lr", type=float, default=1e-5)
    finetune_branching_gp.add_argument("--epsilon-start", type=float, default=0.10)
    finetune_branching_gp.add_argument("--epsilon-end", type=float, default=0.02)
    finetune_branching_gp.add_argument(
        "--epsilon-decay", type=float, default=0.995
    )
    finetune_branching_gp.add_argument("--seed", type=int, default=4)
    finetune_branching_gp.add_argument("--device", default="auto")
    finetune_branching_gp.add_argument("--resume", action="store_true")
    finetune_branching_gp.add_argument(
        "--skip-historical-test", action="store_true"
    )
    finetune_branching_gp.set_defaults(
        handler=handle_finetune_branching_with_gp
    )
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


def flat_rule_config(args) -> HRLConfig:
    return HRLConfig(
        episodes=args.episodes,
        scenario_pool_size=args.scenario_pool_size,
        budget=args.budget,
        refund_rate=args.refund_rate,
        gamma=args.gamma,
        n_step=args.n_step,
        architecture_lr=args.lr,
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


def branching_config(args) -> BranchingDQNConfig:
    return BranchingDQNConfig(
        episodes=args.episodes,
        max_env_steps=args.max_env_steps,
        scenario_pool_size=args.scenario_pool_size,
        budget=args.budget,
        refund_rate=args.refund_rate,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        device=resolve_device(args.device),
        log_interval=args.log_interval,
    )


def validate_scheduler(agent) -> None:
    if agent.obs_dim != 25:
        raise ValueError("scheduler checkpoint must use the 25-dimensional state.")
    if agent.action_dim != Rule.RULE_NUM:
        raise ValueError("scheduler checkpoint must contain four standard rules.")


def validate_flat_rule_model(agent, reference_config) -> None:
    if agent.action_dim != flat_rules.JOINT_ACTION_SIZE:
        raise ValueError("flat-rule checkpoint must contain 24 joint actions.")
    if not math.isclose(agent.config.budget, reference_config.budget):
        raise ValueError("flat-rule and HRL checkpoints must use the same budget.")
    if not math.isclose(agent.config.refund_rate, reference_config.refund_rate):
        raise ValueError(
            "flat-rule and HRL checkpoints must use the same refund rate."
        )


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
    is_branching = isinstance(scheduler_agent, BranchingDQNAgent)
    config = scheduler_agent.config if is_branching else architecture_agent.config
    scenarios = list(
        adaptive_pool(
            config,
            seed=args.eval_seed,
            size=args.eval_episodes,
        ).scenarios
    )
    if is_branching:
        reference_label = "architecture_branching"
        primary_rows = branching.evaluate_branching(
            architecture_agent,
            scheduler_agent,
            scenarios,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        for row in primary_rows:
            row["model"] = reference_label
        rows = list(primary_rows)
        fixed_rows = branching.evaluate_branching(
            hierarchical.FixedArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        for row in fixed_rows:
            row["model"] = "fixed_architecture_rules"
        rows.extend(fixed_rows)
    else:
        validate_scheduler(scheduler_agent)
        reference_label = "hrl"
        primary_rows = hierarchical.evaluate_hrl(
            architecture_agent,
            scheduler_agent,
            scenarios,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        for row in primary_rows:
            row["model"] = reference_label
        rows = list(primary_rows)
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
    flat_rule_metadata = {}
    flat_rule_labels = []
    reserved_labels = {
        "hrl",
        "architecture_branching",
        "static_initial",
        "full_system_upper_bound",
        "fixed_architecture_rules",
        "random_architecture_rules",
        "flat_intdqn",
    }
    for label, path in args.flat_rule_models or []:
        if label in reserved_labels or label in flat_rule_metadata:
            raise ValueError(f"duplicate or reserved flat-rule model label: {label}")
        flat_rule_agent, flat_rule_checkpoint = FlatRuleDQNAgent.load_checkpoint(
            path,
            device=device,
            load_optimizer=False,
        )
        validate_flat_rule_model(flat_rule_agent, config)
        rows.extend(
            flat_rules.evaluate_flat_rules(
                flat_rule_agent,
                scenarios,
                label=label,
                budget=config.budget,
                refund_rate=config.refund_rate,
            )
        )
        flat_rule_labels.append(label)
        flat_rule_metadata[label] = {
            "path": str(path.resolve()),
            "training_state": flat_rule_checkpoint.get("training_state", {}),
            "config": flat_rule_checkpoint["config"],
        }
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
    if is_branching:
        random_rows = branching.evaluate_branching(
            hierarchical.RandomArchitectureRulePolicy(),
            scheduler_agent,
            scenarios,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        for row in random_rows:
            row["model"] = "random_architecture_rules"
        rows.extend(random_rows)
    else:
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
    paired_comparisons = (
        evaluation.paired_adaptive_comparisons(
            rows,
            reference_label=reference_label,
            candidate_labels=flat_rule_labels,
        )
        if flat_rule_labels
        else []
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "results.csv", rows)
    save_csv(args.output_dir / "summary.csv", summary)
    if paired_comparisons:
        save_csv(args.output_dir / "paired_comparisons.csv", paired_comparisons)
    manifest = {
        "eval_seed": args.eval_seed,
        "eval_episodes": args.eval_episodes,
        "paired_scenarios": True,
        "unseen_missions": True,
        "checkpoint": str(args.checkpoint.resolve()),
        "training_state": checkpoint.get("training_state", {}),
        "architecture_actions": list(ArchitectureRule.RULE_NAMES),
        "scheduler_kind": getattr(scheduler_agent, "checkpoint_kind", "scheduler"),
        "scheduler_actions": (
            ["frontier_task", "active_system"]
            if is_branching
            else list(Rule.RULE_NAMES)
        ),
    }
    if flat_rule_metadata:
        manifest["flat_rule_models"] = flat_rule_metadata
        manifest["joint_action_encoding"] = (
            "joint_action = architecture_action * 4 + scheduling_action"
        )
    save_json(
        args.output_dir / "evaluation_manifest.json",
        manifest,
    )
    print_json(summary)


def handle_train_branching_scheduler(args) -> None:
    if args.max_env_steps is not None and args.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive")
    config = branching_config(args)
    architecture_agent, architecture_checkpoint = (
        ArchitectureDQNAgent.load_checkpoint(
            args.architecture_checkpoint,
            device=config.device,
            load_optimizer=False,
        )
    )
    pool = adaptive_pool(config)
    branching_agent, history, training_progress = (
        branching.train_branching_scheduler(
            config,
            architecture_agent,
            pool,
        )
    )
    scenario_metadata = [
        {
            "category": category,
            "scenario_hash": hierarchical.scenario_hash(architecture, mission),
        }
        for architecture, mission, category in pool.scenarios
    ]
    provider_metadata = {
        "checkpoint": str(args.architecture_checkpoint.resolve()),
        "checkpoint_kind": architecture_checkpoint.get(
            "checkpoint_kind",
            "architecture",
        ),
        "training_state": architecture_checkpoint.get("training_state", {}),
        "frozen": True,
        "epsilon": 0.0,
    }
    training_state = {
        "stage": "train-branching-scheduler",
        "algorithm": "constrained_additive_bdqn",
        "episodes": len(history),
        "target_environment_steps": config.max_env_steps,
        "actual_environment_steps": training_progress["total_env_steps"],
        "final_epsilon": training_progress["epsilon"],
        "architecture_provider": provider_metadata,
        "training_scenarios": scenario_metadata,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_path = branching_agent.save_checkpoint(
        args.output_dir / "branching_scheduler.pt",
        training_state=training_state,
    )
    combined_path = save_combined_checkpoint(
        args.output_dir / "architecture_branching.pt",
        architecture_agent,
        branching_agent,
        training_state=training_state,
        metadata={
            "architecture_provider": provider_metadata,
            "training_scenarios": scenario_metadata,
        },
    )
    save_csv(args.output_dir / "branching_history.csv", history)
    save_json(
        args.output_dir / "branching_config.json",
        {
            "training": asdict(config),
            "algorithm": "constrained additive BDQN without interaction terms",
            "action": "(task_idx, sys_idx); op_idx=task_op_indices[task_idx]",
            "feature_schema": {
                "version": FEATURE_SCHEMA_VERSION,
                "global": list(GLOBAL_FEATURE_NAMES),
                "task": list(TASK_FEATURE_NAMES),
                "system": list(SYSTEM_FEATURE_NAMES),
                "dimensions": {
                    "global": GLOBAL_FEATURE_DIM,
                    "task": TASK_FEATURE_DIM,
                    "system": SYSTEM_FEATURE_DIM,
                },
            },
            "architecture_provider": provider_metadata,
            "scenario_pool": scenario_metadata,
        },
    )
    print_json(
        {
            "branching_scheduler_checkpoint": str(scheduler_path),
            "combined_checkpoint": str(combined_path),
            "training_state": training_state,
        }
    )


def handle_train_flat_rules(args) -> None:
    if args.max_env_steps is not None and args.max_env_steps <= 0:
        raise ValueError("--max-env-steps must be positive")
    config = flat_rule_config(args)
    pool = adaptive_pool(config)
    agent, history = flat_rules.train_flat_rules(
        config,
        pool,
        max_env_steps=args.max_env_steps,
    )
    actual_steps = (
        int(history[-1]["cumulative_environment_steps"]) if history else 0
    )
    final_epsilon = history[-1]["epsilon"] if history else config.epsilon_start
    training_state = {
        "stage": "train-flat-rules",
        "algorithm": "flat_rule_dqn",
        "episodes": len(history),
        "target_environment_steps": args.max_env_steps,
        "actual_environment_steps": actual_steps,
        "final_epsilon": final_epsilon,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = agent.save_checkpoint(
        args.output_dir / "flat_rules.pt",
        training_state=training_state,
    )
    save_csv(args.output_dir / "flat_rules_history.csv", history)
    save_json(
        args.output_dir / "flat_rules_config.json",
        {
            "training": asdict(config),
            "interaction_budget": {
                "target_environment_steps": args.max_env_steps,
                "actual_environment_steps": actual_steps,
                "episode_boundary_stop": True,
            },
            "observation_dim": agent.obs_dim,
            "action_dim": agent.action_dim,
            "joint_action_encoding": (
                "joint_action = architecture_action * 4 + scheduling_action"
            ),
        },
    )
    print_json(
        {
            "checkpoint": str(checkpoint_path),
            "training_state": training_state,
        }
    )


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


def handle_generate_gp_scenarios(args) -> None:
    from .workflows import gp_architecture

    paths = gp_architecture.generate_gp_scenario_manifests(
        args.output_dir,
        base_seed=args.base_seed,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        ood_size=args.ood_size,
    )
    print_json({name: str(path.resolve()) for name, path in paths.items()})


def handle_train_gp_architecture(args) -> None:
    from .workflows import gp_architecture

    config = GPArchitectureConfig(
        population_size=args.population_size,
        generations=args.generations,
        independent_runs=args.runs,
        train_batch_size=args.train_batch_size,
        anchor_size=args.anchor_size,
        anchor_interval=args.anchor_interval,
        anchor_top_k=args.anchor_top_k,
        workers=args.workers,
        base_seed=args.base_seed,
        feature_set=args.feature_set,
    )
    outputs = gp_architecture.train_gp_architecture(
        scheduler_checkpoint=args.scheduler_checkpoint,
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        config=config,
        device=resolve_device(args.device),
        resume_state=args.resume_state,
    )
    print_json({name: str(path.resolve()) for name, path in outputs.items()})


def handle_evaluate_gp_stack(args) -> None:
    from .workflows import gp_architecture

    outputs = gp_architecture.evaluate_gp_stack(
        gp_policy=args.gp_policy,
        scheduler_checkpoint=args.scheduler_checkpoint,
        scenario_manifest=args.scenario_manifest,
        output_dir=args.output_dir,
        baselines=args.baselines,
        manual_architecture_checkpoint=args.manual_architecture_checkpoint,
        device=resolve_device(args.device),
        collect_schedule=args.collect_schedule,
    )
    print_json({name: str(path.resolve()) for name, path in outputs.items()})


def handle_finetune_branching_with_gp(args) -> None:
    from .workflows.branching_gp_finetune import (
        finetune_branching_with_frozen_gp,
    )

    outputs = finetune_branching_with_frozen_gp(
        scheduler_checkpoint=args.scheduler_checkpoint,
        gp_policy=args.gp_policy,
        scenario_dir=args.scenario_dir,
        output_dir=args.output_dir,
        extra_env_steps=args.extra_env_steps,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        lr=args.lr,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        skip_historical_test=args.skip_historical_test,
    )
    print_json({name: str(path.resolve()) for name, path in outputs.items()})


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "eval_episodes", 1) <= 0:
        parser.error("--eval-episodes must be positive")
    args.handler(args)
