"""Rule-DQN training in the exact Round-1 fixed-provider environment."""

from __future__ import annotations

from dataclasses import asdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Sequence

import numpy as np
import torch

from ..gp.artifact import sha256_file
from ..gp.evolution import episode_objective
from ..gp.provider import FixedArchitectureProvider
from ..rl.agent import DQNAgent
from ..rl.config import DQNConfig
from ..rules import scheduling as scheduling_rules
from .branching import branching_episode_row
from .branching_gp_finetune import StratifiedManifestSampler
from .gp_architecture import load_scenario_manifest
from .round1_study import _episode_outcome, _scenario_environment
from .scheduler import set_seed
from .scheduler_backends import run_rule_dqn_episode, scheduler_parameter_hash


FIXED_RULE_RUN_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_STEPS = (
    0,
    20000,
    40000,
    60000,
    80000,
    120000,
    160000,
    200000,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _canonical_steps(values: Sequence[int]) -> tuple[int, ...]:
    steps = tuple(sorted({int(value) for value in values}))
    if not steps or steps[0] != 0 or any(value < 0 for value in steps):
        raise ValueError("checkpoint steps must be non-negative and include zero.")
    return steps


def _checkpoint_label(steps: int) -> str:
    return f"{steps // 1000}k" if steps and steps % 1000 == 0 else str(steps)


def _save_checkpoint_atomic(
    agent: DQNAgent,
    path: Path,
    training_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    agent.save_checkpoint(temporary, training_state=training_state)
    temporary.replace(path)


def _save_resume_state(
    path: Path,
    *,
    checkpoint: Path,
    agent: DQNAgent,
    total_steps: int,
    episode: int,
    epsilon: float,
    history: Sequence[dict[str, Any]],
) -> None:
    payload = {
        "schema_version": FIXED_RULE_RUN_SCHEMA_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "total_steps": int(total_steps),
        "episode": int(episode),
        "epsilon": float(epsilon),
        "history": list(history),
        "replay_buffer": list(agent.replay.buffer),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_resume_state(
    path: Path,
    *,
    config: DQNConfig,
    sampler: StratifiedManifestSampler,
) -> tuple[DQNAgent, int, int, float, list[dict[str, Any]], Path]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != FIXED_RULE_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported fixed rule-DQN resume schema.")
    checkpoint = Path(state["checkpoint"]).resolve()
    if sha256_file(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("fixed rule-DQN resume checkpoint hash changed.")
    agent, _ = DQNAgent.load_checkpoint(
        checkpoint, device=config.device, load_optimizer=True
    )
    if asdict(agent.config) != asdict(config):
        raise ValueError("fixed rule-DQN resume configuration changed.")
    agent.replay.buffer.extend(state["replay_buffer"])
    episode = int(state["episode"])
    sampler.advance(episode)
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    cuda_state = state.get("cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return (
        agent,
        int(state["total_steps"]),
        episode,
        float(state["epsilon"]),
        list(state["history"]),
        checkpoint,
    )


def _rule_episode_row(
    episode: int,
    payload: dict[str, Any],
    mission_env,
    result: dict[str, Any],
    epsilon: float,
    *,
    total_steps: int | None = None,
) -> dict[str, Any]:
    row = branching_episode_row(
        episode,
        payload["category"],
        mission_env,
        result,
        epsilon,
        total_env_steps=total_steps,
    )
    for index, name in enumerate(scheduling_rules.Rule.RULE_NAMES):
        row[f"rule_{name.lower()}_count"] = int(
            result["scheduler_rule_counts"][index]
        )
    row["scenario_hash"] = payload["scenario_hash"]
    row["failure_aware_j"] = episode_objective(
        _episode_outcome(mission_env, result)
    )
    return row


def evaluate_fixed_rule_scheduler(
    *,
    scheduler_checkpoint: str | Path,
    scenarios: Sequence[dict[str, Any]],
    device: str,
    model: str,
) -> list[dict[str, Any]]:
    """Evaluate a frozen rule DQN under the Round-1 fixed provider."""

    agent, _ = DQNAgent.load_checkpoint(
        scheduler_checkpoint, device=device, load_optimizer=False
    )
    if str(agent.config.rule_set) != "standard":
        raise ValueError("fixed rule-DQN workflow requires the standard JSP rule set.")
    before_hash = scheduler_parameter_hash(agent)
    replay_size = len(agent.replay)
    provider = FixedArchitectureProvider()
    rows = []
    with torch.no_grad():
        for episode, payload in enumerate(scenarios):
            mission_env = _scenario_environment(payload, "fixed")
            result = run_rule_dqn_episode(
                mission_env,
                provider,
                agent,
                scheduler_epsilon=0.0,
                update_scheduler=False,
                store_experience=False,
            )
            row = _rule_episode_row(
                episode, payload, mission_env, result, 0.0
            )
            row["model"] = str(model)
            row["provider"] = "fixed"
            rows.append(row)
    if len(agent.replay) != replay_size:
        raise RuntimeError("fixed rule-DQN evaluation modified replay.")
    if scheduler_parameter_hash(agent) != before_hash:
        raise RuntimeError("fixed rule-DQN evaluation modified parameters.")
    if any(
        int(row["architecture_changes"]) != 0
        or int(row["provider_invariant_violations"]) != 0
        or int(row["invalid_action_count"]) != 0
        for row in rows
    ):
        raise RuntimeError("fixed rule-DQN evaluation violated an invariant.")
    return rows


def _summarize_checkpoint(step: int, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if bool(row["success"])]
    return {
        "target_environment_steps": int(step),
        "episodes": len(rows),
        "success_count": len(successful),
        "failure_rate": 1.0 - len(successful) / max(len(rows), 1),
        "mean_j": float(np.mean([float(row["failure_aware_j"]) for row in rows])),
        "mean_success_makespan": (
            float(np.mean([float(row["makespan"]) for row in successful]))
            if successful
            else float("inf")
        ),
        "invalid_action_count": sum(int(row["invalid_action_count"]) for row in rows),
        "provider_invariant_violations": sum(
            int(row["provider_invariant_violations"]) for row in rows
        ),
        "architecture_changes": sum(int(row["architecture_changes"]) for row in rows),
    }


def train_fixed_rule_scheduler(
    *,
    output_dir: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    config: DQNConfig,
    max_env_steps: int = 200000,
    checkpoint_steps: Sequence[int] = DEFAULT_CHECKPOINT_STEPS,
    stop_after_checkpoint: int | None = None,
) -> dict[str, Path]:
    """Train, resume, validate, and select a rule DQN in the fixed environment."""

    if str(config.rule_set) != "standard":
        raise ValueError("fixed rule-DQN training requires rule_set='standard'.")
    steps = _canonical_steps(checkpoint_steps)
    if int(max_env_steps) != int(steps[-1]):
        raise ValueError("max_env_steps must equal the final checkpoint step.")

    destination = Path(output_dir).resolve()
    manifest_path = destination / "run_manifest.json"
    resuming = manifest_path.is_file()
    if not resuming:
        destination.mkdir(parents=True, exist_ok=False)
    train = load_scenario_manifest(train_manifest)
    validation = load_scenario_manifest(validation_manifest)
    if str(train.get("split", "")).startswith("test") or str(
        validation.get("split", "")
    ).startswith("test"):
        raise ValueError("test manifests are forbidden during scheduler selection.")
    for payload in (*train["scenarios"], *validation["scenarios"]):
        _scenario_environment(payload, "fixed")

    inputs = {
        "train_manifest": {
            "path": str(Path(train_manifest).resolve()),
            "sha256": sha256_file(train_manifest),
            "manifest_hash": train["manifest_hash"],
        },
        "validation_manifest": {
            "path": str(Path(validation_manifest).resolve()),
            "sha256": sha256_file(validation_manifest),
            "manifest_hash": validation["manifest_hash"],
        },
        "provider_sha256": "static-feasible-keep-v1",
    }
    expected_manifest = {
        "schema_version": FIXED_RULE_RUN_SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "code_commit": _git_commit(),
        "provider": "fixed",
        "scheduler": "rule-dqn",
        "seed": int(config.seed),
        "config": asdict(config),
        "max_env_steps": int(max_env_steps),
        "checkpoint_steps": list(steps),
        "inputs": inputs,
    }
    if resuming:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = (
            "schema_version",
            "provider",
            "scheduler",
            "seed",
            "config",
            "max_env_steps",
            "checkpoint_steps",
            "inputs",
        )
        changed = [
            key for key in immutable if manifest.get(key) != expected_manifest.get(key)
        ]
        if changed:
            raise ValueError(f"fixed rule-DQN run inputs changed: {changed}")
        if manifest.get("status") == "complete":
            selected = Path(manifest["selection"]["checkpoint"])
            if sha256_file(selected) != manifest["selection"]["checkpoint_sha256"]:
                raise ValueError("completed fixed rule-DQN selection changed.")
            return {
                "manifest": manifest_path,
                "selected_checkpoint": selected,
                "selection": destination / "validation" / "selection.json",
            }
        manifest["resumed_at"] = _utc_now()
    else:
        manifest = expected_manifest
        _write_json(manifest_path, manifest)

    set_seed(int(config.seed))
    sampler = StratifiedManifestSampler(train["scenarios"], seed=int(config.seed))
    checkpoints_dir = destination / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[int, Path] = {}
    for path in checkpoints_dir.glob("checkpoint_*.pt"):
        label = path.stem.removeprefix("checkpoint_")
        checkpoint_paths[
            int(label[:-1]) * 1000 if label.endswith("k") else int(label)
        ] = path
    resume_path = destination / "resume_state.pt"

    if resuming:
        if not resume_path.is_file():
            raise FileNotFoundError("incomplete fixed rule-DQN run has no resume state.")
        agent, total_steps, episode, epsilon, history, resumed_checkpoint = (
            _restore_resume_state(
                resume_path, config=config, sampler=sampler
            )
        )
        if resumed_checkpoint not in checkpoint_paths.values():
            raise ValueError("resume checkpoint is outside the registered run.")
    else:
        first_env = _scenario_environment(train["scenarios"][0], "fixed")
        agent = DQNAgent(
            int(first_env.schedule_observation().shape[0]),
            scheduling_rules.Rule.RULE_NUM,
            config,
        )
        manifest["initial_parameter_sha256"] = scheduler_parameter_hash(agent)
        _write_json(manifest_path, manifest)
        zero_path = checkpoints_dir / "checkpoint_0.pt"
        _save_checkpoint_atomic(
            agent,
            zero_path,
            {
                "stage": "fixed_rule_dqn",
                "target_environment_steps": 0,
                "actual_environment_steps": 0,
                "seed": int(config.seed),
                "provider_sha256": "static-feasible-keep-v1",
                "train_manifest_hash": train["manifest_hash"],
            },
        )
        checkpoint_paths[0] = zero_path
        total_steps = 0
        episode = 0
        epsilon = float(config.epsilon_start)
        history: list[dict[str, Any]] = []
        _save_resume_state(
            resume_path,
            checkpoint=zero_path,
            agent=agent,
            total_steps=total_steps,
            episode=episode,
            epsilon=epsilon,
            history=history,
        )

    # Older partial runs did not persist the learning-rate trace.  Backfill it
    # deterministically so resume and uninterrupted histories have one schema.
    for row in history:
        row_episode = int(row["episode"])
        row.setdefault(
            "learning_rate", config.learning_rate_at_episode(row_episode)
        )
        row.setdefault(
            "next_learning_rate",
            config.learning_rate_at_episode(row_episode + 1),
        )

    provider = FixedArchitectureProvider()
    while total_steps < steps[-1]:
        payload = sampler.next_payload()
        mission_env = _scenario_environment(payload, "fixed")
        used_epsilon = float(epsilon)
        used_learning_rate = config.learning_rate_at_episode(episode)
        for parameter_group in agent.optimizer.param_groups:
            parameter_group["lr"] = used_learning_rate
        result = run_rule_dqn_episode(
            mission_env,
            provider,
            agent,
            scheduler_epsilon=used_epsilon,
            update_scheduler=True,
            store_experience=True,
        )
        total_steps += int(result["assignment_steps"])
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row = _rule_episode_row(
            episode,
            payload,
            mission_env,
            result,
            used_epsilon,
            total_steps=total_steps,
        )
        row.update(
            {
                "provider": "fixed",
                "seed": int(config.seed),
                "next_epsilon": float(epsilon),
                "learning_rate": float(used_learning_rate),
                "next_learning_rate": float(
                    config.learning_rate_at_episode(episode + 1)
                ),
                "replay_size": len(agent.replay),
            }
        )
        history.append(row)
        episode += 1

        for threshold in steps[1:]:
            if total_steps < threshold or threshold in checkpoint_paths:
                continue
            path = checkpoints_dir / f"checkpoint_{_checkpoint_label(threshold)}.pt"
            _save_checkpoint_atomic(
                agent,
                path,
                {
                    "stage": "fixed_rule_dqn",
                    "target_environment_steps": int(threshold),
                    "actual_environment_steps": int(total_steps),
                    "episodes": int(episode),
                    "epsilon": float(epsilon),
                    "learning_rate": float(used_learning_rate),
                    "next_learning_rate": float(
                        config.learning_rate_at_episode(episode)
                    ),
                    "seed": int(config.seed),
                    "provider_sha256": "static-feasible-keep-v1",
                    "train_manifest_hash": train["manifest_hash"],
                },
            )
            checkpoint_paths[int(threshold)] = path
            _write_csv(destination / "training_history.csv", history)
            _save_resume_state(
                resume_path,
                checkpoint=path,
                agent=agent,
                total_steps=total_steps,
                episode=episode,
                epsilon=epsilon,
                history=history,
            )
            manifest.update(
                {
                    "last_checkpoint_step": int(threshold),
                    "actual_environment_steps": int(total_steps),
                    "episodes": int(episode),
                }
            )
            _write_json(manifest_path, manifest)
            if stop_after_checkpoint is not None and threshold >= int(
                stop_after_checkpoint
            ):
                return {
                    "manifest": manifest_path,
                    "resume_state": resume_path,
                    "training_history": destination / "training_history.csv",
                }

    _write_csv(destination / "training_history.csv", history)
    if set(checkpoint_paths) != set(steps):
        raise RuntimeError("not all fixed rule-DQN checkpoints were produced.")
    if any(
        int(row["architecture_changes"]) != 0
        or int(row["provider_invariant_violations"]) != 0
        or int(row["invalid_action_count"]) != 0
        for row in history
    ):
        raise RuntimeError("fixed rule-DQN training violated an invariant.")

    validation_rows: list[dict[str, Any]] = []
    summaries = []
    for threshold in steps:
        rows = evaluate_fixed_rule_scheduler(
            scheduler_checkpoint=checkpoint_paths[threshold],
            scenarios=validation["scenarios"],
            device=config.device,
            model=f"fixed_rule_seed{int(config.seed)}_{_checkpoint_label(threshold)}",
        )
        for row in rows:
            row["target_environment_steps"] = int(threshold)
            row["seed"] = int(config.seed)
        validation_rows.extend(rows)
        summaries.append(_summarize_checkpoint(threshold, rows))

    valid_summaries = [
        row
        for row in summaries
        if int(row["invalid_action_count"]) == 0
        and int(row["provider_invariant_violations"]) == 0
        and int(row["architecture_changes"]) == 0
    ]
    if not valid_summaries:
        raise RuntimeError("no valid fixed rule-DQN checkpoint is selectable.")
    selected_summary = min(
        valid_summaries,
        key=lambda row: (
            float(row["failure_rate"]),
            float(row["mean_success_makespan"]),
            int(row["target_environment_steps"]),
        ),
    )
    selected_step = int(selected_summary["target_environment_steps"])
    selected_checkpoint = checkpoint_paths[selected_step].resolve()
    selection = {
        "selection_rule": (
            "minimize failure_rate, then mean_success_makespan, then "
            "target_environment_steps; mean_j is reporting-only"
        ),
        "selected_step": selected_step,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": sha256_file(selected_checkpoint),
        "parameter_sha256": scheduler_parameter_hash(
            DQNAgent.load_checkpoint(
                selected_checkpoint, device="cpu", load_optimizer=False
            )[0]
        ),
        "metrics": selected_summary,
    }
    validation_dir = destination / "validation"
    _write_csv(validation_dir / "checkpoint_results.csv", validation_rows)
    _write_csv(validation_dir / "checkpoint_summary.csv", summaries)
    _write_json(validation_dir / "selection.json", selection)
    manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "actual_environment_steps": int(total_steps),
            "episodes": int(episode),
            "selection": selection,
            "checkpoints": {
                str(step): {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for step, path in sorted(checkpoint_paths.items())
            },
        }
    )
    _write_json(manifest_path, manifest)
    resume_path.unlink(missing_ok=True)
    return {
        "manifest": manifest_path,
        "selected_checkpoint": selected_checkpoint,
        "selection": validation_dir / "selection.json",
        "training_history": destination / "training_history.csv",
        "validation_results": validation_dir / "checkpoint_results.csv",
        "validation_summary": validation_dir / "checkpoint_summary.csv",
    }
