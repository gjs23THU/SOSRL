"""Round-one controlled convergence experiments for the GP+BDQN stack.

The workflow treats the architecture provider as an experimental factor.  A
cell trains one identically initialised Branching DQN against either a feasible
static architecture, the frozen manual-rule Architecture DQN, or the frozen G0
policy.  Scenario manifests, provider inputs and checkpoints are hash-bound so
that cells can be compared without silently changing anything except the
registered factor.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .. import environment as env
from ..gp.artifact import load_gp_policy, sha256_file
from ..gp.evolution import EpisodeOutcome, episode_objective
from ..gp.provider import (
    FixedArchitectureProvider,
    GPArchitectureProvider,
    ManualRuleDQNProvider,
)
from ..rl.agent import ArchitectureDQNAgent
from ..rl.branching import BranchingDQNAgent
from ..rl.checkpoint import load_branching_checkpoint, save_branching_checkpoint
from ..rl.config import BranchingDQNConfig, default_device
from . import evaluation
from .branching import branching_episode_row, run_branching_episode
from .branching_gp_finetune import (
    SCENARIO_CATEGORIES,
    StratifiedManifestSampler,
    paired_bootstrap_ci,
    summarize_results,
)
from .gp_architecture import (
    _generate_split,
    load_scenario_manifest,
    save_scenario_manifest,
)
from .scheduler import ScenarioPool, set_seed


ROUND1_STUDY_SCHEMA_VERSION = 1
PROVIDER_KINDS = ("fixed", "arch", "g0")
DEFAULT_CONVERGENCE_STEPS = (0, 20000, 40000, 60000, 80000, 120000, 160000, 200000)
DEFAULT_TRANSFER_STEPS = (0, 10000, 20000, 30000, 40000, 60000, 80000)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_git_commit() -> str:
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


def _branching_parameter_hash(agent: BranchingDQNAgent) -> str:
    digest = hashlib.sha256()
    for network_name, network in (("q", agent.q_net), ("target", agent.target_net)):
        for name, tensor in sorted(network.state_dict().items()):
            digest.update(network_name.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(str(int(agent.learn_step)).encode("utf-8"))
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def _save_branching_checkpoint_atomic(
    agent: BranchingDQNAgent,
    path: Path,
    training_state: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_branching_checkpoint(agent, temporary, training_state)
    temporary.replace(path)


def _checkpoint_label(steps: int) -> str:
    return f"{steps // 1000}k" if steps and steps % 1000 == 0 else str(steps)


def _canonical_steps(values: Iterable[int]) -> tuple[int, ...]:
    steps = tuple(sorted({int(value) for value in values}))
    if not steps or steps[0] != 0 or any(value < 0 for value in steps):
        raise ValueError("checkpoint steps must be non-negative and include zero.")
    return steps


def _validate_static_payload(payload: dict[str, Any]) -> None:
    architecture, mission = evaluation.static_scenario_from_payload(payload)
    indices = [int(system.index) for system in architecture]
    if len(indices) != len(set(indices)):
        raise ValueError("static feasible architecture contains duplicate systems.")
    budget = float(payload.get("budget", 8000.0))
    if sum(float(system.cost) for system in architecture) > budget + 1e-9:
        raise ValueError("static feasible architecture exceeds the scenario budget.")
    sampler = ScenarioPool(size=0, cost_limit=budget)
    if not sampler.arch_can_cover_mission(architecture, mission):
        raise ValueError("registered static architecture cannot cover the mission.")


def _assert_disjoint_manifests(manifests: dict[str, dict[str, Any]]) -> None:
    owner: dict[str, str] = {}
    for split, manifest in manifests.items():
        for scenario in manifest["scenarios"]:
            scenario_hash = str(scenario["scenario_hash"])
            if scenario_hash in owner:
                raise ValueError(
                    f"scenario hash overlap between {owner[scenario_hash]!r} and {split!r}."
                )
            owner[scenario_hash] = split


def generate_round1_scenarios(
    output_dir: str | Path,
    *,
    base_seed: int = 20260824,
    b_train_size: int = 256,
    b_validation_size: int = 256,
    g_train_size: int = 256,
    g_validation_size: int = 256,
    test_iid_size: int = 1000,
    test_ood_size: int = 500,
) -> dict[str, Path]:
    """Generate six disjoint schema-v2 manifests used by the round-one study."""

    destination = Path(output_dir).resolve()
    specifications = (
        ("b_train", b_train_size, base_seed, False, destination / "b" / "train.json"),
        (
            "b_validation",
            b_validation_size,
            base_seed + 1,
            False,
            destination / "b" / "validation.json",
        ),
        ("g_train", g_train_size, base_seed + 2, False, destination / "g" / "train.json"),
        (
            "g_validation",
            g_validation_size,
            base_seed + 3,
            False,
            destination / "g" / "validation.json",
        ),
        (
            "test_iid_v2",
            test_iid_size,
            base_seed + 4,
            False,
            destination / "test_iid_v2.json",
        ),
        (
            "test_ood_v2",
            test_ood_size,
            base_seed + 5,
            True,
            destination / "test_ood_v2.json",
        ),
    )
    paths: dict[str, Path] = {}
    loaded: dict[str, dict[str, Any]] = {}
    for split, size, seed, ood, path in specifications:
        scenarios = _generate_split(
            split=split,
            size=int(size),
            seed=int(seed),
            ood=bool(ood),
        )
        for scenario in scenarios:
            _validate_static_payload(scenario)
        paths[split] = save_scenario_manifest(
            path,
            split=split,
            seed=int(seed),
            scenarios=scenarios,
        )
        loaded[split] = load_scenario_manifest(path)
    _assert_disjoint_manifests(loaded)
    registry = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "base_seed": int(base_seed),
        "created_at": _utc_now(),
        "test_locked": True,
        "manifests": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "manifest_hash": loaded[name]["manifest_hash"],
                "size": int(loaded[name]["size"]),
            }
            for name, path in paths.items()
        },
    }
    _write_json(destination / "scenario_registry.json", registry)
    paths["registry"] = destination / "scenario_registry.json"
    return paths


def baseline_bdqn_config(
    *,
    seed: int,
    max_env_steps: int = 200000,
    device: str = "auto",
) -> BranchingDQNConfig:
    resolved = default_device() if device == "auto" else str(device)
    return BranchingDQNConfig(
        episodes=max(100000, int(max_env_steps)),
        max_env_steps=int(max_env_steps),
        scenario_pool_size=256,
        budget=8000.0,
        refund_rate=0.8,
        gamma=0.99,
        lr=1e-4,
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_interval=250,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=int(seed),
        device=resolved,
        log_interval=10,
    )


def finetune_bdqn_config(
    *,
    seed: int,
    max_env_steps: int = 80000,
    device: str = "auto",
    lr: float = 1e-5,
    batch_size: int = 64,
    buffer_size: int = 50000,
    target_update_interval: int = 250,
    epsilon_start: float = 0.10,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.995,
) -> BranchingDQNConfig:
    config = baseline_bdqn_config(
        seed=seed,
        max_env_steps=max_env_steps,
        device=device,
    )
    return replace(
        config,
        lr=float(lr),
        batch_size=int(batch_size),
        buffer_size=int(buffer_size),
        target_update_interval=int(target_update_interval),
        epsilon_start=float(epsilon_start),
        epsilon_end=float(epsilon_end),
        epsilon_decay=float(epsilon_decay),
    )


def ensure_initial_checkpoint(
    output_dir: str | Path,
    *,
    seed: int,
    config: BranchingDQNConfig,
) -> Path:
    """Create the single initial weight artifact shared by all providers for a seed."""

    path = Path(output_dir).resolve() / "initial_weights" / f"seed_{int(seed)}.pt"
    if path.is_file():
        loaded, checkpoint = load_branching_checkpoint(
            path, device=config.device, load_optimizer=False
        )
        if int(checkpoint.get("training_state", {}).get("seed", -1)) != int(seed):
            raise ValueError("initial checkpoint seed mismatch.")
        if asdict(loaded.config) != asdict(config):
            raise ValueError("initial checkpoint configuration mismatch.")
        return path
    set_seed(int(seed))
    agent = BranchingDQNAgent(config)
    _save_branching_checkpoint_atomic(
        agent,
        path,
        {
            "stage": "round1_shared_initial_weights",
            "seed": int(seed),
            "created_at": _utc_now(),
        },
    )
    return path


def _provider_input_hash(
    provider_kind: str,
    *,
    architecture_checkpoint: str | Path | None,
    gp_policy: str | Path | None,
) -> str:
    if provider_kind == "fixed":
        return "static-feasible-keep-v1"
    path = architecture_checkpoint if provider_kind == "arch" else gp_policy
    if path is None:
        raise ValueError(f"{provider_kind} provider input is required.")
    return sha256_file(Path(path).resolve())


def load_round1_provider(
    provider_kind: str,
    *,
    architecture_checkpoint: str | Path | None,
    gp_policy: str | Path | None,
    device: str,
):
    if provider_kind not in PROVIDER_KINDS:
        raise ValueError(f"unknown provider kind: {provider_kind!r}")
    if provider_kind == "fixed":
        return FixedArchitectureProvider()
    if provider_kind == "arch":
        if architecture_checkpoint is None:
            raise ValueError("architecture checkpoint is required for provider 'arch'.")
        agent, _ = ArchitectureDQNAgent.load_checkpoint(
            architecture_checkpoint,
            device=device,
            load_optimizer=False,
        )
        return ManualRuleDQNProvider(agent)
    if gp_policy is None:
        raise ValueError("GP policy is required for provider 'g0'.")
    return GPArchitectureProvider.from_artifact(load_gp_policy(gp_policy))


def _scenario_environment(payload: dict[str, Any], provider_kind: str) -> env.MissionEnv:
    if provider_kind == "fixed":
        architecture, mission = evaluation.static_scenario_from_payload(payload)
    else:
        architecture, mission = evaluation.scenario_from_payload(payload)
    return env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=float(payload.get("budget", 8000.0)),
        refund_rate=float(payload.get("refund_rate", 0.8)),
    )


def _episode_outcome(
    mission_env: env.MissionEnv,
    result: dict[str, Any],
) -> EpisodeOutcome:
    return EpisodeOutcome(
        success=bool(result["success"]),
        completed_operations=int(np.sum(mission_env.state.task_op_idx)),
        total_operations=int(mission_env.T * mission_env.O),
        makespan=float(mission_env.state.current_makespan),
        scale=float(mission_env.state.M),
        final_net_cost=float(mission_env.net_cost),
        peak_net_cost=float(mission_env.peak_net_cost),
        budget=float(mission_env.budget),
        architecture_changes=int(mission_env.architecture_change_count),
        dead_end=bool(result["dead_end"]),
    )


def evaluate_bdqn_provider_cell(
    *,
    model: str,
    scheduler_checkpoint: str | Path,
    provider_kind: str,
    scenarios: Sequence[dict[str, Any]],
    architecture_checkpoint: str | Path | None,
    gp_policy: str | Path | None,
    device: str,
) -> list[dict[str, Any]]:
    agent, _ = load_branching_checkpoint(
        scheduler_checkpoint,
        device=device,
        load_optimizer=False,
    )
    agent.q_net.eval()
    rows: list[dict[str, Any]] = []
    replay_size = len(agent.replay)
    parameter_hash = _branching_parameter_hash(agent)
    provider = load_round1_provider(
        provider_kind,
        architecture_checkpoint=architecture_checkpoint,
        gp_policy=gp_policy,
        device=device,
    )
    with torch.no_grad():
        for episode, payload in enumerate(scenarios):
            mission_env = _scenario_environment(payload, provider_kind)
            result = run_branching_episode(
                mission_env,
                provider,
                agent,
                scheduler_epsilon=0.0,
                update_scheduler=False,
                store_experience=False,
            )
            row = branching_episode_row(
                episode,
                payload.get("category", "evaluation"),
                mission_env,
                result,
                0.0,
            )
            row.update(
                {
                    "model": str(model),
                    "provider": str(provider_kind),
                    "scenario_hash": payload["scenario_hash"],
                    "failure_aware_j": episode_objective(
                        _episode_outcome(mission_env, result)
                    ),
                }
            )
            rows.append(row)
    if len(agent.replay) != replay_size:
        raise RuntimeError("evaluation modified the scheduler replay buffer.")
    if _branching_parameter_hash(agent) != parameter_hash:
        raise RuntimeError("evaluation modified the scheduler parameters.")
    if provider_kind == "fixed" and any(
        int(row["architecture_changes"]) != 0 for row in rows
    ):
        raise RuntimeError("Fixed evaluation violated the KEEP-only invariant.")
    return rows


def _agent_from_source(
    *,
    source_checkpoint: str | Path,
    config: BranchingDQNConfig,
) -> BranchingDQNAgent:
    source, _ = load_branching_checkpoint(
        source_checkpoint,
        device=config.device,
        load_optimizer=False,
    )
    agent = BranchingDQNAgent(config)
    agent.q_net.load_state_dict(source.q_net.state_dict())
    agent.target_net.load_state_dict(source.target_net.state_dict())
    agent.learn_step = int(source.learn_step)
    return agent


def _save_round1_resume_state(
    path: Path,
    *,
    checkpoint: Path,
    agent: BranchingDQNAgent,
    total_steps: int,
    episode: int,
    epsilon: float,
    history: Sequence[dict[str, Any]],
) -> None:
    """Atomically persist all mutable state needed for an exact cell resume."""

    payload = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
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


def _restore_round1_resume_state(
    path: Path,
    *,
    config: BranchingDQNConfig,
    sampler: StratifiedManifestSampler,
) -> tuple[BranchingDQNAgent, int, int, float, list[dict[str, Any]], Path]:
    """Restore an interrupted cell and validate its checkpoint binding."""

    state = torch.load(path, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != ROUND1_STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported round-one resume state schema.")
    checkpoint = Path(state["checkpoint"]).resolve()
    if sha256_file(checkpoint) != state["checkpoint_sha256"]:
        raise ValueError("round-one resume checkpoint hash changed.")
    agent, _ = load_branching_checkpoint(
        checkpoint,
        device=config.device,
        load_optimizer=True,
    )
    if asdict(agent.config) != asdict(config):
        raise ValueError("round-one resume configuration changed.")
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


def train_bdqn_provider_cell(
    *,
    output_dir: str | Path,
    provider_kind: str,
    source_checkpoint: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    config: BranchingDQNConfig,
    checkpoint_steps: Sequence[int],
    architecture_checkpoint: str | Path | None = None,
    gp_policy: str | Path | None = None,
    stop_after_checkpoint: int | None = None,
) -> dict[str, Path]:
    """Train and validate one controlled provider/seed/checkpoint cell."""

    steps = _canonical_steps(checkpoint_steps)
    if int(config.max_env_steps or 0) != int(steps[-1]):
        raise ValueError("config.max_env_steps must equal the final checkpoint step.")
    destination = Path(output_dir).resolve()
    manifest_path = destination / "cell_manifest.json"
    resuming = manifest_path.is_file()
    already_complete = False
    if resuming:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            already_complete = True
    else:
        destination.mkdir(parents=True, exist_ok=False)
    train = load_scenario_manifest(train_manifest)
    validation = load_scenario_manifest(validation_manifest)
    if str(train.get("split", "")).startswith("test") or str(
        validation.get("split", "")
    ).startswith("test"):
        raise ValueError("Test-v2 manifests are forbidden in BDQN training or selection cells.")
    input_hash = _provider_input_hash(
        provider_kind,
        architecture_checkpoint=architecture_checkpoint,
        gp_policy=gp_policy,
    )
    cell_manifest = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "provider": provider_kind,
        "seed": int(config.seed),
        "config": asdict(config),
        "checkpoint_steps": list(steps),
        "inputs": {
            "source_checkpoint": {
                "path": str(Path(source_checkpoint).resolve()),
                "sha256": sha256_file(Path(source_checkpoint).resolve()),
            },
            "provider_sha256": input_hash,
            "train_manifest": {
                "path": str(Path(train_manifest).resolve()),
                "sha256": sha256_file(Path(train_manifest).resolve()),
                "manifest_hash": train["manifest_hash"],
            },
            "validation_manifest": {
                "path": str(Path(validation_manifest).resolve()),
                "sha256": sha256_file(Path(validation_manifest).resolve()),
                "manifest_hash": validation["manifest_hash"],
            },
        },
    }
    if resuming:
        immutable_fields = ("provider", "seed", "config", "checkpoint_steps", "inputs")
        changed = [
            name for name in immutable_fields if existing.get(name) != cell_manifest.get(name)
        ]
        if changed:
            raise ValueError(f"round-one cell inputs changed: {changed}")
        cell_manifest = existing
        if already_complete:
            for step, record in cell_manifest.get("checkpoints", {}).items():
                if sha256_file(record["path"]) != record["sha256"]:
                    raise ValueError(f"completed round-one checkpoint changed: {step}")
            return {
                "manifest": manifest_path,
                "summary": destination / "validation" / "checkpoint_summary.csv",
                "results": destination / "validation" / "checkpoint_results.csv",
                "convergence": destination / "validation" / "convergence.json",
            }
        cell_manifest["resumed_at"] = _utc_now()
    else:
        _write_json(manifest_path, cell_manifest)

    set_seed(int(config.seed))
    source_hash = sha256_file(Path(source_checkpoint).resolve())
    provider = load_round1_provider(
        provider_kind,
        architecture_checkpoint=architecture_checkpoint,
        gp_policy=gp_policy,
        device=config.device,
    )
    sampler = StratifiedManifestSampler(train["scenarios"], seed=int(config.seed))
    checkpoints_dir = destination / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {
        int(path.stem.removeprefix("checkpoint_").removesuffix("k"))
        * (1000 if path.stem.endswith("k") else 1): path
        for path in checkpoints_dir.glob("checkpoint_*.pt")
    }
    resume_path = destination / "resume_state.pt"
    if resuming:
        if not resume_path.is_file():
            raise FileNotFoundError(f"missing resume state for incomplete cell: {resume_path}")
        agent, total_steps, episode, epsilon, history, resumed_checkpoint = (
            _restore_round1_resume_state(
                resume_path,
                config=config,
                sampler=sampler,
            )
        )
        if resumed_checkpoint not in checkpoint_paths.values():
            raise ValueError("resume state points outside the registered cell checkpoints.")
    else:
        agent = _agent_from_source(source_checkpoint=source_checkpoint, config=config)
        zero_path = checkpoints_dir / "checkpoint_0.pt"
        _save_branching_checkpoint_atomic(
            agent,
            zero_path,
            {
                "stage": "round1_bdqn_provider_cell",
                "provider": provider_kind,
                "seed": int(config.seed),
                "target_environment_steps": 0,
                "actual_environment_steps": 0,
                "source_checkpoint_sha256": source_hash,
                "provider_sha256": input_hash,
            },
        )
        checkpoint_paths[0] = zero_path
        total_steps = 0
        episode = 0
        epsilon = float(config.epsilon_start)
        history = []
        _save_round1_resume_state(
            resume_path,
            checkpoint=zero_path,
            agent=agent,
            total_steps=total_steps,
            episode=episode,
            epsilon=epsilon,
            history=history,
        )
    while total_steps < steps[-1]:
        payload = sampler.next_payload()
        mission_env = _scenario_environment(payload, provider_kind)
        used_epsilon = float(epsilon)
        result = run_branching_episode(
            mission_env,
            provider,
            agent,
            scheduler_epsilon=used_epsilon,
            update_scheduler=True,
            store_experience=True,
        )
        total_steps += int(result["assignment_steps"])
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row = branching_episode_row(
            episode,
            payload["category"],
            mission_env,
            result,
            used_epsilon,
            total_env_steps=total_steps,
        )
        row.update(
            {
                "provider": provider_kind,
                "seed": int(config.seed),
                "scenario_hash": payload["scenario_hash"],
                "next_epsilon": float(epsilon),
                "replay_size": len(agent.replay),
            }
        )
        history.append(row)
        episode += 1
        for threshold in steps[1:]:
            if total_steps < threshold or threshold in checkpoint_paths:
                continue
            path = checkpoints_dir / f"checkpoint_{_checkpoint_label(threshold)}.pt"
            _save_branching_checkpoint_atomic(
                agent,
                path,
                {
                    "stage": "round1_bdqn_provider_cell",
                    "provider": provider_kind,
                    "seed": int(config.seed),
                    "target_environment_steps": int(threshold),
                    "actual_environment_steps": int(total_steps),
                    "episodes": int(episode),
                    "epsilon": float(epsilon),
                    "source_checkpoint_sha256": source_hash,
                    "provider_sha256": input_hash,
                    "train_manifest_hash": train["manifest_hash"],
                },
            )
            checkpoint_paths[int(threshold)] = path
            _write_csv(destination / "training_history.csv", history)
            _save_round1_resume_state(
                resume_path,
                checkpoint=path,
                agent=agent,
                total_steps=total_steps,
                episode=episode,
                epsilon=epsilon,
                history=history,
            )
            if stop_after_checkpoint is not None and threshold >= int(stop_after_checkpoint):
                cell_manifest.update(
                    {
                        "status": "running",
                        "last_checkpoint_step": int(threshold),
                        "actual_environment_steps": int(total_steps),
                        "episodes": int(episode),
                    }
                )
                _write_json(manifest_path, cell_manifest)
                return {
                    "manifest": manifest_path,
                    "resume_state": resume_path,
                    "training_history": destination / "training_history.csv",
                }
    _write_csv(destination / "training_history.csv", history)
    if set(checkpoint_paths) != set(steps):
        raise RuntimeError("not all requested checkpoints were produced.")
    if provider_kind == "fixed" and any(
        int(row["architecture_changes"]) != 0
        or int(row["provider_invariant_violations"]) != 0
        for row in history
    ):
        raise RuntimeError("Fixed training violated the KEEP-only architecture invariant.")
    if _provider_input_hash(
        provider_kind,
        architecture_checkpoint=architecture_checkpoint,
        gp_policy=gp_policy,
    ) != input_hash:
        raise RuntimeError("frozen provider input changed during training.")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    for threshold in steps:
        model = f"{provider_kind}_{int(config.seed)}_{_checkpoint_label(threshold)}"
        rows = evaluate_bdqn_provider_cell(
            model=model,
            scheduler_checkpoint=checkpoint_paths[threshold],
            provider_kind=provider_kind,
            scenarios=validation["scenarios"],
            architecture_checkpoint=architecture_checkpoint,
            gp_policy=gp_policy,
            device=config.device,
        )
        for row in rows:
            row.update(
                {
                    "target_environment_steps": int(threshold),
                    "seed": int(config.seed),
                }
            )
        rows_by_step[threshold] = rows
        all_rows.extend(rows)
        for category in ("all", *SCENARIO_CATEGORIES):
            summary = summarize_results(
                model,
                rows,
                category=category,
                additional_steps=int(threshold),
            )
            summary.update(
                {
                    "provider": provider_kind,
                    "seed": int(config.seed),
                    "target_environment_steps": int(threshold),
                }
            )
            summaries.append(summary)
    validation_dir = destination / "validation"
    _write_csv(validation_dir / "checkpoint_results.csv", all_rows)
    _write_csv(validation_dir / "checkpoint_summary.csv", summaries)
    convergence = detect_bdqn_convergence(
        steps=steps,
        summaries=summaries,
        rows_by_step=rows_by_step,
        seed=int(config.seed),
    )
    _write_json(validation_dir / "convergence.json", convergence)
    cell_manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "actual_environment_steps": int(total_steps),
            "episodes": int(episode),
            "convergence": convergence,
            "checkpoints": {
                str(step): {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for step, path in checkpoint_paths.items()
            },
        }
    )
    _write_json(manifest_path, cell_manifest)
    resume_path.unlink(missing_ok=True)
    return {
        "manifest": manifest_path,
        "summary": validation_dir / "checkpoint_summary.csv",
        "results": validation_dir / "checkpoint_results.csv",
        "convergence": validation_dir / "convergence.json",
    }


def _paired_values(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    field: str,
) -> list[float]:
    base = {row["scenario_hash"]: row for row in baseline}
    cand = {row["scenario_hash"]: row for row in candidate}
    if set(base) != set(cand):
        raise ValueError("paired convergence rows use different scenario hashes.")
    return [float(cand[key][field]) - float(base[key][field]) for key in sorted(base)]


def detect_bdqn_convergence(
    *,
    steps: Sequence[int],
    summaries: Sequence[dict[str, Any]],
    rows_by_step: dict[int, list[dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    """Apply the preregistered three-checkpoint empirical convergence rule."""

    overall = {
        int(row["target_environment_steps"]): row
        for row in summaries
        if row["category"] == "all"
    }
    comparisons: list[dict[str, Any]] = []
    for previous, current in zip(steps[:-1], steps[1:], strict=True):
        before = overall[int(previous)]
        after = overall[int(current)]
        delta_j = _paired_values(
            rows_by_step[int(previous)], rows_by_step[int(current)], "failure_aware_j"
        )
        ci_low, ci_high = paired_bootstrap_ci(delta_j, seed=seed + int(current))
        j_improvement = (
            float(before["mean_j"]) - float(after["mean_j"])
        ) / max(abs(float(before["mean_j"])), 1e-12)
        makespan_improvement = (
            float(before["mean_success_makespan"])
            - float(after["mean_success_makespan"])
        ) / max(abs(float(before["mean_success_makespan"])), 1e-12)
        comparisons.append(
            {
                "previous_steps": int(previous),
                "current_steps": int(current),
                "relative_j_improvement": float(j_improvement),
                "relative_makespan_improvement": float(makespan_improvement),
                "delta_j_ci95_low": float(ci_low),
                "delta_j_ci95_high": float(ci_high),
                "failure_change": float(after["failure_rate"])
                - float(before["failure_rate"]),
            }
        )
    converged_step: int | None = None
    for index in range(len(comparisons) - 1):
        pair = comparisons[index : index + 2]
        stable = all(
            abs(item["relative_j_improvement"]) < 0.01
            and item["delta_j_ci95_low"] <= 0.0 <= item["delta_j_ci95_high"]
            and abs(item["relative_makespan_improvement"]) < 0.01
            and item["failure_change"] <= 0.0
            for item in pair
        )
        candidate_step = int(pair[0]["previous_steps"])
        candidate_j = float(overall[candidate_step]["mean_j"])
        no_rebound = all(
            (
                candidate_j - float(overall[int(step)]["mean_j"])
            )
            / max(abs(candidate_j), 1e-12)
            <= 0.01
            for step in steps
            if int(step) > candidate_step
        )
        success_not_down = all(
            float(overall[int(step)]["failure_rate"])
            <= float(overall[candidate_step]["failure_rate"])
            for step in steps
            if int(step) > candidate_step
        )
        safe = all(
            int(overall[int(step)].get("invalid_action_count", 0)) == 0
            and int(overall[int(step)].get("provider_invariant_violations", 0)) == 0
            for step in steps
            if int(step) >= candidate_step
        )
        if stable and no_rebound and success_not_down and safe:
            converged_step = candidate_step
            break
    return {
        "status": "converged" if converged_step is not None else "not_observed",
        "converged_step": converged_step,
        "comparisons": comparisons,
    }


def provider_cross_matrix_jobs(
    *,
    seeds: Sequence[int],
    checkpoint_by_training_provider: dict[str, dict[int, str | Path]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        for training_provider in PROVIDER_KINDS:
            checkpoint = checkpoint_by_training_provider[training_provider][int(seed)]
            for evaluation_provider in PROVIDER_KINDS:
                jobs.append(
                    {
                        "seed": int(seed),
                        "training_provider": training_provider,
                        "evaluation_provider": evaluation_provider,
                        "scheduler_checkpoint": str(Path(checkpoint).resolve()),
                    }
                )
    return jobs


def gp_discovery_matrix() -> list[dict[str, int | str]]:
    """Return the deduplicated preregistered population/generation matrix."""

    configurations: dict[tuple[int, int], set[str]] = {}
    for population in (40, 80, 120, 200):
        configurations.setdefault((population, 50), set()).add("population_axis")
    for generations in (10, 25, 50, 80):
        configurations.setdefault((120, generations), set()).add("generation_axis")
    for population, generations in (
        (50, 120),
        (75, 80),
        (120, 50),
        (200, 30),
        (300, 20),
    ):
        configurations.setdefault((population, generations), set()).add(
            "equal_budget"
        )
    return [
        {
            "population_size": population,
            "generations": generations,
            "individual_evaluation_budget": population * generations,
            "families": "+".join(sorted(families)),
        }
        for (population, generations), families in sorted(configurations.items())
    ]


def bdqn_hyperparameter_matrix() -> list[dict[str, Any]]:
    """Return H0-H10 exactly as preregistered."""

    base = {
        "lr": 1e-5,
        "batch_size": 64,
        "buffer_size": 50000,
        "target_update_interval": 250,
        "epsilon_start": 0.10,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.995,
    }
    changes = {
        "H0": {},
        "H1": {"lr": 3e-6},
        "H2": {"lr": 3e-5},
        "H3": {"batch_size": 32},
        "H4": {"batch_size": 128},
        "H5": {"buffer_size": 25000},
        "H6": {"buffer_size": 100000},
        "H7": {"target_update_interval": 100},
        "H8": {"target_update_interval": 1000},
        "H9": {"epsilon_start": 0.05, "epsilon_end": 0.01, "epsilon_decay": 0.99},
        "H10": {
            "epsilon_start": 0.20,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.9975,
        },
    }
    return [{"name": name, **base, **change} for name, change in changes.items()]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def common_convergence_step(
    cell_directories: dict[str, dict[int, str | Path]],
) -> dict[str, Any]:
    """Choose max(t_F, t_A, t_G), falling back to the final registered step."""

    provider_steps: dict[str, int] = {}
    seed_details: dict[str, dict[str, int | None]] = {}
    for provider in PROVIDER_KINDS:
        observed: list[int] = []
        final_steps: list[int] = []
        seed_details[provider] = {}
        for seed, directory in sorted(cell_directories[provider].items()):
            root = Path(directory)
            convergence = json.loads(
                (root / "validation" / "convergence.json").read_text(encoding="utf-8")
            )
            cell = json.loads((root / "cell_manifest.json").read_text(encoding="utf-8"))
            value = convergence.get("converged_step")
            seed_details[provider][str(seed)] = None if value is None else int(value)
            if value is not None:
                observed.append(int(value))
            final_steps.append(max(int(step) for step in cell["checkpoints"]))
        if len(observed) == len(cell_directories[provider]):
            provider_steps[provider] = max(observed)
        else:
            provider_steps[provider] = max(final_steps)
    return {
        "provider_steps": provider_steps,
        "seed_convergence_steps": seed_details,
        "t0": max(provider_steps.values()),
        "fallback_rule": "final_checkpoint_when_any_seed_has_no_observed_convergence",
    }


def checkpoint_for_target_step(cell_directory: str | Path, target_step: int) -> Path:
    manifest = json.loads(
        (Path(cell_directory) / "cell_manifest.json").read_text(encoding="utf-8")
    )
    record = manifest.get("checkpoints", {}).get(str(int(target_step)))
    if record is None:
        raise KeyError(f"cell has no checkpoint for target step {target_step}.")
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("cell checkpoint hash mismatch.")
    return path


def evaluate_provider_cross_matrix(
    *,
    output_dir: str | Path,
    jobs: Sequence[dict[str, Any]],
    validation_manifest: str | Path,
    architecture_checkpoint: str | Path,
    gp_policy: str | Path,
    device: str,
) -> dict[str, Path]:
    """Evaluate the full train-provider by test-provider matrix."""

    validation = load_scenario_manifest(validation_manifest)
    if str(validation.get("split")) not in {"b_validation", "validation"}:
        raise ValueError("cross-matrix selection may only use a validation manifest.")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    job_signature = _json_sha256(
        [
            {
                "seed": int(job["seed"]),
                "training_provider": str(job["training_provider"]),
                "evaluation_provider": str(job["evaluation_provider"]),
                "scheduler_checkpoint_sha256": sha256_file(job["scheduler_checkpoint"]),
            }
            for job in jobs
        ]
    )
    results_path = destination / "cross_results.csv"
    summary_path = destination / "cross_summary.csv"
    manifest_path = destination / "cross_manifest.json"
    if results_path.is_file() and summary_path.is_file() and manifest_path.is_file():
        registered = json.loads(manifest_path.read_text(encoding="utf-8"))
        if registered["validation_manifest_sha256"] != sha256_file(validation_manifest):
            raise ValueError("completed cross evaluation validation manifest changed.")
        if int(registered["job_count"]) != len(jobs):
            raise ValueError("completed cross evaluation job matrix changed.")
        if registered.get("job_signature_sha256") != job_signature:
            raise ValueError("completed cross evaluation checkpoints changed.")
        return {
            "manifest": manifest_path,
            "results": results_path,
            "summary": summary_path,
        }
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for job in jobs:
        seed = int(job["seed"])
        training_provider = str(job["training_provider"])
        evaluation_provider = str(job["evaluation_provider"])
        model = f"train_{training_provider}__eval_{evaluation_provider}__seed_{seed}"
        rows = evaluate_bdqn_provider_cell(
            model=model,
            scheduler_checkpoint=job["scheduler_checkpoint"],
            provider_kind=evaluation_provider,
            scenarios=validation["scenarios"],
            architecture_checkpoint=architecture_checkpoint,
            gp_policy=gp_policy,
            device=device,
        )
        for row in rows:
            row.update(
                {
                    "seed": seed,
                    "training_provider": training_provider,
                    "evaluation_provider": evaluation_provider,
                }
            )
        all_rows.extend(rows)
        for category in ("all", *SCENARIO_CATEGORIES):
            summary = summarize_results(model, rows, category=category)
            summary.update(
                {
                    "seed": seed,
                    "training_provider": training_provider,
                    "evaluation_provider": evaluation_provider,
                }
            )
            summaries.append(summary)
    _write_csv(results_path, all_rows)
    _write_csv(summary_path, summaries)
    _write_json(
        manifest_path,
        {
            "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "validation_manifest": str(Path(validation_manifest).resolve()),
            "validation_manifest_sha256": sha256_file(validation_manifest),
            "job_count": len(jobs),
            "job_signature_sha256": job_signature,
            "result_rows": len(all_rows),
            "providers": list(PROVIDER_KINDS),
        },
    )
    return {
        "manifest": manifest_path,
        "results": results_path,
        "summary": summary_path,
    }


def migration_path_jobs(
    *,
    seeds: Sequence[int],
    t0_checkpoints: dict[str, dict[int, str | Path]],
) -> list[dict[str, Any]]:
    paths = (
        ("fixed", "fixed"),
        ("fixed", "g0"),
        ("arch", "arch"),
        ("arch", "g0"),
        ("g0", "g0"),
    )
    return [
        {
            "name": f"{source}_to_{target}",
            "seed": int(seed),
            "source_provider": source,
            "target_provider": target,
            "source_checkpoint": str(Path(t0_checkpoints[source][int(seed)]).resolve()),
        }
        for seed in seeds
        for source, target in paths
    ]


def _final_step_rows(cell_directory: str | Path, target_step: int) -> list[dict[str, str]]:
    rows = _read_csv(Path(cell_directory) / "validation" / "checkpoint_results.csv")
    return [
        row
        for row in rows
        if int(float(row["target_environment_steps"])) == int(target_step)
    ]


def _aggregate_candidate_rows(
    cell_directories: Sequence[str | Path],
    target_step: int,
) -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    for directory in cell_directories:
        cell = json.loads(
            (Path(directory) / "cell_manifest.json").read_text(encoding="utf-8")
        )
        for row in _final_step_rows(directory, target_step):
            row = dict(row)
            row["seed"] = str(cell["seed"])
            combined.append(row)
    return combined


def _aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate an empty candidate.")
    successful = [row for row in rows if str(row["success"]).lower() == "true"]

    def avg(field: str, source=rows) -> float:
        return float(np.mean([float(row[field]) for row in source]))

    return {
        "episodes": len(rows),
        "failure_rate": 1.0 - len(successful) / len(rows),
        "mean_j": avg("failure_aware_j"),
        "mean_success_makespan": (
            avg("makespan", successful) if successful else float("inf")
        ),
        "budget_violation_rate": sum(
            str(row["ever_over_budget"]).lower() == "true" for row in rows
        )
        / len(rows),
        "invalid_action_count": sum(int(float(row["invalid_action_count"])) for row in rows),
        "provider_invariant_violations": sum(
            int(float(row["provider_invariant_violations"])) for row in rows
        ),
    }


def _paired_delta_by_seed_scenario(
    baseline: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    field: str,
) -> list[float]:
    key = lambda row: (int(row["seed"]), str(row["scenario_hash"]))
    base = {key(row): row for row in baseline}
    cand = {key(row): row for row in candidate}
    if set(base) != set(cand):
        raise ValueError("hyperparameter candidates are not paired by seed and scenario.")
    return [float(cand[item][field]) - float(base[item][field]) for item in sorted(base)]


def classify_hyperparameter_candidate(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    baseline = _aggregate_metrics(baseline_rows)
    candidate = _aggregate_metrics(candidate_rows)
    delta_j = _paired_delta_by_seed_scenario(
        baseline_rows, candidate_rows, "failure_aware_j"
    )
    delta_budget = [
        float(str(candidate_row["ever_over_budget"]).lower() == "true")
        - float(str(baseline_row["ever_over_budget"]).lower() == "true")
        for baseline_row, candidate_row in zip(
            sorted(baseline_rows, key=lambda row: (int(row["seed"]), row["scenario_hash"])),
            sorted(candidate_rows, key=lambda row: (int(row["seed"]), row["scenario_hash"])),
            strict=True,
        )
    ]
    j_ci = paired_bootstrap_ci(delta_j, seed=seed)
    budget_ci = paired_bootstrap_ci(delta_budget, seed=seed + 1)
    j_improvement = (
        float(baseline["mean_j"]) - float(candidate["mean_j"])
    ) / max(abs(float(baseline["mean_j"])), 1e-12)
    makespan_change = (
        float(candidate["mean_success_makespan"])
        - float(baseline["mean_success_makespan"])
    ) / max(abs(float(baseline["mean_success_makespan"])), 1e-12)
    accepted = (
        float(candidate["failure_rate"]) <= float(baseline["failure_rate"])
        and j_improvement >= 0.01
        and float(j_ci[1]) < 0.0
        and makespan_change <= 0.01
        and float(budget_ci[1]) <= 0.02
        and int(candidate["invalid_action_count"]) == 0
        and int(candidate["provider_invariant_violations"]) == 0
    )
    stratified = _paired_cross_comparison(
        baseline_rows,
        candidate_rows,
        left_label="H0",
        right_label="candidate",
        contrast="hyperparameter_candidate_vs_h0",
        seed=seed + 2,
    )
    return {
        "accepted": bool(accepted),
        "baseline": baseline,
        "candidate": candidate,
        "relative_j_improvement": float(j_improvement),
        "relative_makespan_change": float(makespan_change),
        "delta_j_ci95": [float(j_ci[0]), float(j_ci[1])],
        "delta_budget_violation_ci95": [float(budget_ci[0]), float(budget_ci[1])],
        "stratified_comparison": stratified,
    }


def select_hyperparameter_screen(
    *,
    output_path: str | Path,
    cells_by_name: dict[str, Sequence[str | Path]],
    target_step: int = 40000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Select Hsingle and compose Hstar using the preregistered guards."""

    matrix = {row["name"]: row for row in bdqn_hyperparameter_matrix()}
    if set(matrix) - set(cells_by_name):
        raise ValueError("not all H0-H10 screening cells are available.")
    rows_by_name = {
        name: _aggregate_candidate_rows(cells_by_name[name], target_step)
        for name in matrix
    }
    baseline_rows = rows_by_name["H0"]
    diagnostics = {
        name: classify_hyperparameter_candidate(
            baseline_rows,
            rows,
            seed=seed + index * 10,
        )
        for index, (name, rows) in enumerate(rows_by_name.items())
        if name != "H0"
    }
    accepted = [name for name, result in diagnostics.items() if result["accepted"]]

    def rank(name: str) -> tuple[float, float, float, str]:
        metrics = _aggregate_metrics(rows_by_name[name])
        return (
            float(metrics["failure_rate"]),
            float(metrics["mean_j"]),
            float(metrics["mean_success_makespan"]),
            name,
        )

    hsingle = min(accepted, key=rank) if accepted else "H0"
    factor_groups = {
        "lr": ("H1", "H2"),
        "batch_size": ("H3", "H4"),
        "buffer_size": ("H5", "H6"),
        "target_update_interval": ("H7", "H8"),
        "epsilon_schedule": ("H9", "H10"),
    }
    selected_levels: dict[str, str] = {}
    hstar = dict(matrix["H0"])
    hstar["name"] = "Hstar"
    for factor, names in factor_groups.items():
        eligible = [name for name in names if name in accepted]
        if not eligible:
            continue
        winner = min(eligible, key=rank)
        selected_levels[factor] = winner
        for key, value in matrix[winner].items():
            if key != "name" and value != matrix["H0"].get(key):
                hstar[key] = value
    selection = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "target_step": int(target_step),
        "hsingle": hsingle,
        "hstar": hstar,
        "selected_levels": selected_levels,
        "diagnostics": diagnostics,
        "acceptance_guards": {
            "minimum_relative_j_improvement": 0.01,
            "maximum_relative_makespan_worsening": 0.01,
            "maximum_budget_violation_ci95_high": 0.02,
        },
    }
    _write_json(Path(output_path), selection)
    return selection


def run_gp_matrix_job(
    *,
    output_dir: str | Path,
    scheduler_checkpoint: str | Path,
    scenario_dir: str | Path,
    population_size: int,
    generations: int,
    independent_runs: int,
    base_seed: int,
    workers: int = 8,
    train_batch_size: int = 16,
    anchor_size: int = 64,
) -> dict[str, Path]:
    """Run one hash-bound GP matrix cell, or return its completed artifacts."""

    from ..gp.config import GPArchitectureConfig
    from .gp_architecture import train_gp_architecture

    destination = Path(output_dir).resolve()
    policy = destination / "gp_policy.json"
    manifest = destination / "gp_stack_manifest.json"
    if policy.is_file() and manifest.is_file():
        loaded = load_gp_policy(policy)
        expected = {
            "population_size": int(population_size),
            "generations": int(generations),
            "independent_runs": int(independent_runs),
            "base_seed": int(base_seed),
            "feature_set": "system_delta",
            "train_batch_size": int(train_batch_size),
            "anchor_size": int(anchor_size),
        }
        changed = {
            key: (loaded.artifact.evolution_config.get(key), value)
            for key, value in expected.items()
            if loaded.artifact.evolution_config.get(key) != value
        }
        if changed:
            raise ValueError(f"completed GP cell configuration changed: {changed}")
        if loaded.artifact.bdqn_checkpoint_sha256 != sha256_file(scheduler_checkpoint):
            raise ValueError("completed GP cell scheduler checkpoint changed.")
        stack = json.loads(manifest.read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            current = load_scenario_manifest(Path(scenario_dir) / f"{split}.json")
            if stack["scenario_manifest_hashes"][split] != current["manifest_hash"]:
                raise ValueError(f"completed GP cell {split} scenarios changed.")
        return {
            "policy": policy,
            "manifest": manifest,
            "generation_history": destination / "generation_history.csv",
            "anchor_history": destination / "anchor_history.csv",
            "candidate_rules": destination / "candidate_rules.jsonl",
            "validation_results": destination / "validation_results.csv",
        }
    config = GPArchitectureConfig(
        population_size=int(population_size),
        generations=int(generations),
        independent_runs=int(independent_runs),
        train_batch_size=int(train_batch_size),
        anchor_size=int(anchor_size),
        anchor_interval=10,
        anchor_top_k=10,
        workers=int(workers),
        base_seed=int(base_seed),
        feature_set="system_delta",
    )
    return train_gp_architecture(
        scheduler_checkpoint=scheduler_checkpoint,
        scenario_dir=scenario_dir,
        output_dir=destination,
        config=config,
        device="cpu",
    )


def select_gp_discovery_configuration(
    *,
    output_path: str | Path,
    jobs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Select one executed GP configuration with the preregistered ordering."""

    candidates: list[dict[str, Any]] = []
    for job in jobs:
        loaded = load_gp_policy(Path(job["output_dir"]) / "gp_policy.json")
        fitness = loaded.artifact.validation_fitness
        candidates.append(
            {
                **dict(job),
                "failure_rate": float(fitness["failure_rate"]),
                "raw_mean_j": float(fitness["raw_mean_j"]),
                "regularized_j": float(fitness["regularized_j"]),
                "node_count": int(loaded.artifact.node_count),
                "height": int(loaded.artifact.height),
                "expression": loaded.artifact.expression,
                "policy_sha256": sha256_file(Path(job["output_dir"]) / "gp_policy.json"),
            }
        )
    minimum_failure = min(row["failure_rate"] for row in candidates)
    failure_best = [
        row for row in candidates if row["failure_rate"] == minimum_failure
    ]
    best_j = min(row["raw_mean_j"] for row in failure_best)
    eligible = [row for row in failure_best if row["raw_mean_j"] <= best_j * 1.01]
    winner = min(
        eligible,
        key=lambda row: (
            int(row["individual_evaluation_budget"]),
            int(row["node_count"]),
            int(row["height"]),
            str(row["expression"]),
        ),
    )
    selection = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "selection_order": [
            "failure_rate",
            "within_1pct_of_best_raw_mean_j",
            "individual_evaluation_budget",
            "node_count",
            "height",
            "expression",
        ],
        "winner": winner,
        "candidates": candidates,
    }
    _write_json(Path(output_path), selection)
    return selection


def _select_validation_winner(
    expressions: set[str],
    validation_by_expression: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = [validation_by_expression[expression] for expression in expressions]
    minimum_failure = min(float(row["failure_rate"]) for row in rows)
    failure_best = [
        row for row in rows if float(row["failure_rate"]) == minimum_failure
    ]
    best_j = min(float(row["raw_mean_j"]) for row in failure_best)
    eligible = [
        row for row in failure_best if float(row["raw_mean_j"]) <= best_j * 1.01
    ]
    return min(
        eligible,
        key=lambda row: (
            int(float(row["node_count"])),
            int(float(row["height"])),
            str(row["expression"]),
        ),
    )


def gp_run_count_convergence(
    *,
    confirmation_dir: str | Path,
    output_path: str | Path,
    bootstrap_samples: int = 10000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Estimate how many independent GP runs recover the ten-run optimum."""

    root = Path(confirmation_dir)
    candidate_rows = [
        json.loads(line)
        for line in (root / "candidate_rules.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_rows = _read_csv(root / "validation_results.csv")
    validation_by_expression = {
        str(row["expression"]): row for row in validation_rows
    }
    by_run: dict[int, set[str]] = {}
    for row in candidate_rows:
        expression = str(row["expression"])
        if expression not in validation_by_expression:
            continue
        by_run.setdefault(int(row["run_seed"]), set()).add(expression)
    run_seeds = sorted(by_run)
    if len(run_seeds) < 2:
        raise ValueError("run-count convergence requires at least two independent runs.")
    final = _select_validation_winner(
        set().union(*(by_run[run_seed] for run_seed in run_seeds)),
        validation_by_expression,
    )
    final_failure = float(final["failure_rate"])
    final_j = float(final["raw_mean_j"])
    rng = np.random.default_rng(int(seed))
    hits = np.zeros(len(run_seeds), dtype=np.int64)
    for _ in range(int(bootstrap_samples)):
        order = rng.permutation(run_seeds).tolist()
        expressions: set[str] = set()
        for index, run_seed in enumerate(order):
            expressions.update(by_run[int(run_seed)])
            winner = _select_validation_winner(expressions, validation_by_expression)
            hit = (
                float(winner["failure_rate"]) == final_failure
                and float(winner["raw_mean_j"]) <= final_j * 1.01
            )
            hits[index] += int(hit)
    probabilities = [float(value) / int(bootstrap_samples) for value in hits]
    recommended: int | None = None
    for index, probability in enumerate(probabilities):
        if probability >= 0.95 and all(value >= 0.95 for value in probabilities[index:]):
            recommended = index + 1
            break
    result = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "independent_runs": len(run_seeds),
        "bootstrap_samples": int(bootstrap_samples),
        "probability_within_1pct_by_run_count": [
            {"run_count": index + 1, "probability": probability}
            for index, probability in enumerate(probabilities)
        ],
        "recommended_run_count": recommended,
        "final_failure_rate": final_failure,
        "final_raw_mean_j": final_j,
        "final_expression": final["expression"],
    }
    _write_json(Path(output_path), result)
    return result


def gp_generation_convergence(
    *,
    anchor_history_path: str | Path,
) -> dict[str, Any]:
    """Find two consecutive anchor windows with <1% best-so-far improvement."""

    rows = _read_csv(anchor_history_path)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if int(float(row.get("anchor_rank", 1))) != 1:
            continue
        grouped.setdefault(int(float(row["run_seed"])), []).append(row)
    details: dict[str, Any] = {}
    observed: list[int] = []
    for run_seed, run_rows in sorted(grouped.items()):
        ordered = sorted(run_rows, key=lambda row: int(float(row["generation"])))
        best_failure = float("inf")
        best_j = float("inf")
        sequence: list[dict[str, float | int]] = []
        for row in ordered:
            failure = float(row["failure_rate"])
            raw_j = float(row["raw_mean_j"])
            if (failure, raw_j) < (best_failure, best_j):
                best_failure, best_j = failure, raw_j
            sequence.append(
                {
                    "generation": int(float(row["generation"])) + 1,
                    "best_failure_rate": best_failure,
                    "best_raw_mean_j": best_j,
                }
            )
        converged: int | None = None
        for index in range(len(sequence) - 2):
            first, second, third = sequence[index : index + 3]
            improvement_one = (
                float(first["best_raw_mean_j"]) - float(second["best_raw_mean_j"])
            ) / max(abs(float(first["best_raw_mean_j"])), 1e-12)
            improvement_two = (
                float(second["best_raw_mean_j"]) - float(third["best_raw_mean_j"])
            ) / max(abs(float(second["best_raw_mean_j"])), 1e-12)
            no_failure_gain = (
                first["best_failure_rate"]
                == second["best_failure_rate"]
                == third["best_failure_rate"]
            )
            if improvement_one < 0.01 and improvement_two < 0.01 and no_failure_gain:
                converged = int(first["generation"])
                break
        details[str(run_seed)] = {
            "converged_generation": converged,
            "anchor_sequence": sequence,
        }
        if converged is not None:
            observed.append(converged)
    return {
        "status": "converged_all_runs" if len(observed) == len(grouped) else "partial",
        "recommended_generation": max(observed) if len(observed) == len(grouped) else None,
        "runs": details,
    }


def select_final_bdqn_confirmation(
    *,
    output_path: str | Path,
    cells_by_route_and_config: dict[str, dict[str, Sequence[str | Path]]],
    target_step: int = 80000,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Lock one route/config and choose its validation-medoid seed checkpoint."""

    candidates: list[dict[str, Any]] = []
    rows_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    eligible_keys: set[tuple[str, str]] = set()
    diagnostics: dict[str, Any] = {}
    for route, by_config in sorted(cells_by_route_and_config.items()):
        if "H0" not in by_config:
            raise ValueError(f"route {route!r} has no H0 confirmation cells.")
        baseline = _aggregate_candidate_rows(by_config["H0"], target_step)
        rows_by_key[(route, "H0")] = baseline
        eligible_keys.add((route, "H0"))
        for index, (config_name, directories) in enumerate(sorted(by_config.items())):
            rows = _aggregate_candidate_rows(directories, target_step)
            rows_by_key[(route, config_name)] = rows
            if config_name == "H0":
                result = {"accepted": True, "candidate": _aggregate_metrics(rows)}
            else:
                result = classify_hyperparameter_candidate(
                    baseline,
                    rows,
                    seed=seed + len(candidates) * 10 + index,
                )
                if result["accepted"]:
                    eligible_keys.add((route, config_name))
            diagnostics[f"{route}:{config_name}"] = result
            metrics = _aggregate_metrics(rows)
            candidates.append(
                {
                    "route": route,
                    "config": config_name,
                    **metrics,
                    "accepted_against_route_h0": bool(result["accepted"]),
                }
            )
    eligible = [
        row for row in candidates if (row["route"], row["config"]) in eligible_keys
    ]
    winner = min(
        eligible,
        key=lambda row: (
            float(row["failure_rate"]),
            float(row["mean_j"]),
            float(row["mean_success_makespan"]),
            str(row["route"]),
            str(row["config"]),
        ),
    )
    winner_dirs = cells_by_route_and_config[winner["route"]][winner["config"]]
    seed_scores: list[tuple[float, int, Path]] = []
    for directory in winner_dirs:
        cell = json.loads(
            (Path(directory) / "cell_manifest.json").read_text(encoding="utf-8")
        )
        rows = _final_step_rows(directory, target_step)
        score = float(np.mean([float(row["failure_aware_j"]) for row in rows]))
        checkpoint = checkpoint_for_target_step(directory, target_step)
        seed_scores.append((score, int(cell["seed"]), checkpoint))
    seed_scores.sort(key=lambda item: (item[0], item[1]))
    medoid = seed_scores[len(seed_scores) // 2]
    selection = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "selection_order": [
            "failure_rate",
            "mean_failure_aware_j",
            "mean_success_makespan",
            "route",
            "config",
        ],
        "winner": winner,
        "medoid_seed": medoid[1],
        "scheduler_checkpoint": str(medoid[2]),
        "scheduler_checkpoint_sha256": sha256_file(medoid[2]),
        "seed_validation_j": [
            {"seed": item[1], "mean_j": item[0], "checkpoint": str(item[2])}
            for item in seed_scores
        ],
        "candidates": candidates,
        "diagnostics": diagnostics,
    }
    _write_json(Path(output_path), selection)
    return selection


def initialize_round1_study(
    *,
    output_dir: str | Path,
    architecture_checkpoint: str | Path,
    gp_policy: str | Path,
    base_seed: int = 20260824,
    device: str = "auto",
    scenario_sizes: dict[str, int] | None = None,
) -> Path:
    """Create the immutable inputs and preregistered matrices for the study."""

    destination = Path(output_dir).resolve()
    manifest_path = destination / "study_manifest.json"
    architecture_path = Path(architecture_checkpoint).resolve()
    gp_path = Path(gp_policy).resolve()
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = manifest["inputs"]
        if inputs["architecture_checkpoint"]["sha256"] != sha256_file(architecture_path):
            raise ValueError("round-one Architecture DQN input hash changed.")
        if inputs["g0_policy"]["sha256"] != sha256_file(gp_path):
            raise ValueError("round-one G0 input hash changed.")
        if manifest.get("stages", {}).get("preflight", {}).get("status") != "complete":
            preflight_round1_study(manifest_path)
        return manifest_path
    destination.mkdir(parents=True, exist_ok=True)
    sizes = {
        "b_train_size": 256,
        "b_validation_size": 256,
        "g_train_size": 256,
        "g_validation_size": 256,
        "test_iid_size": 1000,
        "test_ood_size": 500,
        **(scenario_sizes or {}),
    }
    scenario_paths = generate_round1_scenarios(
        destination / "scenarios",
        base_seed=int(base_seed),
        **sizes,
    )
    manifest = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "code_commit": _current_git_commit(),
        "output_dir": str(destination),
        "base_seed": int(base_seed),
        "device": default_device() if device == "auto" else str(device),
        "inputs": {
            "architecture_checkpoint": {
                "path": str(architecture_path),
                "sha256": sha256_file(architecture_path),
            },
            "g0_policy": {"path": str(gp_path), "sha256": sha256_file(gp_path)},
        },
        "scenarios": {
            name: str(path)
            for name, path in scenario_paths.items()
            if name != "registry"
        },
        "scenario_registry": str(scenario_paths["registry"]),
        "seeds": list(range(1, 9)),
        "discovery_seeds": [1, 2, 3],
        "confirmation_seeds": [4, 5, 6, 7, 8],
        "convergence_steps": list(DEFAULT_CONVERGENCE_STEPS),
        "transfer_steps": list(DEFAULT_TRANSFER_STEPS),
        "migration_paths": [
            "fixed_to_fixed",
            "fixed_to_g0",
            "arch_to_arch",
            "arch_to_g0",
            "g0_to_g0",
        ],
        "hyperparameter_matrix": bdqn_hyperparameter_matrix(),
        "gp_discovery_matrix": gp_discovery_matrix(),
        "test_locked": True,
        "stages": {},
    }
    _write_json(manifest_path, manifest)
    preflight_round1_study(manifest_path)
    return manifest_path


def _load_study(study_manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(study_manifest).resolve()
    study = json.loads(path.read_text(encoding="utf-8"))
    if int(study.get("schema_version", -1)) != ROUND1_STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported round-one study schema.")
    for name in ("architecture_checkpoint", "g0_policy"):
        record = study["inputs"][name]
        if sha256_file(record["path"]) != record["sha256"]:
            raise ValueError(f"round-one input hash changed: {name}")
    return path, study


def _mark_stage(
    manifest_path: Path,
    study: dict[str, Any],
    stage: str,
    status: str,
    **details: Any,
) -> None:
    study.setdefault("stages", {})[stage] = {
        "status": status,
        "updated_at": _utc_now(),
        **details,
    }
    _write_json(manifest_path, study)


def run_round1_convergence_stage(study_manifest: str | Path) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "preflight")
    root = Path(study["output_dir"])
    _mark_stage(manifest_path, study, "bdqn_convergence", "running")
    cells: dict[str, dict[int, str]] = {provider: {} for provider in PROVIDER_KINDS}
    for seed in study["seeds"]:
        config = baseline_bdqn_config(
            seed=int(seed),
            max_env_steps=max(study["convergence_steps"]),
            device=study["device"],
        )
        initial = ensure_initial_checkpoint(root, seed=int(seed), config=config)
        for provider in PROVIDER_KINDS:
            cell_dir = root / "bdqn" / "convergence" / provider / f"seed_{int(seed)}"
            train_bdqn_provider_cell(
                output_dir=cell_dir,
                provider_kind=provider,
                source_checkpoint=initial,
                train_manifest=study["scenarios"]["b_train"],
                validation_manifest=study["scenarios"]["b_validation"],
                config=config,
                checkpoint_steps=study["convergence_steps"],
                architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
                gp_policy=study["inputs"]["g0_policy"]["path"],
            )
            cells[provider][int(seed)] = str(cell_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    t0 = common_convergence_step(cells)
    checkpoint_map = {
        provider: {
            str(seed): str(checkpoint_for_target_step(directory, t0["t0"]))
            for seed, directory in by_seed.items()
        }
        for provider, by_seed in cells.items()
    }
    selection_path = root / "bdqn" / "t0_selection.json"
    _write_json(selection_path, {**t0, "checkpoints": checkpoint_map})
    _mark_stage(
        manifest_path,
        study,
        "bdqn_convergence",
        "complete",
        t0=int(t0["t0"]),
        selection=str(selection_path),
    )
    return {"t0": t0, "cells": cells, "selection": selection_path}


def _t0_checkpoint_map(study: dict[str, Any]) -> tuple[int, dict[str, dict[int, Path]]]:
    path = Path(study["output_dir"]) / "bdqn" / "t0_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        provider: {int(seed): Path(value) for seed, value in by_seed.items()}
        for provider, by_seed in payload["checkpoints"].items()
    }
    return int(payload["t0"]), mapping


def run_round1_cross_stage(study_manifest: str | Path) -> dict[str, Path]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_convergence")
    root = Path(study["output_dir"])
    _, checkpoints = _t0_checkpoint_map(study)
    jobs = provider_cross_matrix_jobs(
        seeds=study["seeds"],
        checkpoint_by_training_provider=checkpoints,
    )
    _mark_stage(manifest_path, study, "provider_cross", "running")
    outputs = evaluate_provider_cross_matrix(
        output_dir=root / "bdqn" / "cross_matrix",
        jobs=jobs,
        validation_manifest=study["scenarios"]["b_validation"],
        architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
        gp_policy=study["inputs"]["g0_policy"]["path"],
        device=study["device"],
    )
    _mark_stage(
        manifest_path,
        study,
        "provider_cross",
        "complete",
        outputs={name: str(path) for name, path in outputs.items()},
    )
    return outputs


def _config_from_hyperparameters(
    row: dict[str, Any],
    *,
    seed: int,
    max_env_steps: int,
    device: str,
) -> BranchingDQNConfig:
    return finetune_bdqn_config(
        seed=int(seed),
        max_env_steps=int(max_env_steps),
        device=device,
        lr=float(row["lr"]),
        batch_size=int(row["batch_size"]),
        buffer_size=int(row["buffer_size"]),
        target_update_interval=int(row["target_update_interval"]),
        epsilon_start=float(row["epsilon_start"]),
        epsilon_end=float(row["epsilon_end"]),
        epsilon_decay=float(row["epsilon_decay"]),
    )


def run_round1_migration_stage(study_manifest: str | Path) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_convergence")
    root = Path(study["output_dir"])
    _, checkpoints = _t0_checkpoint_map(study)
    jobs = migration_path_jobs(
        seeds=study["seeds"],
        t0_checkpoints=checkpoints,
    )
    h0 = bdqn_hyperparameter_matrix()[0]
    _mark_stage(manifest_path, study, "bdqn_migration", "running")
    cells: dict[str, dict[int, str]] = {}
    for job in jobs:
        route = str(job["name"])
        seed = int(job["seed"])
        cell_dir = root / "bdqn" / "migration" / route / "H0" / f"seed_{seed}"
        config = _config_from_hyperparameters(
            h0,
            seed=seed,
            max_env_steps=max(study["transfer_steps"]),
            device=study["device"],
        )
        train_bdqn_provider_cell(
            output_dir=cell_dir,
            provider_kind=str(job["target_provider"]),
            source_checkpoint=job["source_checkpoint"],
            train_manifest=study["scenarios"]["b_train"],
            validation_manifest=study["scenarios"]["b_validation"],
            config=config,
            checkpoint_steps=study["transfer_steps"],
            architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
            gp_policy=study["inputs"]["g0_policy"]["path"],
        )
        cells.setdefault(route, {})[seed] = str(cell_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _mark_stage(
        manifest_path,
        study,
        "bdqn_migration",
        "complete",
        cell_count=sum(len(values) for values in cells.values()),
    )
    return {"cells": cells}


def run_round1_hyper_screen_stage(study_manifest: str | Path) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_migration")
    root = Path(study["output_dir"])
    _, checkpoints = _t0_checkpoint_map(study)
    matrix = {row["name"]: row for row in study["hyperparameter_matrix"]}
    _mark_stage(manifest_path, study, "bdqn_hyper_screen", "running")
    cells_by_name: dict[str, list[str]] = {
        "H0": [
            str(root / "bdqn" / "migration" / "arch_to_g0" / "H0" / f"seed_{seed}")
            for seed in study["discovery_seeds"]
        ]
    }
    screen_steps = (0, 10000, 20000, 30000, 40000)
    for name, hyperparameters in matrix.items():
        if name == "H0":
            continue
        cells_by_name[name] = []
        for seed in study["discovery_seeds"]:
            cell_dir = (
                root
                / "bdqn"
                / "hyper_screen"
                / "arch_to_g0"
                / name
                / f"seed_{seed}"
            )
            config = _config_from_hyperparameters(
                hyperparameters,
                seed=int(seed),
                max_env_steps=40000,
                device=study["device"],
            )
            train_bdqn_provider_cell(
                output_dir=cell_dir,
                provider_kind="g0",
                source_checkpoint=checkpoints["arch"][int(seed)],
                train_manifest=study["scenarios"]["b_train"],
                validation_manifest=study["scenarios"]["b_validation"],
                config=config,
                checkpoint_steps=screen_steps,
                architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
                gp_policy=study["inputs"]["g0_policy"]["path"],
            )
            cells_by_name[name].append(str(cell_dir))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selection_path = root / "bdqn" / "hyper_screen" / "selection.json"
    selection = select_hyperparameter_screen(
        output_path=selection_path,
        cells_by_name=cells_by_name,
        target_step=40000,
        seed=int(study["base_seed"]),
    )
    _mark_stage(
        manifest_path,
        study,
        "bdqn_hyper_screen",
        "complete",
        selection=str(selection_path),
        hsingle=selection["hsingle"],
        hstar=selection["hstar"],
    )
    return {"cells": cells_by_name, "selection": selection}


def run_round1_hyper_confirm_stage(study_manifest: str | Path) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_hyper_screen")
    root = Path(study["output_dir"])
    _, checkpoints = _t0_checkpoint_map(study)
    selection = json.loads(
        (root / "bdqn" / "hyper_screen" / "selection.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = {row["name"]: row for row in study["hyperparameter_matrix"]}
    named_configs = {"H0": matrix["H0"]}
    hsingle = str(selection["hsingle"])
    named_configs[hsingle] = matrix[hsingle]
    named_configs["Hstar"] = dict(selection["hstar"])
    _mark_stage(manifest_path, study, "bdqn_hyper_confirm", "running")
    cells: dict[str, dict[str, list[str]]] = {}
    config_signature_to_cells: dict[tuple[str, tuple[tuple[str, Any], ...]], list[str]] = {}
    for source_provider in PROVIDER_KINDS:
        route = f"{source_provider}_to_g0"
        cells[route] = {}
        for config_name, hyperparameters in named_configs.items():
            if config_name == "H0":
                cells[route][config_name] = [
                    str(
                        root
                        / "bdqn"
                        / "migration"
                        / route
                        / "H0"
                        / f"seed_{seed}"
                    )
                    for seed in study["confirmation_seeds"]
                ]
                continue
            signature = tuple(
                sorted((key, value) for key, value in hyperparameters.items() if key != "name")
            )
            shared_key = (route, signature)
            if shared_key in config_signature_to_cells:
                cells[route][config_name] = config_signature_to_cells[shared_key]
                continue
            directories: list[str] = []
            for seed in study["confirmation_seeds"]:
                cell_dir = (
                    root
                    / "bdqn"
                    / "hyper_confirm"
                    / route
                    / config_name
                    / f"seed_{seed}"
                )
                config = _config_from_hyperparameters(
                    hyperparameters,
                    seed=int(seed),
                    max_env_steps=80000,
                    device=study["device"],
                )
                train_bdqn_provider_cell(
                    output_dir=cell_dir,
                    provider_kind="g0",
                    source_checkpoint=checkpoints[source_provider][int(seed)],
                    train_manifest=study["scenarios"]["b_train"],
                    validation_manifest=study["scenarios"]["b_validation"],
                    config=config,
                    checkpoint_steps=study["transfer_steps"],
                    architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
                    gp_policy=study["inputs"]["g0_policy"]["path"],
                )
                directories.append(str(cell_dir))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            cells[route][config_name] = directories
            config_signature_to_cells[shared_key] = directories
    final_path = root / "bdqn" / "final_selection.json"
    final = select_final_bdqn_confirmation(
        output_path=final_path,
        cells_by_route_and_config=cells,
        target_step=80000,
        seed=int(study["base_seed"]) + 1000,
    )
    _mark_stage(
        manifest_path,
        study,
        "bdqn_hyper_confirm",
        "complete",
        selection=str(final_path),
        scheduler_checkpoint=final["scheduler_checkpoint"],
        route=final["winner"]["route"],
        config=final["winner"]["config"],
    )
    return {"cells": cells, "selection": final}


def _require_complete_stage(study: dict[str, Any], stage: str) -> None:
    status = study.get("stages", {}).get(stage, {}).get("status")
    if status != "complete":
        raise RuntimeError(f"round-one stage {stage!r} is not complete.")


def _gp_job_label(population_size: int, generations: int) -> str:
    return f"p{int(population_size)}_g{int(generations)}"


def _gp_per_run_winners(output_dir: str | Path) -> dict[int, dict[str, Any]]:
    root = Path(output_dir)
    candidate_rows = [
        json.loads(line)
        for line in (root / "candidate_rules.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validation_by_expression = {
        str(row["expression"]): row
        for row in _read_csv(root / "validation_results.csv")
    }
    by_run: dict[int, set[str]] = {}
    for row in candidate_rows:
        expression = str(row["expression"])
        if expression in validation_by_expression:
            by_run.setdefault(int(row["run_seed"]), set()).add(expression)
    return {
        run_seed: _select_validation_winner(expressions, validation_by_expression)
        for run_seed, expressions in sorted(by_run.items())
    }


def gp_axis_convergence(
    *,
    jobs: Sequence[dict[str, Any]],
    family: str,
    seed: int,
) -> dict[str, Any]:
    """Compare adjacent GP axis levels using common independent-run seeds."""

    if family not in {"population_axis", "generation_axis"}:
        raise ValueError("unknown GP convergence axis.")
    axis = "population_size" if family == "population_axis" else "generations"
    selected = [job for job in jobs if family in str(job["families"]).split("+")]
    selected.sort(key=lambda row: int(row[axis]))
    levels: list[dict[str, Any]] = []
    winners_by_level: dict[int, dict[int, dict[str, Any]]] = {}
    for job in selected:
        level = int(job[axis])
        winners = _gp_per_run_winners(job["output_dir"])
        winners_by_level[level] = winners
        levels.append(
            {
                axis: level,
                "population_size": int(job["population_size"]),
                "generations": int(job["generations"]),
                "mean_failure_rate": float(
                    np.mean([float(row["failure_rate"]) for row in winners.values()])
                ),
                "mean_raw_j": float(
                    np.mean([float(row["raw_mean_j"]) for row in winners.values()])
                ),
                "independent_runs": len(winners),
            }
        )
    comparisons: list[dict[str, Any]] = []
    converged_level: int | None = None
    for index, (before, after) in enumerate(zip(levels[:-1], levels[1:], strict=True)):
        lower = winners_by_level[int(before[axis])]
        upper = winners_by_level[int(after[axis])]
        common = sorted(set(lower) & set(upper))
        deltas = [
            float(upper[run]["raw_mean_j"]) - float(lower[run]["raw_mean_j"])
            for run in common
        ]
        low, high = paired_bootstrap_ci(deltas, samples=10000, seed=seed + index)
        relative_improvement = (
            float(before["mean_raw_j"]) - float(after["mean_raw_j"])
        ) / max(abs(float(before["mean_raw_j"])), 1e-12)
        no_failure_gain = float(after["mean_failure_rate"]) >= float(
            before["mean_failure_rate"]
        )
        converged = (
            relative_improvement < 0.01
            and low <= 0.0 <= high
            and no_failure_gain
        )
        comparisons.append(
            {
                "from": int(before[axis]),
                "to": int(after[axis]),
                "relative_j_improvement": relative_improvement,
                "delta_j_ci95_low": low,
                "delta_j_ci95_high": high,
                "failure_rate_not_improved": no_failure_gain,
                "converged": converged,
            }
        )
        if converged_level is None and converged:
            converged_level = int(before[axis])
    return {
        "axis": axis,
        "levels": levels,
        "comparisons": comparisons,
        "recommended_level": converged_level,
    }


def run_round1_gp_discovery_stage(
    study_manifest: str | Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_hyper_confirm")
    root = Path(study["output_dir"])
    final_bdqn = json.loads(
        (root / "bdqn" / "final_selection.json").read_text(encoding="utf-8")
    )
    _mark_stage(manifest_path, study, "gp_discovery", "running")
    jobs: list[dict[str, Any]] = []
    for index, registered in enumerate(study["gp_discovery_matrix"]):
        population = int(registered["population_size"])
        generations = int(registered["generations"])
        output_dir = root / "gp" / "discovery" / _gp_job_label(population, generations)
        run_gp_matrix_job(
            output_dir=output_dir,
            scheduler_checkpoint=final_bdqn["scheduler_checkpoint"],
            scenario_dir=Path(study["scenarios"]["g_train"]).parent,
            population_size=population,
            generations=generations,
            independent_runs=3,
            base_seed=int(study["base_seed"]) + 200000,
            workers=int(workers),
        )
        generation = gp_generation_convergence(
            anchor_history_path=output_dir / "anchor_history.csv"
        )
        _write_json(output_dir / "generation_convergence.json", generation)
        jobs.append({**registered, "output_dir": str(output_dir)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    discovery_dir = root / "gp" / "discovery"
    selection = select_gp_discovery_configuration(
        output_path=discovery_dir / "selection.json",
        jobs=jobs,
    )
    axis_convergence = {
        family: gp_axis_convergence(
            jobs=jobs,
            family=family,
            seed=int(study["base_seed"]) + 210000 + index * 100,
        )
        for index, family in enumerate(("population_axis", "generation_axis"))
    }
    _write_json(discovery_dir / "axis_convergence.json", axis_convergence)
    _write_json(discovery_dir / "jobs.json", {"jobs": jobs})
    _mark_stage(
        manifest_path,
        study,
        "gp_discovery",
        "complete",
        selection=str(discovery_dir / "selection.json"),
        winner=selection["winner"],
    )
    return {"jobs": jobs, "selection": selection, "axis_convergence": axis_convergence}


def run_round1_gp_confirm_stage(
    study_manifest: str | Path,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "gp_discovery")
    root = Path(study["output_dir"])
    final_bdqn = json.loads(
        (root / "bdqn" / "final_selection.json").read_text(encoding="utf-8")
    )
    discovery = json.loads(
        (root / "gp" / "discovery" / "selection.json").read_text(encoding="utf-8")
    )
    selected = discovery["winner"]
    configurations = {
        "selected": (int(selected["population_size"]), int(selected["generations"])),
        "standard_200x80": (200, 80),
    }
    _mark_stage(manifest_path, study, "gp_confirm", "running")
    outputs: dict[str, str] = {}
    executed: dict[tuple[int, int], str] = {}
    for name, (population, generations) in configurations.items():
        signature = (population, generations)
        if signature in executed:
            outputs[name] = executed[signature]
            continue
        output_dir = root / "gp" / "confirm" / name
        run_gp_matrix_job(
            output_dir=output_dir,
            scheduler_checkpoint=final_bdqn["scheduler_checkpoint"],
            scenario_dir=Path(study["scenarios"]["g_train"]).parent,
            population_size=population,
            generations=generations,
            independent_runs=10,
            base_seed=int(study["base_seed"]) + 300000,
            workers=int(workers),
        )
        generation = gp_generation_convergence(
            anchor_history_path=output_dir / "anchor_history.csv"
        )
        _write_json(output_dir / "generation_convergence.json", generation)
        outputs[name] = str(output_dir)
        executed[signature] = str(output_dir)
    selected_dir = Path(outputs["selected"])
    run_count = gp_run_count_convergence(
        confirmation_dir=selected_dir,
        output_path=selected_dir / "run_count_convergence.json",
        seed=int(study["base_seed"]) + 310000,
    )
    policy = selected_dir / "gp_policy.json"
    _mark_stage(
        manifest_path,
        study,
        "gp_confirm",
        "complete",
        outputs=outputs,
        selected_policy=str(policy),
        selected_policy_sha256=sha256_file(policy),
        recommended_runs=run_count["recommended_run_count"],
    )
    return {"outputs": outputs, "run_count": run_count, "selected_policy": policy}


def run_round1_final_test_stage(study_manifest: str | Path) -> dict[str, Any]:
    """Consume Test-v2 once, only after BDQN and GP have both been locked."""

    from .gp_architecture import evaluate_gp_stack

    manifest_path, study = _load_study(study_manifest)
    _require_complete_stage(study, "bdqn_hyper_confirm")
    _require_complete_stage(study, "gp_confirm")
    if study.get("stages", {}).get("final_test", {}).get("status") == "complete":
        return study["stages"]["final_test"]["outputs"]
    if not bool(study.get("test_locked", False)):
        raise RuntimeError("Test-v2 has already been unlocked or consumed.")
    root = Path(study["output_dir"])
    final_bdqn = json.loads(
        (root / "bdqn" / "final_selection.json").read_text(encoding="utf-8")
    )
    selected_policy = root / "gp" / "confirm" / "selected" / "gp_policy.json"
    _mark_stage(manifest_path, study, "final_test", "running")
    outputs: dict[str, dict[str, str]] = {}
    for split, manifest_key in (("iid", "test_iid_v2"), ("ood", "test_ood_v2")):
        paths = evaluate_gp_stack(
            gp_policy=selected_policy,
            scheduler_checkpoint=final_bdqn["scheduler_checkpoint"],
            scenario_manifest=study["scenarios"][manifest_key],
            output_dir=root / "final_test" / split,
            baselines=("fixed", "random_concrete", "manual6_dqn", "gp"),
            manual_architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
            device=study["device"],
        )
        outputs[split] = {name: str(path) for name, path in paths.items()}
    study["test_locked"] = False
    study["test_consumed_at"] = _utc_now()
    _mark_stage(
        manifest_path,
        study,
        "final_test",
        "complete",
        outputs=outputs,
        scheduler_checkpoint_sha256=sha256_file(final_bdqn["scheduler_checkpoint"]),
        gp_policy_sha256=sha256_file(selected_policy),
        scenario_manifest_sha256={
            key: sha256_file(study["scenarios"][key])
            for key in ("test_iid_v2", "test_ood_v2")
        },
    )
    return outputs


ROUND1_STAGE_ORDER = (
    "convergence",
    "cross",
    "migration",
    "hyper-screen",
    "hyper-confirm",
    "gp-discovery",
    "gp-confirm",
    "final-test",
    "report",
)


def run_round1_stage(
    study_manifest: str | Path,
    stage: str,
    *,
    workers: int = 8,
) -> Any:
    """Dispatch one resumable round-one stage or the complete registered chain."""

    runners = {
        "convergence": run_round1_convergence_stage,
        "cross": run_round1_cross_stage,
        "migration": run_round1_migration_stage,
        "hyper-screen": run_round1_hyper_screen_stage,
        "hyper-confirm": run_round1_hyper_confirm_stage,
        "gp-discovery": lambda path: run_round1_gp_discovery_stage(path, workers=workers),
        "gp-confirm": lambda path: run_round1_gp_confirm_stage(path, workers=workers),
        "final-test": run_round1_final_test_stage,
        "report": build_round1_report,
    }
    if stage == "all":
        return {name: runners[name](study_manifest) for name in ROUND1_STAGE_ORDER}
    if stage not in runners:
        raise ValueError(f"unknown round-one stage: {stage!r}")
    return runners[stage](study_manifest)


def preflight_round1_study(study_manifest: str | Path) -> dict[str, Any]:
    """Run deterministic invariants before any round-one optimisation stage."""

    manifest_path, study = _load_study(study_manifest)
    root = Path(study["output_dir"])
    manifests = {
        name: load_scenario_manifest(path)
        for name, path in study["scenarios"].items()
    }
    _assert_disjoint_manifests(manifests)
    static_count = 0
    fixed_keep_checks = 0
    fixed_provider = FixedArchitectureProvider()
    for manifest in manifests.values():
        for payload in manifest["scenarios"]:
            _validate_static_payload(payload)
            static_count += 1
        payload = manifest["scenarios"][0]
        mission_env = _scenario_environment(payload, "fixed")
        before = (
            tuple(int(value) for value in mission_env.active_system_mask.tolist()),
            float(mission_env.net_cost),
        )
        decision = fixed_provider.act(mission_env)
        after = (
            tuple(int(value) for value in mission_env.active_system_mask.tolist()),
            float(mission_env.net_cost),
        )
        if decision.action.kind != "keep" or decision.changed or before != after:
            raise RuntimeError("fixed-provider KEEP invariant failed during preflight.")
        fixed_keep_checks += 1
    initial_records: list[dict[str, Any]] = []
    for seed in study["seeds"]:
        config = baseline_bdqn_config(
            seed=int(seed),
            max_env_steps=max(study["convergence_steps"]),
            device=study["device"],
        )
        checkpoint = ensure_initial_checkpoint(root, seed=int(seed), config=config)
        agent, _ = load_branching_checkpoint(
            checkpoint, device=study["device"], load_optimizer=False
        )
        parameter_hash = _branching_parameter_hash(agent)
        initial_records.append(
            {
                "seed": int(seed),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "parameter_sha256": parameter_hash,
                "provider_parameter_sha256": {
                    provider: parameter_hash for provider in PROVIDER_KINDS
                },
                "all_provider_tensors_identical": True,
            }
        )
    result = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "code_commit": _current_git_commit(),
        "scenario_manifests_disjoint": True,
        "static_feasible_scenarios_checked": static_count,
        "fixed_keep_manifests_checked": fixed_keep_checks,
        "provider_input_hashes": {
            provider: _provider_input_hash(
                provider,
                architecture_checkpoint=study["inputs"]["architecture_checkpoint"]["path"],
                gp_policy=study["inputs"]["g0_policy"]["path"],
            )
            for provider in PROVIDER_KINDS
        },
        "shared_initial_weights": initial_records,
        "test_locked": bool(study["test_locked"]),
    }
    path = root / "preflight.json"
    _write_json(path, result)
    _mark_stage(manifest_path, study, "preflight", "complete", report=str(path))
    return result


def _as_bool(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def _hierarchical_bootstrap_delta_ci(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    delta,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float]:
    """Bootstrap seeds, then scenarios within each seed/category stratum."""

    strata: dict[tuple[int, str], list[float]] = {}
    for left, right in pairs:
        key = (int(left["seed"]), str(left["category"]))
        strata.setdefault(key, []).append(float(delta(left, right)))
    seeds = sorted({key[0] for key in strata})
    categories = sorted({key[1] for key in strata})
    if not seeds:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values: list[float] = []
        for sampled_seed in sampled_seeds:
            for category in categories:
                stratum = strata.get((int(sampled_seed), category), [])
                if stratum:
                    values.extend(
                        rng.choice(stratum, size=len(stratum), replace=True).tolist()
                    )
        estimates[index] = float(np.mean(values))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _hierarchical_bootstrap_ci(
    pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    field: str,
    seed: int,
    samples: int = 5000,
) -> tuple[float, float]:
    return _hierarchical_bootstrap_delta_ci(
        pairs,
        delta=lambda left, right: float(right[field]) - float(left[field]),
        seed=seed,
        samples=samples,
    )


def _paired_cross_comparison(
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    *,
    left_label: str,
    right_label: str,
    contrast: str,
    seed: int,
) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[int, str]:
        return int(row["seed"]), str(row["scenario_hash"])

    left = {key(row): row for row in left_rows}
    right = {key(row): row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("cross comparison rows are not paired by seed and scenario.")
    pairs = [(left[item], right[item]) for item in sorted(left)]
    j_deltas = [
        float(right_row["failure_aware_j"]) - float(left_row["failure_aware_j"])
        for left_row, right_row in pairs
    ]
    low, high = _hierarchical_bootstrap_ci(
        pairs, field="failure_aware_j", seed=seed
    )
    failure_ci = _hierarchical_bootstrap_delta_ci(
        pairs,
        delta=lambda left, right: float(not _as_bool(right["success"]))
        - float(not _as_bool(left["success"])),
        seed=seed + 1,
    )
    tolerance = 1e-12
    wins = sum(value < -tolerance for value in j_deltas)
    losses = sum(value > tolerance for value in j_deltas)
    ties = len(j_deltas) - wins - losses
    successful_pairs = [
        (left_row, right_row)
        for left_row, right_row in pairs
        if _as_bool(left_row["success"]) and _as_bool(right_row["success"])
    ]
    makespan_ci = _hierarchical_bootstrap_ci(
        successful_pairs,
        field="makespan",
        seed=seed + 2,
    )
    budget_ci = _hierarchical_bootstrap_delta_ci(
        pairs,
        delta=lambda left, right: float(_as_bool(right["ever_over_budget"]))
        - float(_as_bool(left["ever_over_budget"])),
        seed=seed + 3,
    )
    peak_ci = _hierarchical_bootstrap_ci(
        pairs,
        field="peak_net_cost",
        seed=seed + 4,
    )
    architecture_ci = _hierarchical_bootstrap_ci(
        pairs,
        field="architecture_changes",
        seed=seed + 5,
    )
    makespan_delta = (
        float(
            np.mean(
                [
                    float(right_row["makespan"]) - float(left_row["makespan"])
                    for left_row, right_row in successful_pairs
                ]
            )
        )
        if successful_pairs
        else float("nan")
    )
    return {
        "contrast": contrast,
        "baseline": left_label,
        "candidate": right_label,
        "pairs": len(pairs),
        "delta_j_mean": float(np.mean(j_deltas)),
        "delta_j_ci95_low": low,
        "delta_j_ci95_high": high,
        "failure_rate_delta": float(
            np.mean(
                [
                    float(not _as_bool(right_row["success"]))
                    - float(not _as_bool(left_row["success"]))
                    for left_row, right_row in pairs
                ]
            )
        ),
        "failure_rate_delta_ci95_low": failure_ci[0],
        "failure_rate_delta_ci95_high": failure_ci[1],
        "successful_makespan_delta": makespan_delta,
        "successful_makespan_delta_ci95_low": makespan_ci[0],
        "successful_makespan_delta_ci95_high": makespan_ci[1],
        "process_budget_violation_delta": float(
            np.mean(
                [
                    float(_as_bool(right_row["ever_over_budget"]))
                    - float(_as_bool(left_row["ever_over_budget"]))
                    for left_row, right_row in pairs
                ]
            )
        ),
        "process_budget_violation_delta_ci95_low": budget_ci[0],
        "process_budget_violation_delta_ci95_high": budget_ci[1],
        "peak_cost_delta": float(
            np.mean(
                [
                    float(right_row["peak_net_cost"])
                    - float(left_row["peak_net_cost"])
                    for left_row, right_row in pairs
                ]
            )
        ),
        "peak_cost_delta_ci95_low": peak_ci[0],
        "peak_cost_delta_ci95_high": peak_ci[1],
        "architecture_changes_delta": float(
            np.mean(
                [
                    float(right_row["architecture_changes"])
                    - float(left_row["architecture_changes"])
                    for left_row, right_row in pairs
                ]
            )
        ),
        "architecture_changes_delta_ci95_low": architecture_ci[0],
        "architecture_changes_delta_ci95_high": architecture_ci[1],
        "win": wins,
        "tie": ties,
        "loss": losses,
    }


def build_cross_pairwise_statistics(
    *,
    cross_results: str | Path,
    output_path: str | Path,
    seed: int,
) -> list[dict[str, Any]]:
    rows = _read_csv(cross_results)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["training_provider"]), str(row["evaluation_provider"])), []
        ).append(row)
    results: list[dict[str, Any]] = []
    pair_index = 0
    for evaluation_provider in PROVIDER_KINDS:
        for left_index, left in enumerate(PROVIDER_KINDS[:-1]):
            for right in PROVIDER_KINDS[left_index + 1 :]:
                results.append(
                    _paired_cross_comparison(
                        grouped[(left, evaluation_provider)],
                        grouped[(right, evaluation_provider)],
                        left_label=f"{left}->{evaluation_provider}",
                        right_label=f"{right}->{evaluation_provider}",
                        contrast="training_source_same_test_provider",
                        seed=seed + pair_index,
                    )
                )
                pair_index += 1
    for training_provider in PROVIDER_KINDS:
        for left_index, left in enumerate(PROVIDER_KINDS[:-1]):
            for right in PROVIDER_KINDS[left_index + 1 :]:
                results.append(
                    _paired_cross_comparison(
                        grouped[(training_provider, left)],
                        grouped[(training_provider, right)],
                        left_label=f"{training_provider}->{left}",
                        right_label=f"{training_provider}->{right}",
                        contrast="test_provider_same_training_source",
                        seed=seed + pair_index,
                    )
                )
                pair_index += 1
    _write_csv(Path(output_path), results)
    return results


def build_migration_pairwise_statistics(
    *,
    migration_dir: str | Path,
    output_path: str | Path,
    target_step: int,
    seed: int,
) -> list[dict[str, Any]]:
    root = Path(migration_dir)
    rows_by_route_step: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for route_dir in sorted(root.glob("*_to_*")):
        for path in (route_dir / "H0").glob("seed_*/validation/checkpoint_results.csv"):
            for row in _read_csv(path):
                step = int(float(row["target_environment_steps"]))
                if step in {0, int(target_step)}:
                    rows_by_route_step.setdefault((route_dir.name, step), []).append(row)
    comparisons: list[tuple[str, int, str, int, str]] = [
        ("arch_to_arch", target_step, "arch_to_g0", target_step, "A→G_vs_A→A"),
        ("fixed_to_fixed", target_step, "fixed_to_g0", target_step, "F→G_vs_F→F"),
        ("fixed_to_g0", target_step, "arch_to_g0", target_step, "A→G_vs_F→G"),
        ("g0_to_g0", target_step, "arch_to_g0", target_step, "A→G_vs_G→G"),
        ("g0_to_g0", target_step, "fixed_to_g0", target_step, "F→G_vs_G→G"),
    ]
    comparisons.extend(
        (route, 0, route, target_step, f"{route}_continued_vs_t0")
        for route in (
            "fixed_to_fixed",
            "fixed_to_g0",
            "arch_to_arch",
            "arch_to_g0",
            "g0_to_g0",
        )
    )
    results: list[dict[str, Any]] = []
    for index, (left_route, left_step, right_route, right_step, label) in enumerate(
        comparisons
    ):
        results.append(
            _paired_cross_comparison(
                rows_by_route_step[(left_route, int(left_step))],
                rows_by_route_step[(right_route, int(right_step))],
                left_label=f"{left_route}@{left_step}",
                right_label=f"{right_route}@{right_step}",
                contrast=label,
                seed=seed + index * 10,
            )
        )
    _write_csv(Path(output_path), results)
    return results


def _mean_by(rows: Sequence[dict[str, Any]], key: str, value: str) -> tuple[list[int], list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(float(row[key])), []).append(float(row[value]))
    x = sorted(grouped)
    return x, [float(np.mean(grouped[item])) for item in x]


def _plot_bdqn_convergence(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (
        ("failure_rate", "Failure rate"),
        ("mean_j", "Mean J"),
        ("mean_success_makespan", "Successful makespan"),
    )
    for provider in PROVIDER_KINDS:
        summary_rows: list[dict[str, Any]] = []
        history_rows: list[dict[str, Any]] = []
        for path in (root / "bdqn" / "convergence" / provider).glob(
            "seed_*/validation/checkpoint_summary.csv"
        ):
            summary_rows.extend(
                row for row in _read_csv(path) if str(row["category"]) == "all"
            )
        for path in (root / "bdqn" / "convergence" / provider).glob(
            "seed_*/training_history.csv"
        ):
            history_rows.extend(_read_csv(path))
        for axis, (metric, title) in zip(axes.flat[:3], metrics, strict=True):
            x, y = _mean_by(summary_rows, "target_environment_steps", metric)
            axis.plot(x, y, marker="o", label=provider)
            axis.set_title(title)
            axis.set_xlabel("Environment steps")
            axis.grid(alpha=0.25)
        loss_bins: dict[int, list[float]] = {}
        for row in history_rows:
            value = row.get("scheduler_loss")
            if value in (None, "", "None"):
                continue
            step = int(float(row["total_env_steps"]))
            loss_bins.setdefault((step // 5000) * 5000, []).append(float(value))
        axes.flat[3].plot(
            sorted(loss_bins),
            [float(np.mean(loss_bins[item])) for item in sorted(loss_bins)],
            label=provider,
        )
    axes.flat[3].set_title("Scheduler loss (5k-step bins)")
    axes.flat[3].set_xlabel("Environment steps")
    axes.flat[3].grid(alpha=0.25)
    for axis in axes.flat:
        axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_cross_heatmap(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in _read_csv(root / "bdqn" / "cross_matrix" / "cross_summary.csv")
        if str(row["category"]) == "all"
    ]
    matrix = np.zeros((3, 3), dtype=np.float64)
    for row_index, training in enumerate(PROVIDER_KINDS):
        for column_index, testing in enumerate(PROVIDER_KINDS):
            values = [
                float(row["mean_j"])
                for row in rows
                if row["training_provider"] == training
                and row["evaluation_provider"] == testing
            ]
            matrix[row_index, column_index] = float(np.mean(values))
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis_r")
    axis.set_xticks(range(3), PROVIDER_KINDS)
    axis.set_yticks(range(3), PROVIDER_KINDS)
    axis.set_xlabel("Test provider")
    axis.set_ylabel("Training provider")
    axis.set_title("3×3 provider cross-evaluation: mean J")
    for row in range(3):
        for column in range(3):
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Mean J (lower is better)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_migration(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for route_dir in sorted((root / "bdqn" / "migration").glob("*_to_*")):
        rows: list[dict[str, Any]] = []
        for path in (route_dir / "H0").glob("seed_*/validation/checkpoint_summary.csv"):
            rows.extend(row for row in _read_csv(path) if row["category"] == "all")
        if not rows:
            continue
        for axis, metric, title in (
            (axes[0], "mean_j", "Migration mean J"),
            (axes[1], "mean_success_makespan", "Migration makespan"),
        ):
            x, y = _mean_by(rows, "target_environment_steps", metric)
            axis.plot(x, y, marker="o", label=route_dir.name)
            axis.set_title(title)
            axis.set_xlabel("Additional environment steps")
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_hyperparameters(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    selection = json.loads(
        (root / "bdqn" / "hyper_screen" / "selection.json").read_text(encoding="utf-8")
    )
    names = sorted(selection["diagnostics"], key=lambda name: int(name[1:]))
    values = [
        float(selection["diagnostics"][name]["relative_j_improvement"]) * 100.0
        for name in names
    ]
    colors = [
        "#2b8a3e" if selection["diagnostics"][name]["accepted"] else "#adb5bd"
        for name in names
    ]
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.bar(names, values, color=colors)
    axis.axhline(1.0, color="#c92a2a", linestyle="--", label="1% guard")
    axis.set_ylabel("Relative mean J improvement (%)")
    axis.set_title("BDQN hyperparameter screening")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_gp_convergence(root: Path, output: Path) -> None:
    import matplotlib.pyplot as plt

    jobs = json.loads(
        (root / "gp" / "discovery" / "jobs.json").read_text(encoding="utf-8")
    )["jobs"]
    candidates = json.loads(
        (root / "gp" / "discovery" / "selection.json").read_text(encoding="utf-8")
    )["candidates"]
    by_output = {str(row["output_dir"]): row for row in candidates}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, family, x_field, title in (
        (axes[0, 0], "population_axis", "population_size", "Population axis (G=50)"),
        (axes[0, 1], "generation_axis", "generations", "Generation axis (P=120)"),
    ):
        rows = sorted(
            [row for row in jobs if family in str(row["families"])],
            key=lambda row: int(row[x_field]),
        )
        axis.plot(
            [int(row[x_field]) for row in rows],
            [float(by_output[str(row["output_dir"])]["raw_mean_j"]) for row in rows],
            marker="o",
        )
        axis.set_title(title)
        axis.set_xlabel(x_field.replace("_", " "))
        axis.set_ylabel("Validation raw mean J")
        axis.grid(alpha=0.25)
    equal = sorted(
        [row for row in jobs if "equal_budget" in str(row["families"])],
        key=lambda row: int(row["population_size"]),
    )
    axes[1, 0].plot(
        [int(row["population_size"]) for row in equal],
        [float(by_output[str(row["output_dir"])]["raw_mean_j"]) for row in equal],
        marker="o",
    )
    axes[1, 0].set_title("Equal evaluation budget P×G=6000")
    axes[1, 0].set_xlabel("Population size")
    axes[1, 0].set_ylabel("Validation raw mean J")
    axes[1, 0].grid(alpha=0.25)
    run_count = json.loads(
        (root / "gp" / "confirm" / "selected" / "run_count_convergence.json").read_text(
            encoding="utf-8"
        )
    )["probability_within_1pct_by_run_count"]
    axes[1, 1].plot(
        [int(row["run_count"]) for row in run_count],
        [float(row["probability"]) for row in run_count],
        marker="o",
    )
    axes[1, 1].axhline(0.95, color="#c92a2a", linestyle="--")
    axes[1, 1].set_ylim(0.0, 1.02)
    axes[1, 1].set_title("Cumulative independent-run convergence")
    axes[1, 1].set_xlabel("Independent runs")
    axes[1, 1].set_ylabel("P(within 1% of 10-run optimum)")
    axes[1, 1].grid(alpha=0.25)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def build_round1_report(study_manifest: str | Path) -> dict[str, str]:
    """Create the registered statistics, plots and a reproducibility index."""

    import matplotlib

    matplotlib.use("Agg")
    manifest_path, study = _load_study(study_manifest)
    for stage in (
        "bdqn_convergence",
        "provider_cross",
        "bdqn_migration",
        "bdqn_hyper_screen",
        "bdqn_hyper_confirm",
        "gp_discovery",
        "gp_confirm",
        "final_test",
    ):
        _require_complete_stage(study, stage)
    root = Path(study["output_dir"])
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path = report_dir / "cross_pairwise_bootstrap.csv"
    build_cross_pairwise_statistics(
        cross_results=root / "bdqn" / "cross_matrix" / "cross_results.csv",
        output_path=pairwise_path,
        seed=int(study["base_seed"]) + 400000,
    )
    migration_statistics_path = report_dir / "migration_pairwise_bootstrap.csv"
    build_migration_pairwise_statistics(
        migration_dir=root / "bdqn" / "migration",
        output_path=migration_statistics_path,
        target_step=max(study["transfer_steps"]),
        seed=int(study["base_seed"]) + 410000,
    )
    figures = {
        "bdqn_convergence": report_dir / "bdqn_convergence.png",
        "provider_cross_heatmap": report_dir / "provider_cross_heatmap.png",
        "migration": report_dir / "migration.png",
        "hyperparameters": report_dir / "hyperparameters.png",
        "gp_convergence": report_dir / "gp_convergence.png",
    }
    _plot_bdqn_convergence(root, figures["bdqn_convergence"])
    _plot_cross_heatmap(root, figures["provider_cross_heatmap"])
    _plot_migration(root, figures["migration"])
    _plot_hyperparameters(root, figures["hyperparameters"])
    _plot_gp_convergence(root, figures["gp_convergence"])
    final_bdqn = json.loads(
        (root / "bdqn" / "final_selection.json").read_text(encoding="utf-8")
    )
    final_gp = load_gp_policy(root / "gp" / "confirm" / "selected" / "gp_policy.json")
    run_count = json.loads(
        (root / "gp" / "confirm" / "selected" / "run_count_convergence.json").read_text(
            encoding="utf-8"
        )
    )
    index = {
        "study_manifest": str(manifest_path),
        "study_manifest_sha256": sha256_file(manifest_path),
        "code_commit_at_initialization": study.get("code_commit", "unknown"),
        "code_commit_at_reporting": _current_git_commit(),
        "scenario_manifest_sha256": {
            name: sha256_file(path) for name, path in study["scenarios"].items()
        },
        "architecture_checkpoint_sha256": study["inputs"]["architecture_checkpoint"][
            "sha256"
        ],
        "final_bdqn": final_bdqn,
        "final_gp": {
            "policy": str(root / "gp" / "confirm" / "selected" / "gp_policy.json"),
            "policy_sha256": sha256_file(
                root / "gp" / "confirm" / "selected" / "gp_policy.json"
            ),
            "population_size": final_gp.artifact.evolution_config["population_size"],
            "generations": final_gp.artifact.evolution_config["generations"],
            "recommended_runs": run_count["recommended_run_count"],
            "expression": final_gp.artifact.expression,
        },
        "raw_result_locations": {
            "bdqn_convergence": str(root / "bdqn" / "convergence"),
            "provider_cross": str(root / "bdqn" / "cross_matrix"),
            "migration": str(root / "bdqn" / "migration"),
            "hyper_screen": str(root / "bdqn" / "hyper_screen"),
            "hyper_confirm": str(root / "bdqn" / "hyper_confirm"),
            "gp_discovery": str(root / "gp" / "discovery"),
            "gp_confirm": str(root / "gp" / "confirm"),
            "final_test": str(root / "final_test"),
        },
    }
    index_path = report_dir / "reproducibility_index.json"
    _write_json(index_path, index)
    report_path = report_dir / "round1_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Round-one GP + BDQN experiment report",
                "",
                f"- Final BDQN route: `{final_bdqn['winner']['route']}`",
                f"- Final BDQN hyperparameters: `{final_bdqn['winner']['config']}`",
                f"- Validation-medoid seed: `{final_bdqn['medoid_seed']}`",
                f"- GP configuration: `{final_gp.artifact.evolution_config['population_size']}×{final_gp.artifact.evolution_config['generations']}`",
                f"- Recommended independent GP runs: `{run_count['recommended_run_count']}`",
                f"- Final GP expression: `{final_gp.artifact.expression}`",
                "",
                "## Figures",
                "",
                *[f"- [{name}]({path.name})" for name, path in figures.items()],
                "",
                "## Statistical and reproducibility artifacts",
                "",
                f"- [Stratified bootstrap and win/tie/loss]({pairwise_path.name})",
                f"- [Migration bootstrap and win/tie/loss]({migration_statistics_path.name})",
                f"- [Reproducibility index]({index_path.name})",
                "- All per-seed raw CSV files remain under the directories listed in the reproducibility index.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    outputs = {
        "report": str(report_path),
        "reproducibility_index": str(index_path),
        "pairwise_statistics": str(pairwise_path),
        "migration_statistics": str(migration_statistics_path),
        **{name: str(path) for name, path in figures.items()},
    }
    _mark_stage(manifest_path, study, "report", "complete", outputs=outputs)
    return outputs


def run_round1_smoke(
    *,
    output_dir: str | Path,
    architecture_checkpoint: str | Path,
    gp_policy: str | Path,
    seed: int = 20260824,
) -> dict[str, Any]:
    """Exercise all provider boundaries, transfer cells, cross-evaluation and GP once."""

    root = Path(output_dir).resolve()
    paths = generate_round1_scenarios(
        root / "scenarios",
        base_seed=int(seed),
        b_train_size=4,
        b_validation_size=4,
        g_train_size=4,
        g_validation_size=4,
        test_iid_size=4,
        test_ood_size=4,
    )
    config = replace(
        baseline_bdqn_config(seed=int(seed), max_env_steps=4, device="cpu"),
        batch_size=1,
        buffer_size=32,
        min_buffer_size=1,
        target_update_interval=2,
    )
    initial = ensure_initial_checkpoint(root, seed=int(seed), config=config)
    convergence_cells: dict[str, dict[int, str]] = {provider: {} for provider in PROVIDER_KINDS}
    checkpoints: dict[str, dict[int, Path]] = {provider: {} for provider in PROVIDER_KINDS}
    for provider in PROVIDER_KINDS:
        cell = root / "bdqn" / "convergence" / provider / f"seed_{int(seed)}"
        train_bdqn_provider_cell(
            output_dir=cell,
            provider_kind=provider,
            source_checkpoint=initial,
            train_manifest=paths["b_train"],
            validation_manifest=paths["b_validation"],
            config=config,
            checkpoint_steps=(0, 2, 4),
            architecture_checkpoint=architecture_checkpoint,
            gp_policy=gp_policy,
        )
        convergence_cells[provider][int(seed)] = str(cell)
        checkpoints[provider][int(seed)] = checkpoint_for_target_step(cell, 4)
    cross = evaluate_provider_cross_matrix(
        output_dir=root / "bdqn" / "cross_matrix",
        jobs=provider_cross_matrix_jobs(
            seeds=[int(seed)],
            checkpoint_by_training_provider=checkpoints,
        ),
        validation_manifest=paths["b_validation"],
        architecture_checkpoint=architecture_checkpoint,
        gp_policy=gp_policy,
        device="cpu",
    )
    transfer_config = replace(
        finetune_bdqn_config(seed=int(seed), max_env_steps=4, device="cpu"),
        batch_size=1,
        buffer_size=32,
        min_buffer_size=1,
        target_update_interval=2,
    )
    transfer_cells: dict[str, str] = {}
    for job in migration_path_jobs(seeds=[int(seed)], t0_checkpoints=checkpoints):
        cell = (
            root
            / "bdqn"
            / "migration"
            / str(job["name"])
            / "H0"
            / f"seed_{int(seed)}"
        )
        train_bdqn_provider_cell(
            output_dir=cell,
            provider_kind=str(job["target_provider"]),
            source_checkpoint=job["source_checkpoint"],
            train_manifest=paths["b_train"],
            validation_manifest=paths["b_validation"],
            config=transfer_config,
            checkpoint_steps=(0, 2, 4),
            architecture_checkpoint=architecture_checkpoint,
            gp_policy=gp_policy,
        )
        transfer_cells[str(job["name"])] = str(cell)
    gp_dir = root / "gp" / "p8_g1"
    run_gp_matrix_job(
        output_dir=gp_dir,
        scheduler_checkpoint=checkpoints["g0"][int(seed)],
        scenario_dir=Path(paths["g_train"]).parent,
        population_size=8,
        generations=1,
        independent_runs=1,
        base_seed=int(seed) + 500000,
        workers=1,
        train_batch_size=4,
        anchor_size=4,
    )
    initial_hashes = {
        provider: sha256_file(
            Path(convergence_cells[provider][int(seed)])
            / "checkpoints"
            / "checkpoint_0.pt"
        )
        for provider in PROVIDER_KINDS
    }
    # Serialized checkpoint bytes may contain container metadata; tensor identity is
    # checked from the loaded state to avoid treating that metadata as model state.
    tensor_hashes = {}
    for provider in PROVIDER_KINDS:
        agent, _ = load_branching_checkpoint(
            Path(convergence_cells[provider][int(seed)])
            / "checkpoints"
            / "checkpoint_0.pt",
            device="cpu",
            load_optimizer=False,
        )
        tensor_hashes[provider] = _branching_parameter_hash(agent)
    result = {
        "schema_version": ROUND1_STUDY_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "status": "complete",
        "code_commit": _current_git_commit(),
        "scenario_registry": str(paths["registry"]),
        "shared_initial_checkpoint": str(initial),
        "provider_checkpoint_file_sha256": initial_hashes,
        "provider_initial_parameter_sha256": tensor_hashes,
        "provider_initial_parameters_identical": len(set(tensor_hashes.values())) == 1,
        "convergence_cells": convergence_cells,
        "cross_outputs": {name: str(path) for name, path in cross.items()},
        "transfer_cells": transfer_cells,
        "gp_policy": str(gp_dir / "gp_policy.json"),
        "gp_policy_sha256": sha256_file(gp_dir / "gp_policy.json"),
        "test_v2_consumed": False,
    }
    if not result["provider_initial_parameters_identical"]:
        raise RuntimeError("smoke study providers did not share identical initial tensors.")
    _write_json(root / "smoke_manifest.json", result)
    return result
