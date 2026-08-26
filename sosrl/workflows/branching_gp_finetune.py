"""Warm-start a branching scheduler under one frozen GP architecture policy.

The workflow in this module is intentionally one-sided: the deployed GP
expression is never changed, while the lower Branching DQN is adapted on the
GP training manifest and selected on the GP validation manifest.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .. import environment as env
from ..gp.artifact import load_gp_policy, sha256_file
from ..gp.evolution import EpisodeOutcome, episode_objective
from ..gp.provider import ArchitectureDecision, GPArchitectureProvider
from ..rl.branching import BranchingDQNAgent
from ..rl.checkpoint import load_branching_checkpoint, save_branching_checkpoint
from ..rl.config import BranchingDQNConfig, default_device
from . import evaluation
from .branching import branching_episode_row, run_branching_episode
from .gp_architecture import load_scenario_manifest
from .scheduler import set_seed


RUN_MANIFEST_SCHEMA_VERSION = 1
SCENARIO_CATEGORIES = (
    "feasible_suboptimal",
    "capacity_tight",
    "missing_capability",
    "redundant_overbudget",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class StratifiedManifestSampler:
    """Deterministic round-robin sampling across the four GP categories."""

    def __init__(self, scenarios: Sequence[dict[str, Any]], *, seed: int):
        grouped = {category: [] for category in SCENARIO_CATEGORIES}
        for scenario in scenarios:
            category = str(scenario.get("category", ""))
            if category not in grouped:
                raise ValueError(f"unknown GP scenario category: {category!r}")
            grouped[category].append(dict(scenario))
        if any(not grouped[category] for category in SCENARIO_CATEGORIES):
            raise ValueError("every GP scenario category must be represented.")
        self._rng = random.Random(int(seed))
        self._grouped = grouped
        self._offsets = {category: 0 for category in SCENARIO_CATEGORIES}
        self._position = 0
        for values in self._grouped.values():
            self._rng.shuffle(values)

    @property
    def position(self) -> int:
        return int(self._position)

    def next_payload(self) -> dict[str, Any]:
        category = SCENARIO_CATEGORIES[self._position % len(SCENARIO_CATEGORIES)]
        values = self._grouped[category]
        offset = self._offsets[category]
        if offset >= len(values):
            self._rng.shuffle(values)
            offset = 0
        payload = dict(values[offset])
        self._offsets[category] = offset + 1
        self._position += 1
        return payload

    def advance(self, episodes: int) -> None:
        for _ in range(int(episodes)):
            self.next_payload()


def prepare_finetune_agent(
    scheduler_checkpoint: str | Path,
    config: BranchingDQNConfig,
) -> tuple[BranchingDQNAgent, dict[str, Any]]:
    """Copy B0 networks into a B1 agent with fresh optimizer and replay."""

    source_agent, checkpoint = load_branching_checkpoint(
        scheduler_checkpoint,
        device=config.device,
        load_optimizer=False,
    )
    agent = BranchingDQNAgent(config)
    agent.q_net.load_state_dict(source_agent.q_net.state_dict())
    agent.target_net.load_state_dict(source_agent.target_net.state_dict())
    agent.learn_step = int(source_agent.learn_step)
    agent.q_net.train()
    agent.target_net.eval()
    return agent, checkpoint


def initialize_run_directory(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    inputs: dict[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Create a non-overwriting run directory or validate an exact resume."""

    destination = Path(output_dir)
    manifest_path = destination / "run_manifest.json"
    if resume:
        if not manifest_path.exists():
            raise FileNotFoundError("resume requested but run_manifest.json is missing.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported B1 run manifest schema.")
        if manifest.get("config") != config or manifest.get("inputs") != inputs:
            raise ValueError("resume configuration or input hashes do not match.")
        verify_input_hashes(manifest["inputs"])
        return manifest

    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config,
        "inputs": inputs,
        "stages": {},
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def verify_input_hashes(inputs: dict[str, Any]) -> None:
    for name, record in inputs.items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            continue
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"input {name!r} is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise RuntimeError(f"input {name!r} changed during the B1 run.")


def update_run_manifest(
    output_dir: str | Path,
    manifest: dict[str, Any],
    *,
    stage: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    record = {"status": str(status), "updated_at": _utc_now()}
    if details:
        record.update(details)
    manifest.setdefault("stages", {})[str(stage)] = record
    manifest["updated_at"] = _utc_now()
    _write_json_atomic(Path(output_dir) / "run_manifest.json", manifest)


def _relative_change(base: float, candidate: float) -> float:
    return (float(base) - float(candidate)) / max(abs(float(base)), 1e-12)


def classify_adaptation(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    delta_j_ci: tuple[float, float],
    delta_makespan_ci: tuple[float, float] = (-float("inf"), float("inf")),
) -> str:
    """Apply the pre-registered B0/B1 decision rule."""

    base_failure = float(baseline["failure_rate"])
    candidate_failure = float(candidate["failure_rate"])
    invalid = int(candidate.get("invalid_action_count", 0))
    invariant = int(candidate.get("provider_invariant_violations", 0))
    j_improvement = _relative_change(baseline["mean_j"], candidate["mean_j"])
    makespan_improvement = _relative_change(
        baseline["mean_success_makespan"],
        candidate["mean_success_makespan"],
    )
    budget_increase = float(candidate["budget_violation_rate"]) - float(
        baseline["budget_violation_rate"]
    )
    base_changes = float(baseline["mean_architecture_changes"])
    change_increase = (
        float(candidate["mean_architecture_changes"]) - base_changes
    ) / max(abs(base_changes), 1.0)

    if candidate_failure > base_failure or invalid or invariant:
        return "reject_b1_revisit_scheduler"
    if j_improvement <= -0.01 and makespan_improvement < 0.01:
        return "reject_b1_revisit_scheduler"

    objective_win = j_improvement >= 0.01 or float(delta_j_ci[1]) < 0.0
    makespan_guard = makespan_improvement >= -0.01
    budget_guard = budget_increase <= 0.02
    change_guard = change_increase <= 0.05
    if objective_win and makespan_guard and budget_guard and change_guard:
        return "accept_b1_no_gp_tuning"
    if makespan_improvement >= 0.01:
        return "accept_b1_consider_gp_tuning"

    ci_crosses_zero = float(delta_j_ci[0]) <= 0.0 <= float(delta_j_ci[1])
    makespan_ci_crosses_zero = (
        float(delta_makespan_ci[0])
        <= 0.0
        <= float(delta_makespan_ci[1])
    )
    if (
        candidate_failure == base_failure
        and abs(j_improvement) < 0.01
        and abs(makespan_improvement) < 0.01
        and ci_crosses_zero
        and makespan_ci_crosses_zero
    ):
        return "inconclusive"
    return "reject_b1_revisit_scheduler"


class TracingGPArchitectureProvider:
    """Record concrete G0 decisions while delegating all policy behavior."""

    def __init__(self, provider: GPArchitectureProvider):
        self.provider = provider
        self.trace: list[dict[str, Any]] = []

    def act(self, mission_env: env.MissionEnv) -> ArchitectureDecision:
        step = len(self.trace)
        decision = self.provider.act(mission_env)
        self.trace.append(
            {
                "step": int(step),
                "kind": decision.action.kind,
                "old_system": decision.action.old_system,
                "new_system": decision.action.new_system,
                "candidate_count": int(decision.candidate_count),
                "score": float(decision.score),
                "valid": bool(decision.valid),
            }
        )
        return decision


def _manifest_paths(scenario_dir: str | Path) -> dict[str, Path]:
    root = Path(scenario_dir).resolve()
    paths = {
        "train_manifest": root / "train.json",
        "validation_manifest": root / "validation.json",
        "test_iid_manifest": root / "test_iid.json",
        "test_ood_manifest": root / "test_ood.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing scenario manifests: " + ", ".join(missing))
    return paths


def collect_input_records(
    *,
    scheduler_checkpoint: str | Path,
    gp_policy: str | Path,
    scenario_dir: str | Path,
) -> dict[str, dict[str, str]]:
    paths = {
        "b0_scheduler": Path(scheduler_checkpoint).resolve(),
        "g0_policy": Path(gp_policy).resolve(),
        **_manifest_paths(scenario_dir),
    }
    records: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"input {name!r} is missing: {path}")
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
    return records


def _scenario_environment(payload: dict[str, Any]) -> env.MissionEnv:
    architecture, mission = evaluation.scenario_from_payload(payload)
    return env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=float(payload.get("budget", 8000.0)),
        refund_rate=float(payload.get("refund_rate", 0.8)),
    )


def _outcome(mission_env: env.MissionEnv, result: dict[str, Any]) -> EpisodeOutcome:
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


def evaluate_scheduler_with_frozen_gp(
    *,
    model: str,
    scheduler_checkpoint: str | Path,
    loaded_gp_policy,
    scenarios: Sequence[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one scheduler without replay writes or network updates."""

    scheduler_path = Path(scheduler_checkpoint).resolve()
    agent, _ = load_branching_checkpoint(
        scheduler_path,
        device=device,
        load_optimizer=False,
    )
    agent.q_net.eval()
    actual_hash = sha256_file(scheduler_path)
    training_scheduler = dict(loaded_gp_policy.artifact.training_scheduler)
    bound_hash = str(training_scheduler.get("checkpoint_sha256", ""))
    binding = "matched" if actual_hash == bound_hash else "diagnostic_crossed"
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode, payload in enumerate(scenarios):
            mission_env = _scenario_environment(payload)
            provider = TracingGPArchitectureProvider(
                GPArchitectureProvider.from_artifact(loaded_gp_policy)
            )
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
                    "scenario_hash": payload["scenario_hash"],
                    "failure_aware_j": episode_objective(
                        _outcome(mission_env, result)
                    ),
                    "g0_bound_scheduler_sha256": bound_hash,
                    "g0_training_scheduler_backend": training_scheduler.get(
                        "kind", "unknown"
                    ),
                    "actual_scheduler_sha256": actual_hash,
                    "checkpoint_binding": binding,
                }
            )
            rows.append(row)
            for trace in provider.trace:
                traces.append(
                    {
                        "model": str(model),
                        "scenario_hash": payload["scenario_hash"],
                        "category": payload.get("category", "evaluation"),
                        **trace,
                    }
                )
    if len(agent.replay) != 0:
        raise RuntimeError("validation evaluation unexpectedly wrote scheduler replay.")
    return rows, traces


def summarize_results(
    model: str,
    rows: Sequence[dict[str, Any]],
    *,
    category: str = "all",
    additional_steps: int = 0,
) -> dict[str, Any]:
    selected = [
        row for row in rows if category == "all" or row["category"] == category
    ]
    if not selected:
        raise ValueError(f"no rows available for category {category!r}.")
    successful = [row for row in selected if bool(row["success"])]
    failures = len(selected) - len(successful)

    def avg(field: str, source=selected) -> float:
        return float(mean(float(row[field]) for row in source))

    return {
        "model": str(model),
        "category": str(category),
        "additional_steps": int(additional_steps),
        "episodes": len(selected),
        "success_count": len(successful),
        "failure_count": failures,
        "success_rate": len(successful) / len(selected),
        "failure_rate": failures / len(selected),
        "dead_end_rate": sum(bool(row["dead_end"]) for row in selected)
        / len(selected),
        "mean_j": avg("failure_aware_j"),
        "mean_success_makespan": (
            avg("makespan", successful) if successful else float("inf")
        ),
        "mean_final_cost": avg("final_net_cost"),
        "mean_peak_cost": avg("peak_net_cost"),
        "mean_gross_charge": avg("gross_charge"),
        "mean_refund": avg("total_refund"),
        "budget_violation_rate": sum(
            bool(row["ever_over_budget"]) for row in selected
        )
        / len(selected),
        "mean_architecture_changes": avg("architecture_changes"),
        "invalid_action_count": sum(
            int(row["invalid_action_count"]) for row in selected
        ),
        "provider_invariant_violations": sum(
            int(row["provider_invariant_violations"]) for row in selected
        ),
        "mean_scheduler_reward": avg("scheduler_reward"),
    }


def summarize_models(
    rows_by_model: dict[str, Sequence[dict[str, Any]]],
    step_by_model: dict[str, int],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model, rows in rows_by_model.items():
        summaries.append(
            summarize_results(
                model,
                rows,
                additional_steps=step_by_model.get(model, 0),
            )
        )
        categories = sorted({str(row["category"]) for row in rows})
        for category in categories:
            summaries.append(
                summarize_results(
                    model,
                    rows,
                    category=category,
                    additional_steps=step_by_model.get(model, 0),
                )
            )
    return summaries


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    samples: int = 2000,
    seed: int = 4,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return (float("nan"), float("nan"))
    if array.size == 1:
        value = float(array[0])
        return (value, value)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(array, size=array.size, replace=True)))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _paired_metric(values: Sequence[float], *, seed: int) -> dict[str, Any]:
    low, high = paired_bootstrap_ci(values, samples=2000, seed=seed)
    return {
        "count": len(values),
        "mean": float(mean(values)) if values else None,
        "ci95_low": None if math.isnan(low) else low,
        "ci95_high": None if math.isnan(high) else high,
    }


def compare_paired_results(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    include_categories: bool = True,
) -> dict[str, Any]:
    baseline = {row["scenario_hash"]: row for row in baseline_rows}
    candidate = {row["scenario_hash"]: row for row in candidate_rows}
    if baseline.keys() != candidate.keys():
        raise ValueError("paired evaluation scenario hashes do not match.")
    pairs = [(baseline[key], candidate[key]) for key in sorted(baseline)]

    def differences(field: str, *, successful_only: bool = False) -> list[float]:
        return [
            float(right[field]) - float(left[field])
            for left, right in pairs
            if not successful_only or (bool(left["success"]) and bool(right["success"]))
        ]

    j_values = differences("failure_aware_j")
    tolerance = 1e-12
    metrics = {
        "delta_j": _paired_metric(j_values, seed=seed),
        "delta_successful_makespan": _paired_metric(
            differences("makespan", successful_only=True), seed=seed + 1
        ),
        "delta_final_cost": _paired_metric(
            differences("final_net_cost"), seed=seed + 2
        ),
        "delta_peak_cost": _paired_metric(
            differences("peak_net_cost"), seed=seed + 3
        ),
        "delta_architecture_changes": _paired_metric(
            differences("architecture_changes"), seed=seed + 4
        ),
        "delta_success_rate": _paired_metric(
            [float(right["success"]) - float(left["success"]) for left, right in pairs],
            seed=seed + 5,
        ),
        "delta_dead_end_rate": _paired_metric(
            [float(right["dead_end"]) - float(left["dead_end"]) for left, right in pairs],
            seed=seed + 6,
        ),
    }
    metrics["j_win_tie_loss"] = {
        "win": sum(value < -tolerance for value in j_values),
        "tie": sum(abs(value) <= tolerance for value in j_values),
        "loss": sum(value > tolerance for value in j_values),
    }
    metrics["by_category"] = {}
    if include_categories:
        categories = sorted({str(row["category"]) for row in baseline_rows})
        for category in categories:
            base_subset = [row for row in baseline_rows if row["category"] == category]
            candidate_subset = [
                row for row in candidate_rows if row["category"] == category
            ]
            metrics["by_category"][category] = compare_paired_results(
                base_subset,
                candidate_subset,
                seed=seed + 100 + categories.index(category),
                include_categories=False,
            )
    return metrics


def compare_action_traces(
    baseline_traces: Sequence[dict[str, Any]],
    candidate_traces: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[tuple[Any, ...]]]] = defaultdict(
        lambda: {"baseline": [], "candidate": []}
    )
    for label, traces in (("baseline", baseline_traces), ("candidate", candidate_traces)):
        for row in traces:
            grouped[row["scenario_hash"]][label].append(
                (row["kind"], row["old_system"], row["new_system"])
            )
    exact = 0
    aligned_matches = 0
    aligned_total = 0
    for traces in grouped.values():
        left = traces["baseline"]
        right = traces["candidate"]
        exact += left == right
        aligned_total += min(len(left), len(right))
        aligned_matches += sum(a == b for a, b in zip(left, right))

    def kind_distribution(traces: Sequence[dict[str, Any]]) -> dict[str, float]:
        counts = Counter(str(row["kind"]) for row in traces)
        total = max(sum(counts.values()), 1)
        return {
            kind: counts.get(kind, 0) / total
            for kind in ("keep", "add", "remove", "replace")
        }

    def system_distribution(rows: Sequence[dict[str, Any]], suffix: str) -> dict[str, float]:
        counts = {
            str(index): sum(int(row.get(f"system_{index}_{suffix}_count", 0)) for row in rows)
            for index in range(env.N)
        }
        total = max(sum(counts.values()), 1)
        return {key: value / total for key, value in counts.items()}

    return {
        "scenario_count": len(grouped),
        "exact_sequence_rate": exact / max(len(grouped), 1),
        "aligned_action_agreement": aligned_matches / max(aligned_total, 1),
        "aligned_step_count": aligned_total,
        "action_kind_distribution": {
            "baseline": kind_distribution(baseline_traces),
            "candidate": kind_distribution(candidate_traces),
        },
        "system_distribution": {
            suffix: {
                "baseline": system_distribution(baseline_rows, suffix),
                "candidate": system_distribution(candidate_rows, suffix),
            }
            for suffix in ("added", "removed", "used")
        },
    }


def _selection_key(summary: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(summary["failure_rate"]),
        float(summary["mean_j"]),
        float(summary["mean_success_makespan"]),
        int(summary["additional_steps"]),
    )


def select_validation_checkpoint(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    overall = [item for item in summaries if item["category"] == "all"]
    baseline = next(item for item in overall if item["model"] == "B0")
    best = min(overall, key=_selection_key)
    if best["model"] != "B0" and _selection_key(best) < _selection_key(baseline):
        return {"selected_model": best["model"], "adaptation_accepted": True}
    return {"selected_model": "B0", "adaptation_accepted": False}


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _training_config(
    *,
    extra_env_steps: int,
    lr: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay: float,
    seed: int,
    device: str,
) -> BranchingDQNConfig:
    return BranchingDQNConfig(
        episodes=max(100000, int(extra_env_steps)),
        max_env_steps=int(extra_env_steps),
        scenario_pool_size=256,
        budget=8000.0,
        refund_rate=0.8,
        gamma=0.99,
        lr=float(lr),
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_interval=250,
        epsilon_start=float(epsilon_start),
        epsilon_end=float(epsilon_end),
        epsilon_decay=float(epsilon_decay),
        seed=int(seed),
        device=str(device),
        log_interval=10,
    )


def _checkpoint_thresholds(
    extra_env_steps: int,
    checkpoint_interval_steps: int,
) -> list[int]:
    if extra_env_steps <= 0 or checkpoint_interval_steps <= 0:
        raise ValueError("step counts must be positive.")
    if extra_env_steps % checkpoint_interval_steps != 0:
        raise ValueError("--extra-env-steps must be divisible by checkpoint interval.")
    return list(
        range(
            int(checkpoint_interval_steps),
            int(extra_env_steps) + 1,
            int(checkpoint_interval_steps),
        )
    )


def _checkpoint_label(threshold: int) -> str:
    return (
        f"{int(threshold) // 1000}k"
        if int(threshold) % 1000 == 0
        else f"{int(threshold)}steps"
    )


def _checkpoint_path(output_dir: Path, threshold: int) -> Path:
    return output_dir / "training" / f"checkpoint_{_checkpoint_label(threshold)}.pt"


def _resume_training_state(
    output_dir: Path,
    thresholds: Sequence[int],
    *,
    device: str,
) -> tuple[BranchingDQNAgent | None, dict[str, Any] | None, Path | None]:
    available = [
        (_checkpoint_path(output_dir, threshold), threshold)
        for threshold in thresholds
        if _checkpoint_path(output_dir, threshold).is_file()
    ]
    if not available:
        return None, None, None
    checkpoint_path, _ = available[-1]
    agent, checkpoint = load_branching_checkpoint(
        checkpoint_path,
        device=device,
        load_optimizer=True,
    )
    state = dict(checkpoint.get("training_state", {}))
    if state.get("stage") != "finetune_branching_with_g0":
        raise ValueError("resume checkpoint has the wrong training stage.")
    return agent, state, checkpoint_path


def train_finetuned_scheduler(
    *,
    output_dir: Path,
    scheduler_checkpoint: Path,
    loaded_gp_policy,
    train_scenarios: Sequence[dict[str, Any]],
    train_manifest_hash: str,
    input_records: dict[str, dict[str, str]],
    config: BranchingDQNConfig,
    thresholds: Sequence[int],
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[int, Path], dict[str, Any]]:
    """Warm-start B0 and save B1 candidates after complete episodes."""

    set_seed(config.seed)
    registered_min_replay = int(config.min_buffer_size)
    output_dir.joinpath("training").mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training" / "training_history.csv"
    sampler = StratifiedManifestSampler(train_scenarios, seed=config.seed)
    replay_reinitialized = False
    resumed_from: Path | None = None
    if resume:
        agent, state, resumed_from = _resume_training_state(
            output_dir,
            thresholds,
            device=config.device,
        )
    else:
        agent, state = None, None

    if agent is None:
        agent, _ = prepare_finetune_agent(scheduler_checkpoint, config)
        epsilon = float(config.epsilon_start)
        total_steps = 0
        episode = 0
        history: list[dict[str, Any]] = []
    else:
        expected = asdict(config)
        actual = asdict(agent.config)
        if actual != expected:
            raise ValueError("resume checkpoint training configuration does not match.")
        expected_state_hashes = {
            "base_scheduler_sha256": input_records["b0_scheduler"]["sha256"],
            "g0_policy_sha256": input_records["g0_policy"]["sha256"],
            "train_manifest_sha256": input_records["train_manifest"]["sha256"],
        }
        if any(state.get(key) != value for key, value in expected_state_hashes.items()):
            raise ValueError("resume checkpoint source hashes do not match the run manifest.")
        epsilon = float(state["epsilon"])
        total_steps = int(state["actual_environment_steps"])
        episode = int(state["episodes"])
        sampler.advance(episode)
        history = [
            row
            for row in _read_csv(history_path)
            if int(float(row.get("total_env_steps", 0))) <= total_steps
        ]
        replay_reinitialized = True
        if len(agent.replay) != 0:
            raise RuntimeError("replay must be empty after checkpoint restore.")

    source_expression = str(loaded_gp_policy.artifact.expression)
    checkpoints = {
        threshold: _checkpoint_path(output_dir, threshold)
        for threshold in thresholds
        if _checkpoint_path(output_dir, threshold).is_file()
    }
    provider = GPArchitectureProvider.from_artifact(loaded_gp_policy)
    while total_steps < int(config.max_env_steps or 0):
        payload = sampler.next_payload()
        mission_env = _scenario_environment(payload)
        used_epsilon = float(epsilon)
        # The first 1,000 new transitions are collection-only.  Requiring
        # 1,001 entries here makes transition number 1,000 ineligible while
        # preserving the registered steady-state min replay value of 1,000.
        agent.config.min_buffer_size = (
            registered_min_replay + 1
            if len(agent.replay) < registered_min_replay
            else registered_min_replay
        )
        result = run_branching_episode(
            mission_env,
            provider,
            agent,
            scheduler_epsilon=used_epsilon,
            update_scheduler=True,
            store_experience=True,
        )
        agent.config.min_buffer_size = registered_min_replay
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
                "scenario_hash": payload["scenario_hash"],
                "next_epsilon": float(epsilon),
                "replay_size": len(agent.replay),
                "learning_enabled": len(agent.replay) > registered_min_replay,
            }
        )
        history.append(row)
        episode += 1
        _write_csv(history_path, history)
        for threshold in thresholds:
            checkpoint_path = _checkpoint_path(output_dir, threshold)
            if total_steps < threshold or checkpoint_path.is_file():
                continue
            training_state = {
                "stage": "finetune_branching_with_g0",
                "target_environment_steps": int(threshold),
                "actual_environment_steps": int(total_steps),
                "episodes": int(episode),
                "epsilon": float(epsilon),
                "base_scheduler_sha256": input_records["b0_scheduler"]["sha256"],
                "g0_policy_sha256": input_records["g0_policy"]["sha256"],
                "train_manifest_sha256": input_records["train_manifest"]["sha256"],
                "train_manifest_hash": str(train_manifest_hash),
                "replay_reinitialized_on_resume": bool(replay_reinitialized),
            }
            save_branching_checkpoint(agent, checkpoint_path, training_state)
            checkpoints[int(threshold)] = checkpoint_path

    if set(checkpoints) != set(thresholds):
        raise RuntimeError("not all requested B1 checkpoints were produced.")
    if str(loaded_gp_policy.artifact.expression) != source_expression:
        raise RuntimeError("G0 expression changed during B1 training.")
    progress = {
        "episodes": int(episode),
        "actual_environment_steps": int(total_steps),
        "epsilon": float(epsilon),
        "replay_reinitialized_on_resume": bool(replay_reinitialized),
        "resumed_from": str(resumed_from) if resumed_from else None,
    }
    return history, checkpoints, progress


def _comparison_rows(
    baseline_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline = {row["scenario_hash"]: row for row in baseline_rows}
    candidate = {row["scenario_hash"]: row for row in candidate_rows}
    rows = []
    for scenario_hash in sorted(baseline):
        left, right = baseline[scenario_hash], candidate[scenario_hash]
        both_success = bool(left["success"]) and bool(right["success"])
        rows.append(
            {
                "scenario_hash": scenario_hash,
                "category": left["category"],
                "baseline_success": left["success"],
                "candidate_success": right["success"],
                "delta_j": float(right["failure_aware_j"])
                - float(left["failure_aware_j"]),
                "delta_makespan_both_success": (
                    float(right["makespan"]) - float(left["makespan"])
                    if both_success
                    else None
                ),
                "delta_final_cost": float(right["final_net_cost"])
                - float(left["final_net_cost"]),
                "delta_peak_cost": float(right["peak_net_cost"])
                - float(left["peak_net_cost"]),
                "delta_architecture_changes": int(right["architecture_changes"])
                - int(left["architecture_changes"]),
            }
        )
    return rows


def _plot_training_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [float(row["total_env_steps"]) for row in history]
    reward = [float(row["scheduler_reward"]) for row in history]
    makespan = [float(row["makespan"]) for row in history]
    loss = [
        float(row["scheduler_loss"])
        if row.get("scheduler_loss") not in (None, "", "None")
        else float("nan")
        for row in history
    ]
    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(steps, reward, linewidth=0.8)
    axes[0].set_ylabel("scheduler reward")
    axes[1].plot(steps, makespan, linewidth=0.8)
    axes[1].set_ylabel("makespan")
    axes[2].plot(steps, loss, linewidth=0.8)
    axes[2].set_ylabel("loss")
    axes[2].set_xlabel("additional assignment steps")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_paired_effects(path: Path, paired: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [
        "delta_j",
        "delta_successful_makespan",
        "delta_final_cost",
        "delta_peak_cost",
        "delta_architecture_changes",
    ]
    labels = ["J", "makespan", "final cost", "peak cost", "arch changes"]
    figure, axes = plt.subplots(1, len(names), figsize=(12, 4.5))
    for axis, name, label in zip(axes, names, labels, strict=True):
        record = paired[name]
        value = float(record["mean"] or 0.0)
        low = float(record["ci95_low"] if record["ci95_low"] is not None else value)
        high = float(record["ci95_high"] if record["ci95_high"] is not None else value)
        color = "#2e7d32" if value < 0 else "#c62828"
        axis.bar([0], [value], color=color, width=0.6)
        axis.vlines(0.0, low, high, color="black", linewidth=1.0)
        axis.hlines((low, high), -0.08, 0.08, color="black", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks([])
        axis.set_title(label)
        axis.set_ylabel("B1 - B0")
        axis.text(
            0.0,
            value,
            f" {value:.3g}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    figure.suptitle("Paired validation effects with 95% bootstrap CI")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    summaries: Sequence[dict[str, Any]],
    selection: dict[str, Any],
    paired: dict[str, Any],
    action_comparison: dict[str, Any],
    historical_summaries: dict[str, dict[str, Any]] | None = None,
) -> None:
    overall = [row for row in summaries if row["category"] == "all"]
    lines = [
        "# G0 固定条件下 B0/B1 微调对比",
        "",
        f"- Validation 选择：`{selection['selected_model']}`",
        f"- 是否接受适配：`{str(selection['adaptation_accepted']).lower()}`",
        f"- 正式判断：`{selection['decision_status']}`",
        f"- 实际配对候选：`{selection['compared_model']}`",
        "- GP 状态：G0 expression 与原 artifact 均未修改；B1 组合仅为 diagnostic crossed binding。",
        "",
        "## Validation 总体结果",
        "",
        "|模型|失败率|mean J|成功 makespan|预算违规率|架构变化|新增步数|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "|{model}|{failure_rate:.4f}|{mean_j:.6f}|{mean_success_makespan:.4f}|"
            "{budget_violation_rate:.4f}|{mean_architecture_changes:.4f}|{additional_steps}|".format(
                **row
            )
        )
    delta_j = paired["delta_j"]
    lines.extend(
        [
            "",
            "## 配对结果",
            "",
            f"- ΔJ 均值：{delta_j['mean']:.6f}，95% CI "
            f"[{delta_j['ci95_low']:.6f}, {delta_j['ci95_high']:.6f}]。",
            f"- J win/tie/loss：{paired['j_win_tie_loss']}。",
            f"- 完整 G0 动作序列一致率：{action_comparison['exact_sequence_rate']:.4f}。",
            f"- 对齐步骤动作一致率：{action_comparison['aligned_action_agreement']:.4f}。",
        ]
    )
    if historical_summaries:
        lines.extend(
            [
                "",
                "## 历史 IID/OOD 描述性复核",
                "",
                "阶段标记：`historical_test_diagnostic`。",
                "",
                "|数据集|模型|失败率|mean J|成功 makespan|预算违规率|架构变化|",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        historical_delta_lines = []
        for split, summary in historical_summaries.items():
            for row in summary["summaries"]:
                if row["category"] != "all":
                    continue
                lines.append(
                    "|{split}|{model}|{failure_rate:.4f}|{mean_j:.6f}|"
                    "{mean_success_makespan:.4f}|{budget_violation_rate:.4f}|"
                    "{mean_architecture_changes:.4f}|".format(split=split.upper(), **row)
                )
            delta = summary["paired"]["delta_j"]
            historical_delta_lines.append(
                f"{split.upper()} ΔJ={delta['mean']:.6f}，95% CI "
                f"[{delta['ci95_low']:.6f}, {delta['ci95_high']:.6f}]。"
            )
        lines.extend(["", *historical_delta_lines])
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "动作轨迹差异只用于解释 B0/B1 如何改变 G0 所见状态，不参与是否微调 G 的正式判定。"
            "旧 IID/OOD 结果仅为历史诊断，也不会回选 checkpoint 或改写 Validation 判断。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_historical_split(
    *,
    output_dir: Path,
    split: str,
    scenarios: Sequence[dict[str, Any]],
    scheduler_paths: dict[str, Path],
    loaded_gp_policy,
    device: str,
    seed: int,
) -> dict[str, Any]:
    split_dir = output_dir / "historical_test" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    traces_by_model: dict[str, list[dict[str, Any]]] = {}
    for model, scheduler_path in scheduler_paths.items():
        rows, traces = evaluate_scheduler_with_frozen_gp(
            model=model,
            scheduler_checkpoint=scheduler_path,
            loaded_gp_policy=loaded_gp_policy,
            scenarios=scenarios,
            device=device,
        )
        rows_by_model[model] = rows
        traces_by_model[model] = traces
    all_rows = [row for rows in rows_by_model.values() for row in rows]
    all_traces = [row for rows in traces_by_model.values() for row in rows]
    _write_csv(split_dir / "results.csv", all_rows)
    _write_csv(split_dir / "action_traces.csv", all_traces)
    summary = {
        "stage": "historical_test_diagnostic",
        "split": split,
        "summaries": summarize_models(
            rows_by_model,
            {model: 0 for model in rows_by_model},
        ),
        "paired": compare_paired_results(
            rows_by_model["B0"],
            rows_by_model["B1"],
            seed=seed,
        ),
    }
    _write_json_atomic(split_dir / "summary.json", summary)
    return summary


def finetune_branching_with_frozen_gp(
    *,
    scheduler_checkpoint: str | Path,
    gp_policy: str | Path,
    scenario_dir: str | Path,
    output_dir: str | Path,
    extra_env_steps: int = 40000,
    checkpoint_interval_steps: int = 10000,
    lr: float = 1e-5,
    epsilon_start: float = 0.10,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.995,
    seed: int = 4,
    device: str = "auto",
    resume: bool = False,
    skip_historical_test: bool = False,
) -> dict[str, Path]:
    """Run the protected G0+B0 -> G0+B1 adaptation and comparison workflow."""

    resolved_device = default_device() if device == "auto" else str(device)
    thresholds = _checkpoint_thresholds(
        int(extra_env_steps), int(checkpoint_interval_steps)
    )
    scheduler_path = Path(scheduler_checkpoint).resolve()
    gp_path = Path(gp_policy).resolve()
    destination = Path(output_dir).resolve()
    input_records = collect_input_records(
        scheduler_checkpoint=scheduler_path,
        gp_policy=gp_path,
        scenario_dir=scenario_dir,
    )
    workflow_config = {
        "extra_env_steps": int(extra_env_steps),
        "checkpoint_interval_steps": int(checkpoint_interval_steps),
        "lr": float(lr),
        "epsilon_start": float(epsilon_start),
        "epsilon_end": float(epsilon_end),
        "epsilon_decay": float(epsilon_decay),
        "seed": int(seed),
        "device": resolved_device,
        "skip_historical_test": bool(skip_historical_test),
        "gamma": 0.99,
        "buffer_size": 50000,
        "batch_size": 64,
        "min_replay": 1000,
        "target_update": 250,
        "gradient_clip": 10.0,
    }
    loaded_gp = load_gp_policy(gp_path)
    manifest = initialize_run_directory(
        destination,
        config=workflow_config,
        inputs=input_records,
        resume=resume,
    )
    original_gp = loaded_gp.artifact.to_dict()
    manifest_paths = _manifest_paths(scenario_dir)
    train_manifest = load_scenario_manifest(manifest_paths["train_manifest"])
    validation_manifest = load_scenario_manifest(manifest_paths["validation_manifest"])
    test_iid_manifest = load_scenario_manifest(manifest_paths["test_iid_manifest"])
    test_ood_manifest = load_scenario_manifest(manifest_paths["test_ood_manifest"])
    if len(train_manifest["scenarios"]) != 256:
        raise ValueError("the B1 training manifest must contain exactly 256 scenarios.")
    category_counts = Counter(row["category"] for row in train_manifest["scenarios"])
    if any(category_counts.get(category, 0) != 64 for category in SCENARIO_CATEGORIES):
        raise ValueError("the B1 training manifest must contain 64 scenarios per category.")

    for path in (
        destination / "baseline",
        destination / "validation",
        destination / "historical_test" / "iid",
        destination / "historical_test" / "ood",
        destination / "report",
    ):
        path.mkdir(parents=True, exist_ok=True)
    update_run_manifest(destination, manifest, stage="baseline", status="running")
    baseline_rows, baseline_traces = evaluate_scheduler_with_frozen_gp(
        model="B0",
        scheduler_checkpoint=scheduler_path,
        loaded_gp_policy=loaded_gp,
        scenarios=validation_manifest["scenarios"],
        device=resolved_device,
    )
    _write_csv(destination / "baseline" / "validation_results.csv", baseline_rows)
    update_run_manifest(
        destination,
        manifest,
        stage="baseline",
        status="complete",
        details={"validation_scenarios": len(baseline_rows)},
    )

    config = _training_config(
        extra_env_steps=extra_env_steps,
        lr=lr,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        seed=seed,
        device=resolved_device,
    )
    update_run_manifest(destination, manifest, stage="training", status="running")
    history, checkpoints, training_progress = train_finetuned_scheduler(
        output_dir=destination,
        scheduler_checkpoint=scheduler_path,
        loaded_gp_policy=loaded_gp,
        train_scenarios=train_manifest["scenarios"],
        train_manifest_hash=train_manifest["manifest_hash"],
        input_records=input_records,
        config=config,
        thresholds=thresholds,
        resume=resume,
    )
    update_run_manifest(
        destination,
        manifest,
        stage="training",
        status="complete",
        details=training_progress,
    )

    update_run_manifest(destination, manifest, stage="validation", status="running")
    rows_by_model: dict[str, list[dict[str, Any]]] = {"B0": baseline_rows}
    traces_by_model: dict[str, list[dict[str, Any]]] = {"B0": baseline_traces}
    model_paths: dict[str, Path] = {"B0": scheduler_path}
    step_by_model = {"B0": 0}
    for threshold in thresholds:
        model = f"B{_checkpoint_label(threshold)}"
        model_paths[model] = checkpoints[threshold]
        step_by_model[model] = int(threshold)
        rows, traces = evaluate_scheduler_with_frozen_gp(
            model=model,
            scheduler_checkpoint=checkpoints[threshold],
            loaded_gp_policy=loaded_gp,
            scenarios=validation_manifest["scenarios"],
            device=resolved_device,
        )
        rows_by_model[model] = rows
        traces_by_model[model] = traces
    summaries = summarize_models(rows_by_model, step_by_model)
    _write_csv(destination / "validation" / "checkpoint_comparison.csv", summaries)
    _write_csv(
        destination / "validation" / "all_results.csv",
        [row for rows in rows_by_model.values() for row in rows],
    )
    _write_csv(
        destination / "validation" / "action_traces.csv",
        [row for rows in traces_by_model.values() for row in rows],
    )
    selection = select_validation_checkpoint(summaries)
    nonbaseline = [
        summary
        for summary in summaries
        if summary["category"] == "all" and summary["model"] != "B0"
    ]
    comparison_model = (
        selection["selected_model"]
        if selection["selected_model"] != "B0"
        else min(nonbaseline, key=_selection_key)["model"]
    )
    paired = compare_paired_results(
        baseline_rows,
        rows_by_model[comparison_model],
        seed=seed,
    )
    paired_rows = _comparison_rows(baseline_rows, rows_by_model[comparison_model])
    _write_csv(destination / "validation" / "paired_results.csv", paired_rows)
    baseline_summary = next(
        row for row in summaries if row["model"] == "B0" and row["category"] == "all"
    )
    candidate_summary = next(
        row
        for row in summaries
        if row["model"] == comparison_model and row["category"] == "all"
    )
    delta_j_ci = (
        float(paired["delta_j"]["ci95_low"]),
        float(paired["delta_j"]["ci95_high"]),
    )
    makespan_ci_record = paired["delta_successful_makespan"]
    delta_makespan_ci = (
        (
            float(makespan_ci_record["ci95_low"]),
            float(makespan_ci_record["ci95_high"]),
        )
        if makespan_ci_record["ci95_low"] is not None
        else (float("nan"), float("nan"))
    )
    decision_status = classify_adaptation(
        baseline_summary,
        candidate_summary,
        delta_j_ci=delta_j_ci,
        delta_makespan_ci=delta_makespan_ci,
    )
    if not selection["adaptation_accepted"] and decision_status.startswith("accept_"):
        decision_status = "inconclusive"
    action_comparison = compare_action_traces(
        baseline_traces,
        traces_by_model[comparison_model],
        baseline_rows,
        rows_by_model[comparison_model],
    )
    selection.update(
        {
            "selected_checkpoint": str(model_paths[selection["selected_model"]]),
            "selected_scheduler_sha256": sha256_file(
                model_paths[selection["selected_model"]]
            ),
            "compared_model": comparison_model,
            "compared_checkpoint": str(model_paths[comparison_model]),
            "decision_status": decision_status,
            "selection_order": [
                "failure_rate",
                "mean_failure_aware_j",
                "mean_successful_makespan",
                "fewer_additional_steps",
            ],
            "paired_statistics": paired,
            "action_trace_comparison": action_comparison,
            "checkpoint_binding": (
                "matched"
                if selection["selected_model"] == "B0"
                else "diagnostic_crossed"
            ),
        }
    )
    _write_json_atomic(destination / "validation" / "selection.json", selection)
    update_run_manifest(
        destination,
        manifest,
        stage="validation",
        status="complete",
        details={
            "selected_model": selection["selected_model"],
            "adaptation_accepted": selection["adaptation_accepted"],
            "decision_status": decision_status,
        },
    )

    historical_status: dict[str, Any]
    historical_summaries: dict[str, dict[str, Any]] = {}
    if skip_historical_test:
        historical_status = {"status": "skipped_by_request"}
    elif not selection["adaptation_accepted"]:
        historical_status = {
            "status": "skipped_no_accepted_b1",
            "reason": "Validation retained B0; no B1 was locked for historical testing.",
        }
    else:
        update_run_manifest(
            destination, manifest, stage="historical_test", status="running"
        )
        historical_schedulers = {
            "B0": scheduler_path,
            "B1": model_paths[selection["selected_model"]],
        }
        historical_summaries["iid"] = _evaluate_historical_split(
            output_dir=destination,
            split="iid",
            scenarios=test_iid_manifest["scenarios"],
            scheduler_paths=historical_schedulers,
            loaded_gp_policy=loaded_gp,
            device=resolved_device,
            seed=seed + 1000,
        )
        historical_summaries["ood"] = _evaluate_historical_split(
            output_dir=destination,
            split="ood",
            scenarios=test_ood_manifest["scenarios"],
            scheduler_paths=historical_schedulers,
            loaded_gp_policy=loaded_gp,
            device=resolved_device,
            seed=seed + 2000,
        )
        historical_status = {
            "status": "complete",
            "stage": "historical_test_diagnostic",
        }
    _write_json_atomic(
        destination / "historical_test" / "diagnostic_status.json",
        historical_status,
    )
    update_run_manifest(
        destination,
        manifest,
        stage="historical_test",
        status=historical_status["status"],
        details={key: value for key, value in historical_status.items() if key != "status"},
    )

    _plot_training_history(destination / "report" / "stage_curves.png", history)
    _plot_paired_effects(destination / "report" / "paired_effects.png", paired)
    _write_report(
        destination / "report" / "b0_b1_summary.md",
        summaries=summaries,
        selection=selection,
        paired=paired,
        action_comparison=action_comparison,
        historical_summaries=historical_summaries,
    )
    if loaded_gp.artifact.to_dict() != original_gp:
        raise RuntimeError("loaded G0 policy changed during the workflow.")
    verify_input_hashes(input_records)
    update_run_manifest(
        destination,
        manifest,
        stage="input_integrity",
        status="complete",
        details={"verified_at": _utc_now()},
    )
    return {
        "run_manifest": destination / "run_manifest.json",
        "selection": destination / "validation" / "selection.json",
        "report": destination / "report" / "b0_b1_summary.md",
        "stage_curves": destination / "report" / "stage_curves.png",
        "paired_effects": destination / "report" / "paired_effects.png",
    }
