"""Resumable Rule-DQN, GP, and Branching-DQN Pareto tuning workflow."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import html
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from ..gp.artifact import load_gp_policy, sha256_file
from ..gp.config import GPArchitectureConfig
from ..rl.config import BranchingDQNConfig, DQNConfig, default_device
from .alternating_stack import _finalize_cell
from .fixed_rule_scheduler import (
    DEFAULT_CHECKPOINT_STEPS,
    evaluate_fixed_rule_scheduler,
    train_fixed_rule_scheduler,
)
from .gp_architecture import (
    _generate_split,
    load_scenario_manifest,
    save_scenario_manifest,
    train_gp_architecture,
)
from .round1_study import (
    ensure_initial_checkpoint,
    evaluate_bdqn_provider_cell,
    train_bdqn_provider_cell,
)
from .tuning_statistics import (
    decide_rule_lr_package,
    paired_difference_ci,
    robust_pareto_selection,
    select_aggregate_checkpoint,
    summarize_rows,
)


TUNING_SCHEMA_VERSION = 1
RULE_CONFIG_NAMES = ("R0", "R1")
GP_CONFIG_NAMES = tuple(f"G-H{index}" for index in range(6))
BDQN_CONFIG_NAMES = tuple(f"B-H{index}" for index in range(6))


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


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return destination
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(destination)
    return destination


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return []
    with source.open(newline="", encoding="utf-8-sig") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _verify_input_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(record["path"])).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen input {label}: {path}")
    if sha256_file(path) != str(record["sha256"]):
        raise ValueError(f"frozen input {label!r} changed: {path}")
    return path


def _default_gp_candidates() -> list[dict[str, Any]]:
    base = {
        "parent_population_fraction": 0.30,
        "crossover_probability": 0.75,
        "mutation_probability": 0.20,
        "reproduction_probability": 0.05,
        "parsimony_coefficient": 0.001,
    }
    changes = {
        "G-H0": {},
        "G-H1": {"parent_population_fraction": 0.15},
        "G-H2": {"parent_population_fraction": 0.50},
        "G-H3": {
            "crossover_probability": 0.60,
            "mutation_probability": 0.35,
        },
        "G-H4": {"parsimony_coefficient": 0.0},
        "G-H5": {"parsimony_coefficient": 0.005},
    }
    return [{"name": name, **base, **change} for name, change in changes.items()]


def _default_bdqn_candidates() -> list[dict[str, Any]]:
    base = {
        "gamma": 0.99,
        "target_update_interval": 250,
        "epsilon_start": 0.10,
        "epsilon_end": 0.02,
        "epsilon_decay": 0.995,
    }
    changes = {
        "B-H0": {},
        "B-H1": {
            "epsilon_start": 0.05,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
        },
        "B-H2": {
            "epsilon_start": 0.20,
            "epsilon_end": 0.05,
            "epsilon_decay": 0.9975,
        },
        "B-H3": {"target_update_interval": 100},
        "B-H4": {"target_update_interval": 1000},
        "B-H5": {"gamma": 0.97},
    }
    return [{"name": name, **base, **change} for name, change in changes.items()]


def create_gp_bdqn_tuning_spec(
    output_path: str | Path,
    *,
    b_scenario_dir: str | Path,
    g_scenario_dir: str | Path,
    base_rule_checkpoint: str | Path,
    base_gp_policy: str | Path,
    base_bdqn_checkpoints: Sequence[str | Path],
    existing_manifests: Sequence[str | Path] = (),
) -> Path:
    """Write the frozen, formal one-day tuning specification."""

    if len(base_bdqn_checkpoints) != 3:
        raise ValueError("exactly three B0 checkpoints are required.")
    b_root = Path(b_scenario_dir).resolve()
    g_root = Path(g_scenario_dir).resolve()
    b_train = _input_record(b_root / "train.json")
    b_validation = _input_record(b_root / "validation.json")
    g_train = _input_record(g_root / "train.json")
    g_validation = _input_record(g_root / "validation.json")
    registered = []
    seen_paths = set()
    for path in (
        b_root / "train.json",
        b_root / "validation.json",
        g_root / "train.json",
        g_root / "validation.json",
        *map(Path, existing_manifests),
    ):
        resolved = Path(path).resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        registered.append(_input_record(resolved))
    spec = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "code_commit": _git_commit(),
        "inputs": {
            "b_train": b_train,
            "b_validation": b_validation,
            "g_train": g_train,
            "g_validation": g_validation,
            "base_rule_checkpoint": _input_record(base_rule_checkpoint),
            "base_gp_policy": _input_record(base_gp_policy),
            "base_bdqn_checkpoints": [
                _input_record(path) for path in base_bdqn_checkpoints
            ],
            "registered_manifests": registered,
        },
        "rule_lr": {
            "seeds": [4, 5, 6],
            "max_env_steps": 200000,
            "checkpoint_steps": list(DEFAULT_CHECKPOINT_STEPS),
            "minimum_relative_improvement": 0.01,
            "configs": [
                {"name": "R0", "lr": 1e-3, "lr_end": None, "lr_decay": 1.0},
                {"name": "R1", "lr": 1e-4, "lr_end": 1e-5, "lr_decay": 0.9975},
            ],
        },
        "gp": {
            "population_size": 120,
            "screen_generations": 20,
            "max_generations": 50,
            "runs": 3,
            "workers": 12,
            "train_batch_size": 16,
            "anchor_size": 64,
            "anchor_interval": 5,
            "anchor_top_k": 10,
            "min_generations": 20,
            "validation_candidates_per_run": 5,
            "base_seed": 20261020,
            "candidates": _default_gp_candidates(),
        },
        "bdqn": {
            "screen_steps": 15000,
            "max_env_steps": 40000,
            "checkpoint_interval": 5000,
            "screen_seeds": [4, 5, 6],
            "confirm_seeds": [7, 8, 9],
            "parallel_jobs": 3,
            "lr": 1e-4,
            "lr_end": 1e-5,
            "lr_decay": 0.9975,
            "batch_size": 64,
            "buffer_size": 50000,
            "min_buffer_size": 1000,
            "candidates": _default_bdqn_candidates(),
        },
        "evaluation": {
            "screen_validation_size": 64,
            "bootstrap_samples": 5000,
            "budget_guard": 0.02,
            "rule_gate_iid": {"size": 256, "seed": 20261001, "ood": False},
            "rule_gate_ood": {"size": 128, "seed": 20261002, "ood": True},
            "gate_iid": {"size": 512, "seed": 20261003, "ood": False},
            "gate_ood": {"size": 256, "seed": 20261004, "ood": True},
            "final_iid": {"size": 1000, "seed": 20261005, "ood": False},
            "final_ood": {"size": 500, "seed": 20261006, "ood": True},
        },
    }
    return _write_json(output_path, spec)


def load_tuning_spec(path: str | Path) -> dict[str, Any]:
    spec = _read_json(path)
    if int(spec.get("schema_version", -1)) != TUNING_SCHEMA_VERSION:
        raise ValueError("unsupported GP/BDQN tuning spec schema.")
    current_commit = _git_commit()
    if current_commit != "unknown" and str(spec.get("code_commit")) != current_commit:
        raise ValueError(
            "tuning spec code commit does not match the running checkout."
        )
    inputs = spec["inputs"]
    for name in (
        "b_train",
        "b_validation",
        "g_train",
        "g_validation",
        "base_rule_checkpoint",
        "base_gp_policy",
    ):
        _verify_input_record(inputs[name], label=name)
    if len(inputs.get("base_bdqn_checkpoints", ())) != 3:
        raise ValueError("tuning spec requires exactly three B0 checkpoints.")
    for index, record in enumerate(inputs["base_bdqn_checkpoints"]):
        _verify_input_record(record, label=f"base_bdqn_checkpoints[{index}]")
    for index, record in enumerate(inputs.get("registered_manifests", ())):
        _verify_input_record(record, label=f"registered_manifests[{index}]")
    if [row["name"] for row in spec["rule_lr"]["configs"]] != list(
        RULE_CONFIG_NAMES
    ):
        raise ValueError("rule learning-rate matrix changed names or order.")
    if [row["name"] for row in spec["gp"]["candidates"]] != list(
        GP_CONFIG_NAMES
    ):
        raise ValueError("GP tuning matrix changed names or order.")
    if [row["name"] for row in spec["bdqn"]["candidates"]] != list(
        BDQN_CONFIG_NAMES
    ):
        raise ValueError("BDQN tuning matrix changed names or order.")
    return spec


def generate_tuning_evaluation_scenarios(
    output_dir: str | Path,
    *,
    registered_manifests: Sequence[str | Path],
    evaluation_spec: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Generate all six hash-disjoint confirmation/gate/final manifests."""

    destination = Path(output_dir).resolve()
    registry_path = destination / "scenario_registry.json"
    if registry_path.is_file():
        registry = _read_json(registry_path)
        outputs = {
            name: Path(record["path"])
            for name, record in registry["generated"].items()
        }
        for name, path in outputs.items():
            if sha256_file(path) != registry["generated"][name]["sha256"]:
                raise ValueError(f"generated tuning scenario {name!r} changed.")
        outputs["registry"] = registry_path
        return outputs

    occupied: set[str] = set()
    registered = []
    for path in registered_manifests:
        manifest = load_scenario_manifest(path)
        hashes = {str(row["scenario_hash"]) for row in manifest["scenarios"]}
        occupied.update(hashes)
        registered.append(
            {
                **_input_record(path),
                "manifest_hash": str(manifest["manifest_hash"]),
                "size": int(manifest["size"]),
            }
        )
    outputs: dict[str, Path] = {}
    generated = {}
    for name in (
        "rule_gate_iid",
        "rule_gate_ood",
        "gate_iid",
        "gate_ood",
        "final_iid",
        "final_ood",
    ):
        definition = evaluation_spec[name]
        scenarios = _generate_split(
            split=name,
            size=int(definition["size"]),
            seed=int(definition["seed"]),
            ood=bool(definition["ood"]),
        )
        hashes = {str(row["scenario_hash"]) for row in scenarios}
        if len(hashes) != len(scenarios) or occupied & hashes:
            raise ValueError(f"generated tuning split {name!r} overlaps prior data.")
        occupied.update(hashes)
        path = save_scenario_manifest(
            destination / f"{name}.json",
            split=name,
            seed=int(definition["seed"]),
            scenarios=scenarios,
        )
        outputs[name] = path
        manifest = load_scenario_manifest(path)
        generated[name] = {
            **_input_record(path),
            "manifest_hash": str(manifest["manifest_hash"]),
            "size": int(manifest["size"]),
            "ood": bool(definition["ood"]),
        }
    outputs["registry"] = _write_json(
        registry_path,
        {
            "schema_version": TUNING_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "registered": registered,
            "generated": generated,
            "all_hashes_unique": True,
        },
    )
    return outputs


def _rule_config(definition: Mapping[str, Any], *, seed: int, device: str) -> DQNConfig:
    return DQNConfig(
        episodes=0,
        scenario_pool_size=0,
        scenario_order="sequential",
        shared_mission=False,
        rule_set="standard",
        selected_system_num=None,
        min_system_num=3,
        max_system_num=22,
        cost_limit=8000.0,
        gamma=0.99,
        lr=float(definition["lr"]),
        lr_end=(
            None if definition.get("lr_end") is None else float(definition["lr_end"])
        ),
        lr_decay=float(definition["lr_decay"]),
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_interval=250,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        hidden_dim=128,
        seed=int(seed),
        device=str(device),
    )


def _checkpoint_path(cell_dir: str | Path, step: int) -> Path:
    label = f"{int(step) // 1000}k" if step and int(step) % 1000 == 0 else str(step)
    return Path(cell_dir) / "checkpoints" / f"checkpoint_{label}.pt"


def run_rule_lr_stage(
    *,
    spec: Mapping[str, Any],
    scenario_paths: Mapping[str, Path],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Train both LR packages and make the preregistered R0/R1 decision."""

    destination = Path(output_dir).resolve()
    selection_path = destination / "rule_lr_selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for record in selection["selected_checkpoints"].values():
            for checkpoint in record.values():
                _verify_input_record(checkpoint, label="completed rule LR checkpoint")
        return selection

    definition = spec["rule_lr"]
    inputs = spec["inputs"]
    cells: dict[str, dict[int, Path]] = {name: {} for name in RULE_CONFIG_NAMES}
    validation_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in RULE_CONFIG_NAMES
    }
    for config_definition in definition["configs"]:
        name = str(config_definition["name"])
        for seed in definition["seeds"]:
            cell = destination / name / f"seed_{int(seed)}"
            train_fixed_rule_scheduler(
                output_dir=cell,
                train_manifest=inputs["b_train"]["path"],
                validation_manifest=inputs["b_validation"]["path"],
                config=_rule_config(config_definition, seed=int(seed), device=device),
                max_env_steps=int(definition["max_env_steps"]),
                checkpoint_steps=tuple(int(step) for step in definition["checkpoint_steps"]),
            )
            cells[name][int(seed)] = cell
            rows = _read_csv(cell / "validation" / "checkpoint_results.csv")
            for row in rows:
                row.update({"config": name, "seed": int(seed)})
            validation_rows[name].extend(rows)

    matched_pair_audit = {}
    for seed in map(int, definition["seeds"]):
        r0_manifest = _read_json(cells["R0"][seed] / "run_manifest.json")
        r1_manifest = _read_json(cells["R1"][seed] / "run_manifest.json")
        r0_history = _read_csv(cells["R0"][seed] / "training_history.csv")
        r1_history = _read_csv(cells["R1"][seed] / "training_history.csv")
        same_initial = (
            r0_manifest.get("initial_parameter_sha256")
            == r1_manifest.get("initial_parameter_sha256")
        )
        common_episodes = min(len(r0_history), len(r1_history))
        same_scenarios = [
            row["scenario_hash"] for row in r0_history[:common_episodes]
        ] == [row["scenario_hash"] for row in r1_history[:common_episodes]]
        same_exploration_stream = [
            row["epsilon"] for row in r0_history[:common_episodes]
        ] == [row["epsilon"] for row in r1_history[:common_episodes]]
        if not (same_initial and same_scenarios and same_exploration_stream):
            raise RuntimeError(
                f"Rule-DQN paired LR invariant failed for seed {seed}."
            )
        matched_pair_audit[str(seed)] = {
            "initial_parameter_sha256": r0_manifest[
                "initial_parameter_sha256"
            ],
            "shared_initial_weights": True,
            "shared_scenario_order": True,
            "shared_epsilon_random_stream": True,
            "common_episode_count": common_episodes,
            "r0_episode_count": len(r0_history),
            "r1_episode_count": len(r1_history),
        }

    aggregate_selection = {
        name: select_aggregate_checkpoint(
            rows,
            samples=int(spec["evaluation"]["bootstrap_samples"]),
            seed=20261010 + index * 100,
        )
        for index, (name, rows) in enumerate(validation_rows.items())
    }
    selected_checkpoints: dict[str, dict[str, dict[str, str]]] = {}
    gate_rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"iid": [], "ood": []} for name in RULE_CONFIG_NAMES
    }
    for name in RULE_CONFIG_NAMES:
        step = int(aggregate_selection[name]["selected_step"])
        selected_checkpoints[name] = {}
        for seed, cell in sorted(cells[name].items()):
            checkpoint = _checkpoint_path(cell, step).resolve()
            selected_checkpoints[name][str(seed)] = _input_record(checkpoint)
            for split, manifest_name in (
                ("iid", "rule_gate_iid"),
                ("ood", "rule_gate_ood"),
            ):
                scenarios = load_scenario_manifest(scenario_paths[manifest_name])[
                    "scenarios"
                ]
                rows = evaluate_fixed_rule_scheduler(
                    scheduler_checkpoint=checkpoint,
                    scenarios=scenarios,
                    device=device,
                    model=f"{name}_seed{seed}",
                )
                for row in rows:
                    row.update({"config": name, "seed": int(seed), "split": split})
                gate_rows[name][split].extend(rows)

    decision = decide_rule_lr_package(
        gate_rows["R0"],
        gate_rows["R1"],
        minimum_relative_improvement=float(definition["minimum_relative_improvement"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20261100,
    )
    winner = str(decision["winner"])
    deployment_seed = str(int(definition["seeds"][0]))
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "winner": winner,
        "performance_estimate": "three-seed aggregate with hierarchical 95% CI",
        "performance_statement": (
            "aggregate means and 95% confidence intervals across three "
            "independent repeats"
        ),
        "deployment_seed": int(deployment_seed),
        "selected_deployment_seed": int(deployment_seed),
        "deployment_checkpoint": selected_checkpoints[winner][deployment_seed],
        "aggregate_checkpoint_selection": aggregate_selection,
        "selected_checkpoints": selected_checkpoints,
        "matched_pair_audit": matched_pair_audit,
        "decision": decision,
    }
    flat_gate_rows = [
        row
        for name in RULE_CONFIG_NAMES
        for split in ("iid", "ood")
        for row in gate_rows[name][split]
    ]
    _write_csv(destination / "validation_results.csv", [*validation_rows["R0"], *validation_rows["R1"]])
    _write_csv(destination / "gate_results.csv", flat_gate_rows)
    _write_json(selection_path, selection)
    return selection


def _evaluate_stack_repeats(
    *,
    gp_policies: Sequence[str | Path],
    bdqn_checkpoints: Sequence[str | Path],
    scenarios: Sequence[dict[str, Any]],
    device: str,
    model: str,
    split: str,
) -> list[dict[str, Any]]:
    if not gp_policies or not bdqn_checkpoints:
        raise ValueError("stack evaluation requires GP and BDQN artifacts.")
    count = max(len(gp_policies), len(bdqn_checkpoints))
    if len(gp_policies) not in {1, count} or len(bdqn_checkpoints) not in {1, count}:
        raise ValueError("stack repeat artifacts must be singleton or equally sized.")
    rows = []
    for index in range(count):
        gp_policy = gp_policies[0 if len(gp_policies) == 1 else index]
        checkpoint = bdqn_checkpoints[0 if len(bdqn_checkpoints) == 1 else index]
        repeat_rows = evaluate_bdqn_provider_cell(
            model=f"{model}_repeat{index + 1}",
            scheduler_checkpoint=checkpoint,
            provider_kind="g0",
            scenarios=scenarios,
            architecture_checkpoint=None,
            gp_policy=gp_policy,
            device=device,
        )
        for row in repeat_rows:
            row.update(
                {
                    "stack": str(model),
                    "repeat": int(index + 1),
                    "split": str(split),
                    "gp_policy_sha256": sha256_file(gp_policy),
                    "bdqn_checkpoint_sha256": sha256_file(checkpoint),
                }
            )
        rows.extend(repeat_rows)
    return rows


def _policy_node_count(path: str | Path) -> int:
    return int(load_gp_policy(path).artifact.node_count)


def _gp_config(
    spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    generations: int,
    repeat_index: int,
) -> GPArchitectureConfig:
    definition = spec["gp"]
    candidate_index = GP_CONFIG_NAMES.index(str(candidate["name"]))
    return GPArchitectureConfig(
        population_size=int(definition["population_size"]),
        generations=int(generations),
        independent_runs=1,
        tournament_size=5,
        elite_count=min(2, int(definition["population_size"]) - 1),
        crossover_probability=float(candidate["crossover_probability"]),
        mutation_probability=float(candidate["mutation_probability"]),
        reproduction_probability=float(candidate["reproduction_probability"]),
        max_height=6,
        max_nodes=40,
        train_batch_size=int(definition["train_batch_size"]),
        anchor_size=int(definition["anchor_size"]),
        anchor_interval=int(definition["anchor_interval"]),
        anchor_top_k=min(
            int(definition["anchor_top_k"]), int(definition["population_size"])
        ),
        convergence_interval=int(definition["anchor_interval"]),
        convergence_threshold=0.01,
        convergence_patience=2,
        convergence_confirmation_windows=1,
        min_generations=min(int(definition["min_generations"]), int(generations)),
        parent_population_fraction=float(candidate["parent_population_fraction"]),
        parsimony_coefficient=float(candidate["parsimony_coefficient"]),
        base_seed=(
            int(definition["base_seed"])
            + candidate_index * 10000
            + int(repeat_index) * 100
        ),
        workers=int(definition["workers"]),
        feature_set="system_delta",
    )


def _nearest_deployment_member(
    paths: Sequence[str | Path],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose a deployment artifact without using it as a performance estimate."""

    summary = summarize_rows(rows, samples=1000, seed=20261200)
    target_m = float(summary["mean_success_makespan"])
    target_c = float(summary["mean_final_cost"])
    by_repeat: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_repeat.setdefault(int(row["repeat"]), []).append(row)
    scored = []
    for index, path in enumerate(paths, 1):
        member = summarize_rows(by_repeat[index], samples=500, seed=20261200 + index)
        distance = (
            abs(float(member["mean_success_makespan"]) - target_m)
            / max(abs(target_m), 1e-12)
            + abs(float(member["mean_final_cost"]) - target_c)
            / max(abs(target_c), 1e-12)
        )
        scored.append((distance, str(Path(path).resolve()), member))
    distance, path, member = min(scored, key=lambda item: (item[0], item[1]))
    return {
        "path": path,
        "sha256": sha256_file(path),
        "selection_role": "deployment-only nearest member to aggregate means",
        "distance_to_aggregate": float(distance),
        "member_metrics": member,
    }


def _two_promoted_candidates(
    selection: Mapping[str, Any],
    *,
    candidate_names: Sequence[str],
) -> list[str]:
    front = [
        name for name in selection["pareto_front"] if name in set(candidate_names)
    ]
    accepted = [
        name
        for name in candidate_names
        if selection["safety"].get(name, {}).get("accepted", False)
    ]
    ranked = sorted(
        accepted,
        key=lambda name: (
            int(selection["summaries"][name]["failure_count"]),
            float(selection["summaries"][name]["mean_success_makespan"]),
            float(selection["summaries"][name]["mean_final_cost"]),
            name,
        ),
    )
    promoted = []
    for name in [*front, *ranked]:
        if name not in promoted:
            promoted.append(name)
        if len(promoted) == 2:
            break
    if len(promoted) != 2:
        raise RuntimeError("fewer than two safe tuning candidates can be promoted.")
    return promoted


def run_gp_screen_stage(
    *,
    spec: Mapping[str, Any],
    base_gp_policy: str | Path,
    base_bdqn_checkpoints: Sequence[str | Path],
    deployment_bdqn_checkpoint: str | Path,
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Screen six GP configurations, extend two, and lock the G1 knee."""

    destination = Path(output_dir).resolve()
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for paths in selection["candidate_policies"].values():
            for record in paths:
                _verify_input_record(record, label="completed GP candidate")
        _verify_input_record(selection["deployment_policy"], label="G1 deployment")
        return selection

    validation = load_scenario_manifest(spec["inputs"]["g_validation"]["path"])[
        "scenarios"
    ]
    candidates = {row["name"]: row for row in spec["gp"]["candidates"]}
    screen_rows: dict[str, list[dict[str, Any]]] = {
        "G-parent": _evaluate_stack_repeats(
            gp_policies=[base_gp_policy],
            bdqn_checkpoints=base_bdqn_checkpoints,
            scenarios=validation,
            device=device,
            model="G-parent",
            split="validation",
        )
    }
    policies: dict[str, list[Path]] = {}
    for name, candidate in candidates.items():
        policies[name] = []
        rows = []
        for repeat_index in range(int(spec["gp"]["runs"])):
            run_dir = destination / "candidates" / name / f"repeat_{repeat_index + 1}"
            outputs = train_gp_architecture(
                scheduler_checkpoint=deployment_bdqn_checkpoint,
                scheduler_backend="branching-dqn",
                scenario_dir=Path(spec["inputs"]["g_train"]["path"]).parent,
                output_dir=run_dir,
                config=_gp_config(
                    spec,
                    candidate,
                    generations=int(spec["gp"]["screen_generations"]),
                    repeat_index=repeat_index,
                ),
                device="cpu",
                parent_gp_policy=base_gp_policy,
                skip_test_evaluation=True,
                validation_candidates_per_run=int(
                    spec["gp"]["validation_candidates_per_run"]
                ),
            )
            policy = outputs["gp_policy"].resolve()
            policies[name].append(policy)
            repeat_rows = _evaluate_stack_repeats(
                gp_policies=[policy],
                bdqn_checkpoints=[base_bdqn_checkpoints[repeat_index]],
                scenarios=validation,
                device=device,
                model=name,
                split="validation",
            )
            for row in repeat_rows:
                row["repeat"] = repeat_index + 1
            rows.extend(repeat_rows)
        screen_rows[name] = rows

    metadata = {
        name: {
            "gp_nodes": int(round(np.mean([_policy_node_count(path) for path in paths]))),
            "bdqn_step": 0,
        }
        for name, paths in policies.items()
    }
    metadata["G-parent"] = {
        "gp_nodes": _policy_node_count(base_gp_policy),
        "bdqn_step": 0,
    }
    screen_selection = robust_pareto_selection(
        screen_rows,
        baseline="G-parent",
        metadata=metadata,
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20261300,
    )
    promoted = _two_promoted_candidates(
        screen_selection, candidate_names=GP_CONFIG_NAMES
    )
    _write_json(destination / "screen_selection.json", {
        **screen_selection,
        "promoted": promoted,
    })
    _write_csv(
        destination / "screen_results.csv",
        [row for rows in screen_rows.values() for row in rows],
    )

    final_rows = {"G-parent": screen_rows["G-parent"]}
    for name in promoted:
        candidate = candidates[name]
        rows = []
        extended_paths = []
        for repeat_index in range(int(spec["gp"]["runs"])):
            run_dir = destination / "candidates" / name / f"repeat_{repeat_index + 1}"
            outputs = train_gp_architecture(
                scheduler_checkpoint=deployment_bdqn_checkpoint,
                scheduler_backend="branching-dqn",
                scenario_dir=Path(spec["inputs"]["g_train"]["path"]).parent,
                output_dir=run_dir,
                config=_gp_config(
                    spec,
                    candidate,
                    generations=int(spec["gp"]["max_generations"]),
                    repeat_index=repeat_index,
                ),
                device="cpu",
                parent_gp_policy=base_gp_policy,
                skip_test_evaluation=True,
                validation_candidates_per_run=int(
                    spec["gp"]["validation_candidates_per_run"]
                ),
            )
            policy = outputs["gp_policy"].resolve()
            extended_paths.append(policy)
            repeat_rows = _evaluate_stack_repeats(
                gp_policies=[policy],
                bdqn_checkpoints=[base_bdqn_checkpoints[repeat_index]],
                scenarios=validation,
                device=device,
                model=name,
                split="validation",
            )
            for row in repeat_rows:
                row["repeat"] = repeat_index + 1
            rows.extend(repeat_rows)
        policies[name] = extended_paths
        final_rows[name] = rows
        metadata[name]["gp_nodes"] = int(
            round(np.mean([_policy_node_count(path) for path in extended_paths]))
        )

    final_selection = robust_pareto_selection(
        final_rows,
        baseline="G-parent",
        metadata=metadata,
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20261400,
    )
    knee = str(final_selection["knee"])
    deployment_policies = {
        name: _nearest_deployment_member(policies[name], final_rows[name])
        for name in promoted
    }
    if knee == "G-parent":
        deployment = {
            **_input_record(base_gp_policy),
            "selection_role": "deployment-only parent selected as Pareto knee",
        }
        selected_hyperparameters = dict(candidates["G-H0"])
    else:
        deployment = deployment_policies[knee]
        selected_hyperparameters = dict(candidates[knee])
    candidate_policies = {
        "G-parent": [_input_record(base_gp_policy)],
        **{
            name: [_input_record(path) for path in policies[name]] for name in promoted
        },
    }
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "performance_estimate": "three independent GP runs with hierarchical 95% CI",
        "screen": screen_selection,
        "promoted": promoted,
        "final": final_selection,
        "knee": knee,
        "selected_hyperparameters": selected_hyperparameters,
        "candidate_policies": candidate_policies,
        "deployment_policies": deployment_policies,
        "deployment_policy": deployment,
    }
    _write_csv(
        destination / "final_results.csv",
        [row for rows in final_rows.values() for row in rows],
    )
    _write_json(selection_path, selection)
    return selection


def _bdqn_config(
    spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    device: str,
) -> BranchingDQNConfig:
    definition = spec["bdqn"]
    return BranchingDQNConfig(
        episodes=max(100000, int(definition["max_env_steps"])),
        max_env_steps=int(definition["max_env_steps"]),
        scenario_pool_size=256,
        budget=8000.0,
        refund_rate=0.8,
        gamma=float(candidate["gamma"]),
        lr=float(definition["lr"]),
        lr_end=float(definition["lr_end"]),
        lr_decay=float(definition["lr_decay"]),
        batch_size=int(definition["batch_size"]),
        buffer_size=int(definition["buffer_size"]),
        min_buffer_size=int(definition["min_buffer_size"]),
        target_update_interval=int(candidate["target_update_interval"]),
        epsilon_start=float(candidate["epsilon_start"]),
        epsilon_end=float(candidate["epsilon_end"]),
        epsilon_decay=float(candidate["epsilon_decay"]),
        seed=int(seed),
        device=str(device),
        log_interval=10,
    )


def _run_bdqn_job(job: Mapping[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    config = BranchingDQNConfig(**job["config"])
    outputs = train_bdqn_provider_cell(
        output_dir=job["cell_dir"],
        provider_kind="g0",
        source_checkpoint=job["source_checkpoint"],
        train_manifest=job["train_manifest"],
        validation_manifest=job["validation_manifest"],
        config=config,
        checkpoint_steps=job["checkpoint_steps"],
        gp_policy=job["gp_policy"],
        stop_after_checkpoint=int(job["stop_after_checkpoint"]),
        reset_learn_step=True,
    )
    return {
        "seed": int(config.seed),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def _dispatch_bdqn_jobs(
    jobs: Sequence[Mapping[str, Any]], *, parallel_jobs: int
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if int(parallel_jobs) <= 1:
        return [_run_bdqn_job(job) for job in jobs]
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=min(int(parallel_jobs), len(jobs)),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_run_bdqn_job, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: int(row["seed"]))


def _stratified_subset(
    scenarios: Sequence[dict[str, Any]], size: int
) -> list[dict[str, Any]]:
    if int(size) >= len(scenarios):
        return list(scenarios)
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in scenarios:
        categories.setdefault(str(row.get("category", "all")), []).append(row)
    for rows in categories.values():
        rows.sort(key=lambda row: str(row["scenario_hash"]))
    selected = []
    ordered_categories = sorted(categories)
    cursor = {name: 0 for name in ordered_categories}
    while len(selected) < int(size):
        progressed = False
        for name in ordered_categories:
            index = cursor[name]
            if index >= len(categories[name]):
                continue
            selected.append(categories[name][index])
            cursor[name] += 1
            progressed = True
            if len(selected) >= int(size):
                break
        if not progressed:
            break
    if len(selected) != int(size):
        raise ValueError("could not construct requested stratified validation subset.")
    return selected


def _bdqn_cells(
    destination: Path, candidate_name: str, seeds: Sequence[int]
) -> dict[int, Path]:
    return {
        int(seed): destination / "candidates" / candidate_name / f"seed_{int(seed)}"
        for seed in seeds
    }


def _train_bdqn_candidate_to(
    *,
    spec: Mapping[str, Any],
    candidate: Mapping[str, Any],
    source_checkpoint: str | Path,
    gp_policy: str | Path,
    destination: Path,
    seeds: Sequence[int],
    threshold: int,
    device: str,
) -> dict[int, Path]:
    definition = spec["bdqn"]
    checkpoints = tuple(
        range(
            0,
            int(definition["max_env_steps"]) + 1,
            int(definition["checkpoint_interval"]),
        )
    )
    cells = _bdqn_cells(destination, str(candidate["name"]), seeds)
    jobs = []
    for seed, cell in cells.items():
        if _checkpoint_path(cell, int(threshold)).is_file():
            continue
        jobs.append(
            {
                "cell_dir": str(cell),
                "source_checkpoint": str(Path(source_checkpoint).resolve()),
                "train_manifest": str(Path(spec["inputs"]["g_train"]["path"])),
                "validation_manifest": str(
                    Path(spec["inputs"]["g_validation"]["path"])
                ),
                "gp_policy": str(Path(gp_policy).resolve()),
                "checkpoint_steps": list(checkpoints),
                "stop_after_checkpoint": int(threshold),
                "config": asdict(
                    _bdqn_config(spec, candidate, seed=seed, device=device)
                ),
            }
        )
    _dispatch_bdqn_jobs(
        jobs, parallel_jobs=int(definition.get("parallel_jobs", 1))
    )
    return cells


def _evaluate_bdqn_cells(
    *,
    cells: Mapping[int, Path],
    step: int,
    gp_policy: str | Path,
    scenarios: Sequence[dict[str, Any]],
    device: str,
    model: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for repeat, (seed, cell) in enumerate(sorted(cells.items()), 1):
        checkpoint = _checkpoint_path(cell, int(step))
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        repeat_rows = evaluate_bdqn_provider_cell(
            model=f"{model}_seed{seed}_{step}",
            scheduler_checkpoint=checkpoint,
            provider_kind="g0",
            scenarios=scenarios,
            architecture_checkpoint=None,
            gp_policy=gp_policy,
            device=device,
        )
        for row in repeat_rows:
            row.update(
                {
                    "stack": model,
                    "seed": int(seed),
                    "repeat": int(repeat),
                    "split": split,
                    "target_environment_steps": int(step),
                }
            )
        rows.extend(repeat_rows)
    return rows


def _initial_bdqn_plateau() -> dict[str, Any]:
    return {
        "stable_windows": 0,
        "provisional_step": None,
        "confirmed_step": None,
        "comparisons": [],
    }


def _update_bdqn_plateau(
    state: Mapping[str, Any],
    *,
    previous_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    current_step: int,
    threshold: float = 0.01,
    min_steps: int = 20000,
    seed: int = 20261500,
) -> tuple[dict[str, Any], bool]:
    updated = json.loads(json.dumps(state))
    comparison = _bdqn_plateau_comparison(
        previous_rows,
        current_rows,
        threshold=threshold,
        samples=1000,
        seed=seed,
    )
    stable = bool(comparison["stable"])
    if stable:
        updated["stable_windows"] = int(updated["stable_windows"]) + 1
    else:
        updated["stable_windows"] = 0
        updated["provisional_step"] = None
    candidate = (
        int(updated["stable_windows"]) >= 2
        and int(current_step) >= int(min_steps)
    )
    if candidate:
        updated["provisional_step"] = int(current_step)
    updated["comparisons"].append(
        {
            "current_step": int(current_step),
            **comparison,
            "stable_windows": int(updated["stable_windows"]),
            "candidate_convergence": bool(candidate),
            "confirmed": False,
        }
    )
    return updated, bool(candidate)


def _bdqn_plateau_comparison(
    reference_rows: Sequence[Mapping[str, Any]],
    current_rows: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 0.01,
    samples: int = 1000,
    seed: int = 20261500,
) -> dict[str, Any]:
    """Compare a checkpoint with its frozen parent for stopping decisions."""

    previous = summarize_rows(reference_rows, samples=samples, seed=seed)
    current = summarize_rows(current_rows, samples=samples, seed=seed + 1)
    makespan = paired_difference_ci(
        reference_rows,
        current_rows,
        "makespan",
        both_success=True,
        samples=samples,
        seed=seed + 2,
    )
    cost = paired_difference_ci(
        reference_rows,
        current_rows,
        "final_net_cost",
        both_success=True,
        samples=samples,
        seed=seed + 3,
    )
    previous_m = float(previous["mean_success_makespan"])
    previous_c = float(previous["mean_final_cost"])
    relative_m = (previous_m - float(current["mean_success_makespan"])) / max(
        abs(previous_m), 1e-12
    )
    relative_c = (previous_c - float(current["mean_final_cost"])) / max(
        abs(previous_c), 1e-12
    )
    credible_m = float(makespan["ci95"][1]) < -float(threshold) * abs(previous_m)
    credible_c = float(cost["ci95"][1]) < -float(threshold) * abs(previous_c)
    stable = (
        int(previous["failure_count"]) == int(current["failure_count"])
        and relative_m < float(threshold)
        and relative_c < float(threshold)
        and not credible_m
        and not credible_c
        and int(current["invalid_action_count"]) == 0
        and int(current["provider_invariant_violations"]) == 0
    )
    return {
        "failure_unchanged": int(previous["failure_count"])
        == int(current["failure_count"]),
        "relative_makespan_improvement": float(relative_m),
        "relative_final_cost_improvement": float(relative_c),
        "credible_makespan_improvement_ge_1pct": bool(credible_m),
        "credible_final_cost_improvement_ge_1pct": bool(credible_c),
        "invalid_action_count": int(current["invalid_action_count"]),
        "provider_invariant_violations": int(
            current["provider_invariant_violations"]
        ),
        "stable": bool(stable),
    }


def _adaptive_full_steps(
    small_rows_by_step: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    stopped_step: int,
) -> list[int]:
    available = sorted(step for step in small_rows_by_step if step <= int(stopped_step))
    selected = {0, 15000, int(stopped_step)}
    selected.update(step for step in (20000, 30000, 40000) if step in available)
    summaries = {
        step: summarize_rows(rows, samples=500, seed=20261600 + step)
        for step, rows in small_rows_by_step.items()
    }
    for middle in (25000, 35000):
        lower = middle - 5000
        upper = middle + 5000
        if not all(step in summaries for step in (lower, middle, upper)):
            continue
        if (
            float(summaries[middle]["mean_success_makespan"])
            < min(
                float(summaries[lower]["mean_success_makespan"]),
                float(summaries[upper]["mean_success_makespan"]),
            )
            and float(summaries[middle]["mean_final_cost"])
            < min(
                float(summaries[lower]["mean_final_cost"]),
                float(summaries[upper]["mean_final_cost"]),
            )
        ):
            selected.add(middle)
    return sorted(step for step in selected if step in available)


def run_bdqn_screen_stage(
    *,
    spec: Mapping[str, Any],
    base_bdqn_checkpoints: Sequence[str | Path],
    deployment_bdqn_checkpoint: str | Path,
    gp_policy: str | Path,
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Screen six BDQN configurations and retain two aggregate candidates."""

    destination = Path(output_dir).resolve()
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for records in selection["candidate_checkpoints"].values():
            for record in records:
                _verify_input_record(record, label="completed BDQN candidate")
        return selection

    definition = spec["bdqn"]
    frozen_gp_sha256 = sha256_file(gp_policy)
    seeds = [int(seed) for seed in definition["screen_seeds"]]
    candidates = {row["name"]: row for row in definition["candidates"]}
    validation = load_scenario_manifest(spec["inputs"]["g_validation"]["path"])[
        "scenarios"
    ]
    small = _stratified_subset(
        validation, int(spec["evaluation"]["screen_validation_size"])
    )
    screen_step = int(definition["screen_steps"])
    small_rows: dict[str, dict[int, list[dict[str, Any]]]] = {}
    full_screen_rows: dict[str, list[dict[str, Any]]] = {}

    for name, candidate in candidates.items():
        cells = _train_bdqn_candidate_to(
            spec=spec,
            candidate=candidate,
            source_checkpoint=deployment_bdqn_checkpoint,
            gp_policy=gp_policy,
            destination=destination,
            seeds=seeds,
            threshold=screen_step,
            device=device,
        )
        small_rows[name] = {}
        for step in range(0, screen_step + 1, int(definition["checkpoint_interval"])):
            small_rows[name][step] = _evaluate_bdqn_cells(
                cells=cells,
                step=step,
                gp_policy=gp_policy,
                scenarios=small,
                device=device,
                model=name,
                split="screen64",
            )
        full_screen_rows[name] = _evaluate_bdqn_cells(
            cells=cells,
            step=screen_step,
            gp_policy=gp_policy,
            scenarios=validation,
            device=device,
            model=name,
            split="validation",
        )

    parent_rows = _evaluate_stack_repeats(
        gp_policies=[gp_policy],
        bdqn_checkpoints=base_bdqn_checkpoints,
        scenarios=validation,
        device=device,
        model="B-parent",
        split="validation",
    )
    screen_selection = robust_pareto_selection(
        {"B-parent": parent_rows, **full_screen_rows},
        baseline="B-parent",
        metadata={
            "B-parent": {
                "gp_nodes": _policy_node_count(gp_policy),
                "bdqn_step": 0,
            },
            **{
                name: {
                    "gp_nodes": _policy_node_count(gp_policy),
                    "bdqn_step": screen_step,
                }
                for name in candidates
            },
        },
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20261700,
    )
    promoted = _two_promoted_candidates(
        screen_selection, candidate_names=BDQN_CONFIG_NAMES
    )
    _write_json(
        destination / "screen_selection.json",
        {**screen_selection, "promoted": promoted},
    )
    _write_csv(
        destination / "screen_results.csv",
        [parent_row for parent_row in parent_rows]
        + [row for rows in full_screen_rows.values() for row in rows],
    )

    for name, candidate in candidates.items():
        if name in promoted:
            continue
        for cell in _bdqn_cells(destination, name, seeds).values():
            manifest = _read_json(cell / "cell_manifest.json")
            if manifest.get("status") != "complete":
                _finalize_cell(
                    cell,
                    stopped_step=screen_step,
                    stop_reason="pruned_at_screen",
                )

    convergence_records = {}
    full_rows_by_candidate_step: dict[str, dict[int, list[dict[str, Any]]]] = {}
    candidate_checkpoints: dict[str, list[dict[str, str]]] = {}
    deployment_checkpoints: dict[str, dict[str, Any]] = {}
    step_selections = {}
    for name in promoted:
        candidate = candidates[name]
        cells = _bdqn_cells(destination, name, seeds)
        convergence_path = destination / "candidates" / name / "convergence.json"
        convergence = (
            _read_json(convergence_path)
            if convergence_path.is_file()
            else _initial_bdqn_plateau()
        )
        stopped_step = int(convergence.get("actual_environment_steps", screen_step))
        for completed_step in range(
            screen_step + int(definition["checkpoint_interval"]),
            stopped_step + 1,
            int(definition["checkpoint_interval"]),
        ):
            small_rows[name][completed_step] = _evaluate_bdqn_cells(
                cells=cells,
                step=completed_step,
                gp_policy=gp_policy,
                scenarios=small,
                device=device,
                model=name,
                split="screen64",
            )
        parent_small_rows = small_rows[name][0]
        full_rows_by_candidate_step[name] = {
            screen_step: full_screen_rows[name]
        }
        if convergence.get("confirmed_step") is None and stopped_step < int(
            definition["max_env_steps"]
        ):
            for threshold in range(
                max(screen_step + int(definition["checkpoint_interval"]), stopped_step + int(definition["checkpoint_interval"])),
                int(definition["max_env_steps"]) + 1,
                int(definition["checkpoint_interval"]),
            ):
                cells = _train_bdqn_candidate_to(
                    spec=spec,
                    candidate=candidate,
                    source_checkpoint=deployment_bdqn_checkpoint,
                    gp_policy=gp_policy,
                    destination=destination,
                    seeds=seeds,
                    threshold=threshold,
                    device=device,
                )
                current_rows = _evaluate_bdqn_cells(
                    cells=cells,
                    step=threshold,
                    gp_policy=gp_policy,
                    scenarios=small,
                    device=device,
                    model=name,
                    split="screen64",
                )
                small_rows[name][threshold] = current_rows
                convergence, candidate_convergence = _update_bdqn_plateau(
                    convergence,
                    previous_rows=parent_small_rows,
                    current_rows=current_rows,
                    current_step=threshold,
                    threshold=0.01,
                    min_steps=20000,
                    seed=20261800 + threshold + promoted.index(name) * 100,
                )
                stopped_step = threshold
                confirmed = False
                if candidate_convergence:
                    if 0 not in full_rows_by_candidate_step[name]:
                        full_rows_by_candidate_step[name][0] = _evaluate_bdqn_cells(
                            cells=cells,
                            step=0,
                            gp_policy=gp_policy,
                            scenarios=validation,
                            device=device,
                            model=f"{name}@0",
                            split="validation",
                        )
                    full_rows_by_candidate_step[name][threshold] = (
                        _evaluate_bdqn_cells(
                            cells=cells,
                            step=threshold,
                            gp_policy=gp_policy,
                            scenarios=validation,
                            device=device,
                            model=f"{name}@{threshold}",
                            split="validation",
                        )
                    )
                    confirmation = _bdqn_plateau_comparison(
                        full_rows_by_candidate_step[name][0],
                        full_rows_by_candidate_step[name][threshold],
                        threshold=0.01,
                        samples=int(spec["evaluation"]["bootstrap_samples"]),
                        seed=20261850 + threshold + promoted.index(name) * 100,
                    )
                    convergence["full_validation_confirmation"] = {
                        "step": int(threshold),
                        **confirmation,
                    }
                    confirmed = bool(confirmation["stable"])
                    convergence["comparisons"][-1]["confirmed"] = confirmed
                    if confirmed:
                        convergence["confirmed_step"] = int(threshold)
                    else:
                        convergence["stable_windows"] = 0
                        convergence["provisional_step"] = None
                convergence.update(
                    {
                        "actual_environment_steps": int(stopped_step),
                        "max_environment_steps": int(definition["max_env_steps"]),
                        "status": "converged" if confirmed else "running",
                    }
                )
                _write_json(convergence_path, convergence)
                if confirmed:
                    break
        stop_reason = (
            "converged"
            if convergence.get("confirmed_step") is not None
            else "max_env_steps_reached"
        )
        convergence.update(
            {
                "status": stop_reason,
                "actual_environment_steps": int(stopped_step),
                "max_environment_steps": int(definition["max_env_steps"]),
            }
        )
        _write_json(convergence_path, convergence)
        convergence_records[name] = convergence
        for cell in cells.values():
            manifest = _read_json(cell / "cell_manifest.json")
            if manifest.get("status") != "complete":
                _finalize_cell(
                    cell,
                    stopped_step=stopped_step,
                    stop_reason=stop_reason,
                )

        full_steps = _adaptive_full_steps(
            small_rows[name], stopped_step=stopped_step
        )
        step_candidates = {}
        for step in full_steps:
            rows = full_rows_by_candidate_step[name].get(step)
            if rows is None:
                rows = _evaluate_bdqn_cells(
                    cells=cells,
                    step=step,
                    gp_policy=gp_policy,
                    scenarios=validation,
                    device=device,
                    model=f"{name}@{step}",
                    split="validation",
                )
            full_rows_by_candidate_step[name][step] = rows
            step_candidates[f"{name}@{step}"] = rows
        baseline_name = f"{name}@0"
        step_selection = robust_pareto_selection(
            step_candidates,
            baseline=baseline_name,
            metadata={
                candidate_name: {
                    "gp_nodes": _policy_node_count(gp_policy),
                    "bdqn_step": int(candidate_name.rsplit("@", 1)[1]),
                }
                for candidate_name in step_candidates
            },
            budget_guard=float(spec["evaluation"]["budget_guard"]),
            samples=int(spec["evaluation"]["bootstrap_samples"]),
            seed=20261900 + promoted.index(name) * 100,
        )
        selected_step = int(str(step_selection["knee"]).rsplit("@", 1)[1])
        step_selection["selected_step"] = selected_step
        step_selections[name] = step_selection
        paths = [
            _checkpoint_path(cells[seed], selected_step).resolve()
            for seed in sorted(cells)
        ]
        candidate_checkpoints[name] = [_input_record(path) for path in paths]
        deployment_checkpoints[name] = _nearest_deployment_member(
            paths, full_rows_by_candidate_step[name][selected_step]
        )

    _write_csv(
        destination / "small_validation_results.csv",
        [
            row
            for name_rows in small_rows.values()
            for step_rows in name_rows.values()
            for row in step_rows
        ],
    )
    _write_csv(
        destination / "full_validation_results.csv",
        [
            row
            for name_rows in full_rows_by_candidate_step.values()
            for step_rows in name_rows.values()
            for row in step_rows
        ],
    )
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "performance_estimate": "three-seed aggregate with hierarchical 95% CI",
        "screen": screen_selection,
        "promoted": promoted,
        "convergence": convergence_records,
        "step_selection": step_selections,
        "candidate_hyperparameters": {
            name: dict(candidates[name]) for name in promoted
        },
        "candidate_checkpoints": candidate_checkpoints,
        "deployment_checkpoints": deployment_checkpoints,
        "frozen_gp_sha256_before": frozen_gp_sha256,
        "frozen_gp_sha256_after": sha256_file(gp_policy),
    }
    if selection["frozen_gp_sha256_after"] != frozen_gp_sha256:
        raise RuntimeError("BDQN screening mutated its frozen GP policy.")
    _write_json(selection_path, selection)
    return selection


def _b0_config(*, seed: int, device: str, max_env_steps: int = 80000) -> BranchingDQNConfig:
    return BranchingDQNConfig(
        episodes=max(100000, int(max_env_steps)),
        max_env_steps=int(max_env_steps),
        scenario_pool_size=256,
        budget=8000.0,
        refund_rate=0.8,
        gamma=0.99,
        lr=1e-4,
        lr_end=1e-5,
        lr_decay=0.9975,
        batch_size=64,
        buffer_size=50000,
        min_buffer_size=1000,
        target_update_interval=250,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=int(seed),
        device=str(device),
        log_interval=10,
    )


def prepare_s0_prefix(
    *,
    spec: Mapping[str, Any],
    rule_selection: Mapping[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Reuse R0 lineage or retrain Rule-DQN -> G0 -> B0 after an R1 win."""

    destination = Path(output_dir).resolve()
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        _verify_input_record(selection["gp_policy"], label="completed G0 policy")
        for record in selection["bdqn_checkpoints"]:
            _verify_input_record(record, label="completed B0 checkpoint")
        _verify_input_record(
            selection["deployment_bdqn_checkpoint"], label="completed B0 deployment"
        )
        return selection

    if str(rule_selection["winner"]) == "R0":
        selection = {
            "schema_version": TUNING_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "rule_lr_winner": "R0",
            "prefix_retrained": False,
            "rule_checkpoint": dict(spec["inputs"]["base_rule_checkpoint"]),
            "gp_policy": dict(spec["inputs"]["base_gp_policy"]),
            "bdqn_checkpoints": [
                dict(record) for record in spec["inputs"]["base_bdqn_checkpoints"]
            ],
            "deployment_bdqn_checkpoint": dict(
                spec["inputs"]["base_bdqn_checkpoints"][0]
            ),
            "performance_estimate": "existing three-seed B0 group; no prefix retraining",
        }
        _write_json(selection_path, selection)
        return selection

    rule_checkpoint = _verify_input_record(
        rule_selection["deployment_checkpoint"], label="R1 deployment checkpoint"
    )
    gp_definition = spec["gp"]
    g0_dir = destination / "g0"
    g0_config = GPArchitectureConfig(
        population_size=int(gp_definition["population_size"]),
        generations=int(gp_definition["max_generations"]),
        independent_runs=3,
        tournament_size=5,
        elite_count=min(2, int(gp_definition["population_size"]) - 1),
        crossover_probability=0.75,
        mutation_probability=0.20,
        reproduction_probability=0.05,
        max_height=6,
        max_nodes=40,
        train_batch_size=int(gp_definition["train_batch_size"]),
        anchor_size=int(gp_definition["anchor_size"]),
        anchor_interval=int(gp_definition["anchor_interval"]),
        anchor_top_k=min(
            int(gp_definition["anchor_top_k"]), int(gp_definition["population_size"])
        ),
        convergence_interval=int(gp_definition["anchor_interval"]),
        convergence_threshold=0.01,
        convergence_patience=2,
        convergence_confirmation_windows=1,
        min_generations=min(
            int(gp_definition["min_generations"]),
            int(gp_definition["max_generations"]),
        ),
        parent_population_fraction=0.30,
        parsimony_coefficient=0.001,
        base_seed=int(gp_definition["base_seed"]) - 100000,
        workers=int(gp_definition["workers"]),
        feature_set="system_delta",
    )
    g0_outputs = train_gp_architecture(
        scheduler_checkpoint=rule_checkpoint,
        scheduler_backend="rule-dqn",
        scenario_dir=Path(spec["inputs"]["g_train"]["path"]).parent,
        output_dir=g0_dir,
        config=g0_config,
        device="cpu",
        skip_test_evaluation=True,
    )
    g0_policy = g0_outputs["gp_policy"].resolve()

    b0_root = destination / "b0"
    b0_steps = (0, 10000, 20000, 30000, 40000, 60000, 80000)
    validation_rows = []
    cells = {}
    for seed in (1, 2, 3):
        config = _b0_config(seed=seed, device=device)
        initial = ensure_initial_checkpoint(b0_root, seed=seed, config=config)
        cell = b0_root / "cells" / f"seed_{seed}"
        train_bdqn_provider_cell(
            output_dir=cell,
            provider_kind="g0",
            source_checkpoint=initial,
            train_manifest=spec["inputs"]["g_train"]["path"],
            validation_manifest=spec["inputs"]["g_validation"]["path"],
            config=config,
            checkpoint_steps=b0_steps,
            gp_policy=g0_policy,
            reset_learn_step=True,
        )
        cells[seed] = cell
        rows = _read_csv(cell / "validation" / "checkpoint_results.csv")
        for row in rows:
            row.update({"seed": seed, "repeat": seed})
        validation_rows.extend(rows)
    by_step = {}
    for step in b0_steps:
        by_step[f"B0@{step}"] = [
            row
            for row in validation_rows
            if int(float(row["target_environment_steps"])) == step
        ]
    step_selection = robust_pareto_selection(
        by_step,
        baseline="B0@0",
        metadata={
            name: {"gp_nodes": _policy_node_count(g0_policy), "bdqn_step": int(name[3:])}
            for name in by_step
        },
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20262000,
    )
    selected_step = int(str(step_selection["knee"]).split("@", 1)[1])
    paths = [
        _checkpoint_path(cells[seed], selected_step).resolve() for seed in sorted(cells)
    ]
    selected_rows = by_step[f"B0@{selected_step}"]
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "rule_lr_winner": "R1",
        "prefix_retrained": True,
        "rule_checkpoint": _input_record(rule_checkpoint),
        "gp_policy": _input_record(g0_policy),
        "b0_step_selection": step_selection,
        "selected_b0_step": selected_step,
        "bdqn_checkpoints": [_input_record(path) for path in paths],
        "deployment_bdqn_checkpoint": _nearest_deployment_member(paths, selected_rows),
        "performance_estimate": "three-seed B0 aggregate with hierarchical 95% CI",
    }
    _write_csv(b0_root / "aggregate_validation_results.csv", validation_rows)
    _write_json(selection_path, selection)
    return selection


def run_crossplay_stage(
    *,
    spec: Mapping[str, Any],
    scenario_paths: Mapping[str, Path],
    prefix: Mapping[str, Any],
    gp_selection: Mapping[str, Any],
    bdqn_selection: Mapping[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate the preregistered 3x3 matched-repeat cross-play matrix."""

    destination = Path(output_dir).resolve()
    selection_path = destination / "pareto_front.json"
    if selection_path.is_file():
        return _read_json(selection_path)
    gp_groups: dict[str, list[str]] = {
        "G-parent": [str(prefix["gp_policy"]["path"])],
    }
    for name in gp_selection["promoted"]:
        gp_groups[name] = [
            str(record["path"]) for record in gp_selection["candidate_policies"][name]
        ]
    bdqn_groups: dict[str, list[str]] = {
        "B-parent": [str(record["path"]) for record in prefix["bdqn_checkpoints"]],
    }
    for name in bdqn_selection["promoted"]:
        bdqn_groups[name] = [
            str(record["path"])
            for record in bdqn_selection["candidate_checkpoints"][name]
        ]
    rows_by_stack = {}
    all_rows = []
    for gp_name, gp_paths in gp_groups.items():
        for bdqn_name, bdqn_paths in bdqn_groups.items():
            stack = f"{gp_name}+{bdqn_name}"
            rows = []
            for split, manifest_name in (("iid", "gate_iid"), ("ood", "gate_ood")):
                scenarios = load_scenario_manifest(scenario_paths[manifest_name])[
                    "scenarios"
                ]
                rows.extend(
                    _evaluate_stack_repeats(
                        gp_policies=gp_paths,
                        bdqn_checkpoints=bdqn_paths,
                        scenarios=scenarios,
                        device=device,
                        model=stack,
                        split=split,
                    )
                )
            rows_by_stack[stack] = rows
            all_rows.extend(rows)
    baseline = "G-parent+B-parent"
    metadata = {}
    for stack in rows_by_stack:
        gp_name, bdqn_name = stack.split("+", 1)
        gp_path = gp_groups[gp_name][0]
        bdqn_step = (
            0
            if bdqn_name == "B-parent"
            else int(bdqn_selection["step_selection"][bdqn_name]["selected_step"])
        )
        metadata[stack] = {
            "gp_nodes": _policy_node_count(gp_path),
            "bdqn_step": bdqn_step,
        }
    pareto = robust_pareto_selection(
        rows_by_stack,
        baseline=baseline,
        metadata=metadata,
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20262100,
    )
    selected_gp, selected_bdqn = str(pareto["knee"]).split("+", 1)
    if selected_gp == "G-parent":
        deployment_gp = dict(prefix["gp_policy"])
        gp_hyperparameters = next(
            row for row in spec["gp"]["candidates"] if row["name"] == "G-H0"
        )
    else:
        deployment_gp = dict(gp_selection["deployment_policies"][selected_gp])
        gp_hyperparameters = next(
            row for row in spec["gp"]["candidates"] if row["name"] == selected_gp
        )
    if selected_bdqn == "B-parent":
        deployment_bdqn = dict(prefix["deployment_bdqn_checkpoint"])
        bdqn_hyperparameters = next(
            row for row in spec["bdqn"]["candidates"] if row["name"] == "B-H0"
        )
    else:
        deployment_bdqn = dict(
            bdqn_selection["deployment_checkpoints"][selected_bdqn]
        )
        bdqn_hyperparameters = next(
            row for row in spec["bdqn"]["candidates"] if row["name"] == selected_bdqn
        )
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        **pareto,
        "all_stacks": sorted(rows_by_stack),
        "selection_basis": (
            "aggregate means and paired 95% confidence intervals across "
            "preregistered repeats"
        ),
        "selected_gp": selected_gp,
        "selected_bdqn": selected_bdqn,
        "selected_gp_hyperparameters": dict(gp_hyperparameters),
        "selected_bdqn_hyperparameters": dict(bdqn_hyperparameters),
        "deployment_gp_policy": deployment_gp,
        "deployment_bdqn_checkpoint": deployment_bdqn,
        "gp_groups": gp_groups,
        "bdqn_groups": bdqn_groups,
        "pairing_rule": "repeat i GP policy is paired with repeat i BDQN checkpoint",
    }
    _write_csv(destination / "results.csv", all_rows)
    _write_json(selection_path, selection)
    return selection


def run_g2_confirmation_stage(
    *,
    spec: Mapping[str, Any],
    crossplay: Mapping[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for record in selection["policies"]:
            _verify_input_record(record, label="completed G2 policy")
        _verify_input_record(selection["deployment_policy"], label="G2 deployment")
        return selection
    candidate = dict(crossplay["selected_gp_hyperparameters"])
    parent = str(crossplay["deployment_gp_policy"]["path"])
    scheduler = str(crossplay["deployment_bdqn_checkpoint"]["path"])
    validation = load_scenario_manifest(spec["inputs"]["g_validation"]["path"])[
        "scenarios"
    ]
    b1_paths = crossplay["bdqn_groups"][crossplay["selected_bdqn"]]
    policies = []
    rows = []
    for repeat_index in range(int(spec["gp"]["runs"])):
        config = _gp_config(
            spec,
            candidate,
            generations=int(spec["gp"]["max_generations"]),
            repeat_index=repeat_index,
        )
        config = replace(config, base_seed=int(config.base_seed) + 900000)
        outputs = train_gp_architecture(
            scheduler_checkpoint=scheduler,
            scheduler_backend="branching-dqn",
            scenario_dir=Path(spec["inputs"]["g_train"]["path"]).parent,
            output_dir=destination / f"repeat_{repeat_index + 1}",
            config=config,
            device="cpu",
            parent_gp_policy=parent,
            skip_test_evaluation=True,
            validation_candidates_per_run=int(
                spec["gp"]["validation_candidates_per_run"]
            ),
        )
        policy = outputs["gp_policy"].resolve()
        policies.append(policy)
        repeat_rows = _evaluate_stack_repeats(
            gp_policies=[policy],
            bdqn_checkpoints=[b1_paths[repeat_index]],
            scenarios=validation,
            device=device,
            model="G2",
            split="validation",
        )
        for row in repeat_rows:
            row["repeat"] = repeat_index + 1
        rows.extend(repeat_rows)
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "parent_policy": _input_record(parent),
        "frozen_scheduler": _input_record(scheduler),
        "hyperparameters": candidate,
        "policies": [_input_record(path) for path in policies],
        "deployment_policy": _nearest_deployment_member(policies, rows),
        "aggregate_metrics": summarize_rows(
            rows,
            samples=int(spec["evaluation"]["bootstrap_samples"]),
            seed=20262200,
        ),
    }
    _write_csv(destination / "validation_results.csv", rows)
    _write_json(selection_path, selection)
    return selection


def run_b2_confirmation_stage(
    *,
    spec: Mapping[str, Any],
    crossplay: Mapping[str, Any],
    g2: Mapping[str, Any],
    output_dir: str | Path,
    device: str,
) -> dict[str, Any]:
    destination = Path(output_dir).resolve()
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for record in selection["checkpoints"]:
            _verify_input_record(record, label="completed B2 checkpoint")
        return selection
    candidate = {**dict(crossplay["selected_bdqn_hyperparameters"]), "name": "B2"}
    source = str(crossplay["deployment_bdqn_checkpoint"]["path"])
    gp_policy = str(g2["deployment_policy"]["path"])
    frozen_gp_sha256 = sha256_file(gp_policy)
    seeds = [int(seed) for seed in spec["bdqn"]["confirm_seeds"]]
    definition = spec["bdqn"]
    validation = load_scenario_manifest(spec["inputs"]["g_validation"]["path"])[
        "scenarios"
    ]
    small = _stratified_subset(
        validation, int(spec["evaluation"]["screen_validation_size"])
    )
    convergence_path = destination / "convergence.json"
    convergence = (
        _read_json(convergence_path)
        if convergence_path.is_file()
        else _initial_bdqn_plateau()
    )
    small_rows_by_step = {}
    stopped_step = int(convergence.get("actual_environment_steps", 0))
    first_threshold = int(definition["checkpoint_interval"])
    cells = _train_bdqn_candidate_to(
        spec=spec,
        candidate=candidate,
        source_checkpoint=source,
        gp_policy=gp_policy,
        destination=destination,
        seeds=seeds,
        threshold=max(first_threshold, stopped_step),
        device=device,
    )
    zero_rows = _evaluate_bdqn_cells(
        cells=cells,
        step=0,
        gp_policy=gp_policy,
        scenarios=small,
        device=device,
        model="B2",
        split="screen64",
    )
    small_rows_by_step[0] = zero_rows
    for completed_step in range(
        first_threshold,
        stopped_step + 1,
        int(definition["checkpoint_interval"]),
    ):
        small_rows_by_step[completed_step] = _evaluate_bdqn_cells(
            cells=cells,
            step=completed_step,
            gp_policy=gp_policy,
            scenarios=small,
            device=device,
            model="B2",
            split="screen64",
        )
    full_rows_by_step = {}
    thresholds = (
        ()
        if convergence.get("confirmed_step") is not None
        else range(
            first_threshold if stopped_step == 0 else stopped_step + first_threshold,
            int(definition["max_env_steps"]) + 1,
            int(definition["checkpoint_interval"]),
        )
    )
    for threshold in thresholds:
        if threshold > max(first_threshold, stopped_step):
            cells = _train_bdqn_candidate_to(
                spec=spec,
                candidate=candidate,
                source_checkpoint=source,
                gp_policy=gp_policy,
                destination=destination,
                seeds=seeds,
                threshold=threshold,
                device=device,
            )
        current_rows = _evaluate_bdqn_cells(
            cells=cells,
            step=threshold,
            gp_policy=gp_policy,
            scenarios=small,
            device=device,
            model="B2",
            split="screen64",
        )
        small_rows_by_step[threshold] = current_rows
        stopped_step = threshold
        convergence, candidate_convergence = _update_bdqn_plateau(
            convergence,
            previous_rows=zero_rows,
            current_rows=current_rows,
            current_step=threshold,
            threshold=0.01,
            min_steps=20000,
            seed=20262300 + threshold,
        )
        confirmed = False
        if candidate_convergence:
            if 0 not in full_rows_by_step:
                full_rows_by_step[0] = _evaluate_bdqn_cells(
                    cells=cells,
                    step=0,
                    gp_policy=gp_policy,
                    scenarios=validation,
                    device=device,
                    model="B2@0",
                    split="validation",
                )
            full_rows_by_step[threshold] = _evaluate_bdqn_cells(
                cells=cells,
                step=threshold,
                gp_policy=gp_policy,
                scenarios=validation,
                device=device,
                model=f"B2@{threshold}",
                split="validation",
            )
            confirmation = _bdqn_plateau_comparison(
                full_rows_by_step[0],
                full_rows_by_step[threshold],
                threshold=0.01,
                samples=int(spec["evaluation"]["bootstrap_samples"]),
                seed=20262350 + threshold,
            )
            convergence["full_validation_confirmation"] = {
                "step": int(threshold),
                **confirmation,
            }
            confirmed = bool(confirmation["stable"])
            convergence["comparisons"][-1]["confirmed"] = confirmed
            if confirmed:
                convergence["confirmed_step"] = int(threshold)
            else:
                convergence["stable_windows"] = 0
                convergence["provisional_step"] = None
        convergence.update(
            {
                "status": "converged" if confirmed else "running",
                "actual_environment_steps": int(stopped_step),
                "max_environment_steps": int(definition["max_env_steps"]),
            }
        )
        _write_json(convergence_path, convergence)
        if confirmed:
            break
    stop_reason = (
        "converged"
        if convergence.get("confirmed_step") is not None
        else "max_env_steps_reached"
    )
    convergence.update(
        {
            "status": stop_reason,
            "actual_environment_steps": int(stopped_step),
            "max_environment_steps": int(definition["max_env_steps"]),
        }
    )
    _write_json(convergence_path, convergence)
    for cell in cells.values():
        manifest = _read_json(cell / "cell_manifest.json")
        if manifest.get("status") != "complete":
            _finalize_cell(cell, stopped_step=stopped_step, stop_reason=stop_reason)
    for step in _adaptive_full_steps(
        small_rows_by_step, stopped_step=stopped_step
    ):
        if step not in full_rows_by_step:
            full_rows_by_step[step] = _evaluate_bdqn_cells(
                cells=cells,
                step=step,
                gp_policy=gp_policy,
                scenarios=validation,
                device=device,
                model=f"B2@{step}",
                split="validation",
            )
    rows_by_name = {
        f"B2@{step}": rows for step, rows in full_rows_by_step.items()
    }
    step_selection = robust_pareto_selection(
        rows_by_name,
        baseline="B2@0",
        metadata={
            name: {
                "gp_nodes": _policy_node_count(gp_policy),
                "bdqn_step": int(name.split("@", 1)[1]),
            }
            for name in rows_by_name
        },
        budget_guard=float(spec["evaluation"]["budget_guard"]),
        samples=int(spec["evaluation"]["bootstrap_samples"]),
        seed=20262400,
    )
    selected_step = int(str(step_selection["knee"]).split("@", 1)[1])
    paths = [
        _checkpoint_path(cells[seed], selected_step).resolve() for seed in sorted(cells)
    ]
    selection = {
        "schema_version": TUNING_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_checkpoint": _input_record(source),
        "frozen_gp_policy": _input_record(gp_policy),
        "hyperparameters": candidate,
        "convergence": convergence,
        "step_selection": step_selection,
        "selected_step": selected_step,
        "checkpoints": [_input_record(path) for path in paths],
        "deployment_checkpoint": _nearest_deployment_member(
            paths, full_rows_by_step[selected_step]
        ),
        "performance_estimate": "three-seed aggregate with hierarchical 95% CI",
        "frozen_gp_sha256_before": frozen_gp_sha256,
        "frozen_gp_sha256_after": sha256_file(gp_policy),
    }
    if selection["frozen_gp_sha256_after"] != frozen_gp_sha256:
        raise RuntimeError("B2 training mutated its frozen GP policy.")
    _write_csv(
        destination / "small_validation_results.csv",
        [row for rows in small_rows_by_step.values() for row in rows],
    )
    _write_csv(
        destination / "full_validation_results.csv",
        [row for rows in full_rows_by_step.values() for row in rows],
    )
    _write_json(selection_path, selection)
    return selection


def _lineage_groups(
    *,
    prefix: Mapping[str, Any],
    crossplay: Mapping[str, Any],
    g2: Mapping[str, Any],
    b2: Mapping[str, Any],
) -> dict[str, dict[str, list[str]]]:
    g0 = [str(prefix["gp_policy"]["path"])]
    b0 = [str(record["path"]) for record in prefix["bdqn_checkpoints"]]
    g1 = [str(path) for path in crossplay["gp_groups"][crossplay["selected_gp"]]]
    b1 = [
        str(path) for path in crossplay["bdqn_groups"][crossplay["selected_bdqn"]]
    ]
    g2_paths = [str(record["path"]) for record in g2["policies"]]
    b2_paths = [str(record["path"]) for record in b2["checkpoints"]]
    return {
        "S0": {"gp": g0, "bdqn": b0},
        "SG1": {"gp": g1, "bdqn": b0},
        "S1": {"gp": g1, "bdqn": b1},
        "SG2": {"gp": g2_paths, "bdqn": b1},
        "S2": {"gp": g2_paths, "bdqn": b2_paths},
    }


def _flatten_summary(stage: str, split: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "stage": stage,
        "split": split,
        "repeat_count": len(summary.get("repeats", ())),
    }
    for name, value in summary.items():
        if name.endswith("_ci95") and isinstance(value, (list, tuple)) and len(value) == 2:
            stem = name[: -len("_ci95")]
            row[f"{stem}_ci_low"] = value[0]
            row[f"{stem}_ci_high"] = value[1]
        row[name] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def _format_ci(mean: float, ci: Sequence[float], digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} [{float(ci[0]):.{digits}f}, {float(ci[1]):.{digits}f}]"


def run_lineage_evaluation(
    *,
    spec: Mapping[str, Any],
    scenario_paths: Mapping[str, Path],
    groups: Mapping[str, Mapping[str, Sequence[str]]],
    output_dir: str | Path,
    device: str,
) -> dict[str, Path]:
    """Evaluate gate and final exactly once after the lineage is frozen."""

    destination = Path(output_dir).resolve()
    report_path = destination / "final" / "tuning_report.html"
    if report_path.is_file():
        return {
            "gate_results": destination / "gate" / "results.csv",
            "final_results": destination / "final" / "results.csv",
            "metrics": destination / "final" / "metrics.csv",
            "paired_ci": destination / "final" / "paired_ci.csv",
            "report": report_path,
        }
    all_results: dict[str, list[dict[str, Any]]] = {"gate": [], "final": []}
    split_mapping = {
        "gate": (("iid", "gate_iid"), ("ood", "gate_ood")),
        "final": (("iid", "final_iid"), ("ood", "final_ood")),
    }
    for family, splits in split_mapping.items():
        for stage, artifacts in groups.items():
            for split, manifest_name in splits:
                scenarios = load_scenario_manifest(scenario_paths[manifest_name])[
                    "scenarios"
                ]
                all_results[family].extend(
                    _evaluate_stack_repeats(
                        gp_policies=list(artifacts["gp"]),
                        bdqn_checkpoints=list(artifacts["bdqn"]),
                        scenarios=scenarios,
                        device=device,
                        model=stage,
                        split=split,
                    )
                )
        _write_csv(destination / family / "results.csv", all_results[family])

    final_rows = all_results["final"]
    metrics = []
    raw_summaries = {}
    for stage in groups:
        raw_summaries[stage] = {}
        stage_rows = [row for row in final_rows if row["stack"] == stage]
        for split in ("iid", "ood", "all"):
            rows = (
                stage_rows
                if split == "all"
                else [row for row in stage_rows if row["split"] == split]
            )
            summary = summarize_rows(
                rows,
                samples=int(spec["evaluation"]["bootstrap_samples"]),
                seed=20262500 + list(groups).index(stage) * 100 + (0 if split == "iid" else 1 if split == "ood" else 2),
            )
            raw_summaries[stage][split] = summary
            metrics.append(_flatten_summary(stage, split, summary))
    paired_rows = []
    baseline = [row for row in final_rows if row["stack"] == "S0"]
    metric_definitions = (
        ("makespan", True),
        ("final_net_cost", True),
        ("peak_net_cost", True),
        ("failure_aware_j", False),
    )
    for stage_index, stage in enumerate(groups):
        if stage == "S0":
            continue
        candidate = [row for row in final_rows if row["stack"] == stage]
        for metric_index, (field, both_success) in enumerate(metric_definitions):
            result = paired_difference_ci(
                baseline,
                candidate,
                field,
                both_success=both_success,
                samples=int(spec["evaluation"]["bootstrap_samples"]),
                seed=20262600 + stage_index * 100 + metric_index,
            )
            paired_rows.append(
                {
                    "baseline": "S0",
                    "candidate": stage,
                    "metric": field,
                    "mean_candidate_minus_baseline": result["mean_difference"],
                    "ci95_low": result["ci95"][0],
                    "ci95_high": result["ci95"][1],
                    "paired_rows": result["n"],
                }
            )
    _write_csv(destination / "final" / "metrics.csv", metrics)
    _write_csv(destination / "final" / "paired_ci.csv", paired_rows)
    _write_json(destination / "final" / "metrics.json", raw_summaries)

    table_rows = []
    for stage in groups:
        item = raw_summaries[stage]["all"]
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(stage)}</td>"
            f"<td>{_format_ci(item['failure_rate'], item['failure_rate_ci95'], 4)}</td>"
            f"<td>{_format_ci(item['mean_success_makespan'], item['mean_success_makespan_ci95'])}</td>"
            f"<td>{_format_ci(item['mean_final_cost'], item['mean_final_cost_ci95'])}</td>"
            f"<td>{_format_ci(item['mean_peak_cost'], item['mean_peak_cost_ci95'])}</td>"
            f"<td>{_format_ci(item['budget_violation_rate'], item['budget_violation_rate_ci95'], 4)}</td>"
            f"<td>{_format_ci(item['mean_architecture_changes'], item['mean_architecture_changes_ci95'])}</td>"
            f"<td>{_format_ci(item['mean_j'], item['mean_j_ci95'], 4)}</td>"
            "</tr>"
        )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>GP + BDQN tuning report</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#1f2937}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d1d5db;padding:.55rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f3f4f6}}code{{background:#f3f4f6;padding:.1rem .3rem}}</style></head>
<body><h1>GP + BDQN 快速调参结果</h1>
<p>所有数值均为三个独立重复的聚合均值与层级bootstrap 95% CI；单个部署checkpoint不作为性能估计。</p>
<table><thead><tr><th>Stack</th><th>Failure rate</th><th>Makespan</th><th>Final cost</th><th>Peak cost</th><th>Budget violation</th><th>Architecture changes</th><th>J（仅报告）</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p>配对差值见 <code>paired_ci.csv</code>，原始结果见 <code>results.csv</code>。</p></body></html>"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "gate_results": destination / "gate" / "results.csv",
        "final_results": destination / "final" / "results.csv",
        "metrics": destination / "final" / "metrics.csv",
        "paired_ci": destination / "final" / "paired_ci.csv",
        "report": report_path,
    }


def _mark_stage(
    manifest_path: Path,
    manifest: dict[str, Any],
    stage: str,
    outputs: Mapping[str, Any],
) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": "complete",
        "completed_at": _utc_now(),
        "outputs": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in outputs.items()
        },
    }
    _write_json(manifest_path, manifest)


def run_gp_bdqn_tuning(
    *,
    spec_path: str | Path,
    output_dir: str | Path,
    rule_device: str = "auto",
    bdqn_device: str = "auto",
    stop_after_stage: str | None = None,
) -> dict[str, Path]:
    """Execute and resume the complete formal tuning lineage."""

    spec_path = Path(spec_path).resolve()
    spec = load_tuning_spec(spec_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "tuning_manifest.json"
    frozen_spec = _input_record(spec_path)
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("spec") != frozen_spec:
            raise ValueError("tuning resume spec changed.")
        if manifest.get("status") == "complete":
            return {
                "manifest": manifest_path,
                "rule_lr_selection": destination / "rule_lr" / "rule_lr_selection.json",
                "gp_selection": destination / "gp_screen" / "selection.json",
                "bdqn_selection": destination / "bdqn_screen" / "selection.json",
                "pareto_front": destination / "crossplay" / "pareto_front.json",
                "lineage": destination / "continuation_lineage.json",
                "final_metrics": destination / "final" / "metrics.csv",
                "final_paired_ci": destination / "final" / "paired_ci.csv",
                "report": destination / "final" / "tuning_report.html",
            }
    else:
        manifest = {
            "schema_version": TUNING_SCHEMA_VERSION,
            "status": "running",
            "created_at": _utc_now(),
            "code_commit": _git_commit(),
            "spec": frozen_spec,
            "spec_sha256": frozen_spec["sha256"],
            "stages": {},
        }
        _write_json(manifest_path, manifest)

    resolved_rule_device = default_device() if rule_device == "auto" else str(rule_device)
    resolved_bdqn_device = default_device() if bdqn_device == "auto" else str(bdqn_device)
    registered = [
        str(record["path"]) for record in spec["inputs"]["registered_manifests"]
    ]
    scenarios = generate_tuning_evaluation_scenarios(
        destination / "scenarios",
        registered_manifests=registered,
        evaluation_spec=spec["evaluation"],
    )
    _mark_stage(manifest_path, manifest, "scenarios", scenarios)

    rule = run_rule_lr_stage(
        spec=spec,
        scenario_paths=scenarios,
        output_dir=destination / "rule_lr",
        device=resolved_rule_device,
    )
    _mark_stage(
        manifest_path,
        manifest,
        "rule_lr",
        {"selection": destination / "rule_lr" / "rule_lr_selection.json", "winner": rule["winner"]},
    )
    prefix = prepare_s0_prefix(
        spec=spec,
        rule_selection=rule,
        output_dir=destination / "prefix",
        device=resolved_bdqn_device,
    )
    _mark_stage(
        manifest_path,
        manifest,
        "s0_prefix",
        {"selection": destination / "prefix" / "selection.json", "retrained": prefix["prefix_retrained"]},
    )
    gp_selection = run_gp_screen_stage(
        spec=spec,
        base_gp_policy=prefix["gp_policy"]["path"],
        base_bdqn_checkpoints=[record["path"] for record in prefix["bdqn_checkpoints"]],
        deployment_bdqn_checkpoint=prefix["deployment_bdqn_checkpoint"]["path"],
        output_dir=destination / "gp_screen",
        device=resolved_bdqn_device,
    )
    _mark_stage(
        manifest_path,
        manifest,
        "gp_screen",
        {"selection": destination / "gp_screen" / "selection.json", "promoted": gp_selection["promoted"]},
    )
    bdqn_selection = run_bdqn_screen_stage(
        spec=spec,
        base_bdqn_checkpoints=[record["path"] for record in prefix["bdqn_checkpoints"]],
        deployment_bdqn_checkpoint=prefix["deployment_bdqn_checkpoint"]["path"],
        gp_policy=gp_selection["deployment_policy"]["path"],
        output_dir=destination / "bdqn_screen",
        device=resolved_bdqn_device,
    )
    _mark_stage(
        manifest_path,
        manifest,
        "bdqn_screen",
        {"selection": destination / "bdqn_screen" / "selection.json", "promoted": bdqn_selection["promoted"]},
    )
    if stop_after_stage == "bdqn_screen":
        return {"manifest": manifest_path}
    crossplay = run_crossplay_stage(
        spec=spec,
        scenario_paths=scenarios,
        prefix=prefix,
        gp_selection=gp_selection,
        bdqn_selection=bdqn_selection,
        output_dir=destination / "crossplay",
        device=resolved_bdqn_device,
    )
    _mark_stage(
        manifest_path,
        manifest,
        "crossplay",
        {"selection": destination / "crossplay" / "pareto_front.json", "knee": crossplay["knee"]},
    )
    g2 = run_g2_confirmation_stage(
        spec=spec,
        crossplay=crossplay,
        output_dir=destination / "g2",
        device=resolved_bdqn_device,
    )
    _mark_stage(manifest_path, manifest, "g2", {"selection": destination / "g2" / "selection.json"})
    b2 = run_b2_confirmation_stage(
        spec=spec,
        crossplay=crossplay,
        g2=g2,
        output_dir=destination / "b2",
        device=resolved_bdqn_device,
    )
    _mark_stage(manifest_path, manifest, "b2", {"selection": destination / "b2" / "selection.json"})

    groups = _lineage_groups(prefix=prefix, crossplay=crossplay, g2=g2, b2=b2)
    lineage_path = _write_json(
        destination / "continuation_lineage.json",
        {
            "schema_version": TUNING_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "rule_lr_winner": rule["winner"],
            "prefix_retrained": prefix["prefix_retrained"],
            "crossplay_knee": crossplay["knee"],
            "groups": groups,
            "artifact_hashes": {
                stage: {
                    "gp": [sha256_file(path) for path in artifacts["gp"]],
                    "bdqn": [sha256_file(path) for path in artifacts["bdqn"]],
                }
                for stage, artifacts in groups.items()
            },
        },
    )
    evaluation_outputs = run_lineage_evaluation(
        spec=spec,
        scenario_paths=scenarios,
        groups=groups,
        output_dir=destination,
        device=resolved_bdqn_device,
    )
    _mark_stage(manifest_path, manifest, "final", evaluation_outputs)
    manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "lineage": str(lineage_path),
            "report": str(evaluation_outputs["report"]),
            "completed_stages": list(manifest.get("stages", {})),
        }
    )
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "rule_lr_selection": destination / "rule_lr" / "rule_lr_selection.json",
        "gp_selection": destination / "gp_screen" / "selection.json",
        "bdqn_selection": destination / "bdqn_screen" / "selection.json",
        "pareto_front": destination / "crossplay" / "pareto_front.json",
        "lineage": lineage_path,
        "final_metrics": evaluation_outputs["metrics"],
        "final_paired_ci": evaluation_outputs["paired_ci"],
        "report": evaluation_outputs["report"],
    }
