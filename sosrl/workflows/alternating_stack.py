"""Two-round GP/Branching-DQN alternation with online convergence barriers."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from ..gp.artifact import load_gp_policy, sha256_file
from ..gp.config import GPArchitectureConfig
from ..rl.config import BranchingDQNConfig, default_device
from .branching_gp_finetune import (
    compare_paired_results,
    paired_bootstrap_ci,
    summarize_results,
)
from .gp_architecture import (
    _generate_split,
    load_scenario_manifest,
    save_scenario_manifest,
    train_gp_architecture,
)
from .round1_study import (
    evaluate_bdqn_provider_cell,
    train_bdqn_provider_cell,
)


ALTERNATION_SCHEMA_VERSION = 1
STAGE_ORDER = ("S0", "SG1", "S1", "SG2", "S2")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
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
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(file):
            row: dict[str, Any] = {}
            for field, value in raw.items():
                if value == "True":
                    row[field] = True
                elif value == "False":
                    row[field] = False
                else:
                    row[field] = value
            rows.append(row)
        return rows


def _checkpoint_label(step: int) -> str:
    return f"{int(step) // 1000}k" if step and int(step) % 1000 == 0 else str(step)


def _checkpoint_path(cell_dir: str | Path, step: int) -> Path:
    return Path(cell_dir) / "checkpoints" / f"checkpoint_{_checkpoint_label(step)}.pt"


def _input_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def generate_alternation_evaluation_scenarios(
    output_dir: str | Path,
    *,
    existing_manifests: Sequence[str | Path],
    gate_iid_size: int = 512,
    gate_ood_size: int = 256,
    final_iid_size: int = 1000,
    final_ood_size: int = 500,
    gate_iid_seed: int = 20260910,
    gate_ood_seed: int = 20260911,
    final_iid_seed: int = 20260912,
    final_ood_seed: int = 20260913,
) -> dict[str, Path]:
    """Generate four frozen manifests disjoint from every registered Round1 split."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    occupied: set[str] = set()
    existing_records = []
    for path in existing_manifests:
        manifest = load_scenario_manifest(path)
        hashes = {str(row["scenario_hash"]) for row in manifest["scenarios"]}
        overlap = occupied & hashes
        occupied.update(hashes)
        existing_records.append(
            {
                **_input_record(path),
                "manifest_hash": str(manifest["manifest_hash"]),
                "size": int(manifest["size"]),
                "duplicate_hashes_already_registered": len(overlap),
            }
        )
    specifications = (
        ("gate_iid", int(gate_iid_size), int(gate_iid_seed), False),
        ("gate_ood", int(gate_ood_size), int(gate_ood_seed), True),
        ("final_iid", int(final_iid_size), int(final_iid_seed), False),
        ("final_ood", int(final_ood_size), int(final_ood_seed), True),
    )
    paths: dict[str, Path] = {}
    generated_records = {}
    for split, size, seed, ood in specifications:
        scenarios = _generate_split(split=split, size=size, seed=seed, ood=ood)
        hashes = {str(row["scenario_hash"]) for row in scenarios}
        if len(hashes) != len(scenarios) or occupied & hashes:
            raise ValueError(f"generated split {split!r} overlaps an existing scenario.")
        occupied.update(hashes)
        path = save_scenario_manifest(
            destination / f"{split}.json",
            split=split,
            seed=seed,
            scenarios=scenarios,
        )
        paths[split] = path
        manifest = load_scenario_manifest(path)
        generated_records[split] = {
            **_input_record(path),
            "manifest_hash": str(manifest["manifest_hash"]),
            "size": int(manifest["size"]),
            "ood": bool(ood),
        }
    registry = {
        "schema_version": ALTERNATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "test_locked": True,
        "existing_manifests": existing_records,
        "manifests": generated_records,
    }
    paths["registry"] = _write_json(destination / "scenario_registry.json", registry)
    return paths


def initial_bdqn_convergence_state() -> dict[str, Any]:
    return {
        "stable_windows": 0,
        "provisional_step": None,
        "confirmed_step": None,
        "comparisons": [],
    }


def update_bdqn_convergence(
    state: dict[str, Any],
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    delta_j_ci: tuple[float, float],
    threshold: float = 0.01,
    patience: int = 2,
    confirmation_windows: int = 1,
    min_steps: int = 20000,
) -> tuple[dict[str, Any], bool]:
    """Update the synchronized three-seed plateau monitor."""

    updated = json.loads(json.dumps(state))
    previous_j = float(previous["mean_j"])
    current_j = float(current["mean_j"])
    previous_makespan = float(previous["mean_success_makespan"])
    current_makespan = float(current["mean_success_makespan"])
    relative_j = (previous_j - current_j) / max(abs(previous_j), 1e-12)
    relative_makespan = (
        previous_makespan - current_makespan
    ) / max(abs(previous_makespan), 1e-12)
    failure_unchanged = int(previous["failure_count"]) == int(
        current["failure_count"]
    )
    safe = (
        int(current.get("invalid_action_count", 0)) == 0
        and int(current.get("provider_invariant_violations", 0)) == 0
    )
    ci_crosses_zero = float(delta_j_ci[0]) <= 0.0 <= float(delta_j_ci[1])
    candidate_stable = (
        failure_unchanged
        and abs(relative_j) < float(threshold)
        and abs(relative_makespan) < float(threshold)
        and ci_crosses_zero
        and safe
    )
    confirmation_stable = (
        failure_unchanged
        and relative_j < float(threshold)
        and relative_makespan < float(threshold)
        and safe
    )
    confirming = state.get("provisional_step") is not None
    stable = confirmation_stable if confirming else candidate_stable
    if stable:
        updated["stable_windows"] = int(updated["stable_windows"]) + 1
    else:
        updated["stable_windows"] = 0
        updated["provisional_step"] = None
    stable_windows = int(updated["stable_windows"])
    current_step = int(current["target_environment_steps"])
    if stable_windows >= int(patience) and updated["provisional_step"] is None:
        updated["provisional_step"] = current_step
    confirmed = (
        stable_windows >= int(patience) + int(confirmation_windows)
        and current_step >= int(min_steps)
    )
    if confirmed:
        updated["confirmed_step"] = current_step
    updated["comparisons"].append(
        {
            "previous_steps": int(previous["target_environment_steps"]),
            "current_steps": current_step,
            "failure_unchanged": bool(failure_unchanged),
            "relative_j_improvement": float(relative_j),
            "relative_makespan_improvement": float(relative_makespan),
            "delta_j_ci95_low": float(delta_j_ci[0]),
            "delta_j_ci95_high": float(delta_j_ci[1]),
            "safe": bool(safe),
            "candidate_window_stable": bool(candidate_stable),
            "confirmation_window_stable": bool(confirmation_stable),
            "confirmation_window": bool(confirming),
            "stable": bool(stable),
            "stable_windows": stable_windows,
            "confirmed": bool(confirmed),
        }
    )
    return updated, bool(confirmed)


def _bdqn_config(
    *,
    seed: int,
    max_env_steps: int,
    device: str,
) -> BranchingDQNConfig:
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
        epsilon_start=0.10,
        epsilon_end=0.02,
        epsilon_decay=0.995,
        seed=int(seed),
        device=str(device),
        log_interval=10,
    )


def _run_bdqn_chunk(job: dict[str, Any]) -> dict[str, Any]:
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
        reset_learn_step=bool(job.get("reset_learn_step", False)),
    )
    return {
        "seed": int(config.seed),
        "outputs": {key: str(value) for key, value in outputs.items()},
    }


def _dispatch_bdqn_chunks(
    jobs: Sequence[dict[str, Any]], *, parallel_jobs: int
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if int(parallel_jobs) <= 1:
        return [_run_bdqn_chunk(job) for job in jobs]
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=min(int(parallel_jobs), len(jobs)),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(_run_bdqn_chunk, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda row: int(row["seed"]))


def _evaluate_bdqn_step(
    *,
    cell_dirs: dict[int, Path],
    step: int,
    gp_policy: Path,
    scenarios: Sequence[dict[str, Any]],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, cell_dir in sorted(cell_dirs.items()):
        checkpoint = _checkpoint_path(cell_dir, step)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        seed_rows = evaluate_bdqn_provider_cell(
            model=f"seed_{seed}_{_checkpoint_label(step)}",
            scheduler_checkpoint=checkpoint,
            provider_kind="g0",
            scenarios=scenarios,
            architecture_checkpoint=None,
            gp_policy=gp_policy,
            device=device,
        )
        for row in seed_rows:
            row.update(
                {
                    "seed": int(seed),
                    "target_environment_steps": int(step),
                }
            )
        rows.extend(seed_rows)
    return rows


def _paired_step_ci(
    previous_rows: Sequence[dict[str, Any]],
    current_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> tuple[float, float]:
    def key(row):
        return int(row["seed"]), str(row["scenario_hash"])

    before = {key(row): row for row in previous_rows}
    after = {key(row): row for row in current_rows}
    if before.keys() != after.keys():
        raise ValueError("BDQN convergence checkpoints use different paired rows.")
    deltas = [
        float(after[item]["failure_aware_j"])
        - float(before[item]["failure_aware_j"])
        for item in sorted(before)
    ]
    return paired_bootstrap_ci(deltas, samples=2000, seed=seed)


def _finalize_cell(
    cell_dir: Path,
    *,
    stopped_step: int,
    stop_reason: str,
) -> None:
    history = _read_csv(cell_dir / "training_history.csv")
    if any(
        int(float(row.get("invalid_action_count", 0))) != 0
        or int(float(row.get("provider_invariant_violations", 0))) != 0
        for row in history
    ):
        raise RuntimeError("BDQN training produced an illegal assignment or provider violation.")
    manifest_path = cell_dir / "cell_manifest.json"
    manifest = _read_json(manifest_path)
    checkpoints = {}
    for path in sorted((cell_dir / "checkpoints").glob("checkpoint_*.pt")):
        label = path.stem.removeprefix("checkpoint_")
        step = int(label[:-1]) * 1000 if label.endswith("k") else int(label)
        if step <= int(stopped_step):
            checkpoints[str(step)] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
    manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "stopped_step": int(stopped_step),
            "stop_reason": str(stop_reason),
            "checkpoints": checkpoints,
        }
    )
    _write_json(manifest_path, manifest)
    (cell_dir / "resume_state.pt").unlink(missing_ok=True)


def _selection_key(summary: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(summary["failure_rate"]),
        float(summary["mean_j"]),
        float(summary["mean_success_makespan"]),
        int(summary["target_environment_steps"]),
    )


def train_convergent_bdqn_stage(
    *,
    source_checkpoint: str | Path,
    gp_policy: str | Path,
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
    seeds: Sequence[int],
    max_env_steps: int = 40000,
    checkpoint_interval: int = 5000,
    min_convergence_steps: int = 20000,
    convergence_threshold: float = 0.01,
    convergence_patience: int = 2,
    convergence_confirmation_windows: int = 1,
    device: str = "auto",
    parallel_jobs: int = 3,
) -> dict[str, Path]:
    """Train three warm-start seeds behind synchronized validation barriers."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / "selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        for name, record in selection.get("frozen_inputs", {}).items():
            if sha256_file(record["path"]) != record["sha256"]:
                raise ValueError(f"completed BDQN stage input {name!r} changed.")
        selected = Path(selection["selected_checkpoint"])
        if sha256_file(selected) != selection["selected_checkpoint_sha256"]:
            raise ValueError("completed BDQN stage checkpoint hash changed.")
        return {
            "selection": selection_path,
            "selected_checkpoint": selected,
            "convergence": destination / "convergence.json",
        }
    resolved_device = default_device() if device == "auto" else str(device)
    source = Path(source_checkpoint).resolve()
    gp_path = Path(gp_policy).resolve()
    source_hash = sha256_file(source)
    gp_hash = sha256_file(gp_path)
    validation = load_scenario_manifest(validation_manifest)
    thresholds = tuple(range(0, int(max_env_steps) + 1, int(checkpoint_interval)))
    if thresholds[-1] != int(max_env_steps):
        raise ValueError("max_env_steps must be divisible by checkpoint_interval.")
    if len(seeds) != 3 or len({int(seed) for seed in seeds}) != 3:
        raise ValueError("a convergent BDQN stage requires exactly three unique seeds.")
    cell_dirs = {int(seed): destination / "cells" / f"seed_{int(seed)}" for seed in seeds}
    configs = {
        int(seed): _bdqn_config(
            seed=int(seed), max_env_steps=int(max_env_steps), device=resolved_device
        )
        for seed in seeds
    }
    stage_manifest = {
        "schema_version": ALTERNATION_SCHEMA_VERSION,
        "status": "running",
        "created_at": _utc_now(),
        "source_checkpoint": _input_record(source),
        "gp_policy": _input_record(gp_path),
        "train_manifest": _input_record(train_manifest),
        "validation_manifest": _input_record(validation_manifest),
        "seeds": [int(seed) for seed in seeds],
        "configs": {str(seed): asdict(config) for seed, config in configs.items()},
        "checkpoint_steps": list(thresholds),
    }
    manifest_path = destination / "stage_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        for field in (
            "source_checkpoint",
            "gp_policy",
            "train_manifest",
            "validation_manifest",
            "seeds",
            "configs",
            "checkpoint_steps",
        ):
            if existing.get(field) != stage_manifest.get(field):
                raise ValueError(f"resumed BDQN stage changed immutable field {field!r}.")
        stage_manifest = existing
    else:
        _write_json(manifest_path, stage_manifest)

    results_path = destination / "validation" / "checkpoint_results.csv"
    summaries_path = destination / "validation" / "checkpoint_summary.csv"
    all_rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(results_path)]
    summaries: list[dict[str, Any]] = [dict(row) for row in _read_csv(summaries_path)]
    rows_by_step: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        rows_by_step.setdefault(int(float(row["target_environment_steps"])), []).append(row)
    convergence_path = destination / "convergence.json"
    convergence = (
        _read_json(convergence_path)
        if convergence_path.is_file()
        else initial_bdqn_convergence_state()
    )
    stopped_step = max(rows_by_step, default=0)
    stop_reason = None
    if convergence.get("confirmed_step") is not None:
        stopped_step = int(convergence["confirmed_step"])
        stop_reason = "converged"

    for threshold in (() if stop_reason == "converged" else thresholds[1:]):
        jobs = []
        for seed in seeds:
            cell_dir = cell_dirs[int(seed)]
            if _checkpoint_path(cell_dir, threshold).is_file():
                continue
            jobs.append(
                {
                    "cell_dir": str(cell_dir),
                    "source_checkpoint": str(source),
                    "train_manifest": str(Path(train_manifest).resolve()),
                    "validation_manifest": str(Path(validation_manifest).resolve()),
                    "gp_policy": str(gp_path),
                    "checkpoint_steps": list(thresholds),
                    "stop_after_checkpoint": int(threshold),
                    "reset_learn_step": True,
                    "config": asdict(configs[int(seed)]),
                }
            )
        _dispatch_bdqn_chunks(jobs, parallel_jobs=int(parallel_jobs))
        evaluate_steps = [threshold]
        if 0 not in rows_by_step:
            evaluate_steps.insert(0, 0)
        for step in evaluate_steps:
            if step in rows_by_step:
                continue
            rows = _evaluate_bdqn_step(
                cell_dirs=cell_dirs,
                step=step,
                gp_policy=gp_path,
                scenarios=validation["scenarios"],
                device=resolved_device,
            )
            rows_by_step[int(step)] = rows
            all_rows.extend(rows)
            summary = summarize_results(
                f"bdqn_{_checkpoint_label(step)}",
                rows,
                additional_steps=int(step),
            )
            summary["target_environment_steps"] = int(step)
            summaries.append(summary)
        _write_csv(results_path, all_rows)
        _write_csv(summaries_path, summaries)

        previous_step = int(threshold) - int(checkpoint_interval)
        processed_steps = {
            int(row["current_steps"])
            for row in convergence.get("comparisons", [])
        }
        if (
            previous_step >= int(checkpoint_interval)
            and int(threshold) not in processed_steps
        ):
            summary_by_step = {
                int(float(row["target_environment_steps"])): row for row in summaries
            }
            ci = _paired_step_ci(
                rows_by_step[previous_step],
                rows_by_step[int(threshold)],
                seed=sum(int(seed) for seed in seeds) + int(threshold),
            )
            convergence, confirmed = update_bdqn_convergence(
                convergence,
                previous=summary_by_step[previous_step],
                current=summary_by_step[int(threshold)],
                delta_j_ci=ci,
                threshold=float(convergence_threshold),
                patience=int(convergence_patience),
                confirmation_windows=int(convergence_confirmation_windows),
                min_steps=int(min_convergence_steps),
            )
            _write_json(convergence_path, convergence)
            if confirmed:
                stopped_step = int(threshold)
                stop_reason = "converged"
                break
        stopped_step = int(threshold)
    if stop_reason is None:
        stop_reason = "max_env_steps_reached"
    convergence.update(
        {
            "status": stop_reason,
            "actual_environment_steps": int(stopped_step),
            "max_environment_steps": int(max_env_steps),
        }
    )
    _write_json(convergence_path, convergence)

    for cell_dir in cell_dirs.values():
        _finalize_cell(
            cell_dir,
            stopped_step=stopped_step,
            stop_reason=stop_reason,
        )
    overall = [
        row for row in summaries if str(row.get("category", "all")) == "all"
    ]
    winner = min(overall, key=_selection_key)
    selected_step = int(float(winner["target_environment_steps"]))
    seed_scores = []
    for seed in seeds:
        seed_rows = [
            row
            for row in rows_by_step[selected_step]
            if int(float(row["seed"])) == int(seed)
        ]
        seed_scores.append(
            (
                float(mean(float(row["failure_aware_j"]) for row in seed_rows)),
                int(seed),
                _checkpoint_path(cell_dirs[int(seed)], selected_step),
            )
        )
    seed_scores.sort(key=lambda item: (item[0], item[1]))
    selected_seed = seed_scores[len(seed_scores) // 2]
    selection = {
        "schema_version": ALTERNATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "selection_order": [
            "failure_rate",
            "mean_failure_aware_j",
            "mean_success_makespan",
            "fewer_environment_steps",
        ],
        "selected_step": selected_step,
        "selected_seed": selected_seed[1],
        "selected_checkpoint": str(selected_seed[2].resolve()),
        "selected_checkpoint_sha256": sha256_file(selected_seed[2]),
        "selected_step_metrics": winner,
        "seed_order_by_validation_j": [item[1] for item in seed_scores],
        "seed_validation_j": [
            {"seed": item[1], "mean_j": item[0], "checkpoint": str(item[2])}
            for item in seed_scores
        ],
        "frozen_inputs": {
            "source_checkpoint": {
                "path": str(source),
                "sha256": source_hash,
            },
            "gp_policy": {
                "path": str(gp_path),
                "sha256": gp_hash,
            },
        },
        "convergence": convergence,
    }
    _write_json(selection_path, selection)
    stage_manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "stopped_step": int(stopped_step),
            "stop_reason": stop_reason,
            "selection": str(selection_path),
        }
    )
    _write_json(manifest_path, stage_manifest)
    if sha256_file(source) != source_hash or sha256_file(gp_path) != gp_hash:
        raise RuntimeError("frozen BDQN-stage input changed during training.")
    return {
        "selection": selection_path,
        "selected_checkpoint": selected_seed[2],
        "convergence": convergence_path,
        "results": results_path,
        "summary": summaries_path,
    }


def evaluate_alternating_stack(
    *,
    model: str,
    gp_policy: str | Path,
    scheduler_checkpoint: str | Path,
    iid_manifest: str | Path,
    ood_manifest: str | Path,
    output_dir: str | Path,
    device: str = "auto",
) -> dict[str, Path]:
    """Evaluate one stack on paired IID/OOD manifests without mutating inputs."""

    destination = Path(output_dir).resolve()
    summary_path = destination / "summary.json"
    if summary_path.is_file():
        return {
            "summary": summary_path,
            "results": destination / "all_results.csv",
        }
    resolved_device = default_device() if device == "auto" else str(device)
    gp_path = Path(gp_policy).resolve()
    scheduler_path = Path(scheduler_checkpoint).resolve()
    before = {
        "gp_policy": sha256_file(gp_path),
        "scheduler_checkpoint": sha256_file(scheduler_path),
    }
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for split, manifest_path in (("iid", iid_manifest), ("ood", ood_manifest)):
        manifest = load_scenario_manifest(manifest_path)
        rows = evaluate_bdqn_provider_cell(
            model=model,
            scheduler_checkpoint=scheduler_path,
            provider_kind="g0",
            scenarios=manifest["scenarios"],
            architecture_checkpoint=None,
            gp_policy=gp_path,
            device=resolved_device,
        )
        for row in rows:
            row["split"] = split
        all_rows.extend(rows)
        summaries[split] = summarize_results(model, rows)
    summaries["combined"] = summarize_results(model, all_rows)
    if any(
        int(summaries[split]["invalid_action_count"]) != 0
        or int(summaries[split]["provider_invariant_violations"]) != 0
        for split in summaries
    ):
        raise RuntimeError("alternating-stack evaluation violated a scheduler invariant.")
    after = {
        "gp_policy": sha256_file(gp_path),
        "scheduler_checkpoint": sha256_file(scheduler_path),
    }
    if after != before:
        raise RuntimeError("stack inputs changed during frozen evaluation.")
    _write_csv(destination / "all_results.csv", all_rows)
    _write_json(
        summary_path,
        {
            "schema_version": ALTERNATION_SCHEMA_VERSION,
            "model": str(model),
            "gp_policy": _input_record(gp_path),
            "scheduler_checkpoint": _input_record(scheduler_path),
            "manifests": {
                "iid": _input_record(iid_manifest),
                "ood": _input_record(ood_manifest),
            },
            "metrics": summaries,
        },
    )
    return {"summary": summary_path, "results": destination / "all_results.csv"}


def _stage_selection_key(
    stage: str, summary: dict[str, Any]
) -> tuple[float, float, float, int]:
    metrics = summary["metrics"]["combined"]
    return (
        float(metrics["failure_rate"]),
        float(metrics["mean_j"]),
        float(metrics["mean_success_makespan"]),
        STAGE_ORDER.index(stage),
    )


def _update_outer_stage(
    manifest_path: Path,
    manifest: dict[str, Any],
    stage: str,
    status: str,
    **details: Any,
) -> None:
    manifest.setdefault("stages", {})[stage] = {
        "status": str(status),
        "updated_at": _utc_now(),
        **details,
    }
    manifest["updated_at"] = _utc_now()
    _write_json(manifest_path, manifest)


def _write_effect_report(
    output_dir: Path,
    stage_results: dict[str, Path],
) -> Path:
    contrasts = (
        ("S0", "SG1", "gp_round_1"),
        ("SG1", "S1", "bdqn_round_1"),
        ("S1", "SG2", "gp_round_2"),
        ("SG2", "S2", "bdqn_round_2"),
        ("S0", "S1", "full_round_1"),
        ("S1", "S2", "full_round_2"),
        ("S0", "S2", "cumulative"),
    )
    effects = {}
    for index, (left, right, label) in enumerate(contrasts):
        effects[label] = {
            "baseline": left,
            "candidate": right,
            "paired": compare_paired_results(
                _read_csv(stage_results[left]),
                _read_csv(stage_results[right]),
                seed=20260920 + index * 10,
            ),
        }
    return _write_json(output_dir / "paired_effects.json", effects)


def run_gp_bdqn_alternation(
    *,
    base_gp_policy: str | Path,
    base_scheduler_checkpoint: str | Path,
    scenario_dir: str | Path,
    gate_iid_manifest: str | Path,
    gate_ood_manifest: str | Path,
    final_iid_manifest: str | Path,
    final_ood_manifest: str | Path,
    output_dir: str | Path,
    gp_population_size: int = 120,
    gp_max_generations: int = 50,
    gp_runs: int = 3,
    gp_workers: int = 12,
    gp_min_generations: int = 20,
    gp_convergence_interval: int = 5,
    gp_base_seed: int = 20260820,
    bdqn_max_env_steps: int = 40000,
    bdqn_checkpoint_interval: int = 5000,
    bdqn_min_convergence_steps: int = 20000,
    bdqn_round1_seeds: Sequence[int] = (4, 5, 6),
    bdqn_round2_seeds: Sequence[int] = (7, 8, 9),
    bdqn_parallel_jobs: int = 3,
    gp_device: str = "cpu",
    bdqn_device: str = "auto",
) -> dict[str, Path]:
    """Execute S0 -> SG1 -> S1 -> SG2 -> S2 without outer promotion gates."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "alternation_manifest.json"
    scenario_root = Path(scenario_dir).resolve()
    inputs = {
        "g0": _input_record(base_gp_policy),
        "b0": _input_record(base_scheduler_checkpoint),
        "train": _input_record(scenario_root / "train.json"),
        "validation": _input_record(scenario_root / "validation.json"),
        "gate_iid": _input_record(gate_iid_manifest),
        "gate_ood": _input_record(gate_ood_manifest),
        # Final manifests are hash-bound here but not parsed until the lineage is locked.
        "final_iid": _input_record(final_iid_manifest),
        "final_ood": _input_record(final_ood_manifest),
    }
    config = {
        "rounds": 2,
        "gp": {
            "population_size": int(gp_population_size),
            "max_generations": int(gp_max_generations),
            "runs": int(gp_runs),
            "workers": int(gp_workers),
            "min_generations": int(gp_min_generations),
            "convergence_interval": int(gp_convergence_interval),
            "convergence_threshold": 0.01,
            "convergence_patience": 2,
            "convergence_confirmation_windows": 1,
            "parent_population_fraction": 0.30,
        },
        "bdqn": {
            "max_env_steps": int(bdqn_max_env_steps),
            "checkpoint_interval": int(bdqn_checkpoint_interval),
            "min_convergence_steps": int(bdqn_min_convergence_steps),
            "round1_seeds": [int(seed) for seed in bdqn_round1_seeds],
            "round2_seeds": [int(seed) for seed in bdqn_round2_seeds],
            "parallel_jobs": int(bdqn_parallel_jobs),
            "lr": 1e-4,
            "lr_end": 1e-5,
            "lr_decay": 0.9975,
        },
    }
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("inputs") != inputs or manifest.get("config") != config:
            raise ValueError("alternation resume inputs or configuration changed.")
        if manifest.get("status") == "complete":
            return {
                "manifest": manifest_path,
                "selection": Path(manifest["selection"]),
                "final_summary": Path(manifest["final_summary"]),
                "paired_effects": Path(manifest["paired_effects"]),
                "report": Path(manifest["report"]),
            }
    else:
        manifest = {
            "schema_version": ALTERNATION_SCHEMA_VERSION,
            "status": "running",
            "created_at": _utc_now(),
            "inputs": inputs,
            "config": config,
            "stages": {},
        }
        _write_json(manifest_path, manifest)

    gp_by_name: dict[str, Path] = {"G0": Path(base_gp_policy).resolve()}
    bdqn_by_name: dict[str, Path] = {
        "B0": Path(base_scheduler_checkpoint).resolve()
    }
    training_scheduler_hash_by_gp = {
        "G0": sha256_file(bdqn_by_name["B0"]),
    }
    stack_definitions: dict[str, tuple[str, str]] = {"S0": ("G0", "B0")}
    gate_summaries: dict[str, Path] = {}
    gate_results: dict[str, Path] = {}

    def evaluate_gate(stage: str) -> None:
        gp_name, bdqn_name = stack_definitions[stage]
        outputs = evaluate_alternating_stack(
            model=stage,
            gp_policy=gp_by_name[gp_name],
            scheduler_checkpoint=bdqn_by_name[bdqn_name],
            iid_manifest=gate_iid_manifest,
            ood_manifest=gate_ood_manifest,
            output_dir=destination / "gate" / stage,
            device=bdqn_device,
        )
        gate_summaries[stage] = outputs["summary"]
        gate_results[stage] = outputs["results"]

    evaluate_gate("S0")
    _update_outer_stage(manifest_path, manifest, "S0_gate", "complete")

    for round_index in (1, 2):
        parent_gp_name = "G0" if round_index == 1 else "G1"
        scheduler_name = "B0" if round_index == 1 else "B1"
        gp_name = f"G{round_index}"
        gp_dir = destination / f"gp_round_{round_index}"
        gp_stage_name = f"train_{gp_name}"
        gp_stage = manifest.get("stages", {}).get(gp_stage_name, {})
        if gp_stage.get("status") == "complete":
            gp_by_name[gp_name] = Path(gp_stage["policy"])
            if sha256_file(gp_by_name[gp_name]) != gp_stage["sha256"]:
                raise ValueError(f"completed {gp_name} policy hash changed.")
        else:
            _update_outer_stage(manifest_path, manifest, gp_stage_name, "running")
            gp_config = GPArchitectureConfig(
                population_size=int(gp_population_size),
                generations=int(gp_max_generations),
                independent_runs=int(gp_runs),
                elite_count=min(2, int(gp_population_size) - 1),
                train_batch_size=16,
                anchor_size=64,
                anchor_interval=int(gp_convergence_interval),
                anchor_top_k=min(10, int(gp_population_size)),
                convergence_interval=int(gp_convergence_interval),
                convergence_threshold=0.01,
                convergence_patience=2,
                convergence_confirmation_windows=1,
                min_generations=min(
                    int(gp_min_generations), int(gp_max_generations)
                ),
                parent_population_fraction=0.30,
                workers=int(gp_workers),
                base_seed=int(gp_base_seed) + (round_index - 1) * 10000,
                feature_set="system_delta",
            )
            gp_outputs = train_gp_architecture(
                scheduler_checkpoint=bdqn_by_name[scheduler_name],
                scheduler_backend="branching-dqn",
                scenario_dir=scenario_root,
                output_dir=gp_dir,
                config=gp_config,
                device=gp_device,
                parent_gp_policy=gp_by_name[parent_gp_name],
                skip_test_evaluation=True,
            )
            gp_by_name[gp_name] = gp_outputs["gp_policy"]
            _update_outer_stage(
                manifest_path,
                manifest,
                gp_stage_name,
                "complete",
                policy=str(gp_outputs["gp_policy"]),
                sha256=sha256_file(gp_outputs["gp_policy"]),
                convergence=str(gp_outputs["convergence"]),
            )
        training_scheduler_hash_by_gp[gp_name] = sha256_file(
            bdqn_by_name[scheduler_name]
        )
        sg_stage = f"SG{round_index}"
        stack_definitions[sg_stage] = (gp_name, scheduler_name)
        evaluate_gate(sg_stage)
        _update_outer_stage(
            manifest_path, manifest, f"{sg_stage}_gate", "complete"
        )

        bdqn_name = f"B{round_index}"
        source_name = "B0" if round_index == 1 else "B1"
        bdqn_dir = destination / f"bdqn_round_{round_index}"
        seeds = bdqn_round1_seeds if round_index == 1 else bdqn_round2_seeds
        bdqn_stage_name = f"train_{bdqn_name}"
        bdqn_stage = manifest.get("stages", {}).get(bdqn_stage_name, {})
        if bdqn_stage.get("status") == "complete":
            bdqn_by_name[bdqn_name] = Path(bdqn_stage["checkpoint"])
            if sha256_file(bdqn_by_name[bdqn_name]) != bdqn_stage["sha256"]:
                raise ValueError(f"completed {bdqn_name} checkpoint hash changed.")
        else:
            _update_outer_stage(
                manifest_path, manifest, bdqn_stage_name, "running"
            )
            bdqn_outputs = train_convergent_bdqn_stage(
                source_checkpoint=bdqn_by_name[source_name],
                gp_policy=gp_by_name[gp_name],
                train_manifest=scenario_root / "train.json",
                validation_manifest=scenario_root / "validation.json",
                output_dir=bdqn_dir,
                seeds=seeds,
                max_env_steps=int(bdqn_max_env_steps),
                checkpoint_interval=int(bdqn_checkpoint_interval),
                min_convergence_steps=int(bdqn_min_convergence_steps),
                device=bdqn_device,
                parallel_jobs=int(bdqn_parallel_jobs),
            )
            bdqn_by_name[bdqn_name] = bdqn_outputs["selected_checkpoint"]
            _update_outer_stage(
                manifest_path,
                manifest,
                bdqn_stage_name,
                "complete",
                checkpoint=str(bdqn_outputs["selected_checkpoint"]),
                sha256=sha256_file(bdqn_outputs["selected_checkpoint"]),
                convergence=str(bdqn_outputs["convergence"]),
            )
        stage = f"S{round_index}"
        stack_definitions[stage] = (gp_name, bdqn_name)
        evaluate_gate(stage)
        _update_outer_stage(manifest_path, manifest, f"{stage}_gate", "complete")

    cross_rows = []
    for gp_name, gp_path in sorted(gp_by_name.items()):
        for bdqn_name, checkpoint in sorted(bdqn_by_name.items()):
            model = f"{gp_name}_{bdqn_name}"
            outputs = evaluate_alternating_stack(
                model=model,
                gp_policy=gp_path,
                scheduler_checkpoint=checkpoint,
                iid_manifest=gate_iid_manifest,
                ood_manifest=gate_ood_manifest,
                output_dir=destination / "crossplay" / model,
                device=bdqn_device,
            )
            record = _read_json(outputs["summary"])
            cross_rows.append(
                {
                    "gp": gp_name,
                    "bdqn": bdqn_name,
                    "binding": (
                        "matched"
                        if training_scheduler_hash_by_gp[gp_name]
                        == sha256_file(checkpoint)
                        else "crossed"
                    ),
                    **record["metrics"]["combined"],
                }
            )
    _write_csv(destination / "crossplay" / "summary.csv", cross_rows)
    _update_outer_stage(manifest_path, manifest, "crossplay", "complete")

    stage_summary_payloads = {
        stage: _read_json(gate_summaries[stage]) for stage in STAGE_ORDER
    }
    selected_stage = min(
        STAGE_ORDER,
        key=lambda stage: _stage_selection_key(stage, stage_summary_payloads[stage]),
    )
    selected_gp_name, selected_bdqn_name = stack_definitions[selected_stage]
    selection = {
        "schema_version": ALTERNATION_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "selection_order": [
            "combined_failure_rate",
            "combined_mean_failure_aware_j",
            "combined_mean_success_makespan",
            "earlier_stage",
        ],
        "selected_stage": selected_stage,
        "selected_gp": {
            "name": selected_gp_name,
            **_input_record(gp_by_name[selected_gp_name]),
        },
        "selected_bdqn": {
            "name": selected_bdqn_name,
            **_input_record(bdqn_by_name[selected_bdqn_name]),
        },
        "stage_metrics": {
            stage: stage_summary_payloads[stage]["metrics"] for stage in STAGE_ORDER
        },
        "final_test_locked": True,
    }
    selection_path = _write_json(destination / "selection.json", selection)
    effects_path = _write_effect_report(destination, gate_results)
    _update_outer_stage(manifest_path, manifest, "gate_lock", "complete")

    final_summaries = {}
    for stage in STAGE_ORDER:
        gp_name, bdqn_name = stack_definitions[stage]
        outputs = evaluate_alternating_stack(
            model=stage,
            gp_policy=gp_by_name[gp_name],
            scheduler_checkpoint=bdqn_by_name[bdqn_name],
            iid_manifest=final_iid_manifest,
            ood_manifest=final_ood_manifest,
            output_dir=destination / "final" / stage,
            device=bdqn_device,
        )
        final_summaries[stage] = _read_json(outputs["summary"])["metrics"]
    final_summary_path = _write_json(
        destination / "final" / "summary.json", final_summaries
    )
    _update_outer_stage(manifest_path, manifest, "final_test", "complete")

    report_lines = [
        "# GP–BDQN two-round alternation",
        "",
        f"Selected gate stack: `{selected_stage}` (`{selected_gp_name}+{selected_bdqn_name}`).",
        "",
        "|Stage|Failure rate|Mean J|Successful makespan|Final cost|Peak cost|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGE_ORDER:
        metrics = final_summaries[stage]["combined"]
        report_lines.append(
            "|{stage}|{failure_rate:.6f}|{mean_j:.6f}|"
            "{mean_success_makespan:.3f}|{mean_final_cost:.3f}|{mean_peak_cost:.3f}|".format(
                stage=stage, **metrics
            )
        )
    report_path = destination / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    for name, record in inputs.items():
        if sha256_file(record["path"]) != record["sha256"]:
            raise RuntimeError(f"alternation input {name!r} changed during the run.")
    manifest.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "lineage": {
                stage: {
                    "gp": stack_definitions[stage][0],
                    "bdqn": stack_definitions[stage][1],
                }
                for stage in STAGE_ORDER
            },
            "selection": str(selection_path),
            "final_summary": str(final_summary_path),
            "paired_effects": str(effects_path),
            "report": str(report_path),
        }
    )
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "selection": selection_path,
        "final_summary": final_summary_path,
        "paired_effects": effects_path,
        "report": report_path,
    }
