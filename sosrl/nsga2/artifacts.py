"""Scenario-manifest orchestration and auditable NSGA-II artifacts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..gp.architecture import architecture_action_table_hash
from ..multiobjective_artifacts import (
    front_row as _shared_front_row,
    git_commit as _shared_git_commit,
    milestone_summary_row as _shared_milestone_summary_row,
    representative_payload as _shared_representative_payload,
    write_csv as _shared_write_csv,
    write_json as _shared_write_json,
)
from ..workflows import evaluation
from .config import NSGA2Config
from .model import DecodeResult
from .solver import (
    NSGA2ScenarioResult,
    select_representatives,
    solve_scenario_nsga2,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            encoded = {
                key: (
                    json.dumps(
                        _json_safe(value),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _json_safe(value)
                )
                for key, value in row.items()
            }
            writer.writerow(encoded)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _front_row(
    result: DecodeResult,
    *,
    scenario_idx: int,
    scenario_hash: str,
    source_seeds: list[int],
) -> dict[str, Any]:
    return {
        "scenario_idx": int(scenario_idx),
        "scenario_hash": scenario_hash,
        "source_run_seeds": source_seeds,
        **result.summary(),
        "os": result.chromosome.os,
        "ms": result.chromosome.ms,
        "aa": result.chromosome.aa,
        "effective_os": result.effective_os,
        "effective_ms": result.effective_ms,
        "effective_aa": result.effective_aa,
    }


def _representative_payload(result: DecodeResult) -> dict[str, Any]:
    return {
        **result.summary(),
        "chromosome": {
            "os": result.chromosome.os,
            "ms": result.chromosome.ms,
            "aa": result.chromosome.aa,
        },
        "effective_phenotype": {
            "os": result.effective_os,
            "ms": result.effective_ms,
            "aa": result.effective_aa,
        },
    }


def _milestone_summary_row(
    *,
    milestone: int,
    front: tuple[DecodeResult, ...],
    scenario_idx: int,
    scenario: dict[str, Any],
    run_count: int,
) -> dict[str, Any]:
    scale = max(
        1.0,
        float(
            sum(
                operation["duration"]
                for task in scenario["mission"]
                for operation in task["operations"]
            )
        ),
    )
    budget = max(float(scenario.get("budget", 8000.0)), 1.0)
    row = {
        "scenario_idx": int(scenario_idx),
        "scenario_hash": str(scenario["scenario_hash"]),
        "split": scenario.get("split"),
        "category": scenario.get("category"),
        "evaluation_budget_per_run": int(milestone),
        "independent_runs": int(run_count),
        "total_evaluations": int(milestone) * int(run_count),
        "success": bool(front),
        "pareto_front_size": len(front),
        "mission_scale": scale,
        "budget": budget,
    }
    if not front:
        row.update(
            {
                "min_makespan": None,
                "min_effective_cost": None,
                "compromise_makespan": None,
                "compromise_effective_cost": None,
                "gp_aligned_j": None,
                "gp_aligned_makespan": None,
                "gp_aligned_effective_cost": None,
                "gp_aligned_final_net_cost": None,
                "gp_aligned_phenotype_hash": None,
            }
        )
        return row

    representatives = select_representatives(front)
    compromise = representatives["compromise"]
    gp_aligned = min(
        front,
        key=lambda result: (
            10.0 * float(result.makespan) / scale
            + float(result.effective_cost) / budget,
            result.makespan,
            result.effective_cost,
            result.final_net_cost,
            result.phenotype_hash,
        ),
    )
    gp_aligned_j = (
        10.0 * float(gp_aligned.makespan) / scale
        + float(gp_aligned.effective_cost) / budget
    )
    row.update(
        {
            "min_makespan": float(representatives["min_makespan"].makespan),
            "min_effective_cost": float(
                representatives["min_cost"].effective_cost
            ),
            "compromise_makespan": float(compromise.makespan),
            "compromise_effective_cost": float(compromise.effective_cost),
            "gp_aligned_j": float(gp_aligned_j),
            "gp_aligned_makespan": float(gp_aligned.makespan),
            "gp_aligned_effective_cost": float(gp_aligned.effective_cost),
            "gp_aligned_final_net_cost": float(gp_aligned.final_net_cost),
            "gp_aligned_phenotype_hash": gp_aligned.phenotype_hash,
        }
    )
    return row


# Keep the established NSGA-II private names while routing artifact formatting
# through the algorithm-neutral implementation.  This preserves its on-disk
# schema and compatibility for callers that already rely on this module.
_write_json = _shared_write_json
_write_csv = _shared_write_csv
_git_commit = _shared_git_commit
_front_row = _shared_front_row
_representative_payload = _shared_representative_payload
_milestone_summary_row = _shared_milestone_summary_row


def write_scenario_artifacts(
    output_dir: Path,
    *,
    scenario_idx: int,
    scenario: dict[str, Any],
    result: NSGA2ScenarioResult,
    config: NSGA2Config,
    started_at: str,
    wall_seconds: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_hash = str(scenario["scenario_hash"])
    source_seeds: dict[str, list[int]] = {}
    for run in result.runs:
        for item in run.front:
            source_seeds.setdefault(item.phenotype_hash, []).append(int(run.seed))
    front_rows = [
        _front_row(
            item,
            scenario_idx=scenario_idx,
            scenario_hash=scenario_hash,
            source_seeds=source_seeds.get(item.phenotype_hash, []),
        )
        for item in result.combined_front
    ]
    _write_csv(output_dir / "pareto_front.csv", front_rows)

    history_rows = []
    for run in result.runs:
        history_rows.extend(
            {"run_seed": int(run.seed), **row}
            for row in run.history
        )
    _write_csv(output_dir / "generation_history.csv", history_rows)
    _write_json(
        output_dir / "selected_solutions.json",
        {
            role: _representative_payload(item)
            for role, item in result.representatives.items()
        },
    )

    milestone_summary_rows = []
    for milestone, milestone_front in sorted(result.milestone_fronts.items()):
        milestone_sources: dict[str, list[int]] = {}
        for run in result.runs:
            for item in run.milestone_fronts[milestone]:
                milestone_sources.setdefault(item.phenotype_hash, []).append(
                    int(run.seed)
                )
        milestone_dir = output_dir / "milestones" / f"eval_{milestone:06d}"
        _write_csv(
            milestone_dir / "pareto_front.csv",
            [
                _front_row(
                    item,
                    scenario_idx=scenario_idx,
                    scenario_hash=scenario_hash,
                    source_seeds=milestone_sources.get(item.phenotype_hash, []),
                )
                for item in milestone_front
            ],
        )
        _write_json(
            milestone_dir / "selected_solutions.json",
            {
                role: _representative_payload(item)
                for role, item in select_representatives(
                    milestone_front
                ).items()
            },
        )
        milestone_summary_rows.append(
            _milestone_summary_row(
                milestone=milestone,
                front=milestone_front,
                scenario_idx=scenario_idx,
                scenario=scenario,
                run_count=len(result.runs),
            )
        )
    _write_csv(output_dir / "milestone_summary.csv", milestone_summary_rows)

    compromise = result.representatives.get("compromise")
    _write_csv(
        output_dir / "schedule.csv",
        []
        if compromise is None
        else sorted(
            compromise.schedule,
            key=lambda row: (row["start_time"], row["task_idx"], row["op_idx"]),
        ),
    )
    _write_csv(
        output_dir / "architecture_trace.csv",
        [] if compromise is None else compromise.architecture_trace,
    )

    summary_rows = []
    if compromise is not None:
        min_makespan = result.representatives["min_makespan"]
        min_cost = result.representatives["min_cost"]
        summary_rows.append(
            {
                "model": "nsga2",
                "solution_role": "compromise",
                "scenario_idx": int(scenario_idx),
                "scenario_hash": scenario_hash,
                "category": scenario.get("category"),
                "architecture_size": len(
                    scenario["architecture_system_indices"]
                ),
                "architecture_cost": float(scenario["architecture_cost"]),
                "success": bool(compromise.success),
                "dead_end": bool(compromise.dead_end),
                "makespan": float(compromise.makespan),
                "net_cost": float(compromise.final_net_cost),
                "effective_cost": float(compromise.effective_cost),
                "gp_cost_score": float(compromise.gp_cost_score),
                "architecture_change_penalty": float(
                    compromise.architecture_change_penalty
                ),
                "peak_budget_penalty": float(compromise.peak_budget_penalty),
                "assigned_ops": int(compromise.completed_operations),
                "budget_violation": bool(
                    compromise.metrics["final_over_budget"]
                ),
                "pareto_front_size": len(result.combined_front),
                "min_makespan": float(min_makespan.makespan),
                "min_makespan_effective_cost": float(
                    min_makespan.effective_cost
                ),
                "min_makespan_net_cost": float(min_makespan.final_net_cost),
                "min_cost": float(min_cost.effective_cost),
                "min_cost_net_cost": float(min_cost.final_net_cost),
                "min_cost_makespan": float(min_cost.makespan),
                **compromise.metrics,
                **compromise.repair_counts,
            }
        )
    _write_csv(output_dir / "scenario_summary.csv", summary_rows)

    manifest = {
        "schema_version": 2,
        "algorithm": "dynamic_architecture_scheduling_nsga2",
        "encoding_version": "os-ms-aa-gp203-v1",
        "architecture_action_table_hash": architecture_action_table_hash(),
        "gp_aligned_cost_defaults": config.gp_aligned_cost_defaults,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "wall_seconds": float(wall_seconds),
        "scenario_idx": int(scenario_idx),
        "scenario_hash": scenario_hash,
        "category": scenario.get("category"),
        "budget": float(scenario.get("budget", 8000.0)),
        "refund_rate": float(scenario.get("refund_rate", 0.8)),
        "initial_architecture_system_indices": scenario[
            "architecture_system_indices"
        ],
        "config": config.to_dict(),
        "run_seeds": [int(run.seed) for run in result.runs],
        "run_evaluations": [int(run.evaluations) for run in result.runs],
        "run_wall_seconds": [float(run.wall_seconds) for run in result.runs],
        "pymoo_version": None if not result.runs else result.runs[0].pymoo_version,
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pymoo": None if not result.runs else result.runs[0].pymoo_version,
        },
        "git_commit": _git_commit(),
        "combined_front_size": len(result.combined_front),
        "milestone_front_sizes": {
            str(milestone): len(front)
            for milestone, front in result.milestone_fronts.items()
        },
        "representative_roles": sorted(result.representatives),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return {
        "scenario_idx": int(scenario_idx),
        "scenario_hash": scenario_hash,
        "output_dir": str(output_dir.resolve()),
        "front_size": len(result.combined_front),
        "summary_rows": summary_rows,
        "milestone_summary_rows": milestone_summary_rows,
    }


def load_scenario_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        payload = {"scenarios": payload}
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenario manifest must contain a non-empty scenarios list.")
    for scenario in scenarios:
        evaluation.verify_scenario_payload(scenario)
    return payload


def solve_manifest_nsga2(
    scenario_manifest: str | Path,
    output_dir: str | Path,
    *,
    config: NSGA2Config | None = None,
    scenario_indices: Iterable[int] | None = None,
    refund_rate: float | None = None,
) -> dict[str, Any]:
    config = config or NSGA2Config()
    payload = load_scenario_manifest(scenario_manifest)
    scenarios = payload["scenarios"]
    indices = (
        list(range(len(scenarios)))
        if scenario_indices is None
        else [int(value) for value in scenario_indices]
    )
    if not indices:
        raise ValueError("at least one scenario index is required.")
    if any(index < 0 or index >= len(scenarios) for index in indices):
        raise ValueError("scenario index is outside the manifest.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    def solve_one(scenario_idx: int) -> dict[str, Any]:
        scenario = scenarios[scenario_idx]
        architecture, mission = evaluation.scenario_from_payload(scenario)
        scenario_refund = (
            float(refund_rate)
            if refund_rate is not None
            else float(scenario.get("refund_rate", 0.8))
        )
        started_at = datetime.now(timezone.utc).isoformat()
        started = perf_counter()
        result = solve_scenario_nsga2(
            architecture,
            mission,
            budget=float(scenario.get("budget", 8000.0)),
            refund_rate=scenario_refund,
            config=config,
        )
        wall_seconds = perf_counter() - started
        scenario_dir = output_dir / (
            f"scenario_{scenario_idx:05d}_{str(scenario['scenario_hash'])[:8]}"
        )
        return write_scenario_artifacts(
            scenario_dir,
            scenario_idx=scenario_idx,
            scenario={**scenario, "refund_rate": scenario_refund},
            result=result,
            config=config,
            started_at=started_at,
            wall_seconds=wall_seconds,
        )

    if config.workers == 1 or len(indices) == 1:
        solved_records = [solve_one(scenario_idx) for scenario_idx in indices]
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            solved_records = list(executor.map(solve_one, indices))

    summaries = []
    milestone_summaries = []
    records = []
    for record in solved_records:
        records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"summary_rows", "milestone_summary_rows"}
            }
        )
        summaries.extend(record["summary_rows"])
        milestone_summaries.extend(record["milestone_summary_rows"])

    _write_csv(output_dir / "scenario_summary.csv", summaries)
    _write_csv(output_dir / "milestone_summary.csv", milestone_summaries)
    root_manifest = {
        "schema_version": 2,
        "algorithm": "dynamic_architecture_scheduling_nsga2",
        "encoding_version": "os-ms-aa-gp203-v1",
        "architecture_action_table_hash": architecture_action_table_hash(),
        "gp_aligned_cost_defaults": config.gp_aligned_cost_defaults,
        "source_manifest": str(Path(scenario_manifest).resolve()),
        "source_manifest_hash": payload.get("manifest_hash"),
        "config": config.to_dict(),
        "scenario_indices": indices,
        "scenarios": records,
    }
    _write_json(output_dir / "nsga2_manifest.json", root_manifest)
    return root_manifest
