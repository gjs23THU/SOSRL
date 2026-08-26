"""Scenario orchestration and auditable artifacts for random-key MOPSO."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np

from ..gp.architecture import architecture_action_table_hash
from ..multiobjective_artifacts import (
    front_row,
    git_commit,
    milestone_summary_row,
    representative_payload,
    write_csv,
    write_json,
)
from ..nsga2.decoder import DynamicScheduleDecoder
from ..nsga2.model import DecodeResult
from ..nsga2.solver import select_representatives
from ..workflows import evaluation
from .codec import RandomKeyCodec
from .config import MOPSOConfig
from .solver import MOPSOScenarioResult, solve_scenario_mopso


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


def _keys(
    codec: RandomKeyCodec,
    position: np.ndarray,
) -> dict[str, Sequence[float]]:
    return codec.split(position)


def _result_row(
    result: DecodeResult,
    *,
    codec: RandomKeyCodec,
    positions: dict[str, np.ndarray],
    scenario_idx: int,
    scenario_hash: str,
    source_seeds: Sequence[int],
) -> dict[str, Any]:
    position = positions[result.phenotype_hash]
    return front_row(
        result,
        scenario_idx=scenario_idx,
        scenario_hash=scenario_hash,
        source_seeds=source_seeds,
        particle_position=position,
        random_keys=_keys(codec, position),
    )


def _representative(
    result: DecodeResult,
    *,
    codec: RandomKeyCodec,
    positions: dict[str, np.ndarray],
) -> dict[str, Any]:
    position = positions[result.phenotype_hash]
    return representative_payload(
        result,
        particle_position=position,
        random_keys=_keys(codec, position),
    )


def write_scenario_artifacts(
    output_dir: Path,
    *,
    scenario_idx: int,
    scenario: dict[str, Any],
    result: MOPSOScenarioResult,
    codec: RandomKeyCodec,
    config: MOPSOConfig,
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

    write_csv(
        output_dir / "pareto_front.csv",
        [
            _result_row(
                item,
                codec=codec,
                positions=result.positions,
                scenario_idx=scenario_idx,
                scenario_hash=scenario_hash,
                source_seeds=source_seeds.get(item.phenotype_hash, ()),
            )
            for item in result.combined_front
        ],
    )
    write_csv(
        output_dir / "iteration_history.csv",
        [
            {"run_seed": int(run.seed), **row}
            for run in result.runs
            for row in run.history
        ],
    )
    write_json(
        output_dir / "selected_solutions.json",
        {
            role: _representative(
                item, codec=codec, positions=result.positions
            )
            for role, item in result.representatives.items()
        },
    )

    milestone_rows = []
    for milestone, milestone_front in sorted(result.milestone_fronts.items()):
        milestone_sources: dict[str, list[int]] = {}
        for run in result.runs:
            for item in run.milestone_fronts[milestone]:
                milestone_sources.setdefault(item.phenotype_hash, []).append(
                    int(run.seed)
                )
        milestone_dir = output_dir / "milestones" / f"eval_{milestone:06d}"
        write_csv(
            milestone_dir / "pareto_front.csv",
            [
                _result_row(
                    item,
                    codec=codec,
                    positions=result.positions,
                    scenario_idx=scenario_idx,
                    scenario_hash=scenario_hash,
                    source_seeds=milestone_sources.get(item.phenotype_hash, ()),
                )
                for item in milestone_front
            ],
        )
        write_json(
            milestone_dir / "selected_solutions.json",
            {
                role: _representative(
                    item, codec=codec, positions=result.positions
                )
                for role, item in select_representatives(
                    milestone_front
                ).items()
            },
        )
        milestone_rows.append(
            milestone_summary_row(
                milestone=milestone,
                front=milestone_front,
                scenario_idx=scenario_idx,
                scenario=scenario,
                run_count=len(result.runs),
            )
        )
    write_csv(output_dir / "milestone_summary.csv", milestone_rows)

    compromise = result.representatives.get("compromise")
    write_csv(
        output_dir / "schedule.csv",
        []
        if compromise is None
        else sorted(
            compromise.schedule,
            key=lambda row: (
                row["start_time"],
                row["task_idx"],
                row["op_idx"],
            ),
        ),
    )
    write_csv(
        output_dir / "architecture_trace.csv",
        [] if compromise is None else compromise.architecture_trace,
    )

    summary_rows = []
    if compromise is not None:
        min_makespan = result.representatives["min_makespan"]
        min_cost = result.representatives["min_cost"]
        summary_rows.append(
            {
                "model": "mopso",
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
    write_csv(output_dir / "scenario_summary.csv", summary_rows)

    manifest = {
        "schema_version": 2,
        "algorithm": "dynamic_architecture_scheduling_mopso_cd",
        "encoding_version": "random-key-os-ms-aa-gp203-v1",
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
        "git_commit": git_commit(),
        "combined_front_size": len(result.combined_front),
        "milestone_front_sizes": {
            str(milestone): len(front)
            for milestone, front in result.milestone_fronts.items()
        },
        "representative_roles": sorted(result.representatives),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return {
        "scenario_idx": int(scenario_idx),
        "scenario_hash": scenario_hash,
        "output_dir": str(output_dir.resolve()),
        "front_size": len(result.combined_front),
        "summary_rows": summary_rows,
        "milestone_summary_rows": milestone_rows,
    }


def solve_manifest_mopso(
    scenario_manifest: str | Path,
    output_dir: str | Path,
    *,
    config: MOPSOConfig | None = None,
    scenario_indices: Iterable[int] | None = None,
    refund_rate: float | None = None,
) -> dict[str, Any]:
    config = config or MOPSOConfig()
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
        result = solve_scenario_mopso(
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
        codec = RandomKeyCodec(
            DynamicScheduleDecoder(architecture, mission).layout
        )
        return write_scenario_artifacts(
            scenario_dir,
            scenario_idx=scenario_idx,
            scenario={**scenario, "refund_rate": scenario_refund},
            result=result,
            codec=codec,
            config=config,
            started_at=started_at,
            wall_seconds=wall_seconds,
        )

    if config.workers == 1 or len(indices) == 1:
        solved_records = [solve_one(index) for index in indices]
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
    write_csv(output_dir / "scenario_summary.csv", summaries)
    write_csv(output_dir / "milestone_summary.csv", milestone_summaries)
    root_manifest = {
        "schema_version": 2,
        "algorithm": "dynamic_architecture_scheduling_mopso_cd",
        "encoding_version": "random-key-os-ms-aa-gp203-v1",
        "architecture_action_table_hash": architecture_action_table_hash(),
        "gp_aligned_cost_defaults": config.gp_aligned_cost_defaults,
        "source_manifest": str(Path(scenario_manifest).resolve()),
        "source_manifest_hash": payload.get("manifest_hash"),
        "config": config.to_dict(),
        "scenario_indices": indices,
        "scenarios": records,
    }
    write_json(output_dir / "mopso_manifest.json", root_manifest)
    return root_manifest

