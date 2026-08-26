"""Algorithm-neutral helpers for auditable multi-objective artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    from .nsga2.model import DecodeResult


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            json_safe(value),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        if isinstance(value, (dict, list, tuple, np.ndarray))
                        else json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def git_commit() -> str | None:
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


def front_row(
    result: "DecodeResult",
    *,
    scenario_idx: int,
    scenario_hash: str,
    source_seeds: Sequence[int],
    particle_position: np.ndarray | None = None,
    random_keys: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    row = {
        "scenario_idx": int(scenario_idx),
        "scenario_hash": scenario_hash,
        "source_run_seeds": [int(seed) for seed in source_seeds],
        **result.summary(),
        "os": result.chromosome.os,
        "ms": result.chromosome.ms,
        "aa": result.chromosome.aa,
        "effective_os": result.effective_os,
        "effective_ms": result.effective_ms,
        "effective_aa": result.effective_aa,
    }
    if particle_position is not None:
        row["particle_position"] = np.asarray(
            particle_position, dtype=np.float64
        )
    if random_keys is not None:
        row.update(random_keys)
    return row


def representative_payload(
    result: "DecodeResult",
    *,
    particle_position: np.ndarray | None = None,
    random_keys: dict[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    payload = {
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
    if particle_position is not None:
        payload["particle_position"] = np.asarray(
            particle_position, dtype=np.float64
        )
    if random_keys is not None:
        payload["random_keys"] = random_keys
    return payload


def milestone_summary_row(
    *,
    milestone: int,
    front: Sequence["DecodeResult"],
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

    from .nsga2.solver import select_representatives

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
    row.update(
        {
            "min_makespan": float(representatives["min_makespan"].makespan),
            "min_effective_cost": float(
                representatives["min_cost"].effective_cost
            ),
            "compromise_makespan": float(compromise.makespan),
            "compromise_effective_cost": float(compromise.effective_cost),
            "gp_aligned_j": float(
                10.0 * float(gp_aligned.makespan) / scale
                + float(gp_aligned.effective_cost) / budget
            ),
            "gp_aligned_makespan": float(gp_aligned.makespan),
            "gp_aligned_effective_cost": float(gp_aligned.effective_cost),
            "gp_aligned_final_net_cost": float(gp_aligned.final_net_cost),
            "gp_aligned_phenotype_hash": gp_aligned.phenotype_hash,
        }
    )
    return row
