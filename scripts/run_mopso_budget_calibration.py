"""Run resumable, sequential MOPSO budget-calibration shards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from sosrl.mopso import MOPSOConfig, solve_manifest_mopso
from sosrl.multiobjective_calibration import run_calibration


MILESTONES = (50, 100, 150, 200, 300, 400, 500)


def write_status(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def complete_shard(shard_dir: Path, *, scenario_idx: int) -> bool:
    root_manifest = shard_dir / "mopso_manifest.json"
    if not root_manifest.is_file():
        return False
    try:
        root = json.loads(root_manifest.read_text(encoding="utf-8"))
        if root.get("scenario_indices") != [scenario_idx]:
            return False
        scenario_dirs = [
            path for path in shard_dir.glob("scenario_*") if path.is_dir()
        ]
        if len(scenario_dirs) != 1:
            return False
        run = json.loads(
            (scenario_dirs[0] / "run_manifest.json").read_text(encoding="utf-8")
        )
        return bool(
            run.get("run_evaluations") == [500, 500, 500]
            and sorted(int(value) for value in run.get("milestone_front_sizes", {}))
            == list(MILESTONES)
            and (scenario_dirs[0] / "milestone_summary.csv").is_file()
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b-manifest",
        type=Path,
        default=Path("runs/round1_formal/scenarios/b/validation.json"),
    )
    parser.add_argument(
        "--g-manifest",
        type=Path,
        default=Path("runs/round1_formal/scenarios/g/validation.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/mopso_budget_calibration_20260825"),
    )
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")

    config = MOPSOConfig(
        swarm_size=50,
        max_evaluations=500,
        independent_runs=3,
        evaluation_milestones=MILESTONES,
        base_seed=20260825,
        workers=1,
    )
    shards = [
        (split, index, manifest)
        for split, manifest in (("b", args.b_manifest), ("g", args.g_manifest))
        for index in range(4)
    ]
    completed_dirs = []
    for split, scenario_idx, manifest in shards:
        shard_dir = args.output_dir / "shards" / f"{split}_{scenario_idx}"
        status_path = shard_dir / "shard_status.json"
        if complete_shard(shard_dir, scenario_idx=scenario_idx):
            previous = {}
            if status_path.is_file():
                previous = json.loads(status_path.read_text(encoding="utf-8"))
            write_status(
                status_path,
                {
                    **previous,
                    "status": "complete",
                    "split": split,
                    "scenario_idx": scenario_idx,
                    "skipped_complete": True,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            completed_dirs.append(shard_dir)
            print(f"skip complete shard {split}_{scenario_idx}", flush=True)
            continue

        attempts = 0
        last_error = None
        started = perf_counter()
        for attempt in range(args.retries + 1):
            attempts = attempt + 1
            try:
                print(
                    f"run shard {split}_{scenario_idx} attempt {attempts}",
                    flush=True,
                )
                solve_manifest_mopso(
                    manifest,
                    shard_dir,
                    config=config,
                    scenario_indices=[scenario_idx],
                )
                if not complete_shard(shard_dir, scenario_idx=scenario_idx):
                    raise RuntimeError("shard outputs failed completeness check")
                last_error = None
                break
            except Exception as error:  # preserve status before trying next shard
                last_error = f"{type(error).__name__}: {error}"
                write_status(
                    status_path,
                    {
                        "status": "retrying" if attempt < args.retries else "failed",
                        "split": split,
                        "scenario_idx": scenario_idx,
                        "attempts": attempts,
                        "last_error": last_error,
                        "wall_seconds": perf_counter() - started,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        status = "complete" if last_error is None else "failed"
        write_status(
            status_path,
            {
                "status": status,
                "split": split,
                "scenario_idx": scenario_idx,
                "attempts": attempts,
                "last_error": last_error,
                "wall_seconds": perf_counter() - started,
                "skipped_complete": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if status == "complete":
            completed_dirs.append(shard_dir)
        else:
            print(f"failed shard {split}_{scenario_idx}: {last_error}", flush=True)

    if len(completed_dirs) != len(shards):
        raise SystemExit(
            f"only {len(completed_dirs)}/{len(shards)} shards completed; "
            "rerun to resume"
        )
    summary = run_calibration(
        completed_dirs,
        args.output_dir,
        algorithm="MOPSO-CD",
        max_budget=500,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

