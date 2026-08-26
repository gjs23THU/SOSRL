"""Replay MOPSO Pareto and representative artifacts through the shared decoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from sosrl.mopso import RandomKeyCodec
from sosrl.nsga2 import Chromosome, DynamicScheduleDecoder
from sosrl.workflows import evaluation


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def chromosome(payload: dict[str, object]) -> Chromosome:
    return Chromosome(
        np.asarray(payload["os"], dtype=np.int32),
        np.asarray(payload["ms"], dtype=np.int32),
        np.asarray(payload["aa"], dtype=np.int32),
    )


def assert_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise AssertionError(f"{name} mismatch: {actual} != {expected}")


def replay_row(
    row: dict[str, str],
    decoder: DynamicScheduleDecoder,
    codec: RandomKeyCodec,
) -> None:
    encoded = chromosome(
        {
            name: json.loads(row[name])
            for name in ("os", "ms", "aa")
        }
    )
    position = np.asarray(json.loads(row["particle_position"]), dtype=float)
    if not np.array_equal(codec.decode(position).flat, encoded.flat):
        raise AssertionError("particle position does not decode to saved chromosome")
    replayed = decoder.decode(encoded)
    if replayed.phenotype_hash != row["phenotype_hash"]:
        raise AssertionError("replayed phenotype hash differs from artifact")
    assert_close(replayed.makespan, float(row["makespan"]), "makespan")
    assert_close(
        replayed.effective_cost,
        float(row["effective_cost"]),
        "effective_cost",
    )
    assert_close(
        replayed.constraint_violation,
        float(row["constraint_violation"]),
        "constraint_violation",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = json.loads(
        (args.output_dir / "mopso_manifest.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        Path(root["source_manifest"]).read_text(encoding="utf-8")
    )
    config = root["config"]
    replayed_fronts = 0
    replayed_representatives = 0
    for record in root["scenarios"]:
        scenario = source["scenarios"][int(record["scenario_idx"])]
        architecture, mission = evaluation.scenario_from_payload(scenario)
        decoder = DynamicScheduleDecoder(
            architecture,
            mission,
            budget=float(scenario.get("budget", 8000.0)),
            refund_rate=float(scenario.get("refund_rate", 0.8)),
            architecture_change_weight=float(
                config["architecture_change_weight"]
            ),
            peak_budget_penalty=float(config["peak_budget_penalty"]),
        )
        codec = RandomKeyCodec(decoder.layout)
        scenario_dir = Path(record["output_dir"])
        front_paths = [scenario_dir / "pareto_front.csv"] + sorted(
            scenario_dir.glob("milestones/eval_*/pareto_front.csv")
        )
        for front_path in front_paths:
            for row in read_csv(front_path):
                if float(row["constraint_violation"]) != 0.0:
                    raise AssertionError(
                        f"nonzero CV in reported front: {front_path}"
                    )
                replay_row(row, decoder, codec)
                replayed_fronts += 1
        selected = json.loads(
            (scenario_dir / "selected_solutions.json").read_text(
                encoding="utf-8"
            )
        )
        for payload in selected.values():
            encoded = chromosome(payload["chromosome"])
            position = np.asarray(payload["particle_position"], dtype=float)
            if not np.array_equal(codec.decode(position).flat, encoded.flat):
                raise AssertionError(
                    "representative position does not match its chromosome"
                )
            result = decoder.decode(encoded)
            if result.phenotype_hash != payload["phenotype_hash"]:
                raise AssertionError("representative phenotype hash mismatch")
            assert_close(result.makespan, payload["makespan"], "makespan")
            assert_close(
                result.effective_cost,
                payload["effective_cost"],
                "effective_cost",
            )
            replayed_representatives += 1
    print(
        json.dumps(
            {
                "scenarios": len(root["scenarios"]),
                "replayed_front_rows": replayed_fronts,
                "replayed_representatives": replayed_representatives,
                "all_reported_fronts_cv_zero": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

