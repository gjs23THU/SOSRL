"""Check fixed-seed HRL behavior and CPU evaluation performance."""

import argparse
import json
from pathlib import Path
import time

from sosrl.rl.checkpoint import load_combined_checkpoint
from sosrl.workflows import hierarchical, scheduler


FINGERPRINT_FIELDS = (
    "scenario_hash",
    "category",
    "success",
    "dead_end",
    "makespan",
    "net_cost",
    "active_cost",
    "total_refund",
    "architecture_changes",
    "budget_violation",
    "assigned_ops",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-seconds", type=float, default=4.97)
    return parser.parse_args()


def fingerprint(row):
    result = {key: row[key] for key in FINGERPRINT_FIELDS}
    result["architecture_rule_counts"] = {
        key: value
        for key, value in row.items()
        if key.startswith("arch_") and key.endswith("_count")
    }
    result["scheduler_rule_counts"] = {
        key: value
        for key, value in row.items()
        if key.startswith("schedule_") and key.endswith("_count")
    }
    return result


def main() -> None:
    args = parse_args()
    architecture_agent, scheduler_agent, _ = load_combined_checkpoint(
        args.checkpoint,
        device=args.device,
        load_optimizer=False,
    )
    config = architecture_agent.config
    scheduler.set_seed(args.seed)
    scenarios = hierarchical.AdaptiveScenarioPool(
        args.episodes,
        config,
    ).scenarios
    start = time.perf_counter()
    rows = hierarchical.evaluate_hrl(
        architecture_agent,
        scheduler_agent,
        scenarios,
        budget=config.budget,
        refund_rate=config.refund_rate,
    )
    elapsed = time.perf_counter() - start
    actual = [fingerprint(row) for row in rows]

    if args.expected is not None:
        expected = json.loads(args.expected.read_text(encoding="utf-8"))
        if actual != expected:
            raise AssertionError("fixed-seed HRL behavior differs from snapshot")
    if elapsed > args.max_seconds:
        raise AssertionError(
            f"evaluation took {elapsed:.3f}s; limit is {args.max_seconds:.3f}s"
        )
    print(
        json.dumps(
            {"elapsed_seconds": elapsed, "fingerprint": actual},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
