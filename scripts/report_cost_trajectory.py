"""Re-evaluate an HRL checkpoint and export every architecture cost event."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sosrl.environment import MissionEnv
from sosrl.rl.checkpoint import load_combined_checkpoint
from sosrl.workflows import hierarchical, scheduler


class CostEventEnvironment(MissionEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_events = []
        self._record_cost("initial")

    def _record_cost(self, event: str) -> None:
        self.cost_events.append({"event": event, **self.cost_metrics()})

    def add_system(self, sys_idx: int, *, refresh: bool = True):
        result = super().add_system(sys_idx, refresh=refresh)
        if result.get("valid", False):
            self._record_cost("add")
        return result

    def remove_system(self, sys_idx: int, *, refresh: bool = True):
        result = super().remove_system(sys_idx, refresh=refresh)
        if result.get("valid", False):
            self._record_cost("remove")
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=20260724)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    architecture_agent, scheduler_agent, _ = load_combined_checkpoint(
        args.checkpoint,
        device=args.device,
        load_optimizer=False,
    )
    config = architecture_agent.config
    scheduler.set_seed(args.eval_seed)
    scenarios = hierarchical.AdaptiveScenarioPool(
        args.episodes,
        config,
    ).scenarios

    summaries = []
    events = []
    for episode, (architecture, mission, category) in enumerate(scenarios):
        mission_env = CostEventEnvironment(
            architecture,
            mission,
            adaptive=True,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        result = hierarchical.run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=0.0,
            scheduler_epsilon=0.0,
            update_architecture=False,
            update_scheduler=False,
            store_experience=False,
        )
        scenario_id = hierarchical.scenario_hash(architecture, mission)
        summaries.append(
            {
                "episode": episode,
                "category": category,
                "scenario_hash": scenario_id,
                "success": result["success"],
                "architecture_changes": mission_env.architecture_change_count,
                **mission_env.cost_metrics(),
            }
        )
        for event_index, event in enumerate(mission_env.cost_events):
            events.append(
                {
                    "episode": episode,
                    "category": category,
                    "scenario_hash": scenario_id,
                    "event_index": event_index,
                    **event,
                }
            )
    write_csv(args.output_dir / "cost_trajectory_summary.csv", summaries)
    write_csv(args.output_dir / "cost_trajectory_events.csv", events)


if __name__ == "__main__":
    main()
