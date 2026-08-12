import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import torch

import intdqn
import syn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one DQN for joint scheduling and system selection."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--fixed-mission", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--min-buffer-size", type=int, default=1000)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def save_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_seed = args.eval_seed
    if eval_seed is None:
        eval_seed = args.seed + 100000

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"runs/intdqn_seed_{args.seed}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = intdqn.IntDQNConfig(
        episodes=args.episodes,
        fixed_mission=args.fixed_mission,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer_size=args.min_buffer_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        device=device,
        log_interval=args.log_interval,
    )

    print(
        f"training IntDQN: episodes={config.episodes} "
        f"device={config.device} seed={config.seed} "
        f"fixed_mission={config.fixed_mission}"
    )
    agent, history = intdqn.train_intdqn(config)
    checkpoint_path = agent.save_checkpoint(
        output_dir / "model.pt",
        training_state={
            "episodes": config.episodes,
            "final_epsilon": history[-1]["epsilon"] if history else config.epsilon_start,
        },
    )

    eval_results = intdqn.evaluate_intdqn(
        agent,
        episodes=args.eval_episodes,
        eval_seed=eval_seed,
        collect_schedule=True,
    )
    eval_rows = []
    schedule = []
    for result in eval_results:
        eval_rows.append(
            {key: value for key, value in result.items() if key != "schedule"}
        )
        schedule.extend(result.get("schedule", []))

    save_csv(output_dir / "train_history.csv", history)
    save_csv(output_dir / "eval_results.csv", eval_rows)
    save_csv(output_dir / "eval_schedule.csv", schedule)
    save_json(
        output_dir / "config.json",
        {
            "training": asdict(config),
            "evaluation": {
                "episodes": args.eval_episodes,
                "seed": eval_seed,
            },
            "environment": syn.CONFIG,
        },
    )

    success_count = sum(result["success"] for result in eval_rows)
    print(f"model saved: {checkpoint_path}")
    print(f"evaluation success: {success_count}/{len(eval_rows)}")
    print(f"outputs saved: {output_dir}")


if __name__ == "__main__":
    main()
