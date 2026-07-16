import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import dqn


def parse_selected_system_num(value):
    if value is None or value.lower() == "none":
        return None
    if "," in value:
        low, high = value.split(",", 1)
        return int(low), int(high)
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--scenario-pool-size", type=int, default=20)
    parser.add_argument("--scenario-order", choices=["random", "sequential"], default="random")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--selected-system-num", type=str, default="none")
    parser.add_argument("--min-system-num", type=int, default=3)
    parser.add_argument("--max-system-num", type=int, default=22)
    parser.add_argument("--cost-limit", type=float, default=8000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--target-update-interval", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_outputs(output_dir, config, history, eval_results):
    eval_rows = []
    schedule_rows = []
    for result in eval_results:
        eval_rows.append({key: value for key, value in result.items() if key != "schedule"})
        schedule_rows.extend(result.get("schedule", []))

    save_csv(output_dir / "train_history.csv", history)
    save_csv(output_dir / "eval_results.csv", eval_rows)
    save_csv(output_dir / "eval_schedule.csv", schedule_rows)
    save_json(output_dir / "config.json", asdict(config))
    print(f"logs saved to: {output_dir}")


def main():
    args = parse_args()
    config = dqn.DQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        scenario_pool_size=args.scenario_pool_size,
        scenario_order=args.scenario_order,
        selected_system_num=parse_selected_system_num(args.selected_system_num),
        min_system_num=args.min_system_num,
        max_system_num=args.max_system_num,
        cost_limit=args.cost_limit,
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
    )

    dqn.set_seed(config.seed)
    train_pool = dqn.ScenarioPool(
        size=config.scenario_pool_size,
        selected_system_num=config.selected_system_num,
        min_system_num=config.min_system_num,
        max_system_num=config.max_system_num,
        cost_limit=config.cost_limit,
    )
    eval_pool = train_pool
    if not args.eval_on_train:
        eval_pool = dqn.ScenarioPool(
            size=args.eval_episodes,
            selected_system_num=config.selected_system_num,
            min_system_num=config.min_system_num,
            max_system_num=config.max_system_num,
            cost_limit=config.cost_limit,
        )

    agent, history = dqn.train_dqn(config, train_pool)
    best_episode = max(history, key=lambda row: (row["done_ops"], -row["makespan"]))
    print("best train episode:", best_episode)

    eval_results = dqn.evaluate_dqn(
        agent,
        eval_pool,
        episodes=args.eval_episodes,
        max_steps=config.max_steps,
        collect_schedule=True,
    )
    print("eval results:")
    for result in eval_results:
        print({key: value for key, value in result.items() if key != "schedule"})

    run_name = args.run_name or f"seed_{config.seed}"
    save_outputs(Path(args.log_dir) / run_name / "op", config, history, eval_results)


if __name__ == "__main__":
    main()
