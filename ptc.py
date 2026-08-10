import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MOVING_AVERAGE_WINDOW = 10


def moving_average(values, window=MOVING_AVERAGE_WINDOW):
    if window <= 0:
        raise ValueError("Moving-average window must be positive.")

    averages = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        averages.append(running_total / min(index + 1, window))
    return averages


def resolve_history_path(path):
    path = Path(path)
    candidates = [path]
    if path.is_dir():
        candidates.extend(
            [
                path / "train_history.csv",
                path / "rule_sa" / "train_history.csv",
            ]
        )
    else:
        run_dir = Path("runs") / path
        candidates.extend(
            [
                run_dir / "train_history.csv",
                run_dir / "rule_sa" / "train_history.csv",
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Training history not found. Searched: {searched}")


def read_history(path):
    rows = []
    history_path = resolve_history_path(path)
    with history_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "makespan": float(row["makespan"]),
                    "action_mode": row.get("action_mode") or history_path.parent.name,
                }
            )
    return rows


def label_for(path, rows):
    if rows:
        return rows[0]["action_mode"]
    return Path(path).parent.name


def plot_curves(inputs, output):
    histories = [(path, read_history(path)) for path in inputs]
    if not any(rows for _, rows in histories):
        raise ValueError("No train history rows to plot.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for path, rows in histories:
        if not rows:
            continue
        label = label_for(path, rows)
        episodes = [row["episode"] for row in rows]
        rewards = [row["reward"] for row in rows]
        makespans = [row["makespan"] for row in rows]
        reward_line = axes[0].plot(episodes, rewards, label=label, alpha=0.25)[0]
        axes[0].plot(
            episodes,
            moving_average(rewards),
            color=reward_line.get_color(),
            linewidth=2,
            label=f"{label} (MA{MOVING_AVERAGE_WINDOW})",
        )
        makespan_line = axes[1].plot(episodes, makespans, label=label, alpha=0.25)[0]
        axes[1].plot(
            episodes,
            moving_average(makespans),
            color=makespan_line.get_color(),
            linewidth=2,
            label=f"{label} (MA{MOVING_AVERAGE_WINDOW})",
        )

    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward by episode")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Makespan")
    axes[1].set_title("Makespan by episode")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("history", nargs="+")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: next to the first training history)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output:
        output = Path(args.output)
    else:
        output = resolve_history_path(args.history[0]).parent / "training_curves.png"
    plot_curves(args.history, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
