import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def resolve_history_path(path):
    path = Path(path)
    if path.is_dir():
        return path / "train_history.csv"
    return path


def read_history(path):
    rows = []
    with resolve_history_path(path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "makespan": float(row["makespan"]),
                    "action_mode": row["action_mode"],
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
        axes[0].plot(episodes, rewards, label=label)
        axes[1].plot(episodes, makespans, label=label)

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
    parser.add_argument("--output", type=str, default="training_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output)
    plot_curves(args.history, output)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
