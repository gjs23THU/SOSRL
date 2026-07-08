import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_schedule(path, episode):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if int(row["episode"]) != episode:
                continue
            row["task_idx"] = int(row["task_idx"])
            row["op_idx"] = int(row["op_idx"])
            row["sys_idx"] = int(row["sys_idx"])
            row["start_time"] = float(row["start_time"])
            row["finish_time"] = float(row["finish_time"])
            row["duration"] = float(row["duration"])
            rows.append(row)
    return sorted(rows, key=lambda item: (item["sys_idx"], item["start_time"]))


def plot_gantt(rows, output, title):
    if not rows:
        raise ValueError("No schedule rows to plot.")

    systems = sorted({row["sys_name"] for row in rows})
    y_pos = {name: idx for idx, name in enumerate(systems)}
    func_types = sorted({row["func_type"] for row in rows})
    cmap = plt.get_cmap("tab20")
    colors = {func_type: cmap(idx % 20) for idx, func_type in enumerate(func_types)}

    fig_height = max(4, 0.35 * len(systems) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    for row in rows:
        y = y_pos[row["sys_name"]]
        ax.barh(
            y,
            row["duration"],
            left=row["start_time"],
            height=0.65,
            color=colors[row["func_type"]],
            edgecolor="black",
            linewidth=0.4,
        )
        ax.text(
            row["start_time"] + row["duration"] / 2,
            y,
            f"T{row['task_idx']}-O{row['op_idx']}",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
        )

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.set_xlabel("Time")
    ax.set_ylabel("System")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[func_type]) for func_type in func_types]
    ax.legend(handles, func_types, title="Function", loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("schedule_csv")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    schedule_csv = Path(args.schedule_csv)
    output = Path(args.output) if args.output else schedule_csv.with_name(f"gantt_episode_{args.episode}.png")
    rows = read_schedule(schedule_csv, args.episode)
    action_mode = rows[0]["action_mode"] if rows else ""
    plot_gantt(rows, output, f"{action_mode} schedule - episode {args.episode}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
