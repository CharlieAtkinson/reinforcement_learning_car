import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def extract_scalar_events(log_dir, target_tags=None):
    event_files = glob.glob(os.path.join(log_dir, "**", "*tfevents*"), recursive=True)
    if not event_files:
        return {}

    event_files.sort(key=os.path.getmtime)
    tag_data = {}

    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            event_file, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()

        for tag in ea.Tags().get("scalars", []):
            if target_tags and not any(t == tag or t in tag for t in target_tags):
                continue
            records = [
                {"step": e.step, "value": e.value, "wall_time": e.wall_time}
                for e in ea.Scalars(tag)
            ]
            if tag not in tag_data:
                tag_data[tag] = []
            tag_data[tag].extend(records)

    return {
        tag: pd.DataFrame(records).drop_duplicates(subset=["step"]).sort_values("step")
        for tag, records in tag_data.items()
    }


def smooth(scalars, weight=0.85):
    if len(scalars) == 0:
        return np.array([])
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def main():
    runs = {
        "V1 Baseline": "./logs/tensorboard",
        "V2 Shaped": "./logs/v2_tensorboard"
    }

    # Explicitly include the requested eval tags
    target_metrics = [
        "eval/mean_reward",
        "eval/mean_ep_length",
        "rollout/ep_rew_mean",
        "train/entropy_loss",
        "train/value_loss"
    ]

    data = {}
    for run_name, path in runs.items():
        if os.path.exists(path):
            data[run_name] = extract_scalar_events(path, target_tags=target_metrics)

    print("\n" + "=" * 75)
    print("           EVALUATION METRIC BREAKDOWN (Terminal Output)           ")
    print("=" * 75)

    for run_name, metrics in data.items():
        print(f"\n[{run_name}]")

        # 1. Output eval/mean_reward
        if "eval/mean_reward" in metrics:
            df = metrics["eval/mean_reward"]
            print("\n  tag: eval/mean_reward")
            print(f"    • Initial Eval Reward : {df['value'].iloc[0]:.2f}")
            print(
                f"    • Peak Eval Reward    : {df['value'].max():.2f} (at step {df.loc[df['value'].idxmax(), 'step']:,})")
            print(f"    • Latest Eval Reward  : {df['value'].iloc[-1]:.2f}")
            print("    • All Recorded Points:")
            print(df[['step', 'value']].to_string(index=False))
        else:
            print("\n  tag: eval/mean_reward -> No eval events recorded in this log directory.")

        # 2. Output eval/mean_ep_length
        if "eval/mean_ep_length" in metrics:
            df = metrics["eval/mean_ep_length"]
            print("\n  tag: eval/mean_ep_length")
            print(f"    • Initial Ep Length   : {df['value'].iloc[0]:.1f}")
            print(
                f"    • Peak Ep Length      : {df['value'].max():.1f} (at step {df.loc[df['value'].idxmax(), 'step']:,})")
            print(f"    • Latest Ep Length    : {df['value'].iloc[-1]:.1f} / 1000")
            print("    • All Recorded Points:")
            print(df[['step', 'value']].to_string(index=False))
        else:
            print("\n  tag: eval/mean_ep_length -> No ep_length events recorded in this log directory.")

    # Save to CSV
    os.makedirs("logs/exported_csv", exist_ok=True)
    for run_name, metrics in data.items():
        clean_name = run_name.lower().replace(" ", "_")
        for tag in ["eval/mean_reward", "eval/mean_ep_length"]:
            if tag in metrics:
                clean_tag = tag.replace("/", "_")
                csv_path = f"logs/exported_csv/{clean_name}_{clean_tag}.csv"
                metrics[tag].to_csv(csv_path, index=False)

    # Plot evaluation graphs
    os.makedirs("assets", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"V1 Baseline": "#1f77b4", "V2 Shaped": "#2ca02c"}

    # Left: eval/mean_reward
    ax1 = axes[0]
    for run_name, metrics in data.items():
        if "eval/mean_reward" in metrics:
            df = metrics["eval/mean_reward"]
            c = colors.get(run_name, "#333333")
            ax1.plot(df["step"], df["value"], marker="o", color=c, linewidth=2, label=f"{run_name}")

    ax1.axhline(900, color="crimson", linestyle=":", linewidth=1.8, label="Solved (900)")
    ax1.set_title("Evaluation Mean Reward (eval/mean_reward)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Timesteps", fontsize=11)
    ax1.set_ylabel("Reward", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Right: eval/mean_ep_length
    ax2 = axes[1]
    for run_name, metrics in data.items():
        if "eval/mean_ep_length" in metrics:
            df = metrics["eval/mean_ep_length"]
            c = colors.get(run_name, "#333333")
            ax2.plot(df["step"], df["value"], marker="s", color=c, linewidth=2, label=f"{run_name}")

    ax2.axhline(1000, color="gray", linestyle="--", linewidth=1.5, label="Max (1000)")
    ax2.set_title("Evaluation Episode Length (eval/mean_ep_length)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Timesteps", fontsize=11)
    ax2.set_ylabel("Steps Survived", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    chart_path = "assets/eval_metrics_comparison.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\n--> Saved evaluation chart to: {chart_path}")


if __name__ == "__main__":
    main()