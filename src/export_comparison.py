import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

def extract_events(log_dir, target_tags):
    event_files = glob.glob(os.path.join(log_dir, "**", "*tfevents*"), recursive=True)
    if not event_files:
        return {}
    event_files.sort(key=os.path.getmtime)
    tag_data = {}
    for ef in event_files:
        ea = event_accumulator.EventAccumulator(ef, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if any(t in tag for t in target_tags):
                records = [{"step": e.step, "value": e.value} for e in ea.Scalars(tag)]
                if tag not in tag_data:
                    tag_data[tag] = []
                tag_data[tag].extend(records)
    return {k: pd.DataFrame(v).drop_duplicates(subset=["step"]).sort_values("step") for k, v in tag_data.items()}

def main():
    runs = {
        "V1 Baseline (Continuous)": "./logs/tensorboard",
        "V2 Shaped (Continuous)": "./logs/v2_tensorboard",
        "V4 Production (Discrete)": "./logs/v4_tensorboard"
    }

    target_tags = ["eval/mean_reward", "eval_v4/mean_reward", "rollout/ep_rew_mean"]
    data = {k: extract_events(v, target_tags) for k, v in runs.items() if os.path.exists(v)}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    colors = {"V1 Baseline (Continuous)": "#d62728", "V2 Shaped (Continuous)": "#ff7f0e", "V4 Production (Discrete)": "#2ca02c"}

    # Left: Training Rollouts
    for name, metrics in data.items():
        if "rollout/ep_rew_mean" in metrics:
            df = metrics["rollout/ep_rew_mean"]
            ax1.plot(df["step"], df["value"], color=colors[name], label=name, alpha=0.85, linewidth=1.5)

    ax1.set_title("Training Rollout Returns (ep_rew_mean)", fontweight="bold")
    ax1.set_xlabel("Timesteps")
    ax1.set_ylabel("Mean Reward")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Right: Deterministic Evaluation Checks
    for name, metrics in data.items():
        tag = "eval_v4/mean_reward" if "eval_v4/mean_reward" in metrics else "eval/mean_reward"
        if tag in metrics:
            df = metrics[tag]
            ax2.plot(df["step"], df["value"], marker="o", markersize=3, color=colors[name], label=name, linewidth=2)

    ax2.axhline(900, color="gold", linestyle=":", linewidth=2, label="Solved (900)")
    ax2.set_title("Deterministic Evaluation Return (eval/mean_reward)", fontweight="bold")
    ax2.set_xlabel("Timesteps")
    ax2.set_ylabel("Evaluation Reward")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    chart_path = "assets/v4_diagnostics/overall_training_progression.png"
    plt.savefig(chart_path, dpi=300)
    print(f"--> Multi-model comparison saved to: {chart_path}")

if __name__ == "__main__":
    main()
