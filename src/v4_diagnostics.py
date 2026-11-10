import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.wrappers import CropDashboard, GrayScaleObservation

def run_v4_diagnostics(model_path, n_episodes=20):
    print(f"\n============================================================")
    print(f"  RUNNING V4 PRODUCTION BENCHMARK ({n_episodes} UNSEEN TRACKS)")
    print(f"  Target Checkpoint: {model_path}")
    print(f"============================================================\n")

    env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
    env = CropDashboard(env)
    env = GrayScaleObservation(env)

    model = PPO.load(model_path)

    rewards = []
    lengths = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    action_names = {0: "Coast", 1: "Steer Left", 2: "Steer Right", 3: "Gas", 4: "Brake"}

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 250 + 777)
        done = False
        ep_rew = 0.0
        steps = 0

        frame_buffer = np.zeros((4, 84, 96), dtype=np.uint8)
        gray = obs.squeeze(-1)
        for i in range(4):
            frame_buffer[i] = gray

        while not done:
            action, _ = model.predict(frame_buffer, deterministic=True)
            act_int = int(action)
            action_counts[act_int] += 1

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += reward
            steps += 1
            done = terminated or truncated

            frame_buffer = np.roll(frame_buffer, shift=-1, axis=0)
            frame_buffer[-1] = obs.squeeze(-1)

        rewards.append(ep_rew)
        lengths.append(steps)
        status = "SOLVED (900+)" if ep_rew >= 900 else ("HIGH (800+)" if ep_rew >= 800 else "PARTIAL")
        print(f"Track Seed {ep+1:02d}: Score = {ep_rew:6.2f} | Steps = {steps:4d} | Status = {status}")

    env.close()

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    total_actions = sum(action_counts.values())

    rate_900 = np.sum(rewards >= 900) / n_episodes
    rate_800 = np.sum(rewards >= 800) / n_episodes
    rate_500 = np.sum(rewards >= 500) / n_episodes
    fails = np.sum(rewards < 500) / n_episodes

    print("\n" + "=" * 65)
    print("                 V4 PRODUCTION SUMMARY REPORT                 ")
    print("=" * 65)
    print(f"  * Total Tracks Tested     : {n_episodes}")
    print(f"  * Mean Reward             : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  * Median Reward           : {np.median(rewards):.2f}")
    print(f"  * Peak Lap Score          : {np.max(rewards):.2f}")
    print(f"  * Minimum Lap Score       : {np.min(rewards):.2f}")
    print(f"  * 900+ Solved Rate        : {rate_900 * 100:.1f}% ({int(rate_900 * n_episodes)}/{n_episodes})")
    print(f"  * 800+ High Performance   : {rate_800 * 100:.1f}% ({int(rate_800 * n_episodes)}/{n_episodes})")
    print(f"  * 500-799 Intermediate    : {((rate_500 - rate_800) * 100):.1f}%")
    print(f"  * Under 500 Failures      : {fails * 100:.1f}% ({int(fails * n_episodes)}/{n_episodes})")
    print(f"  * Mean Steps per Episode  : {np.mean(lengths):.1f} / 1000")

    print("\n[Discrete Action Usage Breakdown]")
    for act_id, count in action_counts.items():
        pct = (count / total_actions) * 100
        print(f"  * Action {act_id} ({action_names[act_id]:<11}): {count:6d} ({pct:5.2f}%)")
    print("=" * 65)

    Path("assets/v4_diagnostics").mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.hist(rewards, bins=10, color="#107c41", edgecolor="black", alpha=0.85)
    ax1.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rewards):.1f}")
    ax1.axvline(np.median(rewards), color="orange", linestyle="-.", linewidth=2, label=f"Median: {np.median(rewards):.1f}")
    ax1.axvline(900, color="gold", linestyle=":", linewidth=2.5, label="Solved (900)")
    ax1.set_title(f"V4 Reward Distribution ({n_episodes} Seeds)", fontweight="bold")
    ax1.set_xlabel("Reward Score")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    labels = [action_names[i] for i in range(5)]
    percentages = [(action_counts[i] / total_actions) * 100 for i in range(5)]
    colors = ["#7f7f7f", "#1f77b4", "#aec7e8", "#2ca02c", "#d62728"]
    ax2.bar(labels, percentages, color=colors, edgecolor="black", alpha=0.85)
    ax2.set_title("Discrete Action Execution Ratio", fontweight="bold")
    ax2.set_ylabel("Usage Percentage (%)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = "assets/v4_diagnostics/v4_performance_profile.png"
    plt.savefig(chart_path, dpi=300)
    print(f"\n--> Diagnostic chart saved to: {chart_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/v4_prod/best_model.zip")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    run_v4_diagnostics(args.model, args.episodes)
