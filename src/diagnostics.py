import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.wrappers import CropDashboard, GrayScaleObservation, FrameSkip

def run_diagnostics(model_path, n_episodes=15, skip=2):
    print(f"\nRunning {n_episodes}-Episode Diagnostics on: {model_path} (FrameSkip={skip})")
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    env = CropDashboard(env)
    env = GrayScaleObservation(env)
    if skip > 1:
        env = FrameSkip(env, skip=skip)

    model = PPO.load(model_path)

    rewards = []
    lengths = []
    actions_taken = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 100 + 42)
        done = False
        ep_rew = 0.0
        steps = 0

        # Maintain 4-frame grayscale buffer (4, 84, 96)
        frame_buffer = np.zeros((4, 84, 96), dtype=np.uint8)
        gray = obs.squeeze(-1)
        for i in range(4):
            frame_buffer[i] = gray

        while not done:
            action, _ = model.predict(frame_buffer, deterministic=True)
            actions_taken.append(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += reward
            steps += 1
            done = terminated or truncated

            frame_buffer = np.roll(frame_buffer, shift=-1, axis=0)
            frame_buffer[-1] = obs.squeeze(-1)

        rewards.append(ep_rew)
        lengths.append(steps)
        print(f"Track Seed {ep+1:02d}: Score = {ep_rew:6.2f} | Policy Steps = {steps:4d}")

    env.close()

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    actions_taken = np.array(actions_taken)

    median_score = np.median(rewards)
    completed_laps = np.sum(rewards >= 750)
    moderate_laps = np.sum((rewards >= 500) & (rewards < 750))
    failed_laps = np.sum(rewards < 500)

    print("\n" + "=" * 60)
    print("           V3 EPISODIC PERFORMANCE PROFILE (15 SEEDS)          ")
    print("=" * 60)
    print(f"  • Episodes Evaluated    : {n_episodes}")
    print(f"  • Mean Reward           : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  • Median Reward         : {median_score:.2f}")
    print(f"  • Peak Score            : {np.max(rewards):.2f}")
    print(f"  • Minimum Score         : {np.min(rewards):.2f}")
    print(f"  • Clean / Solved (>=750): {completed_laps}/{n_episodes} ({completed_laps/n_episodes*100:.1f}%)")
    print(f"  • Moderate Laps (500-749): {moderate_laps}/{n_episodes} ({moderate_laps/n_episodes*100:.1f}%)")
    print(f"  • Low / Spinouts (<500) : {failed_laps}/{n_episodes} ({failed_laps/n_episodes*100:.1f}%)")
    print(f"  • Mean Steps Survived   : {np.mean(lengths):.1f} / {1000 // skip}")

    print("\n[Continuous Action Distributions]")
    print(f"  • Mean Steering (|steer|): {np.mean(np.abs(actions_taken[:, 0])):.3f}")
    print(f"  • Mean Throttle (gas)    : {np.mean(actions_taken[:, 1]):.3f}")
    print(f"  • Mean Braking (brake)   : {np.mean(actions_taken[:, 2]):.3f}")
    print("=" * 60)

    Path("assets").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=10, color="#1b7837", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(rewards), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(rewards):.1f}")
    ax.axvline(median_score, color="orange", linestyle="-.", linewidth=2, label=f"Median: {median_score:.1f}")
    ax.axvline(900, color="green", linestyle=":", linewidth=2, label="Solved (900)")
    ax.set_title(f"V3 Evaluation Reward Distribution ({n_episodes} Tracks)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reward", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("assets/v3_evaluation_distribution.png", dpi=300)
    print("--> Distribution histogram saved to: assets/v3_evaluation_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/v3_best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--skip", type=int, default=2)
    args = parser.parse_args()

    run_diagnostics(args.model, args.episodes, args.skip)