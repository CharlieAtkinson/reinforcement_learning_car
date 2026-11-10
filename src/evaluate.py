import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.wrappers import CropDashboard, GrayScaleObservation


def evaluate(model_path, episodes=10, record=False, video_folder="./assets/v4_videos"):
    Path(video_folder).mkdir(parents=True, exist_ok=True)

    def base_env():
        return gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")

    if record:
        env = RecordVideo(
            base_env(),
            video_folder=video_folder,
            episode_trigger=lambda ep: True,
            name_prefix="v4_eval",
        )
    else:
        env = base_env()

    env = CropDashboard(env)
    env = GrayScaleObservation(env)

    print(f"Loading checkpoint from: {model_path}")
    model = PPO.load(model_path)

    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep * 100 + 42)
        done = False
        ep_rew = 0.0

        frame_buffer = np.zeros((4, 84, 96), dtype=np.uint8)
        gray = obs.squeeze(-1)
        for i in range(4):
            frame_buffer[i] = gray

        while not done:
            action, _ = model.predict(frame_buffer, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_rew += reward
            done = terminated or truncated

            frame_buffer = np.roll(frame_buffer, shift=-1, axis=0)
            frame_buffer[-1] = obs.squeeze(-1)

        rewards.append(ep_rew)
        print(f"Episode {ep+1:02d}/{episodes:02d} - Reward: {ep_rew:6.2f}")

    env.close()
    mean_rew = np.mean(rewards)
    std_rew = np.std(rewards)
    rate_900 = np.sum(np.array(rewards) >= 900) / episodes

    print("\n" + "=" * 50)
    print(f"Benchmark over {episodes} episode(s):")
    print(f"Mean Reward       : {mean_rew:.2f} +/- {std_rew:.2f}")
    print(f"900+ Solved Rate  : {rate_900 * 100:.1f}% ({int(rate_900 * episodes)}/{episodes})")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./models/v4_prod/best_model.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--video_folder", type=str, default="./assets/v4_videos")
    args = parser.parse_args()

    evaluate(args.model, args.episodes, args.record, args.video_folder)