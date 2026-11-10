import os
import glob
import gymnasium as gym
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from stable_baselines3.common.callbacks import BaseCallback

from src.wrappers import CropDashboard, GrayScaleObservation


class MilestoneVideoRecorderCallback(BaseCallback):
    """
    Periodically runs the current policy on a raw RGB CarRacing environment,
    records the run, and exports a compressed GIF to `assets/milestones/`.
    """

    def __init__(self, eval_freq=120000, save_path="./assets/milestones", verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Trigger at initial step (untrained agent), then every eval_freq steps
        if self.n_calls == 1 or self.n_calls % self.eval_freq == 0:
            step_count = self.num_timesteps
            print(f"\n[Milestone Recorder] Capturing benchmark run at {step_count} steps...")

            # Use raw RGB human-renderable CarRacing for the viewer GIF
            env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

            frames = []
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            # Stack buffer for prediction
            stacked_obs = np.zeros((4, 84, 96), dtype=np.float32)

            def preprocess(rgb_frame):
                cropped = rgb_frame[:84, :, :]
                gray = np.dot(cropped[..., :3], [0.299, 0.587, 0.114])
                return gray.astype(np.float32)

            # Initialize first stack
            initial_gray = preprocess(obs)
            for i in range(4):
                stacked_obs[i] = initial_gray

            step_limit = 1000
            while not done and len(frames) < step_limit:
                frame = env.render()
                frames.append(frame)

                # Query policy with normalized channel-first input
                action, _ = self.model.predict(stacked_obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

                # Shift and append new frame
                gray = preprocess(obs)
                stacked_obs = np.roll(stacked_obs, shift=-1, axis=0)
                stacked_obs[-1] = gray

            env.close()

            # Save GIF using moviepy
            gif_filename = os.path.join(self.save_path, f"stage_{step_count:07d}_rew_{int(total_reward)}.gif")
            clip = ImageSequenceClip(frames, fps=30)
            # Resize slightly to keep README loading snappy (<5MB)
            clip = clip.resized(0.7)
            clip.write_gif(gif_filename, fps=20, logger=None)
            print(f"[Milestone Recorder] Saved GIF: {gif_filename} (Reward: {total_reward:.1f})")

        return True