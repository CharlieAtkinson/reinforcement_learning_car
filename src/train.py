import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from src.wrappers import CropDashboard, GrayScaleObservation


class StagnationFilter(gym.Wrapper):
    """
    Prevents prolonged spinning on grass or reverse drift by truncating
    episodes that spend too many consecutive steps without forward progress.
    """
    def __init__(self, env, max_stagnant_steps=100):
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.stagnant_steps = 0

    def reset(self, **kwargs):
        self.stagnant_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward <= 0:
            self.stagnant_steps += 1
            if self.stagnant_steps >= self.max_stagnant_steps:
                truncated = True
        else:
            self.stagnant_steps = 0
        return obs, reward, terminated, truncated, info


def make_env(rank: int, seed: int = 42, is_eval: bool = False):
    def _init():
        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        env = CropDashboard(env)
        env = GrayScaleObservation(env)
        if not is_eval:
            env = StagnationFilter(env, max_stagnant_steps=100)
        return env
    return _init


def linear_schedule(initial_value: float, final_value: float = 5e-5):
    """Linear learning rate scheduler for smooth convergence."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func


class ConsistencyAndBacktrackCallback(BaseCallback):
    def __init__(
        self,
        eval_env_fn,
        eval_freq=12000,
        n_eval_episodes=20,
        target_success_rate=0.80,
        patience=3,
        drop_threshold=120.0,
        save_dir="./models/v4_prod",
        verbose=1,
    ):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.target_success_rate = target_success_rate
        self.patience = patience
        self.drop_threshold = drop_threshold
        self.save_dir = save_dir
        self.best_model_path = os.path.join(save_dir, "best_model.zip")

        self.best_mean_reward = -float("inf")
        self.best_900_rate = 0.0
        self.underperforming_checks = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        print(f"\n{'='*70}")
        print(f"--> [Evaluation Check at Step {self.num_timesteps:,}] Testing {self.n_eval_episodes} Tracks...")

        eval_env = self.eval_env_fn()
        scores = []

        for ep in range(self.n_eval_episodes):
            obs = eval_env.reset()
            done = False
            ep_rew = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, _ = eval_env.step(action)
                ep_rew += reward[0]
                done = dones[0]
            scores.append(ep_rew)

        eval_env.close()

        scores = np.array(scores)
        mean_rew = float(np.mean(scores))
        std_rew = float(np.std(scores))
        median_rew = float(np.median(scores))
        rate_900 = float(np.sum(scores >= 900) / self.n_eval_episodes)
        rate_800 = float(np.sum(scores >= 800) / self.n_eval_episodes)

        print(f"    • Mean Reward      : {mean_rew:.2f} +/- {std_rew:.2f}")
        print(f"    • Median Reward    : {median_rew:.2f}")
        print(f"    • 900+ Solved Rate : {rate_900 * 100:.1f}% ({int(rate_900 * self.n_eval_episodes)}/{self.n_eval_episodes})")
        print(f"    • 800+ High Rate   : {rate_800 * 100:.1f}% ({int(rate_800 * self.n_eval_episodes)}/{self.n_eval_episodes})")

        self.logger.record("eval_v4/mean_reward", mean_rew)
        self.logger.record("eval_v4/median_reward", median_rew)
        self.logger.record("eval_v4/rate_900_plus", rate_900)
        self.logger.record("eval_v4/rate_800_plus", rate_800)
        self.logger.dump(self.num_timesteps)

        improved = False
        if rate_900 > self.best_900_rate or (rate_900 == self.best_900_rate and mean_rew > self.best_mean_reward):
            print(f"    >>> NEW BEST MODEL! (900+ Rate: {rate_900*100:.1f}%, Mean: {mean_rew:.2f})")
            self.best_900_rate = rate_900
            self.best_mean_reward = mean_rew
            self.underperforming_checks = 0
            self.model.save(self.best_model_path)
            improved = True

        if rate_900 >= self.target_success_rate:
            print(f"\n{'*'*70}")
            print(f"TARGET REACHED: Consistent 900+ score on {rate_900*100:.1f}% of tracks!")
            print(f"{'*'*70}\n")
            return False

        if not improved:
            if (self.best_mean_reward - mean_rew) > self.drop_threshold and os.path.exists(self.best_model_path):
                self.underperforming_checks += 1
                print(f"    [Warning] Underperforming check {self.underperforming_checks}/{self.patience} "
                      f"(Drop: {self.best_mean_reward - mean_rew:.1f} pts below peak)")

                if self.underperforming_checks >= self.patience:
                    print(f"\n    [BACKTRACKING] Rolling back weights to {self.best_model_path}...")
                    reloaded_model = PPO.load(self.best_model_path, env=self.training_env)
                    self.model.policy.load_state_dict(reloaded_model.policy.state_dict())
                    self.underperforming_checks = 0
            else:
                self.underperforming_checks = 0

        print(f"{'='*70}\n")
        return True


def create_eval_env_stack(n_stack=4):
    def _thunk():
        env = SubprocVecEnv([make_env(rank=900, seed=1337, is_eval=True)])
        env = VecFrameStack(env, n_stack=n_stack)
        env = VecTransposeImage(env)
        return env
    return _thunk


def main():
    parser = argparse.ArgumentParser()
    # Optimized defaults for Ryzen 5 3600 (6 cores) & 16GB RAM
    parser.add_argument("--timesteps", type=int, default=1000000, help="Max training timesteps")
    parser.add_argument("--n_envs", type=int, default=6, help="Worker count (optimal for 6-core R5 3600)")
    parser.add_argument("--target_rate", type=float, default=0.80, help="Target fraction of tracks >= 900")
    args = parser.parse_args()

    save_dir = "./models/v4_prod"
    log_dir = "./logs/v4_tensorboard"
    video_dir = "./assets/v4_videos"
    for d in [save_dir, log_dir, video_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--> Initializing {args.n_envs} parallel environments on Ryzen 5 3600...")
    train_env = SubprocVecEnv([make_env(i, seed=42) for i in range(args.n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    # Eval frequency matched to buffer step cadence
    eval_callback = ConsistencyAndBacktrackCallback(
        eval_env_fn=create_eval_env_stack(n_stack=4),
        eval_freq=12000,
        n_eval_episodes=20,
        target_success_rate=args.target_rate,
        patience=3,
        drop_threshold=120.0,
        save_dir=save_dir,
    )

    print("--> Initializing PPO with decaying learning rate & hardware-tuned buffers...")
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=linear_schedule(2.5e-4, 5e-5),  # Prevents late-stage policy jitter
        n_steps=1024,                                 # 6 * 1024 = 6,144 steps/rollout (fits L3 cache)
        batch_size=128,                               # 48 minibatches per epoch
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
    )

    print(f"--> Starting production training (Target: {args.target_rate*100:.0f}% @ 900+)...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\nTraining manually interrupted by user.")

    final_model_path = os.path.join(save_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"--> Model saved to: {final_model_path}")

    train_env.close()


if __name__ == "__main__":
    main()