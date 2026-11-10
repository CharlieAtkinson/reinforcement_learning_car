import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from src.wrappers import CropDashboard, GrayScaleObservation, FrameSkip, V3RewardShaping


def make_env(rank: int, seed: int = 42, use_shaping: bool = True):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
        env.reset(seed=seed + rank)
        env = CropDashboard(env)
        env = GrayScaleObservation(env)
        env = FrameSkip(env, skip=2)
        if use_shaping:
            env = V3RewardShaping(env)
        return env

    return _init


def main():
    checkpoint_to_load = "./models/v2_best/best_model.zip"
    if not os.path.exists(checkpoint_to_load):
        print(f"Error: Could not find checkpoint at {checkpoint_to_load}")
        return

    n_envs = 4
    total_timesteps = 150000

    log_dir = "./logs/v3_tensorboard"
    model_save_dir = "./models/v3_checkpoints"
    eval_model_dir = "./models/v3_best"

    for d in [log_dir, model_save_dir, eval_model_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"--> Initializing {n_envs} parallel environments (FrameSkip=2, V3 Shaping)...")
    train_env = SubprocVecEnv([make_env(i, use_shaping=True) for i in range(n_envs)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    # Evaluation uses unshaped environment to benchmark pure task reward
    eval_env = SubprocVecEnv([make_env(i + 100, use_shaping=False) for i in range(2)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_model_dir,
        log_path=log_dir,
        eval_freq=2500,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_save_dir,
        name_prefix="ppo_v3_step",
    )

    print(f"--> Warm-starting from checkpoint: {checkpoint_to_load}")
    model = PPO.load(
        checkpoint_to_load,
        env=train_env,
        tensorboard_log=log_dir,
        learning_rate=7e-5,      # Conservative learning rate to preserve core driving skills
        ent_coef=0.015,          # Slight bump to encourage brake / deceleration exploration
    )

    print(f"--> Starting V3 fine-tuning for {total_timesteps:,} steps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=True,  # Start V3 step count cleanly from 0 in logs
        progress_bar=True,
    )

    final_path = "./models/v3_final_model.zip"
    model.save(final_path)
    print(f"--> V3 Training Complete! Saved to: {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()