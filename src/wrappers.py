import math
import gymnasium as gym
import numpy as np


class CropDashboard(gym.ObservationWrapper):
    """Crops the bottom 12 pixels containing instrumentation (speedometer, steering gauge).
    Input shape:  (96, 96, 3) or (96, 96, 1)
    Output shape: (84, 96, 3) or (84, 96, 1)
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(84, obs_shape[1], obs_shape[2]),
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation[:84, :, :]


class GrayScaleObservation(gym.ObservationWrapper):
    """Converts 3-channel RGB observations into a single-channel luminance array (ITU-R 601-2).
    Input shape:  (H, W, 3)
    Output shape: (H, W, 1)
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, observation):
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        return np.expand_dims(gray.astype(np.uint8), axis=-1)


class FrameSkip(gym.Wrapper):
    """Repeats each action across `skip` consecutive frames and accumulates rewards.
    Doubles planning horizon and reduces temporal credit-assignment friction.
    """

    def __init__(self, env, skip=2):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = term or terminated
            truncated = trunc or truncated
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class V3RewardShaping(gym.Wrapper):
    """Reward shaping for V3:
    1. Discourages excessive throttle when steering angle is high.
    2. Penalizes high centrifugal velocity on sharp turns (forces braking).
    3. Truncates / penalizes lingering on grass or driving in reverse.
    """

    def __init__(self, env, max_stagnant_steps=40):
        super().__init__(env)
        self.max_stagnant_steps = max_stagnant_steps
        self.stagnant_steps = 0

    def reset(self, **kwargs):
        self.stagnant_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        steer = float(action[0])
        gas = float(action[1])
        brake = float(action[2])

        # 1. Cornering throttle scrub
        if abs(steer) > 0.35 and gas > 0.4:
            reward -= 0.15 * gas * abs(steer)

        # 2. Centrifugal velocity penalty (checks actual Box2D linear speed)
        car = getattr(self.env.unwrapped, "car", None)
        if car is not None and hasattr(car, "hull"):
            vx = car.hull.linearVelocity.x
            vy = car.hull.linearVelocity.y
            speed = math.sqrt(vx * vx + vy * vy)

            # High speed while turning sharply triggers severe tire slip penalty
            if abs(steer) > 0.4 and speed > 35.0:
                reward -= 0.05 * (speed - 35.0) * abs(steer)

            # Explicit small reward bonus for actually applying brakes when carrying speed
            if brake > 0.2 and speed > 30.0 and abs(steer) > 0.3:
                reward += 0.05

        # 3. Off-track / Stagnation penalty
        if reward <= 0:
            self.stagnant_steps += 1
            if self.stagnant_steps > self.max_stagnant_steps:
                reward -= 0.3
                truncated = True  # Cut short unrecoverable grass slides
        else:
            self.stagnant_steps = 0

        return obs, reward, terminated, truncated, info