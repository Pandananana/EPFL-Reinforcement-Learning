from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self.epsilon = epsilon
        self.count = epsilon
        self.mean = np.zeros(self.observation_space.shape, dtype=np.float64)
        self.var = np.ones(self.observation_space.shape, dtype=np.float64)

    def _update(self, obs: np.ndarray):
        obs = np.asarray(obs, dtype=np.float64)
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        self.var += delta * (obs - self.mean)

    def _norm(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(np.maximum(self.var / self.count, self.epsilon))
        return ((obs - self.mean) / std).astype(np.float32)

    def observation(self, obs):
        self._update(obs)
        return self._norm(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update(obs)
        return self._norm(obs), info


class MountainCarDenseReward(gym.Wrapper):
    def __init__(self, env, scale: float = 10.0):
        super().__init__(env)
        self.scale = scale
        self._prev_pos: float | None = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_pos = float(obs[0]) if len(obs) else None
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
            terminated, truncated = done, False
        if self._prev_pos is not None:
            reward = float(reward) + self.scale * abs(float(obs[0]) - self._prev_pos)
        self._prev_pos = float(obs[0])
        if len(out) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info


def episode_success(env_id: str, obs: np.ndarray, terminated: bool, info: dict) -> bool:
    if env_id.startswith("MountainCar"):
        return float(obs[0]) >= 0.5
    if env_id.startswith("Acrobot"):
        return bool(info.get("_terminated")) and not bool(info.get("_truncated"))
    if env_id == "CartPole-v1":
        return bool(terminated) is False
    if env_id == "Pendulum-v1":
        return False
    return False


def make_env(
    env_id: str,
    seed: int | None = None,
    normalize_obs: bool = False,
    reward_shaping: bool = False,
    render_mode: str | None = None,
):
    kwargs = {"render_mode": render_mode} if render_mode else {}
    env = gym.make(env_id, **kwargs)
    if reward_shaping and env_id == "MountainCar-v0":
        env = MountainCarDenseReward(env, scale=10.0)
    if normalize_obs:
        env = NormalizeObservation(env)
    if seed is not None:
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    obs_dim = int(np.prod(env.observation_space.shape))
    continuous = isinstance(env.action_space, spaces.Box)
    act_dim = int(np.prod(env.action_space.shape)) if continuous else env.action_space.n
    return env, obs_dim, act_dim, continuous


def reset_env(env, seed: int | None = None):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def step_env(env, action):
    if isinstance(action, (int, np.integer)):
        action = int(action)
    else:
        action = np.asarray(action, dtype=np.float32)
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    info = dict(info) if info else {}
    info["_terminated"] = terminated
    info["_truncated"] = truncated
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return obs, float(reward), bool(done), info
