from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class PPOConfig:
    env_id: str
    total_timesteps: int = 200_000
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_init: float = 0.0
    eval_interval: int = 10_000
    n_eval_episodes: int = 10
    normalize_obs: bool = False
    reward_shaping: bool = False


ENV_CONFIGS: dict[str, PPOConfig] = {
    "CartPole-v1": PPOConfig(
        env_id="CartPole-v1",
        total_timesteps=100_000,
        ent_coef=0.0,
    ),
    "MountainCar-v0": PPOConfig(
        env_id="MountainCar-v0",
        total_timesteps=1_000_000,
        ent_coef=0.05,
        hidden_sizes=(128, 128),
        normalize_obs=True,
        reward_shaping=False,
    ),
    "MountainCarContinuous-v0": PPOConfig(
        env_id="MountainCarContinuous-v0",
        total_timesteps=500_000,
        ent_coef=0.02,
        hidden_sizes=(128, 128),
        normalize_obs=True,
        log_std_init=-0.5,
    ),
    "Acrobot-v1": PPOConfig(
        env_id="Acrobot-v1",
        total_timesteps=1_000_000,
        ent_coef=0.02,
        hidden_sizes=(128, 128),
        normalize_obs=True,
    ),
    "Pendulum-v1": PPOConfig(
        env_id="Pendulum-v1",
        total_timesteps=300_000,
        ent_coef=0.0,
        normalize_obs=True,
    ),
}

MOUNTAINCAR_SHAPED = replace(ENV_CONFIGS["MountainCar-v0"], reward_shaping=True)
PROJECT_ENVS = list(ENV_CONFIGS.keys())
FAST_OVERRIDES = dict(n_steps=4096, batch_size=256, n_epochs=8)


def get_config(env_id: str, **overrides) -> PPOConfig:
    if env_id not in ENV_CONFIGS:
        raise KeyError(f"Unknown env '{env_id}'. Choose from {PROJECT_ENVS}")
    cfg = ENV_CONFIGS[env_id]
    return replace(cfg, **overrides) if overrides else cfg
