"""Training loop. Called by single-run, sweep, and final-run scripts."""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import numpy as np
import torch

from sac import SAC, ReplayBuffer


@dataclass
class TrainConfig:
    env_name: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 30_000
    start_steps: int = 1_000          # uniform-random actions before policy kicks in
    update_after: int = 1_000         # don't train until buffer has some data
    update_every: int = 1             # one gradient step per env step
    eval_every: int = 1_000
    n_eval_episodes: int = 5
    # Hyperparams that the sweep will mutate:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    auto_alpha: bool = True
    init_alpha: float = 0.2
    # Plumbing:
    device: str = "cpu"
    log_path: str | None = None
    checkpoint_path: str | None = None
    verbose: bool = False


def _make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def evaluate(env, agent: SAC, n_episodes: int) -> tuple[float, float]:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += float(r)
        returns.append(ret)
    return float(np.mean(returns)), float(np.std(returns))


def train(
    cfg: TrainConfig,
    progress_callback: Callable[[int, float], bool] | None = None,
) -> list[dict]:
    """Run one SAC training session. Returns the list of eval log rows.

    `progress_callback(step, eval_return) -> stop_flag` lets the caller (Optuna)
    early-stop a trial.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = _make_env(cfg.env_name, cfg.seed)
    eval_env = _make_env(cfg.env_name, cfg.seed + 10_000)

    assert isinstance(env.action_space, gym.spaces.Box), (
        f"SAC requires a continuous action space. Got {type(env.action_space).__name__} "
        f"for {cfg.env_name}. Try Pendulum-v1 or MountainCarContinuous-v0."
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SAC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=cfg.hidden_dim,
        gamma=cfg.gamma,
        tau=cfg.tau,
        lr=cfg.lr,
        auto_alpha=cfg.auto_alpha,
        init_alpha=cfg.init_alpha,
        device=cfg.device,
    )
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim, device=cfg.device)

    log_rows: list[dict] = []
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0
    t0 = time.time()

    for step in range(1, cfg.total_steps + 1):
        if step < cfg.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(obs)

        next_obs, r, term, trunc, _ = env.step(a)
        # Don't bootstrap through time-limit truncations: only `term` counts as a
        # real terminal. This is the standard SAC/SpinningUp trick.
        buffer.add(obs, a, float(r), next_obs, float(term))
        obs = next_obs
        ep_ret += float(r)
        ep_len += 1
        if term or trunc:
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step >= cfg.update_after and step % cfg.update_every == 0:
            for _ in range(cfg.update_every):
                agent.update(buffer.sample(cfg.batch_size))

        if step % cfg.eval_every == 0:
            eval_ret, eval_std = evaluate(eval_env, agent, cfg.n_eval_episodes)
            row = {
                "step": step,
                "eval_return": eval_ret,
                "eval_std": eval_std,
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            if cfg.verbose:
                print(
                    f"[{cfg.env_name} seed={cfg.seed}] "
                    f"step={step:>6} eval={eval_ret:>8.2f} ± {eval_std:>6.2f} "
                    f"alpha={agent.alpha.item():.3f}"
                )
            if progress_callback is not None:
                if progress_callback(step, eval_ret):
                    break

    env.close()
    eval_env.close()

    if cfg.log_path is not None:
        os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
        with open(cfg.log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["step", "eval_return", "eval_std", "elapsed_s"]
            )
            writer.writeheader()
            writer.writerows(log_rows)

    if cfg.checkpoint_path is not None:
        os.makedirs(os.path.dirname(cfg.checkpoint_path) or ".", exist_ok=True)
        agent.save(
            cfg.checkpoint_path,
            extra={
                "env_name": cfg.env_name,
                "seed": cfg.seed,
                "hidden_dim": cfg.hidden_dim,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "act_limit": act_limit,
            },
        )

    return log_rows


if __name__ == "__main__":
    # Quick single-run sanity check: `python train.py`
    train(TrainConfig(env_name="Pendulum-v1", total_steps=15_000, verbose=True))
