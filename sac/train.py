from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from tqdm.auto import tqdm

from sac import SAC, ReplayBuffer


@dataclass
class TrainConfig:
    env_name: str = "Pendulum-v1"
    seed: int = 0
    total_episodes: int = 200
    start_steps: int = 10_000   
    update_after: int = 1_000        
    warmup_action_repeat: int = 1
    update_every: int = 1          
    eval_every: int = 1_000
    n_eval_episodes: int = 5
    buffer_size: int = 1_000_000
    batch_size: int = 256
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    alpha: float = 0.2
    device: str = "cpu"
    log_path: str | None = None
    checkpoint_path: str | None = None
    verbose: bool = False
    progress_position: int | None = None
    progress_desc: str | None = None


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
        alpha=cfg.alpha,
        device=cfg.device,
    )
    buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim, device=cfg.device)

    log_rows: list[dict] = []
    obs, _ = env.reset(seed=cfg.seed)
    ep_ret, ep_len = 0.0, 0
    episode = 0
    t0 = time.time()
    warmup_a: np.ndarray | None = None
    warmup_a_left: int = 0
    best_eval = -float("inf")
    best_step = 0

    pbar: tqdm | None = None
    if cfg.progress_position is not None:
        pbar = tqdm(
            total=cfg.total_episodes,
            desc=cfg.progress_desc or f"{cfg.env_name} seed={cfg.seed}",
            position=cfg.progress_position,
            leave=False,
            dynamic_ncols=True,
        )

    step = 0
    while episode < cfg.total_episodes:
        step += 1
        if step < cfg.start_steps:
            if warmup_a_left == 0:
                warmup_a = env.action_space.sample()
                warmup_a_left = cfg.warmup_action_repeat
            a = warmup_a
            warmup_a_left -= 1
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
            episode += 1
            log_rows.append({
                "episode": episode,
                "step": step,
                "episode_return": ep_ret,
                "elapsed_s": time.time() - t0,
            })
            ep_ret, ep_len = 0.0, 0
            warmup_a_left = 0  # fresh random action at start of next episode
            if pbar is not None:
                pbar.update(1)

        if step >= cfg.update_after and step % cfg.update_every == 0:
            for _ in range(cfg.update_every):
                agent.update(buffer.sample(cfg.batch_size))

        if step % cfg.eval_every == 0:
            eval_ret, eval_std = evaluate(eval_env, agent, cfg.n_eval_episodes)
            if cfg.verbose:
                msg = (
                    f"[{cfg.env_name} seed={cfg.seed}] "
                    f"step={step:>6} eval={eval_ret:>8.2f} ± {eval_std:>6.2f} "
                    f"alpha={agent.alpha:.3f}"
                )
                # tqdm.write keeps the bar intact when other bars share the screen.
                (tqdm.write if pbar is not None else print)(msg)
            if pbar is not None:
                pbar.set_postfix_str(f"eval={eval_ret:.1f}")
            # Save the best-eval checkpoint. SAC can suffer from policy collapse
            # late in training (especially on MCC with gamma close to 1, where Q
            # targets are large and unstable); keeping the best snapshot makes
            # eval reproducible regardless of how the curve degrades later.
            if cfg.checkpoint_path is not None and eval_ret > best_eval:
                best_eval = eval_ret
                best_step = step
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
                        "best_eval": best_eval,
                        "best_step": best_step,
                    },
                )
            if progress_callback is not None:
                if progress_callback(step, eval_ret):
                    break

    if pbar is not None:
        pbar.close()

    env.close()
    eval_env.close()

    if cfg.log_path is not None:
        os.makedirs(os.path.dirname(cfg.log_path) or ".", exist_ok=True)
        with open(cfg.log_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode", "step", "episode_return", "elapsed_s"]
            )
            writer.writeheader()
            writer.writerows(log_rows)

    if cfg.checkpoint_path is not None and cfg.verbose:
        print(
            f"[{cfg.env_name} seed={cfg.seed}] "
            f"best eval={best_eval:.2f} at step={best_step} -> {cfg.checkpoint_path}"
        )

    return log_rows


if __name__ == "__main__":
    # Quick single-run sanity check: `python train.py`
    train(TrainConfig(env_name="Pendulum-v1", total_episodes=75, verbose=True))
