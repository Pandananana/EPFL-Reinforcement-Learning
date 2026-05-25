"""Diagnostic: track training episode returns + goal-reach count for MCC.

The hypothesis is that eval looks like 0 because eval is deterministic
(act = mu_head output, which is ~0), while training-time behavior may be
stochastic. We want to see if the *training* trajectories ever reach the goal.
"""
from __future__ import annotations

import sys

import gymnasium as gym
import numpy as np
import torch

from sac import SAC, ReplayBuffer


def run(
    total_steps: int = 30_000,
    start_steps: int = 10_000,
    update_after: int = 1_000,
    gamma: float = 0.9999,
    alpha: float = 0.2,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make("MountainCarContinuous-v0")
    env.reset(seed=seed)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    agent = SAC(obs_dim, act_dim, act_limit, gamma=gamma, alpha=alpha)
    buf = ReplayBuffer(1_000_000, obs_dim, act_dim)

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len = 0.0, 0
    ep_idx = 0
    ep_returns: list[float] = []
    max_pos_seen = -1.2
    goal_reaches = 0

    for step in range(1, total_steps + 1):
        if step < start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(obs)

        next_obs, r, term, trunc, _ = env.step(a)
        buf.add(obs, a, float(r), next_obs, float(term))
        obs = next_obs
        ep_ret += float(r)
        ep_len += 1
        max_pos_seen = max(max_pos_seen, float(next_obs[0]))

        if term or trunc:
            ep_idx += 1
            if term:  # only true `term` means goal was reached in MCC
                goal_reaches += 1
            ep_returns.append(ep_ret)
            if ep_idx % 5 == 0 or term:
                phase = "RANDOM" if step < start_steps else "POLICY"
                tag = " <-- GOAL!" if term else ""
                # show pre-tanh mu/log_std if in policy phase
                pol_info = ""
                if step >= start_steps:
                    with torch.no_grad():
                        ot = torch.as_tensor(np.zeros(obs_dim), dtype=torch.float32).unsqueeze(0)
                        h = agent.actor.trunc if False else agent.actor.trunk(ot)
                        mu = agent.actor.mu_head(h).item()
                        log_std = agent.actor.log_std_head(h).item()
                    pol_info = f" mu={mu:+.2f} log_std={log_std:+.2f}"
                print(
                    f"[step={step:>6} ep={ep_idx:>3} {phase}] "
                    f"ret={ep_ret:>8.2f} len={ep_len:>3} "
                    f"max_pos={max_pos_seen:+.3f} goals={goal_reaches}{pol_info}{tag}"
                )
            ep_ret, ep_len = 0.0, 0
            obs, _ = env.reset()
            max_pos_seen = float(obs[0])

        if step >= update_after:
            agent.update(buf.sample(256))

    print(f"\n=== summary: episodes={ep_idx} goal_reaches={goal_reaches} ===")
    if ep_returns:
        print(f"last 10 train-ep returns: {[round(x, 2) for x in ep_returns[-10:]]}")


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    run(total_steps=total, alpha=alpha)
