"""Evaluate trained SAC checkpoints on held-out env seeds.

Loads every models/<env>_seed*.pt for each env, runs each checkpoint for
`--episodes` deterministic episodes on fresh env seeds, and writes the
combined mean ± std per env to results/eval.csv.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import gymnasium as gym
import numpy as np
import torch

from sac import SAC

ENVS = ["MountainCarContinuous-v0", "Pendulum-v1"]
MODELS_DIR = "models"
RESULTS_DIR = "results"
EVAL_SEED_OFFSET = 10_000  # keep eval seeds disjoint from training seeds


def eval_checkpoint(env_name: str, checkpoint: str, episodes: int) -> list[float]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    hidden_dim = ck.get("extra", {}).get("hidden_dim", 256)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = SAC(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_limit=act_limit,
        hidden_dim=hidden_dim,
        device="cpu",
    )
    agent.load(checkpoint)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + ep)
        done = False
        ret = 0.0
        while not done:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += float(r)
        returns.append(ret)
    env.close()
    return returns


def main(episodes: int) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "eval.csv")

    rows = []
    for env_name in ENVS:
        checkpoints = sorted(glob.glob(os.path.join(MODELS_DIR, f"{env_name}_seed*.pt")))
        if not checkpoints:
            print(f"[warn] no checkpoints found for {env_name}, skipping")
            continue

        all_returns: list[float] = []
        for ck in checkpoints:
            rets = eval_checkpoint(env_name, ck, episodes)
            print(f"  {os.path.basename(ck)}: {np.mean(rets):.2f} ± {np.std(rets):.2f}")
            all_returns.extend(rets)

        mean = float(np.mean(all_returns))
        std = float(np.std(all_returns))
        print(f"{env_name}: {mean:.2f} ± {std:.2f}  "
              f"(n={len(all_returns)} = {len(checkpoints)} seeds × {episodes} eps)")
        rows.append({"Env": env_name, "Mean": f"{mean:.2f}", "Std": f"{std:.2f}"})

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Env", "Mean", "Std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10,
                   help="Eval episodes per checkpoint (default: 10).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(episodes=args.episodes)
