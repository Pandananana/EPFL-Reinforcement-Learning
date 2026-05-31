"""Load a trained SAC checkpoint and render it on the env in a window."""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from sac import SAC


def play(
    env_name: str,
    checkpoint: str,
    episodes: int = 5,
    deterministic: bool = True,
    fps: int | None = None,
) -> None:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Did you run `python main.py final --env <env>` first?"
        )

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    extra = ck.get("extra", {})
    hidden_dim = extra.get("hidden_dim", 256)
    saved_env = extra.get("env_name", env_name)
    if saved_env != env_name:
        print(f"[warn] checkpoint was trained on {saved_env}, you're rendering on {env_name}")

    env = gym.make(env_name, render_mode="human")
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

    print(f"Playing {env_name} from {checkpoint} | {episodes} episodes | "
          f"deterministic={deterministic}")
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ret = 0.0
        while not done:
            a = agent.act(obs, deterministic=deterministic)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ret += float(r)
            if fps:
                time.sleep(1.0 / fps)
        print(f"  episode {ep}: return = {ret:.2f}")
        returns.append(ret)
    env.close()

    print(f"\nMean return over {episodes} episodes: "
          f"{np.mean(returns):.2f} ± {np.std(returns):.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt file. Defaults to models/<env>_seed0.pt.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed checkpoint to load (used only if --checkpoint is unset).")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy (the entropy-bonus behavior). "
                        "Default is deterministic (mean action), which is what the paper "
                        "uses for evaluation.")
    p.add_argument("--fps", type=int, default=None,
                   help="Cap rendering speed to N fps (default: as fast as the env allows).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint = args.checkpoint or f"models/{args.env}_seed{args.seed}.pt"
    play(
        env_name=args.env,
        checkpoint=checkpoint,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        fps=args.fps,
    )
