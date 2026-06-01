from __future__ import annotations

import argparse
import csv
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def load_seed_csvs(env_name: str, results_dir: str = "results", tag: str = ""):
    # tag selects a run variant, e.g. tag="vanilla" reads <env>_vanilla_seed*.csv.
    suffix = f"_{tag}" if tag else ""
    pattern = f"{env_name}{suffix}_seed*.csv"
    paths = sorted(glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No CSVs found matching {pattern} in {results_dir}/")

    per_seed_episodes: list[list[int]] = []
    per_seed_returns: list[list[float]] = []
    for p in paths:
        episodes, rets = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                episodes.append(int(row["episode"]))
                rets.append(float(row["episode_return"]))
        per_seed_episodes.append(episodes)
        per_seed_returns.append(rets)

    # Seeds can finish with different eval-row counts (truncation timing varies).
    # Truncate all series to the shortest seed so the stack is rectangular.
    n = min(len(r) for r in per_seed_returns)
    episodes_ref = per_seed_episodes[0][:n]
    returns_per_seed = [r[:n] for r in per_seed_returns]

    return np.asarray(episodes_ref), np.asarray(returns_per_seed), paths


def plot_env(env_name: str, out_dir: str = "plots", tag: str = "") -> str:
    episodes, returns, paths = load_seed_csvs(env_name, tag=tag)
    mean = returns.mean(axis=0)
    std = returns.std(axis=0)

    suffix = f"_{tag}" if tag else ""
    variant = f" ({tag})" if tag else ""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env_name}{suffix}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(episodes, mean, label=f"SAC mean (n={len(paths)} seeds)", color="#d4801f")
    plt.fill_between(episodes, mean - std, mean + std, alpha=0.25, color="#d4801f")
    for i, row in enumerate(returns):
        plt.plot(episodes, row, alpha=0.25, linewidth=0.8, label=f"seed {i}")
    plt.xlabel("Episodes")
    plt.ylabel("Episode return")
    plt.title(f"SAC on {env_name}{variant}")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"Wrote {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--out-dir", default="plots")
    p.add_argument(
        "--tag",
        default="",
        help="Run variant to plot, e.g. 'vanilla' reads <env>_vanilla_seed*.csv "
        "and writes plots/<env>_vanilla.png. Empty = tuned runs.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_env(args.env, out_dir=args.out_dir, tag=args.tag)
