"""Plot eval-return curves averaged across seeds (mean ± std)."""
from __future__ import annotations

import argparse
import csv
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np


def load_seed_csvs(env_name: str, results_dir: str = "results"):
    paths = sorted(glob(os.path.join(results_dir, f"{env_name}_seed*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found for {env_name} in {results_dir}/")

    steps_ref = None
    returns_per_seed = []
    for p in paths:
        steps, rets = [], []
        with open(p) as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                rets.append(float(row["eval_return"]))
        if steps_ref is None:
            steps_ref = steps
        n = min(len(steps_ref), len(rets))
        returns_per_seed.append(rets[:n])
        steps_ref = steps_ref[:n]

    return np.asarray(steps_ref), np.asarray(returns_per_seed), paths


def plot_env(env_name: str, out_dir: str = "plots") -> str:
    steps, returns, paths = load_seed_csvs(env_name)
    mean = returns.mean(axis=0)
    std = returns.std(axis=0)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{env_name}.png")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean, label=f"SAC mean (n={len(paths)} seeds)", color="#d4801f")
    plt.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#d4801f")
    for i, row in enumerate(returns):
        plt.plot(steps, row, alpha=0.25, linewidth=0.8, label=f"seed {i}")
    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(f"SAC on {env_name}")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_env(args.env, out_dir=args.out_dir)
