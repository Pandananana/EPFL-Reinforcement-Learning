"""Compare RL algorithms per environment.

Produces one figure per environment with every algorithm that has results for
that environment overlaid on the same axes, so the learning curves can be
compared directly.

Data sources (benchmark / vanilla results only):
  - DQN : dqn/dqn_benchmark_results.csv         wide, single run (no seeds)
  - TD3 : td3/td3_benchmark_results.csv          wide, single run (no seeds)
  - PPO : ppo/ppo_vanilla_<Env>_episodes.csv     long, 3 seeds (unequal length)
  - SAC : sac/results/<Env>_vanilla_seed*.csv     3 seeds (episode_return)

Seed handling: where an algorithm has multiple seeds, each seed's smoothed
curve is interpolated onto a common episode grid (capped at the shortest
seed's horizon, so no seed is extrapolated) and averaged across seeds. Only
the across-seed mean is shown. Single-run algorithms are plotted as-is. Each
figure's x-axis is capped at the shortest algorithm's horizon.

Run:  uv run python plot/compare.py    (or .venv/bin/python plot/compare.py)
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths (resolved relative to the repo root, i.e. the parent of this file's dir)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(HERE, "figs")

DQN_CSV = os.path.join(ROOT, "dqn", "dqn_benchmark_results.csv")
TD3_CSV = os.path.join(ROOT, "td3", "td3_benchmark_results.csv")

# The five environments and a friendly title for each.
ENVS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "Pendulum-v1",
    "MountainCarContinuous-v0",
]

# Consistent colour per algorithm across all figures.
COLORS = {"DQN": "#1f77b4", "PPO": "#ff7f0e", "SAC": "#2ca02c", "TD3": "#d62728"}

# Number of points on the common grid used when averaging seeds.
GRID_N = 300


# ---------------------------------------------------------------------------
# Loaders: each returns a list of (episodes, returns) numpy-array pairs,
# one pair per seed.  Returns an empty list if the env is not covered.
# ---------------------------------------------------------------------------
def load_wide_single(path: str, env: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """DQN / TD3: a wide CSV with an 'Episode' column and one column per env."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if env not in reader.fieldnames:
            return []
        eps, rets = [], []
        for row in reader:
            val = row[env]
            if val == "" or val is None:
                continue
            eps.append(float(row["Episode"]))
            rets.append(float(val))
    if not eps:
        return []
    return [(np.asarray(eps), np.asarray(rets))]


def load_ppo(env: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """PPO: long CSV per env with a 'seed' column; lengths differ per seed."""
    fname = f"ppo_vanilla_{env.replace('-', '_')}_episodes.csv"
    path = os.path.join(ROOT, "ppo", fname)
    if not os.path.exists(path):
        return []
    per_seed: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            per_seed[row["seed"]].append((float(row["episode"]), float(row["reward"])))
    out = []
    for seed in sorted(per_seed):
        rows = sorted(per_seed[seed])
        eps = np.asarray([r[0] for r in rows])
        rets = np.asarray([r[1] for r in rows])
        out.append((eps, rets))
    return out


def load_sac(env: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """SAC: one vanilla CSV per seed, column 'episode_return'."""
    pattern = os.path.join(ROOT, "sac", "results", f"{env}_vanilla_seed*.csv")
    out = []
    for path in sorted(glob.glob(pattern)):
        eps, rets = [], []
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                eps.append(float(row["episode"]))
                rets.append(float(row["episode_return"]))
        if eps:
            out.append((np.asarray(eps), np.asarray(rets)))
    return out


# Which loader provides each algorithm.
LOADERS = {
    "DQN": lambda env: load_wide_single(DQN_CSV, env),
    "TD3": lambda env: load_wide_single(TD3_CSV, env),
    "PPO": load_ppo,
    "SAC": load_sac,
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean that keeps the original length (edge-padded)."""
    window = max(1, min(window, len(y)))
    if window == 1:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="same")[pad : pad + len(y)]


def aggregate(seeds: list[tuple[np.ndarray, np.ndarray]]):
    """Return (x, mean, std_or_None) for a set of seed curves.

    Single seed -> smoothed curve, no band.
    Multiple seeds -> smoothed curves interpolated onto a common episode grid
    (capped at the shortest seed's horizon) then averaged; std across seeds.
    """
    if not seeds:
        return None

    if len(seeds) == 1:
        eps, rets = seeds[0]
        win = max(1, len(rets) // 20)
        return eps, rolling_mean(rets, win), None

    # Smooth each seed first, then interpolate onto a shared grid.
    lo = max(float(eps.min()) for eps, _ in seeds)
    hi = min(float(eps.max()) for eps, _ in seeds)
    grid = np.linspace(lo, hi, GRID_N)

    curves = []
    for eps, rets in seeds:
        win = max(1, len(rets) // 40)
        smoothed = rolling_mean(rets, win)
        curves.append(np.interp(grid, eps, smoothed))
    stack = np.vstack(curves)
    return grid, stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_env(env: str) -> bool:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    max_x_per_algo = []

    for algo, loader in LOADERS.items():
        agg = aggregate(loader(env))
        if agg is None:
            continue
        x, mean, _ = agg
        label = f"{algo}"
        ax.plot(x, mean, color=COLORS[algo], label=label, linewidth=1.8)
        max_x_per_algo.append(float(x.max()))

    if not max_x_per_algo:
        plt.close(fig)
        return False

    # X-axis only as long as the shortest result, so all curves overlap fully.
    ax.set_xlim(left=0, right=min(max_x_per_algo))

    ax.set_title(env)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode return (rolling mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUTDIR, f"compare_vanilla_{env}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
    return True


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    for env in ENVS:
        plot_env(env)


if __name__ == "__main__":
    main()
