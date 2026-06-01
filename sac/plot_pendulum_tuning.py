from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from plot import load_seed_csvs


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average; returns an array shortened by window-1."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_variant(ax, env_name: str, tag: str, label: str, color: str, window: int):
    episodes, returns, paths = load_seed_csvs(env_name, tag=tag)
    mean = returns.mean(axis=0)
    std = returns.std(axis=0)

    mean_s, std_s = smooth(mean, window), smooth(std, window)
    x = episodes[window - 1:] if window > 1 else episodes

    ax.plot(x, mean_s, color=color, label=f"{label} (n={len(paths)} seeds)")
    ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2, color=color)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument(
        "--out",
        default=os.path.join("plots", "pendulum_vanilla_vs_tuned.png"),
        help="Output PNG path (matches the \\includegraphics path in report.tex).",
    )
    p.add_argument("--smooth", type=int, default=5, help="Moving-average window (1 = none).")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_variant(ax, args.env, "vanilla", "Default", "#7f7f7f", args.smooth)
    plot_variant(ax, args.env, "", "Tuned", "#d4801f", args.smooth)

    ax.set_xlabel("Episodes")
    ax.set_ylabel("Episode return")
    ax.set_title(f"SAC on {args.env}: default vs. tuned hyperparameters")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
