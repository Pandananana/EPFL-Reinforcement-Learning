from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import FAST_OVERRIDES, PROJECT_ENVS, get_config
from .ppo import train_ppo


def smooth(x: np.ndarray, window: int = 10) -> np.ndarray:
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        out[i] = x[max(0, i - window + 1) : i + 1].mean()
    return out


def align_episode_curves(curves: list[np.ndarray], max_len: int | None = None) -> np.ndarray:
    if not curves:
        return np.empty((0, 0))
    T = max_len or max(len(c) for c in curves)
    out = np.full((len(curves), T), np.nan, dtype=np.float32)
    for i, c in enumerate(curves):
        c = np.asarray(c, dtype=np.float32)
        out[i, : min(len(c), T)] = c[:T]
    return out


_SEED_RAW_COLORS = ("#9e9e9e", "#8ec8e8", "#f4b183")
_MEAN_LINE_COLOR = "#1f77b4"


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def plot_ppo_training_curve(
    episode_returns_per_seed: list[np.ndarray],
    env_id: str,
    seeds: list[int] | None = None,
    smooth_window: int = 10,
    save_path: str | Path | None = None,
    show: bool = True,
):
    _style()
    seeds = seeds if seeds is not None else list(range(len(episode_returns_per_seed)))
    curves = [np.asarray(c, dtype=np.float32) for c in episode_returns_per_seed]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (curve, seed) in enumerate(zip(curves, seeds)):
        x = np.arange(len(curve))
        ax.plot(x, curve, color=_SEED_RAW_COLORS[i % len(_SEED_RAW_COLORS)],
                alpha=0.45, linewidth=0.9, label=f"Seed {seed}")
    mat = align_episode_curves(curves)
    mean_per_ep = np.nanmean(mat, axis=0)
    mean_line = smooth(mean_per_ep, smooth_window) if smooth_window > 1 else mean_per_ep
    ax.plot(np.arange(len(mean_line)), mean_line, color=_MEAN_LINE_COLOR,
            linewidth=3.2, label=f"{smooth_window}-Ep Average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"{env_id} PPO Training Curve", fontsize=16, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_learning_curves(curves, labels=None, title="PPO learning curve", window=10, save_path=None, show=True):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = labels or [f"run {i}" for i in range(len(curves))]
    for group, label in zip(curves, labels):
        if isinstance(group[0], (list, np.ndarray)) and np.asarray(group[0]).ndim == 1:
            mat = align_episode_curves(group)
        else:
            mat = np.asarray(group)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        if window > 1:
            mean = smooth(mean, window)
            std = smooth(std, window)
        x = np.arange(len(mean))
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Undiscounted return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_eval_curves(eval_steps_list, eval_returns_list, seeds, env_id, save_path=None, show=True):
    max_len = max(len(r) for r in eval_returns_list)
    steps_mat = np.full((len(seeds), max_len), np.nan)
    ret_mat = np.full((len(seeds), max_len), np.nan)
    for i, (steps, rets) in enumerate(zip(eval_steps_list, eval_returns_list)):
        steps_mat[i, : len(steps)] = steps
        ret_mat[i, : len(rets)] = rets
    x = np.nanmean(steps_mat, axis=0)
    mean = np.nanmean(ret_mat, axis=0)
    std = np.nanstd(ret_mat, axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean, label="PPO")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Eval return ({env_id})")
    ax.set_title(f"PPO on {env_id} ({len(seeds)} seeds)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_ablation_curves(results, env_id, seeds, save_path=None, show=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, data in results.items():
        steps_l, rets_l = data["eval_steps"], data["eval_returns"]
        max_len = max(len(r) for r in rets_l)
        step_mat = np.full((len(seeds), max_len), np.nan)
        ret_mat = np.full((len(seeds), max_len), np.nan)
        for i, (steps, rets) in enumerate(zip(steps_l, rets_l)):
            step_mat[i, : len(steps)] = steps
            ret_mat[i, : len(rets)] = rets
        x = np.nanmean(step_mat, axis=0)
        y = np.nanmean(ret_mat, axis=0)
        std = np.nanstd(ret_mat, axis=0)
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - std, y + std, alpha=0.25)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Eval return ({env_id})")
    ax.set_title(f"PPO ablation — {env_id}")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def summarize_seeds(final_means: list[float]) -> str:
    arr = np.asarray(final_means)
    return f"mean={arr.mean():.2f} ± {arr.std():.2f} (n={len(arr)})"


def train_multi_seed(
    env_id: str,
    seeds: list[int] | None = None,
    cfg_overrides: dict | None = None,
    results_dir: str | Path = "results",
    device: str | None = None,
    verbose: bool = True,
) -> dict:
    seeds = seeds or [0, 1, 2]
    cfg = get_config(env_id, **(cfg_overrides or {}))
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    all_ep_returns, all_eval_steps, all_eval_returns, final_means, final_succs = [], [], [], [], []
    for seed in seeds:
        if verbose:
            print(f"\n=== Training {env_id} | seed={seed} | device={dev} ===")
        out = train_ppo(cfg, seed=seed, device=dev, log_dir=Path(results_dir), verbose=verbose)
        all_ep_returns.append(out["episode_returns"])
        all_eval_steps.append(out["eval_steps"])
        all_eval_returns.append(out["eval_returns"])
        final_means.append(out["final_mean"])
        final_succs.append(out.get("final_success", 0.0))
    out_dir = Path(results_dir) / env_id
    summary = {
        "env_id": env_id,
        "seeds": seeds,
        "final_eval_mean": float(np.mean(final_means)),
        "final_eval_std_across_seeds": float(np.std(final_means)),
        "per_seed_final": final_means,
        "per_seed_success": final_succs,
        "mean_success_rate": float(np.mean(final_succs)),
        "config": cfg.__dict__,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot_learning_curves([all_ep_returns], labels=["PPO"], title=f"PPO — {env_id}",
                         save_path=out_dir / "learning_curve.png", show=False)
    plot_eval_curves(all_eval_steps, all_eval_returns, seeds, env_id,
                     save_path=out_dir / "eval_curve.png", show=False)
    if verbose:
        print(f"\n{env_id}: {summarize_seeds(final_means)} | success={np.mean(final_succs):.0%}")
        print(f"Plots saved to {out_dir}")
    return {
        "summary": summary,
        "episode_returns": all_ep_returns,
        "eval_steps": all_eval_steps,
        "eval_returns": all_eval_returns,
    }


def train_env(
    env_id: str,
    seeds: list[int] | None = None,
    results_dir: str | Path = "results",
    device: str | None = None,
    fast: bool = False,
    cfg_overrides: dict | None = None,
    verbose: bool = True,
    show_plots: bool = False,
) -> dict:
    overrides = dict(cfg_overrides or {})
    if fast:
        overrides = {**FAST_OVERRIDES, **overrides}
    out = train_multi_seed(
        env_id, seeds=seeds, cfg_overrides=overrides or None,
        results_dir=results_dir, device=device, verbose=verbose,
    )
    if show_plots:
        try:
            from IPython.display import Image, display
            base = Path(results_dir) / env_id
            display(Image(filename=str(base / "learning_curve.png")))
            display(Image(filename=str(base / "eval_curve.png")))
        except ImportError:
            pass
    return out


def run_ablation(
    env_id: str,
    ablations: dict[str, dict],
    seeds: list[int] | None = None,
    results_dir: str | Path = "results",
    device: str | None = None,
    verbose: bool = True,
    base_overrides: dict | None = None,
) -> dict:
    seeds = seeds or [0, 1, 2]
    results = {}
    for label, overrides in ablations.items():
        merged = {**(base_overrides or {}), **overrides}
        print(f"\n{'='*60}\nAblation: {label} | overrides={merged}\n{'='*60}")
        log_root = Path(results_dir) / f"ablation_{label.replace('=', '_').replace(' ', '_')}"
        steps_l, rets_l = [], []
        for seed in seeds:
            cfg = get_config(env_id, **merged)
            r = train_ppo(
                cfg, seed=seed, device=torch.device(device) if device else None,
                log_dir=log_root, verbose=verbose,
            )
            steps_l.append(r["eval_steps"])
            rets_l.append(r["eval_returns"])
        results[label] = {"eval_steps": steps_l, "eval_returns": rets_l}
    return results


def train_all_envs(seeds=None, env_ids=None, results_dir="results", **kwargs):
    env_ids = env_ids or PROJECT_ENVS
    return {e: train_multi_seed(e, seeds=seeds, results_dir=results_dir, **kwargs) for e in env_ids}


def cli_main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Train PPO")
    p.add_argument("--env", default="CartPole-v1", choices=PROJECT_ENVS + ["all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args(argv)
    overrides = {"total_timesteps": args.timesteps} if args.timesteps else {}
    if args.env == "all":
        for e in PROJECT_ENVS:
            train_env(e, seeds=args.seeds, results_dir=args.results_dir, fast=args.fast, cfg_overrides=overrides, device=args.device)
    else:
        train_env(args.env, seeds=args.seeds, results_dir=args.results_dir, fast=args.fast, cfg_overrides=overrides, device=args.device)


if __name__ == "__main__":
    cli_main()
