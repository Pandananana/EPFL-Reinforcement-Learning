from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

JOINT_LOG = {"lr": True}


def _style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def _align(curves: list[list[float]]) -> np.ndarray:
    T = max((len(c) for c in curves), default=0)
    out = np.full((len(curves), T), np.nan, dtype=np.float64)
    for i, c in enumerate(curves):
        out[i, : len(c)] = c
    return out


def _mean_curve(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = _align(run["eval_steps"])
    rets = _align(run["eval_returns"])
    x = np.nanmean(steps, axis=0)
    mean = np.nanmean(rets, axis=0)
    std = np.nanstd(rets, axis=0)
    return x, mean, std


def plot_ofat_curves(sweep: dict[str, Any], save_path=None, show=True):
    _style()
    fig, ax = plt.subplots(figsize=(9, 5))
    default_v = str(sweep.get("default_value"))
    for run in sweep["runs"]:
        x, mean, std = _mean_curve(run)
        is_default = run["label"].split("=", 1)[-1] == default_v
        ax.plot(x, mean, linewidth=3 if is_default else 1.8,
                label=run["label"] + (" (paper default)" if is_default else ""),
                zorder=3 if is_default else 2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Eval return")
    ax.set_title(f"PPO on {sweep['env_id']} — effect of {sweep['hparam']} "
                 f"({len(sweep['seeds'])} seeds)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def plot_ofat_final(sweep: dict[str, Any], save_path=None, show=True):
    _style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    runs = sweep["runs"]
    labels = [str(v) for v in sweep["values"]]
    scores = [r["score"] for r in runs]
    errs = [r["score_std"] for r in runs]
    x = np.arange(len(labels))
    bars = ax.bar(x, scores, yerr=errs, capsize=4, alpha=0.85)
    default_v = str(sweep.get("default_value"))
    for i, v in enumerate(labels):
        if v == default_v:
            bars[i].set_edgecolor("black")
            bars[i].set_linewidth(2.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_xlabel(sweep["hparam"])
    ax.set_ylabel("Final eval return")
    ax.set_title(f"{sweep['env_id']} — final score vs. {sweep['hparam']} "
                 f"(outlined = paper default)")
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def plot_sensitivity_bars(report: dict[str, Any], save_path=None, show=True):
    _style()
    ranking = report["ranking"]
    names = [r["hparam"] for r in ranking][::-1]
    ranges = [r["sensitivity_range"] for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
    ax.barh(names, ranges, alpha=0.85)
    ax.set_xlabel("Score range over grid  (max - min)")
    ax.set_title(f"PPO hyperparameter sensitivity — {report['env_id']}")
    for i, (n, rng) in enumerate(zip(names, ranges)):
        ax.text(rng, i, f" {rng:.1f}", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def plot_search_progress(summaries: dict[str, dict[str, Any]], save_path=None, show=True):
    _style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, s in summaries.items():
        y = s.get("best_so_far") or list(np.maximum.accumulate([t["score"] for t in s["trials"]]))
        ax.plot(np.arange(1, len(y) + 1), y, marker="o", markersize=3, label=label)
    env = next(iter(summaries.values())).get("env_id", "")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best eval return so far")
    ax.set_title(f"HPO search efficiency — {env}")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def plot_trial_scatter(summary: dict[str, Any], hparam: str, save_path=None, show=True):
    _style()
    xs = [t["hparams"].get(hparam) for t in summary["trials"]]
    ys = [t["score"] for t in summary["trials"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(xs, ys, alpha=0.8)
    if JOINT_LOG.get(hparam, False):
        ax.set_xscale("log")
    ax.set_xlabel(hparam)
    ax.set_ylabel("Eval return")
    ax.set_title(f"{summary.get('env_id', '')} — {summary['method']}: score vs. {hparam}")
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def plot_optuna_history(study, save_path=None, show=True):
    return _optuna_plot(study, "plot_optimization_history", save_path, show)


def plot_optuna_importances(study, save_path=None, show=True):
    return _optuna_plot(study, "plot_param_importances", save_path, show)


def _optuna_plot(study, fn_name: str, save_path, show):
    try:
        from optuna.visualization import matplotlib as ov
    except ImportError:
        print("optuna not installed; skipping", fn_name)
        return None
    ax = getattr(ov, fn_name)(study)
    fig = ax.figure
    fig.set_facecolor("white")
    for a in fig.axes:
        a.set_facecolor("white")
    fig.tight_layout()
    _save(fig, save_path, show)
    return fig


def compare_to_baseline(baseline_score: float, tuned_summary: dict[str, Any]):
    tuned = tuned_summary.get("best_value") or tuned_summary.get("best_score")
    delta = tuned - baseline_score
    print(f"Baseline (paper default): {baseline_score:.1f}")
    print(f"Tuned ({tuned_summary['method']}):     {tuned:.1f}")
    print(f"Improvement:              {delta:+.1f}  ({100 * delta / abs(baseline_score):+.1f}%)")
    return delta


def _save(fig, save_path, show):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    if show:
        plt.show()
    else:
        plt.close(fig)
