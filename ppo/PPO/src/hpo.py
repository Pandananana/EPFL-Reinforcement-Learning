from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .config import PROJECT_ENVS, get_config
from .ppo import train_ppo

PAPER_BASE: dict[str, Any] = dict(
    n_steps=2048,
    n_epochs=10,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    vf_coef=0.5,
    max_grad_norm=0.5,
    hidden_sizes=(64, 64),
    log_std_init=0.0,
    normalize_obs=False,
    reward_shaping=False,
)

SEARCH_SPACE: dict[str, list[Any]] = {
    "clip_eps": [0.1, 0.2, 0.3],
    "lr": [1e-4, 3e-4, 1e-3],
    "ent_coef": [0.0, 0.01, 0.05],
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def iter_grid_hparams(space: dict[str, list[Any]] | None = None):
    space = space or SEARCH_SPACE
    keys = list(space.keys())
    for values in itertools.product(*(space[k] for k in keys)):
        yield dict(zip(keys, values))


def sample_random_hparams(rng: random.Random, space: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    space = space or SEARCH_SPACE
    return {k: rng.choice(space[k]) for k in space}


def _run_trial(trial_id, hparams, env_ids, seeds, base, results_dir, dev, verbose) -> dict[str, Any]:
    overrides = {**base, **hparams}
    finals: list[float] = []
    per_env: dict[str, float] = {}
    log_root = results_dir / f"trial_{trial_id:04d}"
    for env_id in env_ids:
        env_finals = []
        for seed in seeds:
            cfg = get_config(env_id, **overrides)
            out = train_ppo(cfg, seed=seed, device=dev, log_dir=log_root / env_id, verbose=verbose)
            finals.append(out["final_mean"])
            env_finals.append(out["final_mean"])
        per_env[env_id] = float(np.mean(env_finals))
    return {
        "trial_id": trial_id,
        "hparams": hparams,
        "overrides": overrides,
        "objective": float(np.mean(finals)),
        "objective_std": float(np.std(finals)),
        "per_env_means": per_env,
        "per_seed_finals": finals,
    }


def run_vanilla_hpo(
    method: Literal["random", "grid"] = "random",
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    n_trials: int = 20,
    total_timesteps: int = 500_000,
    eval_interval: int = 10_000,
    results_dir: str | Path = "results/hpo_vanilla",
    search_space: dict[str, list[Any]] | None = None,
    hpo_seed: int = 42,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    env_ids = env_ids or list(PROJECT_ENVS)
    seeds = seeds or [0, 1, 2]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    space = search_space or SEARCH_SPACE
    base = {**PAPER_BASE, "total_timesteps": total_timesteps, "eval_interval": eval_interval}

    if method == "grid":
        hparam_list = list(iter_grid_hparams(space))
    elif method == "random":
        rng = random.Random(hpo_seed)
        hparam_list = [sample_random_hparams(rng, space) for _ in range(n_trials)]
    else:
        raise ValueError(f"method must be 'grid' or 'random', got {method!r}")

    trials: list[dict[str, Any]] = []
    for trial_id, hparams in enumerate(hparam_list):
        if verbose:
            print(f"\n=== Vanilla {method} trial {trial_id}/{len(hparam_list) - 1} | {hparams} ===")
        trials.append(_run_trial(trial_id, hparams, env_ids, seeds, base, results_dir, dev, verbose))

    best = max(trials, key=lambda t: t["objective"])
    summary = {
        "hpo_method": method,
        "sampler": "grid" if method == "grid" else f"random(seed={hpo_seed})",
        "n_trials": len(trials),
        "best_trial": best["trial_id"],
        "best_value": best["objective"],
        "best_params": best["hparams"],
        "best_overrides": best["overrides"],
        "env_ids": env_ids,
        "seeds": seeds,
        "paper_base": base,
        "search_space": space,
        "trials": trials,
    }
    with open(results_dir / "vanilla_trials.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    with open(results_dir / "best_config.json", "w") as f:
        json.dump(_json_safe({k: v for k, v in summary.items() if k != "trials"}), f, indent=2)
    return summary


def run_optuna_hpo(
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    n_trials: int = 20,
    total_timesteps: int = 500_000,
    eval_interval: int = 10_000,
    results_dir: str | Path = "results/hpo_optuna",
    study_name: str = "ppo_optuna_tpe",
    storage: str | None = None,
    sampler_seed: int = 42,
    device: torch.device | str | None = None,
    verbose: bool = True,
    show_progress_bar: bool = True,
) -> tuple[Any, dict[str, Any]]:
    import optuna
    from optuna.samplers import TPESampler

    env_ids = env_ids or list(PROJECT_ENVS)
    seeds = seeds or [0, 1, 2]
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    base = {**PAPER_BASE, "total_timesteps": total_timesteps, "eval_interval": eval_interval}

    def objective(trial: "optuna.Trial") -> float:
        hparams = {
            "clip_eps": trial.suggest_categorical("clip_eps", SEARCH_SPACE["clip_eps"]),
            "lr": trial.suggest_categorical("lr", SEARCH_SPACE["lr"]),
            "ent_coef": trial.suggest_categorical("ent_coef", SEARCH_SPACE["ent_coef"]),
        }
        rec = _run_trial(trial.number, hparams, env_ids, seeds, base, results_dir, dev, verbose)
        trial.set_user_attr("per_env_means", rec["per_env_means"])
        return rec["objective"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
        direction="maximize",
        sampler=TPESampler(seed=sampler_seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    best = study.best_trial
    summary = {
        "hpo_method": "optuna_tpe",
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_overrides": {**base, **best.params},
        "env_ids": env_ids,
        "seeds": seeds,
        "paper_base": base,
        "sampler": "TPESampler",
    }
    with open(results_dir / "best_config.json", "w") as f:
        json.dump(_json_safe(summary), f, indent=2)
    return study, summary


def export_study_plots(study: Any, results_dir: str | Path) -> list[Path]:
    results_dir = Path(results_dir)
    saved: list[Path] = []
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
    except ImportError:
        return saved
    for name, plot_fn in (
        ("optuna_history.png", plot_optimization_history),
        ("optuna_param_importances.png", plot_param_importances),
    ):
        try:
            fig = plot_fn(study)
            path = results_dir / name
            fig.figure.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(path)
        except Exception:
            pass
    return saved
