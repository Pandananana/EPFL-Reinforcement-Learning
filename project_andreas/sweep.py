"""Optuna hyperparameter sweep for SAC.

Search phase: 1 seed per trial (fast). The follow-up validation with 3 seeds for
the final report lives in final_runs.py.

n_jobs default tuned for an M4 MacBook Air (passive cooling, 10 cores). Each
trial runs PyTorch with 1 OMP thread, so n_jobs=4 keeps 4 P-cores busy without
thermal throttling. Crank it up on better hardware.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train import TrainConfig, train

DEFAULT_TRIAL_BUDGET = {
    "Pendulum-v1": 15_000,
    "MountainCarContinuous-v0": 40_000,
}


def make_objective(env_name: str, total_steps: int, seed: int):
    def objective(trial: optuna.Trial) -> float:
        cfg = TrainConfig(
            env_name=env_name,
            seed=seed,
            total_steps=total_steps,
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            tau=trial.suggest_float("tau", 1e-3, 5e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
            hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            gamma=trial.suggest_float("gamma", 0.95, 0.999),
            verbose=False,
        )

        eval_returns: list[float] = []

        def cb(step: int, eval_ret: float) -> bool:
            eval_returns.append(eval_ret)
            trial.report(eval_ret, step)
            return trial.should_prune()

        try:
            train(cfg, progress_callback=cb)
        except optuna.TrialPruned:
            raise

        if not eval_returns:
            return -1e9
        # Score = mean of the last few evals (smooths out single-eval noise).
        return float(np.mean(eval_returns[-5:]))

    return objective


def run_sweep(
    env_name: str,
    n_trials: int,
    total_steps: int | None,
    n_jobs: int,
    seed: int,
    study_name: str | None,
    storage: str,
) -> optuna.Study:
    total_steps = total_steps or DEFAULT_TRIAL_BUDGET.get(env_name, 30_000)
    study_name = study_name or f"sac_{env_name}"

    # Important on macOS with multi-thread Optuna: pin PyTorch to 1 thread so
    # the OMP pools across worker threads don't fight each other.
    torch.set_num_threads(1)

    sampler = TPESampler(seed=seed, n_startup_trials=5)
    pruner = MedianPruner(n_warmup_steps=5, n_startup_trials=5)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    obj = make_objective(env_name, total_steps, seed=seed)
    study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    print()
    print(f"Best value: {study.best_value:.2f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs("results", exist_ok=True)
    out = {
        "env_name": env_name,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "search_total_steps": total_steps,
    }
    with open(f"results/best_{env_name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved best params to results/best_{env_name}.json")
    return study


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Per-trial training budget (defaults depend on env).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel trials. 4 is the sweet spot for an M4 Air; bump higher on workstations.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default="sqlite:///sweeps.db")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        env_name=args.env,
        n_trials=args.n_trials,
        total_steps=args.total_steps,
        n_jobs=args.n_jobs,
        seed=args.seed,
        study_name=args.study_name,
        storage=args.storage,
    )
