"""Optuna hyperparameter sweep for SAC.

Search phase: 1 seed per trial (fast). The follow-up validation with 3 seeds for
the final report lives in final_runs.py.

Per-env search spaces live in `SUGGEST_PARAMS`. MCC needs two extra knobs that
Pendulum doesn't (gamma very close to 1, and a warmup action repeat) -- see
train.py for why iid random warmup fails on momentum-dominated envs.

n_jobs default tuned for an M4 MacBook Air (passive cooling, 10 cores). Each
trial runs PyTorch with 1 OMP thread, so n_jobs=4 keeps 4 P-cores busy without
thermal throttling. Crank it up on better hardware.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
from typing import Callable

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train import TrainConfig, train

# Worker threads grab a position from here so each in-flight trial gets its own
# tqdm row below Optuna's outer trial bar (which sits at position 0).
_position_queue: queue.Queue[int] = queue.Queue()

DEFAULT_TRIAL_BUDGET = {
    "Pendulum-v1": 15_000,
    "MountainCarContinuous-v0": 40_000,
}


def _suggest_pendulum(trial: optuna.Trial) -> dict:
    return dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        tau=trial.suggest_float("tau", 1e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        gamma=trial.suggest_float("gamma", 0.95, 0.999),
        # Fixed temperature is THE knob in classic SAC: equivalent to the
        # paper's reward scale (alpha = 1 / reward_scale). The paper's
        # sensitivity sweep covers reward scales ~1-300, so cover a wide
        # log range here.
        alpha=trial.suggest_float("alpha", 1e-2, 2.0, log=True),
    )


def _suggest_mcc(trial: optuna.Trial) -> dict:
    return dict(
        lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        tau=trial.suggest_float("tau", 1e-3, 5e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
        hidden_dim=trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        # MCC's +100 sparse goal reward arrives ~500 steps deep. At gamma=0.99
        # it discounts to ~0.66 and gets dominated by the action-energy
        # penalty. Search in (1-gamma) log space via well-chosen categoricals.
        gamma=trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995]),
        alpha=trial.suggest_float("alpha", 1e-2, 2.0, log=True),
        # Per-step iid uniform warmup actions average out and never reach the
        # goal on MCC, so the buffer never sees +100. K>=5 builds enough
        # momentum during warmup. K=1 is included as a control (those trials
        # should die fast under the pruner).
        warmup_action_repeat=trial.suggest_categorical(
            "warmup_action_repeat", [1, 5, 10, 20]
        ),
    )


SUGGEST_PARAMS: dict[str, Callable[[optuna.Trial], dict]] = {
    "Pendulum-v1": _suggest_pendulum,
    "MountainCarContinuous-v0": _suggest_mcc,
}


def make_objective(env_name: str, total_steps: int, seed: int):
    suggest = SUGGEST_PARAMS.get(env_name)
    if suggest is None:
        raise KeyError(
            f"No sweep search space registered for env {env_name!r}. "
            f"Add one to SUGGEST_PARAMS in sweep.py. Known: {list(SUGGEST_PARAMS)}"
        )

    def objective(trial: optuna.Trial) -> float:
        position = _position_queue.get()
        try:
            params = suggest(trial)
            cfg = TrainConfig(
                env_name=env_name,
                seed=seed,
                total_steps=total_steps,
                verbose=False,
                progress_position=position,
                progress_desc=f"trial {trial.number}",
                **params,
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
        finally:
            _position_queue.put(position)

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

    # Slots 1..n_jobs sit below Optuna's outer trial bar (position 0). Drain
    # any leftovers from a previous run in the same interpreter first.
    while not _position_queue.empty():
        _position_queue.get_nowait()
    for i in range(1, n_jobs + 1):
        _position_queue.put(i)

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
