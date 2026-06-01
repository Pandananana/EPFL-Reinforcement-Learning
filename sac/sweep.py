from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
from typing import Callable

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from train import TrainConfig, train

# Each worker process owns one tqdm row (single in-flight trial per process).
# Kept as a queue so make_objective stays unchanged across single- and
# multi-process paths.
_position_queue: queue.Queue[int] = queue.Queue()

DEFAULT_TRIAL_BUDGET = {
    "Pendulum-v1": 75,
    "MountainCarContinuous-v0": 40,
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


def make_objective(env_name: str, total_episodes: int, seed: int):
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
                total_episodes=total_episodes,
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


def _run_worker(
    worker_idx: int,
    env_name: str,
    total_episodes: int,
    n_trials_total: int,
    seed: int,
    study_name: str,
    storage: str,
) -> None:
    """Single-process worker. Loads the shared study and runs trials until the
    global budget is exhausted (MaxTrialsCallback polls storage)."""
    # Belt-and-braces: keep each process to one BLAS/OMP thread so 16 workers
    # don't each spawn 16 OMP threads and trash each other.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # Per-worker sampler seed so the TPE samplers don't propose identical
    # trials in lockstep before they've each seen enough history.
    sampler = TPESampler(seed=seed + worker_idx, n_startup_trials=5)
    pruner = MedianPruner(n_warmup_steps=5, n_startup_trials=5)
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    _position_queue.put(worker_idx + 1)

    obj = make_objective(env_name, total_episodes, seed=seed)
    study.optimize(
        obj,
        n_trials=n_trials_total,
        n_jobs=1,
        callbacks=[
            MaxTrialsCallback(
                n_trials_total,
                states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
            )
        ],
        show_progress_bar=False,
    )


def run_sweep(
    env_name: str,
    n_trials: int,
    total_episodes: int | None,
    n_jobs: int,
    seed: int,
    study_name: str | None,
    storage: str,
) -> optuna.Study:
    total_episodes = total_episodes or DEFAULT_TRIAL_BUDGET.get(env_name, 75)
    study_name = study_name or f"sac_{env_name}"

    sampler = TPESampler(seed=seed, n_startup_trials=5)
    pruner = MedianPruner(n_warmup_steps=5, n_startup_trials=5)

    # Materialize the study in storage upfront so workers can load_if_exists
    # without racing on creation.
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    if n_jobs <= 1:
        # Single-process inline path (default for laptops).
        torch.set_num_threads(1)
        while not _position_queue.empty():
            _position_queue.get_nowait()
        _position_queue.put(1)
        obj = make_objective(env_name, total_episodes, seed=seed)
        study.optimize(obj, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    else:
        # Process fan-out. `spawn` (not fork) is required because PyTorch's
        # internal thread pools don't survive fork cleanly.
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(
                target=_run_worker,
                args=(i, env_name, total_episodes, n_trials, seed, study_name, storage),
            )
            for i in range(n_jobs)
        ]
        for p in procs:
            p.start()
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            for p in procs:
                p.join()
            raise
        # Re-read the study so best_value/best_params reflect worker results.
        study = optuna.load_study(study_name=study_name, storage=storage)

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
        "search_total_episodes": total_episodes,
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
        "--total-episodes",
        type=int,
        default=None,
        help="Per-trial training budget in episodes (defaults depend on env).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help=(
            "Parallel worker processes (NOT threads). Each worker runs single-threaded "
            "and shares trial state via the SQLite study. 4 fits an M4 Air; set to ~physical "
            "cores on a workstation (e.g. 16 on a 32-thread EPYC -- SMT siblings fight for FP units)."
        ),
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
        total_episodes=args.total_episodes,
        n_jobs=args.n_jobs,
        seed=args.seed,
        study_name=args.study_name,
        storage=args.storage,
    )
