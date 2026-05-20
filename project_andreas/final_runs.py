"""Validate the sweep's best params with multiple seeds for the final plots.

This is the half of the pipeline that satisfies the "average across at least 3
seeds" requirement from the project brief.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from train import TrainConfig, train

DEFAULT_FINAL_STEPS = {
    "Pendulum-v1": 30_000,
    "MountainCarContinuous-v0": 80_000,
}


def _run_one(env_name: str, seed: int, total_steps: int, params: dict) -> str:
    torch.set_num_threads(1)
    log_path = f"results/{env_name}_seed{seed}.csv"
    ckpt_path = f"models/{env_name}_seed{seed}.pt"
    cfg = TrainConfig(
        env_name=env_name,
        seed=seed,
        total_steps=total_steps,
        log_path=log_path,
        checkpoint_path=ckpt_path,
        verbose=True,
        **params,
    )
    train(cfg)
    return log_path


def run_final(
    env_name: str,
    seeds: list[int],
    total_steps: int | None,
    params_path: str | None,
    n_jobs: int,
) -> None:
    params_path = params_path or f"results/best_{env_name}.json"
    with open(params_path) as f:
        payload = json.load(f)
    params = payload["best_params"]

    total_steps = total_steps or DEFAULT_FINAL_STEPS.get(env_name, 50_000)

    os.makedirs("results", exist_ok=True)
    print(f"Final runs on {env_name} for seeds={seeds}, steps={total_steps}")
    print(f"Params: {params}")

    if n_jobs <= 1 or len(seeds) == 1:
        for s in seeds:
            _run_one(env_name, s, total_steps, params)
        return

    # Each seed runs in its own process; with torch threads pinned to 1, this
    # cleanly parallelises across cores.
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(_run_one, env_name, s, total_steps, params): s for s in seeds
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                path = fut.result()
                print(f"seed={s} done -> {path}")
            except Exception as e:
                print(f"seed={s} failed: {e}")
                raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--params", default=None, help="Path to best_<env>.json")
    p.add_argument("--n-jobs", type=int, default=3, help="Parallel seeds")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_final(
        env_name=args.env,
        seeds=args.seeds,
        total_steps=args.total_steps,
        params_path=args.params,
        n_jobs=args.n_jobs,
    )
