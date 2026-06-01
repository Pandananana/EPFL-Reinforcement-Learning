from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from train import TrainConfig, train

DEFAULT_FINAL_EPISODES = {
    "Pendulum-v1": 200,
    "MountainCarContinuous-v0": 200,
}

PAPER_DEFAULTS = {
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "batch_size": 256,
    "hidden_dim": 256,
}


def _run_one(
    env_name: str, seed: int, total_episodes: int, params: dict, tag: str = ""
) -> str:
    torch.set_num_threads(1)
    suffix = f"_{tag}" if tag else ""
    log_path = f"results/{env_name}{suffix}_seed{seed}.csv"
    ckpt_path = f"models/{env_name}{suffix}_seed{seed}.pt"
    cfg = TrainConfig(
        env_name=env_name,
        seed=seed,
        total_episodes=total_episodes,
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
    total_episodes: int | None,
    params_path: str | None,
    n_jobs: int,
    vanilla: bool = False,
) -> None:
    # Vanilla: paper-default params + a "_vanilla" filename tag, so tuned results
    # (results/<env>_seed*.csv, models/<env>_seed*.pt) are never clobbered.
    if vanilla:
        params = dict(PAPER_DEFAULTS)
        tag = "vanilla"
    else:
        params_path = params_path or f"results/best_{env_name}.json"
        with open(params_path) as f:
            payload = json.load(f)
        params = payload["best_params"]
        tag = ""

    total_episodes = total_episodes or DEFAULT_FINAL_EPISODES.get(env_name, 200)

    os.makedirs("results", exist_ok=True)
    label = "vanilla (paper-default)" if vanilla else "tuned"
    print(f"Final runs [{label}] on {env_name} for seeds={seeds}, episodes={total_episodes}")
    print(f"Params: {params}")

    if n_jobs <= 1 or len(seeds) == 1:
        for s in seeds:
            _run_one(env_name, s, total_episodes, params, tag=tag)
        return

    # Each seed runs in its own process; with torch threads pinned to 1, this
    # cleanly parallelises across cores.
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = {
            ex.submit(_run_one, env_name, s, total_episodes, params, tag): s
            for s in seeds
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
    p.add_argument("--total-episodes", type=int, default=None)
    p.add_argument("--params", default=None, help="Path to best_<env>.json")
    p.add_argument("--n-jobs", type=int, default=3, help="Parallel seeds")
    p.add_argument(
        "--vanilla",
        action="store_true",
        help="Use paper-default hyperparameters (ignore the swept best_<env>.json) "
        "and write to results/<env>_vanilla_seed*.csv. Untuned, unmodified benchmark.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_final(
        env_name=args.env,
        seeds=args.seeds,
        total_episodes=args.total_episodes,
        params_path=args.params,
        n_jobs=args.n_jobs,
        vanilla=args.vanilla,
    )
