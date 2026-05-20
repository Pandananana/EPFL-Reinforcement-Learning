"""End-to-end orchestrator: sweep -> final runs -> plot.

Examples:
    # Full pipeline on Pendulum (quickest):
    python main.py all --env Pendulum-v1

    # Just smoke-test the implementation:
    python main.py smoke

    # Run one phase:
    python main.py sweep --env Pendulum-v1 --n-trials 30
    python main.py final --env Pendulum-v1
    python main.py plot  --env Pendulum-v1
"""
from __future__ import annotations

import argparse

import torch

from final_runs import run_final
from play import play
from plot import plot_env
from sweep import run_sweep
from train import TrainConfig, train


def cmd_smoke(args):
    print("Smoke test: 5k steps on Pendulum-v1, default hyperparams.")
    train(
        TrainConfig(
            env_name="Pendulum-v1",
            total_steps=5_000,
            eval_every=500,
            verbose=True,
        )
    )


def cmd_sweep(args):
    torch.set_num_threads(1)
    run_sweep(
        env_name=args.env,
        n_trials=args.n_trials,
        total_steps=args.total_steps,
        n_jobs=args.n_jobs,
        seed=args.seed,
        study_name=None,
        storage="sqlite:///sweeps.db",
    )


def cmd_final(args):
    run_final(
        env_name=args.env,
        seeds=args.seeds,
        total_steps=args.total_steps,
        params_path=None,
        n_jobs=args.n_jobs,
    )


def cmd_plot(args):
    plot_env(args.env)


def cmd_play(args):
    checkpoint = args.checkpoint or f"models/{args.env}_seed{args.seed}.pt"
    play(
        env_name=args.env,
        checkpoint=checkpoint,
        episodes=args.episodes,
        deterministic=not args.stochastic,
        fps=args.fps,
    )


def cmd_all(args):
    print(f"== Phase 1: sweep on {args.env} ==")
    cmd_sweep(args)
    print(f"\n== Phase 2: multi-seed final runs on {args.env} ==")
    cmd_final(args)
    print(f"\n== Phase 3: plot {args.env} ==")
    cmd_plot(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("smoke", help="Quick sanity training run.")
    sp.set_defaults(func=cmd_smoke)

    sp = sub.add_parser("sweep", help="Run Optuna hyperparameter sweep.")
    sp.add_argument("--env", default="Pendulum-v1")
    sp.add_argument("--n-trials", type=int, default=30)
    sp.add_argument("--total-steps", type=int, default=None)
    sp.add_argument("--n-jobs", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("final", help="Multi-seed runs with the swept best params.")
    sp.add_argument("--env", default="Pendulum-v1")
    sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sp.add_argument("--total-steps", type=int, default=None)
    sp.add_argument("--n-jobs", type=int, default=3)
    sp.set_defaults(func=cmd_final)

    sp = sub.add_parser("plot", help="Plot per-env multi-seed eval curves.")
    sp.add_argument("--env", default="Pendulum-v1")
    sp.set_defaults(func=cmd_plot)

    sp = sub.add_parser("play", help="Render a trained policy in a window.")
    sp.add_argument("--env", default="Pendulum-v1")
    sp.add_argument("--checkpoint", default=None,
                    help="Path to .pt file. Defaults to models/<env>_seed<seed>.pt.")
    sp.add_argument("--seed", type=int, default=0,
                    help="Seed checkpoint to load if --checkpoint is unset.")
    sp.add_argument("--episodes", type=int, default=5)
    sp.add_argument("--stochastic", action="store_true",
                    help="Sample from the policy instead of taking the mean action.")
    sp.add_argument("--fps", type=int, default=None)
    sp.set_defaults(func=cmd_play)

    sp = sub.add_parser("all", help="Run sweep + final + plot for one env.")
    sp.add_argument("--env", default="Pendulum-v1")
    sp.add_argument("--n-trials", type=int, default=30)
    sp.add_argument("--total-steps", type=int, default=None)
    sp.add_argument("--n-jobs", type=int, default=4)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sp.set_defaults(func=cmd_all)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
