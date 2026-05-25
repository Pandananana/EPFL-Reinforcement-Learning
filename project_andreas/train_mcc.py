"""Hand-tuned MCC training run.

Two fixes needed beyond defaults; diagnosed empirically:

1. warmup_action_repeat=10. The per-step iid uniform random actions used by the
   paper's warmup never reach the goal on MCC (max position seen across 25
   episodes was ~-0.21, goal is +0.45). The action-energy variance per step is
   too small to build the velocity required to escape the valley. Repeating
   each random action for 10 steps concentrates the variance so warmup
   reliably reaches max velocity (sigma_vel ~ 0.087 > max_vel 0.07). This
   doesn't touch the SAC update rule.

2. gamma=0.9999. With the project default gamma=0.99 and ~500 steps to reach
   the goal, the +100 reward gets discounted to ~0.66 -- on par with the
   accumulated discounted action penalty. The agent rationally learns "do
   nothing". gamma=0.9999 brings the discounted goal value to ~95.

alpha is kept at 0.2 (paper's working range; equivalent to reward_scale=5).
"""
from __future__ import annotations

import sys

from train import TrainConfig, train


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60_000
    cfg = TrainConfig(
        env_name="MountainCarContinuous-v0",
        seed=seed,
        total_steps=total_steps,
        start_steps=10_000,
        warmup_action_repeat=10,
        update_after=1_000,
        eval_every=2_000,
        n_eval_episodes=5,
        gamma=0.9999,
        alpha=0.2,
        tau=0.005,
        lr=3e-4,
        batch_size=256,
        hidden_dim=256,
        verbose=True,
        log_path=f"results/MountainCarContinuous-v0_seed{seed}.csv",
        checkpoint_path=f"models/MountainCarContinuous-v0_seed{seed}.pt",
    )
    train(cfg)


if __name__ == "__main__":
    main()
