
from __future__ import annotations

import sys

from train import TrainConfig, train


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    total_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    cfg = TrainConfig(
        env_name="MountainCarContinuous-v0",
        seed=seed,
        total_episodes=total_episodes,
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
