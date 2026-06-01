from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import PROJECT_ENVS, PPOConfig, get_config
from .hpo import PAPER_BASE
from .ppo import PPOAgent

PAPER_PPO_OVERRIDES: dict[str, Any] = {
    **PAPER_BASE,
    "clip_eps": 0.2,
    "lr": 3e-4,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
}

PAPER_TOTAL_TIMESTEPS = 200_000
PAPER_EVAL_INTERVAL = 10_000


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _device_info(dev: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device_type": dev.type}
    if dev.type == "cuda":
        info["cuda_device_name"] = torch.cuda.get_device_name(dev)
        info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(dev)))
    return info


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def train_ppo_with_metrics(
    cfg: PPOConfig,
    seed: int = 0,
    device: torch.device | None = None,
    log_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    t_start = time.perf_counter()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(cfg, device=dev, seed=seed)
    log_dir = Path(log_dir or "results") / cfg.env_id / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)

    n_actor = count_parameters(agent.net.actor_body) + (
        count_parameters(agent.net.mu) if agent.continuous else count_parameters(agent.net.logits)
    )
    n_critic = count_parameters(agent.net.critic_body) + count_parameters(agent.net.value_head)
    n_total = count_parameters(agent.net)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    timesteps = 0
    n_updates = 0
    episode_returns: list[float] = []
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    eval_success: list[float] = []
    update_step: list[int] = []
    policy_loss_h: list[float] = []
    value_loss_h: list[float] = []
    entropy_h: list[float] = []
    approx_kl_h: list[float] = []
    clipfrac_h: list[float] = []
    time_rollout, time_update, time_eval = 0.0, 0.0, 0.0

    while timesteps < cfg.total_timesteps:
        t0 = time.perf_counter()
        ep_rets = agent.collect_rollout()
        time_rollout += time.perf_counter() - t0
        episode_returns.extend(ep_rets)
        timesteps += cfg.n_steps

        t0 = time.perf_counter()
        stats = agent.update()
        time_update += time.perf_counter() - t0
        n_updates += 1

        update_step.append(timesteps)
        policy_loss_h.append(stats["policy_loss"])
        value_loss_h.append(stats["value_loss"])
        entropy_h.append(stats["entropy"])
        approx_kl_h.append(stats["approx_kl"])
        clipfrac_h.append(stats["clipfrac"])

        if timesteps % cfg.eval_interval < cfg.n_steps or timesteps >= cfg.total_timesteps:
            t0 = time.perf_counter()
            mean_ret, std_ret, succ = agent.evaluate(cfg.n_eval_episodes)
            time_eval += time.perf_counter() - t0
            eval_steps.append(timesteps)
            eval_returns.append(mean_ret)
            eval_success.append(succ)
            if verbose:
                print(
                    f"[{cfg.env_id}|seed={seed}] steps={timesteps:,} "
                    f"eval={mean_ret:.1f}±{std_ret:.1f} succ={succ:.0%} kl={stats['approx_kl']:.4f}"
                )

    t0 = time.perf_counter()
    final_mean, final_std, final_succ = agent.evaluate(cfg.n_eval_episodes)
    time_eval += time.perf_counter() - t0

    wall_total = time.perf_counter() - t_start
    peak_mem_mb = None
    if dev.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(dev) / (1024**2)

    obs_dim, act_dim, continuous = agent.obs_dim, agent.act_dim, agent.continuous
    agent.save(log_dir / "model.pt")
    agent.close()

    np.savez(
        log_dir / "logs.npz",
        episode_returns=np.array(episode_returns, dtype=np.float32),
        eval_steps=np.array(eval_steps, dtype=np.int64),
        eval_returns=np.array(eval_returns, dtype=np.float32),
        eval_success=np.array(eval_success, dtype=np.float32),
        update_step=np.array(update_step, dtype=np.int64),
        policy_loss=np.array(policy_loss_h, dtype=np.float32),
        value_loss=np.array(value_loss_h, dtype=np.float32),
        entropy=np.array(entropy_h, dtype=np.float32),
        approx_kl=np.array(approx_kl_h, dtype=np.float32),
        clipfrac=np.array(clipfrac_h, dtype=np.float32),
        timesteps=timesteps,
        final_mean=final_mean,
        final_std=final_std,
        final_success=final_succ,
        seed=seed,
    )

    metrics: dict[str, Any] = {
        "algorithm": "PPO",
        "env_id": cfg.env_id,
        "seed": seed,
        "config": cfg.__dict__,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "continuous": continuous,
        "n_parameters": {"actor": n_actor, "critic": n_critic, "total": n_total},
        "training": {
            "timesteps": timesteps,
            "n_updates": n_updates,
            "n_eval_checkpoints": len(eval_steps),
            "n_episodes_completed": len(episode_returns),
        },
        "final_eval": {"mean_return": final_mean, "std_return": final_std, "success_rate": final_succ},
        "compute": {
            "wall_time_total_sec": wall_total,
            "wall_time_rollout_sec": time_rollout,
            "wall_time_update_sec": time_update,
            "wall_time_eval_sec": time_eval,
            "steps_per_sec": timesteps / wall_total if wall_total > 0 else None,
            "sec_per_million_steps": 1e6 * wall_total / timesteps if timesteps > 0 else None,
            "peak_gpu_memory_mb": peak_mem_mb,
        },
        "update_stats_summary": {
            "approx_kl_mean": float(np.mean(approx_kl_h)) if approx_kl_h else None,
            "approx_kl_max": float(np.max(approx_kl_h)) if approx_kl_h else None,
            "clipfrac_mean": float(np.mean(clipfrac_h)) if clipfrac_h else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            **_device_info(dev),
        },
        "log_dir": str(log_dir),
    }

    with open(log_dir / "run_metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    return {
        "seed": seed,
        "metrics": metrics,
        "episode_returns": episode_returns,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "final_mean": final_mean,
        "final_std": final_std,
        "final_success": final_succ,
        "log_dir": str(log_dir),
    }


def run_paper_ppo(
    env_id: str,
    seeds: list[int] | None = None,
    results_dir: str | Path = "results/paper_baseline",
    total_timesteps: int | None = None,
    eval_interval: int | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    seeds = seeds or [0, 1, 2]
    results_dir = Path(results_dir)
    overrides = {
        **PAPER_PPO_OVERRIDES,
        "total_timesteps": total_timesteps or PAPER_TOTAL_TIMESTEPS,
        "eval_interval": eval_interval or PAPER_EVAL_INTERVAL,
        **(cfg_overrides or {}),
    }
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        if verbose:
            print(f"\n=== Paper PPO | {env_id} | seed={seed} ===")
        cfg = get_config(env_id, **overrides)
        out = train_ppo_with_metrics(cfg, seed=seed, device=dev, log_dir=results_dir, verbose=verbose)
        per_seed.append(out)

    finals = [s["final_mean"] for s in per_seed]
    succs = [s["final_success"] for s in per_seed]
    wall_times = [s["metrics"]["compute"]["wall_time_total_sec"] for s in per_seed]
    param_total = per_seed[0]["metrics"]["n_parameters"]["total"]

    env_summary = {
        "env_id": env_id,
        "seeds": seeds,
        "config": get_config(env_id, **overrides).__dict__,
        "final_eval_mean": float(np.mean(finals)),
        "final_eval_std_across_seeds": float(np.std(finals)),
        "per_seed_final": finals,
        "per_seed_success": succs,
        "mean_success_rate": float(np.mean(succs)),
        "compute": {
            "wall_time_total_sec_mean": float(np.mean(wall_times)),
            "wall_time_total_sec_std": float(np.std(wall_times)),
            "sec_per_million_steps_mean": float(
                np.mean([s["metrics"]["compute"]["sec_per_million_steps"] for s in per_seed])
            ),
            "peak_gpu_memory_mb_max": _max_optional(
                [s["metrics"]["compute"]["peak_gpu_memory_mb"] for s in per_seed]
            ),
        },
        "n_parameters_total": param_total,
        "per_seed_metrics_paths": [str(Path(s["log_dir"]) / "run_metrics.json") for s in per_seed],
    }

    env_dir = results_dir / env_id
    env_dir.mkdir(parents=True, exist_ok=True)
    with open(env_dir / "paper_summary.json", "w") as f:
        json.dump(_json_safe(env_summary), f, indent=2)

    return {"env_summary": env_summary, "per_seed": per_seed}


def run_paper_ppo_all_envs(
    env_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    results_dir: str | Path = "results/paper_baseline",
    **kwargs: Any,
) -> dict[str, Any]:
    env_ids = env_ids or list(PROJECT_ENVS)
    results_dir = Path(results_dir)
    all_env: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for env_id in env_ids:
        out = run_paper_ppo(env_id, seeds=seeds, results_dir=results_dir, **kwargs)
        all_env[env_id] = out
        s = out["env_summary"]
        rows.append({
            "env_id": env_id,
            "final_eval_mean": s["final_eval_mean"],
            "final_eval_std": s["final_eval_std_across_seeds"],
            "success_rate": s["mean_success_rate"],
            "wall_time_sec": s["compute"]["wall_time_total_sec_mean"],
            "sec_per_1M_steps": s["compute"]["sec_per_million_steps_mean"],
            "n_params": s["n_parameters_total"],
        })
    benchmark = {
        "description": "Paper-default PPO (no HPO); for cross-algorithm comparison",
        "env_ids": env_ids,
        "seeds": seeds or [0, 1, 2],
        "rows": rows,
        "per_env": {e: all_env[e]["env_summary"] for e in env_ids},
    }
    with open(results_dir / "benchmark_table.json", "w") as f:
        json.dump(_json_safe(benchmark), f, indent=2)
    return {"benchmark": benchmark, "all_env": all_env}


def _max_optional(vals: list[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return float(max(xs)) if xs else None
