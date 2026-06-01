from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from src.config import PROJECT_ENVS, get_config
from src.ppo import train_ppo

PAPER_DEFAULT: dict[str, Any] = dict(
    n_steps=2048,
    n_epochs=10,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=3e-4,
    vf_coef=0.5,
    ent_coef=0.0,
    max_grad_norm=0.5,
    hidden_sizes=(64, 64),
    log_std_init=0.0,
    normalize_obs=False,
    reward_shaping=False,
)

OFAT_GRID: dict[str, list[Any]] = {
    "lr": [3e-4, 1e-4, 1e-3, 3e-3],
    "clip_eps": [0.2, 0.1, 0.3],
    "ent_coef": [0.0, 0.01, 0.05],
    "gae_lambda": [0.95, 0.9, 0.99, 1.0],
    "gamma": [0.99, 0.95, 0.999],
    "n_epochs": [10, 5, 20],
    "batch_size": [64, 128, 256],
    "n_steps": [2048, 1024, 4096],
    "hidden_sizes": [(64, 64), (32, 32), (128, 128)],
    "log_std_init": [0.0, -0.5, -1.0],
}

JOINT_SPACE: dict[str, dict[str, Any]] = {
    "lr":         {"type": "loguniform", "low": 1e-4, "high": 3e-3},
    "clip_eps":   {"type": "categorical", "choices": [0.1, 0.2, 0.3]},
    "ent_coef":   {"type": "categorical", "choices": [0.0, 0.01, 0.05]},
    "gae_lambda": {"type": "uniform", "low": 0.9, "high": 1.0},
    "n_epochs":   {"type": "categorical", "choices": [5, 10, 20]},
    "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
}

JOINT_SPACE_HARD: dict[str, dict[str, Any]] = {
    "lr":           {"type": "loguniform", "low": 5e-5, "high": 1e-2},
    "clip_eps":     {"type": "categorical", "choices": [0.1, 0.2, 0.3, 0.4]},
    "ent_coef":     {"type": "loguniform", "low": 1e-4, "high": 0.1},
    "gae_lambda":   {"type": "uniform", "low": 0.8, "high": 1.0},
    "gamma":        {"type": "categorical", "choices": [0.9, 0.95, 0.99, 0.995, 0.999]},
    "n_epochs":     {"type": "categorical", "choices": [3, 5, 10, 15, 20, 30]},
    "batch_size":   {"type": "categorical", "choices": [32, 64, 128, 256, 512]},
    "n_steps":      {"type": "categorical", "choices": [512, 1024, 2048, 4096]},
    "hidden_sizes": {"type": "categorical", "choices": [(32, 32), (64, 64), (128, 128), (256, 256)]},
}


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@dataclass
class RunResult:
    label: str
    hparams: dict[str, Any]
    final_means: list[float]
    eval_steps: list[list[int]]
    eval_returns: list[list[float]]
    wall_time_sec: float

    @property
    def score(self) -> float:
        return float(np.mean(self.final_means))

    @property
    def score_std(self) -> float:
        return float(np.std(self.final_means))

    def to_dict(self) -> dict[str, Any]:
        return _json_safe({
            "label": self.label,
            "hparams": self.hparams,
            "final_means": self.final_means,
            "score": self.score,
            "score_std": self.score_std,
            "eval_steps": self.eval_steps,
            "eval_returns": self.eval_returns,
            "wall_time_sec": self.wall_time_sec,
        })


def train_point(
    env_id: str,
    hparams: dict[str, Any],
    seeds: Iterable[int],
    base: dict[str, Any],
    log_dir: Path,
    device: torch.device,
    verbose: bool = False,
    progress: Callable[[int, int, float, float], None] | None = None,
    tag: str = "",
) -> RunResult:
    overrides = {**base, **hparams}
    seeds = list(seeds)
    finals, steps_all, rets_all = [], [], []
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds):
        if verbose and tag:
            print(f"  {tag} | seed {seed} training...", flush=True)
        cfg = get_config(env_id, **overrides)
        out = train_ppo(cfg, seed=seed, device=device, log_dir=log_dir, verbose=verbose)
        finals.append(out["final_mean"])
        steps_all.append(list(out["eval_steps"]))
        rets_all.append(list(out["eval_returns"]))
        if progress is not None:
            progress(i + 1, len(seeds), out["final_mean"], time.perf_counter() - t0)
    return RunResult(
        label=str(hparams),
        hparams=hparams,
        final_means=finals,
        eval_steps=steps_all,
        eval_returns=rets_all,
        wall_time_sec=time.perf_counter() - t0,
    )


def sweep_one_hparam(
    env_id: str,
    hparam: str,
    values: list[Any] | None = None,
    seeds: list[int] | None = None,
    total_timesteps: int = 200_000,
    eval_interval: int = 10_000,
    base: dict[str, Any] | None = None,
    results_dir: str | Path = "results/ofat",
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    seeds = seeds or [0, 1, 2]
    values = values if values is not None else OFAT_GRID[hparam]
    base = {**(base or PAPER_DEFAULT), "total_timesteps": total_timesteps, "eval_interval": eval_interval}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(results_dir).resolve() / env_id / hparam
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {out_dir}", flush=True)

    runs: list[RunResult] = []
    n_vals = len(values)
    print(f"OFAT sweep [{env_id}] {hparam}: {n_vals} values x {len(seeds)} seeds "
          f"= {n_vals * len(seeds)} runs", flush=True)
    for vi, v in enumerate(values, 1):
        def _prog(s, n_s, fm, el, _vi=vi, _v=v):
            print(f"  [{_vi}/{n_vals}] {hparam}={_v}  seed {s}/{n_s} done "
                  f"(final={fm:.1f}, {el:.0f}s)", flush=True)
        r = train_point(
            env_id, {hparam: v}, seeds, base,
            log_dir=out_dir / f"{hparam}={_tag(v)}", device=dev, verbose=verbose,
            progress=_prog, tag=f"[{vi}/{n_vals}] {hparam}={v}",
        )
        r.label = f"{hparam}={v}"
        runs.append(r)
        print(f"  -> {hparam}={v}: score={r.score:.1f} +/- {r.score_std:.1f}", flush=True)

    default_v = base.get(hparam, PAPER_DEFAULT.get(hparam))
    summary = {
        "method": "ofat",
        "env_id": env_id,
        "hparam": hparam,
        "values": _json_safe(values),
        "default_value": _json_safe(default_v),
        "seeds": seeds,
        "base": _json_safe(base),
        "runs": [r.to_dict() for r in runs],
        "scores": {str(v): r.score for v, r in zip(values, runs)},
        "best_value": _json_safe(max(zip(values, runs), key=lambda p: p[1].score)[0]),
        "sensitivity_range": float(max(r.score for r in runs) - min(r.score for r in runs)),
    }
    _dump(out_dir / "sweep.json", summary)
    return summary


def sweep_sensitivity_report(
    env_id: str,
    hparams: list[str] | None = None,
    seeds: list[int] | None = None,
    grid: dict[str, list[Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    grid = grid or OFAT_GRID
    hparams = hparams or list(grid.keys())
    sweeps = {h: sweep_one_hparam(env_id, h, values=grid.get(h), seeds=seeds, **kwargs) for h in hparams}
    ranking = sorted(
        ({"hparam": h, "sensitivity_range": s["sensitivity_range"], "best_value": s["best_value"]}
         for h, s in sweeps.items()),
        key=lambda d: d["sensitivity_range"], reverse=True,
    )
    report = {"env_id": env_id, "ranking": ranking, "sweeps": sweeps}
    out_dir = Path(kwargs.get("results_dir", "results/ofat")) / env_id
    _dump(out_dir / "sensitivity_report.json", report)
    return report


def _sample_joint(rng: random.Random, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            out[k] = rng.choice(spec["choices"])
        elif t == "uniform":
            out[k] = rng.uniform(spec["low"], spec["high"])
        elif t == "loguniform":
            lo, hi = np.log10(spec["low"]), np.log10(spec["high"])
            out[k] = float(10 ** rng.uniform(lo, hi))
        else:
            raise ValueError(f"unknown spec type {t!r}")
    return out


def random_search(
    env_id: str,
    n_trials: int = 20,
    seeds: list[int] | None = None,
    space: dict[str, dict[str, Any]] | None = None,
    total_timesteps: int = 200_000,
    eval_interval: int = 10_000,
    base: dict[str, Any] | None = None,
    results_dir: str | Path = "results/random",
    rng_seed: int = 42,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    seeds = seeds or [0, 1, 2]
    space = space or JOINT_SPACE
    base = {**(base or PAPER_DEFAULT), "total_timesteps": total_timesteps, "eval_interval": eval_interval}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(results_dir) / env_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed)

    runs: list[RunResult] = []
    best_so_far: list[float] = []
    n_knobs = len(space)
    print(f"\n{'='*60}", flush=True)
    print(f"Random search [{env_id}]: {n_trials} trials x {len(seeds)} seeds", flush=True)
    print(f"  timesteps/trial: {total_timesteps:,}  |  knobs: {n_knobs}  |  verbose: {verbose}", flush=True)
    print(f"  search space: {list(space.keys())}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for t in range(n_trials):
        hp = _sample_joint(rng, space)
        print(f"\n{'─'*50}", flush=True)
        print(f"Trial {t+1}/{n_trials}", flush=True)
        for k, v in hp.items():
            print(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
        print(f"{'─'*50}", flush=True)

        def _prog(s, n_s, fm, el, _t=t):
            print(f"  [trial {_t+1}] seed {s}/{n_s} done — final={fm:.1f}, time={el:.0f}s", flush=True)

        r = train_point(env_id, hp, seeds, base, log_dir=out_dir / f"trial_{t:04d}",
                        device=dev, verbose=verbose, progress=_prog)
        r.label = f"trial_{t}"
        runs.append(r)
        best_so_far.append(max(x.score for x in runs))
        print(f"\n  > Trial {t+1} score: {r.score:.1f} ± {r.score_std:.1f}  |  best so far: {best_so_far[-1]:.1f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    best = max(runs, key=lambda r: r.score)
    print(f"Random search complete. Best score: {best.score:.1f}", flush=True)
    print(f"Best hparams: {best.hparams}", flush=True)
    print(f"{'='*60}\n", flush=True)

    return _finalize_search("random", env_id, runs, best_so_far, seeds, base, space, out_dir)


def grid_search(
    env_id: str,
    grid: dict[str, list[Any]],
    seeds: list[int] | None = None,
    total_timesteps: int = 200_000,
    eval_interval: int = 10_000,
    base: dict[str, Any] | None = None,
    results_dir: str | Path = "results/grid",
    device: torch.device | str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    seeds = seeds or [0, 1, 2]
    base = {**(base or PAPER_DEFAULT), "total_timesteps": total_timesteps, "eval_interval": eval_interval}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(results_dir) / env_id
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    runs: list[RunResult] = []
    best_so_far: list[float] = []
    print(f"Grid search [{env_id}]: {len(combos)} combos x {len(seeds)} seeds", flush=True)
    for t, vals in enumerate(combos):
        hp = dict(zip(keys, vals))
        print(f"\n=== grid trial {t}/{len(combos) - 1} | {hp} ===", flush=True)
        def _prog(s, n_s, fm, el, _t=t):
            print(f"  trial {_t}: seed {s}/{n_s} done (final={fm:.1f}, {el:.0f}s)", flush=True)
        r = train_point(env_id, hp, seeds, base, log_dir=out_dir / f"trial_{t:04d}",
                        device=dev, verbose=verbose, progress=_prog)
        r.label = f"trial_{t}"
        runs.append(r)
        best_so_far.append(max(x.score for x in runs))
        print(f"  -> trial {t}: score={r.score:.1f}  (best so far={best_so_far[-1]:.1f})", flush=True)

    return _finalize_search("grid", env_id, runs, best_so_far, seeds, base, grid, out_dir)


def optuna_search(
    env_id: str,
    n_trials: int = 20,
    seeds: list[int] | None = None,
    space: dict[str, dict[str, Any]] | None = None,
    total_timesteps: int = 200_000,
    eval_interval: int = 10_000,
    base: dict[str, Any] | None = None,
    results_dir: str | Path = "results/optuna",
    sampler_seed: int = 42,
    device: torch.device | str | None = None,
    verbose: bool = True,
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seeds = seeds or [0, 1, 2]
    space = space or JOINT_SPACE
    base = {**(base or PAPER_DEFAULT), "total_timesteps": total_timesteps, "eval_interval": eval_interval}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(results_dir) / env_id
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: list[RunResult] = []
    n_knobs = len(space)

    def suggest(trial: "optuna.Trial") -> dict[str, Any]:
        hp: dict[str, Any] = {}
        for k, spec in space.items():
            t = spec["type"]
            if t == "categorical":
                hp[k] = trial.suggest_categorical(k, spec["choices"])
            elif t == "uniform":
                hp[k] = trial.suggest_float(k, spec["low"], spec["high"])
            elif t == "loguniform":
                hp[k] = trial.suggest_float(k, spec["low"], spec["high"], log=True)
        return hp

    def objective(trial: "optuna.Trial") -> float:
        hp = suggest(trial)
        print(f"\n{'─'*50}", flush=True)
        print(f"Optuna Trial {trial.number+1}/{n_trials}", flush=True)
        for k, v in hp.items():
            print(f"  {k}: {v:.5g}" if isinstance(v, float) else f"  {k}: {v}", flush=True)
        print(f"{'─'*50}", flush=True)

        def _prog(s, n_s, fm, el, _n=trial.number):
            print(f"  [trial {_n+1}] seed {s}/{n_s} done — final={fm:.1f}, time={el:.0f}s", flush=True)

        r = train_point(env_id, hp, seeds, base, log_dir=out_dir / f"trial_{trial.number:04d}",
                        device=dev, verbose=verbose, progress=_prog)
        r.label = f"trial_{trial.number}"
        runs.append(r)
        trial.set_user_attr("score_std", r.score_std)
        prev = [t.value for t in study.trials if t.value is not None]
        best_so_far = max(prev) if prev else r.score
        print(f"\n  > Trial {trial.number+1} score: {r.score:.1f} ± {r.score_std:.1f}  |  best so far: {max(best_so_far, r.score):.1f}", flush=True)
        return r.score

    print(f"\n{'='*60}", flush=True)
    print(f"Optuna TPE search [{env_id}]: {n_trials} trials x {len(seeds)} seeds", flush=True)
    print(f"  timesteps/trial: {total_timesteps:,}  |  knobs: {n_knobs}  |  verbose: {verbose}", flush=True)
    print(f"  search space: {list(space.keys())}", flush=True)
    print(f"{'='*60}\n", flush=True)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=sampler_seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)

    print(f"\n{'='*60}", flush=True)
    print(f"Optuna search complete. Best score: {study.best_value:.1f}", flush=True)
    print(f"Best hparams: {study.best_params}", flush=True)
    print(f"{'='*60}\n", flush=True)

    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = list(np.maximum.accumulate(values)) if values else []
    summary = _finalize_search("optuna_tpe", env_id, runs, best_so_far, seeds, base, space, out_dir)
    summary["study"] = study
    summary["best_params"] = study.best_params
    summary["best_value"] = study.best_value
    _dump(out_dir / "best_config.json", {
        "method": "optuna_tpe", "env_id": env_id, "best_value": study.best_value,
        "best_params": _json_safe(study.best_params), "best_overrides": _json_safe({**base, **study.best_params}),
        "seeds": seeds,
    })
    return summary


def _finalize_search(method, env_id, runs, best_so_far, seeds, base, space, out_dir) -> dict[str, Any]:
    best = max(runs, key=lambda r: r.score)
    summary = {
        "method": method,
        "env_id": env_id,
        "n_trials": len(runs),
        "seeds": seeds,
        "base": _json_safe(base),
        "search_space": _json_safe(space),
        "best_score": best.score,
        "best_hparams": _json_safe(best.hparams),
        "best_overrides": _json_safe({**base, **best.hparams}),
        "best_so_far": [float(x) for x in best_so_far],
        "trials": [r.to_dict() for r in runs],
    }
    _dump(out_dir / "search.json", summary)
    return summary


def _tag(v: Any) -> str:
    if isinstance(v, (list, tuple)):
        return "x".join(str(x) for x in v)
    return str(v).replace(".", "p").replace("-", "m")


def _dump(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in obj.items() if k != "study"}
    with open(path, "w") as f:
        json.dump(_json_safe(serializable), f, indent=2)


def load_summary(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)
