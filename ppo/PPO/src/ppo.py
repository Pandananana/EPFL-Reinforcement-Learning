from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions import Categorical, Normal

from .config import PPOConfig
from .env import episode_success, make_env, reset_env, step_env


def mlp(sizes: list[int], activation=nn.Tanh) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        continuous: bool,
        hidden_sizes: tuple[int, ...] = (64, 64),
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.continuous = continuous
        self.act_dim = act_dim
        h = list(hidden_sizes)
        self.actor_body = mlp([obs_dim, *h])
        self.critic_body = mlp([obs_dim, *h])
        if continuous:
            self.mu = nn.Linear(h[-1], act_dim)
            self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        else:
            self.logits = nn.Linear(h[-1], act_dim)
        self.value_head = nn.Linear(h[-1], 1)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, obs: torch.Tensor):
        a_feat = self.actor_body(obs)
        c_feat = self.critic_body(obs)
        value = self.value_head(c_feat).squeeze(-1)
        if self.continuous:
            mu = self.mu(a_feat)
            std = self.log_std.exp().expand_as(mu)
            dist = Normal(mu, std)
        else:
            dist = Categorical(logits=self.logits(a_feat))
        return dist, value

    def act(self, obs: np.ndarray, action_low=None, action_high=None, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.forward(obs_t)
            if self.continuous:
                raw = dist.mean if deterministic else dist.sample()
                env_action = raw.squeeze(0).cpu().numpy()
                if action_low is not None and action_high is not None:
                    env_action = np.clip(env_action, action_low, action_high)
                log_prob = dist.log_prob(raw).sum(-1)
                return env_action, float(log_prob.item()), float(value.item()), raw.squeeze(0).cpu().numpy()
            a = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item()), int(a.item())

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, values = self.forward(obs)
        if self.continuous:
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            log_prob = dist.log_prob(actions.long())
            entropy = dist.entropy()
        return log_prob, entropy, values


class RolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, act_dim: int, continuous: bool, gamma: float, gae_lambda: float):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.continuous = continuous
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.obs = np.zeros((n_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim if continuous else 1), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, log_prob, value):
        i = self.ptr
        self.obs[i] = obs
        if self.continuous:
            self.actions[i] = action
        else:
            self.actions[i, 0] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool):
        gae = 0.0
        next_value = last_value
        next_non_terminal = 1.0 - float(last_done)
        for t in reversed(range(self.n_steps)):
            delta = self.rewards[t] + self.gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]
            next_value = self.values[t]
            next_non_terminal = 1.0 - self.dones[t]

    def get_batches(self, batch_size: int):
        n = self.n_steps
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            mb = idx[start : start + batch_size]
            actions = self.actions[mb]
            if not self.continuous:
                actions = actions.squeeze(-1)
            yield (
                self.obs[mb],
                actions,
                self.log_probs[mb],
                self.advantages[mb],
                self.returns[mb],
            )

    def reset(self):
        self.ptr = 0


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PPOAgent:
    def __init__(self, cfg: PPOConfig, device: torch.device | None = None, seed: int = 0):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)
        self.seed = seed
        self.env, obs_dim, act_dim, self.continuous = make_env(
            cfg.env_id,
            seed=seed,
            normalize_obs=cfg.normalize_obs,
            reward_shaping=cfg.reward_shaping,
        )
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.action_low = self.action_high = None
        if self.continuous:
            self.action_low = self.env.action_space.low.astype(np.float32)
            self.action_high = self.env.action_space.high.astype(np.float32)
        self.net = ActorCritic(
            obs_dim, act_dim, self.continuous,
            hidden_sizes=cfg.hidden_sizes,
            log_std_init=cfg.log_std_init,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            cfg.n_steps, obs_dim, act_dim, self.continuous, cfg.gamma, cfg.gae_lambda,
        )

    def _to_action_tensor(self, actions_np: np.ndarray) -> torch.Tensor:
        if self.continuous:
            return torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        return torch.as_tensor(actions_np, dtype=torch.int64, device=self.device)

    def collect_rollout(self) -> list[float]:
        cfg = self.cfg
        obs = reset_env(self.env, seed=None)
        episode_returns: list[float] = []
        ep_ret = 0.0
        last_done = False
        for _ in range(cfg.n_steps):
            if self.continuous:
                env_action, log_prob, value, store_action = self.net.act(
                    obs, self.action_low, self.action_high, deterministic=False,
                )
            else:
                env_action, log_prob, value, store_action = self.net.act(obs, deterministic=False)
            next_obs, reward, done, _ = step_env(self.env, env_action)
            self.buffer.add(obs, store_action, reward, done, log_prob, value)
            last_done = done
            ep_ret += reward
            obs = next_obs
            if done:
                episode_returns.append(ep_ret)
                ep_ret = 0.0
                obs = reset_env(self.env)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_v = self.net(obs_t)
            last_value = float(last_v.squeeze().cpu())
        self.buffer.compute_gae(last_value, last_done=last_done)
        return episode_returns

    def update(self) -> dict[str, float]:
        cfg = self.cfg
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        n_updates = 0
        for _ in range(cfg.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in self.buffer.get_batches(cfg.batch_size):
                obs_t = torch.as_tensor(obs_b, dtype=torch.float32, device=self.device)
                act_t = self._to_action_tensor(act_b)
                old_lp = torch.as_tensor(old_lp_b, device=self.device)
                adv = torch.as_tensor(adv_b, device=self.device)
                ret = torch.as_tensor(ret_b, device=self.device)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                log_prob, entropy, values = self.net.evaluate(obs_t, act_t)
                ratio = (log_prob - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nnf.mse_loss(values, ret)
                ent_loss = -entropy.mean()
                loss = policy_loss + cfg.vf_coef * value_loss + cfg.ent_coef * ent_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    approx_kl = (old_lp - log_prob).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean().item()
                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                n_updates += 1
        self.buffer.reset()
        if n_updates:
            for k in stats:
                stats[k] /= n_updates
        return stats

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> tuple[float, float, float]:
        returns, successes = [], []
        for ep in range(n_episodes):
            obs = reset_env(self.env, seed=self.seed + 10_000 + ep)
            done, ep_ret, steps = False, 0.0, 0
            last_info: dict = {}
            while not done and steps < 1000:
                if self.continuous:
                    action, _, _, _ = self.net.act(
                        obs, self.action_low, self.action_high, deterministic=deterministic,
                    )
                else:
                    action, _, _, _ = self.net.act(obs, deterministic=deterministic)
                obs, reward, done, last_info = step_env(self.env, action)
                ep_ret += reward
                steps += 1
            returns.append(ep_ret)
            successes.append(
                episode_success(self.cfg.env_id, obs, done, last_info)
                or (self.cfg.env_id == "CartPole-v1" and ep_ret >= 499)
            )
        return float(np.mean(returns)), float(np.std(returns)), float(np.mean(successes))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "state_dict": self.net.state_dict(), "seed": self.seed}, path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["state_dict"])

    def close(self):
        self.env.close()


def train_ppo(
    cfg: PPOConfig,
    seed: int = 0,
    device: torch.device | None = None,
    log_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    agent = PPOAgent(cfg, device=device, seed=seed)
    log_dir = Path(log_dir or "results") / cfg.env_id / f"seed_{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    timesteps = 0
    episode_returns: list[float] = []
    eval_steps: list[int] = []
    eval_returns: list[float] = []
    while timesteps < cfg.total_timesteps:
        ep_rets = agent.collect_rollout()
        episode_returns.extend(ep_rets)
        timesteps += cfg.n_steps
        stats = agent.update()
        if timesteps % cfg.eval_interval < cfg.n_steps or timesteps >= cfg.total_timesteps:
            mean_ret, _, succ = agent.evaluate(cfg.n_eval_episodes)
            eval_steps.append(timesteps)
            eval_returns.append(mean_ret)
            if verbose:
                print(
                    f"[{cfg.env_id}|seed={seed}] steps={timesteps:,} "
                    f"eval_return={mean_ret:.1f} success={succ:.0%} kl={stats['approx_kl']:.4f}"
                )
    final_mean, final_std, final_succ = agent.evaluate(cfg.n_eval_episodes)
    agent.save(log_dir / "model.pt")
    agent.close()
    np.savez(
        log_dir / "logs.npz",
        episode_returns=np.array(episode_returns, dtype=np.float32),
        eval_steps=np.array(eval_steps, dtype=np.int64),
        eval_returns=np.array(eval_returns, dtype=np.float32),
        timesteps=timesteps,
        final_mean=final_mean,
        final_std=final_std,
        final_success=final_succ,
        seed=seed,
    )
    return {
        "seed": seed,
        "episode_returns": episode_returns,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
        "final_mean": final_mean,
        "final_std": final_std,
        "final_success": final_succ,
        "log_dir": str(log_dir),
    }
