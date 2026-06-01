from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(act())
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.idx] = obs
        self.act[self.idx] = act
        self.rew[self.idx] = rew
        self.next_obs[self.idx] = next_obs
        self.done[self.idx] = done
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        t = lambda x: torch.from_numpy(x[idx]).to(self.device)
        return t(self.obs), t(self.act), t(self.rew), t(self.next_obs), t(self.done)


class Actor(nn.Module):
    """Gaussian policy with tanh squashing to bound actions to [-act_limit, act_limit]."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_limit: float):
        super().__init__()
        self.trunk = mlp([obs_dim, hidden_dim, hidden_dim], output_activation=nn.ReLU)
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic: bool = False, with_logprob: bool = True):
        h = self.trunk(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mu, std)
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)
        if with_logprob:
            # Change of variables for tanh squashing — numerically stable form of
            # log(1 - tanh(u)**2) = 2 * (log 2 - u - softplus(-2u)).
            log_pi = dist.log_prob(u).sum(-1) - (
                2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
            ).sum(-1)
        else:
            log_pi = None
        return a * self.act_limit, log_pi


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1)).squeeze(-1)


class VNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.v = mlp([obs_dim, hidden_dim, hidden_dim, 1])

    def forward(self, obs):
        return self.v(obs).squeeze(-1)


class SAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = float(alpha)
        self.device = device

        self.actor = Actor(obs_dim, act_dim, hidden_dim, act_limit).to(device)
        self.q1 = QNet(obs_dim, act_dim, hidden_dim).to(device)
        self.q2 = QNet(obs_dim, act_dim, hidden_dim).to(device)
        self.v = VNet(obs_dim, hidden_dim).to(device)
        self.v_t = VNet(obs_dim, hidden_dim).to(device)
        self.v_t.load_state_dict(self.v.state_dict())
        for p in self.v_t.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(obs_t, deterministic=deterministic, with_logprob=False)
        return a.cpu().numpy().squeeze(0)

    def save(self, path: str, extra: dict | None = None) -> None:
        ck = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "v": self.v.state_dict(),
            "v_t": self.v_t.state_dict(),
            "alpha": self.alpha,
            "act_limit": self.actor.act_limit,
        }
        if extra:
            ck["extra"] = extra
        torch.save(ck, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
        self.v.load_state_dict(ck["v"])
        self.v_t.load_state_dict(ck["v_t"])
        self.alpha = float(ck.get("alpha", self.alpha))

    def update(self, batch) -> dict:
        obs, act, rew, next_obs, done = batch

        # --- Q update (eq. 7-9): target uses the target V network ----------------
        with torch.no_grad():
            q_target = rew + self.gamma * (1.0 - done) * self.v_t(next_obs)

        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --- V update (eq. 5-6): V(s) regresses to E_{a~pi}[Q_min(s,a) - alpha log pi(a|s)]
        # Action is freshly sampled from the current policy (paper: single sample).
        new_a, logp = self.actor(obs)
        with torch.no_grad():
            q_min_for_v = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
            v_target = q_min_for_v - self.alpha * logp

        v_pred = self.v(obs)
        v_loss = F.mse_loss(v_pred, v_target)

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # --- Policy update (eq. 10, 12-13) ---------------------------------------
        # Reuse (new_a, logp) — they were sampled with reparameterization.
        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        q_min = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        actor_loss = (self.alpha * logp - q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        # --- Polyak target update on V (Algorithm 1: psibar <- tau*psi + (1-tau)*psibar)
        with torch.no_grad():
            for p, tp in zip(self.v.parameters(), self.v_t.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {
            "q_loss": float(q_loss.detach()),
            "v_loss": float(v_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": self.alpha,
            "entropy": float(-logp.detach().mean()),
        }
