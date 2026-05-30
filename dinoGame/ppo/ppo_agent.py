"""
Actor-critic network and PPO update for vectorised Dino training.

The RolloutBuffer stores transitions from N parallel envs at each of T steps —
internal shape is (T, N, ...). GAE is computed per-env (each env's done flag
breaks bootstrapping correctly), then the buffer is flattened to (T*N, ...)
for the SGD update.
"""

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, device):
        """obs_np: (N, obs_dim). Returns (actions[N], logp[N], values[N]) as np arrays."""
        obs = torch.from_numpy(obs_np).float().to(device)
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (action.cpu().numpy().astype(np.int64),
                dist.log_prob(action).cpu().numpy().astype(np.float32),
                value.cpu().numpy().astype(np.float32))

    def evaluate(self, obs, actions):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RolloutBuffer:
    """Stores T steps × N envs of transitions."""

    def __init__(self, n_envs: int):
        self.n_envs   = n_envs
        self.obs      : List[np.ndarray] = []   # each (N, obs_dim)
        self.actions  : List[np.ndarray] = []   # each (N,)
        self.logprobs : List[np.ndarray] = []   # each (N,)
        self.rewards  : List[np.ndarray] = []   # each (N,)
        self.values   : List[np.ndarray] = []   # each (N,)
        self.dones    : List[np.ndarray] = []   # each (N,) bool

    def add(self, obs, actions, logps, rewards, values, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.logprobs.append(logps)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def __len__(self):
        return len(self.obs)

    def clear(self):
        for attr in ("obs", "actions", "logprobs", "rewards", "values", "dones"):
            getattr(self, attr).clear()

    def stack(self):
        """Returns (T, N, ...) arrays."""
        return (np.stack(self.obs),      np.stack(self.actions),
                np.stack(self.logprobs), np.stack(self.rewards),
                np.stack(self.values),   np.stack(self.dones))


def compute_gae_vec(rewards, values, dones, last_values,
                    gamma: float = 0.99, lam: float = 0.95):
    """
    rewards    : (T, N) float
    values     : (T, N) float
    dones      : (T, N) bool
    last_values: (N,)   float (V(s_T) for bootstrapping)
    Returns (advantages, returns), each (T, N).
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        next_nonterm = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_v * next_nonterm - values[t]
        gae = delta + gamma * lam * next_nonterm * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class PPO:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 lr: float = 3e-4,
                 clip_eps: float = 0.2,
                 epochs: int = 4,
                 batch_size: int = 64,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps      = clip_eps
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update(self, buf: RolloutBuffer, last_values: np.ndarray,
               gamma: float = 0.99, lam: float = 0.95):
        obs, actions, old_logp, rewards, values, dones = buf.stack()  # (T,N,...)
        adv, returns = compute_gae_vec(rewards, values, dones, last_values,
                                       gamma=gamma, lam=lam)

        # Flatten T,N → T*N for the SGD update.
        T, N = rewards.shape
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        obs_f      = flat(obs)
        actions_f  = flat(actions)
        old_logp_f = flat(old_logp)
        adv_f      = flat(adv)
        ret_f      = flat(returns)

        # Normalise advantages. NOTE: do NOT normalise returns here — the value
        # head would learn a per-batch-normalised scale that doesn't match the
        # raw `values` used by compute_gae_vec on the next rollout, silently
        # corrupting GAE. If you want return scaling, do it with a *running*
        # mean/std (à la SB3 VecNormalize) and un-normalise before GAE.
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        device = self.device
        obs_t      = torch.tensor(obs_f,      dtype=torch.float32, device=device)
        actions_t  = torch.tensor(actions_f,  dtype=torch.long,    device=device)
        old_logp_t = torch.tensor(old_logp_f, dtype=torch.float32, device=device)
        adv_t      = torch.tensor(adv_f,      dtype=torch.float32, device=device)
        ret_t      = torch.tensor(ret_f,      dtype=torch.float32, device=device)

        M  = T * N
        idx = np.arange(M)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0,
                 "kl": 0.0, "grad_norm": 0.0}
        n_updates = 0

        target_kl = 0.015
        early_stop = False

        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, M, self.batch_size):
                b   = idx[start:start + self.batch_size]
                b_t = torch.as_tensor(b, dtype=torch.long, device=device)

                new_logp, entropy, value = self.net.evaluate(obs_t[b_t], actions_t[b_t])
                log_ratio = new_logp - old_logp_t[b_t]
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                if approx_kl > target_kl:
                    print(f"[PPO] early stop at epoch {epoch}: approx_kl={approx_kl:.4f} > {target_kl}")
                    early_stop = True
                    break

                ratio = torch.exp(log_ratio)
                s1 = ratio * adv_t[b_t]
                s2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[b_t]
                pi_loss = -torch.min(s1, s2).mean()
                v_loss  = F.mse_loss(value, ret_t[b_t])
                ent     = entropy.mean()
                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent

                self.optim.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    stats["pi_loss"]   += pi_loss.item()
                    stats["v_loss"]    += v_loss.item()
                    stats["entropy"]   += ent.item()
                    stats["kl"]        += (old_logp_t[b_t] - new_logp).mean().item()
                    stats["grad_norm"] += float(gnorm)
                n_updates += 1
            if early_stop:
                break

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats

    def save(self, path: str):
        torch.save({"net": self.net.state_dict(),
                    "optim": self.optim.state_dict()}, path)

    def load(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        if load_optim and "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
