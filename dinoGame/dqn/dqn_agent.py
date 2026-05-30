"""
Double-DQN with dueling head, soft target update, and action masking applied
in three places (action selection, target argmax, target gather).

Three-place masking matters: if you mask actions only during exploration but
leave the Bellman target maxing over ALL actions, the target network
overestimates the value of airborne states by routing through the (invalid)
jump action. Symptom: TD-error stays high forever, Q-values diverge.

Algorithm:
    Y_t = R_t^(k) + γ^k · (1 - terminal) · Q_target(s_{t+k}, argmax_a Q_online(s_{t+k}, a))
                                                                    ↑ masked       ↑ masked
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .q_net import DuelingQNet


def _masked_q(net: nn.Module, obs_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Return Q(s, ·) with invalid actions set to -inf so any subsequent
    argmax/gather will avoid them. mask_t True = valid."""
    q = net(obs_t)
    # Use a very-negative finite number rather than -inf so that the loss is
    # still finite if a downstream consumer accidentally gathers a masked
    # action — argmax behaviour is unchanged.
    return q.masked_fill(~mask_t, -1e9)


class DQNAgent:
    def __init__(self,
                 obs_dim: int,
                 n_actions: int,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 hidden: int = 256,
                 grad_clip: float = 10.0,
                 device: str = None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma     = gamma
        self.tau       = tau
        self.grad_clip = grad_clip
        self.n_actions = n_actions

        self.online = DuelingQNet(obs_dim, n_actions, hidden=hidden).to(self.device)
        self.target = DuelingQNet(obs_dim, n_actions, hidden=hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        self.optim = torch.optim.Adam(self.online.parameters(), lr=lr)

    # ------------------------------------------------------------------ acting

    @torch.no_grad()
    def act_batch(self,
                  obs_np: np.ndarray,
                  mask_np: np.ndarray,
                  epsilon: float) -> np.ndarray:
        """obs_np: (N, obs_dim).  mask_np: (N, n_actions) bool, True=valid.
        Returns (N,) int64 actions. Random choice (over valid actions only)
        with probability epsilon, otherwise masked argmax."""
        N = obs_np.shape[0]
        actions = np.zeros(N, dtype=np.int64)

        obs_t  = torch.from_numpy(obs_np).float().to(self.device)
        mask_t = torch.from_numpy(mask_np).to(self.device)
        q = _masked_q(self.online, obs_t, mask_t).cpu().numpy()
        greedy = np.argmax(q, axis=1)

        explore = np.random.rand(N) < epsilon
        for i in range(N):
            if explore[i]:
                valid = np.flatnonzero(mask_np[i])
                actions[i] = int(np.random.choice(valid))
            else:
                actions[i] = int(greedy[i])
        return actions

    @torch.no_grad()
    def act_greedy(self, obs_np: np.ndarray, mask_np: np.ndarray) -> int:
        """Single-env greedy action with masking. Used by infer + eval."""
        obs_t  = torch.from_numpy(obs_np).float().unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
        q = _masked_q(self.online, obs_t, mask_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------ learning

    def update(self, batch: Tuple[np.ndarray, ...]) -> dict:
        (obs_np, actions_np, rewards_np, next_obs_np,
         terminals_np, gamma_pow_np, next_mask_np) = batch

        device = self.device
        obs_t       = torch.from_numpy(obs_np).float().to(device)
        actions_t   = torch.from_numpy(actions_np).long().to(device)
        rewards_t   = torch.from_numpy(rewards_np).float().to(device)
        next_obs_t  = torch.from_numpy(next_obs_np).float().to(device)
        terminals_t = torch.from_numpy(terminals_np.astype(np.float32)).to(device)
        gamma_pow_t = torch.from_numpy(gamma_pow_np).float().to(device)
        next_mask_t = torch.from_numpy(next_mask_np).to(device)

        # Current Q(s, a).
        q_current = self.online(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double-DQN target with masking in BOTH argmax and gather.
        with torch.no_grad():
            q_online_next = _masked_q(self.online, next_obs_t, next_mask_t)
            next_actions  = q_online_next.argmax(dim=1)                              # (B,)

            q_target_next = _masked_q(self.target, next_obs_t, next_mask_t)
            q_next_val    = q_target_next.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target = rewards_t + gamma_pow_t * (1.0 - terminals_t) * q_next_val

        loss = F.smooth_l1_loss(q_current, target)

        self.optim.zero_grad()
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optim.step()

        # Soft target update.
        with torch.no_grad():
            for tp, op in zip(self.target.parameters(), self.online.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(op.data, alpha=self.tau)

        return {
            "loss":      float(loss.item()),
            "q_mean":    float(q_current.mean().item()),
            "target_mean": float(target.mean().item()),
            "grad_norm": float(gnorm),
        }

    # ------------------------------------------------------------------ I/O

    def save(self, path: str):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optim":  self.optim.state_dict(),
        }, path)

    def load(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        if load_optim and "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
