"""
Dueling Q-network. Separates V(s) from A(s, a) to help the agent learn that
most frames are value-neutral and that only a few are decisive — which is
exactly the problem we have on Chrome Dino.

Architecture:
    obs --(MLP trunk)--> h --+-- V_head ---> V(s)        (scalar)
                             +-- A_head ---> A(s, ·)     (n_actions)
    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
"""

import torch
import torch.nn as nn


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.v_head = nn.Linear(hidden, 1)
        self.a_head = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.v_head(h)                                 # (B, 1)
        a = self.a_head(h)                                 # (B, n_actions)
        # Subtract the mean so V and A are identifiable.
        return v + a - a.mean(dim=-1, keepdim=True)        # (B, n_actions)
