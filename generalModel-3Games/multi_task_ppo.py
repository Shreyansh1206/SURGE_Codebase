
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

TASK_MINIGRID = "minigrid"
TASK_DINO = "dino"
TASK_CARRACING = "carracing"
VALID_TASKS = (TASK_MINIGRID, TASK_DINO, TASK_CARRACING)

MINIGRID_VIEW = 7
MINIGRID_CHANNELS = 20

CARRACING_OBS_SHAPE = (4, 96, 96)
# Fallback default; callers pass the real value from envs.carracing_env. The
# discrete set now includes gas+steer combos (see CARRACING_DISCRETE_ACTIONS).
CARRACING_N_ACTIONS = 9


def _ortho(layer: nn.Module, gain: float = np.sqrt(2)) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    return layer


class MinigridCNNEncoder(nn.Module):

    def __init__(
        self,
        out_dim: int,
        view: int = MINIGRID_VIEW,
        channels: int = MINIGRID_CHANNELS,
    ):
        super().__init__()
        self.view = view
        self.channels = channels
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(channels, 16, kernel_size=2)),
            nn.ReLU(),
            _ortho(nn.Conv2d(16, 32, kernel_size=2)),
            nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=2)),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, view, view)
            n_flat = self.conv(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(_ortho(nn.Linear(n_flat, out_dim)), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        grid = x.reshape(b, self.view, self.view, self.channels).permute(0, 3, 1, 2)
        return self.fc(self.conv(grid).reshape(b, -1))


CARRACING_AUX_DIM = 3  # speed, lateral slip, angular velocity


class CarRacingCNNEncoder(nn.Module):

    def __init__(
        self,
        out_dim: int,
        obs_shape: Tuple[int, int, int] = CARRACING_OBS_SHAPE,
        aux_dim: int = CARRACING_AUX_DIM,
    ):
        super().__init__()
        self.obs_shape = tuple(obs_shape)
        self.aux_dim = aux_dim
        self.n_pixels = 1
        for d in self.obs_shape:
            self.n_pixels *= d
        channels, height, width = self.obs_shape
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            n_flat = self.conv(dummy).reshape(1, -1).shape[1]
        self.fc = nn.Sequential(
            _ortho(nn.Linear(n_flat + aux_dim, out_dim)), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        total_flat = x.shape[1] if x.dim() == 2 else x.numel() // b
        if total_flat > self.n_pixels:
            pixels = x[:, :self.n_pixels].reshape((b,) + self.obs_shape)
            aux = x[:, self.n_pixels:]
        else:
            if x.dim() == 2:
                pixels = x.reshape((b,) + self.obs_shape)
            else:
                pixels = x
            aux = torch.zeros(b, self.aux_dim, device=x.device)
        conv_out = self.conv(pixels).reshape(b, -1)
        return self.fc(torch.cat([conv_out, aux], dim=-1))


class MultiTaskActorCritic(nn.Module):

    def __init__(
        self,
        minigrid_dim: int,
        dino_dim: int,
        carracing_obs_shape: Tuple[int, int, int] = CARRACING_OBS_SHAPE,
        minigrid_actions: int = 7,
        dino_actions: int = 3,
        carracing_actions: int = CARRACING_N_ACTIONS,
        shared_dim: int = 128,
        head_hidden: int = 128,
    ):
        super().__init__()
        self.minigrid_dim = minigrid_dim
        self.dino_dim = dino_dim
        self.carracing_obs_shape = tuple(carracing_obs_shape)

        self.minigrid_encoder = MinigridCNNEncoder(out_dim=shared_dim)
        self.dino_encoder = nn.Sequential(
            nn.Linear(dino_dim, shared_dim),
            nn.Tanh(),
        )
        self.carracing_encoder = CarRacingCNNEncoder(
            out_dim=shared_dim, obs_shape=self.carracing_obs_shape
        )

        self.shared_core = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.Tanh(),
            nn.Linear(shared_dim, shared_dim),
            nn.Tanh(),
        )

        self.minigrid_actor = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, minigrid_actions),
        )
        self.dino_actor = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, dino_actions),
        )
        self.carracing_actor = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, carracing_actions),
        )
        self.minigrid_critic = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )
        self.dino_critic = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )
        self.carracing_critic = nn.Sequential(
            nn.Linear(shared_dim, head_hidden),
            nn.Tanh(),
            nn.Linear(head_hidden, 1),
        )

        for actor in (self.minigrid_actor, self.dino_actor, self.carracing_actor):
            _ortho(actor[0])
            _ortho(actor[-1], gain=0.01)
        for critic in (self.minigrid_critic, self.dino_critic, self.carracing_critic):
            _ortho(critic[0])
            _ortho(critic[-1], gain=1.0)
        for core_layer in self.shared_core:
            _ortho(core_layer)
        _ortho(self.dino_encoder[0])

    def _encode(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        if task_name == TASK_MINIGRID:
            return self.minigrid_encoder(x)
        if task_name == TASK_DINO:
            return self.dino_encoder(x)
        if task_name == TASK_CARRACING:
            return self.carracing_encoder(x)
        raise ValueError(f"Invalid task parameter: {task_name}")

    def _heads(self, h: torch.Tensor, task_name: str):
        if task_name == TASK_MINIGRID:
            return self.minigrid_actor(h), self.minigrid_critic(h).squeeze(-1)
        if task_name == TASK_DINO:
            return self.dino_actor(h), self.dino_critic(h).squeeze(-1)
        if task_name == TASK_CARRACING:
            return self.carracing_actor(h), self.carracing_critic(h).squeeze(-1)
        raise ValueError(f"Invalid task parameter: {task_name}")

    def forward(self, x: torch.Tensor, task_name: str):
        h = self.shared_core(self._encode(x, task_name))
        return self._heads(h, task_name)

    @torch.no_grad()
    def act_batch(self, obs_np: np.ndarray, task_name: str, device):
        obs = torch.from_numpy(obs_np).float().to(device)
        logits, value = self.forward(obs, task_name)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return (
            action.cpu().numpy().astype(np.int64),
            dist.log_prob(action).cpu().numpy().astype(np.float32),
            value.cpu().numpy().astype(np.float32),
        )

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor, task_name: str):
        logits, value = self.forward(obs, task_name)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


class RolloutBuffer:
    def __init__(self, n_envs: int):
        self.n_envs = n_envs
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.logprobs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.dones: List[np.ndarray] = []

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
        return (
            np.stack(self.obs),
            np.stack(self.actions),
            np.stack(self.logprobs),
            np.stack(self.rewards),
            np.stack(self.values),
            np.stack(self.dones),
        )


def compute_gae_vec(
    rewards, values, dones, last_values, gamma: float = 0.99, lam: float = 0.95
):
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


class MultiTaskPPO:
    def __init__(
        self,
        minigrid_dim: int,
        dino_dim: int,
        carracing_obs_shape: Tuple[int, int, int] = CARRACING_OBS_SHAPE,
        minigrid_actions: int = 7,
        dino_actions: int = 3,
        carracing_actions: int = CARRACING_N_ACTIONS,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device=None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MultiTaskActorCritic(
            minigrid_dim=minigrid_dim,
            dino_dim=dino_dim,
            carracing_obs_shape=carracing_obs_shape,
            minigrid_actions=minigrid_actions,
            dino_actions=dino_actions,
            carracing_actions=carracing_actions,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def update_task(
        self,
        task_name: str,
        buf: RolloutBuffer,
        last_values: np.ndarray,
        gamma: float = 0.99,
        lam: float = 0.95,
        target_kl: float = 0.015,
        bc_obs: torch.Tensor | None = None,
        bc_actions: torch.Tensor | None = None,
        bc_coef: float = 0.0,
        bc_weight: torch.Tensor | None = None,
    ):
        if task_name not in VALID_TASKS:
            raise ValueError(f"Invalid task parameter: {task_name}")
        if len(buf) == 0:
            return {
                "pi_loss": 0.0,
                "v_loss": 0.0,
                "entropy": 0.0,
                "kl": 0.0,
                "grad_norm": 0.0,
            }

        obs, actions, old_logp, rewards, values, dones = buf.stack()
        adv, returns = compute_gae_vec(
            rewards, values, dones, last_values, gamma=gamma, lam=lam
        )
        T, N = rewards.shape
        flat = lambda x: x.reshape((T * N,) + x.shape[2:])
        obs_f = flat(obs)
        actions_f = flat(actions)
        old_logp_f = flat(old_logp)
        adv_f = flat(adv)
        ret_f = flat(returns)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)

        device = self.device
        obs_t = torch.tensor(obs_f, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions_f, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(old_logp_f, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_f, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_f, dtype=torch.float32, device=device)

        M = T * N
        idx = np.arange(M)
        stats = {
            "pi_loss": 0.0,
            "v_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "grad_norm": 0.0,
            "bc_loss": 0.0,
        }
        use_bc = bc_obs is not None and bc_coef > 0.0
        n_updates = 0
        early_stop = False

        for epoch in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, M, self.batch_size):
                b = idx[start : start + self.batch_size]
                b_t = torch.as_tensor(b, dtype=torch.long, device=device)
                new_logp, entropy, value = self.net.evaluate(
                    obs_t[b_t], actions_t[b_t], task_name
                )
                log_ratio = new_logp - old_logp_t[b_t]
                with torch.no_grad():
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                if approx_kl > target_kl:
                    early_stop = True
                    break

                ratio = torch.exp(log_ratio)
                s1 = ratio * adv_t[b_t]
                s2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * adv_t[b_t]
                )
                pi_loss = -torch.min(s1, s2).mean()
                v_loss = F.mse_loss(value, ret_t[b_t])
                ent = entropy.mean()
                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent

                bc_loss_val = 0.0
                if use_bc:
                    sel = torch.randint(0, bc_obs.shape[0], (b_t.shape[0],), device=device)
                    bc_logits, _ = self.net.forward(bc_obs[sel], task_name)
                    bc_loss = F.cross_entropy(bc_logits, bc_actions[sel], weight=bc_weight)
                    loss = loss + bc_coef * bc_loss
                    bc_loss_val = bc_loss.item()

                self.optim.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.max_grad_norm
                )
                self.optim.step()

                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["kl"] += (old_logp_t[b_t] - new_logp).mean().item()
                    stats["grad_norm"] += float(gnorm)
                    stats["bc_loss"] += bc_loss_val
                n_updates += 1
            if early_stop:
                break

        for k in stats:
            stats[k] /= max(1, n_updates)
        return stats

    def save(self, path: str):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optim": self.optim.state_dict(),
                "minigrid_dim": self.net.minigrid_dim,
                "dino_dim": self.net.dino_dim,
                "carracing_obs_shape": list(self.net.carracing_obs_shape),
            },
            path,
        )

    def load(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        saved = ckpt["net"]
        own = self.net.state_dict()
        # Load only tensors whose shape matches, so we can change a single head
        # (e.g. the CarRacing action head) without discarding everything else.
        compatible = {
            k: v for k, v in saved.items() if k in own and v.shape == own[k].shape
        }
        skipped = [k for k in saved if k not in compatible]
        own.update(compatible)
        self.net.load_state_dict(own)
        if skipped:
            print(f"[load] reinitialized {len(skipped)} mismatched tensors: {skipped}")
            # Optimizer moments would be misaligned with reinitialized params, so
            # start the optimizer fresh in that case.
            load_optim = False
        if load_optim and "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
