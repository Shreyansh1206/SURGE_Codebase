"""
Behavior-cloning warmup for the Dino pathway.

Starts from a checkpoint whose MiniGrid pathway + shared core are already good
(the anti-stall curriculum), FREEZES the shared core and every MiniGrid/critic
parameter, and trains only the Dino encoder + Dino actor to imitate the scripted
expert's actions. Because the core is frozen, MiniGrid is left bit-for-bit intact
while Dino gets a strong policy prior. The saved checkpoint therefore contains a
ready combined init: MiniGrid (curriculum) + core (curriculum) + Dino (BC).

Run RL fine-tuning afterwards (train_parallel.py --resume <this checkpoint>).

Usage:
    python pretrain_dino_bc.py --resume checkpoints_curriculum_v2/latest.pt \
        --demos demos/dino_demos.npz --out checkpoints_dino_bc/bc_init.pt
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from envs.dino_gym import DINO_N_ACTIONS, DINO_OBS_DIM
from envs.minigrid_env import minigrid_obs_dim
from multi_task_ppo import TASK_DINO, MultiTaskPPO

ACTION_NAMES = {0: "run", 1: "jump", 2: "duck"}


def eval_dino_greedy(ppo, n_eps=15, frames_per_step=4):
    """Greedy rollout score — sanity check that BC produced a useful policy."""
    import dino_env  # noqa: F401
    from dino_env import DinoEnv

    ppo.net.eval()
    scores = []
    for ep in range(n_eps):
        env = DinoEnv(render=False, frames_per_step=frames_per_step, seed=10_000 + ep)
        obs = env.reset()
        done = False
        last = 0
        guard = 0
        while not done and guard < 20000:
            with torch.no_grad():
                logits, _ = ppo.net(
                    torch.from_numpy(obs).float().unsqueeze(0).to(ppo.device), TASK_DINO
                )
            obs, _, done, info = env.step(int(logits.argmax(-1)))
            last = info.get("score", last)
            guard += 1
        scores.append(last)
        env.close()
    return float(np.mean(scores)), int(np.max(scores))


def main():
    p = argparse.ArgumentParser(description="BC warmup for the Dino pathway")
    p.add_argument("--resume", type=str, default="checkpoints_curriculum_v2/latest.pt")
    p.add_argument("--demos", type=str, default="demos/dino_demos.npz")
    p.add_argument("--out", type=str, default="checkpoints_dino_bc/bc_init.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    data = np.load(args.demos)
    obs = torch.from_numpy(data["obs"]).float()
    act = torch.from_numpy(data["actions"]).long()
    n = len(act)
    print(f"[bc] {n} demos | expert mean score {float(data.get('mean_score', 0)):.0f}")

    # Class-weighted loss — run is ~95% of frames, so unweighted CE just predicts run.
    counts = np.bincount(act.numpy(), minlength=DINO_N_ACTIONS).astype(np.float64)
    weights = counts.sum() / (DINO_N_ACTIONS * np.maximum(counts, 1))
    w = torch.tensor(weights, dtype=torch.float32)
    print(f"[bc] class counts {counts.astype(int).tolist()} -> weights {np.round(weights, 2).tolist()}")

    # Train/val split.
    perm = torch.randperm(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    mg_dim = minigrid_obs_dim("MiniGrid-DoorKey-16x16-v0")
    ppo = MultiTaskPPO(minigrid_dim=mg_dim, dino_dim=DINO_OBS_DIM, dino_actions=DINO_N_ACTIONS)
    ppo.load(args.resume, load_optim=False)
    print(f"[bc] resumed {args.resume}")

    net = ppo.net
    w = w.to(ppo.device)

    # Freeze everything, then unfreeze only the Dino encoder + actor.
    for pm in net.parameters():
        pm.requires_grad_(False)
    trainable = list(net.dino_encoder.parameters()) + list(net.dino_actor.parameters())
    for pm in trainable:
        pm.requires_grad_(True)
    n_train_p = sum(p.numel() for p in trainable)
    print(f"[bc] trainable params (dino_encoder+dino_actor): {n_train_p} "
          f"| frozen: core+minigrid+dino_critic")

    opt = torch.optim.Adam(trainable, lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    base_mean, base_max = eval_dino_greedy(ppo)
    print(f"[bc] pre-BC greedy dino: mean {base_mean:.0f} max {base_max}")

    def run_eval():
        net.eval()
        with torch.no_grad():
            xb = obs[val_idx].to(ppo.device)
            yb = act[val_idx].to(ppo.device)
            logits, _ = net(xb, TASK_DINO)
            pred = logits.argmax(-1)
            acc = (pred == yb).float().mean().item()
            per = {}
            for c in range(DINO_N_ACTIONS):
                m = yb == c
                per[ACTION_NAMES[c]] = (pred[m] == c).float().mean().item() if m.any() else float("nan")
        return acc, per

    import copy

    bs = args.batch_size
    best_score = -1.0
    best_state = None
    for ep in range(1, args.epochs + 1):
        net.train()
        ep_idx = tr_idx[torch.randperm(len(tr_idx))]
        tot = 0.0
        nb = 0
        for i in range(0, len(ep_idx), bs):
            b = ep_idx[i:i + bs]
            xb = obs[b].to(ppo.device)
            yb = act[b].to(ppo.device)
            logits, _ = net(xb, TASK_DINO)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        acc, per = run_eval()
        # Select on actual game score, not classification accuracy — high action
        # accuracy doesn't guarantee good play, and late epochs can diverge.
        gmean, gmax = eval_dino_greedy(ppo, n_eps=10)
        per_s = " ".join(f"{k}={v:.2f}" for k, v in per.items())
        flag = ""
        if gmean > best_score:
            best_score = gmean
            best_state = copy.deepcopy(net.state_dict())
            flag = " *best*"
        print(f"[bc] epoch {ep:2d} | loss {tot/nb:.3f} | val acc {acc:.3f} | "
              f"recall[{per_s}] | greedy {gmean:.0f}/{gmax}{flag}")

    if best_state is not None:
        net.load_state_dict(best_state)
    post_mean, post_max = eval_dino_greedy(ppo, n_eps=20)
    print(f"[bc] best-epoch greedy dino: mean {post_mean:.0f} max {post_max}  "
          f"(was {base_mean:.0f}, selected best={best_score:.0f})")

    ppo.save(args.out)
    print(f"[bc] saved combined checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
