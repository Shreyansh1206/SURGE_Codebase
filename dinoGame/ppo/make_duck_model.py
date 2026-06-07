"""Build the ducking model: head-only supervised training on cached states.

Only the DUCK row of the policy head is trained; noop/jump logits stay identical
to the base policy on every state, so cactus jump timing is preserved.

Suppression is FOCUSED on close low obstacles (cacti), not diluted across empty
gaps — that was the fix for duck bleeding onto cacti (~6% deadly leak).

Typical use (from dinoGame/ppo/, dinoGame conda env):
    python make_duck_model.py --reuse-data   # fast: uses _duckdata.npz
    python make_duck_model.py                # collect + train
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from ppo_agent import PPO
from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
from finetune_duck import collect_states, BIRD_Y_NORM_MAX, GUIDE_DX_MAX, DUCK_IDX, NOOP_IDX, JUMP_IDX

FEATURES_PER_FRAME = 12


def cactus_mask(obs_np: np.ndarray) -> np.ndarray:
    """Close low obstacles (cacti / low pterodactyl) — states where duck is deadly."""
    f = obs_np[:, -FEATURES_PER_FRAME:]
    return (f[:, 5] > BIRD_Y_NORM_MAX) & (f[:, 4] < GUIDE_DX_MAX)


def train_head_focused(ppo, bird, other, *, target, cap, lr, iters, cactus_weight, epochs):
    """Head-only: raise duck on bird states; push duck down on close-low obstacles."""
    ph = ppo.net.policy_head
    opt = torch.optim.Adam([ph.weight, ph.bias], lr=lr)
    log_tgt = float(np.log(target))
    log_cap = float(np.log(cap))

    cact = other[cactus_mask(other)]
    empty = other[~cactus_mask(other)]
    print(f"[train] bird={len(bird)} cactus(close-low)={len(cact)} empty/far={len(empty)}")

    B = torch.tensor(bird)
    for it in range(1, iters + 1):
        nb = len(bird)
        nc = min(len(cact), max(1, int(nb * cactus_weight)))
        ne = min(len(empty), nb)
        cs = torch.tensor(cact[np.random.choice(len(cact), nc, replace=False)])
        es = torch.tensor(empty[np.random.choice(len(empty), ne, replace=False)])

        lb = ppo.net(B)[0]
        lpb = F.log_softmax(lb, -1)[:, DUCK_IDX]
        lpc = F.log_softmax(ppo.net(cs)[0], -1)[:, DUCK_IDX]
        lpe = F.log_softmax(ppo.net(es)[0], -1)[:, DUCK_IDX]

        loss = (2.0 * torch.clamp(log_tgt - lpb, min=0.0).mean()
                + 3.0 * torch.clamp(lpc - log_cap, min=0.0).mean()
                + 1.0 * torch.clamp(lpe - log_cap, min=0.0).mean())

        opt.zero_grad()
        loss.backward()
        if ph.weight.grad is not None:
            ph.weight.grad[NOOP_IDX].zero_()
            ph.weight.grad[JUMP_IDX].zero_()
        if ph.bias.grad is not None:
            ph.bias.grad[NOOP_IDX].zero_()
            ph.bias.grad[JUMP_IDX].zero_()
        opt.step()

        if it % 50 == 0 or it == iters:
            with torch.no_grad():
                barg = torch.argmax(lb, -1).numpy()
                carg = torch.argmax(ppo.net(cs)[0], -1).numpy()
                O = torch.tensor(other)
                oarg = torch.argmax(ppo.net(O)[0], -1).numpy()
            of = other[:, -FEATURES_PER_FRAME:]
            danger = np.mean(
                (oarg == DUCK_IDX) & (of[:, 4] < 0.35) & (of[:, 5] > BIRD_Y_NORM_MAX)
            ) * 100
            print(f"  it {it:4d} loss {loss.item():.3f} | "
                  f"bird duck-arg {np.mean(barg == DUCK_IDX) * 100:5.1f}% | "
                  f"cactus duck-arg {np.mean(carg == DUCK_IDX) * 100:4.1f}% | "
                  f"DANGEROUS leak {danger:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints_duck/duck_init.pt")
    p.add_argument("--out", default="checkpoints_duck_v13/best_duck.pt")
    p.add_argument("--data", default="_duckdata.npz")
    p.add_argument("--reuse-data", action="store_true")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--collect-steps", type=int, default=2000)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--target", type=float, default=0.80)
    p.add_argument("--cap", type=float, default=0.02)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--cactus-weight", type=float, default=4.0,
                   help="Cactus samples per bird sample in each step.")
    p.add_argument("--epochs", type=int, default=1, help="Unused (kept for CLI compat).")
    args = p.parse_args()

    ppo = PPO(OBS_DIM, N_ACTIONS)
    ppo.load(args.init, load_optim=False)

    vec = None
    try:
        if args.reuse_data and os.path.exists(args.data):
            d = np.load(args.data)
            bird = d["bird"].astype(np.float32)
            other = d["other"].astype(np.float32)
            print(f"[data] reused {args.data}: bird={len(bird)} other={len(other)}")
        else:
            vec = VecDinoEnv(
                n_envs=args.n_envs, window_size=(700, 320), grid_cols=2,
                headless=True, curriculum_prob=0.9,
                start_speed_range=(8.5, 11.0), start_score=450.0,
                duck_shaping=False,
            )
            birds, others = [], []
            for i in range(args.passes):
                b, o = collect_states(vec, ppo, ppo.device, args.collect_steps, guide_prob=0.9)
                birds.append(b)
                others.append(o)
                print(f"[collect] pass {i + 1}/{args.passes}: bird={len(b)} other={len(o)}")
            bird = np.concatenate(birds)
            other = np.concatenate(others)
            np.savez(args.data, bird=bird, other=other)
            print(f"[collect] TOTAL bird={len(bird)} other={len(other)} -> {args.data}")

        train_head_focused(
            ppo, bird, other,
            target=args.target, cap=args.cap, lr=args.lr,
            iters=args.iters, cactus_weight=args.cactus_weight, epochs=args.epochs,
        )

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        ppo.save(args.out)
        print(f"[done] saved -> {args.out} (checkpoints/best.pt untouched)")
    finally:
        if vec is not None:
            vec.close()
