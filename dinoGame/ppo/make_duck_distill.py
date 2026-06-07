"""
Inject duck behaviour into an existing strong model while preserving jump/noop.

Approach:
1) Collect bird/other states from the target model in bird territory.
2) Train with:
   - hinge raise on bird states: log P(duck) >= log(target_duck)
   - hinge cap on other states: log P(duck) <= log(duck_cap)
   - distillation anchor on jump-noop logit gap vs frozen base net
"""
import argparse
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F

from ppo_agent import PPO
from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
from finetune_duck import collect_states, DUCK_IDX

NOOP_IDX, JUMP_IDX = 0, 1


def train_distill(ppo, bird, other, *, iters, lr, target_duck, duck_cap, distill_w):
    ref = copy.deepcopy(ppo.net)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(ppo.net.parameters(), lr=lr)

    B = torch.tensor(bird, dtype=torch.float32, device=ppo.device)
    O_all = torch.tensor(other, dtype=torch.float32, device=ppo.device)
    ltgt = float(np.log(target_duck))
    lcap = float(np.log(duck_cap))

    for it in range(1, iters + 1):
        k = min(len(other), len(bird))
        idx = np.random.choice(len(other), k, replace=False)
        O = O_all[idx]

        lb, _ = ppo.net(B)
        lo, _ = ppo.net(O)
        with torch.no_grad():
            rb, _ = ref(B)
            ro, _ = ref(O)

        lpb = F.log_softmax(lb, -1)[:, DUCK_IDX]
        lpo = F.log_softmax(lo, -1)[:, DUCK_IDX]
        duck_loss = 2.0 * (
            torch.clamp(ltgt - lpb, min=0.0).mean()
            + torch.clamp(lpo - lcap, min=0.0).mean()
        )

        # Keep jump/noop decision boundary close to base model.
        gap_b = lb[:, JUMP_IDX] - lb[:, NOOP_IDX]
        gap_o = lo[:, JUMP_IDX] - lo[:, NOOP_IDX]
        ref_gap_b = rb[:, JUMP_IDX] - rb[:, NOOP_IDX]
        ref_gap_o = ro[:, JUMP_IDX] - ro[:, NOOP_IDX]
        distill = F.mse_loss(gap_b, ref_gap_b) + F.mse_loss(gap_o, ref_gap_o)

        loss = duck_loss + distill_w * distill
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % 50 == 0 or it == iters:
            with torch.no_grad():
                barg = torch.argmax(lb, -1).cpu().numpy()
                oarg = torch.argmax(lo, -1).cpu().numpy()
            print(
                f"  it {it:4d} | loss {float(loss.item()):.3f} "
                f"| bird duck-arg {np.mean(barg==DUCK_IDX)*100:5.1f}% "
                f"| other duck-arg {np.mean(oarg==DUCK_IDX)*100:4.1f}%"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="checkpoints_scratch_duck/best.pt")
    p.add_argument("--out", default="checkpoints_scratch_duck/duck_distill.pt")
    p.add_argument("--data", default="_duckdata_scratch.npz")
    p.add_argument("--reuse-data", action="store_true")
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--collect-steps", type=int, default=1800)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--target-duck", type=float, default=0.88)
    p.add_argument("--duck-cap", type=float, default=0.05)
    p.add_argument("--distill-w", type=float, default=5.0)
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
                print(f"[collect] pass {i+1}/{args.passes}: bird={len(b)} other={len(o)}")
            bird = np.concatenate(birds)
            other = np.concatenate(others)
            np.savez(args.data, bird=bird, other=other)
            print(f"[collect] TOTAL bird={len(bird)} other={len(other)} -> {args.data}")

        train_distill(
            ppo, bird, other,
            iters=args.iters, lr=args.lr,
            target_duck=args.target_duck, duck_cap=args.duck_cap,
            distill_w=args.distill_w,
        )
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        ppo.save(args.out)
        print(f"[done] saved {args.out}")
    finally:
        if vec is not None:
            vec.close()


if __name__ == "__main__":
    main()
