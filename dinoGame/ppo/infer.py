"""
Run a trained PPO policy in the live Chrome window so you can watch it play.

Defaults are tuned for *viewing*, not benchmarking:
  - loads checkpoints/best.pt automatically
  - deterministic argmax actions (no exploration noise)
  - small step-pause so the action stream is easier to follow by eye
  - prints per-step decision the first few episodes for sanity-checking

Examples:
    python infer.py                                  # watch the best checkpoint
    python infer.py --ckpt checkpoints/latest.pt     # watch most recent
    python infer.py --ckpt checkpoints/bc_init.pt    # watch the BC-warmup policy
    python infer.py --sample --episodes 20           # benchmark with sampling
"""

import argparse
import os
import time

import numpy as np
import torch

from dino_env import DinoEnv, OBS_DIM, N_ACTIONS, FEATURES_PER_FRAME
from ppo_agent import PPO


ACTION_NAMES = ["noop", "jump", "duck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/best.pt",
                   help="Path to checkpoint. Defaults to checkpoints/best.pt.")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--step-pause", type=float, default=0.015,
                   help="Extra sleep per step for human viewing. 0 for max speed.")
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    p.add_argument("--sample", action="store_true",
                   help="Sample from policy instead of argmax (adds exploration noise).")
    p.add_argument("--trace", type=int, default=1,
                   help="Print per-step decision for the first N episodes (0 = silent).")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        # Helpful fallback: if best.pt doesn't exist yet, try latest.pt or bc_init.pt.
        for alt in ("checkpoints/latest.pt", "checkpoints/bc_init.pt"):
            if os.path.exists(alt):
                print(f"[infer] {args.ckpt} not found — falling back to {alt}")
                args.ckpt = alt
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {args.ckpt} (and no fallback exists). "
                f"Train first with: python train.py --n-envs 4 --rollout 256"
            )

    env = DinoEnv(chromedriver_path=args.chromedriver,
                  step_pause=args.step_pause,
                  game_url=args.game_url)
    ppo = PPO(OBS_DIM, N_ACTIONS)
    ppo.load(args.ckpt, load_optim=False)
    ppo.net.eval()
    print(f"[infer] loaded {args.ckpt}  |  mode={'sample' if args.sample else 'argmax'}  "
          f"|  step-pause={args.step_pause:.3f}s")

    scores, returns = [], []
    best_so_far = -1
    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            action_counts = np.zeros(3, dtype=np.int64)
            t0 = time.time()

            while not done:
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(ppo.device)
                with torch.no_grad():
                    logits, value = ppo.net(obs_t)
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                    if args.sample:
                        action = int(np.random.choice(N_ACTIONS, p=probs))
                    else:
                        action = int(np.argmax(probs))

                # Trace: print key decision points for the first few episodes.
                if ep <= args.trace and (steps < 5 or action == 1 or action == 2):
                    f = obs[-FEATURES_PER_FRAME:]
                    print(f"  ep{ep} step{steps:4d} | dx_o1={f[4]:.2f} "
                          f"jumping={f[1]:.0f} speed={f[3]:.2f} | "
                          f"p={probs.round(2).tolist()} → {ACTION_NAMES[action]} "
                          f"V={value.item():+.2f}")

                action_counts[action] += 1
                obs, r, done, info = env.step(action)
                ep_ret += r
                steps += 1

            score = info.get("score", 0)
            scores.append(score)
            returns.append(ep_ret)
            best_so_far = max(best_so_far, score)
            frac = action_counts / max(1, action_counts.sum())
            print(f"ep {ep:3d} | score {score:5d} (best {best_so_far:5d}) | "
                  f"return {ep_ret:7.2f} | steps {steps:5d} | "
                  f"noop/jump/duck {frac.round(2).tolist()} | "
                  f"{time.time()-t0:5.1f}s")

    except KeyboardInterrupt:
        print("\n[interrupt] stopping")
    finally:
        if scores:
            print(f"\n=== {len(scores)} episodes ===")
            print(f"  score   mean={np.mean(scores):6.1f}  max={np.max(scores):5d}  "
                  f"min={np.min(scores):5d}  median={int(np.median(scores)):5d}")
            print(f"  return  mean={np.mean(returns):+7.2f}  max={np.max(returns):+7.2f}")
        env.close()


if __name__ == "__main__":
    main()
