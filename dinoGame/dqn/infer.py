"""
Watch a trained DQN policy play live.

Defaults:
  - loads dqn/checkpoints/best.pt (falls back to latest.pt)
  - greedy with masking (no exploration)
  - small step-pause so action stream is easier to follow by eye
  - trace mode prints key decision points for the first N episodes

Examples:
    python -m dqn.infer
    python -m dqn.infer --ckpt dqn/checkpoints/latest.pt
    python -m dqn.infer --episodes 20 --step-pause 0
"""

import argparse
import os
import time

import numpy as np
import torch

from .dino_env  import DinoEnv, OBS_DIM, N_ACTIONS, FEATURES_PER_FRAME
from .dqn_agent import DQNAgent, _masked_q


ACTION_NAMES = ["noop", "jump", "duck"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="dqn/checkpoints/best.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--step-pause", type=float, default=0.015)
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    p.add_argument("--trace", type=int, default=1,
                   help="Print per-step decision for the first N episodes (0 = silent).")
    p.add_argument("--epsilon", type=float, default=0.0,
                   help="Optional exploration noise during inference (default greedy).")
    args = p.parse_args()

    if not os.path.exists(args.ckpt):
        alt = "dqn/checkpoints/latest.pt"
        if os.path.exists(alt):
            print(f"[infer] {args.ckpt} not found — falling back to {alt}")
            args.ckpt = alt
        else:
            raise FileNotFoundError(
                f"No checkpoint at {args.ckpt}. Train first with: "
                f"python -m dqn.train"
            )

    env = DinoEnv(chromedriver_path=args.chromedriver,
                  step_pause=args.step_pause,
                  game_url=args.game_url)
    agent = DQNAgent(OBS_DIM, N_ACTIONS)
    agent.load(args.ckpt, load_optim=False)
    agent.online.eval()
    print(f"[infer] loaded {args.ckpt}  |  epsilon={args.epsilon:.2f}  |  "
          f"step-pause={args.step_pause:.3f}s")

    scores, returns = [], []
    best_so_far = -1
    try:
        for ep in range(1, args.episodes + 1):
            obs = env.reset()
            done = False
            ep_ret = 0.0
            steps = 0
            action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
            t0 = time.time()

            while not done:
                mask = env.action_mask()
                # Compute Q for tracing even when greedy.
                obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q = _masked_q(agent.online, obs_t, mask_t)[0].cpu().numpy()
                if args.epsilon > 0 and np.random.rand() < args.epsilon:
                    valid = np.flatnonzero(mask)
                    action = int(np.random.choice(valid))
                else:
                    action = int(np.argmax(q))

                if ep <= args.trace and (steps < 5 or action != 0):
                    f = obs[-FEATURES_PER_FRAME:]
                    # Feature layout: 0 dinoY, 1 jumping, 2 ducking, 3 speed,
                    # 4 dx_o1, 5 y_o1, 6 w_o1, 7 h_o1, 8 isbird_o1, ...
                    print(f"  ep{ep} step{steps:4d} | dx_o1={f[4]:.2f} "
                          f"y_o1={f[5]:.2f} bird={f[8]:.0f} | "
                          f"jumping={f[1]:.0f} speed={f[3]:.2f} | "
                          f"Q={q.round(2).tolist()} mask={mask.tolist()} "
                          f"→ {ACTION_NAMES[action]}")

                action_counts[action] += 1
                obs, r, done, info = env.step(action)
                ep_ret += r
                steps  += 1

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
