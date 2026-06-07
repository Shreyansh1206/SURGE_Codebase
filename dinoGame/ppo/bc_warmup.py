"""
Behavior-cloning warmup with a duck-aware scripted teacher.

Collects (obs, action) trajectories where the teacher:
  - ducks under mid-height pterodactyls,
  - no-ops under high birds (run through),
  - jumps over cacti and low birds.

Optionally starts some episodes in bird territory (speed curriculum) so the
dataset includes many bird encounters.

Saves to checkpoints_scratch_duck/ by default — does not touch checkpoints/best.pt
or prior finetune artifacts.

Typical pipeline (dinoGame conda env, from dinoGame/ppo/):
    python bc_warmup.py --headless --episodes 40 --curriculum-prob 0.5
    python train.py --save-dir checkpoints_scratch_duck --resume checkpoints_scratch_duck/bc_init.pt \\
        --headless --curriculum-prob 0.35 --updates 300
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from dino_env import DinoEnv, OBS_DIM, N_ACTIONS
from ppo_agent import PPO
from scripted_teacher import scripted_action, new_memory


def collect(env, n_episodes: int, gamma: float = 0.99):
    """Run scripted policy; return (obs, actions, returns, scores)."""
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    scores = []
    for ep in range(n_episodes):
        obs = env.reset()
        mem = new_memory()
        last_info = {}
        done = False
        while not done:
            a = scripted_action(obs, info=last_info if last_info else None, memory=mem)
            obs_buf.append(obs.copy())
            act_buf.append(a)
            obs, r, done, info = env.step(a)
            last_info = info
            rew_buf.append(r)
            done_buf.append(done)
        scores.append(info.get("score", 0))
        print(f"  bc ep {ep + 1}/{n_episodes}: score={scores[-1]}")

    returns = np.zeros(len(rew_buf), dtype=np.float32)
    R = 0.0
    for t in reversed(range(len(rew_buf))):
        if done_buf[t]:
            R = 0.0
        R = rew_buf[t] + gamma * R
        returns[t] = R

    return (
        np.stack(obs_buf).astype(np.float32),
        np.array(act_buf, dtype=np.int64),
        returns,
        scores,
    )


def main():
    p = argparse.ArgumentParser(
        description="BC warmup with duck-aware scripted teacher.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", type=int, default=40,
                   help="Scripted episodes to collect.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", type=str, default="checkpoints_scratch_duck/bc_init.pt")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    # Bird-territory curriculum during data collection
    p.add_argument("--curriculum-prob", type=float, default=0.5,
                   help="Fraction of episodes that reset into bird territory "
                        "(speed >= 8.5, score ~450).")
    p.add_argument("--start-speed-min", type=float, default=8.5)
    p.add_argument("--start-speed-max", type=float, default=9.5)
    p.add_argument("--start-score", type=float, default=450.0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    print(f"[bc] collecting {args.episodes} episodes | curriculum_prob={args.curriculum_prob} "
          f"speed=({args.start_speed_min}, {args.start_speed_max}) score={args.start_score}")
    env = DinoEnv(
        headless=args.headless,
        chromedriver_path=args.chromedriver,
        game_url=args.game_url,
        curriculum_prob=args.curriculum_prob,
        start_speed_range=(args.start_speed_min, args.start_speed_max),
        start_score=args.start_score,
    )
    obs_np, act_np, ret_np, scores = collect(env, args.episodes)
    env.close()

    n = len(obs_np)
    n_noop = int((act_np == 0).sum())
    n_jump = int((act_np == 1).sum())
    n_duck = int((act_np == 2).sum())
    print(f"[bc] collected {n} samples | scripted score mean={np.mean(scores):.1f} "
          f"max={max(scores)}")
    print(f"[bc] action mix: noop={n_noop} ({100*n_noop/n:.1f}%) "
          f"jump={n_jump} ({100*n_jump/n:.1f}%) "
          f"duck={n_duck} ({100*n_duck/n:.1f}%)")
    if n_duck == 0:
        print("[bc] WARNING: zero duck actions in dataset — raise --curriculum-prob "
              "or --episodes.")

    ppo = PPO(OBS_DIM, N_ACTIONS, lr=args.lr)
    dev = ppo.device
    obs_t = torch.tensor(obs_np, device=dev)
    act_t = torch.tensor(act_np, device=dev)
    ret_t = torch.tensor(ret_np, device=dev)
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    idx = np.arange(n)
    print(f"[bc] training {args.epochs} epochs over {n} samples ...")
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        tot_pi, tot_v, n_batches = 0.0, 0.0, 0
        for start in range(0, n, args.batch_size):
            b = torch.as_tensor(idx[start:start + args.batch_size],
                                dtype=torch.long, device=dev)
            logits, value = ppo.net(obs_t[b])
            pi_loss = F.cross_entropy(logits, act_t[b])
            v_loss = F.mse_loss(value, ret_t[b])
            loss = pi_loss + 0.5 * v_loss
            ppo.optim.zero_grad()
            loss.backward()
            ppo.optim.step()
            tot_pi += pi_loss.item()
            tot_v += v_loss.item()
            n_batches += 1

        with torch.no_grad():
            acc = (ppo.net(obs_t)[0].argmax(-1) == act_t).float().mean().item()
        print(f"  epoch {epoch + 1:2d}: pi_loss={tot_pi / n_batches:.3f} "
              f"v_loss={tot_v / n_batches:.3f}  acc={acc:.3f}")

    ppo.save(args.save)
    save_dir = os.path.dirname(args.save)
    print(f"[bc] saved {args.save}")
    print("[bc] next:")
    print(f"  python train.py --save-dir {save_dir} --resume {args.save} "
          f"--headless --curriculum-prob {args.curriculum_prob} --updates 300")


if __name__ == "__main__":
    main()
