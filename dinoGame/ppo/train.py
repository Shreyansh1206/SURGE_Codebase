"""
PPO training loop for Chrome Dino, vectorised over N parallel browser instances.

Usage:
    python train.py
    python train.py --n-envs 4 --updates 300 --rollout 256
    python train.py --resume checkpoints/latest.pt --updates 500
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
from ppo_agent import PPO, RolloutBuffer


def collect_rollout(vec_env, ppo, buf, n_steps, device, last_obs):
    """Step the vec_env n_steps times. Each step adds N transitions to buf.
    Total samples per call = n_steps × n_envs.

    last_obs : (N, obs_dim) — observation array carried over between rollouts.
    Returns  : (last_obs, last_values, episode_returns, episode_lengths, episode_scores).
    """
    N = vec_env.n_envs
    obs = last_obs
    ep_returns, ep_lens, ep_scores = [], [], []
    # Per-env running stats.
    cur_return = np.zeros(N, dtype=np.float32)
    cur_len    = np.zeros(N, dtype=np.int64)

    for _ in range(n_steps):
        actions, logps, values = ppo.net.act_batch(obs, device)
        next_obs, rewards, dones, infos = vec_env.step(actions)

        buf.add(obs, actions, logps, rewards, values, dones)

        cur_return += rewards
        cur_len    += 1
        for i in range(N):
            if dones[i]:
                ep_returns.append(float(cur_return[i]))
                ep_lens.append(int(cur_len[i]))
                ep_scores.append(int(infos[i].get("score", 0)))
                cur_return[i] = 0.0
                cur_len[i]    = 0
        obs = next_obs

    # Bootstrap V(s_T) for each env.
    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)
        _, last_v = ppo.net(obs_t)
        last_values = last_v.cpu().numpy().astype(np.float32)

    return obs, last_values, ep_returns, ep_lens, ep_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=4,
                   help="Number of parallel browser instances.")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--rollout", type=int, default=256,
                   help="Steps per env per update — total samples = rollout * n_envs.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--step-pause", type=float, default=0.0,
                   help="Extra sleep per step (training: 0).")
    p.add_argument("--window-w", type=int, default=700)
    p.add_argument("--window-h", type=int, default=320)
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--headless", action="store_true",
                   help="Run browsers off-screen. Recommended for background training.")
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[train] launching {args.n_envs} parallel Dino envs "
          f"({args.window_w}x{args.window_h}, {args.grid_cols} cols)")
    vec_env = VecDinoEnv(
        n_envs        = args.n_envs,
        window_size   = (args.window_w, args.window_h),
        grid_cols     = args.grid_cols,
        chromedriver_path = args.chromedriver,
        step_pause    = args.step_pause,
        game_url      = args.game_url,
        headless      = args.headless,
    )
    ppo = PPO(OBS_DIM, N_ACTIONS,
              lr=args.lr, clip_eps=args.clip, epochs=args.epochs,
              batch_size=args.batch_size, entropy_coef=args.entropy)
    if args.resume:
        print(f"[resume] loading {args.resume}")
        ppo.load(args.resume)

    buf = RolloutBuffer(args.n_envs)
    log_path  = os.path.join(args.save_dir, "train_log.jsonl")
    best_score = -1
    t0 = time.time()

    try:
        obs = vec_env.reset()           # (N, obs_dim)
        for update in range(1, args.updates + 1):
            buf.clear()
            t_roll = time.time()
            obs, last_v, ep_ret, ep_len, ep_sc = collect_rollout(
                vec_env, ppo, buf, args.rollout, ppo.device, obs)
            roll_time = time.time() - t_roll

            t_upd = time.time()
            stats = ppo.update(buf, last_v, gamma=args.gamma, lam=args.lam)
            upd_time = time.time() - t_upd

            mean_ret = float(np.mean(ep_ret))  if ep_ret else float("nan")
            mean_len = float(np.mean(ep_len))  if ep_len else float("nan")
            mean_sc  = float(np.mean(ep_sc))   if ep_sc  else float("nan")
            max_sc   = int(np.max(ep_sc))      if ep_sc  else 0
            elapsed  = time.time() - t0
            samples  = args.rollout * args.n_envs
            sps      = samples / max(roll_time, 1e-6)

            print(f"upd {update:4d} | eps {len(ep_ret):3d} | "
                  f"ret {mean_ret:7.2f} | len {mean_len:6.1f} | "
                  f"score avg {mean_sc:6.1f} max {max_sc:5d} | "
                  f"pi {stats['pi_loss']:+.3f} v {stats['v_loss']:.3f} "
                  f"H {stats['entropy']:.3f} g {stats['grad_norm']:.2f} | "
                  f"{sps:5.1f} sps | "
                  f"roll {roll_time:5.1f}s upd {upd_time:4.1f}s | "
                  f"{elapsed:6.0f}s")

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "update": update,
                    "episodes": len(ep_ret),
                    "mean_return": mean_ret,
                    "mean_len": mean_len,
                    "mean_score": mean_sc,
                    "max_score": max_sc,
                    **stats,
                    "samples": samples,
                    "sps": sps,
                    "roll_time": roll_time,
                    "upd_time": upd_time,
                    "elapsed": elapsed,
                }) + "\n")

            if update % args.save_every == 0:
                ppo.save(os.path.join(args.save_dir, f"ppo_upd{update}.pt"))
                ppo.save(os.path.join(args.save_dir, "latest.pt"))
            if max_sc > best_score:
                best_score = max_sc
                ppo.save(os.path.join(args.save_dir, "best.pt"))

    except KeyboardInterrupt:
        print("\n[interrupt] saving and exiting")
    finally:
        ppo.save(os.path.join(args.save_dir, "latest.pt"))
        vec_env.close()


if __name__ == "__main__":
    main()
