"""
Curriculum trainer for MiniGrid DoorKey — the standard way to crack 16x16.

DoorKey-16x16 is a sparse-reward exploration wall: a from-scratch PPO agent
almost never stumbles onto the goal, so it gets no learning signal (see the
flatlined `checkpoints_v3_updated_rewards_mg` run). Instead we ramp difficulty:

    5x5 -> 6x6 -> 8x8 -> 10x10 -> 12x12 -> 14x14 -> 16x16

The agent's view is ALWAYS 7x7, so the network (one-hot CNN encoder) is identical
at every size and weights transfer with zero surgery. Each stage starts from a
policy that already knows "pick up key, open door, reach goal" on a smaller map
and only has to scale that skill up.

A stage is cleared once the agent reliably solves it (solve-rate EMA >= threshold),
then we carry the weights forward. This is MiniGrid-only — no Dino.

Usage:
    python train_minigrid_curriculum.py
    python train_minigrid_curriculum.py --stages 8,10,12,14,16 --rollout 256
    python train_minigrid_curriculum.py --resume checkpoints_curriculum/latest.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from envs.minigrid_env import make_minigrid_env, minigrid_obs_dim
from multi_task_ppo import TASK_MINIGRID, MultiTaskPPO, RolloutBuffer
from train import collect_rollout

DINO_OBS_DIM = 48  # unused head; kept so the shared network shape is stable
DINO_N_ACTIONS = 3

DEFAULT_STAGES = (5, 6, 8, 10, 12, 14, 16)
# A run that clearly reached the goal scores well above the shaping-only ceiling
# (key +0.25, door +0.35 => 0.6 max without the goal). 0.5 cleanly separates them.
SOLVE_RETURN_THRESHOLD = 0.5


def stage_max_steps(size: int) -> int:
    """Tight episode cap: enough headroom to solve, short enough for frequent resets."""
    return max(80, 12 * size)


def make_doorkey_vec_env(size: int, n_envs: int, seed: int, max_steps: int):
    env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"

    def _factory():
        return make_minigrid_env(env_id, max_episode_steps=max_steps)

    vec = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
    vec.reset(seed=seed)
    return vec, env_id


def train_stage(ppo, args, size, stage_idx, log_f, t0):
    max_steps = stage_max_steps(size)
    vec, env_id = make_doorkey_vec_env(size, args.n_envs, args.seed + stage_idx, max_steps)
    buf = RolloutBuffer(args.n_envs)
    obs, _ = vec.reset(seed=args.seed + stage_idx)

    print(
        f"\n{'=' * 72}\n  STAGE {stage_idx + 1}: {env_id}  "
        f"(max_steps={max_steps}, envs={args.n_envs}, rollout={args.rollout})\n{'=' * 72}"
    )

    ema_solve = 0.0
    ema_alpha = 0.2
    cleared = False

    for upd in range(1, args.max_updates_per_stage + 1):
        buf.clear()
        obs, last_v, ep_ret, ep_len, _ = collect_rollout(
            vec, ppo, buf, args.rollout, ppo.device, obs, TASK_MINIGRID
        )
        stats = ppo.update_task(
            TASK_MINIGRID, buf, last_v, gamma=args.gamma, lam=args.lam
        )

        if ep_ret:
            mean_ret = float(np.mean(ep_ret))
            mean_len = float(np.mean(ep_len))
            solve = float(np.mean([r >= SOLVE_RETURN_THRESHOLD for r in ep_ret]))
            ema_solve = (1 - ema_alpha) * ema_solve + ema_alpha * solve
        else:
            mean_ret, mean_len, solve = float("nan"), float("nan"), 0.0

        elapsed = time.time() - t0
        print(
            f"  [{size}x{size}] upd {upd:3d} | ret {mean_ret:6.2f} | "
            f"solve {solve:4.2f} ema {ema_solve:4.2f} | len {mean_len:5.1f} | "
            f"ep {len(ep_ret):3d} | H {stats['entropy']:.2f} | {elapsed:5.0f}s"
        )
        log_f.write(
            json.dumps(
                {
                    "stage": stage_idx,
                    "size": size,
                    "env_id": env_id,
                    "update": upd,
                    "elapsed": elapsed,
                    "mean_return": mean_ret,
                    "mean_len": mean_len,
                    "solve_rate": solve,
                    "ema_solve": ema_solve,
                    "episodes": len(ep_ret),
                    **stats,
                }
            )
            + "\n"
        )
        log_f.flush()

        if update_should_save(upd, args):
            ppo.save(os.path.join(args.save_dir, "latest.pt"))

        if ema_solve >= args.advance_solve and upd >= args.min_stage_updates:
            print(f"  -> cleared {env_id} (ema_solve {ema_solve:.2f}) after {upd} updates")
            cleared = True
            break

    ppo.save(os.path.join(args.save_dir, f"mt_ppo_doorkey{size}.pt"))
    ppo.save(os.path.join(args.save_dir, "latest.pt"))
    vec.close()
    if not cleared:
        print(
            f"  !! stage {env_id} hit the {args.max_updates_per_stage}-update cap "
            f"(ema_solve {ema_solve:.2f}). Advancing anyway with current weights."
        )
    return cleared


def update_should_save(upd: int, args) -> bool:
    return upd % args.save_every == 0


def main():
    p = argparse.ArgumentParser(description="Curriculum PPO for MiniGrid DoorKey")
    p.add_argument("--stages", type=str, default=",".join(map(str, DEFAULT_STAGES)),
                   help="Comma-separated DoorKey sizes, easiest first.")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--max-updates-per-stage", type=int, default=200)
    p.add_argument("--min-stage-updates", type=int, default=10,
                   help="Minimum updates before a stage may be cleared.")
    p.add_argument("--advance-solve", type=float, default=0.8,
                   help="Clear a stage once the solve-rate EMA reaches this.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--entropy", type=float, default=0.03,
                   help="Higher than the multitask default — DoorKey needs exploration.")
    p.add_argument("--save-dir", type=str, default="checkpoints_curriculum")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    stages = [int(s) for s in args.stages.split(",") if s.strip()]
    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mg_dim = minigrid_obs_dim(f"MiniGrid-DoorKey-{stages[0]}x{stages[0]}-v0")
    print(f"[curriculum] MiniGrid obs={mg_dim} | stages={stages} | "
          f"advance@solve_ema>={args.advance_solve}")

    ppo = MultiTaskPPO(
        minigrid_dim=mg_dim,
        dino_dim=DINO_OBS_DIM,
        dino_actions=DINO_N_ACTIONS,
        lr=args.lr,
        clip_eps=args.clip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy,
    )
    if args.resume:
        print(f"[resume] loading {args.resume}")
        ppo.load(args.resume)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    t0 = time.time()
    with open(log_path, "a") as log_f:
        try:
            for stage_idx, size in enumerate(stages):
                train_stage(ppo, args, size, stage_idx, log_f, t0)
        except KeyboardInterrupt:
            print("\n[interrupt] saving and exiting")
        finally:
            ppo.save(os.path.join(args.save_dir, "latest.pt"))

    print(f"\n[done] final weights -> {os.path.join(args.save_dir, 'latest.pt')}")


if __name__ == "__main__":
    main()
