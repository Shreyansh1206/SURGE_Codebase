"""
Collect behavior-cloning demonstrations from the scripted Dino expert.

Runs the heuristic expert (envs/dino_expert.py) across many seeds and records
(stacked_48_obs, expert_action) pairs — the exact observation the policy will
see at decision time, paired with the action the expert took. Saved to a .npz
for the BC pretrainer.

Usage:
    python collect_dino_demos.py --n-transitions 150000 --out demos/dino_demos.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import dino_env  # noqa: F401 — sets up Dino_runGame path
from dino_env import DinoEnv, FRAME_STACK, FEATURES_PER_FRAME, N_ACTIONS
from envs.dino_expert import expert_action

OBS_DIM = FRAME_STACK * FEATURES_PER_FRAME
ACTION_NAMES = {0: "run", 1: "jump", 2: "duck"}


def collect(n_transitions: int, seed_start: int, frames_per_step: int):
    obs_buf = np.empty((n_transitions, OBS_DIM), dtype=np.float32)
    act_buf = np.empty((n_transitions,), dtype=np.int64)
    n = 0
    seed = seed_start
    ep_scores = []

    while n < n_transitions:
        env = DinoEnv(render=False, frames_per_step=frames_per_step, seed=seed)
        obs = env.reset()
        done = False
        last_score = 0
        guard = 0
        while not done and n < n_transitions and guard < 20000:
            state = env.engine.get_state()
            action = expert_action(state)
            obs_buf[n] = obs
            act_buf[n] = action
            n += 1
            obs, _, done, info = env.step(action)
            last_score = info.get("score", last_score)
            guard += 1
        ep_scores.append(last_score)
        env.close()
        seed += 1
        if len(ep_scores) % 25 == 0:
            print(f"  collected {n}/{n_transitions} transitions | "
                  f"{len(ep_scores)} eps | mean score {np.mean(ep_scores):.0f}")

    return obs_buf[:n], act_buf[:n], ep_scores


def main():
    p = argparse.ArgumentParser(description="Collect Dino BC demos from the scripted expert")
    p.add_argument("--n-transitions", type=int, default=150000)
    p.add_argument("--out", type=str, default="demos/dino_demos.npz")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--frames-per-step", type=int, default=4)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[collect] target {args.n_transitions} transitions, frames_per_step={args.frames_per_step}")
    obs, act, scores = collect(args.n_transitions, args.seed_start, args.frames_per_step)

    counts = np.bincount(act, minlength=N_ACTIONS)
    print(f"[collect] done: {len(act)} transitions over {len(scores)} episodes")
    print(f"[collect] expert score: mean {np.mean(scores):.0f}  max {int(np.max(scores))}")
    print(f"[collect] action distribution: " +
          ", ".join(f"{ACTION_NAMES[i]}={counts[i]} ({100*counts[i]/len(act):.1f}%)"
                    for i in range(N_ACTIONS)))
    np.savez_compressed(args.out, obs=obs, actions=act,
                        class_counts=counts, mean_score=float(np.mean(scores)))
    print(f"[collect] saved -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
