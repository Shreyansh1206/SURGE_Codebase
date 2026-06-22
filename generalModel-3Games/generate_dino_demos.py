"""Generate expert demonstrations for Dino BC using the rule-based expert.

The expert uses obstacle position + type to decide jump/duck/run.
Output: demos/dino_demos.npz with keys 'obs' and 'actions' (matching DinoEnv's
48-dim observation space and discrete 3-action space).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dino_env import DinoEnv, N_ACTIONS
from envs.dino_expert import expert_action


def collect_demos(n_episodes: int = 100, seed_base: int = 5000) -> dict:
    all_obs, all_actions = [], []
    scores = []

    for ep in range(n_episodes):
        env = DinoEnv(render=False, seed=seed_base + ep, frames_per_step=4)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 20_000:
            state = env.engine.get_state()
            action = expert_action(state)
            all_obs.append(obs.copy())
            all_actions.append(action)
            obs, _, done, info = env.step(action)
            steps += 1
        scores.append(info.get("score", 0))
        env.close()

    obs_arr = np.array(all_obs, dtype=np.float32)
    act_arr = np.array(all_actions, dtype=np.int64)
    counts = np.bincount(act_arr, minlength=N_ACTIONS)

    print(f"Collected {len(act_arr)} steps from {n_episodes} episodes")
    print(f"  Mean score: {np.mean(scores):.1f} | Max: {np.max(scores)} | Min: {np.min(scores)}")
    print(f"  Action distribution: run={counts[0]} jump={counts[1]} duck={counts[2]}")
    print(f"  Duck %: {100*counts[2]/len(act_arr):.1f}%")

    return {
        "obs": obs_arr,
        "actions": act_arr,
        "mean_score": np.mean(scores),
        "max_score": np.max(scores),
        "scores": np.array(scores),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", type=str, default="demos/dino_demos.npz")
    p.add_argument("--seed", type=int, default=5000)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = collect_demos(args.episodes, args.seed)
    np.savez_compressed(args.out, **data)
    print(f"Saved -> {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)")
