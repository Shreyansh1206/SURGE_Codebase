"""
Run a trained multi-task checkpoint on MiniGrid, Dino, or both.

Examples:
    python infer.py --task both --ckpt checkpoints_parallel/latest.pt
    python infer.py --task minigrid --episodes 10 --render
    python infer.py --task dino --ckpt checkpoints_visible/latest.pt --render
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from dino_env import FEATURES_PER_FRAME, N_ACTIONS as DINO_ACTIONS, OBS_DIM
from envs.dino_gym import DinoGymEnv
from envs.minigrid_env import MINIGRID_ACTIONS, make_minigrid_env, minigrid_obs_dim
from multi_task_ppo import TASK_DINO, TASK_MINIGRID, MultiTaskPPO

MINIGRID_ACTION_NAMES = [
    "left",
    "right",
    "forward",
    "pickup",
    "drop",
    "toggle",
    "done",
]
DINO_ACTION_NAMES = ["noop", "jump", "duck"]


def _resolve_ckpt(path: str) -> str:
    if os.path.exists(path):
        return path
    for alt in (
        "checkpoints_doorkey_16x16/latest.pt",
        "checkpoints_parallel/latest.pt",
        "checkpoints_visible/latest.pt",
        "checkpoints/latest.pt",
    ):
        if os.path.exists(alt):
            print(f"[infer] {path} not found — using {alt}")
            return alt
    raise FileNotFoundError(
        f"No checkpoint at {path}. Train first, e.g. python train_parallel.py"
    )


def _load_agent(ckpt_path: str, minigrid_env_id: str) -> MultiTaskPPO:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mg_dim = int(ckpt.get("minigrid_dim", minigrid_obs_dim(minigrid_env_id)))
    dino_dim = int(ckpt.get("dino_dim", OBS_DIM))
    agent = MultiTaskPPO(
        minigrid_dim=mg_dim,
        dino_dim=dino_dim,
        minigrid_actions=MINIGRID_ACTIONS,
        dino_actions=DINO_ACTIONS,
    )
    agent.load(ckpt_path, load_optim=False)
    agent.net.eval()
    return agent


def _select_action(agent: MultiTaskPPO, obs: np.ndarray, task: str, sample: bool):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(agent.device)
    with torch.no_grad():
        logits, value = agent.net(obs_t, task)
        probs = torch.softmax(logits, dim=-1)[0]
        if sample:
            action = int(torch.distributions.Categorical(probs).sample().item())
        else:
            action = int(probs.argmax().item())
    return action, float(value.item()), probs.cpu().numpy()


def run_minigrid(
    agent: MultiTaskPPO,
    env_id: str,
    episodes: int,
    render: bool,
    sample: bool,
    step_pause: float,
):
    from envs.minigrid_env import refresh_minigrid_display

    render_mode = "human" if render else None
    env = make_minigrid_env(env_id, render_mode=render_mode)
    returns, lengths, successes = [], [], []

    print(f"\n=== MiniGrid inference ({env_id}) ===")
    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_ret = 0.0
            steps = 0
            t0 = time.time()

            while not (terminated or truncated):
                if render:
                    refresh_minigrid_display(env)
                    env.render()
                action, value, probs = _select_action(agent, obs, TASK_MINIGRID, sample)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                steps += 1
                if render and step_pause > 0:
                    time.sleep(step_pause)

            success = ep_ret > 0 or info.get("terminated", False) and ep_ret > 0
            returns.append(ep_ret)
            lengths.append(steps)
            successes.append(success)
            print(
                f"  ep {ep:3d} | return {ep_ret:+6.2f} | steps {steps:4d} | "
                f"last_action={MINIGRID_ACTION_NAMES[action]} | "
                f"{time.time() - t0:5.1f}s"
            )
    finally:
        env.close()

    if returns:
        print(
            f"  summary | mean_return={np.mean(returns):+.2f} | "
            f"mean_len={np.mean(lengths):.1f} | episodes={len(returns)}"
        )
    return returns


def run_dino(
    agent: MultiTaskPPO,
    episodes: int,
    render: bool,
    sample: bool,
    step_pause: float,
    trace: int,
):
    render_mode = "human" if render else None
    env = DinoGymEnv(render_mode=render_mode)
    scores, returns = [], []

    print("\n=== Dino inference ===")
    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            ep_ret = 0.0
            steps = 0
            action_counts = np.zeros(DINO_ACTIONS, dtype=np.int64)
            t0 = time.time()

            while not (terminated or truncated):
                action, value, probs = _select_action(agent, obs, TASK_DINO, sample)
                if ep <= trace and (steps < 5 or action in (1, 2)):
                    f = obs[-FEATURES_PER_FRAME:]
                    print(
                        f"    ep{ep} step{steps:4d} | dx_o1={f[4]:.2f} "
                        f"jump={f[1]:.0f} speed={f[3]:.2f} | "
                        f"p={probs.round(2).tolist()} -> {DINO_ACTION_NAMES[action]} "
                        f"V={value:+.2f}"
                    )
                action_counts[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += float(reward)
                steps += 1
                if render and step_pause > 0:
                    time.sleep(step_pause)

            score = int(info.get("score", 0))
            scores.append(score)
            returns.append(ep_ret)
            frac = action_counts / max(1, action_counts.sum())
            print(
                f"  ep {ep:3d} | score {score:5d} | return {ep_ret:+7.2f} | "
                f"steps {steps:5d} | noop/jump/duck {frac.round(2).tolist()} | "
                f"{time.time() - t0:5.1f}s"
            )
    finally:
        env.close()

    if scores:
        print(
            f"  summary | mean_score={np.mean(scores):.1f} | max={max(scores)} | "
            f"mean_return={np.mean(returns):+.2f} | episodes={len(scores)}"
        )
    return scores


def main():
    p = argparse.ArgumentParser(description="Multi-task inference: MiniGrid + Dino")
    p.add_argument(
        "--task",
        choices=("minigrid", "dino", "both"),
        default="both",
        help="Which game(s) to run inference on.",
    )
    p.add_argument("--ckpt", type=str, default="checkpoints_doorkey_16x16/latest.pt")
    p.add_argument("--episodes", type=int, default=5, help="Episodes per selected game.")
    p.add_argument(
        "--minigrid-episodes",
        type=int,
        default=None,
        help="Override --episodes for MiniGrid only.",
    )
    p.add_argument(
        "--dino-episodes",
        type=int,
        default=None,
        help="Override --episodes for Dino only.",
    )
    p.add_argument("--minigrid-env-id", type=str, default="MiniGrid-DoorKey-16x16-v0")
    p.add_argument(
        "--render",
        action="store_true",
        help="Show game windows while inferring.",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="Sample actions from the policy instead of argmax.",
    )
    p.add_argument(
        "--step-pause",
        type=float,
        default=0.015,
        help="Pause between steps when --render is on.",
    )
    p.add_argument(
        "--trace",
        type=int,
        default=1,
        help="Print per-step Dino decisions for the first N episodes (0=off).",
    )
    args = p.parse_args()

    ckpt_path = _resolve_ckpt(args.ckpt)
    agent = _load_agent(ckpt_path, args.minigrid_env_id)
    mode = "sample" if args.sample else "argmax"
    print(
        f"[infer] ckpt={ckpt_path} | task={args.task} | mode={mode} | "
        f"render={args.render}"
    )

    mg_eps = args.minigrid_episodes if args.minigrid_episodes is not None else args.episodes
    dino_eps = args.dino_episodes if args.dino_episodes is not None else args.episodes

    try:
        if args.task in ("minigrid", "both"):
            run_minigrid(
                agent,
                args.minigrid_env_id,
                mg_eps,
                args.render,
                args.sample,
                args.step_pause,
            )
        if args.task in ("dino", "both"):
            run_dino(
                agent,
                dino_eps,
                args.render,
                args.sample,
                args.step_pause,
                args.trace,
            )
    except KeyboardInterrupt:
        print("\n[infer] interrupted")


if __name__ == "__main__":
    main()
