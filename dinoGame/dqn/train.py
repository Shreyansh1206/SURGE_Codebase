"""
DQN training loop for Chrome Dino.

Phased rollout per the plan:
  - default --n-step 1 (verify the network learns basic obstacle clearance)
  - flip to --n-step 3 once we've confirmed basic learning works

Usage:
    python -m dqn.train
    python -m dqn.train --n-envs 4 --total-steps 500000
    python -m dqn.train --n-step 3 --resume dqn/checkpoints/latest.pt

The loop:
  1. act_batch on current obs+mask (ε-greedy)
  2. vec_env.step → push each env's transition into NStepCollector
  3. once warmup is filled, run 1 gradient step every `train-every` env steps
  4. every `eval-every` env steps: pause, run greedy eval episodes on env 0,
     log eval score, resume
  5. checkpoint latest.pt every `save-every` env steps; best.pt on new max
     eval score

Stopping criterion: eval-mean plateau across `plateau-window` eval cycles,
OR hard cap at `--total-steps` env steps (whichever comes first).
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch

from .dino_env  import VecDinoEnv, OBS_DIM, N_ACTIONS, action_mask_from_obs
from .dqn_agent import DQNAgent
from .replay_buffer import ReplayBuffer, NStepCollector


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    return start + (end - start) * (step / decay_steps)


def run_eval(env, agent, n_episodes: int) -> dict:
    """Run greedy (ε=0) episodes on a single env. Returns score stats.
    `env` is one DinoEnv (not the vec). The caller is responsible for not
    interleaving training stepping with this env."""
    scores = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.act_greedy(obs, mask)
            obs, _, done, info = env.step(action)
        scores.append(info.get("score", 0))
    return {
        "mean":   float(np.mean(scores)),
        "max":    int(np.max(scores)),
        "min":    int(np.min(scores)),
        "scores": scores,
    }


def main():
    p = argparse.ArgumentParser()
    # Env
    p.add_argument("--n-envs",       type=int,   default=4)
    p.add_argument("--window-w",     type=int,   default=700)
    p.add_argument("--window-h",     type=int,   default=320)
    p.add_argument("--grid-cols",    type=int,   default=2)
    p.add_argument("--headless",     action="store_true",
                   help="Run browsers off-screen. Off by default; mirrors PPO.")
    p.add_argument("--chromedriver", type=str,   default=None)
    p.add_argument("--game-url",     type=str,   default=None)
    p.add_argument("--step-pause",   type=float, default=0.0)
    # Algo
    p.add_argument("--n-step",       type=int,   default=1,
                   help="n in n-step returns. Start at 1 to debug; bump to 3 once stable.")
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--tau",          type=float, default=0.005)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--grad-clip",    type=float, default=10.0)
    # Replay / exploration
    p.add_argument("--buffer-cap",   type=int,   default=100_000)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--warmup-steps", type=int,   default=5_000,
                   help="Env steps of random play before first gradient update.")
    p.add_argument("--eps-start",    type=float, default=1.0)
    p.add_argument("--eps-end",      type=float, default=0.05)
    p.add_argument("--eps-decay",    type=int,   default=50_000)
    p.add_argument("--train-every",  type=int,   default=4,
                   help="Env steps per gradient step (per env, so global ratio = train_every / n_envs).")
    # Training schedule
    p.add_argument("--total-steps",  type=int,   default=500_000,
                   help="Hard cap on env steps (across all envs combined).")
    p.add_argument("--eval-every",   type=int,   default=25_000,
                   help="Env steps between eval cycles.")
    p.add_argument("--eval-episodes", type=int,  default=3)
    p.add_argument("--plateau-window", type=int, default=4,
                   help="Stop if eval mean hasn't improved over this many cycles.")
    p.add_argument("--save-dir",     type=str,   default="dqn/checkpoints")
    p.add_argument("--save-every",   type=int,   default=5_000)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ----- env + agent -------------------------------------------------------
    print(f"[dqn-train] launching {args.n_envs} envs ({args.window_w}x{args.window_h})")
    vec_env = VecDinoEnv(
        n_envs       = args.n_envs,
        window_size  = (args.window_w, args.window_h),
        grid_cols    = args.grid_cols,
        chromedriver_path = args.chromedriver,
        step_pause   = args.step_pause,
        game_url     = args.game_url,
        headless     = args.headless,
    )

    agent = DQNAgent(OBS_DIM, N_ACTIONS,
                     lr=args.lr, gamma=args.gamma, tau=args.tau,
                     hidden=args.hidden, grad_clip=args.grad_clip)
    if args.resume:
        print(f"[resume] loading {args.resume}")
        agent.load(args.resume)

    buf  = ReplayBuffer(capacity=args.buffer_cap, obs_dim=OBS_DIM, n_actions=N_ACTIONS)
    coll = NStepCollector(n_envs=args.n_envs, n=args.n_step, gamma=args.gamma)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    eval_path = os.path.join(args.save_dir, "eval_log.jsonl")
    t0 = time.time()

    # ----- main loop ---------------------------------------------------------
    obs   = vec_env.reset()                           # (N, obs_dim)
    masks = vec_env.masks()                           # (N, n_actions)
    env_step      = 0
    grad_step     = 0
    train_counter = 0
    best_eval     = -1.0
    last_save     = 0
    last_eval     = 0
    eval_history  = deque(maxlen=args.plateau_window)
    ep_returns    = [0.0] * args.n_envs
    ep_lens       = [0]    * args.n_envs
    finished_returns  = deque(maxlen=50)
    finished_scores   = deque(maxlen=50)
    finished_lens     = deque(maxlen=50)
    finished_phantoms = deque(maxlen=50)
    ep_phantoms       = [0] * args.n_envs   # per-env phantom counter, reset on done

    try:
        while env_step < args.total_steps:
            eps = linear_epsilon(env_step, args.eps_start, args.eps_end, args.eps_decay)
            actions = agent.act_batch(obs, masks, eps)

            next_obs, rewards, dones, infos, next_masks = vec_env.step(actions)
            env_step += args.n_envs

            for i in range(args.n_envs):
                ep_returns[i] += float(rewards[i])
                ep_lens[i]    += 1
                if infos[i].get("phantom", False):
                    ep_phantoms[i] += 1

                # When the env auto-reset, the obs we just got is the FRESH
                # post-reset state, so the (s, a, r, s') tuple we should store
                # uses the terminal_obs that the vec env preserved for us.
                if dones[i] and "terminal_obs" in infos[i]:
                    real_next_obs  = infos[i]["terminal_obs"]
                    real_next_mask = action_mask_from_obs(real_next_obs)
                else:
                    real_next_obs  = next_obs[i]
                    real_next_mask = next_masks[i]

                coll.add(
                    env_i     = i,
                    obs       = obs[i],
                    action    = int(actions[i]),
                    reward    = float(rewards[i]),
                    next_obs  = real_next_obs,
                    done      = bool(dones[i]),
                    next_mask = real_next_mask,
                    buffer    = buf,
                )

                if dones[i]:
                    finished_returns.append(ep_returns[i])
                    finished_scores.append(int(infos[i].get("score", 0)))
                    finished_lens.append(ep_lens[i])
                    finished_phantoms.append(ep_phantoms[i])
                    ep_returns[i]  = 0.0
                    ep_lens[i]     = 0
                    ep_phantoms[i] = 0

            obs   = next_obs
            masks = next_masks

            # Gradient steps. Only start counting after warmup is satisfied so
            # train_counter doesn't accumulate a huge debt during warmup and
            # cause a burst of 1000+ grad steps the moment warmup ends.
            in_warmup = len(buf) < max(args.warmup_steps, args.batch_size)
            if not in_warmup:
                train_counter += args.n_envs
            if not in_warmup:
                while train_counter >= args.train_every:
                    train_counter -= args.train_every
                    batch = buf.sample(args.batch_size)
                    stats = agent.update(batch)
                    grad_step += 1

                    if grad_step % 200 == 0:
                        avg_ret = np.mean(finished_returns)  if finished_returns  else float("nan")
                        avg_sc  = np.mean(finished_scores)   if finished_scores   else float("nan")
                        max_sc  = max(finished_scores)       if finished_scores   else 0
                        avg_len = np.mean(finished_lens)     if finished_lens     else float("nan")
                        avg_ph  = np.mean(finished_phantoms) if finished_phantoms else 0.0
                        elapsed = time.time() - t0
                        print(f"step {env_step:7d} | grad {grad_step:6d} | "
                              f"eps {eps:.3f} | buf {len(buf):6d} | "
                              f"ret {avg_ret:6.2f} | sc avg {avg_sc:6.1f} max {max_sc:5d} | "
                              f"len {avg_len:5.0f} | phantoms/ep {avg_ph:.2f} | "
                              f"loss {stats['loss']:.4f} | q {stats['q_mean']:+.2f} "
                              f"tgt {stats['target_mean']:+.2f} | g {stats['grad_norm']:5.2f} | "
                              f"{elapsed:6.0f}s")
                        with open(log_path, "a") as f:
                            f.write(json.dumps({
                                "env_step": env_step, "grad_step": grad_step,
                                "epsilon": eps, "buffer": len(buf),
                                "avg_ret": float(avg_ret) if avg_ret == avg_ret else None,
                                "avg_score": float(avg_sc) if avg_sc == avg_sc else None,
                                "max_score": int(max_sc),
                                "avg_len": float(avg_len) if avg_len == avg_len else None,
                                "avg_phantoms_per_ep": float(avg_ph),
                                **stats, "elapsed": elapsed,
                            }) + "\n")

            # Checkpoint.
            if env_step - last_save >= args.save_every:
                last_save = env_step
                agent.save(os.path.join(args.save_dir, "latest.pt"))

            # Eval.
            if env_step - last_eval >= args.eval_every and len(buf) >= args.warmup_steps:
                last_eval = env_step
                print(f"[eval] step {env_step}: running {args.eval_episodes} greedy episodes on env 0")
                # Use env 0 for eval. Its in-flight n-step queue gets dropped
                # because we're about to reset it for a fresh trajectory.
                coll.reset_env(0)
                eval_stats = run_eval(vec_env.envs[0], agent, args.eval_episodes)
                # After eval, the vec_env's `obs[0]` and `masks[0]` are stale.
                # Refresh by reading env 0's current state.
                obs[0]   = vec_env.envs[0].reset()
                masks[0] = vec_env.envs[0].action_mask()
                # Reset that env's running episode counters too.
                ep_returns[0] = 0.0
                ep_lens[0]    = 0

                print(f"[eval] mean={eval_stats['mean']:.1f}  "
                      f"max={eval_stats['max']}  scores={eval_stats['scores']}")
                with open(eval_path, "a") as f:
                    f.write(json.dumps({
                        "env_step": env_step, "grad_step": grad_step,
                        **eval_stats,
                    }) + "\n")

                eval_history.append(eval_stats["mean"])
                if eval_stats["mean"] > best_eval:
                    best_eval = eval_stats["mean"]
                    agent.save(os.path.join(args.save_dir, "best.pt"))
                    print(f"[eval] new best={best_eval:.1f} — saved best.pt")

                # Plateau check: stop if no eval in the trailing window
                # matched best_eval (i.e. the best was set before the window
                # started and nothing has caught up since).
                if len(eval_history) == args.plateau_window and \
                   max(eval_history) < best_eval - 1e-6:
                    print(f"[plateau] no eval improvement in "
                          f"{args.plateau_window} cycles "
                          f"(best={best_eval:.1f}, recent max={max(eval_history):.1f}). "
                          f"Stopping.")
                    break

    except KeyboardInterrupt:
        print("\n[interrupt] saving and exiting")
    finally:
        agent.save(os.path.join(args.save_dir, "latest.pt"))
        vec_env.close()


if __name__ == "__main__":
    main()
