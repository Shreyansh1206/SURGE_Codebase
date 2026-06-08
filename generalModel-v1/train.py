"""
Interleaved multi-task PPO training for MiniGrid + Chrome Dino.

Each update cycle:
  1. Collect on-policy rollouts from MiniGrid vector env
  2. Collect on-policy rollouts from Dino vector env
  3. Run separate PPO updates per task (no summed loss / shared backward)
"""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from envs.dino_gym import DINO_N_ACTIONS, DINO_OBS_DIM, VecDinoGymEnv
from multi_task_ppo import (
    TASK_DINO,
    TASK_MINIGRID,
    MultiTaskPPO,
    RolloutBuffer,
)


def _num_envs(vec_env) -> int:
    return getattr(vec_env, "n_envs", getattr(vec_env, "num_envs", 1))


def _env_info_at(infos, i: int) -> dict:
    """Per-env info from gymnasium vector (dict) or legacy list returns."""
    if isinstance(infos, list):
        return infos[i] if i < len(infos) else {}
    if isinstance(infos, dict):
        out = {}
        for key, val in infos.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    if isinstance(v2, (list, tuple, np.ndarray)) and len(v2) > i:
                        out[k2] = v2[i]
            elif isinstance(val, (list, tuple, np.ndarray)) and len(val) > i:
                item = val[i]
                if isinstance(item, dict):
                    out.update(item)
                else:
                    out[key] = item
        return out
    return {}


def _episode_metric(info: dict, task_name: str, fallback: float) -> float:
    if task_name == TASK_DINO:
        return float(info.get("score", fallback))
    return float(info.get("episode_return", info.get("r", fallback)))


def _render_env(vec_env):
    if hasattr(vec_env, "render"):
        vec_env.render()
    elif hasattr(vec_env, "_vec") and hasattr(vec_env._vec, "render"):
        vec_env._vec.render()


def collect_rollout(
    vec_env, ppo, buf, n_steps, device, last_obs, task_name, render=False, render_delay=0.0
):
    if render and hasattr(vec_env, "prepare_render"):
        vec_env.prepare_render()
    N = _num_envs(vec_env)
    obs = last_obs
    ep_returns, ep_lens, ep_scores = [], [], []
    cur_return = np.zeros(N, dtype=np.float32)
    cur_len = np.zeros(N, dtype=np.int64)

    for _ in range(n_steps):
        actions, logps, values = ppo.net.act_batch(obs, task_name, device)
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        if render:
            _render_env(vec_env)
            if render_delay > 0:
                time.sleep(render_delay)
        dones = np.logical_or(terminated, truncated)
        buf.add(obs, actions, logps, rewards, values, dones)
        cur_return += rewards
        cur_len += 1
        for i in range(N):
            if dones[i]:
                ep_returns.append(float(cur_return[i]))
                ep_lens.append(int(cur_len[i]))
                ep_scores.append(
                    _episode_metric(_env_info_at(infos, i), task_name, cur_return[i])
                )
                cur_return[i] = 0.0
                cur_len[i] = 0
        obs = next_obs

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)
        _, last_v = ppo.net(obs_t, task_name)
        last_values = last_v.cpu().numpy().astype(np.float32)

    return obs, last_values, ep_returns, ep_lens, ep_scores


class MiniGridVecEnv:
    """Thin adapter so MiniGrid vector env matches the Dino vec API."""

    def __init__(self, n_envs: int, env_id: str, seed: int, render: bool = False):
        from envs.minigrid_env import make_minigrid_env

        self.n_envs = n_envs
        self.show_window = render
        self.env_id = env_id
        self._single = None
        self._vec = None

        # Single-env path: correct human render + list-style infos (visible training).
        if render or n_envs == 1:
            self.n_envs = 1
            render_mode = "human" if render else None
            self._single = make_minigrid_env(env_id, render_mode=render_mode)
            self._single.reset(seed=seed)
        else:
            def _factory():
                return make_minigrid_env(env_id)

            self._vec = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
            self._vec.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        if self._single is not None:
            obs, info = self._single.reset(seed=seed)
            return np.asarray([obs], dtype=np.float32), [info]
        return self._vec.reset(seed=seed)

    def step(self, actions):
        if self._single is not None:
            action = int(np.asarray(actions).reshape(-1)[0])
            obs, reward, terminated, truncated, info = self._single.step(action)
            if self.show_window:
                self.prepare_render()
                self._single.render()
            return (
                np.asarray([obs], dtype=np.float32),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated], dtype=bool),
                np.asarray([truncated], dtype=bool),
                [info],
            )
        return self._vec.step(actions)

    def prepare_render(self):
        if self._single is not None and self.show_window:
            from envs.minigrid_env import refresh_minigrid_display

            refresh_minigrid_display(self._single)

    def render(self):
        if self._single is not None:
            return self._single.render()
        return self._vec.render()

    def close(self):
        if self._single is not None:
            self._single.close()
        if self._vec is not None:
            self._vec.close()


def make_minigrid_vec_env(n_envs: int, env_id: str, seed: int, render: bool = False):
    return MiniGridVecEnv(n_envs, env_id, seed, render=render)


def _load_minigrid():
    from envs.minigrid_env import MINIGRID_ACTIONS, make_minigrid_env, minigrid_obs_dim

    return MINIGRID_ACTIONS, make_minigrid_env, minigrid_obs_dim


def main():
    p = argparse.ArgumentParser(description="Multi-task PPO: MiniGrid + Dino")
    p.add_argument("--minigrid-env-id", type=str, default="MiniGrid-DoorKey-16x16-v0")
    p.add_argument("--n-minigrid-envs", type=int, default=8)
    p.add_argument("--n-dino-envs", type=int, default=1)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument(
        "--rollout",
        type=int,
        default=128,
        help="Rollout steps per env for MiniGrid (and Dino if --dino-rollout unset).",
    )
    p.add_argument(
        "--dino-rollout",
        type=int,
        default=512,
        help="Rollout steps per Dino env per update (each step = 4 game frames).",
    )
    p.add_argument(
        "--minigrid-rollout",
        type=int,
        default=None,
        help="Rollout steps per MiniGrid env (default: --rollout).",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dino-bc-demos",
        type=str,
        default=None,
        help="Path to expert demo .npz; adds a BC anchor to the Dino update so RL "
        "fine-tuning can't drift away from the reliable cloned policy.",
    )
    p.add_argument("--dino-bc-coef", type=float, default=0.5,
                   help="Weight of the Dino BC anchor loss (used only with --dino-bc-demos).")
    p.add_argument("--dino-only", action="store_true", help="Skip MiniGrid (smoke test).")
    p.add_argument("--minigrid-only", action="store_true", help="Skip Dino (smoke test).")
    p.add_argument(
        "--render",
        action="store_true",
        help="Show both games on screen during rollout collection.",
    )
    p.add_argument(
        "--render-minigrid",
        action="store_true",
        help="Show the MiniGrid window (Farama minigrid via gymnasium).",
    )
    p.add_argument(
        "--render-dino",
        action="store_true",
        help="Show the pygame Dino window during training.",
    )
    p.add_argument(
        "--render-delay",
        type=float,
        default=0.0,
        help="Seconds to pause after each env step when rendering (e.g. 0.02).",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="Headless parallel training: 8 MiniGrid + multiprocess Dino envs (no windows).",
    )
    args = p.parse_args()

    parallel = args.parallel
    render_minigrid = (args.render or args.render_minigrid) and not parallel
    render_dino = (args.render or args.render_dino) and not parallel
    if parallel and (args.render or args.render_minigrid or args.render_dino):
        print("[parallel] disabling on-screen render — use train_visible.py to watch games.")

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    use_minigrid = not args.dino_only
    use_dino = not args.minigrid_only

    mg_dim = 147  # 7×7×3 partial view (default MiniGrid agent_view_size)
    minigrid_actions = 7
    if use_minigrid:
        minigrid_actions, _, minigrid_obs_dim = _load_minigrid()
        mg_dim = minigrid_obs_dim(args.minigrid_env_id)

    print(
        f"[generalModel-v1] MiniGrid obs={mg_dim} actions={minigrid_actions} | "
        f"Dino obs={DINO_OBS_DIM} actions={DINO_N_ACTIONS}"
    )
    mg_rollout = args.minigrid_rollout if args.minigrid_rollout is not None else args.rollout
    dino_rollout = args.dino_rollout

    if render_minigrid or render_dino:
        print(
            f"[render] minigrid={render_minigrid} dino={render_dino} "
            f"delay={args.render_delay}s | "
            f"MiniGrid uses the official `minigrid` library (gymnasium.make)"
        )
    if parallel:
        print(
            f"[parallel] headless | "
            f"minigrid_envs={args.n_minigrid_envs} | "
            f"dino_envs={args.n_dino_envs} (multiprocess)"
        )
    if use_minigrid and use_dino:
        mg_samples = mg_rollout * args.n_minigrid_envs
        dino_samples = dino_rollout * args.n_dino_envs
        print(
            f"[rollout] minigrid={mg_rollout}×{args.n_minigrid_envs}={mg_samples} | "
            f"dino={dino_rollout}×{args.n_dino_envs}={dino_samples} "
            f"(×4 game frames/step)"
        )

    ppo = MultiTaskPPO(
        minigrid_dim=mg_dim,
        dino_dim=DINO_OBS_DIM,
        minigrid_actions=minigrid_actions,
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

    # Optional Dino BC anchor: keep RL from forgetting the reliable cloned policy.
    bc_obs_t = bc_act_t = bc_w_t = None
    if args.dino_bc_demos:
        _d = np.load(args.dino_bc_demos)
        bc_obs_t = torch.tensor(_d["obs"], dtype=torch.float32, device=ppo.device)
        bc_act_t = torch.tensor(_d["actions"], dtype=torch.long, device=ppo.device)
        _counts = np.bincount(_d["actions"], minlength=DINO_N_ACTIONS).astype(np.float64)
        _w = _counts.sum() / (DINO_N_ACTIONS * np.maximum(_counts, 1))
        bc_w_t = torch.tensor(_w, dtype=torch.float32, device=ppo.device)
        print(f"[dino-bc] anchor on: {len(bc_act_t)} demos, coef={args.dino_bc_coef}, "
              f"class_weights={np.round(_w, 2).tolist()}")

    mg_vec = None
    dino_vec = None
    mg_buf = None
    dino_buf = None
    mg_obs = None
    dino_obs = None

    if use_minigrid:
        mg_vec = make_minigrid_vec_env(
            args.n_minigrid_envs,
            args.minigrid_env_id,
            args.seed,
            render=render_minigrid,
        )
        mg_buf = RolloutBuffer(args.n_minigrid_envs)
        mg_obs, _ = mg_vec.reset(seed=args.seed)

    if use_dino:
        dino_vec = VecDinoGymEnv(
            n_envs=args.n_dino_envs,
            render=render_dino,
            parallel=parallel,
        )
        dino_buf = RolloutBuffer(args.n_dino_envs)
        dino_obs, _ = dino_vec.reset(seed=args.seed)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    t0 = time.time()

    try:
        for update in range(1, args.updates + 1):
            task_stats = {}

            if use_minigrid:
                mg_buf.clear()
                t_roll = time.time()
                mg_obs, mg_last_v, mg_ret, mg_len, mg_sc = collect_rollout(
                    mg_vec,
                    ppo,
                    mg_buf,
                    mg_rollout,
                    ppo.device,
                    mg_obs,
                    TASK_MINIGRID,
                    render=render_minigrid,
                    render_delay=args.render_delay,
                )
                mg_roll_t = time.time() - t_roll
                t_upd = time.time()
                task_stats[TASK_MINIGRID] = ppo.update_task(
                    TASK_MINIGRID, mg_buf, mg_last_v, gamma=args.gamma, lam=args.lam
                )
                task_stats[TASK_MINIGRID]["roll_time"] = mg_roll_t
                task_stats[TASK_MINIGRID]["upd_time"] = time.time() - t_upd
                task_stats[TASK_MINIGRID]["episodes"] = len(mg_ret)
                task_stats[TASK_MINIGRID]["mean_return"] = (
                    float(np.mean(mg_ret)) if mg_ret else float("nan")
                )
                task_stats[TASK_MINIGRID]["mean_len"] = (
                    float(np.mean(mg_len)) if mg_len else float("nan")
                )

            if use_dino:
                dino_buf.clear()
                t_roll = time.time()
                dino_obs, dino_last_v, dino_ret, dino_len, dino_sc = collect_rollout(
                    dino_vec,
                    ppo,
                    dino_buf,
                    dino_rollout,
                    ppo.device,
                    dino_obs,
                    TASK_DINO,
                    render=render_dino,
                    render_delay=args.render_delay,
                )
                dino_roll_t = time.time() - t_roll
                t_upd = time.time()
                task_stats[TASK_DINO] = ppo.update_task(
                    TASK_DINO, dino_buf, dino_last_v, gamma=args.gamma, lam=args.lam,
                    bc_obs=bc_obs_t, bc_actions=bc_act_t,
                    bc_coef=args.dino_bc_coef, bc_weight=bc_w_t,
                )
                task_stats[TASK_DINO]["roll_time"] = dino_roll_t
                task_stats[TASK_DINO]["upd_time"] = time.time() - t_upd
                task_stats[TASK_DINO]["episodes"] = len(dino_ret)
                task_stats[TASK_DINO]["mean_return"] = (
                    float(np.mean(dino_ret)) if dino_ret else float("nan")
                )
                task_stats[TASK_DINO]["mean_score"] = (
                    float(np.mean(dino_sc)) if dino_sc else float("nan")
                )
                task_stats[TASK_DINO]["max_score"] = int(np.max(dino_sc)) if dino_sc else 0

            elapsed = time.time() - t0
            parts = [f"upd {update:4d}"]
            if use_minigrid and TASK_MINIGRID in task_stats:
                s = task_stats[TASK_MINIGRID]
                parts.append(
                    f"mg ret {s['mean_return']:6.2f} pi {s['pi_loss']:+.3f} "
                    f"v {s['v_loss']:.3f} H {s['entropy']:.3f}"
                )
            if use_dino and TASK_DINO in task_stats:
                s = task_stats[TASK_DINO]
                parts.append(
                    f"dino ret {s['mean_return']:6.2f} score {s.get('mean_score', 0):5.1f} "
                    f"pi {s['pi_loss']:+.3f} v {s['v_loss']:.3f}"
                )
            parts.append(f"{elapsed:6.0f}s")
            print(" | ".join(parts))

            with open(log_path, "a") as f:
                f.write(
                    json.dumps(
                        {"update": update, "elapsed": elapsed, "tasks": task_stats}
                    )
                    + "\n"
                )

            if update % args.save_every == 0:
                ppo.save(os.path.join(args.save_dir, f"mt_ppo_upd{update}.pt"))
                ppo.save(os.path.join(args.save_dir, "latest.pt"))

    except KeyboardInterrupt:
        print("\n[interrupt] saving and exiting")
    finally:
        ppo.save(os.path.join(args.save_dir, "latest.pt"))
        if mg_vec is not None:
            mg_vec.close()
        if dino_vec is not None:
            dino_vec.close()


if __name__ == "__main__":
    main()
