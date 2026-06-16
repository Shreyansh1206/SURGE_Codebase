
import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from envs.carracing_env import (
    CARRACING_N_ACTIONS,
    carracing_obs_shape,
    make_carracing_vec_env,
)
from envs.dino_gym import DINO_N_ACTIONS, DINO_OBS_DIM, VecDinoGymEnv
from multi_task_ppo import (
    TASK_CARRACING,
    TASK_DINO,
    TASK_MINIGRID,
    MultiTaskPPO,
    RolloutBuffer,
)


def _num_envs(vec_env) -> int:
    return getattr(vec_env, "n_envs", getattr(vec_env, "num_envs", 1))


def _env_info_at(infos, i: int) -> dict:
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
        if render and not getattr(vec_env, "renders_in_step", False):
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

    def __init__(self, n_envs: int, env_id: str, seed: int, render: bool = False):
        from envs.minigrid_env import make_minigrid_env

        self.n_envs = n_envs
        self.show_window = render
        self.renders_in_step = render
        self.env_id = env_id
        self._single = None
        self._vec = None

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


def _run_task(
    ppo, args, task_name, vec, buf, obs, rollout, render, task_stats
):
    buf.clear()
    t_roll = time.time()
    obs, last_v, ep_ret, ep_len, ep_sc = collect_rollout(
        vec, ppo, buf, rollout, ppo.device, obs, task_name,
        render=render, render_delay=args.render_delay,
    )
    roll_t = time.time() - t_roll
    t_upd = time.time()
    stats = ppo.update_task(task_name, buf, last_v, gamma=args.gamma, lam=args.lam)
    stats["roll_time"] = roll_t
    stats["upd_time"] = time.time() - t_upd
    stats["episodes"] = len(ep_ret)
    stats["mean_return"] = float(np.mean(ep_ret)) if ep_ret else float("nan")
    stats["mean_len"] = float(np.mean(ep_len)) if ep_len else float("nan")
    if task_name == TASK_DINO:
        stats["mean_score"] = float(np.mean(ep_sc)) if ep_sc else float("nan")
        stats["max_score"] = int(np.max(ep_sc)) if ep_sc else 0
    task_stats[task_name] = stats
    return obs


def main():
    p = argparse.ArgumentParser(
        description="Multi-task PPO: MiniGrid + Dino + CarRacing"
    )
    p.add_argument("--minigrid-env-id", type=str, default="MiniGrid-DoorKey-16x16-v0")
    p.add_argument("--n-minigrid-envs", type=int, default=8)
    p.add_argument("--n-dino-envs", type=int, default=4)
    p.add_argument("--n-carracing-envs", type=int, default=4)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--rollout", type=int, default=128,
                   help="Default rollout steps per env (MiniGrid).")
    p.add_argument("--dino-rollout", type=int, default=512,
                   help="Rollout steps per Dino env per update (each step = 4 frames).")
    p.add_argument("--minigrid-rollout", type=int, default=None,
                   help="Rollout steps per MiniGrid env (default: --rollout).")
    p.add_argument("--carracing-rollout", type=int, default=128,
                   help="Rollout steps per CarRacing env per update (each step = frame-skip frames).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--save-dir", type=str, default="checkpoints_3games")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dino-only", action="store_true")
    p.add_argument("--minigrid-only", action="store_true")
    p.add_argument("--carracing-only", action="store_true")
    p.add_argument("--no-minigrid", action="store_true")
    p.add_argument("--no-dino", action="store_true")
    p.add_argument("--no-carracing", action="store_true")
    p.add_argument("--render", action="store_true",
                   help="Show all enabled games on screen during rollout collection.")
    p.add_argument("--render-minigrid", action="store_true")
    p.add_argument("--render-dino", action="store_true")
    p.add_argument("--render-carracing", action="store_true")
    p.add_argument("--render-delay", type=float, default=0.0)
    p.add_argument("--parallel", action="store_true",
                   help="Headless parallel training (multiprocess Dino workers).")
    args = p.parse_args()

    only_flags = [args.minigrid_only, args.dino_only, args.carracing_only]
    if sum(only_flags) > 1:
        raise SystemExit("Use at most one of --minigrid-only / --dino-only / --carracing-only")

    if args.minigrid_only:
        use_minigrid, use_dino, use_carracing = True, False, False
    elif args.dino_only:
        use_minigrid, use_dino, use_carracing = False, True, False
    elif args.carracing_only:
        use_minigrid, use_dino, use_carracing = False, False, True
    else:
        use_minigrid = not args.no_minigrid
        use_dino = not args.no_dino
        use_carracing = not args.no_carracing

    parallel = args.parallel
    render_minigrid = (args.render or args.render_minigrid) and not parallel
    render_dino = (args.render or args.render_dino) and not parallel
    render_carracing = (args.render or args.render_carracing) and not parallel
    if parallel and (args.render or args.render_minigrid or args.render_dino or args.render_carracing):
        print("[parallel] disabling on-screen render.")

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mg_dim = 147
    minigrid_actions = 7
    if use_minigrid:
        minigrid_actions, _, minigrid_obs_dim = _load_minigrid()
        mg_dim = minigrid_obs_dim(args.minigrid_env_id)

    cr_shape = carracing_obs_shape()

    print(
        f"[3games] MiniGrid obs={mg_dim} act={minigrid_actions} | "
        f"Dino obs={DINO_OBS_DIM} act={DINO_N_ACTIONS} | "
        f"CarRacing obs={cr_shape} act={CARRACING_N_ACTIONS}"
    )
    print(
        f"[tasks] minigrid={use_minigrid} dino={use_dino} carracing={use_carracing}"
    )

    mg_rollout = args.minigrid_rollout if args.minigrid_rollout is not None else args.rollout
    dino_rollout = args.dino_rollout
    cr_rollout = args.carracing_rollout

    ppo = MultiTaskPPO(
        minigrid_dim=mg_dim,
        dino_dim=DINO_OBS_DIM,
        carracing_obs_shape=cr_shape,
        minigrid_actions=minigrid_actions,
        dino_actions=DINO_N_ACTIONS,
        carracing_actions=CARRACING_N_ACTIONS,
        lr=args.lr,
        clip_eps=args.clip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy,
    )
    if args.resume:
        print(f"[resume] loading {args.resume}")
        ppo.load(args.resume)

    mg_vec = dino_vec = cr_vec = None
    mg_buf = dino_buf = cr_buf = None
    mg_obs = dino_obs = cr_obs = None

    if use_minigrid:
        mg_vec = make_minigrid_vec_env(
            args.n_minigrid_envs, args.minigrid_env_id, args.seed, render=render_minigrid
        )
        mg_buf = RolloutBuffer(args.n_minigrid_envs)
        mg_obs, _ = mg_vec.reset(seed=args.seed)

    if use_dino:
        dino_vec = VecDinoGymEnv(
            n_envs=args.n_dino_envs, render=render_dino, parallel=parallel
        )
        dino_buf = RolloutBuffer(args.n_dino_envs)
        dino_obs, _ = dino_vec.reset(seed=args.seed)

    if use_carracing:
        cr_vec = make_carracing_vec_env(
            args.n_carracing_envs, seed=args.seed, render=render_carracing
        )
        cr_buf = RolloutBuffer(args.n_carracing_envs)
        cr_obs, _ = cr_vec.reset(seed=args.seed)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    t0 = time.time()

    try:
        for update in range(1, args.updates + 1):
            task_stats = {}

            if use_minigrid:
                mg_obs = _run_task(
                    ppo, args, TASK_MINIGRID, mg_vec, mg_buf, mg_obs,
                    mg_rollout, render_minigrid, task_stats
                )
            if use_dino:
                dino_obs = _run_task(
                    ppo, args, TASK_DINO, dino_vec, dino_buf, dino_obs,
                    dino_rollout, render_dino, task_stats
                )
            if use_carracing:
                cr_obs = _run_task(
                    ppo, args, TASK_CARRACING, cr_vec, cr_buf, cr_obs,
                    cr_rollout, render_carracing, task_stats
                )

            elapsed = time.time() - t0
            parts = [f"upd {update:4d}"]
            if TASK_MINIGRID in task_stats:
                s = task_stats[TASK_MINIGRID]
                parts.append(f"mg ret {s['mean_return']:6.2f} H {s['entropy']:.2f}")
            if TASK_DINO in task_stats:
                s = task_stats[TASK_DINO]
                parts.append(
                    f"dino ret {s['mean_return']:6.2f} score {s.get('mean_score', 0):5.1f}"
                )
            if TASK_CARRACING in task_stats:
                s = task_stats[TASK_CARRACING]
                parts.append(f"car ret {s['mean_return']:7.2f} H {s['entropy']:.2f}")
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
        if cr_vec is not None:
            cr_vec.close()


if __name__ == "__main__":
    main()
