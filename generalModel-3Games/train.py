
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
    if task_name == TASK_CARRACING:
        raw = info.get("raw_return")
        if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
            return float(raw)
    return float(info.get("episode_return", info.get("r", fallback)))


def _render_env(vec_env):
    if hasattr(vec_env, "render"):
        vec_env.render()
    elif hasattr(vec_env, "_vec") and hasattr(vec_env._vec, "render"):
        vec_env._vec.render()


def _attach_aux(obs: np.ndarray, infos, n_envs: int, task_name: str) -> np.ndarray:
    """For CarRacing, flatten frames and append aux scalars to each env's obs."""
    if task_name != TASK_CARRACING:
        return obs
    aux_list = []
    for i in range(n_envs):
        info_i = _env_info_at(infos, i) if infos is not None else {}
        aux_list.append(info_i.get("aux", np.zeros(3, dtype=np.float32)))
    aux = np.stack(aux_list, axis=0).astype(np.float32)
    flat = obs.reshape(n_envs, -1)
    return np.concatenate([flat, aux], axis=-1)


def collect_rollout(
    vec_env, ppo, buf, n_steps, device, last_obs, task_name, render=False, render_delay=0.0
):
    if render and hasattr(vec_env, "prepare_render"):
        vec_env.prepare_render()
    N = _num_envs(vec_env)
    obs = last_obs
    ep_returns, ep_lens, ep_scores, ep_solved = [], [], [], []
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
                ep_solved.append(bool(terminated[i]) and not bool(truncated[i]))
                cur_return[i] = 0.0
                cur_len[i] = 0
        obs = _attach_aux(next_obs, infos, N, task_name)

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)
        _, last_v = ppo.net(obs_t, task_name)
        last_values = last_v.cpu().numpy().astype(np.float32)

    return obs, last_values, ep_returns, ep_lens, ep_scores, ep_solved


class MiniGridVecEnv:

    def __init__(
        self,
        n_envs: int,
        env_id: str,
        seed: int,
        render: bool = False,
        episode_end_pause: float = 0.0,
        max_episode_steps: int | None = None,
    ):
        from envs.minigrid_env import make_minigrid_env

        self.n_envs = n_envs
        self.show_window = render
        self.renders_in_step = render
        self.episode_end_pause = episode_end_pause if render else 0.0
        self.env_id = env_id
        self._single = None
        self._vec = None

        mk_kwargs = {"anti_stall": True}
        if max_episode_steps is not None:
            mk_kwargs["max_episode_steps"] = max_episode_steps

        if render or n_envs == 1:
            self.n_envs = 1
            render_mode = "human" if render else None
            self._single = make_minigrid_env(env_id, render_mode=render_mode, **mk_kwargs)
            self._single.reset(seed=seed)
        else:
            def _factory():
                return make_minigrid_env(env_id, **mk_kwargs)

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
            if terminated or truncated:
                from envs.episode_utils import finalize_episode

                obs, info = finalize_episode(
                    obs,
                    info,
                    lambda: self._single.reset(),
                    pause_s=self.episode_end_pause,
                )
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


def make_minigrid_vec_env(
    n_envs: int,
    env_id: str,
    seed: int,
    render: bool = False,
    episode_end_pause: float = 0.0,
    max_episode_steps: int | None = None,
):
    return MiniGridVecEnv(
        n_envs, env_id, seed, render=render,
        episode_end_pause=episode_end_pause,
        max_episode_steps=max_episode_steps,
    )


DOORKEY_DEFAULT_STAGES = (5, 6, 8, 10, 12, 14, 16)


def doorkey_env_id(size: int) -> str:
    return f"MiniGrid-DoorKey-{size}x{size}-v0"


def doorkey_stage_max_steps(size: int) -> int:
    # Keep episodes short enough that they finish inside one rollout so PPO
    # always gets a learning signal (the old 16x16/1000-step combo never did).
    return max(80, 12 * size)


def _load_minigrid():
    from envs.minigrid_env import MINIGRID_ACTIONS, make_minigrid_env, minigrid_obs_dim

    return MINIGRID_ACTIONS, make_minigrid_env, minigrid_obs_dim


def _run_task(
    ppo, args, task_name, vec, buf, obs, rollout, render, task_stats,
    bc_obs=None, bc_actions=None, bc_coef=0.0, bc_weight=None,
):
    buf.clear()
    t_roll = time.time()
    obs, last_v, ep_ret, ep_len, ep_sc, ep_solved = collect_rollout(
        vec, ppo, buf, rollout, ppo.device, obs, task_name,
        render=render, render_delay=args.render_delay,
    )
    roll_t = time.time() - t_roll
    t_upd = time.time()
    stats = ppo.update_task(
        task_name, buf, last_v, gamma=args.gamma, lam=args.lam,
        bc_obs=bc_obs, bc_actions=bc_actions, bc_coef=bc_coef, bc_weight=bc_weight,
    )
    stats["roll_time"] = roll_t
    stats["upd_time"] = time.time() - t_upd
    stats["episodes"] = len(ep_ret)
    stats["mean_return"] = float(np.mean(ep_ret)) if ep_ret else float("nan")
    stats["mean_len"] = float(np.mean(ep_len)) if ep_len else float("nan")
    if task_name == TASK_DINO:
        stats["mean_score"] = float(np.mean(ep_sc)) if ep_sc else float("nan")
        stats["max_score"] = int(np.max(ep_sc)) if ep_sc else 0
    if task_name == TASK_MINIGRID:
        stats["solve_rate"] = float(np.mean(ep_solved)) if ep_solved else float("nan")
    if task_name == TASK_CARRACING:
        stats["mean_raw_return"] = float(np.mean(ep_sc)) if ep_sc else float("nan")
    task_stats[task_name] = stats
    return obs


def main():
    p = argparse.ArgumentParser(
        description="Multi-task PPO: MiniGrid + Dino + CarRacing"
    )
    p.add_argument("--minigrid-env-id", type=str, default="MiniGrid-DoorKey-16x16-v0")
    p.add_argument("--minigrid-curriculum", action="store_true",
                   help="Auto-advance DoorKey size 5->16 as solve-rate climbs "
                   "(fixes the unlearnable 16x16 cold start).")
    p.add_argument("--minigrid-stages", type=str,
                   default=",".join(map(str, DOORKEY_DEFAULT_STAGES)),
                   help="Comma-separated DoorKey sizes for the curriculum, easiest first.")
    p.add_argument("--minigrid-advance-solve", type=float, default=0.80,
                   help="Advance to the next stage once solve-rate EMA reaches this.")
    p.add_argument("--minigrid-min-stage-updates", type=int, default=10,
                   help="Minimum updates spent on a stage before it can advance.")
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
    p.add_argument(
        "--carracing-max-episode-steps",
        type=int,
        default=1000,
        help="Gymnasium step cap on the base CarRacing env (50 FPS → 1000 ≈ 20s).",
    )
    p.add_argument(
        "--carracing-no-progress-patience",
        type=int,
        default=0,
        help="Truncate after this many base physics frames with no new track tile "
        "(0=disabled). On-track driving on old tiles is not progress, so keep at 0 "
        "unless you want to cut stuck episodes during headless training.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-schedule", type=str, default="linear",
                   choices=["constant", "linear"],
                   help="LR schedule: 'linear' decays to 0 over training (recommended).")
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
    p.add_argument("--normalize-carracing-reward", action="store_true",
                   help="Normalize CarRacing rewards by running std of discounted returns "
                   "(SB3 VecNormalize-style). Critical for stable value learning.")
    p.add_argument("--dino-bc-demos", type=str, default=None,
                   help="Path to expert demo .npz for Dino BC anchor. Adds a weighted "
                   "cross-entropy loss that prevents RL from forgetting duck/jump timing.")
    p.add_argument("--dino-bc-coef", type=float, default=0.5,
                   help="Weight of the Dino BC anchor loss (only with --dino-bc-demos).")
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
    p.add_argument(
        "--episode-end-pause",
        type=float,
        default=0.0,
        help="Seconds to pause on screen when an episode ends (visible training).",
    )
    p.add_argument(
        "--visible-rotate",
        action="store_true",
        help="When rendering, train only one game per update (round-robin). "
        "Avoids rapidly switching between three game windows each update.",
    )
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
    if parallel:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

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
    end_pause = args.episode_end_pause

    active_tasks = []
    if use_minigrid:
        active_tasks.append(TASK_MINIGRID)
    if use_dino:
        active_tasks.append(TASK_DINO)
    if use_carracing:
        active_tasks.append(TASK_CARRACING)
    visible_rotate = args.visible_rotate and len(active_tasks) > 1
    if visible_rotate:
        print(f"[visible-rotate] one game per update: {active_tasks}")

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

    if args.resume and args.normalize_carracing_reward and use_carracing:
        print("[normalize] reinitializing CarRacing critic for new reward scale")
        for m in ppo.net.carracing_critic:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    # Dino behavior cloning anchor
    bc_obs_t = bc_act_t = bc_w_t = None
    if args.dino_bc_demos:
        _d = np.load(args.dino_bc_demos)
        bc_obs_t = torch.tensor(_d["obs"], dtype=torch.float32, device=ppo.device)
        bc_act_t = torch.tensor(_d["actions"], dtype=torch.long, device=ppo.device)
        _counts = np.bincount(_d["actions"], minlength=DINO_N_ACTIONS).astype(np.float64)
        _w = _counts.sum() / (DINO_N_ACTIONS * np.maximum(_counts, 1))
        bc_w_t = torch.tensor(_w, dtype=torch.float32, device=ppo.device)
        print(f"[dino-bc] anchor: {len(bc_act_t)} demos, coef={args.dino_bc_coef}, "
              f"weights={np.round(_w, 2).tolist()}")

    mg_vec = dino_vec = cr_vec = None
    mg_buf = dino_buf = cr_buf = None
    mg_obs = dino_obs = cr_obs = None

    # MiniGrid curriculum state
    mg_stages = [int(s) for s in args.minigrid_stages.split(",") if s.strip()]
    mg_stage_idx = 0
    mg_ema_solve = 0.0
    mg_stage_updates = 0
    mg_env_id = args.minigrid_env_id
    mg_max_steps = None

    if use_minigrid:
        if args.minigrid_curriculum:
            size = mg_stages[mg_stage_idx]
            mg_env_id = doorkey_env_id(size)
            mg_max_steps = doorkey_stage_max_steps(size)
            print(f"[curriculum] MiniGrid stages={mg_stages} | start {mg_env_id} "
                  f"(max_steps={mg_max_steps}) | advance@solve_ema>={args.minigrid_advance_solve}")
        mg_vec = make_minigrid_vec_env(
            args.n_minigrid_envs,
            mg_env_id,
            args.seed,
            render=render_minigrid,
            episode_end_pause=end_pause,
            max_episode_steps=mg_max_steps,
        )
        mg_buf = RolloutBuffer(args.n_minigrid_envs)
        mg_obs, _ = mg_vec.reset(seed=args.seed)

    if use_dino:
        dino_vec = VecDinoGymEnv(
            n_envs=args.n_dino_envs,
            render=render_dino,
            parallel=parallel,
            episode_end_pause=end_pause,
        )
        dino_buf = RolloutBuffer(args.n_dino_envs)
        dino_obs, _ = dino_vec.reset(seed=args.seed)

    if use_carracing:
        cr_vec = make_carracing_vec_env(
            args.n_carracing_envs,
            seed=args.seed,
            render=render_carracing,
            max_episode_steps=args.carracing_max_episode_steps,
            no_progress_patience=args.carracing_no_progress_patience,
            episode_end_pause=end_pause,
            normalize_reward=args.normalize_carracing_reward,
            gamma=args.gamma,
        )
        cr_buf = RolloutBuffer(args.n_carracing_envs)
        cr_obs, cr_info = cr_vec.reset(seed=args.seed)
        cr_obs = _attach_aux(cr_obs, cr_info, args.n_carracing_envs, TASK_CARRACING)

    log_path = os.path.join(args.save_dir, "train_log.jsonl")
    best_path = os.path.join(args.save_dir, "best.pt")
    best_score = float("-inf")
    t0 = time.time()

    try:
        for update in range(1, args.updates + 1):
            if args.lr_schedule == "linear":
                frac = 1.0 - (update - 1) / max(1, args.updates)
                for pg in ppo.optim.param_groups:
                    pg["lr"] = args.lr * frac

            task_stats = {}
            run_only = None
            if visible_rotate:
                run_only = active_tasks[(update - 1) % len(active_tasks)]
                print(f"  [visible] update {update} -> {run_only}")

            if use_minigrid and (run_only is None or run_only == TASK_MINIGRID):
                mg_obs = _run_task(
                    ppo, args, TASK_MINIGRID, mg_vec, mg_buf, mg_obs,
                    mg_rollout, render_minigrid, task_stats
                )
                if args.minigrid_curriculum:
                    s = task_stats[TASK_MINIGRID]
                    sr = s.get("solve_rate", float("nan"))
                    if sr == sr:  # not NaN (episodes finished this update)
                        mg_ema_solve = 0.8 * mg_ema_solve + 0.2 * sr
                    s["ema_solve"] = mg_ema_solve
                    s["stage_size"] = mg_stages[mg_stage_idx]
                    mg_stage_updates += 1
                    can_advance = (
                        mg_stage_idx < len(mg_stages) - 1
                        and mg_stage_updates >= args.minigrid_min_stage_updates
                        and mg_ema_solve >= args.minigrid_advance_solve
                    )
                    if can_advance:
                        mg_stage_idx += 1
                        size = mg_stages[mg_stage_idx]
                        new_id = doorkey_env_id(size)
                        new_max = doorkey_stage_max_steps(size)
                        print(f"  [curriculum] solved {mg_env_id} (ema={mg_ema_solve:.2f}) "
                              f"-> advancing to {new_id} (max_steps={new_max})")
                        mg_vec.close()
                        mg_env_id = new_id
                        mg_max_steps = new_max
                        mg_vec = make_minigrid_vec_env(
                            args.n_minigrid_envs, mg_env_id, args.seed + mg_stage_idx,
                            render=render_minigrid, episode_end_pause=end_pause,
                            max_episode_steps=mg_max_steps,
                        )
                        mg_obs, _ = mg_vec.reset(seed=args.seed + mg_stage_idx)
                        mg_ema_solve = 0.0
                        mg_stage_updates = 0
            if use_dino and (run_only is None or run_only == TASK_DINO):
                dino_obs = _run_task(
                    ppo, args, TASK_DINO, dino_vec, dino_buf, dino_obs,
                    dino_rollout, render_dino, task_stats,
                    bc_obs=bc_obs_t, bc_actions=bc_act_t,
                    bc_coef=args.dino_bc_coef, bc_weight=bc_w_t,
                )
            if use_carracing and (run_only is None or run_only == TASK_CARRACING):
                cr_obs = _run_task(
                    ppo, args, TASK_CARRACING, cr_vec, cr_buf, cr_obs,
                    cr_rollout, render_carracing, task_stats
                )

            elapsed = time.time() - t0
            parts = [f"upd {update:4d}"]
            if TASK_MINIGRID in task_stats:
                s = task_stats[TASK_MINIGRID]
                if args.minigrid_curriculum:
                    parts.append(
                        f"mg[{s.get('stage_size', '?')}] ret {s['mean_return']:5.2f} "
                        f"solve {s.get('solve_rate', float('nan')):.2f} "
                        f"ema {s.get('ema_solve', 0):.2f}"
                    )
                else:
                    parts.append(f"mg ret {s['mean_return']:6.2f} H {s['entropy']:.2f}")
            if TASK_DINO in task_stats:
                s = task_stats[TASK_DINO]
                parts.append(
                    f"dino ret {s['mean_return']:6.2f} score {s.get('mean_score', 0):5.1f}"
                )
            if TASK_CARRACING in task_stats:
                s = task_stats[TASK_CARRACING]
                raw = s.get('mean_raw_return', s['mean_return'])
                parts.append(
                    f"car raw {raw:7.1f} ret {s['mean_return']:7.2f} "
                    f"H {s['entropy']:.2f} gn {s['grad_norm']:.1f}"
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

            # Save best checkpoint by CarRacing raw score once MiniGrid/Dino are
            # acceptable (>=0.9 solve EMA, >=200 score). No composite metric —
            # CarRacing <700 is the only bottleneck right now.
            DINO_ACCEPTABLE = 200
            MG_ACCEPTABLE = 0.9

            mg_ema = (
                task_stats[TASK_MINIGRID].get("ema_solve", 0)
                if TASK_MINIGRID in task_stats
                else MG_ACCEPTABLE
            )
            dino_score = (
                task_stats[TASK_DINO].get("mean_score", 0)
                if TASK_DINO in task_stats
                else DINO_ACCEPTABLE
            )
            car_raw = (
                task_stats[TASK_CARRACING].get("mean_raw_return")
                if TASK_CARRACING in task_stats
                else None
            )
            others_ok = mg_ema >= MG_ACCEPTABLE and dino_score >= DINO_ACCEPTABLE

            if (
                others_ok
                and car_raw is not None
                and car_raw == car_raw
                and car_raw > best_score
            ):
                best_score = float(car_raw)
                ppo.save(best_path)
                print(
                    f"  [best] new best (car_raw={car_raw:.1f}, "
                    f"mg={mg_ema:.2f} dino={dino_score:.0f}) saved to {best_path}"
                )

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
