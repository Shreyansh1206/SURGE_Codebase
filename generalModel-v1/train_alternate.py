
"""
Alternating multi-task training from scratch.

Cycle (repeat):
  1. MiniGrid PPO for N updates (Dino frozen) — DoorKey curriculum, starts first
  2. Dino PPO for M updates (MiniGrid frozen)
  3. MiniGrid again ...
  4. Dino again ...

MiniGrid curriculum (within each MiniGrid block):
  Sizes: 5, 6, 7, 8, 10, 12, 14, 16
  Advance when solve-rate EMA >= 80% AND at least 10 updates on current size.

Usage:
  python train_alternate.py --parallel --no-auto-resume
  python train_alternate.py --parallel   # resume
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from datetime import datetime, timezone

import gymnasium as gym
import numpy as np
import torch

from envs.dino_gym import DINO_N_ACTIONS, DINO_OBS_DIM, VecDinoGymEnv
from envs.minigrid_env import make_minigrid_env, minigrid_obs_dim
from eval_final import run_dino_eval, run_minigrid_eval
from multi_task_ppo import TASK_DINO, TASK_MINIGRID, MultiTaskPPO, RolloutBuffer
from train import collect_rollout

DEFAULT_SAVE_DIR = "checkpoints_alternate"
PIPELINE_STATE = "pipeline_state.json"
BLOCK_SUMMARIES = "block_summaries.json"
MG_STAGES = (5, 6, 7, 8, 10, 12, 14, 16)
SOLVE_RETURN_THRESHOLD = 0.5
EMA_ALPHA = 0.2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _task_for_block(block_idx: int) -> str:
    return "minigrid" if block_idx % 2 == 0 else "dino"


def _block_updates_for_task(args, task: str) -> int:
    return args.mg_block_updates if task == "minigrid" else args.dino_block_updates


def _freeze_task(net, task: str) -> list:
    dino_prefixes = ("dino_encoder", "dino_actor", "dino_critic")
    mg_prefixes = ("minigrid_encoder", "minigrid_actor", "minigrid_critic")
    frozen = dino_prefixes if task == "minigrid" else mg_prefixes
    trainable = []
    for name, param in net.named_parameters():
        freeze = name.startswith(frozen)
        param.requires_grad_(not freeze)
        if not freeze:
            trainable.append(param)
    return trainable


def _rebuild_optimizer(ppo: MultiTaskPPO, trainable, lr: float) -> None:
    ppo.optim = torch.optim.Adam(trainable, lr=lr)


def _try_load_optim(ppo: MultiTaskPPO, ckpt_path: str) -> bool:
    ckpt = torch.load(ckpt_path, map_location=ppo.device)
    if "optim" not in ckpt:
        return False
    try:
        ppo.optim.load_state_dict(ckpt["optim"])
        return True
    except ValueError:
        print("[warn] optimizer state mismatch — fresh optimizer for frozen setup")
        return False


def _load_pipeline(save_dir: str) -> dict:
    path = os.path.join(save_dir, PIPELINE_STATE)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_pipeline(save_dir: str, state: dict) -> None:
    with open(os.path.join(save_dir, PIPELINE_STATE), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _append_block_summary(save_dir: str, record: dict) -> None:
    path = os.path.join(save_dir, BLOCK_SUMMARIES)
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
    rows.append(record)
    with open(path, encoding="utf-8", mode="w") as f:
        json.dump(rows, f, indent=2)


def _stage_max_steps(size: int) -> int:
    return max(80, 12 * size)


def _make_mg_vec(size: int, n_envs: int, seed: int):
    env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
    max_steps = _stage_max_steps(size)

    def _factory():
        return make_minigrid_env(env_id, max_episode_steps=max_steps, anti_stall=True)

    vec = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
    vec.reset(seed=seed)
    return vec, env_id


def _quiet_eval(fn, *args, **kwargs) -> dict:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _eval_dino(ppo: MultiTaskPPO, n_episodes: int) -> dict:
    result = _quiet_eval(run_dino_eval, ppo, n_episodes=n_episodes)
    s = result["summary"]
    return {
        "task": "dino",
        "n_episodes": s["n_episodes"],
        "score_mean": s["score_mean"],
        "score_max": s["score_max"],
        "score_min": s["score_min"],
        "return_mean": s["return_mean"],
        "passes_per_episode": s["passes_per_episode"],
        "entropy_mean": s["entropy_mean"],
        "duck_frac": s.get("action_distribution", {}).get("duck", 0.0),
        "full": result,
    }


def _eval_minigrid(ppo: MultiTaskPPO, sizes: tuple[int, ...], n_seeds: int) -> dict:
    result = _quiet_eval(
        run_minigrid_eval, ppo, sizes=sizes, n_seeds=n_seeds, verbose=False
    )
    ovr = result["overall"]
    return {
        "task": "minigrid",
        "sizes": list(sizes),
        "n_episodes": ovr["total_episodes"],
        "solve_rate": ovr["solve_rate"],
        "loop_fail_rate": ovr["loop_fail_rate"],
        "return_mean": ovr["return_mean"],
        "length_mean": ovr["length_mean"],
        "entropy_mean": ovr["entropy_mean"],
        "per_size": result.get("per_size", []),
        "full": result,
    }


def _save_ckpt(ppo: MultiTaskPPO, path: str) -> None:
    ppo.save(path)


def _setup_agent(mg_dim: int, args) -> MultiTaskPPO:
    return MultiTaskPPO(
        minigrid_dim=mg_dim,
        dino_dim=DINO_OBS_DIM,
        minigrid_actions=7,
        dino_actions=DINO_N_ACTIONS,
        lr=args.lr,
        clip_eps=args.clip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy,
    )


def _load_agent_for_resume(ppo: MultiTaskPPO, ckpt_path: str, task: str, lr: float) -> None:
    ppo.load(ckpt_path, load_optim=False)
    trainable = _freeze_task(ppo.net, task)
    _rebuild_optimizer(ppo, trainable, lr)
    _try_load_optim(ppo, ckpt_path)


def _block_dir(save_dir: str, block_idx: int, task: str) -> str:
    d = os.path.join(save_dir, "blocks", f"block_{block_idx:03d}_{task}")
    os.makedirs(d, exist_ok=True)
    return d


def _log_train(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _finalize_block(
    ppo: MultiTaskPPO,
    save_dir: str,
    block_idx: int,
    task: str,
    pipeline: dict,
    args,
    mg_stage_idx: int,
    interrupted: bool = False,
) -> dict:
    """Run end-of-block eval, write summary files, return summary record."""
    print(f"\n[eval] block {block_idx} ({task}) — running evaluation...")
    t0 = time.time()

    if task == "dino":
        eval_summary = _eval_dino(ppo, n_episodes=args.eval_dino_episodes)
    else:
        sizes_trained = MG_STAGES[: mg_stage_idx + 1]
        eval_summary = _eval_minigrid(
            ppo, sizes=sizes_trained, n_seeds=args.eval_mg_seeds
        )

    eval_wall = time.time() - t0
    block_dir = _block_dir(save_dir, block_idx, task)

    record = {
        "block_index": block_idx,
        "task": task,
        "block_updates_target": _block_updates_for_task(args, task),
        "block_updates_done": pipeline.get(
            "block_update", _block_updates_for_task(args, task)
        ),
        "interrupted": interrupted,
        "finished_at": _utc_now(),
        "eval_wall_s": eval_wall,
        "mg_stage_idx": mg_stage_idx,
        "mg_stage_size": MG_STAGES[mg_stage_idx] if task == "minigrid" else None,
        "eval": {k: v for k, v in eval_summary.items() if k != "full"},
    }

    summary_path = os.path.join(block_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    full_eval_path = os.path.join(block_dir, "eval_full.json")

    def _serialise(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    with open(full_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary.get("full", {}), f, indent=2, default=_serialise)

    _append_block_summary(save_dir, record)

    if task == "dino":
        print(
            f"[eval] Dino: mean score {eval_summary['score_mean']:.1f} | "
            f"max {eval_summary['score_max']} | duck {eval_summary['duck_frac']:.3f}"
        )
    else:
        print(
            f"[eval] MiniGrid sizes {sizes_trained}: solve "
            f"{eval_summary['solve_rate']*100:.1f}% | "
            f"return {eval_summary['return_mean']:+.3f}"
        )
    print(f"[eval] saved -> {summary_path}")
    return record


def _train_dino_block(
    ppo: MultiTaskPPO,
    args,
    block_idx: int,
    start_update: int,
    updates_to_run: int,
    log_path: str,
    save_dir: str,
    pipeline: dict,
) -> int:
    trainable = _freeze_task(ppo.net, "dino")
    _rebuild_optimizer(ppo, trainable, args.dino_lr)
    ppo.batch_size = args.batch_size
    ppo.net.train()

    parallel = args.parallel and not args.render_dino
    n_envs = 1 if args.render_dino else args.n_dino_envs
    dino_vec = VecDinoGymEnv(n_envs=n_envs, render=args.render_dino, parallel=parallel)
    buf = RolloutBuffer(n_envs)
    obs, _ = dino_vec.reset(seed=args.seed + block_idx)

    end = start_update + updates_to_run
    try:
        for upd in range(start_update + 1, end + 1):
            buf.clear()
            t_roll = time.time()
            obs, last_v, ep_ret, _, ep_sc = collect_rollout(
                dino_vec,
                ppo,
                buf,
                args.dino_rollout,
                ppo.device,
                obs,
                TASK_DINO,
                render=args.render_dino,
                render_delay=args.render_delay if args.render_dino else 0.0,
            )
            roll_t = time.time() - t_roll
            stats = ppo.update_task(
                TASK_DINO, buf, last_v, gamma=args.gamma, lam=args.lam
            )
            mean_score = float(np.mean(ep_sc)) if ep_sc else float("nan")
            max_score = int(np.max(ep_sc)) if ep_sc else 0
            print(
                f"  [dino] upd {upd:3d}/{end} | score {mean_score:6.1f} "
                f"(max {max_score}) | pi {stats['pi_loss']:+.3f} | "
                f"roll {roll_t:.1f}s"
            )
            _log_train(
                log_path,
                {
                    "block": block_idx,
                    "task": "dino",
                    "update": upd,
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "stats": stats,
                },
            )
            if upd % args.save_every == 0:
                _save_ckpt(ppo, os.path.join(save_dir, "latest.pt"))
                pipeline["block_update"] = upd
                _save_pipeline(save_dir, pipeline)
    finally:
        dino_vec.close()

    return end


def _train_minigrid_block(
    ppo: MultiTaskPPO,
    args,
    block_idx: int,
    start_update: int,
    updates_to_run: int,
    log_path: str,
    save_dir: str,
    pipeline: dict,
) -> tuple[int, int, float]:
    trainable = _freeze_task(ppo.net, "minigrid")
    _rebuild_optimizer(ppo, trainable, args.mg_lr)
    ppo.batch_size = args.mg_batch_size
    ppo.net.train()

    stage_idx = int(pipeline.get("mg_stage_idx", 0))
    stage_idx = min(stage_idx, len(MG_STAGES) - 1)
    ema_solve = float(pipeline.get("mg_ema_solve", 0.0))
    stage_updates = int(pipeline.get("mg_stage_updates", 0))

    end = start_update + updates_to_run
    vec = None
    buf = None
    obs = None
    current_size = MG_STAGES[stage_idx]

    try:
        for upd in range(start_update + 1, end + 1):
            size = MG_STAGES[stage_idx]
            if vec is None or size != current_size:
                if vec is not None:
                    vec.close()
                current_size = size
                vec, env_id = _make_mg_vec(
                    size, args.n_mg_envs, args.seed + block_idx * 100 + stage_idx
                )
                buf = RolloutBuffer(args.n_mg_envs)
                obs, _ = vec.reset(seed=args.seed + block_idx * 100 + stage_idx)
                print(f"  [minigrid] stage {stage_idx + 1}/{len(MG_STAGES)}: {env_id}")

            buf.clear()
            obs, last_v, ep_ret, ep_len, _ = collect_rollout(
                vec, ppo, buf, args.mg_rollout, ppo.device, obs, TASK_MINIGRID
            )
            stats = ppo.update_task(
                TASK_MINIGRID, buf, last_v, gamma=args.gamma, lam=args.lam
            )

            if ep_ret:
                mean_ret = float(np.mean(ep_ret))
                mean_len = float(np.mean(ep_len))
                solve = float(np.mean([r >= SOLVE_RETURN_THRESHOLD for r in ep_ret]))
                ema_solve = (1 - EMA_ALPHA) * ema_solve + EMA_ALPHA * solve
            else:
                mean_ret, mean_len, solve = float("nan"), float("nan"), 0.0

            stage_updates += 1
            print(
                f"  [mg {size}x{size}] upd {upd:3d}/{end} | ret {mean_ret:6.2f} | "
                f"solve {solve:.2f} ema {ema_solve:.2f} | stage_upd {stage_updates} | "
                f"H {stats['entropy']:.2f}"
            )
            _log_train(
                log_path,
                {
                    "block": block_idx,
                    "task": "minigrid",
                    "update": upd,
                    "size": size,
                    "stage_idx": stage_idx,
                    "stage_updates": stage_updates,
                    "mean_return": mean_ret,
                    "solve_rate": solve,
                    "ema_solve": ema_solve,
                    "stats": stats,
                },
            )

            pipeline["mg_stage_idx"] = stage_idx
            pipeline["mg_ema_solve"] = ema_solve
            pipeline["mg_stage_updates"] = stage_updates
            pipeline["block_update"] = upd
            _save_pipeline(save_dir, pipeline)
            _save_ckpt(ppo, os.path.join(save_dir, "latest.pt"))

            if (
                ema_solve >= args.advance_solve
                and stage_updates >= args.min_stage_updates
                and stage_idx < len(MG_STAGES) - 1
            ):
                print(
                    f"  -> advancing {size}x{size} -> "
                    f"{MG_STAGES[stage_idx + 1]}x{MG_STAGES[stage_idx + 1]} "
                    f"(ema {ema_solve:.2f})"
                )
                stage_idx += 1
                ema_solve = 0.0
                stage_updates = 0
                vec.close()
                vec = None

            if upd % args.save_every == 0:
                _save_ckpt(ppo, os.path.join(save_dir, "latest.pt"))
    finally:
        if vec is not None:
            vec.close()

    pipeline["mg_stage_idx"] = stage_idx
    pipeline["mg_ema_solve"] = ema_solve
    pipeline["mg_stage_updates"] = stage_updates
    return end, stage_idx, ema_solve


def main() -> None:
    p = argparse.ArgumentParser(description="Alternating Dino/MiniGrid training from scratch")
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--no-auto-resume", action="store_true")
    p.add_argument("--blocks", type=int, default=0,
                   help="Total alternating blocks to run (0 = infinite until Ctrl+C)")
    p.add_argument("--mg-block-updates", type=int, default=500,
                   help="PPO updates per MiniGrid block")
    p.add_argument("--dino-block-updates", type=int, default=169,
                   help="PPO updates per Dino block")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-dino-envs", type=int, default=4)
    p.add_argument("--dino-rollout", type=int, default=512)
    p.add_argument("--dino-lr", type=float, default=2e-4)
    p.add_argument("--parallel", action="store_true")
    p.add_argument("--render-dino", action="store_true")

    p.add_argument("--n-mg-envs", type=int, default=8)
    p.add_argument("--mg-rollout", type=int, default=256)
    p.add_argument("--mg-lr", type=float, default=3e-4)
    p.add_argument("--mg-batch-size", type=int, default=256,
                   help="PPO batch size during MiniGrid blocks")
    p.add_argument("--advance-solve", type=float, default=0.8)
    p.add_argument("--min-stage-updates", type=int, default=10)

    p.add_argument("--lr", type=float, default=3e-4, help="Fallback LR for agent init")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--mg-entropy", type=float, default=0.03)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--render-delay", type=float, default=0.012)

    p.add_argument("--eval-dino-episodes", type=int, default=20)
    p.add_argument("--eval-mg-seeds", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    latest = os.path.join(args.save_dir, "latest.pt")
    pipeline = _load_pipeline(args.save_dir) if not args.no_auto_resume else {}
    resuming = bool(pipeline) and os.path.isfile(latest)

    mg_dim = minigrid_obs_dim(f"MiniGrid-DoorKey-{MG_STAGES[0]}x{MG_STAGES[0]}-v0")
    ppo = _setup_agent(mg_dim, args)

    if resuming:
        block_idx = int(pipeline.get("block_index", 0))
        task = pipeline.get("task", _task_for_block(block_idx))
        block_update = int(pipeline.get("block_update", 0))
        lr = args.dino_lr if task == "dino" else args.mg_lr
        print(f"[resume] block {block_idx} ({task}) update {block_update} <- {latest}")
        _load_agent_for_resume(ppo, latest, task, lr)
    else:
        block_idx = 0
        task = _task_for_block(0)
        block_update = 0
        pipeline = {
            "block_index": 0,
            "task": task,
            "block_update": 0,
            "mg_stage_idx": 0,
            "mg_stage_updates": 0,
            "mg_ema_solve": 0.0,
            "started_at": _utc_now(),
        }
        _save_pipeline(args.save_dir, pipeline)
        print(
            f"[fresh] alternating training | block 0 = {task} | "
            f"mg={args.mg_block_updates} dino={args.dino_block_updates} updates/block"
        )

    pipeline.setdefault("evaluated_blocks", [])
    target_blocks = args.blocks

    blocks_done = len(pipeline["evaluated_blocks"])

    try:
        while True:
            if target_blocks > 0 and blocks_done >= target_blocks:
                print(f"[done] completed {target_blocks} blocks")
                break

            task = _task_for_block(block_idx)
            pipeline["block_index"] = block_idx
            pipeline["task"] = task
            block_target = _block_updates_for_task(args, task)
            remaining = block_target - block_update

            if remaining <= 0:
                if block_idx not in pipeline["evaluated_blocks"]:
                    mg_stage_idx = int(pipeline.get("mg_stage_idx", 0))
                    _finalize_block(
                        ppo, args.save_dir, block_idx, task, pipeline, args, mg_stage_idx
                    )
                    pipeline["evaluated_blocks"].append(block_idx)
                    blocks_done = len(pipeline["evaluated_blocks"])
                    _save_pipeline(args.save_dir, pipeline)
                block_idx += 1
                block_update = 0
                pipeline["block_index"] = block_idx
                pipeline["block_update"] = 0
                if target_blocks > 0 and block_idx >= target_blocks:
                    break
                continue

            print(
                f"\n{'=' * 72}\n"
                f"  BLOCK {block_idx} — {task.upper()} "
                f"({remaining}/{block_target} updates remaining)\n"
                f"{'=' * 72}"
            )

            block_dir = _block_dir(args.save_dir, block_idx, task)
            log_path = os.path.join(block_dir, "train_log.jsonl")

            if task == "dino":
                ppo.entropy_coef = args.entropy
                block_update = _train_dino_block(
                    ppo, args, block_idx, block_update, remaining, log_path, args.save_dir,
                    pipeline,
                )
                mg_stage_idx = int(pipeline.get("mg_stage_idx", 0))
            else:
                ppo.entropy_coef = args.mg_entropy
                block_update, mg_stage_idx, ema_solve = _train_minigrid_block(
                    ppo, args, block_idx, block_update, remaining, log_path,
                    args.save_dir, pipeline,
                )
                pipeline["mg_ema_solve"] = ema_solve

            pipeline["block_update"] = block_update
            _save_ckpt(ppo, os.path.join(args.save_dir, "latest.pt"))
            _save_pipeline(args.save_dir, pipeline)

            if block_update >= block_target:
                if block_idx not in pipeline["evaluated_blocks"]:
                    _finalize_block(
                        ppo, args.save_dir, block_idx, task, pipeline, args, mg_stage_idx
                    )
                    pipeline["evaluated_blocks"].append(block_idx)
                    blocks_done = len(pipeline["evaluated_blocks"])
                block_idx += 1
                block_update = 0
                pipeline["block_index"] = block_idx
                pipeline["block_update"] = 0
                pipeline["task"] = _task_for_block(block_idx)
                _save_pipeline(args.save_dir, pipeline)
                _save_ckpt(ppo, os.path.join(args.save_dir, "latest.pt"))

                if target_blocks > 0 and blocks_done >= target_blocks:
                    print(f"[done] completed {target_blocks} blocks")
                    break

    except KeyboardInterrupt:
        print("\n[interrupt] saving + evaluating current block...")
        task = pipeline.get("task", _task_for_block(block_idx))
        mg_stage_idx = int(pipeline.get("mg_stage_idx", 0))
        _save_ckpt(ppo, os.path.join(args.save_dir, "latest.pt"))
        if block_idx not in pipeline.get("evaluated_blocks", []):
            _finalize_block(
                ppo, args.save_dir, block_idx, task, pipeline, args, mg_stage_idx,
                interrupted=True,
            )
            pipeline.setdefault("evaluated_blocks", []).append(block_idx)
        _save_pipeline(args.save_dir, pipeline)
        print("[saved] resume with: python train_alternate.py")
    except Exception:
        _save_ckpt(ppo, os.path.join(args.save_dir, "latest.pt"))
        _save_pipeline(args.save_dir, pipeline)
        raise
    finally:
        _save_ckpt(ppo, os.path.join(args.save_dir, "latest.pt"))


if __name__ == "__main__":
    main()
