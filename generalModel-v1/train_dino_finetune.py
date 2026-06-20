
"""
Further-train the Dino pathway from checkpoints_final/final.pt (multi-task model).

Features:
  - Dino-only PPO (MiniGrid weights frozen by default)
  - Auto-resume from {save_dir}/latest.pt + train_state.json
  - Saves latest.pt on every exit (Ctrl+C, crash, normal end)
  - Promotes best_dino.pt when rollout mean score improves
  - Optional on-screen pygame window (--render-dino)
  - Optional periodic greedy eval (--eval-every)

Usage:
  # Fast headless fine-tune (4 parallel Dino games by default)
  python train_dino_finetune.py --updates 300

  # Watch one game on screen (slower; disables parallel)
  python train_dino_visible.py

  # Resume after interruption (automatic if latest.pt exists in save-dir)
  python train_dino_finetune.py --parallel --updates 200

  # Force fresh start from final.pt (ignore save-dir checkpoints)
  python train_dino_finetune.py --no-auto-resume --parallel --updates 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from envs.dino_gym import DINO_N_ACTIONS, DINO_OBS_DIM, VecDinoGymEnv
from envs.minigrid_env import minigrid_obs_dim
from multi_task_ppo import TASK_DINO, MultiTaskPPO, RolloutBuffer
from train import collect_rollout

DEFAULT_INIT = "checkpoints_final/final.pt"
DEFAULT_SAVE_DIR = "checkpoints_dino_finetune"
DEFAULT_N_DINO_ENVS = 4
STATE_FILE = "train_state.json"
LOG_FILE = "train_log.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_path(save_dir: str) -> str:
    return os.path.join(save_dir, STATE_FILE)


def _load_state(save_dir: str) -> dict:
    path = _state_path(save_dir)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_state(save_dir: str, state: dict) -> None:
    with open(_state_path(save_dir), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _resolve_resume(save_dir: str, init_ckpt: str, no_auto_resume: bool) -> tuple[str, bool]:
    """Return (checkpoint_path, is_resume)."""
    latest = os.path.join(save_dir, "latest.pt")
    if not no_auto_resume and os.path.isfile(latest):
        return latest, True
    if not os.path.isfile(init_ckpt):
        raise FileNotFoundError(
            f"Init checkpoint not found: {init_ckpt}\n"
            f"Place final.pt at checkpoints_final/final.pt or pass --init <path>."
        )
    return init_ckpt, False


def _freeze_minigrid(net, freeze_core: bool = False) -> list:
    """Freeze MiniGrid pathway; return trainable parameter list."""
    frozen_prefixes = ("minigrid_encoder", "minigrid_actor", "minigrid_critic")
    trainable = []
    for name, param in net.named_parameters():
        freeze = name.startswith(frozen_prefixes)
        if freeze_core and name.startswith("shared_core"):
            freeze = True
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
        print(
            "[warn] optimizer state incompatible with frozen MiniGrid setup — "
            "using fresh optimizer (network weights still resumed)"
        )
        return False


@torch.no_grad()
def eval_dino_greedy(ppo: MultiTaskPPO, n_eps: int = 10, seed_start: int = 20_000) -> dict:
    from dino_env import DinoEnv

    ppo.net.eval()
    scores, ducks = [], []
    for ep in range(n_eps):
        env = DinoEnv(render=False, frames_per_step=4, seed=seed_start + ep)
        obs = env.reset()
        done = False
        last = 0
        duck_n = 0
        guard = 0
        while not done and guard < 25_000:
            logits, _ = ppo.net(
                torch.from_numpy(obs).float().unsqueeze(0).to(ppo.device), TASK_DINO
            )
            action = int(logits.argmax(-1).item())
            if action == 2:
                duck_n += 1
            obs, _, done, info = env.step(action)
            last = int(info.get("score", last))
            guard += 1
        scores.append(last)
        ducks.append(duck_n / max(guard, 1))
        env.close()
    ppo.net.train()
    return {
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "duck_rate": float(np.mean(ducks)),
        "n_eps": n_eps,
    }


def _append_log(save_dir: str, record: dict) -> None:
    with open(os.path.join(save_dir, LOG_FILE), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _save_all(
    ppo: MultiTaskPPO,
    save_dir: str,
    state: dict,
    tag: str | None = None,
) -> None:
    ppo.save(os.path.join(save_dir, "latest.pt"))
    _save_state(save_dir, state)
    if tag:
        ppo.save(os.path.join(save_dir, tag))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Dino-only fine-tune from checkpoints_final/final.pt"
    )
    p.add_argument("--init", type=str, default=DEFAULT_INIT,
                   help="Starting weights when save-dir has no latest.pt")
    p.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--no-auto-resume", action="store_true",
                   help="Ignore save-dir/latest.pt; always load --init")
    p.add_argument("--resume", type=str, default=None,
                   help="Explicit checkpoint (overrides auto-resume)")
    p.add_argument("--updates", type=int, default=300,
                   help="Number of PPO updates to run this session")
    p.add_argument(
        "--n-dino-envs",
        type=int,
        default=DEFAULT_N_DINO_ENVS,
        help=f"Number of simultaneous Dino games (default: {DEFAULT_N_DINO_ENVS})",
    )
    p.add_argument("--dino-rollout", type=int, default=512,
                   help="Steps per env per update (each step = 4 game frames)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25,
                   help="Greedy eval episodes (0 = disable)")
    p.add_argument("--eval-eps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-parallel",
        action="store_true",
        help="Use a single Dino game (slower; for debugging)",
    )
    p.add_argument("--render-dino", action="store_true",
                   help="Show one pygame window (forces single env, no parallel)")
    p.add_argument("--render-delay", type=float, default=0.012,
                   help="Pause after each rendered step (seconds)")
    p.add_argument("--freeze-core", action="store_true",
                   help="Also freeze shared_core (train Dino head only)")
    p.add_argument("--unfreeze-minigrid", action="store_true",
                   help="Allow MiniGrid weights to update (not recommended)")
    p.add_argument("--dino-bc-demos", type=str, default=None,
                   help="Expert demo .npz for BC anchor during PPO")
    p.add_argument("--dino-bc-coef", type=float, default=0.3)
    args = p.parse_args()

    parallel = not args.no_parallel and not args.render_dino
    if args.render_dino:
        if args.n_dino_envs != DEFAULT_N_DINO_ENVS:
            print(f"[render] using 1 visible env (ignoring --n-dino-envs {args.n_dino_envs})")
        args.n_dino_envs = 1
        parallel = False

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.resume:
        ckpt_path, is_resume = args.resume, True
    else:
        ckpt_path, is_resume = _resolve_resume(
            args.save_dir, args.init, args.no_auto_resume
        )

    state = _load_state(args.save_dir) if is_resume else {}
    start_update = int(state.get("update", 0))
    best_mean = float(state.get("best_mean_score", -1.0))
    best_max = int(state.get("best_max_score", 0))

    mg_dim = minigrid_obs_dim("MiniGrid-DoorKey-16x16-v0")
    ppo = MultiTaskPPO(
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

    print(f"[dino-finetune] loading {'resume' if is_resume else 'init'}: {ckpt_path}")
    # Load weights only; rebuild optimizer after freezing so param groups match.
    ppo.load(ckpt_path, load_optim=False)

    if not args.unfreeze_minigrid:
        trainable = _freeze_minigrid(ppo.net, freeze_core=args.freeze_core)
        n_train = sum(pm.numel() for pm in trainable)
        print(
            f"[dino-finetune] frozen MiniGrid pathway"
            f"{' + shared_core' if args.freeze_core else ''} | "
            f"trainable params: {n_train:,}"
        )
        _rebuild_optimizer(ppo, trainable, args.lr)
        if is_resume:
            _try_load_optim(ppo, ckpt_path)

    bc_obs_t = bc_act_t = bc_w_t = None
    if args.dino_bc_demos and os.path.isfile(args.dino_bc_demos):
        demos = np.load(args.dino_bc_demos)
        bc_obs_t = torch.tensor(demos["obs"], dtype=torch.float32, device=ppo.device)
        bc_act_t = torch.tensor(demos["actions"], dtype=torch.long, device=ppo.device)
        counts = np.bincount(demos["actions"], minlength=DINO_N_ACTIONS).astype(np.float64)
        weights = counts.sum() / (DINO_N_ACTIONS * np.maximum(counts, 1))
        bc_w_t = torch.tensor(weights, dtype=torch.float32, device=ppo.device)
        print(f"[dino-bc] anchor: {len(bc_act_t)} demos, coef={args.dino_bc_coef}")

    samples_per_update = args.dino_rollout * args.n_dino_envs
    if parallel:
        mode = "parallel (multiprocess)"
        instance_msg = (
            f"{args.n_dino_envs} headless Dino games in separate processes"
        )
    elif args.render_dino:
        mode = "visible (single window)"
        instance_msg = "1 Dino game with on-screen pygame window"
    else:
        mode = "sync (single process)"
        instance_msg = f"{args.n_dino_envs} Dino game(s) in one process"

    print(
        f"[dino-finetune] {instance_msg} | mode={mode} | "
        f"steps/update={args.dino_rollout}×{args.n_dino_envs}={samples_per_update} "
        f"(×4 game frames/step) | session updates={args.updates} | "
        f"resume from upd {start_update}"
    )

    dino_vec = VecDinoGymEnv(
        n_envs=args.n_dino_envs,
        render=args.render_dino,
        parallel=parallel,
    )
    dino_buf = RolloutBuffer(args.n_dino_envs)
    dino_obs, _ = dino_vec.reset(seed=args.seed)

    if not state:
        state = {
            "init_ckpt": args.init,
            "started_at": _utc_now(),
            "update": 0,
            "best_mean_score": best_mean,
            "best_max_score": best_max,
        }

    t0 = time.time()
    end_update = start_update + args.updates
    interrupted = False

    try:
        for update in range(start_update + 1, end_update + 1):
            dino_buf.clear()
            t_roll = time.time()
            dino_obs, dino_last_v, dino_ret, dino_len, dino_sc = collect_rollout(
                dino_vec,
                ppo,
                dino_buf,
                args.dino_rollout,
                ppo.device,
                dino_obs,
                TASK_DINO,
                render=args.render_dino,
                render_delay=args.render_delay if args.render_dino else 0.0,
            )
            roll_t = time.time() - t_roll

            t_upd = time.time()
            stats = ppo.update_task(
                TASK_DINO,
                dino_buf,
                dino_last_v,
                gamma=args.gamma,
                lam=args.lam,
                bc_obs=bc_obs_t,
                bc_actions=bc_act_t,
                bc_coef=args.dino_bc_coef,
                bc_weight=bc_w_t,
            )
            upd_t = time.time() - t_upd

            mean_ret = float(np.mean(dino_ret)) if dino_ret else float("nan")
            mean_score = float(np.mean(dino_sc)) if dino_sc else float("nan")
            max_score = int(np.max(dino_sc)) if dino_sc else 0

            eval_stats = None
            if args.eval_every > 0 and update % args.eval_every == 0:
                eval_stats = eval_dino_greedy(ppo, n_eps=args.eval_eps)
                print(
                    f"  [eval] greedy mean={eval_stats['mean_score']:.1f} "
                    f"max={eval_stats['max_score']} duck={eval_stats['duck_rate']:.3f}"
                )

            promoted = False
            score_for_best = (
                eval_stats["mean_score"] if eval_stats else mean_score
            )
            if score_for_best > best_mean:
                best_mean = score_for_best
                best_max = max(best_max, max_score)
                ppo.save(os.path.join(args.save_dir, "best_dino.pt"))
                promoted = True

            elapsed = time.time() - t0
            flag = " *best*" if promoted else ""
            print(
                f"upd {update:4d} | score {mean_score:6.1f} (max {max_score:4d}) | "
                f"ret {mean_ret:6.2f} | pi {stats['pi_loss']:+.3f} "
                f"v {stats['v_loss']:.3f} H {stats['entropy']:.3f} | "
                f"roll {roll_t:.1f}s upd {upd_t:.1f}s | {elapsed:.0f}s{flag}"
            )

            state.update(
                {
                    "update": update,
                    "best_mean_score": best_mean,
                    "best_max_score": best_max,
                    "last_mean_score": mean_score,
                    "last_max_score": max_score,
                    "last_ckpt": ckpt_path,
                    "updated_at": _utc_now(),
                }
            )

            record = {
                "update": update,
                "elapsed": elapsed,
                "mean_return": mean_ret,
                "mean_score": mean_score,
                "max_score": max_score,
                "episodes": len(dino_ret),
                "stats": stats,
                "roll_time": roll_t,
                "upd_time": upd_t,
                "eval": eval_stats,
                "promoted_best": promoted,
            }
            _append_log(args.save_dir, record)

            if update % args.save_every == 0:
                _save_all(
                    ppo,
                    args.save_dir,
                    state,
                    tag=f"mt_ppo_upd{update}.pt",
                )
            else:
                _save_all(ppo, args.save_dir, state)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] saving checkpoint — rerun the same command to resume")
    except Exception as exc:
        print(f"\n[error] {type(exc).__name__}: {exc}")
        print("[error] saving checkpoint before exit — rerun to resume")
        raise
    finally:
        _save_all(ppo, args.save_dir, state)
        dino_vec.close()
        if interrupted:
            print(f"[saved] {args.save_dir}/latest.pt (update {state.get('update', 0)})")
        else:
            print(
                f"[done] {args.updates} updates | best mean score {best_mean:.1f} | "
                f"checkpoints in {args.save_dir}/"
            )


if __name__ == "__main__":
    main()
