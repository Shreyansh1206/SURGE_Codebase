
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from dino_env import (
    N_ACTIONS as DINO_ACTIONS,
    OBS_DIM as DINO_OBS_DIM,
    DinoEnv,
)
from envs.minigrid_env import make_minigrid_env, minigrid_obs_dim
from multi_task_ppo import TASK_DINO, TASK_MINIGRID, MultiTaskPPO

DEFAULT_CKPT = "checkpoints_final/final.pt"
DEFAULT_MG_SIZES: Tuple[int, ...] = (5, 6, 8, 10, 12, 14, 16)
DEFAULT_MG_SEEDS = 60
DEFAULT_MG_SEED_START = 500
DEFAULT_MG_MAX_STEPS = 300
DEFAULT_DINO_EPISODES = 20
DEFAULT_BENCH_ITERS = 300
DEFAULT_BENCH_WARMUP = 50
BENCH_BATCH_SIZES = (1, 8, 32, 128)

FF_BASELINE_SOLVED = 0.93
FF_BASELINE_LOOPS = 0.07

LOOP_WINDOW = 40
LOOP_UNIQUE_MAX = 6

MINIGRID_ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
MINIGRID_N_ACTIONS = 7
DINO_ACTION_NAMES = ["noop", "jump", "duck"]


def _sep(char: str = "─", width: int = 72) -> str:
    return char * width


def _header(title: str, width: int = 72) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _subheader(title: str, width: int = 72) -> None:
    print(f"\n  {'─' * (width - 2)}")
    print(f"  {title}")
    print(f"  {'─' * (width - 2)}")


def _fmt_num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _fmt_bytes(n: int) -> str:
    for unit, scale in [("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)]:
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n} B"


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _percentile(arr: List[float], p: float) -> float:
    return float(np.percentile(arr, p)) if arr else float("nan")


def _param_count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())


def _trainable_count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def load_agent(ckpt_path: str) -> Tuple[MultiTaskPPO, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Expected at: {os.path.abspath(ckpt_path)}"
        )
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mg_dim = int(raw.get("minigrid_dim", minigrid_obs_dim("MiniGrid-DoorKey-16x16-v0")))
    dino_dim = int(raw.get("dino_dim", DINO_OBS_DIM))
    agent = MultiTaskPPO(minigrid_dim=mg_dim, dino_dim=dino_dim)
    agent.load(ckpt_path, load_optim=False)
    agent.net.eval()
    meta: Dict[str, Any] = {
        "ckpt_path": os.path.abspath(ckpt_path),
        "ckpt_size_bytes": os.path.getsize(ckpt_path),
        "minigrid_dim": mg_dim,
        "dino_dim": dino_dim,
        "device": str(agent.device),
    }
    return agent, meta


def model_summary(agent: MultiTaskPPO, meta: Dict[str, Any]) -> Dict[str, Any]:
    net = agent.net
    _header("MODEL SUMMARY")

    print(f"  Checkpoint  : {meta['ckpt_path']}")
    print(f"  File size   : {_fmt_bytes(meta['ckpt_size_bytes'])}")
    print(f"  Device      : {meta['device']}")
    print(f"  MiniGrid obs dim : {meta['minigrid_dim']}")
    print(f"  Dino obs dim     : {meta['dino_dim']}")

    _subheader("Architecture — Parameter Counts per Submodule")
    submodules = [
        ("minigrid_encoder", net.minigrid_encoder),
        ("dino_encoder    ", net.dino_encoder),
        ("shared_core     ", net.shared_core),
        ("minigrid_actor  ", net.minigrid_actor),
        ("minigrid_critic ", net.minigrid_critic),
        ("dino_actor      ", net.dino_actor),
        ("dino_critic     ", net.dino_critic),
    ]
    param_counts: Dict[str, int] = {}
    for name, mod in submodules:
        n = _param_count(mod)
        param_counts[name.strip()] = n
        print(f"    {name} : {_fmt_num(n):>10}  ({n:>10,} params)")

    total = _param_count(net)
    trainable = _trainable_count(net)
    print(f"  {'─' * 54}")
    print(f"    {'TOTAL':20s}   {_fmt_num(total):>10}  ({total:>10,} params)")
    print(f"    {'trainable':20s}   {_fmt_num(trainable):>10}  ({trainable:>10,} params)")
    param_bytes = total * 4
    print(f"  Parameter storage (fp32) : {_fmt_bytes(param_bytes)}")

    _subheader("Hardware & Runtime")
    print(f"  Python    : {sys.version.split()[0]}")
    print(f"  PyTorch   : {torch.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU       : {props.name}")
        print(f"  VRAM      : {_fmt_bytes(props.total_memory)}")
        print(f"  CUDA      : {torch.version.cuda}")
    else:
        print(f"  GPU       : not available — running on CPU")
    print(f"  CPU       : {platform.processor() or platform.machine()}")
    print(f"  OS        : {platform.system()} {platform.release()}")

    return {
        "param_counts": param_counts,
        "total_params": total,
        "trainable_params": trainable,
        "param_bytes": param_bytes,
        "ckpt_size_bytes": meta["ckpt_size_bytes"],
        "device": meta["device"],
        "torch_version": torch.__version__,
        "has_cuda": torch.cuda.is_available(),
    }


def bench_compute(
    agent: MultiTaskPPO,
    n_iters: int = DEFAULT_BENCH_ITERS,
    n_warmup: int = DEFAULT_BENCH_WARMUP,
) -> Dict[str, Any]:
    _header("COMPUTATION BENCHMARK")

    device = agent.device
    net = agent.net
    results: Dict[str, Any] = {}

    task_specs = [
        (TASK_MINIGRID, torch.zeros(1, agent.net.minigrid_dim, device=device), "MiniGrid"),
        (TASK_DINO,     torch.zeros(1, agent.net.dino_dim,     device=device), "Dino"),
    ]

    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for task_name, dummy_1, label in task_specs:
        _subheader(f"Latency — {label} (batch=1,  {n_iters} iterations)")

        for _ in range(n_warmup):
            with torch.no_grad():
                net(dummy_1, task_name)
        _sync()

        latencies_ms: List[float] = []
        for _ in range(n_iters):
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                net(dummy_1, task_name)
            _sync()
            latencies_ms.append((time.perf_counter() - t0) * 1_000.0)

        lat = np.array(latencies_ms, dtype=np.float64)
        print(f"    mean ± std       : {lat.mean():.3f} ± {lat.std():.3f} ms")
        print(f"    min  / max       : {lat.min():.3f} / {lat.max():.3f} ms")
        print(f"    p50 / p95 / p99  : "
              f"{np.percentile(lat, 50):.3f} / "
              f"{np.percentile(lat, 95):.3f} / "
              f"{np.percentile(lat, 99):.3f} ms")
        print(f"    throughput       : {1_000.0 / lat.mean():.1f} steps/s  (batch=1)")

        _subheader(f"Batched Throughput — {label}")
        throughputs: Dict[int, float] = {}
        for bs in BENCH_BATCH_SIZES:
            obs_batch = dummy_1.expand(bs, -1)
            for _ in range(20):
                with torch.no_grad():
                    net(obs_batch, task_name)
            _sync()
            n_b = max(50, n_iters // 4)
            t0 = time.perf_counter()
            for _ in range(n_b):
                with torch.no_grad():
                    net(obs_batch, task_name)
            _sync()
            elapsed = time.perf_counter() - t0
            sps = (n_b * bs) / elapsed
            ms_per = (elapsed / n_b) * 1_000.0
            throughputs[bs] = sps
            print(f"    batch={bs:4d}  →  {ms_per:8.3f} ms/batch  |  "
                  f"{sps:12.1f} steps/s")

        _subheader(f"Memory — {label}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                net(dummy_1, task_name)
            _sync()
            peak_mb = torch.cuda.max_memory_allocated() / (1 << 20)
            alloc_mb = torch.cuda.memory_allocated() / (1 << 20)
            print(f"    Peak VRAM during forward  : {peak_mb:.2f} MB")
            print(f"    VRAM allocated after step : {alloc_mb:.2f} MB")
        else:
            model_mb = sum(p.numel() * 4 for p in net.parameters()) / (1 << 20)
            print(f"    Model params (fp32)       : {model_mb:.2f} MB")
            print(f"    (GPU not available — activation memory not measured)")

        results[label.lower()] = {
            "latency_mean_ms": float(lat.mean()),
            "latency_std_ms": float(lat.std()),
            "latency_min_ms": float(lat.min()),
            "latency_max_ms": float(lat.max()),
            "latency_p50_ms": float(np.percentile(lat, 50)),
            "latency_p95_ms": float(np.percentile(lat, 95)),
            "latency_p99_ms": float(np.percentile(lat, 99)),
            "throughput_batch1_steps_per_sec": float(1_000.0 / lat.mean()),
            "batched_throughput_steps_per_sec": {
                str(bs): float(v) for bs, v in throughputs.items()
            },
        }

    return results


def _mg_greedy_episode(
    net: nn.Module,
    env: Any,
    device: str,
    seed: int,
    max_steps: int,
) -> Dict[str, Any]:
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0
    ep_ret = 0.0

    action_counts = np.zeros(MINIGRID_N_ACTIONS, dtype=np.int64)
    step_entropies: List[float] = []
    step_values: List[float] = []
    state_history: List[Tuple] = []
    key_picked = False
    door_opened = False
    max_action_prob_sum = 0.0

    while not (terminated or truncated) and steps < max_steps:
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(obs_t, TASK_MINIGRID)
            probs = torch.softmax(logits, dim=-1)[0]
            action = int(probs.argmax().item())
            ent = float(-(probs * (probs.clamp(min=1e-8)).log()).sum().item())
            max_p = float(probs.max().item())

        obs, reward, terminated, truncated, info = env.step(action)
        ep_ret += float(reward)
        steps += 1
        action_counts[action] += 1
        step_entropies.append(ent)
        step_values.append(float(value.item()))
        max_action_prob_sum += max_p

        u = env.unwrapped
        state_history.append(
            (tuple(int(c) for c in u.agent_pos), int(u.agent_dir))
        )

        if info.get("key_pickup_bonus"):
            key_picked = True
        if info.get("door_open_bonus"):
            door_opened = True

    solved = bool(terminated)

    failure_type: Optional[str] = None
    if not solved:
        tail = state_history[-LOOP_WINDOW:]
        unique = len(set(tail))
        failure_type = "loop" if unique <= LOOP_UNIQUE_MAX else "wander"

    mean_ent = float(np.mean(step_entropies)) if step_entropies else 0.0
    mean_val = float(np.mean(step_values)) if step_values else 0.0
    mean_conf = max_action_prob_sum / max(steps, 1)

    return {
        "solved": solved,
        "return": ep_ret,
        "steps": steps,
        "key_picked": key_picked,
        "door_opened": door_opened,
        "failure_type": failure_type,
        "action_counts": action_counts.tolist(),
        "mean_entropy": mean_ent,
        "mean_value": mean_val,
        "mean_action_confidence": mean_conf,
        "unique_states": len(set(state_history)),
        "total_states": len(state_history),
    }


def run_minigrid_eval(
    agent: MultiTaskPPO,
    sizes: Tuple[int, ...] = DEFAULT_MG_SIZES,
    n_seeds: int = DEFAULT_MG_SEEDS,
    seed_start: int = DEFAULT_MG_SEED_START,
    max_steps: int = DEFAULT_MG_MAX_STEPS,
    verbose: bool = False,
) -> Dict[str, Any]:
    _header("MINIGRID DOORKEY EVALUATION")
    print(f"  Seeds per size : {n_seeds}  (start={seed_start})")
    print(f"  Max steps/ep   : {max_steps}")

    device = agent.device
    net = agent.net
    per_size_records: List[Dict[str, Any]] = []
    all_results: Dict[str, Any] = {}

    g_eps = g_solved = g_loop = g_wander = 0
    g_returns: List[float] = []
    g_lengths: List[int] = []
    g_entropies: List[float] = []
    g_values: List[float] = []
    g_confidences: List[float] = []
    g_action_counts = np.zeros(MINIGRID_N_ACTIONS, dtype=np.int64)

    for size in sizes:
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        env = make_minigrid_env(env_id, max_episode_steps=max_steps, anti_stall=False)
        print(f"\n  ── {env_id} ──  ({n_seeds} seeds, max_steps={max_steps})")

        eps_data: List[Dict[str, Any]] = []
        t_size = time.time()

        for i in range(n_seeds):
            seed = seed_start + i
            ep = _mg_greedy_episode(net, env, device, seed, max_steps)
            eps_data.append(ep)
            if verbose:
                outcome = "SOLVED" if ep["solved"] else f"FAIL-{ep['failure_type']}"
                print(f"    seed {seed:5d}  {outcome:14s}  "
                      f"ret={ep['return']:+6.3f}  "
                      f"steps={ep['steps']:4d}  "
                      f"key={'Y' if ep['key_picked'] else 'n'}  "
                      f"door={'Y' if ep['door_opened'] else 'n'}  "
                      f"H={ep['mean_entropy']:.3f}")

        env.close()

        n = len(eps_data)
        solved_eps = [e for e in eps_data if e["solved"]]
        failed_eps = [e for e in eps_data if not e["solved"]]
        loop_eps   = [e for e in failed_eps if e["failure_type"] == "loop"]
        wander_eps = [e for e in failed_eps if e["failure_type"] == "wander"]

        returns    = [e["return"] for e in eps_data]
        lengths    = [e["steps"]  for e in eps_data]
        entropies  = [e["mean_entropy"] for e in eps_data]
        values     = [e["mean_value"]   for e in eps_data]
        confidences = [e["mean_action_confidence"] for e in eps_data]
        unique_states = [e["unique_states"] for e in eps_data]

        key_rate  = sum(e["key_picked"]   for e in eps_data) / n
        door_rate = sum(e["door_opened"]  for e in eps_data) / n

        ac = np.zeros(MINIGRID_N_ACTIONS, dtype=np.int64)
        for e in eps_data:
            ac += np.array(e["action_counts"], dtype=np.int64)
        ac_frac = ac / max(ac.sum(), 1)

        solve_rate  = len(solved_eps) / n
        loop_rate   = len(loop_eps)   / n
        wander_rate = len(wander_eps) / n

        size_rec: Dict[str, Any] = {
            "size": size,
            "env_id": env_id,
            "n_episodes": n,
            "wall_time_s": time.time() - t_size,
            "solve_rate": solve_rate,
            "loop_fail_rate": loop_rate,
            "wander_fail_rate": wander_rate,
            "return_mean": float(np.mean(returns)),
            "return_std": float(np.std(returns)),
            "return_min": float(np.min(returns)),
            "return_max": float(np.max(returns)),
            "return_p25": _percentile(returns, 25),
            "return_p50": _percentile(returns, 50),
            "return_p75": _percentile(returns, 75),
            "return_p90": _percentile(returns, 90),
            "length_mean": float(np.mean(lengths)),
            "length_std": float(np.std(lengths)),
            "length_min": int(np.min(lengths)),
            "length_max": int(np.max(lengths)),
            "key_pickup_rate": key_rate,
            "door_open_rate": door_rate,
            "action_distribution": {
                MINIGRID_ACTION_NAMES[i]: float(ac_frac[i])
                for i in range(MINIGRID_N_ACTIONS)
            },
            "entropy_mean": float(np.mean(entropies)),
            "value_mean": float(np.mean(values)),
            "action_confidence_mean": float(np.mean(confidences)),
            "unique_states_mean": float(np.mean(unique_states)),
        }
        per_size_records.append(size_rec)
        all_results[env_id] = size_rec

        print(f"    solved={_pct(solve_rate):>7}  "
              f"loop={_pct(loop_rate):>7}  "
              f"wander={_pct(wander_rate):>7}  "
              f"ret={np.mean(returns):+.3f}±{np.std(returns):.3f}  "
              f"len={np.mean(lengths):.1f}±{np.std(lengths):.1f}  "
              f"key={_pct(key_rate):>6}  door={_pct(door_rate):>6}  "
              f"H={np.mean(entropies):.3f}  "
              f"({time.time() - t_size:.0f}s)")

        g_eps     += n
        g_solved  += len(solved_eps)
        g_loop    += len(loop_eps)
        g_wander  += len(wander_eps)
        g_returns.extend(returns)
        g_lengths.extend(lengths)
        g_entropies.extend(entropies)
        g_values.extend(values)
        g_confidences.extend(confidences)
        g_action_counts += ac

    _subheader("Per-Size Summary Table")
    hdr = (f"  {'Size':>5} | {'Solved':>7} | {'Loop':>6} | {'Wander':>7} | "
           f"{'Ret μ':>8} | {'Ret σ':>7} | {'Len μ':>7} | "
           f"{'Key%':>5} | {'Door%':>6} | {'H μ':>6} | {'Conf μ':>7}")
    print(hdr)
    print(f"  {_sep('-', len(hdr) - 2)}")
    for r in per_size_records:
        print(
            f"  {r['size']:>5} | "
            f"{r['solve_rate']*100:>6.1f}% | "
            f"{r['loop_fail_rate']*100:>5.1f}% | "
            f"{r['wander_fail_rate']*100:>6.1f}% | "
            f"{r['return_mean']:>+8.3f} | "
            f"{r['return_std']:>7.3f} | "
            f"{r['length_mean']:>7.1f} | "
            f"{r['key_pickup_rate']*100:>4.0f}% | "
            f"{r['door_open_rate']*100:>5.0f}% | "
            f"{r['entropy_mean']:>6.3f} | "
            f"{r['action_confidence_mean']:>6.3f}"
        )

    _subheader("Overall MiniGrid Results (all sizes combined)")
    g_ac_frac = g_action_counts / max(g_action_counts.sum(), 1)
    ovr_solve  = g_solved / g_eps
    ovr_loop   = g_loop   / g_eps
    ovr_wander = g_wander / g_eps

    print(f"  Total episodes : {g_eps:,}")
    print(f"  Solved         : {g_solved:,} / {g_eps:,}  ({_pct(ovr_solve)})")
    print(f"  Loop-fail      : {g_loop:,} / {g_eps:,}  ({_pct(ovr_loop)})")
    print(f"  Wander-fail    : {g_wander:,} / {g_eps:,}  ({_pct(ovr_wander)})")
    print()
    print(f"  Return")
    print(f"    mean ± std   : {np.mean(g_returns):+.4f} ± {np.std(g_returns):.4f}")
    print(f"    min  / max   : {np.min(g_returns):+.4f} / {np.max(g_returns):+.4f}")
    print(f"    p25 / p50 / p75 / p90 : "
          f"{_percentile(g_returns, 25):+.4f} / "
          f"{_percentile(g_returns, 50):+.4f} / "
          f"{_percentile(g_returns, 75):+.4f} / "
          f"{_percentile(g_returns, 90):+.4f}")
    print()
    print(f"  Episode Length")
    print(f"    mean ± std   : {np.mean(g_lengths):.1f} ± {np.std(g_lengths):.1f}")
    print(f"    min  / max   : {np.min(g_lengths)} / {np.max(g_lengths)}")
    print(f"    p25 / p50 / p75 / p90 : "
          f"{_percentile(g_lengths, 25):.0f} / "
          f"{_percentile(g_lengths, 50):.0f} / "
          f"{_percentile(g_lengths, 75):.0f} / "
          f"{_percentile(g_lengths, 90):.0f}")
    print()
    print(f"  Policy quality")
    print(f"    entropy (mean)         : {np.mean(g_entropies):.4f}  "
          f"(0=deterministic, {np.log(MINIGRID_N_ACTIONS):.4f}=uniform)")
    print(f"    value estimate (mean)  : {np.mean(g_values):+.4f}")
    print(f"    action confidence (mean max-prob) : {np.mean(g_confidences):.4f}")
    print()
    print(f"  Action Distribution (all steps across all sizes):")
    for i, name in enumerate(MINIGRID_ACTION_NAMES):
        bar = "█" * int(g_ac_frac[i] * 40)
        print(f"    {name:10s}: {g_ac_frac[i]*100:5.1f}%  {bar}")

    rec16 = next((r for r in per_size_records if r["size"] == 16), None)
    if rec16:
        _subheader("DoorKey-16×16 vs Feedforward Baseline (final.pt context)")
        sr = rec16["solve_rate"] * 100
        lr = rec16["loop_fail_rate"] * 100
        wr = rec16["wander_fail_rate"] * 100
        bl_solve = FF_BASELINE_SOLVED * 100
        bl_loop  = FF_BASELINE_LOOPS  * 100
        print(f"  {'Metric':<30}  {'This model':>12}  {'FF Baseline':>12}  {'Δ':>8}")
        print(f"  {'─' * 68}")
        print(f"  {'Solve rate':<30}  {sr:>11.2f}%  {bl_solve:>11.1f}%  "
              f"{sr - bl_solve:>+7.2f}%")
        print(f"  {'Loop-fail rate':<30}  {lr:>11.2f}%  {bl_loop:>11.1f}%  "
              f"{lr - bl_loop:>+7.2f}%")
        print(f"  {'Wander-fail rate':<30}  {wr:>11.2f}%  {'N/A':>12}  {'N/A':>8}")
        print(f"  {'Mean episode return':<30}  {rec16['return_mean']:>+11.4f}  {'N/A':>12}  {'N/A':>8}")
        print(f"  {'Mean episode length':<30}  {rec16['length_mean']:>11.1f}  {'N/A':>12}  {'N/A':>8}")
        if sr >= bl_solve and lr <= bl_loop * 100 / 100 + 2:
            print(f"\n  [PASS] Solve rate >= baseline and loop-fail within +2pp.")
        else:
            print(f"\n  [CHECK] Targets: solved >= {bl_solve:.0f}% and loop <= {bl_loop + 2:.0f}%.")

    overall: Dict[str, Any] = {
        "total_episodes": g_eps,
        "solve_rate": ovr_solve,
        "loop_fail_rate": ovr_loop,
        "wander_fail_rate": ovr_wander,
        "return_mean": float(np.mean(g_returns)),
        "return_std": float(np.std(g_returns)),
        "return_min": float(np.min(g_returns)),
        "return_max": float(np.max(g_returns)),
        "return_p25": _percentile(g_returns, 25),
        "return_p50": _percentile(g_returns, 50),
        "return_p75": _percentile(g_returns, 75),
        "return_p90": _percentile(g_returns, 90),
        "length_mean": float(np.mean(g_lengths)),
        "length_std": float(np.std(g_lengths)),
        "length_p25": _percentile(g_lengths, 25),
        "length_p50": _percentile(g_lengths, 50),
        "length_p75": _percentile(g_lengths, 75),
        "length_p90": _percentile(g_lengths, 90),
        "entropy_mean": float(np.mean(g_entropies)),
        "value_mean": float(np.mean(g_values)),
        "action_confidence_mean": float(np.mean(g_confidences)),
        "action_distribution": {
            MINIGRID_ACTION_NAMES[i]: float(g_ac_frac[i])
            for i in range(MINIGRID_N_ACTIONS)
        },
        "action_counts": {
            MINIGRID_ACTION_NAMES[i]: int(g_action_counts[i])
            for i in range(MINIGRID_N_ACTIONS)
        },
    }
    return {"per_size": all_results, "overall": overall}


def _dino_greedy_episode(
    net: nn.Module,
    episode_idx: int,
    device: str,
    max_guard: int = 100_000,
) -> Dict[str, Any]:
    seed = episode_idx * 7 + 10_000
    env = DinoEnv(render=False, seed=seed)
    obs = env.reset()
    done = False
    steps = 0
    ep_ret = 0.0

    action_counts = np.zeros(DINO_ACTIONS, dtype=np.int64)
    step_entropies: List[float] = []
    step_values: List[float] = []
    obstacle_passes = 0
    last_speed = 0.0
    death_obstacle = ""

    while not done and steps < max_guard:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, value = net(obs_t, TASK_DINO)
            probs = torch.softmax(logits, dim=-1)[0]
            action = int(probs.argmax().item())
            ent = float(-(probs * probs.clamp(min=1e-8).log()).sum().item())

        obs, reward, done, info = env.step(action)
        ep_ret += float(reward)
        steps += 1
        action_counts[action] += 1
        step_entropies.append(ent)
        step_values.append(float(value.item()))
        last_speed = float(info.get("speed", last_speed))
        if info.get("passed", False):
            obstacle_passes += 1

    score = int(info.get("score", 0))
    if done:
        death_obstacle = str(info.get("death_obstacle", ""))
    env.close()

    return {
        "score": score,
        "return": ep_ret,
        "steps": steps,
        "obstacle_passes": obstacle_passes,
        "action_counts": action_counts.tolist(),
        "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
        "mean_value": float(np.mean(step_values)) if step_values else 0.0,
        "death_speed": last_speed,
        "death_obstacle": death_obstacle,
    }


def run_dino_eval(
    agent: MultiTaskPPO,
    n_episodes: int = DEFAULT_DINO_EPISODES,
    verbose: bool = False,
) -> Dict[str, Any]:
    _header("DINO RUNNER EVALUATION")
    print(f"  Episodes : {n_episodes}")

    device = agent.device
    net = agent.net
    eps_data: List[Dict[str, Any]] = []
    best_score = 0
    t_dino = time.time()

    for ep in range(1, n_episodes + 1):
        data = _dino_greedy_episode(net, episode_idx=ep, device=device)
        eps_data.append(data)
        best_score = max(best_score, data["score"])
        if verbose:
            frac = np.array(data["action_counts"]) / max(sum(data["action_counts"]), 1)
            print(
                f"  ep {ep:3d}  score={data['score']:6d} (best={best_score:6d})  "
                f"ret={data['return']:+8.2f}  steps={data['steps']:6d}  "
                f"passes={data['obstacle_passes']:3d}  "
                f"n/j/d={frac.round(2).tolist()}  "
                f"H={data['mean_entropy']:.3f}"
            )
        else:
            if ep % max(1, n_episodes // 20) == 0:
                print(f"  ... ep {ep}/{n_episodes}  best_score={best_score}", flush=True)

    wall_dino = time.time() - t_dino

    scores   = [e["score"]            for e in eps_data]
    returns  = [e["return"]           for e in eps_data]
    lengths  = [e["steps"]            for e in eps_data]
    passes   = [e["obstacle_passes"]  for e in eps_data]
    speeds   = [e["death_speed"]      for e in eps_data]
    ents     = [e["mean_entropy"]     for e in eps_data]
    vals     = [e["mean_value"]       for e in eps_data]

    ac = np.zeros(DINO_ACTIONS, dtype=np.int64)
    for e in eps_data:
        ac += np.array(e["action_counts"], dtype=np.int64)
    ac_frac = ac / max(ac.sum(), 1)

    death_counts: Dict[str, int] = defaultdict(int)
    for e in eps_data:
        key = e["death_obstacle"] or "unknown"
        death_counts[key] += 1

    _subheader("Score Statistics")
    print(f"  Episodes : {n_episodes}  ({wall_dino:.1f}s total)")
    print(f"  mean ± std     : {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  min  / max     : {int(np.min(scores))} / {int(np.max(scores))}")
    print(f"  median         : {int(np.median(scores))}")
    print(f"  p10  / p25     : {_percentile(scores, 10):.0f} / {_percentile(scores, 25):.0f}")
    print(f"  p75  / p90     : {_percentile(scores, 75):.0f} / {_percentile(scores, 90):.0f}")
    print(f"  p95  / p99     : {_percentile(scores, 95):.0f} / {_percentile(scores, 99):.0f}")

    _subheader("RL Return Statistics")
    print(f"  mean ± std : {np.mean(returns):+.3f} ± {np.std(returns):.3f}")
    print(f"  min  / max : {np.min(returns):+.3f} / {np.max(returns):+.3f}")
    print(f"  p25 / p50 / p75 / p90 : "
          f"{_percentile(returns, 25):+.3f} / "
          f"{_percentile(returns, 50):+.3f} / "
          f"{_percentile(returns, 75):+.3f} / "
          f"{_percentile(returns, 90):+.3f}")

    _subheader("Episode Length (Steps)")
    print(f"  mean ± std : {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  min  / max : {int(np.min(lengths))} / {int(np.max(lengths))}")
    print(f"  p25 / p50 / p75 / p90 : "
          f"{_percentile(lengths, 25):.0f} / "
          f"{_percentile(lengths, 50):.0f} / "
          f"{_percentile(lengths, 75):.0f} / "
          f"{_percentile(lengths, 90):.0f}")

    _subheader("Obstacle Passes")
    passes_per_ep   = float(np.mean(passes))
    passes_per_step = sum(passes) / max(sum(lengths), 1)
    print(f"  passes / episode : {passes_per_ep:.2f}")
    print(f"  passes / step    : {passes_per_step:.5f}")
    print(f"  total passes     : {sum(passes)}")

    _subheader("Action Distribution (all steps)")
    for i, name in enumerate(DINO_ACTION_NAMES):
        bar = "█" * int(ac_frac[i] * 40)
        print(f"  {name:6s}: {ac_frac[i]*100:5.1f}%  {bar}")

    _subheader("Speed at Death")
    print(f"  mean : {np.mean(speeds):.3f}")
    print(f"  std  : {np.std(speeds):.3f}")
    print(f"  min  : {np.min(speeds):.3f}")
    print(f"  max  : {np.max(speeds):.3f}")
    print(f"  p25 / p50 / p75 : "
          f"{_percentile(speeds, 25):.3f} / "
          f"{_percentile(speeds, 50):.3f} / "
          f"{_percentile(speeds, 75):.3f}")

    _subheader("Death Obstacle Breakdown")
    for obs_key, cnt in sorted(death_counts.items(), key=lambda x: -x[1]):
        print(f"  {obs_key or '(empty)':<30}: {cnt:3d}  ({cnt / n_episodes * 100:.1f}%)")

    _subheader("Policy Quality")
    max_ent = float(np.log(DINO_ACTIONS))
    print(f"  entropy — mean : {np.mean(ents):.4f}  "
          f"(0=deterministic, {max_ent:.4f}=uniform)")
    print(f"  entropy — std  : {np.std(ents):.4f}")
    print(f"  entropy — min  : {np.min(ents):.4f}")
    print(f"  entropy — max  : {np.max(ents):.4f}")
    print(f"  value estimate — mean : {np.mean(vals):+.4f}")
    print(f"  value estimate — std  : {np.std(vals):.4f}")

    summary: Dict[str, Any] = {
        "n_episodes": n_episodes,
        "wall_time_s": wall_dino,
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": int(np.min(scores)),
        "score_max": int(np.max(scores)),
        "score_median": int(np.median(scores)),
        "score_p10": _percentile(scores, 10),
        "score_p25": _percentile(scores, 25),
        "score_p50": _percentile(scores, 50),
        "score_p75": _percentile(scores, 75),
        "score_p90": _percentile(scores, 90),
        "score_p95": _percentile(scores, 95),
        "score_p99": _percentile(scores, 99),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "return_p25": _percentile(returns, 25),
        "return_p50": _percentile(returns, 50),
        "return_p75": _percentile(returns, 75),
        "return_p90": _percentile(returns, 90),
        "steps_mean": float(np.mean(lengths)),
        "steps_std": float(np.std(lengths)),
        "steps_min": int(np.min(lengths)),
        "steps_max": int(np.max(lengths)),
        "passes_per_episode": passes_per_ep,
        "passes_per_step": passes_per_step,
        "action_distribution": {
            DINO_ACTION_NAMES[i]: float(ac_frac[i]) for i in range(DINO_ACTIONS)
        },
        "action_counts": {
            DINO_ACTION_NAMES[i]: int(ac[i]) for i in range(DINO_ACTIONS)
        },
        "death_speed_mean": float(np.mean(speeds)),
        "death_speed_std": float(np.std(speeds)),
        "death_speed_max": float(np.max(speeds)),
        "death_obstacle_counts": dict(death_counts),
        "entropy_mean": float(np.mean(ents)),
        "entropy_std": float(np.std(ents)),
        "value_mean": float(np.mean(vals)),
        "value_std": float(np.std(vals)),
    }
    return {"summary": summary, "episodes": eps_data}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Comprehensive evaluation & benchmark for checkpoints_final/final.pt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                   help="Path to the .pt checkpoint file.")
    p.add_argument("--task", choices=("minigrid", "dino", "both"), default="both",
                   help="Which task(s) to evaluate.")
    p.add_argument("--mg-sizes", type=str,
                   default=",".join(map(str, DEFAULT_MG_SIZES)),
                   help="Comma-separated DoorKey grid sizes to evaluate.")
    p.add_argument("--mg-seeds", type=int, default=DEFAULT_MG_SEEDS,
                   help="Number of random seeds per DoorKey size.")
    p.add_argument("--mg-seed-start", type=int, default=DEFAULT_MG_SEED_START,
                   help="Starting seed index (avoids overlap with training seeds).")
    p.add_argument("--mg-max-steps", type=int, default=DEFAULT_MG_MAX_STEPS,
                   help="Per-episode step cap for MiniGrid evaluation.")
    p.add_argument("--dino-episodes", type=int, default=DEFAULT_DINO_EPISODES,
                   help="Number of Dino evaluation episodes.")
    p.add_argument("--bench-iters", type=int, default=DEFAULT_BENCH_ITERS,
                   help="Forward-pass iterations for latency measurement. 0 to skip.")
    p.add_argument("--bench-warmup", type=int, default=DEFAULT_BENCH_WARMUP,
                   help="Warm-up iterations before latency timing.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-episode outcomes for all environments.")
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to save full results as JSON.")
    args = p.parse_args()

    mg_sizes = tuple(int(s) for s in args.mg_sizes.split(",") if s.strip())
    t_total = time.time()

    print("═" * 72)
    print("  MULTI-TASK MODEL — COMPREHENSIVE EVALUATION & BENCHMARK")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Task(s)    : {args.task}")
    print(f"  Started    : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 72)

    agent, meta = load_agent(args.ckpt)

    all_results: Dict[str, Any] = {
        "ckpt_path": meta["ckpt_path"],
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "meta": meta,
    }

    all_results["model_summary"] = model_summary(agent, meta)

    if args.bench_iters > 0:
        all_results["compute_benchmark"] = bench_compute(
            agent, n_iters=args.bench_iters, n_warmup=args.bench_warmup
        )
    else:
        print("\n[skipping computation benchmark (--bench-iters 0)]")

    if args.task in ("minigrid", "both"):
        all_results["minigrid"] = run_minigrid_eval(
            agent,
            sizes=mg_sizes,
            n_seeds=args.mg_seeds,
            seed_start=args.mg_seed_start,
            max_steps=args.mg_max_steps,
            verbose=args.verbose,
        )

    if args.task in ("dino", "both"):
        all_results["dino"] = run_dino_eval(
            agent,
            n_episodes=args.dino_episodes,
            verbose=args.verbose,
        )

    wall_total = time.time() - t_total
    _header("FINAL SUMMARY")
    print(f"  Total wall time : {wall_total:.1f}s")

    if "model_summary" in all_results:
        ms = all_results["model_summary"]
        print(f"  Parameters      : {_fmt_num(ms['total_params'])}  "
              f"({_fmt_bytes(ms['param_bytes'])} fp32)")
        print(f"  Checkpoint size : {_fmt_bytes(ms['ckpt_size_bytes'])}")
        print(f"  Device          : {ms['device']}")

    if args.task in ("minigrid", "both") and "minigrid" in all_results:
        ovr = all_results["minigrid"]["overall"]
        print(f"\n  ┌─ MiniGrid (all sizes, {ovr['total_episodes']:,} episodes) ─────────────────")
        print(f"  │  solve rate     : {_pct(ovr['solve_rate'])}")
        print(f"  │  loop-fail      : {_pct(ovr['loop_fail_rate'])}")
        print(f"  │  wander-fail    : {_pct(ovr['wander_fail_rate'])}")
        print(f"  │  mean return    : {ovr['return_mean']:+.4f}")
        print(f"  │  mean length    : {ovr['length_mean']:.1f} steps")
        print(f"  └─ mean entropy   : {ovr['entropy_mean']:.4f}")

    if args.task in ("dino", "both") and "dino" in all_results:
        ds = all_results["dino"]["summary"]
        print(f"\n  ┌─ Dino ({ds['n_episodes']} episodes) ─────────────────────────────────────")
        print(f"  │  score mean     : {ds['score_mean']:.1f}")
        print(f"  │  score max      : {ds['score_max']}")
        print(f"  │  mean return    : {ds['return_mean']:+.3f}")
        print(f"  │  passes/episode : {ds['passes_per_episode']:.2f}")
        print(f"  └─ mean entropy   : {ds['entropy_mean']:.4f}")

    if "compute_benchmark" in all_results:
        cb = all_results["compute_benchmark"]
        print(f"\n  ┌─ Compute benchmark ──────────────────────────────────────────────────")
        for task_key in ("minigrid", "dino"):
            if task_key in cb:
                td = cb[task_key]
                print(f"  │  {task_key:<12}: latency {td['latency_mean_ms']:.3f}ms  "
                      f"throughput {td['throughput_batch1_steps_per_sec']:.0f} steps/s")
        print(f"  └──────────────────────────────────────────────────────────────────────")

    if args.out:
        def _serialise(obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialise(v) for v in obj]
            return obj

        Path(args.out).write_text(json.dumps(_serialise(all_results), indent=2))
        print(f"\n  Full results saved → {args.out}")

    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    main()
