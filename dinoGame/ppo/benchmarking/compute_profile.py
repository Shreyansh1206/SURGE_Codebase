"""
compute_profile.py — Measure compute cost of the PPO Dino ActorCritic at inference.

Runs repeated timed forward passes (default 25) after warmup to average out OS jitter.
Reports analytical MAC/FLOP estimates, parameter count, memory footprint, and
latency statistics. Optionally profiles the full benchmark-style argmax path.

Usage (from dinoGame/ppo/):
    python benchmarking/compute_profile.py
    python benchmarking/compute_profile.py --ckpt checkpoints_scratch_duck/best_duck.pt --runs 25
    python benchmarking/compute_profile.py --device cuda --runs 25 --out-dir benchmarking/results/compute_best_duck
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dino_env import OBS_DIM, N_ACTIONS, FEATURES_PER_FRAME  # noqa: E402
from ppo_agent import ActorCritic, PPO  # noqa: E402

# Match live env / benchmark cadence.
FRAMES_PER_STEP = 4
GAME_HZ = 60.0
DECISIONS_PER_SEC = GAME_HZ / FRAMES_PER_STEP  # 15 Hz


def descriptive_stats(values):
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n == 0:
        return {}
    mn, std = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * sem
    pcts = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "n": n,
        "mean": mn,
        "std": std,
        "sem": sem,
        "min": float(a.min()),
        "max": float(a.max()),
        "median": float(np.median(a)),
        "p5": float(pcts[1]),
        "p95": float(pcts[5]),
        "p99": float(pcts[8]),
        "ci95_lo": mn - ci,
        "ci95_hi": mn + ci,
        "cv": std / abs(mn) if abs(mn) > 1e-12 else float("nan"),
    }


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def analytical_forward_macs(obs_dim: int, n_actions: int, hidden: int = 128):
    """
    Multiply-accumulate count for one ActorCritic forward (shared trunk + both heads).
    Softmax / argmax (deployment path) counted separately.
    """
    l1 = obs_dim * hidden + hidden          # Linear + bias
    l2 = hidden * hidden + hidden
    policy = hidden * n_actions + n_actions
    value = hidden * 1 + 1
    trunk = l1 + l2
    return {
        "linear_mac_total": int(l1 + l2 + policy + value),
        "layer1_mac": int(l1),
        "layer2_mac": int(l2),
        "policy_head_mac": int(policy),
        "value_head_mac": int(value),
        "trunk_mac": int(trunk),
        "activation_ops_estimate": int(hidden * 2),  # Tanh ~1 op/elem (rough)
        "softmax_argmax_ops_estimate": int(n_actions * 4 + 2),  # exp, sum, div, argmax
    }


def checkpoint_bytes(path: str) -> int:
    return os.path.getsize(path) if os.path.exists(path) else 0


def _sync(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class ProcessResourceSampler:
    """Per-iteration process CPU% and RSS (requires psutil)."""

    def __init__(self):
        import psutil
        self._psutil = psutil
        self.proc = psutil.Process(os.getpid())
        # Prime cpu_percent so the next reads are meaningful.
        self.proc.cpu_percent(interval=None)

    def sample(self):
        rss_mb = self.proc.memory_info().rss / (1024 * 1024)
        cpu_pct = self.proc.cpu_percent(interval=None)
        return float(cpu_pct), round(float(rss_mb), 3)

    @staticmethod
    def system_memory_snapshot():
        import psutil
        vm = psutil.virtual_memory()
        return {
            "system_ram_total_mb": round(vm.total / (1024 * 1024), 2),
            "system_ram_available_mb": round(vm.available / (1024 * 1024), 2),
            "system_ram_used_mb": round(vm.used / (1024 * 1024), 2),
            "system_ram_percent": float(vm.percent),
        }


def _empty_run_samples(n_runs: int):
    return {
        "times_ms": [],
        "cpu_percent": [],
        "ram_rss_mb": [],
        "per_run": [],
    }


def _process_cpu_percent_over_interval(proc, wall_seconds: float, cpu_seconds: float):
    """Process CPU utilization % for a wall-clock interval (can exceed 100% on multi-core)."""
    if wall_seconds <= 0:
        return 0.0
    return 100.0 * cpu_seconds / wall_seconds


def _timed_run_loop(body_fn, sampler, device: str, n_warmup: int, n_runs: int):
    """Warmup, then n_runs timed calls via body_fn(); collect latency + resources."""
    out = _empty_run_samples(n_runs)
    with torch.no_grad():
        for _ in range(n_warmup):
            body_fn()
        _sync(device)
        if sampler is not None:
            sampler.sample()
        block_cpu_pct = None
        t_block0 = t_block1 = None
        if sampler is not None:
            c0 = sampler.proc.cpu_times()
            t_block0 = time.perf_counter()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            body_fn()
            _sync(device)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out["times_ms"].append(elapsed_ms)
            if sampler is not None:
                cpu_pct, ram_mb = sampler.sample()
                out["cpu_percent"].append(cpu_pct)
                out["ram_rss_mb"].append(ram_mb)
                out["per_run"].append({
                    "latency_ms": elapsed_ms,
                    "cpu_percent": cpu_pct,
                    "ram_rss_mb": ram_mb,
                })
            else:
                out["per_run"].append({"latency_ms": elapsed_ms})
        if sampler is not None:
            t_block1 = time.perf_counter()
            c1 = sampler.proc.cpu_times()
            cpu_sec = (c1.user - c0.user) + (c1.system - c0.system)
            block_cpu_pct = _process_cpu_percent_over_interval(
                sampler.proc, t_block1 - t_block0, cpu_sec
            )
        out["block_cpu_percent"] = block_cpu_pct
        if sampler is not None and t_block0 is not None:
            out["block_wall_seconds"] = round(t_block1 - t_block0, 6)
            out["block_cpu_seconds"] = round(cpu_sec, 6)
    return out


def measure_sustained_resources(body_fn, sampler, inner_iters: int = 5000):
    """
    Run many back-to-back forwards so CPU time is measurable on Windows
  (sub-ms single forwards often round to 0% with psutil).
    """
    with torch.no_grad():
        for _ in range(10):
            body_fn()
        if sampler is not None:
            sampler.sample()
        c0 = sampler.proc.cpu_times()
        t_wall0 = time.perf_counter()
        t_proc0 = time.process_time()
        for _ in range(inner_iters):
            body_fn()
        t_proc1 = time.process_time()
        t_wall1 = time.perf_counter()
        c1 = sampler.proc.cpu_times()
    cpu_sec_ps = (c1.user - c0.user) + (c1.system - c0.system)
    wall_sec = t_wall1 - t_wall0
    proc_sec = t_proc1 - t_proc0
    _, ram_mb = sampler.sample()
    return {
        "inner_iterations": inner_iters,
        "wall_seconds": round(wall_sec, 6),
        "process_cpu_seconds_psutil": round(cpu_sec_ps, 6),
        "process_cpu_seconds_time_module": round(proc_sec, 6),
        "cpu_percent_of_one_core_psutil": round(
            _process_cpu_percent_over_interval(sampler.proc, wall_sec, cpu_sec_ps), 4
        ),
        "cpu_percent_of_one_core_process_time": round(
            100.0 * proc_sec / wall_sec if wall_sec > 0 else 0.0, 4
        ),
        "ram_rss_mb_after_burst": ram_mb,
        "inferences_per_second": round(inner_iters / wall_sec, 2) if wall_sec > 0 else 0.0,
    }


def time_forward_only(net, obs_t, device: str, n_warmup: int, n_runs: int,
                      sampler: ProcessResourceSampler | None = None):
    def body():
        net(obs_t)
    return _timed_run_loop(body, sampler, device, n_warmup, n_runs)


def time_argmax_deploy(net, obs_t, device: str, n_warmup: int, n_runs: int,
                       sampler: ProcessResourceSampler | None = None):
    def body():
        logits, value = net(obs_t)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        action = int(np.argmax(probs))
        _ = value, action
    return _timed_run_loop(body, sampler, device, n_warmup, n_runs)


def time_batch_forward(net, obs_batch, device: str, n_warmup: int, n_runs: int,
                       sampler: ProcessResourceSampler | None = None):
    def body():
        net(obs_batch)
    return _timed_run_loop(body, sampler, device, n_warmup, n_runs)


def time_paced_game_loop(body_fn, sampler, device: str, n_warmup: int, n_runs: int,
                         decision_hz: float = DECISIONS_PER_SEC):
    """
    Simulate in-game decision cadence: one inference every 1/decision_hz seconds.
    Sleeps after each decision so slot wall time targets the game rate (15 Hz default).
    """
    period_s = 1.0 / decision_hz
    out = {
        "decision_hz_target": decision_hz,
        "period_ms_target": period_s * 1000.0,
        "inference_ms": [],
        "slot_wall_ms": [],
        "idle_sleep_ms": [],
        "cpu_percent": [],
        "ram_rss_mb": [],
        "per_tick": [],
    }
    with torch.no_grad():
        for _ in range(n_warmup):
            body_fn()
        _sync(device)
        if sampler is not None:
            sampler.sample()

        c0 = sampler.proc.cpu_times() if sampler else None
        t_session0 = time.perf_counter()

        for tick in range(n_runs):
            t_slot0 = time.perf_counter()
            t_deadline = t_slot0 + period_s
            t_inf0 = time.perf_counter()
            body_fn()
            _sync(device)
            inference_ms = (time.perf_counter() - t_inf0) * 1000.0

            remaining_s = t_deadline - time.perf_counter()
            if remaining_s > 0:
                time.sleep(remaining_s)
            slot_wall_ms = (time.perf_counter() - t_slot0) * 1000.0
            idle_ms = max(0.0, slot_wall_ms - inference_ms)

            cpu_pct, ram_mb = (None, None)
            if sampler is not None:
                cpu_pct, ram_mb = sampler.sample()

            out["inference_ms"].append(inference_ms)
            out["slot_wall_ms"].append(slot_wall_ms)
            out["idle_sleep_ms"].append(idle_ms)
            if cpu_pct is not None:
                out["cpu_percent"].append(cpu_pct)
                out["ram_rss_mb"].append(ram_mb)
            tick_rec = {
                "tick": tick + 1,
                "inference_ms": round(inference_ms, 4),
                "slot_wall_ms": round(slot_wall_ms, 4),
                "idle_sleep_ms": round(idle_ms, 4),
            }
            if cpu_pct is not None:
                tick_rec["cpu_percent"] = cpu_pct
                tick_rec["ram_rss_mb"] = ram_mb
            out["per_tick"].append(tick_rec)

        t_session1 = time.perf_counter()
        if sampler is not None:
            c1 = sampler.proc.cpu_times()
            cpu_sec = (c1.user - c0.user) + (c1.system - c0.system)
            wall_sec = t_session1 - t_session0
            out["session_wall_seconds"] = round(wall_sec, 4)
            out["session_cpu_seconds"] = round(cpu_sec, 6)
            out["session_cpu_percent"] = round(
                _process_cpu_percent_over_interval(sampler.proc, wall_sec, cpu_sec), 4
            )
            achieved_hz = n_runs / wall_sec if wall_sec > 0 else 0.0
            out["achieved_decisions_per_second"] = round(achieved_hz, 4)
            inf_arr = np.asarray(out["inference_ms"])
            slot_arr = np.asarray(out["slot_wall_ms"])
            out["model_duty_cycle_percent"] = round(
                100.0 * float(inf_arr.mean()) / float(slot_arr.mean())
                if slot_arr.mean() > 0 else 0.0, 4
            )
    return out


def pack_paced_profile(paced_data: dict):
    """Aggregate paced 15 Hz simulation ticks."""
    packed = {
        "decision_hz_target": paced_data["decision_hz_target"],
        "period_ms_target": paced_data["period_ms_target"],
        "inference_latency_ms": descriptive_stats(paced_data["inference_ms"]),
        "slot_wall_ms": descriptive_stats(paced_data["slot_wall_ms"]),
        "idle_sleep_ms": descriptive_stats(paced_data["idle_sleep_ms"]),
        "per_tick": paced_data["per_tick"],
    }
    if paced_data["cpu_percent"]:
        packed["cpu_percent"] = descriptive_stats(paced_data["cpu_percent"])
        packed["cpu_percent_samples"] = paced_data["cpu_percent"]
    if paced_data["ram_rss_mb"]:
        packed["ram_rss_mb"] = descriptive_stats(paced_data["ram_rss_mb"])
        packed["ram_rss_mb_samples"] = paced_data["ram_rss_mb"]
    for key in (
        "session_wall_seconds", "session_cpu_seconds", "session_cpu_percent",
        "achieved_decisions_per_second", "model_duty_cycle_percent",
    ):
        if key in paced_data:
            packed[key] = paced_data[key]
    return packed


def make_resource_sampler():
    try:
        return ProcessResourceSampler()
    except ImportError:
        return None


def system_memory_snapshot():
    try:
        return ProcessResourceSampler.system_memory_snapshot()
    except ImportError:
        return None


def pack_latency(times_ms):
    stats = descriptive_stats(times_ms)
    thr = descriptive_stats((1000.0 / np.asarray(times_ms)).tolist())
    return {"latency_ms": stats, "throughput_per_sec": thr}


def pack_run_profile(run_data: dict):
    """Latency + throughput + CPU/RAM stats from _timed_run_loop output."""
    packed = {
        **pack_latency(run_data["times_ms"]),
        "per_run": run_data["per_run"],
    }
    if run_data["cpu_percent"]:
        packed["cpu_percent"] = descriptive_stats(run_data["cpu_percent"])
        packed["cpu_percent_samples"] = run_data["cpu_percent"]
    if run_data["ram_rss_mb"]:
        packed["ram_rss_mb"] = descriptive_stats(run_data["ram_rss_mb"])
        packed["ram_rss_mb_samples"] = run_data["ram_rss_mb"]
    if run_data.get("block_cpu_percent") is not None:
        packed["block_cpu_percent"] = round(float(run_data["block_cpu_percent"]), 4)
    return packed


def build_report(data: dict) -> str:
    inf = data["inference"]
    paced = inf.get("paced_game_argmax")
    macs = data["analytical"]["macs_deploy_with_softmax_argmax"]
    m = paced["inference_latency_ms"] if paced else inf.get("argmax_deploy", {}).get(
        "latency_ms", {}
    )
    fo = inf.get("forward_only", {}).get("latency_ms", {})
    lines = [
        "# PPO Dino — Compute Profile",
        "",
        f"**Generated:** {data['run_timestamp']}",
        f"**Checkpoint:** `{data['checkpoint']}`",
        f"**Device:** {data['hardware']['torch_device']}",
        f"**Timed runs:** {data['methodology']['n_timed_runs']} (after {data['methodology']['n_warmup']} warmup)",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Parameters | {data['model']['parameters_total']:,} |",
        f"| Checkpoint size | {data['model']['checkpoint_size_mb']:.3f} MB |",
        f"| Weight memory (FP32) | {data['model']['weight_memory_fp32_mb']:.3f} MB |",
        f"| MACs / deploy (analytical) | {macs:,} |",
        f"| FLOPs / deploy (2× MAC) | {data['analytical']['flops_deploy_estimate']:,} |",
    ]
    if paced:
        sl = paced["slot_wall_ms"]
        lines += [
            f"| **Paced simulation** | **{paced['decision_hz_target']:.1f} Hz** |",
            f"| Inference latency (median) | {m['median']:.4f} ms |",
            f"| Inference latency (mean) | {m['mean']:.4f} ms |",
            f"| Slot period (median) | {sl['median']:.2f} ms |",
            f"| Model duty cycle | {paced.get('model_duty_cycle_percent', 0):.2f}% |",
            f"| Session CPU % | {paced.get('session_cpu_percent', 0):.2f}% |",
            f"| Achieved decision rate | {paced.get('achieved_decisions_per_second', 0):.2f} Hz |",
        ]
        if "cpu_percent" in paced:
            lines.append(f"| CPU % per tick (mean) | {paced['cpu_percent']['mean']:.2f}% |")
            lines.append(f"| RAM RSS (mean) | {paced['ram_rss_mb']['mean']:.2f} MB |")
    if "argmax_deploy" in inf:
        ad = inf["argmax_deploy"]["latency_ms"]
        lines += [
            f"| Latency forward-only (median) | {fo.get('median', float('nan')):.4f} ms |",
            f"| Latency max-speed argmax (median) | {ad['median']:.4f} ms |",
            f"| Max-speed throughput mean | {inf['argmax_deploy']['throughput_per_sec']['mean']:.1f} inf/s |",
        ]
    lines += [
        f"| Game decision rate | {data['deployment']['decisions_per_second']:.1f} Hz |",
        f"| Neural MACs / game second | {data['deployment']['macs_per_game_second']:,} |",
        f"| Neural MACs / episode (median steps) | {data['deployment']['macs_per_episode_median_steps']:,} |",
        f"| GFLOPs / second @ 15 Hz | {data['deployment']['gflops_per_game_second']:.6f} |",
        "",
    ]
    ad_res = inf.get("argmax_deploy") or {}
    if "cpu_percent" in ad_res:
        cpu = ad_res["cpu_percent"]
        ram = ad_res["ram_rss_mb"]
        lines += [
            f"| Process CPU % per-run (median) | {cpu['median']:.2f}% |",
            f"| Process CPU % per-run (mean) | {cpu['mean']:.2f}% |",
            f"| Process CPU % (25-run block) | {ad_res.get('block_cpu_percent', 0):.2f}% |",
            f"| Process RAM RSS (median) | {ram['median']:.2f} MB |",
            f"| Process RAM RSS (mean) | {ram['mean']:.2f} MB |",
        ]
    sys_mem = data.get("system_memory")
    if sys_mem:
        lines += [
            f"| System RAM used (snapshot) | {sys_mem['system_ram_percent']:.1f}% |",
            f"| System RAM total | {sys_mem['system_ram_total_mb']:.0f} MB |",
        ]
    burst = data.get("resources", {}).get("sustained_argmax_burst")
    if burst:
        lines += [
            f"| CPU % (sustained burst, 1 core) | {burst['cpu_percent_of_one_core_process_time']:.2f}% |",
            f"| Inferences/s (sustained burst) | {burst['inferences_per_second']:.0f} |",
            f"| RAM RSS after burst | {burst['ram_rss_mb_after_burst']:.2f} MB |",
        ]
    lines.append("")
    if "batched_forward" in data["inference"]:
        b = data["inference"]["batched_forward"]
        bl = b["latency_ms"]
        lines += [
            f"| Batch-{b['batch_size']} latency (median) | {bl['median']:.4f} ms |",
            f"| Batch-{b['batch_size']} MACs / step | {b['macs_per_step']:,} |",
            "",
        ]
    if paced and "cpu_percent" in paced:
        lines += [
            "",
            "## Paced game simulation — resource usage per tick",
            "",
            "| Stat | Inference ms | Slot ms | CPU % | RAM MB |",
            "|------|-------------:|--------:|------:|-------:|",
        ]
        for key in ("mean", "median", "std", "p95"):
            lines.append(
                f"| {key} | {paced['inference_latency_ms'].get(key, 0):.4f} | "
                f"{paced['slot_wall_ms'].get(key, 0):.2f} | "
                f"{paced['cpu_percent'].get(key, 0):.2f} | "
                f"{paced['ram_rss_mb'].get(key, 0):.2f} |"
            )
    if "forward_only" in inf and "argmax_deploy" in inf:
        lines += [
            "",
            "## Max-speed inference latency (ms)",
            "",
            "| Stat | Forward only | Argmax deploy |",
            "|------|-------------:|--------------:|",
        ]
        for key in ("mean", "std", "median", "min", "max", "p95", "ci95_lo", "ci95_hi"):
            fo_v = inf["forward_only"]["latency_ms"].get(key, float("nan"))
            ad_v = inf["argmax_deploy"]["latency_ms"].get(key, float("nan"))
            lines.append(f"| {key} | {fo_v:.4f} | {ad_v:.4f} |")
    ad_res = inf.get("argmax_deploy") or {}
    if "cpu_percent" in ad_res:
        lines += [
            "",
            "## Resource usage (argmax deploy, per timed run)",
            "",
            "| Stat | CPU % (process) | RAM RSS MB (process) |",
            "|------|----------------:|---------------------:|",
        ]
        for key in ("mean", "std", "median", "min", "max", "p95", "ci95_lo", "ci95_hi"):
            cpu = ad_res["cpu_percent"].get(key, float("nan"))
            ram = ad_res["ram_rss_mb"].get(key, float("nan"))
            lines.append(f"| {key} | {cpu:.2f} | {ram:.2f} |")
    lines += [
        "",
        "## Notes",
        "",
        "- **Paced mode** sleeps between decisions so wall time matches real game rate (15 Hz).",
        "- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).",
        "- Full **browser game loop** is still dominated by Selenium/Chrome; paced mode "
        "profiles **policy inference only** at game cadence.",
        "- MAC counts are analytical (Linear layers); activations use a small fixed estimate.",
        "- **CPU % per-run** is process-scoped (`psutil`) between iterations (often 0% when "
        "each forward is under 1 ms). **block_cpu_percent** covers all timed runs together.",
        "- **RAM RSS** is resident set size of the Python process after each timed run.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Profile PPO Dino inference compute.")
    parser.add_argument("--ckpt", type=str, default="checkpoints_scratch_duck/best_duck.pt")
    parser.add_argument("--runs", type=int, default=25,
                        help="Timed iterations per path (after warmup).")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"],
                        nargs="?", const=None)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Also profile batched forward (training uses 4 envs).")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch.set_num_threads (0 = leave default).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--cpu-burst-iters", type=int, default=5000,
                        help="Extra back-to-back forwards for measurable CPU%%.")
    parser.add_argument("--paced", action="store_true",
                        help="Simulate game decision rate (sleep between inferences).")
    parser.add_argument("--decision-hz", type=float, default=DECISIONS_PER_SEC,
                        help="Target decisions/s for --paced mode (default 15 = game).")
    parser.add_argument("--also-max-speed", action="store_true",
                        help="With --paced, also run the back-to-back max-speed profile.")
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ppo = PPO(OBS_DIM, N_ACTIONS, device=device)
    ppo.load(ckpt, load_optim=False)
    ppo.net.eval()
    net = ppo.net

    n_params, n_trainable = count_parameters(net)
    ckpt_bytes = checkpoint_bytes(ckpt)
    weight_mb = n_params * 4 / (1024 * 1024)

    macs_detail = analytical_forward_macs(OBS_DIM, N_ACTIONS)
    linear_macs = macs_detail["linear_mac_total"]
    act_ops = macs_detail["activation_ops_estimate"]
    deploy_ops = macs_detail["softmax_argmax_ops_estimate"]
    macs_per_forward = linear_macs + act_ops
    macs_per_deploy = macs_per_forward + deploy_ops

    obs = np.random.randn(OBS_DIM).astype(np.float32)
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)

    sampler = make_resource_sampler()
    if sampler is None:
        print("[compute_profile] psutil not installed — CPU/RAM per-run metrics skipped. "
              "Install with: pip install psutil")

    sys_mem = system_memory_snapshot()
    mem_before = sampler.proc.memory_info().rss / (1024 * 1024) if sampler else None

    def argmax_body():
        logits, value = net(obs_t)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        _ = int(np.argmax(probs)), value

    inference = {}
    paced_packed = None
    if args.paced:
        print(f"[compute_profile] Paced simulation @ {args.decision_hz:.1f} Hz "
              f"({args.runs} decisions, period {1000.0/args.decision_hz:.2f} ms)")
        paced_raw = time_paced_game_loop(
            argmax_body, sampler, device, args.warmup, args.runs,
            decision_hz=args.decision_hz,
        )
        paced_packed = pack_paced_profile(paced_raw)
        inference["paced_game_argmax"] = paced_packed

    run_forward = run_argmax = None
    if not args.paced or args.also_max_speed:
        run_forward = time_forward_only(net, obs_t, device, args.warmup, args.runs, sampler)
        run_argmax = time_argmax_deploy(net, obs_t, device, args.warmup, args.runs, sampler)
        inference["forward_only"] = pack_run_profile(run_forward)
        inference["argmax_deploy"] = pack_run_profile(run_argmax)

    mem_after = sampler.proc.memory_info().rss / (1024 * 1024) if sampler else None

    batch_results = {}
    if args.batch_size > 1 and (not args.paced or args.also_max_speed):
        obs_b = torch.randn(args.batch_size, OBS_DIM, device=device)
        run_batch = time_batch_forward(net, obs_b, device, args.warmup, args.runs, sampler)
        batch_results = {
            "batch_size": args.batch_size,
            **pack_run_profile(run_batch),
            "macs_per_step": macs_per_forward * args.batch_size,
        }

    # Reference env steps/ep from scratch_duck_best_duck_30_v2 benchmark when present.
    median_steps_ref = 1100
    ref_metrics = Path(__file__).parent / "results" / "scratch_duck_best_duck_30_v2" / "metrics.json"
    mean_steps_ref = median_steps_ref
    if ref_metrics.is_file():
        with open(ref_metrics, encoding="utf-8") as f:
            ref = json.load(f)
        st = ref["deterministic"]["aggregate"]["step_stats"]
        median_steps_ref = int(st["median"])
        mean_steps_ref = int(round(st["mean"]))

    if batch_results:
        inference["batched_forward"] = batch_results

    resources = {}
    if sampler is not None and (not args.paced or args.also_max_speed):
        resources["sustained_argmax_burst"] = measure_sustained_resources(
            argmax_body, sampler, inner_iters=args.cpu_burst_iters
        )

    deploy = {
        "decisions_per_second": DECISIONS_PER_SEC,
        "frames_per_step": FRAMES_PER_STEP,
        "game_hz": GAME_HZ,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "feature_frames": 4,
        "features_per_frame": FEATURES_PER_FRAME,
        "macs_per_inference_deploy": macs_per_deploy,
        "macs_per_game_second": int(macs_per_deploy * DECISIONS_PER_SEC),
        "macs_per_minute_gameplay": int(macs_per_deploy * DECISIONS_PER_SEC * 60),
        "macs_per_episode_median_steps": int(macs_per_deploy * median_steps_ref),
        "macs_per_episode_mean_steps": int(macs_per_deploy * mean_steps_ref),
        "gflops_per_game_second": round(macs_per_deploy * DECISIONS_PER_SEC * 2 / 1e9, 6),
        "median_env_steps_per_episode_ref": median_steps_ref,
        "mean_env_steps_per_episode_ref": mean_steps_ref,
    }

    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_device": device,
        "cpu_count_logical": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(__file__), "results", f"compute_{run_ts}"
    )
    os.makedirs(out_dir, exist_ok=True)

    result = {
        "run_timestamp": run_ts,
        "checkpoint": ckpt,
        "methodology": {
            "n_warmup": args.warmup,
            "n_timed_runs": args.runs,
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
            "mode": "paced_game" if args.paced and not args.also_max_speed else (
                "paced_game+max_speed" if args.paced else "max_speed"
            ),
            "decision_hz": args.decision_hz if args.paced else None,
            "description": (
                f"{args.runs} ticks at {args.decision_hz:.1f} Hz (paced game simulation)"
                if args.paced else
                f"{args.runs} timed iterations after {args.warmup} warmup passes; "
                "perf_counter wall time including host/device sync on CUDA."
            ),
            "resource_sampling": "psutil process CPU% and RSS per timed iteration"
            if sampler else "unavailable (install psutil)",
        },
        "hardware": hardware,
        "system_memory": sys_mem,
        "model": {
            "architecture": "ActorCritic MLP (2x128 Tanh + policy/value heads)",
            "hidden_size": 128,
            "parameters_total": n_params,
            "parameters_trainable": n_trainable,
            "checkpoint_size_bytes": ckpt_bytes,
            "checkpoint_size_mb": ckpt_bytes / (1024 * 1024),
            "weight_memory_fp32_mb": weight_mb,
            "process_rss_mb_before": round(mem_before, 3) if mem_before is not None else None,
            "process_rss_mb_after": round(mem_after, 3) if mem_after is not None else None,
        },
        "analytical": {
            **macs_detail,
            "macs_forward_with_activations": macs_per_forward,
            "macs_deploy_with_softmax_argmax": macs_per_deploy,
            "flops_forward_estimate": macs_per_forward * 2,
            "flops_deploy_estimate": macs_per_deploy * 2,
        },
        "inference": inference,
        "resources": resources,
        "deployment": deploy,
    }

    metrics_path = os.path.join(out_dir, "compute_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    report_path = os.path.join(out_dir, "compute_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(build_report(result))

    # Console summary
    print()
    print("=" * 60)
    print("  COMPUTE PROFILE")
    print("-" * 60)
    print(f"  Checkpoint     : {ckpt}")
    print(f"  Device         : {device}")
    print(f"  Parameters     : {n_params:,}")
    print(f"  MACs / deploy  : {macs_per_deploy:,}")

    if paced_packed is not None:
        inf = paced_packed["inference_latency_ms"]
        slot = paced_packed["slot_wall_ms"]
        print(f"  Mode           : paced @ {args.decision_hz:.1f} Hz ({args.runs} ticks)")
        print(f"  Inference med  : {inf['median']:.4f} ms  mean {inf['mean']:.4f} ms")
        print(f"  Slot period med: {slot['median']:.2f} ms  (target {paced_packed['period_ms_target']:.2f})")
        print(f"  Model duty     : {paced_packed.get('model_duty_cycle_percent', 0):.2f}% of slot")
        print(f"  Session CPU %  : {paced_packed.get('session_cpu_percent', 0):.2f}%")
        if "cpu_percent" in paced_packed:
            cpu = paced_packed["cpu_percent"]
            ram = paced_packed["ram_rss_mb"]
            print(f"  CPU % per-tick : mean {cpu['mean']:.2f}%  median {cpu['median']:.2f}%")
            print(f"  RAM RSS        : mean {ram['mean']:.2f} MB  median {ram['median']:.2f} MB")
        print(f"  Achieved rate  : {paced_packed.get('achieved_decisions_per_second', 0):.2f} Hz")

    if "argmax_deploy" in inference:
        ad = inference["argmax_deploy"]["latency_ms"]
        print(f"  [max-speed] latency mean : {ad['mean']:.4f} ms")
        print(f"  [max-speed] throughput   : "
              f"{inference['argmax_deploy']['throughput_per_sec']['mean']:.1f} inf/s")

    print(f"  @ 15 Hz MACs/s : {deploy['macs_per_game_second']:,}")
    if sys_mem:
        print(f"  System RAM     : {sys_mem['system_ram_percent']:.1f}% of "
              f"{sys_mem['system_ram_total_mb']:.0f} MB")
    burst = resources.get("sustained_argmax_burst")
    if burst:
        print(f"  CPU % (burst)  : {burst['cpu_percent_of_one_core_process_time']:.2f}%")
    print("-" * 60)
    print(f"  Metrics -> {metrics_path}")
    print(f"  Report  -> {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
