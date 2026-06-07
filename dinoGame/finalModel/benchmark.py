import argparse
import functools
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

print = functools.partial(print, flush=True)
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dino_env import FEATURES_PER_FRAME, N_ACTIONS, OBS_DIM, DinoEnv
from ppo_agent import PPO

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
ACTION_NAMES = ["noop", "jump", "duck"]
_PALETTE = {"noop": "#4C72B0", "jump": "#DD8452", "duck": "#55A868"}
_DARK = {"noop": "#1A3B6E", "jump": "#A0522D", "duck": "#1E6E3A"}


def _ci_t(data, confidence=0.95):
    n = len(data)
    if n < 2:
        m = float(np.mean(data)) if n else float("nan")
        return m, m
    res = _scipy_stats.t.interval(
        confidence,
        df=n - 1,
        loc=np.mean(data),
        scale=_scipy_stats.sem(data),
    )
    return float(res[0]), float(res[1])


def _ci_normal(data, confidence=0.95):
    n = len(data)
    if n < 2:
        m = float(np.mean(data)) if n else float("nan")
        return m, m
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(confidence, 1.960)
    se = float(np.std(data, ddof=1) / math.sqrt(n))
    m = float(np.mean(data))
    return m - z * se, m + z * se


def compute_ci(data, confidence=0.95):
    fn = _ci_t if _HAS_SCIPY else _ci_normal
    return fn(data, confidence)


def descriptive_stats(arr):
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0}
    pcts = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    ci_lo, ci_hi = compute_ci(a)
    std = float(np.std(a, ddof=1)) if n > 1 else 0.0
    mn = float(np.mean(a))
    return {
        "n": n,
        "mean": mn,
        "std": std,
        "sem": std / math.sqrt(n) if n > 1 else 0.0,
        "min": float(np.min(a)),
        "p1": float(pcts[0]),
        "p5": float(pcts[1]),
        "p10": float(pcts[2]),
        "p25": float(pcts[3]),
        "median": float(pcts[4]),
        "p75": float(pcts[5]),
        "p90": float(pcts[6]),
        "p95": float(pcts[7]),
        "p99": float(pcts[8]),
        "max": float(np.max(a)),
        "iqr": float(pcts[5] - pcts[3]),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "cv": std / abs(mn) if abs(mn) > 1e-9 else float("nan"),
        "skewness": float(_scipy_stats.skew(a)) if _HAS_SCIPY else float("nan"),
        "kurtosis": float(_scipy_stats.kurtosis(a)) if _HAS_SCIPY else float("nan"),
    }


def survival_rates(scores, thresholds=(100, 200, 300, 400, 500, 750, 1000)):
    n = len(scores)
    if n == 0:
        return {}
    s = np.asarray(scores)
    return {f"sr_{t}": float(np.sum(s >= t) / n * 100) for t in thresholds}


def action_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    return float(-np.sum(p * np.log(p + 1e-12)))


class EpisodeRecord:
    __slots__ = (
        "episode",
        "score",
        "ep_return",
        "steps",
        "wall_time",
        "action_counts",
        "value_estimates",
        "final_speed",
        "mode",
        "death_cause",
    )

    def __init__(
        self,
        episode,
        score,
        ep_return,
        steps,
        wall_time,
        action_counts,
        value_estimates,
        final_speed,
        mode,
        death_cause="",
    ):
        for k in self.__slots__:
            object.__setattr__(self, k, locals()[k])

    def to_dict(self):
        ac = self.action_counts
        total = max(1, int(ac.sum()))
        ve = self.value_estimates
        return {
            "episode": self.episode,
            "score": int(self.score),
            "return": round(float(self.ep_return), 4),
            "steps": int(self.steps),
            "wall_time_s": round(float(self.wall_time), 3),
            "mode": self.mode,
            "final_speed": round(float(self.final_speed), 3),
            "death_cause": self.death_cause or "unknown",
            "action_counts": {
                "noop": int(ac[0]),
                "jump": int(ac[1]),
                "duck": int(ac[2]),
            },
            "action_fracs": {
                "noop": round(float(ac[0] / total), 4),
                "jump": round(float(ac[1] / total), 4),
                "duck": round(float(ac[2] / total), 4),
            },
            "value_mean": round(float(np.mean(ve)), 4) if ve else None,
            "value_std": round(float(np.std(ve)), 4) if len(ve) > 1 else None,
            "value_min": round(float(np.min(ve)), 4) if ve else None,
            "value_max": round(float(np.max(ve)), 4) if ve else None,
        }


def _run_one_episode(env, ppo, ep, n_episodes, argmax, verbose=True):
    if verbose:
        print(f"  ep {ep:3d}/{n_episodes}  running...", flush=True)
    obs = env.reset()
    done = False
    ep_ret = 0.0
    steps = 0
    action_counts = np.zeros(3, dtype=np.int64)
    value_estimates = []
    final_speed = 0.0
    read_failures = 0
    t0 = time.time()
    while not done:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(ppo.device)
        with torch.no_grad():
            logits, value = ppo.net(obs_t)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            value_estimates.append(float(value.item()))
            action = (
                int(np.argmax(probs))
                if argmax
                else int(np.random.choice(N_ACTIONS, p=probs))
            )
        action_counts[action] += 1
        obs, r, done, info = env.step(action)
        if info.get("read_state_failed"):
            read_failures += 1
        ep_ret += r
        steps += 1
        final_speed = float(info.get("speed", 0.0))
    score = int(info.get("score", 0))
    wall = time.time() - t0
    death_cause = str(info.get("death_obstacle", "") or "unknown")
    if read_failures and death_cause == "unknown":
        death_cause = "read_state_failed"
    return (
        EpisodeRecord(
            episode=ep,
            score=score,
            ep_return=ep_ret,
            steps=steps,
            wall_time=wall,
            action_counts=action_counts,
            value_estimates=value_estimates,
            final_speed=final_speed,
            mode="deterministic" if argmax else "stochastic",
            death_cause=death_cause,
        ),
        wall,
        read_failures,
    )


def warmup_env(env, n_steps=90):
    obs = env.reset()
    for _ in range(n_steps):
        obs, _, done, _ = env.step(0)
        if done:
            obs = env.reset()


def _episode_needs_fresh_browser(rec, bad_score=300, bad_steps=100):
    return rec.score < bad_score or rec.steps < bad_steps


def run_episodes(
    env,
    ppo,
    n_episodes,
    mode,
    verbose=True,
    browser_restart_every=0,
    make_env=None,
    adaptive_restart_streak=2,
):
    records = []
    best = -1
    argmax = mode == "deterministic"
    restart_every = max(0, int(browser_restart_every))
    bad_streak = 0
    streak_limit = max(1, int(adaptive_restart_streak))

    def _restart_browser(reason: str):
        nonlocal env, bad_streak
        if make_env is None:
            return
        if verbose:
            print(f"[benchmark] Restarting browser ({reason}) ...")
        try:
            env.close()
        except Exception:
            pass
        env = make_env()
        bad_streak = 0

    def _maybe_restart_browser(ep_just_finished, rec):
        nonlocal bad_streak
        if make_env is None:
            return
        if _episode_needs_fresh_browser(rec):
            bad_streak += 1
        else:
            bad_streak = 0
        if bad_streak >= streak_limit:
            _restart_browser(
                f"{bad_streak} consecutive weak episodes (latest score {rec.score})"
            )
            return
        if rec.steps >= 1400:
            _restart_browser(
                f"cooldown after long episode (score {rec.score}, {rec.steps} steps)"
            )
            return
        if restart_every > 0 and ep_just_finished % restart_every == 0:
            _restart_browser(f"scheduled every {restart_every} episodes")

    try:
        for ep in range(1, n_episodes + 1):
            try:
                rec, wall, read_failures = _run_one_episode(
                    env, ppo, ep, n_episodes, argmax, verbose=verbose
                )
            except Exception as exc:
                exc_name = type(exc).__name__
                if make_env is None or "InvalidSession" not in exc_name:
                    raise
                print(f"[benchmark] {exc_name} on ep {ep} — relaunching browser ...")
                try:
                    env.close()
                except Exception:
                    pass
                env = make_env()
                rec, wall, read_failures = _run_one_episode(
                    env, ppo, ep, n_episodes, argmax, verbose=verbose
                )
            best = max(best, rec.score)
            records.append(rec)
            if verbose:
                frac = rec.action_counts / max(1, rec.action_counts.sum())
                extra = ""
                if read_failures:
                    extra = f"  read_fail={read_failures}"
                print(
                    f"  ep {ep:3d}/{n_episodes}  score {rec.score:5d} (best {best:5d})  "
                    f"return {rec.ep_return:7.2f}  steps {rec.steps:5d}  "
                    f"n/j/d {frac[0]:.2f}/{frac[1]:.2f}/{frac[2]:.2f}  "
                    f"spd {rec.final_speed:.1f}  {wall:.1f}s{extra}"
                )
            _maybe_restart_browser(ep, rec)
    except KeyboardInterrupt:
        print(
            f"\n[benchmark] Interrupted after {len(records)} completed episode(s) "
            f"— saving partial results."
        )
    return records


def aggregate(records):
    if not records:
        return {}
    scores = [r.score for r in records]
    returns = [r.ep_return for r in records]
    steps = [r.steps for r in records]
    times = [r.wall_time for r in records]
    speeds = [r.final_speed for r in records]
    vals = [v for r in records for v in r.value_estimates]
    total_ac = np.zeros(3, dtype=np.int64)
    for r in records:
        total_ac += r.action_counts
    total_steps = max(1, int(total_ac.sum()))
    per_ep_entropy = [action_entropy(r.action_counts) for r in records]
    death_causes = {}
    for r in records:
        dc = getattr(r, "death_cause", "") or "unknown"
        death_causes[dc] = death_causes.get(dc, 0) + 1
    n_records = max(1, len(records))
    ptero_deaths = death_causes.get("PTERODACTYL", 0)
    return {
        "score_stats": descriptive_stats(scores),
        "return_stats": descriptive_stats(returns),
        "step_stats": descriptive_stats(steps),
        "time_stats": descriptive_stats(times),
        "speed_stats": descriptive_stats(speeds),
        "value_stats": descriptive_stats(vals) if vals else {},
        "action_distribution": {
            act: {
                "count": int(total_ac[i]),
                "frac": round(float(total_ac[i] / total_steps), 4),
            }
            for i, act in enumerate(ACTION_NAMES)
        },
        "global_action_entropy": action_entropy(total_ac),
        "mean_per_episode_action_entropy": float(np.mean(per_ep_entropy)),
        "survival_rates": survival_rates(scores),
        "death_causes": death_causes,
        "pterodactyl_death_frac": float(ptero_deaths / n_records),
    }


def make_figures(records_det, records_sto, out_dir, train_log_path=None):
    if not HAS_MPL:
        print("[benchmark] matplotlib not available — skipping figures.")
        return
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    scores_det = [r.score for r in records_det]
    scores_sto = [r.score for r in records_sto] if records_sto else []
    fig, ax = plt.subplots(figsize=(9, 5))
    all_scores = scores_det + scores_sto
    hi = max(max(all_scores, default=1) * 1.05 + 10, 50)
    bins = np.linspace(0, hi, 35)
    ax.hist(
        scores_det,
        bins=bins,
        alpha=0.72,
        label="Deterministic",
        color=_PALETTE["noop"],
        edgecolor="white",
        linewidth=0.4,
    )
    if scores_sto:
        ax.hist(
            scores_sto,
            bins=bins,
            alpha=0.60,
            label="Stochastic",
            color=_PALETTE["jump"],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axvline(
        np.mean(scores_det),
        color=_DARK["noop"],
        linestyle="--",
        linewidth=1.8,
        label=f"Det. μ = {np.mean(scores_det):.1f}",
    )
    if scores_sto:
        ax.axvline(
            np.mean(scores_sto),
            color=_DARK["jump"],
            linestyle="--",
            linewidth=1.8,
            label=f"Sto. μ = {np.mean(scores_sto):.1f}",
        )
    ax.set_xlabel("Episode Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution — PPO Dino Agent", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_distribution.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, sc, col in [
        ("Deterministic", scores_det, _PALETTE["noop"]),
        ("Stochastic", scores_sto, _PALETTE["jump"]),
    ]:
        if not sc:
            continue
        s = np.sort(sc)
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, cdf, where="post", label=label, color=col, linewidth=2.2)
        ax.axvline(
            float(np.median(s)), color=col, linestyle=":", alpha=0.65, linewidth=1.2
        )
    for thresh in (100, 200, 300, 400, 500):
        ax.axvline(thresh, color="gray", linestyle="--", alpha=0.25, linewidth=0.8)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_ylabel("Cumulative Probability", fontsize=12)
    ax.set_title("Score Empirical CDF", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_cdf.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    ep_det = [r.episode for r in records_det]
    ax.plot(
        ep_det, scores_det, alpha=0.45, color=_PALETTE["noop"], linewidth=0.9, zorder=1
    )
    ax.scatter(
        ep_det,
        scores_det,
        color=_PALETTE["noop"],
        s=18,
        alpha=0.80,
        zorder=2,
        label="Deterministic",
    )
    w = max(3, len(scores_det) // 6)
    if len(scores_det) >= w:
        roll = np.convolve(scores_det, np.ones(w) / w, mode="valid")
        ax.plot(
            ep_det[w - 1 :],
            roll,
            color=_DARK["noop"],
            linewidth=2.0,
            zorder=3,
            label=f"Rolling mean (w={w})",
        )
    if scores_sto:
        ep_sto = [r.episode for r in records_sto]
        ax.plot(
            ep_sto,
            scores_sto,
            alpha=0.35,
            color=_PALETTE["jump"],
            linewidth=0.9,
            zorder=1,
        )
        ax.scatter(
            ep_sto,
            scores_sto,
            color=_PALETTE["jump"],
            s=18,
            alpha=0.70,
            zorder=2,
            label="Stochastic",
        )
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Score per Episode", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "score_progression.png"), dpi=150)
    plt.close(fig)
    datasets = [("Deterministic", records_det)]
    if records_sto:
        datasets.append(("Stochastic", records_sto))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    for ax, (label, recs) in zip(axes, datasets):
        total_ac = np.zeros(3, dtype=np.int64)
        for r in recs:
            total_ac += r.action_counts
        total = max(1, int(total_ac.sum()))
        fracs = total_ac / total
        wedge_cols = [_PALETTE[a] for a in ACTION_NAMES]
        wedges, texts, autotexts = ax.pie(
            fracs,
            labels=ACTION_NAMES,
            colors=wedge_cols,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        )
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title(f"Action Distribution\n({label})", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "action_distribution.png"), dpi=150)
    plt.close(fig)
    vals_det = [v for r in records_det for v in r.value_estimates]
    if vals_det:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            vals_det,
            bins=60,
            color=_PALETTE["noop"],
            alpha=0.80,
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(
            float(np.mean(vals_det)),
            color=_DARK["noop"],
            linestyle="--",
            linewidth=1.8,
            label=f"Mean = {np.mean(vals_det):.3f}",
        )
        ax.axvline(
            float(np.median(vals_det)),
            color=_DARK["noop"],
            linestyle=":",
            linewidth=1.4,
            label=f"Median = {np.median(vals_det):.3f}",
        )
        ax.set_xlabel("V(s)  —  Critic Output", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(
            "Value Function Distribution (Deterministic Mode)",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "value_distribution.png"), dpi=150)
        plt.close(fig)
    if train_log_path and os.path.exists(train_log_path):
        rows = _load_train_log(train_log_path)
        if rows:
            updates = [r["update"] for r in rows]
            mean_sc = [r.get("mean_score", float("nan")) for r in rows]
            max_sc = [r.get("max_score", 0) for r in rows]
            mean_ln = [r.get("mean_len", float("nan")) for r in rows]
            entropies = [r.get("entropy", float("nan")) for r in rows]
            v_losses = [r.get("v_loss", float("nan")) for r in rows]
            pi_losses = [r.get("pi_loss", float("nan")) for r in rows]
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("PPO Training Curve", fontsize=14, fontweight="bold")
            axes[0, 0].plot(
                updates,
                mean_sc,
                color=_PALETTE["noop"],
                linewidth=1.6,
                label="Mean score",
            )
            axes[0, 0].plot(
                updates,
                max_sc,
                color=_PALETTE["jump"],
                linewidth=1.0,
                alpha=0.7,
                label="Max score",
            )
            axes[0, 0].set_title("Score vs. Updates")
            axes[0, 0].set_xlabel("PPO Update")
            axes[0, 0].set_ylabel("Game Score")
            axes[0, 0].legend(fontsize=9)
            axes[0, 0].grid(alpha=0.2)
            axes[0, 1].plot(updates, mean_ln, color=_PALETTE["duck"], linewidth=1.6)
            axes[0, 1].set_title("Mean Episode Length")
            axes[0, 1].set_xlabel("PPO Update")
            axes[0, 1].set_ylabel("Steps")
            axes[0, 1].grid(alpha=0.2)
            axes[0, 2].plot(updates, entropies, color="#8172B2", linewidth=1.6)
            axes[0, 2].set_title("Policy Entropy")
            axes[0, 2].set_xlabel("PPO Update")
            axes[0, 2].set_ylabel("Entropy (nats)")
            axes[0, 2].grid(alpha=0.2)
            axes[1, 0].plot(updates, v_losses, color="#C44E52", linewidth=1.6)
            axes[1, 0].set_title("Value Function Loss (MSE)")
            axes[1, 0].set_xlabel("PPO Update")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].grid(alpha=0.2)
            axes[1, 1].plot(updates, pi_losses, color="#937860", linewidth=1.6)
            axes[1, 1].set_title("Policy Loss")
            axes[1, 1].set_xlabel("PPO Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].grid(alpha=0.2)
            sps = [r.get("sps", float("nan")) for r in rows]
            axes[1, 2].plot(updates, sps, color="#64B5CD", linewidth=1.6)
            axes[1, 2].set_title("Samples per Second")
            axes[1, 2].set_xlabel("PPO Update")
            axes[1, 2].set_ylabel("SPS")
            axes[1, 2].grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "training_curve.png"), dpi=150)
            plt.close(fig)
    print(f"[benchmark] Figures saved -> {fig_dir}")


def _load_train_log(path):
    rows = []
    if not (path and os.path.exists(path)):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _fmt(v, decimals=2):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}f}"
    return str(v)


def _md_table_row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def write_report(
    out_dir,
    ckpt_path,
    n_det,
    n_sto,
    det_agg,
    sto_agg,
    records_det,
    records_sto,
    run_ts,
    train_log_rows=None,
    model_params=None,
):
    lines = []
    A = lines.append
    A("# PPO Chrome Dino Agent — Benchmark Report")
    A("")
    A(f"**Generated:** {run_ts}  ")
    A(f"**Checkpoint:** `{ckpt_path}`  ")
    A(f"**Deterministic episodes:** {n_det}  ")
    if n_sto:
        A(f"**Stochastic episodes:** {n_sto}  ")
    if model_params is not None:
        A(f"**Model parameters:** {model_params:,}  ")
    A(
        f"**scipy available:** {'Yes' if _HAS_SCIPY else 'No (CI via normal approximation)'}  "
    )
    A("")
    A("---")
    A("")
    ss_d = det_agg.get("score_stats", {})
    mean_d = ss_d.get("mean", float("nan"))
    std_d = ss_d.get("std", float("nan"))
    med_d = ss_d.get("median", float("nan"))
    max_d = ss_d.get("max", float("nan"))
    A("## Abstract")
    A("")
    A(
        "We evaluate a Proximal Policy Optimisation (PPO) agent trained to play the "
        "Chrome Dinosaur (T-Rex Runner) browser game. The agent maps a 48-dimensional "
        "observation (12 state features × 4-frame stack) to one of three discrete "
        "actions — no-op, jump, duck — at a nominal decision rate of 15 Hz. "
        "Training uses four parallel browser instances, a 256-step rollout buffer per "
        "environment, and standard PPO hyperparameters (γ = 0.99, λ_GAE = 0.95, "
        "ε_clip = 0.2). "
        f"The deterministic policy achieves a mean score of **{_fmt(mean_d)} ± {_fmt(std_d)}** "
        f"(median {_fmt(med_d, 1)}, max {_fmt(max_d, 0)}) "
        f"across {n_det} evaluation episodes. "
        "This report presents full descriptive statistics, confidence intervals, "
        "survival-rate profiles, action-entropy analysis, and critic value statistics."
    )
    A("")
    A("---")
    A("")
    A("## 1. Experimental Setup")
    A("")
    A("### 1.1 Environment")
    A("")
    A("| Property | Value |")
    A("|----------|-------|")
    A("| Game | Chrome T-Rex Runner (local `file://` HTML) |")
    A("| Observation space | ℝ⁴⁸ (12 features × 4 frame stack) |")
    A("| Action space | Discrete(3): {no-op, jump, duck} |")
    A("| Decision frequency | ~15 Hz (4 game frames @ 60 Hz per step) |")
    A(
        "| State features | dino_y, jumping, ducking, speed, o1_dx, o1_y, o1_w, o1_h, o2_dx, o2_y, o2_w, o2_h |"
    )
    A("| Reward: score delta | +0.01 × Δscore |")
    A("| Reward: obstacle pass | +1.0 |")
    A("| Reward: jump cost | −0.01 per jump action |")
    A("| Reward: death | −10.0 (terminal) |")
    A("| Danger range normalisation | 200 px → [0, 1.5] |")
    A("")
    A("### 1.2 Policy Architecture")
    A("")
    A("| Component | Specification |")
    A("|-----------|--------------|")
    A("| Type | MLP Actor-Critic (shared trunk) |")
    A("| Hidden layers | 2 × 128 units |")
    A("| Activation | Tanh |")
    A("| Policy head | Linear(128 → 3) + Softmax |")
    A("| Value head | Linear(128 → 1) |")
    if model_params is not None:
        A(f"| Total parameters | {model_params:,} |")
    A("")
    A("### 1.3 Training Hyperparameters")
    A("")
    A("| Hyperparameter | Value |")
    A("|----------------|-------|")
    A("| Algorithm | PPO (Clipped Surrogate Objective) |")
    A("| Parallel environments | 4 |")
    A("| Rollout length | 256 steps / env |")
    A("| Samples per update | 1,024 |")
    A("| Learning rate | 3 × 10⁻⁴ (Adam) |")
    A("| Discount factor γ | 0.99 |")
    A("| GAE λ | 0.95 |")
    A("| Clip ε | 0.2 |")
    A("| Entropy coefficient | 0.01 |")
    A("| Value coefficient | 0.5 |")
    A("| Max gradient norm | 0.5 |")
    A("| SGD epochs per update | 4 |")
    A("| Mini-batch size | 128 |")
    A("| KL early-stop threshold | 0.015 |")
    A("")
    A("### 1.4 Evaluation Protocol")
    A("")
    A("| Setting | Value |")
    A("|---------|-------|")
    A(f"| Deterministic episodes | {n_det} |")
    if n_sto:
        A(f"| Stochastic episodes | {n_sto} |")
    A("| Deterministic inference | Argmax over policy logits |")
    A("| Stochastic inference | Categorical sample from softmax |")
    A("| Step pause | 0 s (max speed) |")
    A("| Score metric | `Runner.distanceRan × COEFFICIENT` (in-game) |")
    A("")
    A("---")
    A("")
    A("## 2. Performance Results")
    A("")
    A("### 2.1 Score — Descriptive Statistics")
    A("")
    sto_agg_ok = bool(sto_agg and sto_agg.get("score_stats"))
    ss_s = sto_agg.get("score_stats", {}) if sto_agg_ok else {}
    hdr = ["Metric", "Deterministic"]
    sep = [":------", "------------:"]
    if sto_agg_ok:
        hdr.append("Stochastic")
        sep.append("----------:")
    A(_md_table_row(hdr))
    A(_md_table_row(sep))

    def stat_row(label, key, dec=2):
        row = [label, _fmt(ss_d.get(key, float("nan")), dec)]
        if sto_agg_ok:
            row.append(_fmt(ss_s.get(key, float("nan")), dec))
        A(_md_table_row(row))

    stat_row("N episodes", "n", 0)
    stat_row("Mean", "mean", 2)
    stat_row("Std. deviation", "std", 2)
    stat_row("Std. error of mean", "sem", 3)
    stat_row("95% CI (lower)", "ci95_lo", 2)
    stat_row("95% CI (upper)", "ci95_hi", 2)
    stat_row("Median", "median", 1)
    stat_row("IQR (P25 – P75)", "iqr", 1)
    stat_row("Min", "min", 0)
    stat_row("P1", "p1", 1)
    stat_row("P5", "p5", 1)
    stat_row("P10", "p10", 1)
    stat_row("P25", "p25", 1)
    stat_row("P75", "p75", 1)
    stat_row("P90", "p90", 1)
    stat_row("P95", "p95", 1)
    stat_row("P99", "p99", 1)
    stat_row("Max", "max", 0)
    stat_row("Coeff. of variation", "cv", 4)
    stat_row("Skewness", "skewness", 3)
    stat_row("Excess kurtosis", "kurtosis", 3)
    A("")
    A("### 2.2 Threshold Survival Rates")
    A("")
    A(
        "> Percentage of episodes in which the agent reached or exceeded each score threshold."
    )
    A("")
    hdr2 = ["Score threshold", "Deterministic %"]
    sep2 = [":---------------", "---------------:"]
    if sto_agg_ok:
        hdr2.append("Stochastic %")
        sep2.append("------------:")
    A(_md_table_row(hdr2))
    A(_md_table_row(sep2))
    for t in (100, 200, 300, 400, 500, 750, 1000):
        key = f"sr_{t}"
        det_v = det_agg.get("survival_rates", {}).get(key, 0.0)
        row = [f"≥ {t:,}", f"{det_v:.1f}%"]
        if sto_agg_ok:
            sto_v = sto_agg.get("survival_rates", {}).get(key, 0.0)
            row.append(f"{sto_v:.1f}%")
        A(_md_table_row(row))
    A("")
    A("### 2.3 Episode Length and Shaped Return")
    A("")
    ls_d = det_agg.get("step_stats", {})
    rs_d = det_agg.get("return_stats", {})
    ts_d = det_agg.get("time_stats", {})
    sp_d = det_agg.get("speed_stats", {})
    ls_s = sto_agg.get("step_stats", {}) if sto_agg_ok else {}
    rs_s = sto_agg.get("return_stats", {}) if sto_agg_ok else {}
    ts_s = sto_agg.get("time_stats", {}) if sto_agg_ok else {}
    sp_s = sto_agg.get("speed_stats", {}) if sto_agg_ok else {}
    hdr3 = ["Metric", "Deterministic"]
    sep3 = [":------", "------------:"]
    if sto_agg_ok:
        hdr3.append("Stochastic")
        sep3.append("----------:")
    A(_md_table_row(hdr3))
    A(_md_table_row(sep3))

    def ep_row(label, d_stat, s_stat, key, dec=2):
        row = [label, _fmt(d_stat.get(key, float("nan")), dec)]
        if sto_agg_ok:
            row.append(_fmt(s_stat.get(key, float("nan")), dec))
        A(_md_table_row(row))

    ep_row("Mean steps", ls_d, ls_s, "mean", 1)
    ep_row("Std steps", ls_d, ls_s, "std", 1)
    ep_row("Median steps", ls_d, ls_s, "median", 1)
    ep_row("Max steps", ls_d, ls_s, "max", 0)
    ep_row("Mean shaped return", rs_d, rs_s, "mean", 3)
    ep_row("Std shaped return", rs_d, rs_s, "std", 3)
    ep_row("Max shaped return", rs_d, rs_s, "max", 3)
    ep_row("Mean wall-time (s)", ts_d, ts_s, "mean", 2)
    ep_row("Max wall-time (s)", ts_d, ts_s, "max", 2)
    ep_row("Mean final speed", sp_d, sp_s, "mean", 3)
    ep_row("Max final speed", sp_d, sp_s, "max", 3)
    A("")
    A("---")
    A("")
    A("## 3. Behavioural Analysis")
    A("")
    A("### 3.1 Global Action Distribution")
    A("")
    hdr4 = ["Action", "Det. count", "Det. %"]
    sep4 = [":------", "----------:", "------:"]
    if sto_agg_ok:
        hdr4 += ["Sto. count", "Sto. %"]
        sep4 += ["----------:", "------:"]
    A(_md_table_row(hdr4))
    A(_md_table_row(sep4))
    for act in ACTION_NAMES:
        dd = det_agg.get("action_distribution", {}).get(act, {})
        row = [act, f"{dd.get('count',0):,}", f"{dd.get('frac',0)*100:.2f}%"]
        if sto_agg_ok:
            ds = sto_agg.get("action_distribution", {}).get(act, {})
            row += [f"{ds.get('count',0):,}", f"{ds.get('frac',0)*100:.2f}%"]
        A(_md_table_row(row))
    A("")
    det_gent = det_agg.get("global_action_entropy", float("nan"))
    det_ment = det_agg.get("mean_per_episode_action_entropy", float("nan"))
    A(f"**Deterministic — global action entropy:** {_fmt(det_gent, 4)} nats  ")
    A(
        f"**Deterministic — mean per-episode action entropy:** {_fmt(det_ment, 4)} nats  "
    )
    if sto_agg_ok:
        sto_gent = sto_agg.get("global_action_entropy", float("nan"))
        sto_ment = sto_agg.get("mean_per_episode_action_entropy", float("nan"))
        A(f"**Stochastic — global action entropy:** {_fmt(sto_gent, 4)} nats  ")
        A(
            f"**Stochastic — mean per-episode action entropy:** {_fmt(sto_ment, 4)} nats  "
        )
    A(f"*(Maximum possible entropy for 3 actions: {math.log(3):.4f} nats)*")
    A("")
    vs_d = det_agg.get("value_stats", {})
    if vs_d:
        A("### 3.2 Critic Value Estimates (Deterministic Mode)")
        A("")
        A("| Statistic | Value |")
        A("|:----------|------:|")
        for key, label in [
            ("n", "Total step estimates"),
            ("mean", "Mean V(s)"),
            ("std", "Std V(s)"),
            ("min", "Min V(s)"),
            ("p5", "P5 V(s)"),
            ("median", "Median V(s)"),
            ("p95", "P95 V(s)"),
            ("max", "Max V(s)"),
        ]:
            A(_md_table_row([label, _fmt(vs_d.get(key, float("nan")))]))
        A("")
    dc_d = det_agg.get("death_causes", {})
    if dc_d:
        A("### 3.3 Death-Cause Breakdown (Deterministic Mode)")
        A("")
        A(
            "> Obstacle type the agent collided with at episode end. "
            "`PTERODACTYL` deaths are the ones a working duck is meant to prevent."
        )
        A("")
        A("| Death cause | Episodes | % |")
        A("|:------------|---------:|--:|")
        total_dc = max(1, sum(dc_d.values()))
        for cause, cnt in sorted(dc_d.items(), key=lambda kv: -kv[1]):
            A(_md_table_row([cause, cnt, f"{cnt/total_dc*100:.1f}%"]))
        A("")
        A(
            f"**Pterodactyl-death rate:** "
            f"{det_agg.get('pterodactyl_death_frac', 0.0)*100:.1f}%  "
        )
        A("")
    A("---")
    A("")
    if train_log_rows:
        valid = [
            r
            for r in train_log_rows
            if not math.isnan(r.get("mean_score", float("nan")))
        ]
        A("## 4. Training History")
        A("")
        A(
            f"*Based on `checkpoints/train_log.jsonl` — "
            f"{len(train_log_rows)} PPO updates logged.*"
        )
        A("")
        if valid:
            n = len(train_log_rows)
            a, b = n // 3, 2 * n // 3
            A("### 4.1 Training Phase Summary")
            A("")
            A("| Phase | Updates | Peak score | Mean score | Mean ep. len | Mean SPS |")
            A("|:------|--------:|-----------:|-----------:|-------------:|---------:|")
            for start, end, phase in [
                (1, a, "Early"),
                (a + 1, b, "Mid"),
                (b + 1, n, "Late"),
            ]:
                sub = [
                    r
                    for r in train_log_rows
                    if start <= r["update"] <= end
                    and not math.isnan(r.get("mean_score", float("nan")))
                ]
                if not sub:
                    continue
                peak_sc = max(r["max_score"] for r in sub)
                mean_sc = float(np.mean([r["mean_score"] for r in sub]))
                mean_ln_vals = [
                    r["mean_len"]
                    for r in sub
                    if not math.isnan(r.get("mean_len", float("nan")))
                ]
                mean_ln = float(np.mean(mean_ln_vals)) if mean_ln_vals else float("nan")
                mean_sp = float(np.mean([r["sps"] for r in sub]))
                A(
                    _md_table_row(
                        [
                            phase,
                            f"{start}–{end}",
                            peak_sc,
                            f"{mean_sc:.1f}",
                            f"{mean_ln:.1f}",
                            f"{mean_sp:.1f}",
                        ]
                    )
                )
            A("")
            all_max = [r["max_score"] for r in train_log_rows]
            total_samp = sum(r.get("samples", 0) for r in train_log_rows)
            total_t = train_log_rows[-1].get("elapsed", 0)
            A("### 4.2 Overall Training Statistics")
            A("")
            A("| Metric | Value |")
            A("|:-------|------:|")
            A(f"| PPO updates logged | {len(train_log_rows)} |")
            A(f"| Peak score during training | {max(all_max)} |")
            A(f"| Total env samples | {total_samp:,} |")
            A(f"| Total wall-clock training time | {total_t/60:.1f} min |")
            mean_sp_all = float(np.mean([r.get("sps", 0) for r in train_log_rows]))
            A(f"| Mean samples per second | {mean_sp_all:.1f} |")
            A("")
    A("---")
    A("")
    A("## 5. Discussion")
    A("")
    cv_d = ss_d.get("cv", float("nan"))
    sr200 = det_agg.get("survival_rates", {}).get("sr_200", 0.0)
    sr300 = det_agg.get("survival_rates", {}).get("sr_300", 0.0)
    jmp_f = det_agg.get("action_distribution", {}).get("jump", {}).get("frac", 0.0)
    dck_f = det_agg.get("action_distribution", {}).get("duck", {}).get("frac", 0.0)
    A("### 5.1 Policy Competence")
    A("")
    A(
        f"The deterministic policy achieves a mean score of **{_fmt(mean_d)} ± {_fmt(std_d)}** "
        f"(median {_fmt(med_d, 1)}, max {_fmt(max_d, 0)}) over {n_det} episodes. "
        f"The coefficient of variation (CV = {_fmt(cv_d, 3)}) reflects environmental "
        f"stochasticity — obstacle types, gaps, and the speed ramp-up are non-deterministic "
        f"from the agent's perspective, so score variance is partly irreducible. "
        f"Survival rates of {sr200:.1f}% at score ≥200 and {sr300:.1f}% at score ≥300 "
        f"suggest the policy has internalised basic obstacle-avoidance behaviour and can "
        f"maintain game-play through the early speed-increase phases."
    )
    A("")
    A("### 5.2 Action Selection Behaviour")
    A("")
    A(
        f"Under deterministic inference the agent selects **jump** in {jmp_f*100:.1f}% "
        f"of steps and **duck** in {dck_f*100:.1f}% of steps. "
        "Jump is the primary survival action and its frequency reflects how densely "
        "obstacles appear relative to the nominal decision horizon. "
        "Low duck frequency is expected: pterodactyls only appear at higher speeds "
        "and ducking is rarely the correct action in the early-to-mid score range "
        "captured by most evaluation episodes. "
        f"The global action entropy of {_fmt(det_gent, 4)} nats "
        f"(vs. maximum {math.log(3):.4f} nats) indicates a concentrated, "
        "relatively deterministic behavioural profile."
    )
    A("")
    A("### 5.3 Critic Value Estimates")
    A("")
    if vs_d:
        A(
            f"The critic outputs values in the range [{_fmt(vs_d.get('min',0), 2)}, "
            f"{_fmt(vs_d.get('max',0), 2)}] with a mean of {_fmt(vs_d.get('mean',0), 3)}. "
            "The predominantly negative value range is consistent with the reward "
            "structure: death produces a −10 penalty that dominates shaped rewards, "
            "so the critic learns to predict a slightly negative long-run return at "
            "most states, scaling toward zero as the agent survives longer episodes."
        )
    else:
        A("Value statistics were not collected.")
    A("")
    A("### 5.4 Limitations and Future Work")
    A("")
    A(
        "- **Environment stochasticity.** Obstacle generation is non-deterministic; "
        "score variance is partly irreducible even for a perfect policy.\n"
        "- **Single seed / checkpoint.** A robust benchmark should average over "
        "multiple independently-trained seeds to disentangle policy quality from "
        "random luck.\n"
        "- **Selenium latency.** Browser IPC overhead (~15–30 ms per Selenium call) "
        "inflates wall-clock times and can introduce timing jitter; reported wall "
        "times should not be used to infer in-game timing precision.\n"
        "- **No speed-curriculum evaluation.** The agent is always evaluated from "
        "game start. A more thorough benchmark would measure survival at various "
        "speed injection points.\n"
        "- **No comparison baseline.** Adding a scripted rule-based agent "
        "and a random policy as baselines would contextualise these scores."
    )
    A("")
    A("---")
    A("")
    A(
        "*Report generated automatically by `benchmarking/benchmark.py` — "
        "Chrome Dino PPO project.*"
    )
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[benchmark] Report written -> {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Research-grade benchmarking for the PPO Dino agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best_duck.pt",
        help="Checkpoint path relative to dinoGame/ppo/.",
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Deterministic evaluation episodes."
    )
    parser.add_argument(
        "--stochastic-episodes",
        type=int,
        default=0,
        help="Stochastic-mode episodes (0 = skip).",
    )
    parser.add_argument(
        "--both-modes",
        action="store_true",
        help="Run --episodes episodes in BOTH det. and sto. modes.",
    )
    parser.add_argument(
        "--step-pause",
        type=float,
        default=0.0,
        help="Extra sleep per step in seconds. 0 = max speed.",
    )
    parser.add_argument("--chromedriver", type=str, default=None)
    parser.add_argument("--game-url", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run browser in headless mode."
    )
    parser.add_argument(
        "--browser-restart-every",
        type=int,
        default=5,
        help="Quit and relaunch Chrome every N episodes (0=never).",
    )
    parser.add_argument(
        "--adaptive-restart-streak",
        type=int,
        default=2,
        help="Restart after this many consecutive weak episodes "
        "(score<300 or steps<100); 0=disable.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="NumPy / PyTorch random seed."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: results/run_<TS>).",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        for alt in ("checkpoints/latest.pt", "checkpoints/bc_init.pt"):
            if os.path.exists(alt):
                print(f"[benchmark] {ckpt_path} not found — falling back to {alt}")
                ckpt_path = alt
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {ckpt_path}. "
                "Train first:  python train.py --n-envs 4 --rollout 256"
            )
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(os.path.dirname(__file__), "results", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[benchmark] Output -> {out_dir}")
    ppo = PPO(OBS_DIM, N_ACTIONS)
    ppo.load(ckpt_path, load_optim=False)
    ppo.net.eval()
    model_params = sum(p.numel() for p in ppo.net.parameters())
    print(
        f"[benchmark] Loaded {ckpt_path}  device={ppo.device}  params={model_params:,}"
    )
    n_det = args.episodes
    n_sto = args.stochastic_episodes
    if args.both_modes:
        n_sto = max(n_sto, n_det)

    def make_env():
        return DinoEnv(
            chromedriver_path=args.chromedriver,
            step_pause=args.step_pause,
            game_url=args.game_url,
            headless=args.headless,
        )

    env_holder = [make_env()]
    warmup_env(env_holder[0])
    restart_every = args.browser_restart_every
    adapt_streak = args.adaptive_restart_streak
    if restart_every > 0:
        print(f"[benchmark] Browser restart every {restart_every} episode(s)")
    if adapt_streak > 0:
        print(
            f"[benchmark] Adaptive restart after {adapt_streak} weak episode(s) in a row"
        )

    def relaunch_env():
        try:
            env_holder[0].close()
        except Exception:
            pass
        env_holder[0] = make_env()
        warmup_env(env_holder[0])
        return env_holder[0]

    records_det: list = []
    records_sto: list = []
    try:
        print(f"\n[benchmark] === Deterministic evaluation ({n_det} episodes) ===")
        records_det = run_episodes(
            env_holder[0],
            ppo,
            n_det,
            mode="deterministic",
            verbose=True,
            browser_restart_every=restart_every,
            make_env=relaunch_env,
            adaptive_restart_streak=adapt_streak,
        )
        if n_sto > 0:
            print(f"\n[benchmark] === Stochastic evaluation ({n_sto} episodes) ===")
            records_sto = run_episodes(
                env_holder[0],
                ppo,
                n_sto,
                mode="stochastic",
                verbose=True,
                browser_restart_every=restart_every,
                make_env=relaunch_env,
                adaptive_restart_streak=adapt_streak,
            )
    finally:
        try:
            env_holder[0].close()
        except Exception:
            pass
    if not records_det:
        print("[benchmark] No completed episodes to report — exiting.")
        return
    det_agg = aggregate(records_det)
    sto_agg = aggregate(records_sto) if records_sto else {}
    train_log_path = os.path.join(
        os.path.dirname(__file__), "checkpoints", "train_log.jsonl"
    )
    train_log_rows = _load_train_log(train_log_path)
    if train_log_rows:
        print(f"[benchmark] Loaded {len(train_log_rows)} training updates from log.")

    def _nan_safe(obj):
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    payload = {
        "run_timestamp": run_ts,
        "checkpoint": ckpt_path,
        "seed": args.seed,
        "model_params": model_params,
        "device": str(ppo.device),
        "deterministic": {
            "n_episodes": n_det,
            "aggregate": det_agg,
            "episodes": [r.to_dict() for r in records_det],
        },
    }
    if records_sto:
        payload["stochastic"] = {
            "n_episodes": n_sto,
            "aggregate": sto_agg,
            "episodes": [r.to_dict() for r in records_sto],
        }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_nan_safe)
    print(f"[benchmark] Metrics saved -> {metrics_path}")
    make_figures(
        records_det,
        records_sto,
        out_dir,
        train_log_path if train_log_rows else None,
    )
    write_report(
        out_dir,
        ckpt_path,
        n_det,
        n_sto,
        det_agg,
        sto_agg,
        records_det,
        records_sto,
        run_ts,
        train_log_rows=train_log_rows or None,
        model_params=model_params,
    )
    ss = det_agg.get("score_stats", {})
    sr = det_agg.get("survival_rates", {})
    ad = det_agg.get("action_distribution", {})
    print()
    print("=" * 58)
    print("  BENCHMARK SUMMARY")
    print("-" * 58)
    print(f"  Checkpoint        : {ckpt_path}")
    print(f"  Det. episodes     : {n_det}")
    print(f"  Score mean +/- std: {_fmt(ss.get('mean'))} +/- {_fmt(ss.get('std'))}")
    print(
        f"  Score 95% CI      : [{_fmt(ss.get('ci95_lo'))}, {_fmt(ss.get('ci95_hi'))}]"
    )
    print(f"  Score median      : {_fmt(ss.get('median'), 1)}")
    print(f"  Score max         : {_fmt(ss.get('max'), 0)}")
    print(f"  Survival >= 200  : {sr.get('sr_200', 0):.1f}%")
    print(f"  Survival >= 300  : {sr.get('sr_300', 0):.1f}%")
    print(f"  Jump rate         : {ad.get('jump',{}).get('frac',0)*100:.1f}%")
    print(f"  Duck rate         : {ad.get('duck',{}).get('frac',0)*100:.1f}%")
    print(f"  Pterodactyl deaths: {det_agg.get('pterodactyl_death_frac',0)*100:.1f}%")
    if records_sto:
        ss_s = sto_agg.get("score_stats", {})
        print(
            f"  Sto. mean +/- std : {_fmt(ss_s.get('mean'))} +/- {_fmt(ss_s.get('std'))}"
        )
    print("-" * 58)
    print(f"  Results in: {out_dir}")
    print("=" * 58)


if __name__ == "__main__":
    main()
