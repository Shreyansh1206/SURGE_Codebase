"""
finetune_duck.py — Teach the existing PPO Dino to DUCK without losing what it
already knows.

Why this script exists
----------------------
The shipped policy (`checkpoints/best.pt`) has a *dead* duck action: behaviour
cloning was done against a scripted teacher that only ever jumps or no-ops, so
the duck logit was driven to -inf and PPO (entropy_coef=0.01) can never sample
it back to life. On top of that the agent almost never reaches pterodactyl
territory (birds need game speed >= 8.5), so even a live duck head would get no
bird experience to learn from.

This script fixes both, surgically, on a COPY of the model — it never touches
`checkpoints/`:

  1. revive_duck(): re-open the duck action by copying the no-op row of the
     policy head into the duck row and offsetting the duck bias so that
     P(duck) ~ 5-10% again (a "stay grounded" prior). Saved as
     checkpoints_duck/duck_init.pt.
  2. PPO finetune from duck_init.pt with:
       - a speed CURRICULUM (DinoEnv curriculum_prob / start_speed_range) so a
         large fraction of rollout steps happen where birds actually spawn;
       - duck-targeted reward SHAPING (DinoEnv duck_shaping);
       - stronger exploration (higher entropy coef, lower lr).
  3. The "best" checkpoint is gated on a periodic *eval* (normal game start, no
     curriculum / no shaping) so we never promote a model that regressed on the
     real task.

Everything is written under `checkpoints_duck/`. The original model is the
permanent fallback.

Typical use (run from dinoGame/ppo/):
    # one-off: just build the revived init and inspect it
    python finetune_duck.py --revive-only

    # full finetune (revives automatically if duck_init.pt is missing)
    python finetune_duck.py --n-envs 4 --updates 120 --rollout 256

    # resume a finetune
    python finetune_duck.py --resume checkpoints_duck/latest.pt --updates 60
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ppo_agent import PPO, RolloutBuffer

# These mirror dino_env.OBS_DIM / N_ACTIONS. They are duplicated here so the
# `--revive-only` path works in a Python env without Selenium installed (the
# surgery needs only torch + the checkpoint). main() asserts they match the
# real env values once dino_env is imported.
OBS_DIM_DEFAULT    = 48
N_ACTIONS_DEFAULT  = 3
FEATURES_PER_FRAME = 12     # mirror dino_env.FEATURES_PER_FRAME (last-frame slice)
DUCK_IDX           = 2
NOOP_IDX           = 0

# A "duckable bird is near" when the nearest obstacle is a mid/high pterodactyl
# inside the danger zone. Read from the LAST frame of the stacked observation:
#   index 4 = o1_dx  (normalised: 0 touching dino .. 1.0 edge of decide-zone .. 1.5 far)
#   index 5 = o1_y   (yPos / 150: mid bird 0.50, high bird 0.33; cacti 0.60-0.70,
#                     low bird 0.667 — all of which we DON'T want to duck under)
BIRD_Y_NORM_MAX    = 80.0 / 150.0   # 0.533 — matches dino_env.BIRD_DUCK_YMAX
GUIDE_DX_MAX       = 0.90            # start assisting/collecting while the bird is still
                                    # approaching — ducking early is safe and roughly
                                    # doubles the usable duck-demonstration states


def duckable_bird_mask(obs_np: np.ndarray) -> np.ndarray:
    """Boolean (N,) mask: True where the latest frame shows a mid/high bird in
    range — i.e. a state where DUCK (not jump) is the right answer."""
    f = obs_np[:, -FEATURES_PER_FRAME:]
    o1_dx = f[:, 4]
    o1_y  = f[:, 5]
    return (o1_y > 1e-6) & (o1_y <= BIRD_Y_NORM_MAX) & (o1_dx < GUIDE_DX_MAX)


def jumpable_obstacle_mask(obs_np: np.ndarray) -> np.ndarray:
    """Boolean (N,) mask: True where the nearest in-range obstacle is LOW
    (a cactus or a low pterodactyl, o1_y above the duck threshold) — i.e. a
    state where JUMP, not duck, is the right answer. These are used as
    *contrast* anchors for the imitation loss: without them, pushing duck on
    bird-states bleeds onto visually-similar cactus-states and the agent learns
    to duck into cacti. Anchoring jump here teaches the net to discriminate."""
    f = obs_np[:, -FEATURES_PER_FRAME:]
    o1_dx = f[:, 4]
    o1_y  = f[:, 5]
    return (o1_y > BIRD_Y_NORM_MAX) & (o1_y <= 1.0) & (o1_dx < GUIDE_DX_MAX)


# ===========================================================================
# Step 1 — surgical duck-head revival
# ===========================================================================

def revive_duck(src_ckpt: str,
                dst_ckpt: str,
                margin: float = 2.5,
                obs_dim: int = OBS_DIM_DEFAULT,
                n_actions: int = N_ACTIONS_DEFAULT,
                device: str = "cpu") -> PPO:
    """Re-open the dead duck action on a copy of `src_ckpt`.

    The policy head is Linear(hidden -> 3). Action index 2 is duck. We set:
        weight[duck] = weight[noop].clone()
        bias[duck]   = bias[noop] - margin

    Because the duck row is now identical to the no-op row, for EVERY state the
    duck logit equals the no-op logit minus `margin`. The softmax then gives,
    exactly and state-independently:

        P(duck) = P(noop) * exp(-margin)

    so with margin=2.5 and a policy that picks no-op ~90% of the time, duck gets
    ~7% probability mass — enough for the sampler to start exploring it, while
    inheriting a sane "stay grounded" prior (the correct response to a high bird
    is to stay low, not jump). The value head and shared trunk are untouched.
    """
    if not os.path.exists(src_ckpt):
        raise FileNotFoundError(f"source checkpoint not found: {src_ckpt}")

    ppo = PPO(obs_dim, n_actions, device=device)
    # load_optim=False: we always start the finetune with a fresh optimiser.
    ppo.load(src_ckpt, load_optim=False)

    ph = ppo.net.policy_head
    with torch.no_grad():
        before_b = ph.bias.detach().cpu().numpy().copy()
        ph.weight[DUCK_IDX].copy_(ph.weight[NOOP_IDX])
        ph.bias[DUCK_IDX].copy_(ph.bias[NOOP_IDX] - margin)
        after_b = ph.bias.detach().cpu().numpy().copy()

    os.makedirs(os.path.dirname(dst_ckpt), exist_ok=True)
    ppo.save(dst_ckpt)

    # Report what changed, plus the implied P(duck) under a few plausible
    # no-op probabilities (the relation is state-independent by construction).
    print(f"[revive] loaded {src_ckpt}")
    print(f"[revive] policy-head bias before : noop={before_b[0]:+.3f} "
          f"jump={before_b[1]:+.3f} duck={before_b[2]:+.3f}")
    print(f"[revive] policy-head bias after  : noop={after_b[0]:+.3f} "
          f"jump={after_b[1]:+.3f} duck={after_b[2]:+.3f}  (margin={margin})")
    ratio = float(np.exp(-margin))
    print(f"[revive] P(duck) = P(noop) * exp(-{margin}) = P(noop) * {ratio:.3f}")
    for p_noop in (0.95, 0.90, 0.80):
        print(f"           if P(noop)~{p_noop:.2f}  ->  P(duck)~{p_noop*ratio:.3f}")
    print(f"[revive] saved revived init -> {dst_ckpt}")
    return ppo


# ===========================================================================
# Rollout collection (vectorised) — mirrors train.collect_rollout
# ===========================================================================

@torch.no_grad()
def act_with_guidance(net, obs_np, device, guide_prob: float):
    """Sample actions from the policy, but when a duckable bird is near, force a
    DUCK with probability `guide_prob` (assisted exploration). Because the mask
    stays True across the several steps a bird is approaching, the forced duck is
    naturally *sustained* — which is the whole point: a one-frame duck never
    clears a bird, so unassisted exploration can never discover that ducking
    works. We store the policy's OWN log-prob of the (possibly forced) action so
    PPO's importance ratio stays well-defined; `guide_prob` is annealed to 0 so
    the final policy is fully self-reliant.

    Returns (actions, logps, values, bird_mask).
    """
    obs = torch.from_numpy(obs_np).float().to(device)
    logits, value = net(obs)
    dist = Categorical(logits=logits)
    actions = dist.sample()

    bird_mask = duckable_bird_mask(obs_np)
    if guide_prob > 0.0 and bird_mask.any():
        roll = np.random.random(len(bird_mask)) < guide_prob
        force = bird_mask & roll
        if force.any():
            a_np = actions.cpu().numpy()
            a_np[force] = DUCK_IDX
            actions = torch.from_numpy(a_np).to(device)

    logps = dist.log_prob(actions)
    return (actions.cpu().numpy().astype(np.int64),
            logps.cpu().numpy().astype(np.float32),
            value.cpu().numpy().astype(np.float32),
            bird_mask)


def collect_rollout(vec_env, ppo, buf, n_steps, device, last_obs,
                    guide_prob: float = 0.0, collect_bird_states: bool = False):
    N = vec_env.n_envs
    obs = last_obs
    ep_returns, ep_lens, ep_scores = [], [], []
    cur_return = np.zeros(N, dtype=np.float32)
    cur_len    = np.zeros(N, dtype=np.int64)
    bird_states = []   # observations where DUCK is the correct action
    jump_states = []   # observations where JUMP is the correct action (contrast)

    for _ in range(n_steps):
        actions, logps, values, bird_mask = act_with_guidance(
            ppo.net, obs, device, guide_prob)
        if collect_bird_states:
            if bird_mask.any():
                bird_states.append(obs[bird_mask].copy())
            jmask = jumpable_obstacle_mask(obs)
            if jmask.any():
                jump_states.append(obs[jmask].copy())

        next_obs, rewards, dones, infos = vec_env.step(actions)

        buf.add(obs, actions, logps, rewards, values, dones)

        cur_return += rewards
        cur_len    += 1
        for i in range(N):
            if dones[i]:
                ep_returns.append(float(cur_return[i]))
                ep_lens.append(int(cur_len[i]))
                ep_scores.append(int(infos[i].get("score", 0)))
                cur_return[i] = 0.0
                cur_len[i]    = 0
        obs = next_obs

    with torch.no_grad():
        obs_t = torch.from_numpy(obs).float().to(device)
        _, last_v = ppo.net(obs_t)
        last_values = last_v.cpu().numpy().astype(np.float32)

    bird_obs = (np.concatenate(bird_states, axis=0)
                if bird_states else np.zeros((0, last_obs.shape[1]), dtype=np.float32))
    jump_obs = (np.concatenate(jump_states, axis=0)
                if jump_states else np.zeros((0, last_obs.shape[1]), dtype=np.float32))
    return obs, last_values, ep_returns, ep_lens, ep_scores, bird_obs, jump_obs


JUMP_IDX = 1


def duck_imitation_update(ppo, optim, bird_obs, jump_obs, coef: float,
                          target_duck: float = 0.6, duck_cap: float = 0.05,
                          epochs: int = 4, batch: int = 256,
                          max_grad_norm: float = 0.5, head_only: bool = False):
    """Auxiliary supervised step that controls ONLY the duck logit:
      * on bird-states  -> RAISE  log P(duck) up to `target_duck`;
      * on cactus-states-> LOWER  log P(duck) down to `duck_cap`.

    Both are one-sided hinges that touch *only the duck action's log-prob* and
    do nothing once satisfied. This is deliberately minimal:

      - We never specify noop/jump targets, so PPO keeps its sharp, well-timed
        cactus/early-game behaviour. (An earlier version anchored P(jump) UP on
        cactus-states; because the cactus mask includes obstacles still far away
        (dx<0.9) that pushed the agent to jump too EARLY, landing before the
        cactus and dying — base score collapsed even with duck at 0%.)
      - The cactus-side hinge is a pure anti-bleed guard: it keeps duck small on
        low obstacles so the duck signal learned on birds can't leak into
        'duck into a cactus'.

    Design history of failure modes we ruled out: hard one-hot duck (duck->1.0,
    bleeds), too-weak push (duck stays dead), soft full-distribution target
    (injects entropy, base collapses), jump-up anchor (premature jumps).
    Returns (mean_loss, n_bird)."""
    if coef <= 0.0:
        return 0.0, 0
    device = ppo.device
    log_duck_tgt = float(np.log(target_duck))
    log_duck_cap = float(np.log(duck_cap))
    # (obs, sign): sign=+1 -> raise duck to target; sign=-1 -> push duck below cap
    jobs = [(bird_obs, +1, log_duck_tgt), (jump_obs, -1, log_duck_cap)]

    tot, steps = 0.0, 0
    for _ in range(epochs):
        for obs_arr, sign, log_lim in jobs:
            n = len(obs_arr)
            if n == 0:
                continue
            obs_t = torch.from_numpy(obs_arr).float().to(device)
            idx = np.arange(n)
            np.random.shuffle(idx)
            for start in range(0, n, batch):
                b = torch.as_tensor(idx[start:start + batch],
                                    dtype=torch.long, device=device)
                logits, _ = ppo.net(obs_t[b])
                logp_duck = F.log_softmax(logits, dim=-1)[:, DUCK_IDX]
                # sign=+1: penalise (log_lim - logp) when duck below target
                # sign=-1: penalise (logp - log_lim) when duck above cap
                deficit = torch.clamp(sign * (log_lim - logp_duck), min=0.0)
                active = deficit > 0
                if active.sum() == 0:
                    continue
                loss = coef * deficit[active].mean()
                optim.zero_grad()
                loss.backward()
                if head_only:
                    # Update ONLY the duck row of the policy head. The no-op and
                    # jump logits then stay byte-for-byte identical to the base
                    # policy on EVERY state, so the base policy's jump timing on
                    # cacti / early game is preserved exactly. (Training through
                    # the shared trunk instead silently scrambled that timing and
                    # the agent died at the first cactus, even with duck still 0%
                    # on non-bird states.) The only possible behaviour change is
                    # duck becoming the argmax where we raise it — i.e. on birds.
                    ph = ppo.net.policy_head
                    if ph.weight.grad is not None:
                        ph.weight.grad[NOOP_IDX].zero_()
                        ph.weight.grad[JUMP_IDX].zero_()
                    if ph.bias.grad is not None:
                        ph.bias.grad[NOOP_IDX].zero_()
                        ph.bias.grad[JUMP_IDX].zero_()
                    nn.utils.clip_grad_norm_([ph.weight, ph.bias], max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(ppo.net.parameters(), max_grad_norm)
                optim.step()
                tot += float(loss.item())
                steps += 1
    return tot / max(1, steps), len(bird_obs)


@torch.no_grad()
def collect_states(vec_env, ppo, device, n_steps, guide_prob=0.9):
    """Roll the env to gather observations for SUPERVISED imitation, split into
    'duckable bird in range' (label: duck) vs everything else (label: not-duck).

    The guide forces ducks on bird-states so the agent SURVIVES bird territory
    and we collect many, diverse bird frames across speeds (without the guide the
    agent dies at the first bird and we get almost no bird data). These states
    are used purely as a labelled dataset — they never touch PPO — so the forced
    ducks cannot bias the policy; only the imitation loss shapes the duck logit.
    Returns (bird_obs, other_obs)."""
    obs = vec_env.reset()
    bird, other = [], []
    for _ in range(n_steps):
        a, _, _, bmask = act_with_guidance(ppo.net, obs, device, guide_prob)
        if bmask.any():
            bird.append(obs[bmask].copy())
        if (~bmask).any():
            other.append(obs[~bmask].copy())
        obs, _, _, _ = vec_env.step(a)
    D = obs.shape[1]
    bird_obs  = np.concatenate(bird, axis=0)  if bird  else np.zeros((0, D), np.float32)
    other_obs = np.concatenate(other, axis=0) if other else np.zeros((0, D), np.float32)
    return bird_obs, other_obs


# ===========================================================================
# Step 3b — eval-gated checkpointing (normal start, no curriculum/shaping)
# ===========================================================================

def run_eval(vec_env, ppo, device, n_episodes: int, max_steps: int = 4000,
             curriculum_prob: float = 0.0):
    """Deterministic (argmax) evaluation, always with shaping OFF so the score
    is the true task score.

    `curriculum_prob` controls WHERE episodes start:
      * 0.0 (default) -> NORMAL game start: measures base-task competence /
        non-regression on the early game the agent must not forget.
      * 1.0           -> always jump-start into pterodactyl territory (uses the
        env's configured start_speed_range / start_score): this is the only way
        to actually measure duck-on-bird survival, since a normal start usually
        dies before birds even spawn.
    Restores the env flags before returning.
    """
    saved = []
    for e in vec_env.envs:
        saved.append((getattr(e, "curriculum_prob", 0.0),
                      getattr(e, "duck_shaping", False)))
        e.curriculum_prob = curriculum_prob
        e.duck_shaping     = False

    N = vec_env.n_envs
    scores = []
    action_counts = np.zeros(N_ACTIONS_DEFAULT, dtype=np.int64)
    try:
        obs = vec_env.reset()
        steps = 0
        while len(scores) < n_episodes and steps < max_steps:
            obs_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                logits, _ = ppo.net(obs_t)
                actions = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64)
            for a in actions:
                action_counts[int(a)] += 1
            obs, _, dones, infos = vec_env.step(actions)
            for i in range(N):
                if dones[i]:
                    scores.append(int(infos[i].get("score", 0)))
            steps += 1
    finally:
        for e, (cp, ds) in zip(vec_env.envs, saved):
            e.curriculum_prob = cp
            e.duck_shaping     = ds

    total = max(1, int(action_counts.sum()))
    return {
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
        "max_score":  int(np.max(scores)) if scores else 0,
        "n":          len(scores),
        "scores":     scores,
        "duck_frac":  float(action_counts[DUCK_IDX] / total),
        "jump_frac":  float(action_counts[1] / total),
        "noop_frac":  float(action_counts[0] / total),
    }


# ===========================================================================
# Imitation-only training (no PPO) — the robust, base-preserving path
# ===========================================================================

def run_imitation_only(args, vec_env, ppo):
    """Install duck purely by supervised imitation, no PPO.

    Loop: collect a labelled dataset (bird-states vs everything-else) with the
    guide on so we actually survive birds and gather data, then push the DUCK
    logit up on bird-states and down everywhere else. Because we only touch the
    duck logit, the base policy's jump/no-op behaviour (cacti, early game) is
    preserved by construction. We eval on both a normal start (regression guard)
    and bird territory (the goal) and keep the best ducking model."""
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "imitation_log.jsonl")
    bc_optim = torch.optim.Adam(ppo.net.parameters(), lr=args.duck_bc_lr)
    bird_buf = deque(maxlen=args.duck_bc_bufsize)
    other_buf = deque(maxlen=args.duck_bc_bufsize * 3)
    best_duck = -1.0
    t0 = time.time()

    print(f"[imitation] collect_steps={args.collect_steps} iters={args.imitation_iters} "
          f"recollect_every={args.recollect_every} | duck_target={args.duck_target} "
          f"duck_cap={args.duck_cap} bc_lr={args.duck_bc_lr}")

    for it in range(1, args.imitation_iters + 1):
        if it == 1 or (args.recollect_every and (it - 1) % args.recollect_every == 0):
            bobs, oobs = collect_states(vec_env, ppo, ppo.device,
                                        args.collect_steps, guide_prob=0.9)
            bird_buf.extend(bobs)
            other_buf.extend(oobs)
            print(f"[imitation] it {it:3d} collected bird={len(bobs)} "
                  f"other={len(oobs)} | buffers bird={len(bird_buf)} other={len(other_buf)}")

        def _samp(dq, k):
            pool = np.stack(dq, axis=0)
            if len(pool) > k:
                pool = pool[np.random.choice(len(pool), k, replace=False)]
            return pool

        bird_pool = _samp(bird_buf, args.duck_bc_sample)
        # BALANCE the suppression set to the bird set. A large imbalance (e.g. 5x
        # more 'other' than 'bird') makes the duck-suppression generalisation
        # swamp the duck-raise and pins duck near 0 on birds too — that was the
        # failure mode of the first imitation run.
        n_other = min(len(other_buf), max(1, int(len(bird_pool) * args.other_mult)))
        other_pool = _samp(other_buf, n_other)

        loss, _ = duck_imitation_update(
            ppo, bc_optim, bird_pool, other_pool,
            coef=args.duck_bc_coef,
            target_duck=args.duck_target, duck_cap=args.duck_cap,
            epochs=args.duck_bc_epochs)

        if it % args.eval_every == 0 or it == args.imitation_iters:
            ev = run_eval(vec_env, ppo, ppo.device, args.eval_episodes, curriculum_prob=0.0)
            bv = run_eval(vec_env, ppo, ppo.device, args.eval_episodes, curriculum_prob=1.0)
            print(f"  it {it:3d} | loss {loss:.3f} | "
                  f"BASE {ev['mean_score']:6.1f} duck {ev['duck_frac']*100:4.1f}% | "
                  f"BIRDS {bv['mean_score']:6.1f} duck {bv['duck_frac']*100:4.1f}% "
                  f"jump {bv['jump_frac']*100:4.1f}% | {time.time()-t0:5.0f}s")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "iter": it, "loss": loss,
                    "base_score": ev["mean_score"], "base_duck": ev["duck_frac"],
                    "bird_score": bv["mean_score"], "bird_duck": bv["duck_frac"],
                    "bird_jump": bv["jump_frac"], "elapsed": time.time() - t0,
                }) + "\n")
            base_ok = ev["mean_score"] == ev["mean_score"]
            bird_ok = bv["mean_score"] == bv["mean_score"]
            ppo.save(os.path.join(args.save_dir, "latest.pt"))
            if (base_ok and bird_ok and ev["mean_score"] >= args.base_floor
                    and bv["duck_frac"] >= 0.03 and bv["mean_score"] > best_duck):
                best_duck = bv["mean_score"]
                ppo.save(os.path.join(args.save_dir, "best_duck.pt"))
                print(f"        -> new best DUCKING model (bird {best_duck:.1f}, "
                      f"bird duck {bv['duck_frac']*100:.1f}%, base {ev['mean_score']:.1f}) "
                      f"saved {args.save_dir}/best_duck.pt")
            obs = vec_env.reset()

    vec_env.close()
    print(f"[imitation] done. best DUCKING bird-score = {best_duck:.1f} | "
          f"checkpoints in {args.save_dir}/ | original model untouched.")


# ===========================================================================
# Main finetune loop
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Surgically revive the duck action and PPO-finetune the Dino.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- revival ---
    p.add_argument("--base-ckpt", type=str, default="checkpoints/best.pt",
                   help="Original (frozen) model to revive duck from. Never modified.")
    p.add_argument("--duck-init", type=str, default="checkpoints_duck/duck_init.pt",
                   help="Where the revived init is written / read from.")
    p.add_argument("--margin", type=float, default=2.5,
                   help="Duck-bias offset below no-op. Larger -> smaller initial P(duck).")
    p.add_argument("--revive-only", action="store_true",
                   help="Only perform the surgical revival, then exit (no browser needed).")
    p.add_argument("--imitation-only", action="store_true",
                   help="Skip PPO entirely. Collect a labelled dataset of bird- vs "
                        "non-bird states and install duck purely by SUPERVISED imitation "
                        "(duck up on birds, suppressed elsewhere). This only nudges the "
                        "duck logit, so the base policy's cactus/early-game play is "
                        "preserved by construction — the robust path when PPO finetuning "
                        "keeps degrading the base.")
    p.add_argument("--collect-steps", type=int, default=1500,
                   help="Env steps per data-collection pass for --imitation-only.")
    p.add_argument("--imitation-iters", type=int, default=60,
                   help="Imitation optimiser iterations for --imitation-only.")
    p.add_argument("--recollect-every", type=int, default=15,
                   help="Re-collect states every N imitation iters (DAgger-style) so the "
                        "dataset tracks the improving policy. 0 = collect once.")
    p.add_argument("--other-mult", type=float, default=1.0,
                   help="Suppression-set size as a multiple of the bird-set size in "
                        "--imitation-only. Keep ~1.0: a large imbalance lets duck-"
                        "suppression generalise onto birds and pin duck near 0.")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume finetuning from this checkpoint instead of duck_init.")

    # --- PPO / training ---
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--updates", type=int, default=120)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Lower than scratch-PPO (3e-4) to limit forgetting.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--entropy", type=float, default=0.01,
                   help="Match scratch-PPO (0.01) so the policy can stay SHARP. Heavy "
                        "entropy just adds noise that degrades the base policy; duck "
                        "handled by assisted exploration, not entropy.")
    p.add_argument("--kl", type=float, default=0.02,
                   help="PPO KL early-stop threshold. Kept close to the scratch 0.015 so "
                        "PPO takes SMALL steps and does not wander off the good pretrained "
                        "policy (a loose 0.05 let it degrade the early game). Duck is "
                        "installed by the imitation loss, not by large PPO moves.")

    # --- assisted duck exploration (the actual fix for the held-action problem) ---
    p.add_argument("--guide-prob", type=float, default=0.85,
                   help="Prob. of forcing a (sustained) duck when a duckable bird is "
                        "near, so the agent experiences successful duck-clears. "
                        "Annealed to 0 over --guide-anneal updates.")
    p.add_argument("--guide-anneal", type=int, default=None,
                   help="Updates over which guide-prob and duck-bc-coef decay toward "
                        "their floors. Default = 70%% of --updates.")
    p.add_argument("--duck-bc-coef", type=float, default=2.0,
                   help="Weight of the auxiliary duck-imitation loss on bird-states. "
                        "Annealed toward its floor alongside guidance. 0 disables it.")
    p.add_argument("--duck-bc-epochs", type=int, default=4,
                   help="SGD passes of the duck-imitation loss per update over a sample "
                        "of the bird-state replay buffer.")
    p.add_argument("--duck-bc-lr", type=float, default=5e-4,
                   help="Dedicated LR for the duck-imitation optimiser so imitation can "
                        "lift duck on bird-states without destabilising PPO.")
    p.add_argument("--duck-target", type=float, default=0.6,
                   help="Hinge target for P(duck) on bird-states. Imitation lifts duck "
                        "up to this and then stops, so duck becomes the argmax on birds "
                        "without pinning to 1.0.")
    p.add_argument("--duck-cap", type=float, default=0.05,
                   help="Anti-bleed cap: on cactus/low-obstacle states the imitation "
                        "pushes P(duck) DOWN to at most this. It only touches the duck "
                        "logit (never jump), so the baseline's well-timed jumps are "
                        "preserved while duck is prevented from leaking onto cacti.")
    p.add_argument("--duck-bc-bufsize", type=int, default=6000,
                   help="Size of the rolling bird-state replay buffer that the imitation "
                        "loss trains on (persists across updates).")
    p.add_argument("--duck-bc-sample", type=int, default=2048,
                   help="Bird-states sampled from the buffer for each imitation update.")
    p.add_argument("--guide-floor", type=float, default=0.15,
                   help="Fraction of guide-prob retained after annealing, so a few "
                        "forced ducks keep flowing and duck doesn't regress. Eval is "
                        "always unguided, so it still measures the true policy.")
    p.add_argument("--bc-floor", type=float, default=0.25,
                   help="Fraction of duck-bc-coef retained after annealing — a small "
                        "permanent anchor keeping duck alive on bird-states.")

    # --- curriculum / shaping (opt-in DinoEnv features) ---
    p.add_argument("--curriculum-prob", type=float, default=0.3,
                   help="Fraction of episodes that start in the bird zone. Kept LOW: "
                        "most episodes must start normally (speed 6) so the agent keeps "
                        "practising — and does not FORGET — the early game. The base "
                        "policy already reaches birds (~score 450+) on natural runs, so "
                        "0.3 still yields plenty of bird exposure.")
    p.add_argument("--start-speed-min", type=float, default=8.5)
    p.add_argument("--start-speed-max", type=float, default=11.0)
    p.add_argument("--start-score", type=float, default=450.0,
                   help="In-game score to seed on curriculum starts (sets distanceRan "
                        "so the score readout matches the elevated speed). 0 = leave at 0.")
    p.add_argument("--no-duck-shaping", action="store_true",
                   help="Disable duck-targeted reward shaping during finetune.")

    # --- eval / checkpointing ---
    p.add_argument("--eval-every", type=int, default=5,
                   help="Run an eval (normal start) every N updates to gate best.pt.")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--base-floor", type=float, default=400.0,
                   help="A duck model is only promoted to best_duck.pt if its NORMAL-start "
                        "eval score stays at/above this floor — i.e. it learned to duck "
                        "WITHOUT regressing the early game. (The vec-env baseline is ~450-500 "
                        "with high variance, so 400 leaves headroom for eval noise.)")
    p.add_argument("--save-dir", type=str, default="checkpoints_duck")
    p.add_argument("--save-every", type=int, default=10)

    # --- env plumbing ---
    p.add_argument("--step-pause", type=float, default=0.0)
    p.add_argument("--window-w", type=int, default=700)
    p.add_argument("--window-h", type=int, default=320)
    p.add_argument("--grid-cols", type=int, default=2)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- Step 1: revive (always safe; only reads base-ckpt, writes duck_init) ---
    if args.resume is None:
        if args.revive_only or not os.path.exists(args.duck_init):
            print("[finetune] performing surgical duck revival ...")
            revive_duck(args.base_ckpt, args.duck_init, margin=args.margin)
        else:
            print(f"[finetune] reusing existing revived init: {args.duck_init} "
                  f"(delete it or pass --margin with --revive-only to rebuild)")
    if args.revive_only:
        print("[finetune] --revive-only set; done.")
        return

    # Import the env lazily so --revive-only works without Selenium installed.
    from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
    assert OBS_DIM == OBS_DIM_DEFAULT and N_ACTIONS == N_ACTIONS_DEFAULT, (
        f"env dims ({OBS_DIM},{N_ACTIONS}) differ from script defaults "
        f"({OBS_DIM_DEFAULT},{N_ACTIONS_DEFAULT}); update the constants.")

    start_speed_range = (args.start_speed_min, args.start_speed_max)
    duck_shaping = not args.no_duck_shaping
    guide_anneal = args.guide_anneal or max(1, int(0.7 * args.updates))

    print(f"[finetune] launching {args.n_envs} envs | curriculum_prob="
          f"{args.curriculum_prob} speed={start_speed_range} "
          f"duck_shaping={duck_shaping} entropy={args.entropy} lr={args.lr}")
    print(f"[finetune] assisted exploration: guide_prob={args.guide_prob} "
          f"duck_bc_coef={args.duck_bc_coef} anneal over {guide_anneal} updates | "
          f"target_kl={args.kl}")
    vec_env = VecDinoEnv(
        n_envs            = args.n_envs,
        window_size       = (args.window_w, args.window_h),
        grid_cols         = args.grid_cols,
        chromedriver_path = args.chromedriver,
        step_pause        = args.step_pause,
        game_url          = args.game_url,
        headless          = args.headless,
        curriculum_prob   = args.curriculum_prob,
        start_speed_range = start_speed_range,
        start_score       = args.start_score,
        duck_shaping      = duck_shaping,
    )

    ppo = PPO(OBS_DIM, N_ACTIONS,
              lr=args.lr, clip_eps=args.clip, epochs=args.epochs,
              batch_size=args.batch_size, entropy_coef=args.entropy)
    init_ckpt = args.resume or args.duck_init
    print(f"[finetune] loading policy init: {init_ckpt} (fresh optimiser)")
    ppo.load(init_ckpt, load_optim=False)

    if args.imitation_only:
        run_imitation_only(args, vec_env, ppo)
        return

    buf = RolloutBuffer(args.n_envs)
    bird_buffer = deque(maxlen=args.duck_bc_bufsize)   # rolling duck-demonstration states
    jump_buffer = deque(maxlen=args.duck_bc_bufsize)   # contrast: cactus/low-obstacle states
    # Dedicated optimiser for the imitation loss (higher LR than PPO so it can
    # lift duck on bird-states without destabilising PPO).
    bc_optim = torch.optim.Adam(ppo.net.parameters(), lr=args.duck_bc_lr)
    log_path = os.path.join(args.save_dir, "finetune_log.jsonl")
    best_eval = -1.0     # best eval score overall (non-regression safety net)
    best_duck = -1.0     # best eval score among policies that actually duck
    t0 = time.time()

    try:
        obs = vec_env.reset()
        for update in range(1, args.updates + 1):
            # Anneal assisted exploration from full strength down to a small floor,
            # so the policy increasingly handles birds itself (measured by the
            # unguided eval) while a little guidance/imitation keeps duck from
            # regressing late in training.
            anneal = max(0.0, 1.0 - (update - 1) / max(1, guide_anneal))
            guide_now = args.guide_prob   * (args.guide_floor + (1 - args.guide_floor) * anneal)
            bc_now    = args.duck_bc_coef * (args.bc_floor    + (1 - args.bc_floor)    * anneal)

            buf.clear()
            t_roll = time.time()
            obs, last_v, ep_ret, ep_len, ep_sc, bird_obs, jump_obs = collect_rollout(
                vec_env, ppo, buf, args.rollout, ppo.device, obs,
                guide_prob=guide_now, collect_bird_states=(bc_now > 0.0))
            roll_time = time.time() - t_roll

            stats = ppo.update(buf, last_v, gamma=args.gamma, lam=args.lam,
                               target_kl=args.kl)

            # Accumulate this rollout's duck-demonstration states, then run a
            # STRONG imitation pass over a sample of the whole rolling buffer.
            # Training on thousands of accumulated bird-states (rather than the
            # ~50 from one rollout) lets the duck signal dominate PPO's tendency
            # to suppress ducks that happen to sit inside a losing trajectory.
            if len(bird_obs):
                bird_buffer.extend(bird_obs)
            if len(jump_obs):
                jump_buffer.extend(jump_obs)

            def _sample(dq):
                if len(dq) == 0:
                    return np.zeros((0, obs.shape[1]), dtype=np.float32)
                pool = np.stack(dq, axis=0)
                if len(pool) > args.duck_bc_sample:
                    sel = np.random.choice(len(pool), args.duck_bc_sample, replace=False)
                    pool = pool[sel]
                return pool

            if bc_now > 0.0 and len(bird_buffer) > 0:
                aux_loss, _ = duck_imitation_update(
                    ppo, bc_optim, _sample(bird_buffer), _sample(jump_buffer), bc_now,
                    target_duck=args.duck_target, duck_cap=args.duck_cap,
                    epochs=args.duck_bc_epochs)
            else:
                aux_loss = 0.0
            n_bird = len(bird_buffer)

            mean_ret = float(np.mean(ep_ret)) if ep_ret else float("nan")
            mean_len = float(np.mean(ep_len)) if ep_len else float("nan")
            mean_sc  = float(np.mean(ep_sc))  if ep_sc  else float("nan")
            max_sc   = int(np.max(ep_sc))     if ep_sc  else 0
            elapsed  = time.time() - t0
            sps      = (args.rollout * args.n_envs) / max(roll_time, 1e-6)

            print(f"upd {update:4d} | eps {len(ep_ret):3d} | "
                  f"ret {mean_ret:7.2f} | len {mean_len:6.1f} | "
                  f"score(curr) avg {mean_sc:6.1f} max {max_sc:5d} | "
                  f"H {stats['entropy']:.3f} v {stats['v_loss']:.3f} | "
                  f"guide {guide_now:.2f} bc {bc_now:.2f} birdbuf {n_bird:5d} "
                  f"aux {aux_loss:.3f} | {sps:5.1f} sps | {elapsed:6.0f}s")

            log_row = {
                "update": update, "episodes": len(ep_ret),
                "mean_return": mean_ret, "mean_len": mean_len,
                "mean_score_curriculum": mean_sc, "max_score_curriculum": max_sc,
                **stats, "sps": sps, "elapsed": elapsed,
                "guide_prob": guide_now, "duck_bc_coef": bc_now,
                "bird_buffer_size": int(n_bird), "duck_bc_loss": aux_loss,
            }

            # --- periodic eval: TWO complementary measurements ---
            #   (1) normal start  -> base-task competence (must not regress)
            #   (2) bird-territory -> duck-on-bird survival (the actual goal,
            #       unmeasurable from a normal start that dies before birds)
            if update % args.eval_every == 0 or update == args.updates:
                ev = run_eval(vec_env, ppo, ppo.device, args.eval_episodes,
                              curriculum_prob=0.0)
                bv = run_eval(vec_env, ppo, ppo.device, args.eval_episodes,
                              curriculum_prob=1.0)
                print(f"        [eval] base   score {ev['mean_score']:6.1f} "
                      f"(max {ev['max_score']}) duck {ev['duck_frac']*100:4.1f}% "
                      f"jump {ev['jump_frac']*100:4.1f}%")
                print(f"        [eval] birds  score {bv['mean_score']:6.1f} "
                      f"(max {bv['max_score']}) duck {bv['duck_frac']*100:4.1f}% "
                      f"jump {bv['jump_frac']*100:4.1f}%")
                log_row["eval_mean_score"] = ev["mean_score"]
                log_row["eval_max_score"]  = ev["max_score"]
                log_row["eval_duck_frac"]  = ev["duck_frac"]
                log_row["eval_jump_frac"]  = ev["jump_frac"]
                log_row["bird_mean_score"] = bv["mean_score"]
                log_row["bird_duck_frac"]  = bv["duck_frac"]
                log_row["bird_jump_frac"]  = bv["jump_frac"]

                base_ok = ev["mean_score"] == ev["mean_score"]   # not NaN
                if base_ok and ev["mean_score"] > best_eval:
                    best_eval = ev["mean_score"]
                    ppo.save(os.path.join(args.save_dir, "best.pt"))
                    print(f"        [eval] new best BASE score {best_eval:.1f} "
                          f"-> saved {args.save_dir}/best.pt")
                # The deliverable: ducks on birds AND hasn't wrecked the early
                # game. Gate on bird-eval score, but only among models whose base
                # score stays above the regression floor and that actually duck.
                bird_ok = bv["mean_score"] == bv["mean_score"]
                if (base_ok and bird_ok
                        and ev["mean_score"] >= args.base_floor
                        and bv["duck_frac"] >= 0.03
                        and bv["mean_score"] > best_duck):
                    best_duck = bv["mean_score"]
                    ppo.save(os.path.join(args.save_dir, "best_duck.pt"))
                    print(f"        [eval] new best DUCKING policy "
                          f"(bird score {best_duck:.1f}, bird duck "
                          f"{bv['duck_frac']*100:.1f}%, base {ev['mean_score']:.1f}) "
                          f"-> saved {args.save_dir}/best_duck.pt")
                # Continue training with a clean obs (eval re-reset the envs).
                obs = vec_env.reset()

            with open(log_path, "a") as f:
                f.write(json.dumps(log_row) + "\n")

            if update % args.save_every == 0:
                ppo.save(os.path.join(args.save_dir, f"duck_upd{update}.pt"))
                ppo.save(os.path.join(args.save_dir, "latest.pt"))

    except KeyboardInterrupt:
        print("\n[interrupt] saving and exiting")
    finally:
        ppo.save(os.path.join(args.save_dir, "latest.pt"))
        vec_env.close()
        print(f"[finetune] done. best eval score = {best_eval:.1f} | "
              f"best DUCKING eval score = {best_duck:.1f} | "
              f"checkpoints in {args.save_dir}/ | original model untouched.")
        print("[finetune] benchmark the ducking model with: "
              "python benchmarking/benchmark.py --ckpt "
              f"{args.save_dir}/best_duck.pt --episodes 50")


if __name__ == "__main__":
    main()
