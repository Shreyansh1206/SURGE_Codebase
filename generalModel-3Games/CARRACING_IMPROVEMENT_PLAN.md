# CarRacing Improvement Plan (handoff to implementer)

## Goal
Two targets must BOTH hold for CarRacing, while not regressing the other two games:
1. **Raw score 700+** (consistent, not just occasional peaks). NOTE: target raised from 600
   to 700 deliberately — at a 600 floor the car still spins out on the hardest corner of
   some tracks. A 700 floor effectively forces clean driving through EVERY turn, which is
   the real objective.
2. **Clean cornering** — the car must NOT fly off / spin out on steep/hairpin turns (the original complaint)

Non-regression guardrails:
- MiniGrid DoorKey-16x16: solve EMA >= 0.85
- Chrome Dino: mean score >= 200

Architecture constraint (do NOT change): single shared 128-dim MLP backbone with
per-task encoders + actor/critic heads. All fixes below are env/feature/reward/training
changes only — no new policy distribution, no backbone redesign.

---

## What has already been implemented (code is in place)
These edits are DONE and committed in the working tree. Verify they're present before running.

1. **9 discrete CarRacing actions** (`envs/carracing_env.py` `CARRACING_DISCRETE_ACTIONS`)
   - Added `brake+left` (7) and `brake+right` (8) so the car can slow down WHILE turning.
   - `CARRACING_N_ACTIONS = 9`; `multi_task_ppo.py` default also 9.

2. **Speed/physics aux features** (`envs/carracing_env.py` + `multi_task_ppo.py`)
   - `_get_car_physics()` extracts `[speed, lateral_slip, angular_velocity]` from the Box2D hull.
   - `CarRacingControlWrapper.last_aux` holds the normalized 3-vector; exposed via `info["aux"]`.
   - `CARRACING_AUX_DIM = 3`. `CarRacingCNNEncoder` concatenates aux to the flattened conv
     features before the final FC layer (CNN weights stay checkpoint-compatible; only
     `carracing_encoder.fc.0.weight` resizes, handled by tolerant loading).
   - `train.py` `_attach_aux()` appends aux to the obs vector during rollout + after reset.
   - `infer.py` `_obs_with_aux()` does the same at eval time.

3. **frame_skip 4 -> 2** (`DEFAULT_FRAME_SKIP = 2`) — agent reacts twice as often, key for
   sharp turns. Rollout doubled to 1024 to keep the same game-time per update.

4. **OOM fix** (`DEFAULT_ZOOM_SKIP = 40 -> 20`) — 4 concurrent CarRacing envs at zoom_skip=40
   exhausted pygame memory on reset. 20 is enough to skip the intro zoom and runs stable.

5. **Slip penalty REMOVED** (`DEFAULT_SLIP_PENALTY = 0.0`)
   - It was tried at 0.5 and BACKFIRED: entropy rose from ~1.5 to ~1.69, policy could not
     converge (contradictory gradients vs tile-visit reward). Do NOT re-enable as a flat
     per-step penalty. The aux features already give the agent speed awareness to corner
     better without distorting the reward.

6. **best_7act.pt preserved** — the previous best 7-action checkpoint (CarRacing ~667,
   Dino ~386, MiniGrid 94%, composite 0.940). This is the resume source and the safety net.

7. **Off-track early truncation (NEW, literature-driven)** (`envs/carracing_env.py`)
   - `_wheels_on_track()` reads `car.wheels[i].tiles`; if ALL wheels are off the road for
     `offtrack_patience` physics frames (default 20 ≈ 0.4s grace) the episode TRUNCATES with
     a small sparse penalty (`offtrack_penalty=2.0`, well within the reward clip).
   - Rationale (sources: Columbia-F1-Robotics, felsangom, Mike.W "timeout"): the base env
     only emits its -100 death at the FAR playfield boundary, long after a corner spin-out,
     so the agent wastes rollout in unrecoverable states and gets a delayed, noisy signal.
     Ending the episode right when the wheels leave the road is a clean "left the track ->
     no more reward" signal and directly attacks the steep-turn spin-out complaint.
   - It is SPARSE and goal-aligned (fires only when actually off-track), unlike the dense
     slip penalty that backfired. Default ON for training (via `make_carracing_env` default);
     `infer.py` sets `offtrack_patience=0` so EVAL measures the true native game (spin-offs
     show up as real -100 deaths, which is exactly what we want the 700 floor to catch).

---

## The active run (v4) — current config
Started with:
```
cd c:\Users\shilp\Downloads\surgeShit\generalModel-3Games
$env:PYTHONUNBUFFERED = "1"; $env:SDL_VIDEODRIVER = "dummy"
python -u train_parallel.py --resume checkpoints_3games/best_7act.pt
```
Config (from `train_parallel.py` DEFAULT_ARGS):
- 8 MiniGrid envs (rollout 128), 4 Dino workers (rollout 1024, BC anchor coef 0.5)
- 4 CarRacing envs (rollout 1024), 9 actions, skip=2, aux ON, reward-norm ON, slip OFF
- batch 128, epochs 4, lr linear decay, 2000 updates
- `SDL_VIDEODRIVER=dummy` MUST be set in the shell to avoid pygame OOM.

On resume the loader reinitializes 3 tensors (`carracing_encoder.fc.0.weight`,
`carracing_actor.2.weight/bias`) because of the aux input + 9 actions. The CarRacing critic
is also reinitialized for the reward-norm scale. Expect CarRacing to start NEGATIVE (~-60)
and climb. This is normal.

---

## Monitoring procedure (for the implementer)
- Each update writes one JSON line to `checkpoints_3games/train_log.jsonl`.
- Quick read (PowerShell):
```
$lines = Get-Content "checkpoints_3games/train_log.jsonl" | Select-Object -Last 20
$lines | ForEach-Object { $j = $_ | ConvertFrom-Json; Write-Host ("upd {0,3}: car_raw={1,7:F1} car_H={2,4:F2} dino={3,6:F1} mg_ema={4:F3}" -f $j.update, $j.tasks.carracing.mean_raw_return, $j.tasks.carracing.entropy, $j.tasks.dino.mean_score, $j.tasks.minigrid.ema_solve) }
```
- Healthy signs: `car_H` (entropy) trending DOWN toward ~1.0; `car_raw` trending UP.
- Report to the user every ~15-25 updates with the table above.

### History to compare against (so you know if v4 is actually better)
- 7-action run: reached CarRacing ~600-690 avg by ~update 38-50. Could not brake+steer.
- 9-action (no aux, no skip change): plateaued ~450-520, never beat 7-action.
- v3 (slip penalty 0.5, 2 envs): entropy ROSE to 1.69, stuck ~400 avg. Abandoned.
- v4 (this run) is the cleanest attempt: aux + skip=2 + 4 envs + NO slip penalty.

---

## Decision tree

### Checkpoint at update ~50
- If `car_raw` last-10 avg >= 550 AND entropy < 1.4 -> on track, continue to ~100-200.
- If `car_raw` plateaus < 450 with entropy stuck >= 1.5 -> see "Escalation" below.

### Checkpoint at update ~100-200
- If `car_raw` last-15 avg >= 700 AND entropy ~1.0-1.2 -> SUCCESS path. Go to "Final eval".
- If 550-700 -> let it run further (LR is still decaying; late gains are common). Re-check
  every ~50 updates. Reaching a 700 AVERAGE usually needs more updates than 600 did.
- If still < 550 -> "Escalation".

---

## Escalation options (apply ONE at a time, in this order)
Only if v4 (now WITH off-track truncation) plateaus below target. Each is env/training-only
(architecture preserved). Every option below is backed by the literature review (see bottom).

1. **Per-step reward clip to +1.0** (Mike.W "Solving CarRacing"; UCSD CSE-251B). Clip the raw
   per-step reward to <= +1.0 BEFORE reward-norm. Mike.W's explicit reasoning: uncapped reward
   "gives too many incentives to get into high speeds which translates into losing control over
   tight curves" — i.e. the exact spin-out problem. SPARSE+safe, no contradictory gradient like
   the slip penalty. This is the single most-cited corner-stability fix; promote it to FIRST if
   spin-outs persist after off-track truncation. Add an `offtrack`-style param to the wrapper.

2. **More envs / lower variance.** Raise CarRacing envs 4 -> 6 (watch memory; if OOM, drop
   `zoom_skip` to 10 or stagger resets). More episodes per update = less noisy gradient.

3. **CarRacing-focused phase.** Run `--carracing-only --resume checkpoints_3games/latest.pt`
   for a few hundred updates to let it specialize, then resume joint training. The shared
   backbone is already trained for the other two games via best_7act.pt.

4. **Speed-x-steering coupling penalty** (Nature s41598-025-27702-6, "R2" reward). MORE principled
   than the failed flat slip penalty: penalize only the COMBINATION of high speed AND large
   steering, e.g. `-k * speed_norm * is_turn_action` (we already have speed in aux and know the
   action). Penalizes fast hairpin entry specifically, not all sliding. Keep `k` tiny (~0.05) and
   sparse; abort if entropy rises like the slip-penalty run did.

5. **De-emphasize high-reward episodes** (Ceudan/Car-Racing). Top agents spin out ~1 in 10
   episodes, dropping a 930 avg to ~870. Ceudan fixed this by training LESS on the highest-return
   (dangerous-speed) episodes and MORE on failures. In our PPO: down-weight minibatch samples from
   the top-return CarRacing episodes per rollout. Directly raises the worst-case (which is what the
   700 floor measures), at some cost to peak speed. More invasive — implement only if needed.

6. **Green/grass-contact penalty** (UCSD CSE-251B reached 916 with grass detection + accel aug).
   Penalize per-corner-of-car-on-grass to keep the car centered. NOTE their warning: it makes the
   car drive slowly unless paired with a speed bonus — so it can fight the time-to-finish reward.
   Treat like #4: tiny, monitored. Off-track truncation already captures most of this benefit.

7. **Turn-rate curriculum** (PRISHIta123/Curriculum_RL_for_Driving). Master gentle turns before
   hairpins — same philosophy as our MiniGrid size curriculum. CarRacing-v3 doesn't expose turn
   rate without env patching, so this is heavier; consider only as a last structural lever.

8. **Entropy coef decay.** Lower `--entropy` from 0.01 to 0.005 after update ~80 so the policy
   commits to a cleaner cornering strategy once it's found one.

9. **Frame stack 4 -> 6** (last resort; resizes conv input, larger reinit). More temporal
   context for speed/heading estimation.

---

## Final evaluation (the real acceptance test)
Score alone is NOT sufficient — the user cares about cornering quality.

1. Quantitative (headless), 10 episodes, deterministic argmax:
```
python infer.py --task carracing --ckpt checkpoints_3games/best.pt --episodes 10
```
   - Pass: mean raw return >= 700 AND min episode return not catastrophically negative
     (no -50/-90 wipeouts, which indicate spin-offs). The high mean floor is what proves
     the car isn't messing up on hard corners — a single spin-out tanks the average.
   - Note the action distribution print — `brake+left`/`brake+right` should be NON-zero,
     confirming the car learned to brake through turns.

2. Visual confirmation (the actual complaint), render a few episodes:
```
python infer.py --task carracing --ckpt checkpoints_3games/best.pt --episodes 3 --render
```
   - Watch hairpins specifically. Pass = car slows + stays on track through steep turns,
     no spin-outs.

3. Non-regression sanity:
```
python infer.py --task all --ckpt checkpoints_3games/best.pt --episodes 5
```
   - MiniGrid solves most episodes, Dino mean score >= 200.

4. If `best.pt` (composite metric) underperforms on cornering vs raw peak, also eval
   `latest.pt`. Report both to the user and let them pick.

---

## Key files
- `envs/carracing_env.py` — actions, aux physics extraction, wrappers, reward-norm vec env.
- `multi_task_ppo.py` — `CarRacingCNNEncoder` (aux concat), tolerant `load()`.
- `train.py` — `_attach_aux()`, rollout, curriculum, best.pt (composite) + latest.pt saving.
- `train_parallel.py` — headless launcher with DEFAULT_ARGS (the v4 config).
- `infer.py` — eval with aux features; `reward_clip=0` so it reports TRUE game score.
- `checkpoints_3games/` — `best.pt` (current run best, composite-min metric),
  `best_7act.pt` (safety net), `latest.pt`, `train_log.jsonl`.

## Hard rules
- Always set `SDL_VIDEODRIVER=dummy` for headless multi-env runs (else pygame OOM).
- Do NOT re-add a flat slip penalty.
- Do NOT change the shared-backbone architecture or switch to a continuous policy.
- Keep `best_7act.pt` untouched as the fallback.
- Only create commits if the user explicitly asks.

---

## Literature review (sources behind the choices above)
Surveyed how published / open-source CarRacing agents reach high scores and, specifically, how
they stop spinning out on steep turns. Common ground across all of them and how it maps here:

- **Frame stacking (4)** for velocity/heading — we already do this (4x96x96) plus explicit aux
  physics (speed/slip/ang-vel), which is strictly more information.
- **Reward clipping** so speed-greed doesn't cost cornering control — we clip (+/-10) and now also
  recommend Mike.W's tighter +1.0 per-step cap as escalation #1.
- **Off-track handling / early termination** — the most consistent corner-stability lever across
  sources. IMPLEMENTED as wheel-contact truncation (see "What's implemented" #7).
- **Hyperparameters** (lr 3e-4, gamma 0.99, lambda 0.95, clip 0.2, vf_coef 0.5, grad-clip 0.5,
  ent 0.01) — our config already matches the consensus.

Sources:
- Mike.W, "Solving CarRacing with PPO" (notanymike.github.io) — action/obs design, reward clip
  to +1, off-track timeout.
- Ceudan/Car-Racing (GitHub) — speed vs cornering trade-off; de-emphasize high-reward episodes
  to lift the worst case (930 peak, 870 avg -> stabilized).
- felsangom/GymnasiumAI & Columbia-F1-Robotics/f1_robotics_racing_sim (GitHub) — wheel-contact
  off-track detection + early truncation (`car.wheels[i].tiles`), the basis of our new feature.
- UCSD CSE-251B CarRacing project (dimademler.com) — grass detection + acceleration aug reached
  mean reward 916.8; warns grass penalty alone drives too slowly.
- "Reward design and hyperparameter tuning..." Sci. Reports s41598-025-27702-6 — R2 reward with a
  speed-x-steering coupling penalty for cornering stability.
- PRISHIta123/Curriculum_RL_for_Driving (GitHub) — turn-rate curriculum for PPO on CarRacing.
- ak811/carracing-ppo, frankcholula/ppo-CarRacing-v3, Droid-DevX/AutonomousDriving — confirm the
  consensus PPO/CNN hyperparameters and frame-stack/grayscale preprocessing we already use.
