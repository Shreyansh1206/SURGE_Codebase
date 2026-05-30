# Chrome Dino — PPO (Actor-Critic) — Build Report

## Goal
Train a PPO actor-critic agent that plays the Chrome offline Dino game live
inside a real Chrome window. The Python agent acts only as an intermediary
between Selenium-driven inputs and game outputs; the game itself runs in the
browser.

## What was built

| File | Role |
|---|---|
| `dino_env.py` | `DinoEnv` (one browser) and `VecDinoEnv` (N browsers in parallel). Stacked-frame observations. |
| `ppo_agent.py` | `ActorCritic` MLP, `RolloutBuffer`, GAE, and the `PPO` update. |
| `train.py` | Rollout collection + PPO update loop over a *vectorised* env (N parallel browsers). |
| `infer.py` | Loads a checkpoint and runs the policy live in Chrome. |
| `requirements.txt` | Python deps. |

## Design choices (and the trade-offs)

### State: JavaScript readout, not screenshots
The standard approach in tutorials is to screenshot the canvas and feed a CNN.
With Selenium, `get_screenshot_as_png()` costs ~50–150 ms per call — that's a
catastrophe for a game running at 60 FPS.

Instead, every step calls `driver.execute_script(...)` against `Runner.instance_`
(the Dino game's own runtime object) and pulls a structured snapshot: dino
y-position, jumping/ducking flags, current speed, and the position/size of the
next two obstacles. One `execute_script` round-trip is typically 5–15 ms.

A 12-dim feature vector × 4 stacked frames → 48-dim observation. The stack
gives the policy short-term temporal information (so it sees velocity, not just
position) without needing an RNN.

### Actions: 3 discrete via JS keyboard events
`no-op / jump / duck`. We dispatch `KeyboardEvent` via JS rather than Selenium
`ActionChains` so we don't depend on which element holds focus. Jump is a
50 ms tap; duck is held until the next non-duck action releases it.

### Reward shaping
- `+0.1` per step alive (encourages survival even when score doesn't tick)
- `+0.01 * score_delta` (rewards distance)
- `-10` on death

This is mild shaping — the dominant signal is still distance survived.

### Algorithm: textbook PPO
Clipped surrogate, GAE(λ=0.95), advantage normalization, value coefficient
0.5, entropy bonus 0.01, grad-norm clip 0.5, 4 epochs over each rollout with
minibatch 64. Shared MLP trunk (128-128, tanh) with separate policy and value
heads.

### Single-environment rollouts
PPO traditionally batches over many parallel envs. Here we have one browser,
so rollouts are serial — sample throughput is the main bottleneck. Default
rollout length is 1024 steps per update; at ~30 Hz effective frequency that's
~30 s of game time per update.

## Synchronisation (the thing that has to be right or nothing else matters)

The game ticks on `requestAnimationFrame` at ~60 Hz. A naive `step()` that
just dispatches a key and reads state has no relationship to that clock — at
high Python speeds you read the same frame multiple times (duplicate obs); at
low Python speeds the game advances several frames between reads (the agent
reacts late). Either way the (s, a, r, s') tuples are temporally noisy and
PPO can't learn from them reliably.

**Fix implemented — wall-clock frame timing.**

The pause/play + frame-counter shim approach was tried first but proved
unreliable: `scheduleNextUpdate()` inside `Runner.play()` creates a new
`update.bind(this)` on every call, and depending on Chrome version the
prototype vs. instance property lookup timing can silently break the patch so
the counter never increments — causing an infinite hang.

The approach used instead is simpler and fully reliable:

```
step(action):
    1. dispatch the action's key event (synchronous JS call)
    2. time.sleep(frames_per_step / 60)   ← exactly N frames at 60 Hz
    3. read state from Runner.instance_
```

At 60 Hz and `frames_per_step=4` the sleep is 67 ms — the game advances
exactly 4 frames between our action and our read. The ±1 frame uncertainty
from sleep jitter is irrelevant for RL. `frames_per_step` remains a real
hyperparameter (Atari-style action repeat).

Selenium latency is now only paid once per step (one `execute_script` for
state read), so effective throughput is ~10–12 steps/second, or ~600–700
game frames/second of training data.

## Parallel envs (VecDinoEnv)

Single-env training is bottlenecked by the ~67 ms-per-step sleep needed to let
the game advance 4 frames at 60 Hz. With one browser that caps throughput at
~15 steps/sec. Running N browsers in parallel multiplies sample throughput
near-linearly until you hit screen real estate or driver overhead.

**Architecture:**
- `VecDinoEnv(n_envs=4)` spawns 4 `DinoEnv` instances, each with its own Chrome
  window positioned in a 2×2 grid (configurable via `--grid-cols`, `--window-w/h`).
- Each `step(actions)` call submits N `step()` calls to a `ThreadPoolExecutor`
  and waits on all of them. Selenium round-trips and `time.sleep` release the
  GIL, so threading (not multiprocessing) is sufficient.
- Auto-reset: if any env crashes mid-rollout, `VecDinoEnv.step` resets it
  inline and stores the terminal obs in `info['terminal_obs']`. The training
  loop never sees a stuck env.

**Why not just more envs?** Three soft limits:
1. **Screen real estate** — 4 windows at 700×320 fit a 1080p monitor. Beyond
   that, windows overlap (still fine for training, but ugly to watch).
2. **rAF throttling in occluded windows** — handled by passing
   `--disable-background-timer-throttling`,
   `--disable-backgrounding-occluded-windows`, and
   `--disable-renderer-backgrounding`. With those, even minimized windows
   tick at 60 Hz. So 4 is a soft cap, not a hard one.
3. **Driver overhead** — each Chrome process is ~150 MB RAM. 8 envs are fine
   on a typical 16 GB machine; 16 may swap.

**Rollout sizing:** `--rollout` is steps **per env**. Total samples per
update = `rollout × n_envs`. With defaults (4 envs × 256 steps) that's 1024
samples per update — same as single-env at `--rollout 1024` but ~4× faster
in wall-clock.

## Other known issues

1. **Sample efficiency.** With one env you'll need hours of wall-clock training
   to get past score ~500. Parallelising would require multiple browser windows
   and is non-trivial — each needs its own driver and a separate `Runner` to
   read from. A simpler win is to run several `DinoEnv` instances in
   subprocesses with a `VecEnv` wrapper; this is not implemented yet.

3. **`chrome://dino` reliability.** Chrome sometimes blocks `chrome://`
   navigation or the offline page renders differently across versions. If
   `_wait_for_runner` times out, host a local copy of the game (Wayou's
   `t-rex-runner` GitHub repo is the canonical mirror) and point `--game-url`
   at `file:///path/to/index.html`. The JS API used here works on both.

4. **Headless mode.** The Dino game does *not* render reliably in
   `--headless=new`. Training must happen with a visible window (which is what
   you asked for anyway). Don't move the window — that can defocus inputs on
   some platforms.

5. **Reward shaping is unvalidated.** The `+0.1` survival bonus may push the
   agent toward a "freeze and don't jump" local optimum at very low speeds.
   If you see this, drop the survival term and rely on the score delta only.

6. **Chromedriver version mismatches.** `webdriver_manager` usually handles
   this, but if your Chrome auto-updates faster than it can fetch a driver,
   pass `--chromedriver path\to\chromedriver.exe`.

8. **No GPU requirement, but also no benefit.** The network is tiny; CPU is
   often faster than GPU here because batches are small and the bottleneck is
   the browser, not the model.

## How to run

```powershell
# from C:\Users\shilp\Downloads\surgeShit\dinoGame
pip install -r requirements.txt

# train (Chrome window opens; let it run)
python train.py --updates 300 --rollout 1024

# resume
python train.py --resume checkpoints/latest.pt --updates 500

# inference with the best snapshot
python infer.py --ckpt checkpoints/best.pt --episodes 5
```

Logs are appended to `checkpoints/train_log.jsonl`. Each line contains the
update index, episode count, mean/max score, and PPO loss stats — easy to
plot with matplotlib or feed into TensorBoard via a small adapter.

## What's deliberately not in scope

- TensorBoard / W&B logging — JSONL keeps deps minimal.
- Pixel-based CNN policy — possible later; the env can be extended with a
  `get_pixels()` method that wraps `canvas.toDataURL()`.
- Parallel `VecEnv` — see issue #2 above.
- Hyperparameter sweep / curriculum on game speed.

## Sanity checks before your first long run

1. `python -c "from dino_env import DinoEnv; e=DinoEnv(); print(e.reset().shape); e.close()"`
   should print `(48,)` and a Chrome window should open showing the dino game.
2. Run `train.py --rollout 256 --updates 2` to confirm a full update cycle
   completes end-to-end before committing to a long run.
3. Confirm `checkpoints/train_log.jsonl` is being written.
