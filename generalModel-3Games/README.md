# generalModel-3Games — Multi-Task Game-Playing Agent (3 games)

A single PPO policy with a **shared backbone** and **task-specific encoders + actor/critic heads**, trained to play **three** games in one interleaved loop:

| Task | Environment | Observation | Encoder | Actions |
|------|-------------|-------------|---------|---------|
| **MiniGrid** | `MiniGrid-DoorKey-*` (Farama MiniGrid) | 7×7×20 one-hot egocentric view (flat 980) | small CNN | 7 discrete |
| **Dino** | Chrome T-Rex runner (local pygame engine) | 48-dim physics features (12 × 4-frame stack) | MLP | 3 discrete |
| **CarRacing** | `CarRacing-v3` (Box2D, `continuous=False`) | 4×96×96 grayscale frame stack | NatureCNN | 5 discrete |

This extends `generalModel-v1` (MiniGrid + Dino) by adding **CarRacing** as a third task. The PPO core, rollout buffers, GAE, and per-task isolated updates are unchanged in spirit — each task still gets its own rollout buffer and its own `update_task()` backward pass so gradients never cross between incompatible action spaces.

---

## Design decisions for CarRacing

### 1. CNN encoder (not game variables)
Unlike Dino — whose custom engine exposes structured physics features (obstacle Δx, speed, etc.) — `CarRacing` only returns a **96×96×3 RGB image**. No low-dimensional state vector is available through the gymnasium API. So CarRacing **requires a CNN encoder**, like MiniGrid (but deeper).

- RGB → grayscale (luminosity) → keep full **96×96** (the bottom dashboard bar encodes speed/steering/gyro, useful since we have no state vars).
- **Frame stacking** of 4 frames → `(4, 96, 96)` so the net can infer velocity/heading.
- Encoder is an **Atari-style NatureCNN**: `Conv(8,s4) → Conv(4,s2) → Conv(3,s1) → Linear → Tanh`, projecting into the shared 128-dim space.

### 2. Discrete action space
The whole codebase uses a discrete `Categorical` policy. CarRacing is created with `continuous=False` → `Discrete(5)`:

| ID | Action |
|----|--------|
| 0 | noop |
| 1 | steer left |
| 2 | steer right |
| 3 | gas |
| 4 | brake |

This drops into the existing PPO core with zero changes.

### 3. Reward + decoder
CarRacing ships a **dense native reward**: `−0.1` per frame and `+1000/N` per new track tile visited (episode ends when all tiles are visited). We use this native reward directly. The "decoder block" is just the per-task actor + critic heads (`128→128→{5, 1}`).

Light, training-friendly wrappers (in `envs/carracing_env.py`):
- **Frame skip / action repeat** (default 4) — repeats each action for N frames, summing reward.
- **Zoom skip** (default 40 frames on reset) — skips the start-of-episode zoom-in animation.
- **No-progress early stop** (default 50 aggregated steps with no positive reward) — terminates when the car is stuck / off-track, since CarRacing's per-frame reward is mostly negative between tile crossings.

---

## Quick start

```powershell
cd generalModel-3Games
pip install -r requirements.txt
# Box2D wheels can fail to build on Windows; if so use conda:
#   conda install -c conda-forge box2d-py
```

### Train

```powershell
# All three games, headless parallel (recommended)
python train_parallel.py

# Direct, with custom knobs
python train.py --save-dir checkpoints_3games --updates 500

# Single task (smoke tests)
python train.py --carracing-only --updates 100
python train.py --dino-only
python train.py --minigrid-only

# Drop one task
python train.py --no-minigrid

# Watch a game while training
python train.py --carracing-only --render-carracing
```

### Inference

```powershell
python infer.py --task all --ckpt checkpoints_3games/latest.pt
python infer.py --task carracing --render
python infer.py --task dino --sample
```

---

## Architecture

```
[MiniGrid 7x7x20]   [Dino 48]        [CarRacing 4x96x96]
       │                │                    │
   MiniGrid CNN     Dino MLP           CarRacing NatureCNN
       │                │                    │
       └────────────────┴────────────────────┘
                        ▼
              Shared core (128→128→128, Tanh)
          ┌──────────────┼──────────────┐
          ▼              ▼               ▼
   per-task actor   per-task actor   per-task actor
   per-task critic  per-task critic  per-task critic
```

Per update, the loop runs (when enabled): MiniGrid rollout + update → Dino rollout + update → CarRacing rollout + update. Losses are **never summed across tasks**; `max_grad_norm` clipping protects the shared backbone from any single task dominating.

### Checkpoint format

```python
{
    "net": state_dict,
    "optim": state_dict,
    "minigrid_dim": int,            # e.g. 980
    "dino_dim": int,                # 48
    "carracing_obs_shape": [4,96,96],
}
```

---

## Files

| File | Purpose |
|------|---------|
| `multi_task_ppo.py` | `MultiTaskActorCritic` (3 encoders/heads incl. `CarRacingCNNEncoder`), `RolloutBuffer`, GAE, `MultiTaskPPO` |
| `train.py` | 3-task interleaved training orchestrator + CLI |
| `train_parallel.py` | Headless parallel preset wrapper |
| `infer.py` | Inference for MiniGrid / Dino / CarRacing |
| `envs/carracing_env.py` | CarRacing wrappers (grayscale, frame-stack, frame-skip, early-stop) + vec env |
| `envs/dino_gym.py`, `dino_env.py`, `Dino_runGame/` | Dino game (copied from v1) |
| `envs/minigrid_env.py` | MiniGrid factory + observation flattening (copied from v1) |

---

## Key CarRacing CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n-carracing-envs` | 4 | Parallel CarRacing instances (`SyncVectorEnv`) |
| `--carracing-rollout` | 128 | Rollout steps per CarRacing env per update |
| `--carracing-only` | off | Train CarRacing only |
| `--no-carracing` | off | Disable CarRacing |
| `--render-carracing` | off | Show the CarRacing window (forces single env) |
