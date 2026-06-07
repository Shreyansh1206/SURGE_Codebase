# generalModel-v1 — Multi-Task Game-Playing Agent

A single neural policy trained with **Proximal Policy Optimization (PPO)** to play two different games simultaneously:

| Task | Environment | Backend |
|------|-------------|---------|
| **MiniGrid** | `MiniGrid-DoorKey-16x16-v0` (16×16 map, 7×7 view) | [Farama MiniGrid](https://github.com/Farama-Foundation/Minigrid) via Gymnasium |
| **Dino** | Chrome T-Rex runner (Python port) | Local pygame engine in `Dino_runGame/` |

The agent uses **task-specific encoders and actor/critic heads** with a **shared representation backbone**, and an **interleaved training loop** that prevents gradient crosstalk between incompatible action spaces.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [System Architecture](#system-architecture)
4. [Neural Network Design](#neural-network-design)
5. [Training Algorithm](#training-algorithm)
6. [Environment Specifications](#environment-specifications)
7. [File Reference](#file-reference)
8. [CLI Reference](#cli-reference)
9. [Checkpoints and Logging](#checkpoints-and-logging)
10. [Design Decisions](#design-decisions)
11. [Known Limitations](#known-limitations)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Install

```powershell
cd generalModel-v1
pip install -r requirements.txt
```

### Train

```powershell
# Full multi-task training (MiniGrid + Dino) — fresh run, new checkpoint dir
python train_parallel.py

# Or via train.py directly
python train.py --save-dir checkpoints_doorkey_16x16

# Dino only (no minigrid install needed)
python train.py --dino-only --updates 100

# MiniGrid only
python train.py --minigrid-only --updates 200

# Watch the Dino game while training
python train.py --dino-only --render-dino

# Resume from checkpoint
python train.py --resume checkpoints/latest.pt
```

### Play the original Dino game manually

```powershell
cd Dino_runGame
python main.py
```

The RL pipeline does **not** use `main.py` directly. It uses the headless engine in `Dino_runGame/engine.py`, which shares the same sprites and game logic.

---

## Project Structure

```
generalModel-v1/
├── README.md                  # This document
├── requirements.txt           # Python dependencies
├── train.py                   # Main multi-task training orchestrator
├── multi_task_ppo.py          # MultiTaskActorCritic + MultiTaskPPO agent
├── dino_env.py                # Dino RL environment (obs stacking, rewards)
│
├── envs/
│   ├── __init__.py
│   ├── dino_gym.py            # Gymnasium wrapper around dino_env.py
│   └── minigrid_env.py        # MiniGrid factory + observation flattening
│
├── Dino_runGame/              # User-provided pygame Dino game
│   ├── main.py                # Original playable game (unchanged)
│   ├── engine.py              # Headless, steppable RL game engine
│   └── sprites/               # Game assets (png, wav)
│
└── checkpoints/               # Created at train time (default --save-dir)
    ├── latest.pt
    ├── mt_ppo_upd{N}.pt
    └── train_log.jsonl
```

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           train.py (orchestrator)                       │
│                                                                         │
│   for each update:                                                      │
│     1. collect_rollout(MiniGrid)  →  MiniGrid RolloutBuffer             │
│     2. ppo.update_task("minigrid")  ← separate backward pass            │
│     3. collect_rollout(Dino)      →  Dino RolloutBuffer                 │
│     4. ppo.update_task("dino")    ← separate backward pass              │
└─────────────────────────────────────────────────────────────────────────┘
          │                                        │
          ▼                                        ▼
┌──────────────────────┐              ┌──────────────────────┐
│  MiniGridVecEnv      │              │  VecDinoGymEnv       │
│  (SyncVectorEnv ×8)  │              │  (sequential ×1)     │
└──────────┬───────────┘              └──────────┬───────────┘
           │                                      │
           ▼                                      ▼
┌──────────────────────┐              ┌──────────────────────┐
│  envs/minigrid_env   │              │  envs/dino_gym       │
│  FlatPartialObsWrapper│            │  (Gymnasium API)     │
│  (7×7 partial view) │              └──────────┬───────────┘
└──────────┬───────────┘                         │
           │                                      ▼
           ▼                           ┌──────────────────────┐
┌──────────────────────┐              │  dino_env.py         │
│  gym.make(MiniGrid)  │              │  frame stack + reward│
└──────────────────────┘              └──────────┬───────────┘
                                                 ▼
                                      ┌──────────────────────┐
                                      │  Dino_runGame/       │
                                      │  engine.py (pygame)  │
                                      └──────────────────────┘
```

### Why PPO Instead of DQN?

The original blueprint specified a DQN with dual replay buffers. This implementation uses **PPO** because:

- PPO is **on-policy** — each task collects fresh rollouts every update, naturally isolating task data without mixing off-policy transitions.
- Actor-critic architecture maps cleanly onto the shared-backbone / split-head design.
- Clipped surrogate objective + GAE provides stable multi-task gradient updates.

### Dual Buffer System (On-Policy Rollout Buffers)

Unlike DQN replay buffers, PPO uses **separate on-policy rollout buffers** per task:

| Buffer | Stores | State dim | Actions |
|--------|--------|-----------|---------|
| `mg_buf` | `(obs, action, logprob, reward, value, done)` per MiniGrid step | `(N_mg, obs_dim)` | 7 |
| `dino_buf` | Same tuple structure for Dino steps | `(N_dino, 48)` | 3 |

Buffers are **never merged**. Each task's transitions stay isolated through collection and optimization.

---

## Neural Network Design

### Architecture Diagram

```
[ MiniGrid State (N_mg) ]              [ Dino State (48) ]
           │                                    │
           ▼                                    ▼
   [ MiniGrid Encoder ]                  [ Dino Encoder ]
   Linear(N_mg → 128) + Tanh           Linear(48 → 128) + Tanh
           │                                    │
           └──────────────┬─────────────────────┘
                          ▼
               [ Shared Core Backbone ]
                Linear(128 → 128) + Tanh
                Linear(128 → 128) + Tanh
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
 [ MiniGrid Actor Block ]        [ Dino Actor Block ]
  Linear(128→128) + Tanh          Linear(128→128) + Tanh
  Linear(128→7)  logits           Linear(128→3)  logits
           │                             │
 [ MiniGrid Critic Block]        [ Dino Critic Block]
  Linear(128→128) + Tanh          Linear(128→128) + Tanh
  Linear(128→1)  value            Linear(128→1)  value
```

`N_mg` is the flattened partial-view size. Default MiniGrid egocentric window is **7 × 7 × 3 = 147** (object, color, state per visible cell). Tiles outside the map are **unseen (0, 0, 0)**.

### Task Routing

The `forward(state, task_name)` method routes through exactly one encoder and one actor/critic pair based on `task_name ∈ {"minigrid", "dino"}`. PyTorch builds a computation graph only for the active task block, saving memory and preventing accidental cross-task gradient paths during action selection.

### Parameter Count (approximate)

| Component | Parameters |
|-----------|------------|
| MiniGrid encoder (147→128) | ~19,100 |
| Dino encoder (48→128) | ~6,300 |
| Shared core (128→128→128) | ~33,000 |
| MiniGrid actor + critic | ~33,800 |
| Dino actor + critic | ~33,300 |
| **Total** | **~126,000** |

---

## Training Algorithm

### Interleaved Update Cycle

Each training **update** (one iteration of the outer loop) performs:

```
1. MiniGrid rollout collection
   └─ n_minigrid_envs × rollout_steps transitions → mg_buf

2. MiniGrid PPO update
   └─ GAE → clipped surrogate loss → backward → Adam step
   └─ grad clip (max_norm=0.5)

3. Dino rollout collection
   └─ n_dino_envs × rollout_steps transitions → dino_buf

4. Dino PPO update
   └─ separate GAE → separate backward → separate Adam step
   └─ grad clip (max_norm=0.5)
```

**Critical rule:** losses are **never summed** across tasks before `backward()`. Each `update_task()` call performs its own independent optimization step. This prevents a high-magnitude Dino death penalty from drowning out subtle MiniGrid gradients (or vice versa).

### PPO Hyperparameters (defaults)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `clip_eps` | 0.2 | PPO clip range |
| `epochs` | 4 | Optimization epochs per rollout |
| `batch_size` | 64 | Minibatch size |
| `value_coef` | 0.5 | Value loss weight |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `target_kl` | 0.015 | Early-stop KL divergence limit |

### GAE (Generalized Advantage Estimation)

Advantages are computed per-task using vectorized GAE over the rollout tensor `(T, N_envs)`:

```
δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
A_t = δ_t + γ · λ · (1 - done_t) · A_{t+1}
```

Advantages are normalized (zero mean, unit variance) before the PPO update.

### Samples Per Update

```
total_samples_per_update = (n_minigrid_envs + n_dino_envs) × rollout_steps
```

With defaults: `(8 + 1) × 128 = 1,152` environment steps per update.

---

## Environment Specifications

### MiniGrid

**Source:** `envs/minigrid_env.py`

| Property | Value |
|----------|-------|
| Default env ID | `MiniGrid-DoorKey-16x16-v0` |
| Map size | **16×16** (`DoorKey-16x16`) |
| Observation | **7×7** egocentric field-of-view (default MiniGrid) |
| Obs dim | 147 (7 × 7 × 3) |
| Task | Pick up key, open door, reach goal |
| Action space | 7 discrete |
| Vectorization | `gymnasium.vector.SyncVectorEnv` (default 8 parallel envs) |
| Max episode steps | 1000 |

**Wrappers applied:**
1. *(no full-grid wrapper)* — partial 7×7 view only; out-of-map cells are unseen `(0, 0, 0)`
2. `FlatImageObsWrapper` — flattens `(7, 7, 3)` → float32 vector in [0, 1]

Each visible cell encodes `[object_id, color_id, state_id]` (key, door, wall, goal, etc.).

**Actions:**

| ID | Action |
|----|--------|
| 0 | Turn left |
| 1 | Turn right |
| 2 | Move forward |
| 3 | Pickup |
| 4 | Drop |
| 5 | Toggle |
| 6 | Done |

**Reward:** Native MiniGrid goal reward plus milestone bonuses (no distance shaping):

| Event | Reward |
|-------|--------|
| Pick up key (once per episode) | +0.25 |
| Drop key after pickup | -0.05 (each drop) |
| Open a door (once per door per episode) | +0.35 |
| Reach goal | `1 - 0.9 × (step_count / max_steps)` (native) |
| All other steps | 0 |

---

### Dino (Python / Pygame)

**Source chain:** `Dino_runGame/engine.py` → `dino_env.py` → `envs/dino_gym.py`

| Property | Value |
|----------|-------|
| Canvas size | 600 × 150 pixels |
| Observation | 48-dim float32 (12 features × 4-frame stack) |
| Action space | 3 discrete |
| Vectorization | Sequential (default 1 env; pygame is not thread-safe) |
| Rendering | Off by default; pass `--render-dino` to show window |

**Actions:**

| ID | Action | Effect |
|----|--------|--------|
| 0 | No-op | Release duck |
| 1 | Jump | Tap jump (only if on ground) |
| 2 | Duck | Hold duck pose |

**Per-frame features (12 values):**

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | Dino Y position | ÷ 150 |
| 1 | Jumping flag | 0 or 1 |
| 2 | Ducking flag | 0 or 1 |
| 3 | Game speed | ÷ 13 (clipped) |
| 4 | Nearest obstacle Δx | relative to dino, ÷ 200 |
| 5 | Nearest obstacle Y | ÷ 150 |
| 6 | Nearest obstacle width | ÷ 600 |
| 7 | Nearest obstacle height | ÷ 150 |
| 8–11 | Second-nearest obstacle | same layout as 4–7 |

**Frame stacking:** 4 consecutive frames are concatenated → 48-dim vector. This gives the MLP implicit velocity/timing information without an RNN.

**Reward shaping:**

| Event | Reward |
|-------|--------|
| Death (collision) | -10.0 |
| Alive per game frame | +0.02 |
| Score increase | +0.1 × score_delta |
| Obstacle passed | +1.0 |
| Jump action | -0.01 (discourages spam-jumping) |

**Obstacle pass detection:** Latches when the nearest obstacle enters the approach window, then fires when it despawns or the front slot jumps to a far obstacle. The old Selenium check (`prev_x > DINO_X` and `cur_x > prev_x`) never triggered in pygame because obstacles move left and often clear at negative x.

**Return scale:** A short death is ~-10. A strong run (score ~400+) can reach +100 or more, so episode return now correlates with survival skill rather than clustering near -10.

**Game engine details (`Dino_runGame/engine.py`):**
- Extracted from the original `main.py` by Rohit Rane
- Runs headless by default (`SDL_VIDEODRIVER=dummy`)
- Skips the intro screen — episodes start immediately
- Auto-resets on death (via `VecDinoEnv`)
- Obstacles: cacti (ground) and pterodactyls (flying, after score ~500)
- Speed increases every 700 frames

---

## File Reference

### `train.py`
Main entry point. Parses CLI args, constructs environments and agent, runs the interleaved training loop, logs metrics, saves checkpoints.

### `multi_task_ppo.py`
Contains:
- `MultiTaskActorCritic` — task-routed actor-critic network
- `RolloutBuffer` — on-policy transition storage `(T, N, ...)`
- `compute_gae_vec()` — vectorized GAE
- `MultiTaskPPO` — optimizer wrapper with `update_task()` per task

### `dino_env.py`
Low-level Dino RL interface:
- `DinoEnv` — single environment instance
- `VecDinoEnv` — multi-env wrapper (sequential stepping)
- Exports `OBS_DIM=48`, `N_ACTIONS=3`

### `envs/dino_gym.py`
Gymnasium-compatible wrapper:
- `DinoGymEnv(gym.Env)` — standard `reset()` / `step()` / `close()`
- `VecDinoGymEnv` — batched interface matching MiniGrid vec API
- Returns `(obs, reward, terminated, truncated, info)` tuples

### `envs/minigrid_env.py`
- `make_minigrid_env()` — factory with wrappers
- `minigrid_obs_dim()` — probe observation size for network init
- `MINIGRID_ACTIONS = 7`

### `Dino_runGame/engine.py`
Headless pygame game engine:
- `DinoGameEngine` — `reset()`, `step(action)`, `get_state()`, `close()`
- Sprite loading uses absolute paths (no cwd dependency)
- `render=True` opens a visible pygame window

### `Dino_runGame/main.py`
Original human-playable game. **Not used by the RL pipeline.** Kept unchanged for manual play and reference.

---

## CLI Reference

```
python train.py [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--minigrid-env-id` | `MiniGrid-DoorKey-16x16-v0` | Gymnasium MiniGrid environment ID |
| `--n-minigrid-envs` | 8 | Parallel MiniGrid instances |
| `--n-dino-envs` | 1 | Dino instances (keep at 1; pygame not thread-safe) |
| `--updates` | 500 | Total training updates |
| `--rollout` | 128 | Steps per env per task per update |
| `--lr` | 3e-4 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--lam` | 0.95 | GAE lambda |
| `--clip` | 0.2 | PPO clip epsilon |
| `--epochs` | 4 | PPO epochs per rollout |
| `--batch-size` | 64 | Minibatch size |
| `--entropy` | 0.01 | Entropy coefficient |
| `--save-dir` | `checkpoints` | Checkpoint output directory |
| `--save-every` | 25 | Save checkpoint every N updates |
| `--resume` | None | Path to checkpoint to resume from |
| `--seed` | 0 | Random seed |
| `--dino-only` | off | Train Dino only (skip MiniGrid) |
| `--minigrid-only` | off | Train MiniGrid only (skip Dino) |
| `--render-dino` | off | Show pygame Dino window during training |

---

## Checkpoints and Logging

### Checkpoint format

Saved as PyTorch `.pt` files containing:

```python
{
    "net": state_dict,       # MultiTaskActorCritic weights
    "optim": state_dict,     # Adam optimizer state
    "minigrid_dim": int,     # MiniGrid observation dimension
    "dino_dim": int,         # Dino observation dimension (48)
}
```

**Files written:**
- `checkpoints/latest.pt` — always saved (including on Ctrl+C)
- `checkpoints/mt_ppo_upd{N}.pt` — periodic saves every `--save-every` updates

### Training log

`checkpoints/train_log.jsonl` — one JSON object per update:

```json
{
  "update": 42,
  "elapsed": 183.5,
  "tasks": {
    "minigrid": {
      "pi_loss": -0.012,
      "v_loss": 0.34,
      "entropy": 1.82,
      "kl": 0.008,
      "grad_norm": 0.41,
      "roll_time": 1.2,
      "upd_time": 0.3,
      "episodes": 5,
      "mean_return": 0.85,
      "mean_len": 48.2
    },
    "dino": {
      "pi_loss": -0.021,
      "v_loss": 0.12,
      "entropy": 0.95,
      "kl": 0.005,
      "grad_norm": 0.38,
      "roll_time": 4.8,
      "upd_time": 0.2,
      "episodes": 2,
      "mean_return": -3.5,
      "mean_score": 28.0,
      "max_score": 45
    }
  }
}
```

### Console output

```
upd   42 | mg ret   0.85 pi -0.012 v 0.340 H 1.820 | dino ret  -3.50 score  28.0 pi -0.021 v 0.120 |    184s
```

---

## Design Decisions

### 1. Task-specific encoders, shared backbone
MiniGrid states are 147-dim partial-view encodings (7×7 egocentric window on a 16×16 DoorKey map); Dino states are 48-dim physics features. A single input layer cannot serve both. Task-specific encoders project each into a common 128-dim space where the shared backbone learns transferable representations (spatial reasoning, timing, hazard avoidance).

### 2. Split actor AND critic blocks (not just split policy heads)
Each task gets its own actor trunk (128→128→actions) and critic trunk (128→128→1). Value functions have different scales across games (MiniGrid sparse 0/1 rewards vs Dino shaped -10 to +1), so isolated critics prevent value interference.

### 3. Separate `update_task()` calls, never summed losses
```python
# CORRECT (what we do)
ppo.update_task("minigrid", mg_buf, ...)
ppo.update_task("dino", dino_buf, ...)

# WRONG (would cause gradient crosstalk)
loss = loss_mg + loss_dino
loss.backward()
```

### 4. Gradient clipping on shared backbone
`max_grad_norm=0.5` is applied to **all** network parameters after each task update. The shared core receives gradient injections from both tasks within the same update cycle; clipping prevents one task from causing catastrophic weight changes.

### 5. Headless Dino via extracted engine
The original `main.py` is an event-loop game for human play. `engine.py` reimplements the same logic as a steppable `reset()/step(action)` API without keyboard input or intro screens, suitable for RL batch training.

### 6. Lazy MiniGrid imports
`train.py --dino-only` does not require `minigrid` to be installed. MiniGrid modules are imported only when that task is active.

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| Dino vectorization | Pygame is not thread-safe. `n_dino_envs > 1` steps environments sequentially, not in parallel. |
| No inference script yet | Checkpoints can be loaded via `MultiTaskPPO.load()` but no `infer.py` is bundled. |
| MiniGrid partial view fixed at 7×7 | Obs dim is 147 for standard MiniGrid envs. Map can be 16×16 while the agent only sees 7×7 ahead. |
| Dino obs dim fixed at 48 | Changing `FRAME_STACK` or `FEATURES_PER_FRAME` in `dino_env.py` requires retraining from scratch. |
| Single optimizer | Both tasks share one Adam optimizer. Alternatives (separate optimizers per task) are not implemented. |
| No evaluation mode | Training metrics only; no held-out evaluation loop. |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'minigrid'`
Install dependencies: `pip install -r requirements.txt`, or use `--dino-only` to skip MiniGrid.

### `pygame.error: No video mode has been set`
The engine handles this automatically by calling `pygame.display.set_mode()` before sprite loading. If you see this error, ensure you are importing from `engine.py`, not `main.py`.

### `ModuleNotFoundError: No module named 'dino_env'`
Run all commands from the `generalModel-v1/` directory:
```powershell
cd generalModel-v1
python train.py
```

### Dino scores stay at 0 during early training
Expected. The agent starts with random actions and dies quickly. Returns will show `nan` until at least one episode completes within the rollout window. Increase `--rollout` (e.g. 256) for more stable metrics.

### MiniGrid returns not improving
`DoorKey-16x16` requires pickup + toggle + navigation. If returns stay flat early on, that is normal — try `--minigrid-only` first or increase `--updates`.

### Out of memory (GPU)
Reduce `--n-minigrid-envs`, `--rollout`, or `--batch-size`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network + PPO optimization |
| `numpy` | Array operations |
| `gymnasium` | Standard RL environment API |
| `minigrid` | MiniGrid environments |
| `pygame` | Dino game engine |

---

## Version History

| Version | Description |
|---------|-------------|
| **v1** | Initial multi-task PPO agent: MiniGrid + pygame Dino, shared backbone, interleaved training |
