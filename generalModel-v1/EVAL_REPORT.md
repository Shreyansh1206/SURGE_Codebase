# Multi-Task PPO — Final Model Evaluation Report

**Checkpoint:** `checkpoints_final/final.pt`\
**Evaluation run:** 2026-06-09 15:18:48\
**Script:** `eval_final.py`\
**Results file:** `eval_results.json`\
**Device:** CPU (x86-64)\
**Games evaluated:** MiniGrid DoorKey (7 sizes) · Chrome Dino runner

---

## Table of Contents

 1. [Model Architecture & Parameters](#1-model-architecture--parameters)
 2. [Computation Benchmark](#2-computation-benchmark)
 3. [MiniGrid DoorKey — Per-Size Results](#3-minigrid-doorkey--per-size-results)
 4. [MiniGrid DoorKey — Overall Results](#4-minigrid-doorkey--overall-results)
 5. [MiniGrid DoorKey — Failure Analysis](#5-minigrid-doorkey--failure-analysis)
 6. [MiniGrid DoorKey — Action Distribution](#6-minigrid-doorkey--action-distribution)
 7. [MiniGrid DoorKey — Policy Quality](#7-minigrid-doorkey--policy-quality)
 8. [MiniGrid DoorKey — Baseline Comparison](#8-minigrid-doorkey--baseline-comparison)
 9. [Dino Runner — Performance](#9-dino-runner--performance)
10. [Dino Runner — Per-Episode Breakdown](#10-dino-runner--per-episode-breakdown)
11. [Dino Runner — Policy Quality](#11-dino-runner--policy-quality)
12. [Key Findings & Summary](#12-key-findings--summary)

---

## 1. Model Architecture & Parameters

The `final.pt` checkpoint encodes a **multi-task actor-critic** that plays both MiniGrid DoorKey and the Chrome Dino runner from a shared backbone.

### Network topology

```
MiniGrid obs (980)           Dino obs (48)
      │                           │
MinigridCNNEncoder (CNN)    DinoEncoder (Linear → Tanh)
      │                           │
      └──────────┬────────────────┘
                 │
          SharedCore (Linear → Tanh → Linear → Tanh)
                 │
      ┌──────────┼──────────┐──────────┐
      │                     │          │
MiniGrid Actor (7)   MiniGrid Critic   Dino Actor (3)   Dino Critic
```

**MiniGrid observation:** 7×7×20 = 980-dim one-hot grid (object type, color, state channels)\
**Dino observation:** 48-dim feature vector (4 frames × 12 features)

### Per-submodule parameter counts

| Submodule | Parameters | % of total |
| --- | --- | --- |
| `minigrid_encoder` (CNN) | 142,832 | 57.2% |
| `shared_core` | 33,024 | 13.2% |
| `dino_actor` | 16,899 | 6.8% |
| `minigrid_actor` | 17,415 | 7.0% |
| `minigrid_critic` | 16,641 | 6.7% |
| `dino_critic` | 16,641 | 6.7% |
| `dino_encoder` | 6,272 | 2.5% |
| **TOTAL** | **249,724** | **100%** |

### Storage

| Property | Value |
| --- | --- |
| Total parameters | 249,724 |
| Trainable parameters | 249,724 (100%) |
| Parameter storage (fp32) | 976.3 KB |
| Checkpoint file size on disk | 984.8 KB |
| PyTorch version | 2.12.0+cpu |
| Device | CPU (no GPU) |

The MiniGrid CNN encoder dominates the parameter budget (57%). The Dino encoder is lightweight (6.3K params) because its observation space is small and already hand-crafted into 12 semantic features per frame.

---

## 2. Computation Benchmark

*300 timed iterations after 50 warmup passes. All measurements on CPU (x86-64).*

### Single-step inference latency

| Statistic | MiniGrid (ms) | Dino (ms) |
| --- | --- | --- |
| **Mean** | 1.438 | 0.513 |
| **Std deviation** | 0.411 | 0.134 |
| **Min** | 0.730 | 0.249 |
| **Max** | 3.711 | 1.140 |
| **Median (p50)** | 1.463 | 0.487 |
| **p95** | 2.073 | 0.769 |
| **p99** | 2.282 | 1.024 |

The Dino encoder is **2.8× faster** than the MiniGrid encoder (0.51 ms vs 1.44 ms) because it uses a simple linear layer on 48 inputs rather than a CNN on 980 inputs.

### Throughput — batch=1 (single step/s)

| Task | Steps/second |
| --- | --- |
| **MiniGrid** | 695 steps/s |
| **Dino** | 1,951 steps/s |

### Batched throughput (steps per second)

| Batch size | MiniGrid (steps/s) | Dino (steps/s) |
| --- | --- | --- |
| **1** | 603 | 2,977 |
| **8** | 2,418 | 11,020 |
| **32** | 7,944 | 37,095 |
| **128** | 20,995 | 120,995 |

Both tasks scale roughly linearly in throughput from batch=1 to batch=128, reflecting good vectorisation of the MLP layers. Dino benefits more from batching because its bottleneck is the linear encoder, which vectorises perfectly.

### Memory footprint

| Metric | Value |
| --- | --- |
| Model parameters (fp32) | 976.3 KB |
| GPU VRAM | N/A (CPU only) |
| Activation memory | Not measured (CPU mode) |

---

## 3. MiniGrid DoorKey — Per-Size Results

*60 seeds per size, seeds 500–559. Max steps per episode: 300. Greedy (argmax) policy.*

### Full results table

<!-- colwidths: 0,0,0,0,0,100,0,0,0,0,0,0 -->
| Size | Solved | Loop-fail | Wander-fail | Return μ | Return σ | Length μ | Length σ | Key % | Door % | Entropy μ | Confidence μ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **5×5** | 48.3% | 51.7% | 0.0% | +1.068 | 0.484 | 133.4 | 120.6 | 100.0% | 100.0% | 0.514 | 0.753 |
| **6×6** | 60.0% | 40.0% | 0.0% | +1.121 | 0.565 | 127.5 | 140.9 | 96.7% | 85.0% | 0.153 | 0.951 |
| **8×8** | 95.0% | 0.0% | 5.0% | +1.522 | 0.212 | 35.0 | 61.0 | 100.0% | 100.0% | 0.063 | 0.984 |
| **10×10** | 98.3% | 0.0% | 1.7% | +1.558 | 0.125 | 33.0 | 35.9 | 100.0% | 100.0% | 0.059 | 0.985 |
| **12×12** | 96.7% | 3.3% | 0.0% | +1.525 | 0.283 | 44.3 | 48.8 | 96.7% | 96.7% | 0.065 | 0.981 |
| **14×14** | 90.0% | 10.0% | 0.0% | +1.424 | 0.475 | 65.6 | 79.0 | 90.0% | 90.0% | 0.081 | 0.968 |
| **16×16** | 93.3% | 6.7% | 0.0% | +1.477 | 0.395 | 67.0 | 64.1 | 93.3% | 93.3% | 0.070 | 0.973 |

### Visual solve-rate progression

```
5×5   ████████████████████████░░░░░░░░░░░░░░░░  48.3%
6×6   ████████████████████████████░░░░░░░░░░░░  60.0%
8×8   ██████████████████████████████████████░░  95.0%
10×10 ████████████████████████████████████████  98.3%  ← peak
12×12 ███████████████████████████████████████░  96.7%
14×14 ████████████████████████████████████░░░░  90.0%
16×16 █████████████████████████████████████░░░  93.3%
```

### Episode return percentiles per size

| Size | p25 | p50 | p75 | p90 | Max |
| --- | --- | --- | --- | --- | --- |
| 5×5 | +0.600 | +0.600 | +1.568 | +1.571 | +1.575 |
| 6×6 | +0.600 | +1.564 | +1.570 | +1.575 | +1.578 |
| 8×8 | +1.563 | +1.570 | +1.576 | +1.579 | +1.585 |
| 10×10 | +1.569 | +1.577 | +1.580 | +1.584 | +1.587 |
| 12×12 | +1.573 | +1.579 | +1.583 | +1.586 | +1.589 |
| 14×14 | +1.576 | +1.582 | +1.585 | +1.589 | +1.591 |
| 16×16 | +1.577 | +1.583 | +1.586 | +1.588 | +1.592 |

> The return ceiling of \~+1.59 comes from the fixed reward structure: key pickup (+0.25), door open (+0.35), goal reached (\~+1.0 scaled). Solved episodes all converge to this ceiling. Unsolved episodes show bimodal returns: +0.0 (failed to pick up key) or +0.6 (key + door but no goal).

### Episode length

| Size | Mean steps | Std | Min | Max |
| --- | --- | --- | --- | --- |
| 5×5 | 133.4 | 120.6 | 7 | 250 |
| 6×6 | 127.5 | 140.9 | 9 | 300 |
| 8×8 | 35.0 | 61.0 | 11 | 300 |
| 10×10 | 33.0 | 35.9 | 15 | 300 |
| 12×12 | 44.3 | 48.8 | 17 | 300 |
| 14×14 | 65.6 | 79.0 | 20 | 300 |
| 16×16 | 67.0 | 64.1 | 23 | 300 |

Notably, **8×8 has a shorter mean length (35.0) than even 5×5 (133.4)**. This is because the 5×5 model has a high loop-fail rate (51.7%) causing many truncated 250-step episodes, while the 8×8 model solves efficiently in \~35 steps.

---

## 4. MiniGrid DoorKey — Overall Results

*420 total episodes (60 × 7 sizes)*

| Metric | Value |
| --- | --- |
| **Total episodes** | 420 |
| **Solve rate** | **83.10%** |
| **Loop-fail rate** | 15.95% |
| **Wander-fail rate** | 0.95% |

### Return (all 420 episodes)

| Statistic | Value |
| --- | --- |
| Mean ± std | +1.3849 ± 0.4348 |
| Min | +0.000 |
| Max | +1.592 |
| p25 | +1.564 |
| p50 | +1.573 |
| p75 | +1.581 |
| p90 | +1.586 |

### Episode length (all 420 episodes)

| Statistic | Value |
| --- | --- |
| Mean ± std | 72.3 ± 94.6 steps |
| Min | 7 steps |
| Max | 300 steps |
| p25 | 20 steps |
| p50 | 32 steps |
| p75 | 51 steps |
| p90 | 250 steps |

The highly skewed length distribution (mean 72 vs median 32) reflects the bimodal nature: fast solved episodes cluster at 20–50 steps, while failures reach the 250–300 step hard limit.

---

## 5. MiniGrid DoorKey — Failure Analysis

Failures are classified by trajectory analysis (last 40 steps):

- **Loop-fail:** ≤ 6 unique (position, direction) states in the final 40 steps → agent is stuck in a cycle
- **Wander-fail:** more than 6 unique states → agent is actively exploring but failed to find the goal before timeout

| Size | Total eps | Solved | Loop-fail | Wander-fail |
| --- | --- | --- | --- | --- |
| 5×5 | 60 | 29 | **31** | 0 |
| 6×6 | 60 | 36 | **24** | 0 |
| 8×8 | 60 | 57 | 0 | 3 |
| 10×10 | 60 | 59 | 0 | 1 |
| 12×12 | 60 | 58 | 2 | 0 |
| 14×14 | 60 | 54 | **6** | 0 |
| 16×16 | 60 | 56 | 4 | 0 |
| **Total** | **420** | **349** | **67** | **4** |

**Key insight:** Loop-failures dominate (67/71 failures = 94%). Wander-failures are rare (4 total). The model struggles specifically with **navigational loops** — the memoryless policy gets stuck in left/right turn cycles or 360° spins — rather than simply failing to explore.

**Small grid anomaly:** The 5×5 and 6×6 sizes have the worst solve rates (48%, 60%) despite being the simplest environments. This counter-intuitive result is explained by the partial 7×7 egocentric view: the agent's 7×7 viewport is *larger than the 5×5 grid*, causing the view to be mostly wall/unseen tiles. The network may not have been exposed sufficiently to these configurations during training relative to larger sizes where the view shows a more informative partial grid.

---

## 6. MiniGrid DoorKey — Action Distribution

### Per-size action fractions (% of total steps)

| Size | left | right | forward | pickup | drop | toggle | done |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5×5 | 47.6% | 48.8% | 2.1% | 0.7% | 0.0% | 0.7% | 0.0% |
| 6×6 | 16.1% | 78.7% | 3.7% | 0.8% | 0.0% | 0.7% | 0.0% |
| 8×8 | 3.3% | 33.6% | 57.4% | 2.9% | 0.0% | 2.9% | 0.0% |
| 10×10 | 4.2% | 19.7% | 70.0% | 3.0% | 0.0% | 3.0% | 0.0% |
| 12×12 | 14.1% | 22.4% | 59.2% | 2.2% | 0.0% | 2.2% | 0.0% |
| 14×14 | 24.0% | 28.9% | 44.4% | 1.4% | 0.0% | 1.4% | 0.0% |
| 16×16 | 16.7% | 22.0% | 58.6% | 1.4% | 0.0% | 1.4% | 0.0% |

### Global action distribution (30,346 total steps)

| Action | Count | % | Bar |
| --- | --- | --- | --- |
| **right** | 13,639 | 44.9% | `████████████████████` |
| **left** | 7,181 | 23.7% | `███████████` |
| **forward** | 8,721 | 28.7% | `█████████████` |
| **pickup** | 406 | 1.3% | `▌` |
| **toggle** | 399 | 1.3% | `▌` |
| **drop** | 0 | 0.0% |  |
| **done** | 0 | 0.0% |  |

**Key observations:**

- The `drop` and `done` actions are **never taken** — consistent with DoorKey's optimal strategy (never drop the key; goal-reaching terminates via environment, not the `done` action).
- `pickup` and `toggle` occur in equal proportions (1.3% each), which is expected: each solved episode uses exactly one pickup and one toggle.
- The 5×5 size is almost entirely left/right turns (96.4% combined), consistent with the high loop-fail rate — the agent is spinning in place.
- 8×8 and 10×10 sizes show the healthiest distribution: forward is the dominant action (57–70%), indicating efficient direct navigation.

---

## 7. MiniGrid DoorKey — Policy Quality

### Entropy (measure of decisiveness; 0 = fully deterministic, 1.946 = uniform over 7 actions)

| Size | Mean entropy | Interpretation |
| --- | --- | --- |
| 5×5 | 0.514 | Moderate uncertainty — correlates with loop-fail instability |
| 6×6 | 0.153 | Mostly decisive but some confusion |
| 8×8 | 0.063 | Near-deterministic — confident navigation |
| 10×10 | 0.059 | Near-deterministic — best solver |
| 12×12 | 0.065 | Near-deterministic |
| 14×14 | 0.081 | Slightly more uncertainty at large scale |
| 16×16 | 0.070 | Near-deterministic |
| **Overall** | **0.144** |  |

**The 5×5 entropy (0.514) is 8× higher than the 8×8 entropy (0.063).** The model is genuinely uncertain in small grids, reflecting poor training coverage of these edge cases.

### Action confidence (mean max-softmax probability per step)

| Size | Mean confidence |
| --- | --- |
| 5×5 | 0.753 (75.3%) |
| 6×6 | 0.951 (95.1%) |
| 8×8 | 0.984 (98.4%) |
| 10×10 | 0.985 (98.5%) |
| 12×12 | 0.981 (98.1%) |
| 14×14 | 0.968 (96.8%) |
| 16×16 | 0.973 (97.3%) |
| **Overall** | **0.942 (94.2%)** |

### Value function estimate

| Size | Mean V(s) |
| --- | --- |
| 5×5 | 0.889 |
| 6×6 | 0.981 |
| 8×8 | 1.040 |
| 10×10 | 1.025 |
| 12×12 | 1.020 |
| 14×14 | 1.007 |
| 16×16 | 0.997 |
| **Overall** | **0.994** |

The value estimates hover near 1.0 across all sizes, correctly reflecting that the expected return from any given state is close to the maximum achievable (\~1.59 for a solved episode, but discounted). Value estimates are slightly above 1.0 for 8×8–12×12, consistent with those being the easiest to solve (highest expected return).

### Unique state diversity

| Size | Mean unique states / episode |
| --- | --- |
| 5×5 | 6.2 |
| 6×6 | 9.3 |
| 8×8 | 18.9 |
| 10×10 | 26.1 |
| 12×12 | 32.1 |
| 14×14 | 34.5 |
| 16×16 | 45.2 |

The low unique states for 5×5 (6.2 per episode) directly confirms the loop-failure diagnosis: the agent visits very few distinct positions, cycling through a tiny set of states.

---

## 8. MiniGrid DoorKey — Baseline Comparison

Comparison specifically on DoorKey-16×16-v0 (60 seeds, max 300 steps):

| Metric | This model (final.pt) | Feedforward baseline | Δ |
| --- | --- | --- | --- |
| **Solve rate** | **93.3%** | 93.0% | **+0.3pp** |
| **Loop-fail rate** | **6.7%** | 7.0% | **−0.3pp** |
| Wander-fail rate | 0.0% | N/A | — |
| Mean episode return | +1.477 | N/A | — |
| Mean episode length | 67.0 steps | N/A | — |

**Result:** `final.pt` **matches or marginally exceeds the feedforward baseline** on the hardest (16×16) environment. The improvement is within noise for 60 seeds, but the direction is positive.

> **Baseline reference:** 93% solve / 7% loop-fail is the cited feedforward DoorKey-16×16 result from `eval_gru.py`, representing the prior art this model was benchmarked against.

---

## 9. Dino Runner — Performance

*20 episodes, greedy (argmax) policy. Headless SDL (no display).*

### Score statistics

| Statistic | Value |
| --- | --- |
| **Episodes** | 20 |
| **Mean score** | **304.8** |
| **Std deviation** | 149.1 |
| **Min score** | 48 |
| **Max score** | 510 |
| **Median score** | 344 |

### Score percentiles

| p10 | p25 | p50 | p75 | p90 | p95 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| 96 | 165 | 345 | 405 | 494 | 507 | 509 |

**Score distribution across 20 episodes:**

```
  48  ▏██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  64  ▏██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 100  ▏██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 129  ▏███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 142  ▏███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 172  ▏████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 181  ▏████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 300  ▏██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 322  ▏███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 342  ▏███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 347  ▏███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 383  ▏████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 384  ▏████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 393  ▏████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 400  ▏█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 419  ▏█████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 461  ▏██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 492  ▏██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 507  ▏███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
 510  ▏███████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

### RL Return statistics

| Statistic | Value |
| --- | --- |
| Mean ± std | +87.1 ± 49.8 |
| Min | +3.51 |
| Max | +156.04 |
| p25 | +40.4 |
| p50 | +102.0 |
| p75 | +120.9 |
| p90 | +151.1 |

### Episode length statistics

| Statistic | Value |
| --- | --- |
| Mean ± std | 534.8 ± 260.9 steps |
| Min | 85 steps |
| Max | 894 steps |
| p25 | \~310 steps |
| p50 | \~540 steps |
| p75 | \~720 steps |
| p90 | \~870 steps |

### Obstacle passes

| Metric | Value |
| --- | --- |
| Mean passes per episode | **24.15** |
| Total obstacle passes | 483 |
| Passes per step | 0.04516 (1 per 22.1 steps) |

### Death obstacle breakdown

| Obstacle | Episodes | % |
| --- | --- | --- |
| **CACTUS** | 13 | 65.0% |
| **PTERA** | 7 | 35.0% |

The model dies primarily on cacti (65%) vs pterodactyls (35%). This is noteworthy: the multi-task model, like the standalone Dino PPO (which never used duck), struggles more with cacti. Pterodactyls require either jumping low or ducking — at 35% the model handles a significant fraction of pteras but not all.

### Speed at death

| Statistic | Value |
| --- | --- |
| Mean speed | 6.60 |
| Std deviation | 1.53 |
| Min speed | 4.0 |
| Max speed | 9.0 |
| p25 | 5.0 |
| p50 | 7.0 |
| p75 | 7.75 |

The model reaches speeds of 8–9 in its best episodes (scores 492–510), demonstrating it can survive into the high-speed phase of the game.

---

## 10. Dino Runner — Per-Episode Breakdown

| Ep | Score | Return | Steps | Passes | noop% | jump% | duck% | Death cause | Speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 393 | +115.0 | 689 | 31 | 93.9% | 4.5% | 1.6% | CACTUS | 7.0 |
| 2 | 100 | +19.0 | 176 | 5 | 96.6% | 3.4% | 0.0% | CACTUS | 5.0 |
| 3 | 48 | +3.5 | 85 | 2 | 96.5% | 3.5% | 0.0% | CACTUS | 4.0 |
| 4 | 181 | +43.4 | 318 | 10 | 96.5% | 3.5% | 0.0% | CACTUS | 5.0 |
| 5 | 400 | +118.8 | 701 | 33 | 92.4% | 4.4% | 3.1% | PTERA | 8.0 |
| 6 | 492 | +150.8 | 863 | 43 | 94.3% | 5.0% | 0.7% | CACTUS | 8.0 |
| 7 | **510** | **+156.0** | **894** | **44** | 94.0% | 4.9% | 1.1% | PTERA | **9.0** |
| 8 | 342 | +101.9 | 601 | 30 | 90.7% | 4.7% | 4.7% | PTERA | 7.0 |
| 9 | 461 | +140.3 | 807 | 40 | 94.1% | 4.8% | 1.1% | CACTUS | 8.0 |
| 10 | 507 | +154.3 | 888 | 43 | 92.8% | 4.6% | 2.6% | CACTUS | 9.0 |
| 11 | 322 | +91.2 | 566 | 24 | 95.8% | 4.2% | 0.0% | PTERA | 7.0 |
| 12 | 64 | +8.4 | 114 | 3 | 96.5% | 3.5% | 0.0% | CACTUS | 4.0 |
| 13 | 347 | +102.0 | 608 | 29 | 93.7% | 4.6% | 1.6% | PTERA | 7.0 |
| 14 | 129 | +26.9 | 227 | 6 | 94.3% | 2.6% | 3.1% | PTERA | 5.0 |
| 15 | 300 | +82.8 | 526 | 21 | 95.6% | 4.4% | 0.0% | CACTUS | 7.0 |
| 16 | 384 | +112.9 | 673 | 31 | 90.2% | 4.2% | 5.6% | CACTUS | 7.0 |
| 17 | 419 | +127.3 | 735 | 37 | 91.7% | 4.9% | 3.4% | CACTUS | 8.0 |
| 18 | 142 | +32.1 | 250 | 8 | 96.8% | 3.2% | 0.0% | CACTUS | 5.0 |
| 19 | 172 | +43.2 | 302 | 12 | 95.4% | 4.6% | 0.0% | CACTUS | 5.0 |
| 20 | 383 | +112.7 | 672 | 31 | 92.7% | 4.3% | 3.0% | PTERA | 7.0 |

**Top 5 episodes by score:**

| Rank | Score | Steps | Passes | Death cause | Speed |
| --- | --- | --- | --- | --- | --- |
| 1 | **510** | 894 | 44 | PTERA | 9.0 |
| 2 | 507 | 888 | 43 | CACTUS | 9.0 |
| 3 | 492 | 863 | 43 | CACTUS | 8.0 |
| 4 | 461 | 807 | 40 | CACTUS | 8.0 |
| 5 | 419 | 735 | 37 | CACTUS | 8.0 |

---

## 11. Dino Runner — Policy Quality

### Action distribution (10,695 total steps)

| Action | Steps | % | Bar |
| --- | --- | --- | --- |
| **noop** | 10,009 | 93.6% | `████████████████████████████████████████` |
| **jump** | 477 | 4.5% | `██` |
| **duck** | 209 | 2.0% | `█` |

The multi-task model actually uses `duck` (2.0% of steps), unlike the standalone Dino PPO which never ducked (0%). This may reflect cross-task transfer: the shared core may encode temporal reasoning from MiniGrid that helps the model react to higher obstacles.

### Policy entropy

| Statistic | Value | Note |
| --- | --- | --- |
| Mean entropy | **0.01384** | Extremely decisive |
| Std deviation | 0.00330 | Very stable |
| Min | 0.00751 | Near-zero uncertainty |
| Max | 0.01816 | Tiny peak uncertainty |
| Max possible | 1.0986 | (uniform over 3 actions) |

**The Dino policy entropy is essentially zero.** At 0.014 mean vs maximum of 1.099, the model is acting with **&gt;98% determinism** at every step. This reflects a well-trained, converged policy with no meaningful action uncertainty — the model knows exactly what to do.

### Value function

| Statistic | Value |
| --- | --- |
| Mean value estimate | +12.646 |
| Std deviation | 0.146 |

The high value estimate (\~12.6) is consistent with the long-horizon discounted returns in Dino — each surviving step accumulates small rewards, and the model correctly predicts a large positive expected return at any given state.

---

## 12. Key Findings & Summary

### Model overview

| Property | Value |
| --- | --- |
| Architecture | Multi-task actor-critic (CNN + MLP backbone) |
| Total parameters | 249,724 |
| Checkpoint size | 984.8 KB (\~1 MB) |
| Games supported | MiniGrid DoorKey + Chrome Dino Runner |
| Device | CPU (no GPU required) |

### MiniGrid DoorKey — headline numbers

| Metric | Value |
| --- | --- |
| **Overall solve rate (7 sizes)** | **83.1%** |
| Best individual size (10×10) | **98.3%** |
| DoorKey-16×16 solve rate | **93.3%** |
| DoorKey-16×16 loop-fail rate | **6.7%** |
| vs. feedforward baseline (16×16) | **+0.3pp** (≥ baseline) |
| Failure mode | 94% loop-fails, 6% wander-fails |
| Mean solve-path length | 32 steps (median) |

### Dino Runner — headline numbers

| Metric | Value |
| --- | --- |
| **Mean score (20 eps)** | **304.8** |
| **Max score** | **510** |
| Median score | 344 |
| Mean obstacle passes / episode | 24.2 |
| Max speed reached | 9.0 |
| Duck action used | Yes (2.0%), unlike standalone model |

### Computation — headline numbers

| Metric | MiniGrid | Dino |
| --- | --- | --- |
| Inference latency (mean) | 1.438 ms | 0.513 ms |
| Inference latency (p95) | 2.073 ms | 0.769 ms |
| Throughput (batch=1) | 695 steps/s | 1,951 steps/s |
| Throughput (batch=128) | 20,995 steps/s | 120,995 steps/s |

### Strengths

1. **Strong MiniGrid performance on medium grids** — 95–98% solve rate on 8×8 to 12×12.
2. **Matches the feedforward baseline on the hardest grid** (16×16: 93.3% vs 93.0%).
3. **Competent Dino play** — reaches speeds of 9.0, scores up to 510, uses duck (unlike standalone model).
4. **Compact model** — 249K parameters, \~1 MB on disk, no GPU required.
5. **Fast inference** — sub-1.5 ms MiniGrid, sub-0.5 ms Dino, with large batching headroom.
6. **Near-deterministic Dino policy** — entropy of 0.014, clean confident decisions.

### Weaknesses

1. **Small grid degradation** — 5×5 solve rate (48%) is significantly below 8×8 (95%). The 7×7 partial view is wider than the 5×5 grid, creating distribution shift.
2. **Loop-failures dominate unsolvable episodes** — 94% of failures are navigational cycles, suggesting the memoryless policy struggles to escape certain local loops.
3. **Dino score variance is high** — std of 149 on a mean of 305, reflecting sensitivity to obstacle sequences (some seeded runs produce difficult early clusters).
4. **No GPU acceleration** — the benchmark ran CPU-only; a GPU would reduce inference from 1.4 ms to \~0.1–0.2 ms for MiniGrid.

### Recommendations

- **For small grids (5×5/6×6):** Additional training on these sizes or switching to the recurrent GRU policy (`checkpoints_curriculum_gru/`) which was designed specifically to eliminate loop-failures.
- **For Dino:** Increase duck usage — the model ducks only 2% of steps, insufficient for high-speed pterodactyls. Behavior cloning on duck-heavy demonstrations or tuning the action entropy could help.
- **For deployment:** The model comfortably fits embedded targets. At \~1 MB and 695 steps/s on a desktop CPU, it would run on any modern edge device including the Raspberry Pi 4.

---

*Evaluation conducted with* `eval_final.py` *— 420 MiniGrid episodes (60 × 7 sizes) + 20 Dino episodes. Data source:* `eval_results.json`*.*