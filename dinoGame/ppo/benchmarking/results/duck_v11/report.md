# PPO Chrome Dino Agent — Benchmark Report

**Generated:** 20260531_214248  
**Checkpoint:** `checkpoints_duck_v11/best_duck.pt`  
**Deterministic episodes:** 25  
**Model parameters:** 23,300  
**scipy available:** No (CI via normal approximation)  

---

## Abstract

We evaluate a Proximal Policy Optimisation (PPO) agent trained to play the Chrome Dinosaur (T-Rex Runner) browser game. The agent maps a 48-dimensional observation (12 state features × 4-frame stack) to one of three discrete actions — no-op, jump, duck — at a nominal decision rate of 15 Hz. Training uses four parallel browser instances, a 256-step rollout buffer per environment, and standard PPO hyperparameters (γ = 0.99, λ_GAE = 0.95, ε_clip = 0.2). The deterministic policy achieves a mean score of **62.48 ± 13.60** (median 59.0, max 95) across 25 evaluation episodes. This report presents full descriptive statistics, confidence intervals, survival-rate profiles, action-entropy analysis, and critic value statistics.

---

## 1. Experimental Setup

### 1.1 Environment

| Property | Value |
|----------|-------|
| Game | Chrome T-Rex Runner (local `file://` HTML) |
| Observation space | ℝ⁴⁸ (12 features × 4 frame stack) |
| Action space | Discrete(3): {no-op, jump, duck} |
| Decision frequency | ~15 Hz (4 game frames @ 60 Hz per step) |
| State features | dino_y, jumping, ducking, speed, o1_dx, o1_y, o1_w, o1_h, o2_dx, o2_y, o2_w, o2_h |
| Reward: score delta | +0.01 × Δscore |
| Reward: obstacle pass | +1.0 |
| Reward: jump cost | −0.01 per jump action |
| Reward: death | −10.0 (terminal) |
| Danger range normalisation | 200 px → [0, 1.5] |

### 1.2 Policy Architecture

| Component | Specification |
|-----------|--------------|
| Type | MLP Actor-Critic (shared trunk) |
| Hidden layers | 2 × 128 units |
| Activation | Tanh |
| Policy head | Linear(128 → 3) + Softmax |
| Value head | Linear(128 → 1) |
| Total parameters | 23,300 |

### 1.3 Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Algorithm | PPO (Clipped Surrogate Objective) |
| Parallel environments | 4 |
| Rollout length | 256 steps / env |
| Samples per update | 1,024 |
| Learning rate | 3 × 10⁻⁴ (Adam) |
| Discount factor γ | 0.99 |
| GAE λ | 0.95 |
| Clip ε | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Max gradient norm | 0.5 |
| SGD epochs per update | 4 |
| Mini-batch size | 128 |
| KL early-stop threshold | 0.015 |

### 1.4 Evaluation Protocol

| Setting | Value |
|---------|-------|
| Deterministic episodes | 25 |
| Deterministic inference | Argmax over policy logits |
| Stochastic inference | Categorical sample from softmax |
| Step pause | 0 s (max speed) |
| Score metric | `Runner.distanceRan × COEFFICIENT` (in-game) |

---

## 2. Performance Results

### 2.1 Score — Descriptive Statistics

| Metric | Deterministic |
| :------ | ------------: |
| N episodes | 25 |
| Mean | 62.48 |
| Std. deviation | 13.60 |
| Std. error of mean | 2.721 |
| 95% CI (lower) | 57.15 |
| 95% CI (upper) | 67.81 |
| Median | 59.0 |
| IQR (P25 – P75) | 19.0 |
| Min | 42 |
| P1 | 42.2 |
| P5 | 44.2 |
| P10 | 49.0 |
| P25 | 51.0 |
| P75 | 70.0 |
| P90 | 81.0 |
| P95 | 87.0 |
| P99 | 93.3 |
| Max | 95 |
| Coeff. of variation | 0.2177 |
| Skewness | N/A |
| Excess kurtosis | N/A |

### 2.2 Threshold Survival Rates

> Percentage of episodes in which the agent reached or exceeded each score threshold.

| Score threshold | Deterministic % |
| :--------------- | ---------------: |
| ≥ 100 | 0.0% |
| ≥ 200 | 0.0% |
| ≥ 300 | 0.0% |
| ≥ 400 | 0.0% |
| ≥ 500 | 0.0% |
| ≥ 750 | 0.0% |
| ≥ 1,000 | 0.0% |

### 2.3 Episode Length and Shaped Return

| Metric | Deterministic |
| :------ | ------------: |
| Mean steps | 78.2 |
| Std steps | 16.8 |
| Median steps | 75.0 |
| Max steps | 119 |
| Mean shaped return | -9.776 |
| Std shaped return | 0.035 |
| Max shaped return | -9.670 |
| Mean wall-time (s) | 6.40 |
| Max wall-time (s) | 9.75 |
| Mean final speed | 6.427 |
| Max final speed | 6.899 |

---

## 3. Behavioural Analysis

### 3.1 Global Action Distribution

| Action | Det. count | Det. % |
| :------ | ----------: | ------: |
| noop | 1,014 | 51.84% |
| jump | 942 | 48.16% |
| duck | 0 | 0.00% |

**Deterministic — global action entropy:** 0.6925 nats  
**Deterministic — mean per-episode action entropy:** 0.6772 nats  
*(Maximum possible entropy for 3 actions: 1.0986 nats)*

### 3.2 Critic Value Estimates (Deterministic Mode)

| Statistic | Value |
|:----------|------:|
| Total step estimates | 1956 |
| Mean V(s) | 2.25 |
| Std V(s) | 2.79 |
| Min V(s) | -3.33 |
| P5 V(s) | -2.61 |
| Median V(s) | 1.56 |
| P95 V(s) | 6.73 |
| Max V(s) | 8.38 |

### 3.3 Death-Cause Breakdown (Deterministic Mode)

> Obstacle type the agent collided with at episode end. `PTERODACTYL` deaths are the ones a working duck is meant to prevent.

| Death cause | Episodes | % |
|:------------|---------:|--:|
| CACTUS_SMALL | 19 | 76.0% |
| CACTUS_LARGE | 6 | 24.0% |

**Pterodactyl-death rate:** 0.0%  

---

## 4. Training History

*Based on `checkpoints/train_log.jsonl` — 50 PPO updates logged.*

### 4.1 Training Phase Summary

| Phase | Updates | Peak score | Mean score | Mean ep. len | Mean SPS |
|:------|--------:|-----------:|-----------:|-------------:|---------:|
| Early | 1–16 | 268 | 100.9 | 69.2 | 40.9 |
| Mid | 17–33 | 430 | 178.4 | 93.5 | 42.2 |
| Late | 34–50 | 534 | 181.8 | 56.8 | 42.5 |

### 4.2 Overall Training Statistics

| Metric | Value |
|:-------|------:|
| PPO updates logged | 50 |
| Peak score during training | 534 |
| Total env samples | 51,200 |
| Total wall-clock training time | 20.6 min |
| Mean samples per second | 42.0 |

---

## 5. Discussion

### 5.1 Policy Competence

The deterministic policy achieves a mean score of **62.48 ± 13.60** (median 59.0, max 95) over 25 episodes. The coefficient of variation (CV = 0.218) reflects environmental stochasticity — obstacle types, gaps, and the speed ramp-up are non-deterministic from the agent's perspective, so score variance is partly irreducible. Survival rates of 0.0% at score ≥200 and 0.0% at score ≥300 suggest the policy has internalised basic obstacle-avoidance behaviour and can maintain game-play through the early speed-increase phases.

### 5.2 Action Selection Behaviour

Under deterministic inference the agent selects **jump** in 48.2% of steps and **duck** in 0.0% of steps. Jump is the primary survival action and its frequency reflects how densely obstacles appear relative to the nominal decision horizon. Low duck frequency is expected: pterodactyls only appear at higher speeds and ducking is rarely the correct action in the early-to-mid score range captured by most evaluation episodes. The global action entropy of 0.6925 nats (vs. maximum 1.0986 nats) indicates a concentrated, relatively deterministic behavioural profile.

### 5.3 Critic Value Estimates

The critic outputs values in the range [-3.33, 8.38] with a mean of 2.252. The predominantly negative value range is consistent with the reward structure: death produces a −10 penalty that dominates shaped rewards, so the critic learns to predict a slightly negative long-run return at most states, scaling toward zero as the agent survives longer episodes.

### 5.4 Limitations and Future Work

- **Environment stochasticity.** Obstacle generation is non-deterministic; score variance is partly irreducible even for a perfect policy.
- **Single seed / checkpoint.** A robust benchmark should average over multiple independently-trained seeds to disentangle policy quality from random luck.
- **Selenium latency.** Browser IPC overhead (~15–30 ms per Selenium call) inflates wall-clock times and can introduce timing jitter; reported wall times should not be used to infer in-game timing precision.
- **No speed-curriculum evaluation.** The agent is always evaluated from game start. A more thorough benchmark would measure survival at various speed injection points.
- **No comparison baseline.** Adding a scripted rule-based agent and a random policy as baselines would contextualise these scores.

---

*Report generated automatically by `benchmarking/benchmark.py` — Chrome Dino PPO project.*
