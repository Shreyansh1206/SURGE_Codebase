# Scratch retrain (duck-aware BC + bird curriculum)

Fresh training pipeline. Does **not** overwrite `checkpoints/best.pt` or `checkpoints_duck_*` finetune runs.

## 0. Verify the teacher first (important)

```bash
conda run --no-capture-output -n dinoGame python run_scripted.py --headless --episodes 20
```

You want **mean score ~550+** on normal starts before burning time on BC. With birds:

```bash
conda run --no-capture-output -n dinoGame python run_scripted.py --headless --episodes 15 --curriculum-prob 0.5
```

## 1. BC warmup (duck-aware teacher)

```bash
conda run --no-capture-output -n dinoGame python bc_warmup.py \
  --headless --episodes 40 --curriculum-prob 0.5 \
  --save checkpoints_scratch_duck/bc_init.pt
```

Teacher rules (from `scripted_teacher.py`):
- **Mid birds** (y ≈ 75): duck while in range
- **High birds** (y ≈ 50): no-op (run under)
- **Cacti / low birds**: jump when close

Half of BC episodes can start at speed 8.5–9.5 / score ~450 so the dataset includes birds.

## 2. PPO from BC init

```bash
conda run --no-capture-output -n dinoGame python train.py \
  --save-dir checkpoints_scratch_duck \
  --resume checkpoints_scratch_duck/bc_init.pt \
  --headless --n-envs 4 --rollout 256 --updates 300 \
  --curriculum-prob 0.35 --entropy 0.01
```

## 3. Benchmark

```bash
conda run --no-capture-output -n dinoGame python benchmarking/benchmark.py \
  --ckpt checkpoints_scratch_duck/best.pt --episodes 50 --headless
```

Compare against `checkpoints/best.pt` (original) and `checkpoints_duck_v13/best_duck.pt` (head-only finetune).
