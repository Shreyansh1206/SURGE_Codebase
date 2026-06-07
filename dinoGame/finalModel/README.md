# Chrome Dino — Final Model Bundle

Self-contained copy of the **winning** training and evaluation pipeline (scratch BC → PPO → duck distillation). Intermediate failed approaches (`finetune_duck.py`, `checkpoints_duck_v*`, surgical head finetunes) are **not** included.

## Prerequisites

- Conda env **`dinoGame`** (PyTorch, Selenium, Chrome, chromedriver)
- Game HTML at **`dinoGame/t-rex-runner/index.html`** (sibling of this folder; `dino_env.py` resolves `../t-rex-runner/index.html`)

## Layout

| File | Role |
|------|------|
| `dino_env.py` | Selenium env, curriculum, obstacle/death `info` |
| `ppo_agent.py` | MLP actor-critic + PPO |
| `scripted_teacher.py` | Duck-aware BC / scripted teacher |
| `bc_warmup.py` | Behaviour cloning warm-start |
| `train.py` | PPO training |
| `duck_collect.py` | Bird-state collection for distillation |
| `make_duck_distill.py` | Hinge duck + logit-gap distillation |
| `run_scripted.py` | Verify scripted teacher (~1.8k mean) |
| `infer.py` | Play episodes with a checkpoint |
| `benchmark.py` | 30–50 ep evaluation + figures + `report.md` |
| `compute_profile.py` | Inference timing / MACs / CPU-RAM (optional `--paced`) |
| `run_pipeline.py` | Resumable BC → PPO → distill → benchmark |
| `checkpoints/best_duck.pt` | **Final model** |
| `checkpoints/scratch_best.pt` | Pre-duck PPO (~617 mean); distill init reference |

## Quick use (inference / benchmark only)

```powershell
cd dinoGame\finalModel
conda activate dinoGame

python infer.py --ckpt checkpoints/best_duck.pt --episodes 5 --headless
python benchmark.py --ckpt checkpoints/best_duck.pt --episodes 30 --headless
python compute_profile.py --ckpt checkpoints/best_duck.pt --runs 25 --paced
```

On Windows, prefer direct `python -u` (not buffered `conda run`) for long benchmarks.

## Full retrain (~8h)

```powershell
python run_pipeline.py
```

Or step by step:

```powershell
python bc_warmup.py --headless --curriculum-prob 0.5 --save checkpoints/bc_init.pt
python train.py --save-dir checkpoints --resume checkpoints/bc_init.pt --headless `
  --n-envs 4 --rollout 256 --updates 140 --curriculum-prob 0.35 --entropy 0.01 --lr 2e-4
python make_duck_distill.py --init checkpoints/best.pt --out checkpoints/best_duck.pt
python benchmark.py --ckpt checkpoints/best_duck.pt --episodes 30 --headless
```

Re-run distillation only (reuse collected states):

```powershell
python make_duck_distill.py --reuse-data --init checkpoints/scratch_best.pt
```

## Reference results

Bundled checkpoint was benchmarked in the original tree as `ppo/benchmarking/results/scratch_duck_distill_v1_30/` (30 eps, mean ~2653, duck ~7.8%). See `docs/reference_benchmark_report.md` for the archived report.

## What is excluded

- Original untouched baseline: `ppo/checkpoints/best.pt` (~587 mean, no duck)
- Failed duck finetune path: `finetune_duck.py`, `make_duck_model.py`, `checkpoints_duck_v*`
- Probe / overnight scratch logs under `ppo/`
