# PPO Dino — Compute Profile

**Generated:** 20260601_154520
**Checkpoint:** `checkpoints_scratch_duck/best_duck.pt`
**Device:** cpu
**Timed runs:** 25 (after 10 warmup)

## Summary

| Metric | Value |
|--------|------:|
| Parameters | 23,300 |
| Checkpoint size | 0.093 MB |
| Weight memory (FP32) | 0.089 MB |
| MACs / deploy (analytical) | 23,570 |
| FLOPs / deploy (2× MAC) | 47,140 |
| Latency forward-only (median) | 0.0610 ms |
| Latency argmax deploy (median) | 0.0816 ms |
| Latency argmax deploy (mean) | 0.0940 ms |
| Latency argmax deploy 95% CI | [0.0810, 0.1070] ms |
| Throughput (argmax deploy) mean | 11368.1 inf/s |
| Game decision rate | 15.0 Hz |
| Neural MACs / game second | 353,550 |
| Neural MACs / episode (median steps) | 38,183,400 |
| GFLOPs / second @ 15 Hz | 0.000707 |

| Process CPU % per-run (median) | 0.00% |
| Process CPU % per-run (mean) | 0.00% |
| Process CPU % (25-run block) | 0.00% |
| Process RAM RSS (median) | 255.40 MB |
| Process RAM RSS (mean) | 255.41 MB |
| System RAM used (snapshot) | 81.0% |
| System RAM total | 16055 MB |
| CPU % (sustained burst, 1 core) | 94.66% |
| Inferences/s (sustained burst) | 10097 |
| RAM RSS after burst | 256.28 MB |

## Inference latency (ms) — deterministic deploy path

| Stat | Forward only | Argmax deploy |
|------|-------------:|--------------:|
| mean | 0.0666 | 0.0940 |
| std | 0.0127 | 0.0332 |
| median | 0.0610 | 0.0816 |
| min | 0.0588 | 0.0793 |
| max | 0.1149 | 0.2077 |
| p95 | 0.0653 | 0.0843 |
| ci95_lo | 0.0616 | 0.0810 |
| ci95_hi | 0.0716 | 0.1070 |

## Resource usage (argmax deploy, per timed run)

| Stat | CPU % (process) | RAM RSS MB (process) |
|------|----------------:|---------------------:|
| mean | 0.00 | 255.41 |
| std | 0.00 | 0.00 |
| median | 0.00 | 255.40 |
| min | 0.00 | 255.40 |
| max | 0.00 | 255.41 |
| p95 | 0.00 | 255.41 |
| ci95_lo | 0.00 | 255.40 |
| ci95_hi | 0.00 | 255.41 |

## Notes

- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).
- Full **game loop** time is dominated by Selenium/Chrome (~15 env steps/s); neural net cost is a tiny fraction of wall time.
- MAC counts are analytical (Linear layers); activations use a small fixed estimate.
- **CPU % per-run** is process-scoped (`psutil`) between iterations (often 0% when each forward is under 1 ms). **block_cpu_percent** covers all timed runs together.
- **RAM RSS** is resident set size of the Python process after each timed run.
