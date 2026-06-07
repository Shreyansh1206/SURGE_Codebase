# PPO Dino — Compute Profile

**Generated:** 20260601_154732
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
| Latency forward-only (median) | 0.0514 ms |
| Latency argmax deploy (median) | 0.0871 ms |
| Latency argmax deploy (mean) | 0.0944 ms |
| Latency argmax deploy 95% CI | [0.0841, 0.1048] ms |
| Throughput (argmax deploy) mean | 11374.2 inf/s |
| Game decision rate | 15.0 Hz |
| Neural MACs / game second | 353,550 |
| Neural MACs / episode (median steps) | 38,183,400 |
| GFLOPs / second @ 15 Hz | 0.000707 |

| Process CPU % per-run (median) | 0.00% |
| Process CPU % per-run (mean) | 0.00% |
| Process CPU % (25-run block) | 0.00% |
| Process RAM RSS (median) | 255.81 MB |
| Process RAM RSS (mean) | 255.81 MB |
| System RAM used (snapshot) | 79.0% |
| System RAM total | 16055 MB |
| CPU % (sustained burst, 1 core) | 85.78% |
| Inferences/s (sustained burst) | 15250 |
| RAM RSS after burst | 256.67 MB |

## Inference latency (ms) — deterministic deploy path

| Stat | Forward only | Argmax deploy |
|------|-------------:|--------------:|
| mean | 0.0543 | 0.0944 |
| std | 0.0108 | 0.0264 |
| median | 0.0514 | 0.0871 |
| min | 0.0497 | 0.0677 |
| max | 0.1044 | 0.1464 |
| p95 | 0.0528 | 0.1107 |
| ci95_lo | 0.0500 | 0.0841 |
| ci95_hi | 0.0585 | 0.1048 |

## Resource usage (argmax deploy, per timed run)

| Stat | CPU % (process) | RAM RSS MB (process) |
|------|----------------:|---------------------:|
| mean | 0.00 | 255.81 |
| std | 0.00 | 0.00 |
| median | 0.00 | 255.81 |
| min | 0.00 | 255.81 |
| max | 0.00 | 255.82 |
| p95 | 0.00 | 255.81 |
| ci95_lo | 0.00 | 255.81 |
| ci95_hi | 0.00 | 255.81 |

## Notes

- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).
- Full **game loop** time is dominated by Selenium/Chrome (~15 env steps/s); neural net cost is a tiny fraction of wall time.
- MAC counts are analytical (Linear layers); activations use a small fixed estimate.
- **CPU % per-run** is process-scoped (`psutil`) between iterations (often 0% when each forward is under 1 ms). **block_cpu_percent** covers all timed runs together.
- **RAM RSS** is resident set size of the Python process after each timed run.
