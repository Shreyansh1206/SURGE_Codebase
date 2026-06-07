# PPO Dino — Compute Profile

**Generated:** 20260601_155633
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
| **Paced simulation** | **15.0 Hz** |
| Inference latency (median) | 0.3187 ms |
| Inference latency (mean) | 0.5949 ms |
| Slot period (median) | 79.15 ms |
| Model duty cycle | 0.77% |
| Session CPU % | 0.81% |
| Achieved decision rate | 12.90 Hz |
| CPU % per tick (mean) | 1.01% |
| RAM RSS (mean) | 256.14 MB |
| Game decision rate | 15.0 Hz |
| Neural MACs / game second | 353,550 |
| Neural MACs / episode (median steps) | 38,183,400 |
| GFLOPs / second @ 15 Hz | 0.000707 |

| System RAM used (snapshot) | 81.0% |
| System RAM total | 16055 MB |


## Paced game simulation — resource usage per tick

| Stat | Inference ms | Slot ms | CPU % | RAM MB |
|------|-------------:|--------:|------:|-------:|
| mean | 0.5949 | 77.38 | 1.01 | 256.14 |
| median | 0.3187 | 79.15 | 0.00 | 256.14 |
| std | 0.8019 | 4.11 | 5.04 | 0.01 |
| p95 | 0.3381 | 79.64 | 0.00 | 256.14 |

## Notes

- **Paced mode** sleeps between decisions so wall time matches real game rate (15 Hz).
- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).
- Full **browser game loop** is still dominated by Selenium/Chrome; paced mode profiles **policy inference only** at game cadence.
- MAC counts are analytical (Linear layers); activations use a small fixed estimate.
- **CPU % per-run** is process-scoped (`psutil`) between iterations (often 0% when each forward is under 1 ms). **block_cpu_percent** covers all timed runs together.
- **RAM RSS** is resident set size of the Python process after each timed run.
