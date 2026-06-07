# PPO Dino — Compute Profile

**Generated:** 20260601_155513
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
| Inference latency (median) | 0.3807 ms |
| Inference latency (mean) | 0.5271 ms |
| Slot period (median) | 79.52 ms |
| Model duty cycle | 0.68% |
| Session CPU % | 0.80% |
| Achieved decision rate | 12.85 Hz |
| CPU % per tick (mean) | 0.67% |
| RAM RSS (mean) | 256.67 MB |
| Game decision rate | 15.0 Hz |
| Neural MACs / game second | 353,550 |
| Neural MACs / episode (median steps) | 38,183,400 |
| GFLOPs / second @ 15 Hz | 0.000707 |

| System RAM used (snapshot) | 83.7% |
| System RAM total | 16055 MB |


## Paced game simulation — resource usage per tick

| Stat | Inference ms | Slot ms | CPU % | RAM MB |
|------|-------------:|--------:|------:|-------:|
| mean | 0.5271 | 77.47 | 0.67 | 256.67 |
| median | 0.3807 | 79.52 | 0.00 | 256.67 |
| std | 0.6560 | 3.48 | 3.36 | 0.01 |
| p95 | 0.4515 | 79.80 | 0.00 | 256.67 |

## Notes

- **Paced mode** sleeps between decisions so wall time matches real game rate (15 Hz).
- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).
- Full **browser game loop** is still dominated by Selenium/Chrome; paced mode profiles **policy inference only** at game cadence.
- MAC counts are analytical (Linear layers); activations use a small fixed estimate.
- **CPU % per-run** is process-scoped (`psutil`) between iterations (often 0% when each forward is under 1 ms). **block_cpu_percent** covers all timed runs together.
- **RAM RSS** is resident set size of the Python process after each timed run.
