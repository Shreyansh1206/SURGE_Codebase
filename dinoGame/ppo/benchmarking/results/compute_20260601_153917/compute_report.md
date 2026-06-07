# PPO Dino — Compute Profile

**Generated:** 20260601_153917
**Checkpoint:** `checkpoints_scratch_duck/best_duck.pt`
**Device:** cpu
**Timed runs:** 20 (after 10 warmup)

## Summary

| Metric | Value |
|--------|------:|
| Parameters | 23,300 |
| Checkpoint size | 0.093 MB |
| Weight memory (FP32) | 0.089 MB |
| MACs / deploy (analytical) | 23,570 |
| FLOPs / deploy (2× MAC) | 47,140 |
| Latency forward-only (median) | 0.0435 ms |
| Latency argmax deploy (median) | 0.0603 ms |
| Latency argmax deploy (mean) | 0.0874 ms |
| Latency argmax deploy 95% CI | [0.0634, 0.1114] ms |
| Throughput (argmax deploy) mean | 13979.9 inf/s |
| Game decision rate | 15.0 Hz |
| Neural MACs / game second | 353,550 |
| Neural MACs / episode (median steps) | 38,183,400 |
| GFLOPs / second @ 15 Hz | 0.000707 |

## Inference latency (ms) — deterministic deploy path

| Stat | Forward only | Argmax deploy |
|------|-------------:|--------------:|
| mean | 0.0449 | 0.0874 |
| std | 0.0041 | 0.0548 |
| median | 0.0435 | 0.0603 |
| min | 0.0429 | 0.0572 |
| max | 0.0612 | 0.2661 |
| p95 | 0.0444 | 0.0879 |
| ci95_lo | 0.0431 | 0.0634 |
| ci95_hi | 0.0467 | 0.1114 |

## Notes

- **Argmax deploy** matches `benchmark.py` / `infer.py` (forward + softmax + argmax).
- Full **game loop** time is dominated by Selenium/Chrome (~15 env steps/s); neural net cost is a tiny fraction of wall time.
- MAC counts are analytical (Linear layers); activations use a small fixed estimate.
