
"""
Watch the Dino game while fine-tuning from checkpoints_final/final.pt.

Ctrl+C saves latest.pt automatically; rerun to resume.

  python train_dino_visible.py
  python train_dino_visible.py --updates 100 --render-delay 0.02
"""

from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--render-dino",
    "--n-dino-envs",
    "1",
    "--dino-rollout",
    "256",
    "--render-delay",
    "0.012",
    "--save-dir",
    "checkpoints_dino_finetune",
    "--save-every",
    "5",
    "--eval-every",
    "10",
    "--eval-eps",
    "5",
    "--updates",
    "200",
    "--lr",
    "2e-4",
]


def main() -> None:
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-v1 — DINO FINE-TUNE (VISIBLE)")
    print("  1 on-screen Dino game (parallel disabled for visual check).")
    print("  For fast training use: python train_dino_finetune.py  (4 parallel games)")
    print("  Ctrl+C saves latest.pt; rerun to resume.")
    print("=" * 72)

    from train_dino_finetune import main as finetune_main

    finetune_main()


if __name__ == "__main__":
    main()
