"""
Fast headless parallel training — multiple MiniGrid + Dino instances, no windows.

Usage:
    python train_parallel.py
    python train_parallel.py --updates 500
    python train_parallel.py --n-dino-envs 8 --n-minigrid-envs 16

Equivalent to:
    python train.py --parallel --n-minigrid-envs 8 --n-dino-envs 4 ...
"""

from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--parallel",
    "--minigrid-env-id",
    "MiniGrid-DoorKey-16x16-v0",
    "--n-minigrid-envs",
    "8",
    "--n-dino-envs",
    "4",
    "--rollout",
    "128",
    "--dino-rollout",
    "512",
    "--save-dir",
    "checkpoints_doorkey_16x16",
    "--save-every",
    "25",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-v1 — PARALLEL TRAINING (headless)")
    print("  MiniGrid: gymnasium.vector.SyncVectorEnv")
    print("  Dino:     multiprocess workers (SDL_VIDEODRIVER=dummy)")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
