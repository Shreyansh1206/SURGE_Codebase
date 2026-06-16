
from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--render",
    "--minigrid-env-id",
    "MiniGrid-DoorKey-16x16-v0",
    "--n-minigrid-envs",
    "1",
    "--n-dino-envs",
    "1",
    "--rollout",
    "128",
    "--dino-rollout",
    "512",
    "--render-delay",
    "0.015",
    "--save-dir",
    "checkpoints_doorkey_16x16",
    "--save-every",
    "10",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-v1 — VISIBLE TRAINING")
    print("  MiniGrid: Farama minigrid library (gymnasium.make)")
    print("  Dino:     local pygame engine (Dino_runGame/engine.py)")
    print("  Press Ctrl+C to stop; latest checkpoint is saved automatically.")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
