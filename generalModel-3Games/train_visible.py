
from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--render",
    "--minigrid-env-id",
    "MiniGrid-DoorKey-5x5-v0",
    "--n-minigrid-envs",
    "1",
    "--n-dino-envs",
    "1",
    "--n-carracing-envs",
    "1",
    "--rollout",
    "64",
    "--dino-rollout",
    "128",
    "--carracing-rollout",
    "64",
    "--render-delay",
    "0.015",
    "--updates",
    "100",
    "--save-dir",
    "checkpoints_visible",
    "--save-every",
    "10",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-3Games — VISIBLE TRAINING (one window per game)")
    print("  MiniGrid / Dino / CarRacing — single env each, on-screen render")
    print("  Press Ctrl+C to stop; latest checkpoint is saved automatically.")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
