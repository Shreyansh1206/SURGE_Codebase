
from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--render",
    "--visible-rotate",
    "--episode-end-pause",
    "0.5",
    "--minigrid-env-id",
    "MiniGrid-DoorKey-5x5-v0",
    "--n-minigrid-envs",
    "1",
    "--n-dino-envs",
    "1",
    "--n-carracing-envs",
    "1",
    "--rollout",
    "128",
    "--dino-rollout",
    "256",
    "--carracing-rollout",
    "128",
    "--render-delay",
    "0.01",
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
    print("  generalModel-3Games — VISIBLE TRAINING")
    print("  One game per update (round-robin), auto-reset with 0.5s end pause")
    print("  Press Ctrl+C to stop; latest checkpoint is saved automatically.")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
