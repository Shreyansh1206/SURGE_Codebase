
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
    "--n-carracing-envs",
    "4",
    "--rollout",
    "128",
    "--dino-rollout",
    "512",
    "--carracing-rollout",
    "128",
    "--save-dir",
    "checkpoints_3games",
    "--save-every",
    "25",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-3Games — PARALLEL TRAINING (headless)")
    print("  MiniGrid:  gymnasium.vector.SyncVectorEnv")
    print("  Dino:      multiprocess workers (SDL_VIDEODRIVER=dummy)")
    print("  CarRacing: gymnasium.vector.SyncVectorEnv (Box2D)")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
