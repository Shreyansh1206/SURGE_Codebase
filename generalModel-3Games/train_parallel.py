
from __future__ import annotations

import sys

DEFAULT_ARGS = [
    "--parallel",
    "--minigrid-curriculum",
    "--minigrid-stages", "16",
    "--n-minigrid-envs", "8",
    "--n-dino-envs", "4",
    "--n-carracing-envs", "4",
    "--rollout", "128",
    "--dino-rollout", "1024",
    "--carracing-rollout", "1024",
    "--updates", "2000",
    "--save-dir", "checkpoints_3games",
    "--save-every", "25",
    "--epochs", "4",
    "--batch-size", "128",
    "--lr-schedule", "linear",
    "--normalize-carracing-reward",
    "--dino-bc-demos", "demos/dino_demos.npz",
    "--dino-bc-coef", "0.5",
]


def main():
    sys.argv = [sys.argv[0]] + DEFAULT_ARGS + sys.argv[1:]

    print("=" * 72)
    print("  generalModel-3Games — LONG TRAINING RUN (headless)")
    print("  MiniGrid:  DoorKey-16x16 (SyncVectorEnv x8, rollout=128)")
    print("  Dino:      4 workers, rollout=1024 (BC anchor coef=0.5)")
    print("  CarRacing: 4 envs, rollout=1024, 9-action, skip=2, aux, offtrack-trunc, reward-norm ON")
    print("  LR:        linear decay | Updates: 2000 | Batch: 128")
    print("=" * 72)

    from train import main as train_main

    train_main()


if __name__ == "__main__":
    main()
