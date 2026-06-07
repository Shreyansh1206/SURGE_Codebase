"""
Launch multi-task training with both games visible on screen.

Defaults are tuned for watching the agent learn:
  - MiniGrid: official Farama `minigrid` library via gymnasium.make()
  - Dino: local pygame engine with a live window
  - 1 env per game (one window each, no overlapping pop-ups)
  - Small per-step delay so movement is easy to follow

Usage:
    python train_visible.py
    python train_visible.py --updates 200
    python train_visible.py --dino-only
    python train_visible.py --render-delay 0.03

Any flag supported by train.py can be appended to override these defaults.
"""

from __future__ import annotations

import sys


DEFAULT_ARGS = [
    "--render",              # show MiniGrid + Dino windows
    "--minigrid-env-id",
    "MiniGrid-DoorKey-16x16-v0",
    "--n-minigrid-envs",
    "1",                     # one MiniGrid window
    "--n-dino-envs",
    "1",                     # one Dino window
    "--rollout",
    "128",                   # minigrid steps per update
    "--dino-rollout",
    "512",                   # dino steps per update (×4 game frames each)
    "--render-delay",
    "0.015",                 # ~15 ms pause per step while rendering
    "--save-dir",
    "checkpoints_doorkey_16x16",
    "--save-every",
    "10",
]


def main():
    # Prepend defaults, then pass through any user-supplied CLI overrides.
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
