"""
Scripted heuristic expert for the Dino game — a teacher for behavior cloning.

It reads the raw engine state (nearest obstacle position/type, speed) rather than
the normalised 48-dim policy obs, so its decisions are clean and tunable:

  - cactus (ground obstacle): JUMP when it enters a speed-scaled approach window.
  - ptera (bird): DUCK under high birds, JUMP over low/mid birds.
  - otherwise: RUN.

The jump window widens with speed (jump earlier when the game is faster) and is
kept wide enough to survive the 4-game-frames-per-step coarseness of training.

Action map (from engine._apply_action): 0 = run, 1 = jump (ground only), 2 = duck.
"""

from __future__ import annotations

from dataclasses import dataclass

import dino_env  # noqa: F401 — ensures Dino_runGame is on sys.path
from engine import DINO_X, HEIGHT  # noqa: E402

RUN, JUMP, DUCK = 0, 1, 2
_NO_OBSTACLE_X = 9999.0


@dataclass
class ExpertConfig:
    jump_gap_min: float = -10.0     # allow firing slightly late (obstacle at the dino)
    jump_gap_base: float = 60.0     # base front edge of the jump window (pixels)
    jump_gap_speed_k: float = 3.5   # window extends by this × speed (jump earlier when fast)
    bird_duck_centery: float = 100.0  # birds with centre above this (smaller y) → duck
    duck_gap_base: float = 80.0     # ducking starts a bit earlier and is held longer
    duck_gap_speed_k: float = 3.5


DEFAULT_CONFIG = ExpertConfig()


def expert_action(state: dict, cfg: ExpertConfig = DEFAULT_CONFIG) -> int:
    """Return the heuristic action (0/1/2) for the current engine state."""
    x, y, w, h = state["o1"]
    if x >= _NO_OBSTACLE_X:
        return RUN

    gap = x - DINO_X
    speed = float(state["speed"])

    # Already airborne — can't jump again and nothing useful to do mid-air.
    if state.get("jumping"):
        return RUN

    if state.get("o1type") == "PTERA":
        centery = y + h / 2.0
        if centery <= cfg.bird_duck_centery:  # high bird → duck under it
            duck_hi = cfg.duck_gap_base + cfg.duck_gap_speed_k * speed
            if cfg.jump_gap_min <= gap <= duck_hi:
                return DUCK
            return RUN
        # low / mid bird → jump over it (fall through to jump logic)

    jump_hi = cfg.jump_gap_base + cfg.jump_gap_speed_k * speed
    if cfg.jump_gap_min <= gap <= jump_hi:
        return JUMP
    return RUN
