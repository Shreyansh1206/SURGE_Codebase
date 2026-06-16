
from __future__ import annotations

from dataclasses import dataclass

import dino_env  # noqa
from engine import DINO_X, HEIGHT  # noqa

RUN, JUMP, DUCK = 0, 1, 2
_NO_OBSTACLE_X = 9999.0


@dataclass
class ExpertConfig:
    jump_gap_min: float = -10.0
    jump_gap_base: float = 60.0
    jump_gap_speed_k: float = 3.5
    bird_duck_centery: float = 100.0
    duck_gap_base: float = 80.0
    duck_gap_speed_k: float = 3.5


DEFAULT_CONFIG = ExpertConfig()


def expert_action(state: dict, cfg: ExpertConfig = DEFAULT_CONFIG) -> int:
    x, y, w, h = state["o1"]
    if x >= _NO_OBSTACLE_X:
        return RUN

    gap = x - DINO_X
    speed = float(state["speed"])

    if state.get("jumping"):
        return RUN

    if state.get("o1type") == "PTERA":
        centery = y + h / 2.0
        if centery <= cfg.bird_duck_centery:
            duck_hi = cfg.duck_gap_base + cfg.duck_gap_speed_k * speed
            if cfg.jump_gap_min <= gap <= duck_hi:
                return DUCK
            return RUN

    jump_hi = cfg.jump_gap_base + cfg.jump_gap_speed_k * speed
    if cfg.jump_gap_min <= gap <= jump_hi:
        return JUMP
    return RUN
