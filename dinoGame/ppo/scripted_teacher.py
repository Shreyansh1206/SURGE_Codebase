"""
Strong hand-coded Chrome Dino policy for BC warmup and sanity checks.

Design goals:
  - Score competitively with the old ~587 PPO baseline (not cap at ~600).
  - Teach duck on *mid* pterodactyls, no-op under *high* birds, jump cacti/low birds.
  - Speed-adaptive jump timing + two-obstacle lookahead + sustained duck (hysteresis).

Use `run_scripted.py` to benchmark before BC. Pass `info` from env.step() when available
(obstacle types + raw positions); obs-only fallback still works but is weaker on birds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

FEATURES_PER_FRAME = 12
NOOP, JUMP, DUCK = 0, 1, 2

CANVAS_W = 600.0
CANVAS_H = 150.0
DINO_X = 44.0
DANGER_RANGE = 200.0
SPEED_MAX = 13.0

# Pterodactyl yPos bands from t-rex-runner (150px canvas height)
BIRD_Y_HIGH = 55.0   # run under — do not duck or jump
BIRD_Y_MID  = 80.0   # duck when in range (75 is the classic mid spawn)
BIRD_Y_LOW  = 100.0  # treat like jump obstacle


@dataclass
class TeacherMemory:
    """Per-episode state for sustained ducking."""
    duck_hold: bool = False


def _frame(obs):
    return obs[-FEATURES_PER_FRAME:]


def _norm_to_px(dx_norm: float) -> float:
    if dx_norm >= 1.45:
        return 9999.0
    return DINO_X + max(0.0, dx_norm) * DANGER_RANGE


def _classify(otype: str, y_raw: float) -> str:
    """ground | bird_high | bird_mid | bird_low | empty"""
    if y_raw < 1.0 or (otype == "" and y_raw < 1.0):
        return "empty"
    if otype == "PTERODACTYL":
        if y_raw <= BIRD_Y_HIGH:
            return "bird_high"
        if y_raw <= BIRD_Y_MID:
            return "bird_mid"
        return "bird_low"
    # Cacti and other ground obstacles
    return "ground"


def _jump_lead_px(speed: float) -> float:
    """How far ahead (px from dino feet) to trigger a jump — earlier when faster."""
    s = max(6.0, min(SPEED_MAX, speed))
    return 78.0 + 7.5 * (s - 6.0)


def _duck_enter_px(speed: float) -> float:
    s = max(6.0, min(SPEED_MAX, speed))
    return 150.0 + 4.0 * (s - 6.0)


def _duck_exit_px() -> float:
    return DINO_X + DANGER_RANGE + 25.0


def _parse(info: Optional[Dict[str, Any]], obs) -> Dict[str, Any]:
    f = _frame(obs)
    speed = float(f[3]) * SPEED_MAX
    out = {
        "speed": speed,
        "jumping": f[1] > 0.5,
        "ducking": f[2] > 0.5,
        "o1_dx": f[4],
        "o1_dy": f[5],
        "o2_dx": f[8],
        "o2_dy": f[9],
    }
    if info:
        out["o1_x"] = float(info.get("o1_x", _norm_to_px(f[4])))
        out["o1_y"] = float(info.get("o1_y", f[5] * CANVAS_H))
        out["o2_x"] = float(info.get("o2_x", _norm_to_px(f[8])))
        out["o2_y"] = float(info.get("o2_y", f[9] * CANVAS_H))
        out["o1_type"] = str(info.get("o1_type", "") or "")
        out["o2_type"] = str(info.get("o2_type", "") or "")
    else:
        out["o1_x"] = _norm_to_px(f[4])
        out["o1_y"] = f[5] * CANVAS_H
        out["o2_x"] = _norm_to_px(f[8])
        out["o2_y"] = f[9] * CANVAS_H
        out["o1_type"] = ""
        out["o2_type"] = ""
    out["o1_kind"] = _classify(out["o1_type"], out["o1_y"])
    out["o2_kind"] = _classify(out["o2_type"], out["o2_y"])
    return out


def scripted_action(
    obs,
    info: Optional[Dict[str, Any]] = None,
    memory: Optional[TeacherMemory] = None,
) -> int:
    """Choose {noop, jump, duck}. Pass `memory` for sustained duck across steps."""
    mem = memory if memory is not None else TeacherMemory()
    s = _parse(info, obs)
    speed = s["speed"]
    o1x, o1y = s["o1_x"], s["o1_y"]
    o2x, o2y = s["o2_x"], s["o2_y"]
    k1, k2 = s["o1_kind"], s["o2_kind"]

    # --- Sustained duck (mid bird) -------------------------------------------
    if mem.duck_hold:
        if o1x > _duck_exit_px() or k1 in ("empty", "bird_high", "ground", "bird_low"):
            mem.duck_hold = False
        else:
            return DUCK

    # --- Nearest obstacle (o1) ------------------------------------------------
    if k1 != "empty":
        if k1 == "bird_high":
            # Run under — never jump into it
            return NOOP

        if k1 == "bird_mid":
            if o1x < _duck_enter_px(speed):
                mem.duck_hold = True
                return DUCK
            return NOOP

        if k1 in ("bird_low", "ground"):
            if not s["jumping"] and o1x < DINO_X + _jump_lead_px(speed):
                return JUMP
            return NOOP

    # --- Lookahead (o2) when o1 is clear or far ------------------------------
    if k2 != "empty" and o1x > DINO_X + 120.0:
        if k2 == "bird_mid" and o2x < _duck_enter_px(speed) + 40.0:
            mem.duck_hold = True
            return DUCK
        if k2 == "bird_high":
            return NOOP
        if k2 in ("bird_low", "ground") and not s["jumping"]:
            if o2x < DINO_X + _jump_lead_px(speed) + 30.0:
                return JUMP

    # --- Obs-only fallback (no type in info yet) ------------------------------
    if not info or not info.get("o1_type"):
        dy = s["o1_dy"]
        dx = s["o1_dx"]
        if dy < 1e-6 or dx >= 1.0:
            return NOOP
        if dy <= BIRD_Y_HIGH / CANVAS_H:
            return NOOP
        if dy <= BIRD_Y_MID / CANVAS_H:
            if dx < 0.82:
                mem.duck_hold = True
                return DUCK
            return NOOP
        if not s["jumping"] and dx < 0.28 + 0.025 * (speed / SPEED_MAX):
            return JUMP

    return NOOP


def new_memory() -> TeacherMemory:
    return TeacherMemory()
