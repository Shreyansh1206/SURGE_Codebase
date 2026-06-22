
from __future__ import annotations

import time
from typing import Any, Callable, Tuple


def finalize_episode(
    obs: Any,
    info: dict,
    reset_fn: Callable[[], Tuple[Any, dict]],
    *,
    pause_s: float = 0.0,
) -> Tuple[Any, dict]:
    """Keep terminal flags for the caller; return first obs of the next episode."""
    info = dict(info)
    info["terminal_obs"] = obs
    if pause_s > 0:
        time.sleep(pause_s)
    next_obs, next_info = reset_fn()
    info.update(next_info)
    return next_obs, info
