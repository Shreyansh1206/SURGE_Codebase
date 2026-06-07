"""MiniGrid factory — 7×7 egocentric observations via the official minigrid library."""

import gymnasium as gym
import minigrid  # noqa: F401 — registers env IDs
import numpy as np
import pygame
from gymnasium import ObservationWrapper, Wrapper, spaces
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Key

MINIGRID_ACTIONS = 7
DEFAULT_ENV_ID = "MiniGrid-DoorKey-16x16-v0"
# Default MiniGrid agent view: 7×7 cells in front of the agent.
# Tiles outside the map are encoded as unseen (0, 0, 0) by the environment.
AGENT_VIEW_SIZE = 7

# The MiniGrid "image" obs is NOT pixels — each cell is the symbolic triple
# [object_id, color_id, state_id]. object_id ∈ [0,10], color_id ∈ [0,5],
# state_id ∈ [0,2]. We one-hot each channel so the network can actually tell a
# wall from a key from the goal, instead of dividing tiny integer ids by 255
# (which crushed every token into [0, 0.04] and made learning near-impossible).
N_OBJECT = len(OBJECT_TO_IDX)   # 11
N_COLOR = len(COLOR_TO_IDX)     # 6
N_STATE = len(STATE_TO_IDX)     # 3
N_CHANNELS = N_OBJECT + N_COLOR + N_STATE  # 20
GRID_OBS_SHAPE = (AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, N_CHANNELS)  # (7, 7, 20)
DEFAULT_OBS_DIM = AGENT_VIEW_SIZE * AGENT_VIEW_SIZE * N_CHANNELS  # 980

# DoorKey is built in for 5x5/6x6/8x8/16x16 only. Register intermediate sizes so a
# curriculum can ramp 8x8 → 16x16 smoothly. The partial view stays 7×7 at every
# size, so the same network transfers across all stages with no surgery.
_EXTRA_DOORKEY_SIZES = (10, 12, 14)


def _register_extra_doorkey_envs() -> None:
    for size in _EXTRA_DOORKEY_SIZES:
        env_id = f"MiniGrid-DoorKey-{size}x{size}-v0"
        if env_id in gym.registry:
            continue
        gym.register(
            id=env_id,
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": size},
        )


_register_extra_doorkey_envs()

KEY_PICKUP_REWARD = 0.25
KEY_DROP_PENALTY = -0.05
DOOR_OPEN_REWARD = 0.35


class DoorKeyProgressRewardWrapper(Wrapper):
    """Milestone rewards/penalties for key pickup, key drop, and door opening."""

    def __init__(self, env):
        super().__init__(env)
        self._key_reward_given = False
        self._door_open_rewarded: set[tuple[int, int]] = set()

    def _base_env(self):
        return self.unwrapped

    def _carrying_key(self) -> bool:
        carrying = getattr(self._base_env(), "carrying", None)
        return isinstance(carrying, Key)

    def _open_door_positions(self) -> frozenset[tuple[int, int]]:
        grid = self._base_env().grid
        open_doors = set()
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None and cell.type == "door" and cell.is_open:
                    open_doors.add((x, y))
        return frozenset(open_doors)

    def reset(self, *, seed=None, options=None):
        self._key_reward_given = False
        self._door_open_rewarded = set()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        had_key = self._carrying_key()
        open_before = self._open_door_positions()

        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping = 0.0
        key_bonus = 0.0
        key_drop_penalty = 0.0
        door_bonus = 0.0

        if not had_key and self._carrying_key() and not self._key_reward_given:
            key_bonus = KEY_PICKUP_REWARD
            self._key_reward_given = True

        if had_key and not self._carrying_key():
            key_drop_penalty = KEY_DROP_PENALTY

        open_after = self._open_door_positions()
        for pos in open_after - open_before:
            if pos not in self._door_open_rewarded:
                door_bonus += DOOR_OPEN_REWARD
                self._door_open_rewarded.add(pos)

        shaping = key_bonus + key_drop_penalty + door_bonus
        if shaping:
            reward = float(reward) + shaping
            info = dict(info)
            if key_bonus:
                info["key_pickup_bonus"] = key_bonus
            if key_drop_penalty:
                info["key_drop_penalty"] = key_drop_penalty
            if door_bonus:
                info["door_open_bonus"] = door_bonus

        return obs, reward, terminated, truncated, info


class OneHotFlatObsWrapper(ObservationWrapper):
    """
    One-hot the symbolic (object, color, state) channels and flatten.

    Output is a float32 vector of length 7×7×20 = 980, laid out in (H, W, C)
    order so a CNN can reshape it back to (H, W, C) → (C, H, W).
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(DEFAULT_OBS_DIM,), dtype=np.float32
        )
        # Precompute identity matrices for fast one-hot lookup.
        self._eye_obj = np.eye(N_OBJECT, dtype=np.float32)
        self._eye_col = np.eye(N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(N_STATE, dtype=np.float32)

    def observation(self, obs):
        image = obs["image"].astype(np.int64)  # (7, 7, 3)
        obj = np.clip(image[:, :, 0], 0, N_OBJECT - 1)
        col = np.clip(image[:, :, 1], 0, N_COLOR - 1)
        state = np.clip(image[:, :, 2], 0, N_STATE - 1)
        onehot = np.concatenate(
            [self._eye_obj[obj], self._eye_col[col], self._eye_state[state]],
            axis=-1,
        )  # (7, 7, 20)
        return onehot.reshape(-1)


def make_minigrid_env(
    env_id: str = DEFAULT_ENV_ID,
    max_episode_steps: int | None = 1000,
    render_mode=None,
):
    """
    Create a MiniGrid env with the default 7×7 egocentric view.

    Uses gymnasium.make + Farama minigrid partial observability only — no
    FullyObsWrapper. Out-of-map cells in the 7×7 window are unseen (0, 0, 0).
    Pass ``max_episode_steps=None`` to use MiniGrid's own size-scaled default
    (10·size²), or an explicit cap for a tighter learning signal.
    """
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = DoorKeyProgressRewardWrapper(env)
    env = OneHotFlatObsWrapper(env)
    return env


def minigrid_obs_dim(env_id: str = DEFAULT_ENV_ID) -> int:
    env = make_minigrid_env(env_id)
    dim = int(env.observation_space.shape[0])
    env.close()
    return dim


def refresh_minigrid_display(env) -> None:
    """
    Reclaim a full-size MiniGrid window after Dino (or another env) resized pygame.
    """
    base = env.unwrapped
    screen_size = getattr(base, "screen_size", 640) or 640
    base.window = None
    if not pygame.get_init():
        pygame.init()
        pygame.display.init()
    pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("MiniGrid - Training")
