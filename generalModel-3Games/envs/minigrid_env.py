
import os

import gymnasium as gym
import minigrid  # noqa
import numpy as np
import pygame
from gymnasium import ObservationWrapper, Wrapper, spaces
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Key

MINIGRID_ACTIONS = 7
DEFAULT_ENV_ID = "MiniGrid-DoorKey-16x16-v0"
AGENT_VIEW_SIZE = 7

N_OBJECT = len(OBJECT_TO_IDX)
N_COLOR = len(COLOR_TO_IDX)
N_STATE = len(STATE_TO_IDX)
N_CHANNELS = N_OBJECT + N_COLOR + N_STATE
GRID_OBS_SHAPE = (AGENT_VIEW_SIZE, AGENT_VIEW_SIZE, N_CHANNELS)
DEFAULT_OBS_DIM = AGENT_VIEW_SIZE * AGENT_VIEW_SIZE * N_CHANNELS

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

STALL_PATIENCE = 4
STALL_PENALTY_PER_STEP = 0.02
STALL_PENALTY_CAP = 0.1


class DoorKeyProgressRewardWrapper(Wrapper):

    def __init__(self, env, anti_stall: bool = False):
        super().__init__(env)
        self._key_reward_given = False
        self._door_open_rewarded: set[tuple[int, int]] = set()
        self._anti_stall = anti_stall
        self._prev_pos: tuple[int, int] | None = None
        self._stuck_steps = 0

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
        self._prev_pos = None
        self._stuck_steps = 0
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

        if self._anti_stall and not (terminated or truncated):
            pos = tuple(self._base_env().agent_pos)
            if self._prev_pos is not None and pos == self._prev_pos:
                self._stuck_steps += 1
            else:
                self._stuck_steps = 0
            self._prev_pos = pos
            if self._stuck_steps >= STALL_PATIENCE:
                over = self._stuck_steps - STALL_PATIENCE + 1
                stall_penalty = -min(STALL_PENALTY_PER_STEP * over, STALL_PENALTY_CAP)
                reward = float(reward) + stall_penalty
                info = dict(info)
                info["stall_penalty"] = stall_penalty

        return obs, reward, terminated, truncated, info


class OneHotFlatObsWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(DEFAULT_OBS_DIM,), dtype=np.float32
        )
        self._eye_obj = np.eye(N_OBJECT, dtype=np.float32)
        self._eye_col = np.eye(N_COLOR, dtype=np.float32)
        self._eye_state = np.eye(N_STATE, dtype=np.float32)

    def observation(self, obs):
        image = obs["image"].astype(np.int64)
        obj = np.clip(image[:, :, 0], 0, N_OBJECT - 1)
        col = np.clip(image[:, :, 1], 0, N_COLOR - 1)
        state = np.clip(image[:, :, 2], 0, N_STATE - 1)
        onehot = np.concatenate(
            [self._eye_obj[obj], self._eye_col[col], self._eye_state[state]],
            axis=-1,
        )
        return onehot.reshape(-1)


def make_minigrid_env(
    env_id: str = DEFAULT_ENV_ID,
    max_episode_steps: int | None = 1000,
    render_mode=None,
    anti_stall: bool | None = None,
):
    if anti_stall is None:
        anti_stall = os.environ.get("MINIGRID_ANTI_STALL", "0") == "1"
    env = gym.make(env_id, render_mode=render_mode, max_episode_steps=max_episode_steps)
    env = DoorKeyProgressRewardWrapper(env, anti_stall=anti_stall)
    env = OneHotFlatObsWrapper(env)
    return env


def minigrid_obs_dim(env_id: str = DEFAULT_ENV_ID) -> int:
    env = make_minigrid_env(env_id)
    dim = int(env.observation_space.shape[0])
    env.close()
    return dim


def refresh_minigrid_display(env) -> None:
    base = env.unwrapped
    screen_size = getattr(base, "screen_size", 640) or 640
    base.window = None
    if not pygame.get_init():
        pygame.init()
        pygame.display.init()
    pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("MiniGrid - Training")
