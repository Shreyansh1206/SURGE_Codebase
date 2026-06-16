
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces

CARRACING_ENV_ID = "CarRacing-v3"

CARRACING_N_ACTIONS = 5
CARRACING_ACTION_NAMES = ["noop", "left", "right", "gas", "brake"]

FRAME_SIZE = 96
FRAME_STACK = 4
CARRACING_OBS_SHAPE = (FRAME_STACK, FRAME_SIZE, FRAME_SIZE)

DEFAULT_FRAME_SKIP = 4
DEFAULT_ZOOM_SKIP = 40
DEFAULT_NO_PROGRESS_PATIENCE = 50
DEFAULT_MAX_EPISODE_STEPS = 1000

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)

CARRACING_WINDOW_W = 1000
CARRACING_WINDOW_H = 800


def _ensure_carracing_window() -> None:
    import pygame

    pygame.display.init()
    surf = pygame.display.get_surface()
    want = (CARRACING_WINDOW_W, CARRACING_WINDOW_H)
    if surf is None or surf.get_size() != want:
        pygame.display.set_mode(want)
    pygame.display.set_caption("CarRacing - Training")


def _present_carracing(env) -> None:
    base = env
    while hasattr(base, "env"):
        base = base.env
    _ensure_carracing_window()
    if hasattr(base, "render"):
        base.render()


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32) @ _GRAY_WEIGHTS
    return gray / 255.0


class CarRacingControlWrapper(Wrapper):

    def __init__(
        self,
        env,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        zoom_skip: int = DEFAULT_ZOOM_SKIP,
        no_progress_patience: int = DEFAULT_NO_PROGRESS_PATIENCE,
        visible: bool = False,
    ):
        super().__init__(env)
        self.frame_skip = max(1, int(frame_skip))
        self.zoom_skip = max(0, int(zoom_skip))
        self.no_progress_patience = max(0, int(no_progress_patience))
        self.visible = visible
        self._no_progress = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._no_progress = 0
        for _ in range(self.zoom_skip):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
                break
        if self.visible:
            _present_carracing(self)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        obs = None
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        if total_reward > 0.0:
            self._no_progress = 0
        else:
            self._no_progress += 1

        if (
            self.no_progress_patience
            and self._no_progress >= self.no_progress_patience
            and not (terminated or truncated)
        ):
            truncated = True
            info = dict(info)
            info["no_progress_stop"] = True

        return obs, total_reward, terminated, truncated, info


class GrayFrameStackWrapper(Wrapper):

    def __init__(self, env, frame_stack: int = FRAME_STACK):
        super().__init__(env)
        self.frame_stack = max(1, int(frame_stack))
        self.frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.frame_stack, FRAME_SIZE, FRAME_SIZE),
            dtype=np.float32,
        )

    def _stacked(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        gray = _to_gray(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(gray)
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(_to_gray(obs))
        return self._stacked(), float(reward), terminated, truncated, info


def make_carracing_env(
    render_mode=None,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    frame_stack: int = FRAME_STACK,
    zoom_skip: int = DEFAULT_ZOOM_SKIP,
    no_progress_patience: int = DEFAULT_NO_PROGRESS_PATIENCE,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    visible: bool = False,
):
    visible = visible or render_mode == "human"
    if visible:
        frame_skip = 1
    env = gym.make(
        CARRACING_ENV_ID,
        continuous=False,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    env = CarRacingControlWrapper(
        env,
        frame_skip=frame_skip,
        zoom_skip=zoom_skip,
        no_progress_patience=no_progress_patience,
        visible=visible,
    )
    env = GrayFrameStackWrapper(env, frame_stack=frame_stack)
    return env


def carracing_obs_shape(frame_stack: int = FRAME_STACK) -> tuple[int, int, int]:
    return (frame_stack, FRAME_SIZE, FRAME_SIZE)


class CarRacingVecEnv:

    def __init__(
        self,
        n_envs: int,
        seed: int = 0,
        render: bool = False,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        frame_stack: int = FRAME_STACK,
        zoom_skip: int = DEFAULT_ZOOM_SKIP,
        no_progress_patience: int = DEFAULT_NO_PROGRESS_PATIENCE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    ):
        self.show_window = render
        self._single = None
        self._vec = None
        self.frame_stack = frame_stack

        env_kwargs = dict(
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            zoom_skip=zoom_skip,
            no_progress_patience=no_progress_patience,
            max_episode_steps=max_episode_steps,
        )

        if render or n_envs == 1:
            self.n_envs = 1
            render_mode = "human" if render else None
            self.renders_in_step = bool(render)
            self._single = make_carracing_env(
                render_mode=render_mode, visible=render, **env_kwargs
            )
            self._single.reset(seed=seed)
        else:
            self.n_envs = n_envs
            self.renders_in_step = False

            def _factory():
                return make_carracing_env(**env_kwargs)

            self._vec = gym.vector.SyncVectorEnv([_factory for _ in range(n_envs)])
            self._vec.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        if self._single is not None:
            obs, info = self._single.reset(seed=seed)
            return np.asarray([obs], dtype=np.float32), [info]
        return self._vec.reset(seed=seed)

    def step(self, actions):
        if self._single is not None:
            action = int(np.asarray(actions).reshape(-1)[0])
            obs, reward, terminated, truncated, info = self._single.step(action)
            return (
                np.asarray([obs], dtype=np.float32),
                np.asarray([reward], dtype=np.float32),
                np.asarray([terminated], dtype=bool),
                np.asarray([truncated], dtype=bool),
                [info],
            )
        return self._vec.step(np.asarray(actions, dtype=np.int64))

    def prepare_render(self):
        if self._single is not None and self.show_window:
            _ensure_carracing_window()
            _present_carracing(self._single)

    def render(self):
        self.prepare_render()

    def close(self):
        if self._single is not None:
            self._single.close()
        if self._vec is not None:
            self._vec.close()


def make_carracing_vec_env(
    n_envs: int,
    seed: int = 0,
    render: bool = False,
    **kwargs,
) -> CarRacingVecEnv:
    return CarRacingVecEnv(n_envs, seed=seed, render=render, **kwargs)
