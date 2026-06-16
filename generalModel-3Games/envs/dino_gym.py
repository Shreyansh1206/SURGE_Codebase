
import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dino_env import (  # noqa: E402
    DinoEnv,
    MPVecDinoEnv,
    N_ACTIONS,
    OBS_DIM,
    VecDinoEnv,
)


class DinoGymEnv(gym.Env):

    metadata = {"render_modes": ["human", None]}

    def __init__(self, render_mode=None, **dino_kwargs):
        super().__init__()
        render = render_mode == "human"
        self._env = DinoEnv(render=render, **dino_kwargs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self._env.reset()
        return np.asarray(obs, dtype=np.float32), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        terminated = bool(done)
        truncated = False
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward),
            terminated,
            truncated,
            dict(info),
        )

    def close(self):
        self._env.close()


class VecDinoGymEnv:

    def __init__(
        self,
        n_envs: int = 1,
        render: bool = False,
        parallel: bool = False,
        **env_kwargs,
    ):
        self.n_envs = n_envs
        self.render = render
        self.renders_in_step = render
        if parallel:
            if render:
                raise ValueError("parallel=True requires render=False (headless workers).")
            frames_per_step = env_kwargs.get("frames_per_step", 4)
            self._vec = MPVecDinoEnv(n_envs=n_envs, frames_per_step=frames_per_step)
        else:
            if render:
                env_kwargs = dict(env_kwargs, frames_per_step=1)
            self._vec = VecDinoEnv(n_envs=n_envs, render=render, **env_kwargs)
        self.single_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_envs, OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * n_envs)

    def reset(self, *, seed=None, options=None):
        obs = self._vec.reset()
        return np.asarray(obs, dtype=np.float32), [{} for _ in range(self.n_envs)]

    def step(self, actions):
        obs, rewards, dones, infos = self._vec.step(np.asarray(actions, dtype=np.int64))
        terminated = dones.astype(bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        return (
            np.asarray(obs, dtype=np.float32),
            rewards.astype(np.float32),
            terminated,
            truncated,
            list(infos),
        )

    def prepare_render(self):
        self._vec.prepare_render()

    def render(self):
        if self.render and hasattr(self._vec, "envs") and self._vec.envs:
            self._vec.envs[0].prepare_render()

    def close(self):
        self._vec.close()


DINO_OBS_DIM = OBS_DIM
DINO_N_ACTIONS = N_ACTIONS
