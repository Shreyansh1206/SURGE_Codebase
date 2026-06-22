
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper, spaces

from envs.episode_utils import finalize_episode

CARRACING_ENV_ID = "CarRacing-v3"

# Discrete action set mapped onto continuous control [steer, gas, brake].
# Crucially this includes gas+steer combos so the car can accelerate THROUGH
# corners instead of coasting/drifting — the plain 5-action discrete set cannot
# steer and apply gas simultaneously, which forces sloppy, off-line driving.
CARRACING_DISCRETE_ACTIONS = np.array(
    [
        [0.0, 0.0, 0.0],   # 0 noop / coast
        [-1.0, 0.0, 0.0],  # 1 steer left
        [1.0, 0.0, 0.0],   # 2 steer right
        [0.0, 1.0, 0.0],   # 3 gas (straight)
        [0.0, 0.0, 0.8],   # 4 brake
        [-1.0, 0.5, 0.0],  # 5 gas + left
        [1.0, 0.5, 0.0],   # 6 gas + right
        [-1.0, 0.0, 0.8],  # 7 brake + left
        [1.0, 0.0, 0.8],   # 8 brake + right
    ],
    dtype=np.float32,
)
CARRACING_ACTION_NAMES = [
    "noop", "left", "right", "gas", "brake",
    "gas+left", "gas+right", "brake+left", "brake+right",
]
CARRACING_N_ACTIONS = len(CARRACING_DISCRETE_ACTIONS)

FRAME_SIZE = 96
FRAME_STACK = 4
CARRACING_OBS_SHAPE = (FRAME_STACK, FRAME_SIZE, FRAME_SIZE)

DEFAULT_FRAME_SKIP = 2
DEFAULT_ZOOM_SKIP = 20
# Base physics frames (50 FPS) without a new-tile reward before truncate.
# Default 0 = off. Values like 50 were ~1s in visible mode while still on track.
DEFAULT_NO_PROGRESS_PATIENCE = 0
DEFAULT_MAX_EPISODE_STEPS = 1000
# Physics frames the car may have ALL wheels off-track before we truncate the
# episode (0 = disabled). ~20 frames @50FPS ≈ 0.4s: enough to recover from a
# brief drift, but ends genuine spin-offs fast. Training-only; eval keeps it off
# so it measures the true native game score.
DEFAULT_OFFTRACK_PATIENCE = 20
# Small sparse penalty applied once when an off-track truncation fires. Kept well
# within the reward clip so it never destabilizes the critic (unlike the dense
# slip penalty that backfired).
DEFAULT_OFFTRACK_PENALTY = 2.0
# Clip per-step reward magnitude (0 = off). The base env emits -100 on leaving
# the playfield, which otherwise explodes the value loss under PPO.
DEFAULT_REWARD_CLIP = 10.0

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
INFO_PANEL_HEIGHT = 12  # bottom pixels to zero-out (speedometer / controls bar)

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
    gray = gray / 255.0
    gray[-INFO_PANEL_HEIGHT:] = 0.0
    return gray


class DiscretizeCarRacing(Wrapper):
    """Map a discrete action index onto the continuous [steer, gas, brake] vector."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(CARRACING_N_ACTIONS)
        self._actions = CARRACING_DISCRETE_ACTIONS

    def step(self, action):
        return self.env.step(self._actions[int(action)])


CARRACING_AUX_DIM = 3  # [speed_norm, lateral_slip_norm, angular_vel_norm]


def _base_car(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    base = getattr(base, "unwrapped", base)
    return getattr(base, "car", None)


def _get_car_physics(env) -> tuple[float, float, float]:
    """Extract speed, lateral slip, angular velocity from the base CarRacing env."""
    car = _base_car(env)
    if car is None or not hasattr(car, "hull"):
        return 0.0, 0.0, 0.0
    vx, vy = car.hull.linearVelocity
    speed = float(np.sqrt(vx ** 2 + vy ** 2))
    angle = float(car.hull.angle)
    fwd = np.array([np.cos(angle), np.sin(angle)])
    vel = np.array([float(vx), float(vy)])
    lateral = float(np.abs(fwd[0] * vel[1] - fwd[1] * vel[0]))
    ang_vel = float(abs(car.hull.angularVelocity))
    return speed, lateral, ang_vel


def _wheels_on_track(env) -> bool:
    """True if at least one wheel is touching a track tile.

    The base env only emits its -100 death at the far playfield boundary, long
    after a corner spin-out. Detecting wheel-off-track lets us truncate early so
    the agent gets a clean 'left the road -> episode over' signal (literature:
    Columbia-F1-Robotics, felsangom, Mike.W timeout).
    """
    car = _base_car(env)
    if car is None or not hasattr(car, "wheels") or not car.wheels:
        return True  # fail-safe: assume on track if we can't introspect
    for w in car.wheels:
        if len(getattr(w, "tiles", []) or []) > 0:
            return True
    return False


class CarRacingControlWrapper(Wrapper):

    def __init__(
        self,
        env,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        zoom_skip: int = DEFAULT_ZOOM_SKIP,
        no_progress_patience: int = DEFAULT_NO_PROGRESS_PATIENCE,
        reward_clip: float = DEFAULT_REWARD_CLIP,
        visible: bool = False,
        slip_penalty: float = 0.0,
        offtrack_patience: int = DEFAULT_OFFTRACK_PATIENCE,
        offtrack_penalty: float = DEFAULT_OFFTRACK_PENALTY,
    ):
        super().__init__(env)
        self.frame_skip = max(1, int(frame_skip))
        self.zoom_skip = max(0, int(zoom_skip))
        self.no_progress_patience = max(0, int(no_progress_patience))
        self.reward_clip = float(reward_clip)
        self.slip_penalty = float(slip_penalty)
        self.offtrack_patience = max(0, int(offtrack_patience))
        self.offtrack_penalty = float(offtrack_penalty)
        self.visible = visible
        self._no_progress_frames = 0
        self._offtrack_frames = 0
        self._episodes_started = 0
        self.last_aux = np.zeros(CARRACING_AUX_DIM, dtype=np.float32)

    def _update_aux(self):
        speed, lateral, ang_vel = _get_car_physics(self)
        self.last_aux[0] = min(speed / 100.0, 1.0)
        self.last_aux[1] = min(lateral / 50.0, 1.0)
        self.last_aux[2] = min(ang_vel / 5.0, 1.0)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._no_progress_frames = 0
        self._offtrack_frames = 0
        self._episodes_started += 1
        self.last_aux[:] = 0.0
        zoom = self.zoom_skip
        if self.visible and self._episodes_started > 1:
            zoom = 0
        for _ in range(zoom):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(seed=seed, options=options)
                break
        self._update_aux()
        if self.visible:
            _present_carracing(self)
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        obs = None
        visited_new_tile = False
        n_physics = 0
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            n_physics += 1
            total_reward += float(reward)
            if reward > 0.0:
                visited_new_tile = True
            if terminated or truncated:
                break

        self._update_aux()

        if visited_new_tile:
            self._no_progress_frames = 0
        else:
            self._no_progress_frames += n_physics

        if (
            self.no_progress_patience > 0
            and self._no_progress_frames >= self.no_progress_patience
            and not (terminated or truncated)
        ):
            truncated = True
            info = dict(info)
            info["no_progress_stop"] = True

        # Off-track early truncation: end the episode shortly after all wheels
        # leave the road instead of waiting for the far-boundary -100 death.
        if self.offtrack_patience > 0 and not (terminated or truncated):
            if _wheels_on_track(self):
                self._offtrack_frames = 0
            else:
                self._offtrack_frames += n_physics
                if self._offtrack_frames >= self.offtrack_patience:
                    truncated = True
                    total_reward -= self.offtrack_penalty
                    info = dict(info)
                    info["offtrack_stop"] = True

        if self.slip_penalty > 0:
            lateral_slip = self.last_aux[1]
            total_reward -= self.slip_penalty * lateral_slip

        raw_reward = total_reward
        if self.reward_clip > 0:
            total_reward = float(np.clip(total_reward, -self.reward_clip, self.reward_clip))
        if raw_reward != total_reward:
            info = dict(info)
            info["raw_reward"] = raw_reward

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

    def _get_aux(self) -> np.ndarray:
        ctrl = self.env
        while ctrl is not None and not isinstance(ctrl, CarRacingControlWrapper):
            ctrl = getattr(ctrl, "env", None)
        if ctrl is not None:
            return ctrl.last_aux.copy()
        return np.zeros(CARRACING_AUX_DIM, dtype=np.float32)

    def _stacked(self) -> np.ndarray:
        return np.stack(self.frames, axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        gray = _to_gray(obs)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(gray)
        info["aux"] = self._get_aux()
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(_to_gray(obs))
        info = dict(info) if not isinstance(info, dict) else info
        info["aux"] = self._get_aux()
        return self._stacked(), float(reward), terminated, truncated, info


DEFAULT_SLIP_PENALTY = 0.0


def make_carracing_env(
    render_mode=None,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    frame_stack: int = FRAME_STACK,
    zoom_skip: int = DEFAULT_ZOOM_SKIP,
    no_progress_patience: int = DEFAULT_NO_PROGRESS_PATIENCE,
    reward_clip: float = DEFAULT_REWARD_CLIP,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    visible: bool = False,
    slip_penalty: float = DEFAULT_SLIP_PENALTY,
    offtrack_patience: int = DEFAULT_OFFTRACK_PATIENCE,
    offtrack_penalty: float = DEFAULT_OFFTRACK_PENALTY,
):
    visible = visible or render_mode == "human"
    if visible:
        frame_skip = 1
    env = gym.make(
        CARRACING_ENV_ID,
        continuous=True,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    env = DiscretizeCarRacing(env)
    env = CarRacingControlWrapper(
        env,
        frame_skip=frame_skip,
        zoom_skip=zoom_skip,
        no_progress_patience=no_progress_patience,
        reward_clip=reward_clip,
        visible=visible,
        slip_penalty=slip_penalty,
        offtrack_patience=offtrack_patience,
        offtrack_penalty=offtrack_penalty,
    )
    env = GrayFrameStackWrapper(env, frame_stack=frame_stack)
    return env


def carracing_obs_shape(frame_stack: int = FRAME_STACK) -> tuple[int, int, int]:
    return (frame_stack, FRAME_SIZE, FRAME_SIZE)


class _RunningMeanStd:
    """Welford-style running mean/variance for reward normalization."""
    __slots__ = ("mean", "var", "count")

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: np.ndarray):
        batch = np.asarray(x, dtype=np.float64).ravel()
        batch_mean = float(np.mean(batch))
        batch_var = float(np.var(batch))
        batch_count = len(batch)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = (self.var * self.count + batch_var * batch_count
              + delta ** 2 * self.count * batch_count / total)
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


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
        reward_clip: float = DEFAULT_REWARD_CLIP,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        episode_end_pause: float = 0.0,
        normalize_reward: bool = False,
        gamma: float = 0.99,
    ):
        self.show_window = render
        self.episode_end_pause = episode_end_pause if render else 0.0
        self._single = None
        self._vec = None
        self.frame_stack = frame_stack

        env_kwargs = dict(
            frame_skip=frame_skip,
            frame_stack=frame_stack,
            zoom_skip=zoom_skip,
            no_progress_patience=no_progress_patience,
            reward_clip=reward_clip,
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

        self.normalize_reward = normalize_reward
        self._rew_rms = _RunningMeanStd()
        self._disc_return = np.zeros(self.n_envs, dtype=np.float64)
        self._raw_ep_return = np.zeros(self.n_envs, dtype=np.float64)
        self._gamma = gamma

    def reset(self, *, seed=None, options=None):
        self._raw_ep_return[:] = 0.0
        self._disc_return[:] = 0.0
        if self._single is not None:
            obs, info = self._single.reset(seed=seed)
            return np.asarray([obs], dtype=np.float32), [info]
        return self._vec.reset(seed=seed)

    def _normalize_rewards(self, rewards, dones):
        """VecNormalize-style: divide by running std of discounted returns."""
        self._disc_return = self._disc_return * self._gamma + rewards.astype(np.float64)
        self._rew_rms.update(self._disc_return)
        self._disc_return[dones] = 0.0
        return np.clip(rewards / self._rew_rms.std, -10.0, 10.0).astype(np.float32)

    def step(self, actions):
        if self._single is not None:
            action = int(np.asarray(actions).reshape(-1)[0])
            obs, reward, terminated, truncated, info = self._single.step(action)
            self._raw_ep_return[0] += float(reward)
            if terminated or truncated:
                raw_ret = float(self._raw_ep_return[0])
                self._raw_ep_return[0] = 0.0
                obs, info = finalize_episode(
                    obs,
                    info,
                    lambda: self._single.reset(),
                    pause_s=self.episode_end_pause,
                )
                info["raw_return"] = raw_ret
            rewards = np.asarray([reward], dtype=np.float32)
            if self.normalize_reward:
                done_arr = np.array([terminated or truncated])
                rewards = self._normalize_rewards(rewards, done_arr)
            return (
                np.asarray([obs], dtype=np.float32),
                rewards,
                np.asarray([terminated], dtype=bool),
                np.asarray([truncated], dtype=bool),
                [info],
            )

        # --- vectorized mode ---
        obs, rewards, terminated, truncated, infos = self._vec.step(
            np.asarray(actions, dtype=np.int64)
        )
        self._raw_ep_return += rewards
        dones = np.logical_or(terminated, truncated)
        done_idx = np.where(dones)[0]
        if len(done_idx) > 0:
            raw_rets = np.full(self.n_envs, np.nan, dtype=np.float32)
            raw_rets[done_idx] = self._raw_ep_return[done_idx].astype(np.float32)
            if isinstance(infos, dict):
                infos = dict(infos)
                infos["raw_return"] = raw_rets
            elif isinstance(infos, list):
                for idx in done_idx:
                    if idx < len(infos):
                        infos[idx] = dict(infos[idx]) if isinstance(infos[idx], dict) else {}
                        infos[idx]["raw_return"] = float(self._raw_ep_return[idx])
            self._raw_ep_return[dones] = 0.0
        if self.normalize_reward:
            rewards = self._normalize_rewards(rewards, dones)
        return obs, rewards, terminated, truncated, infos

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
    normalize_reward: bool = False,
    gamma: float = 0.99,
    **kwargs,
) -> CarRacingVecEnv:
    return CarRacingVecEnv(
        n_envs, seed=seed, render=render,
        normalize_reward=normalize_reward, gamma=gamma,
        **kwargs,
    )
