"""
Gymnasium-compatible Dino environment backed by the local pygame engine.
"""

from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

_ENGINE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dino_runGame")
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from engine import DinoGameEngine, DINO_X, HEIGHT, WIDTH  # noqa: E402

FEATURES_PER_FRAME = 12
FRAME_STACK = 4
OBS_DIM = FEATURES_PER_FRAME * FRAME_STACK
N_ACTIONS = 3

CANVAS_W = float(WIDTH)
CANVAS_H = float(HEIGHT)
SPEED_MAX = 13.0
DANGER_RANGE = 200.0
DEATH_REWARD = -10.0
SCORE_REWARD = 0.1
ALIVE_REWARD = 0.02
JUMP_COST = 0.01
PASS_BONUS = 1.0
DEFAULT_FRAMES_PER_STEP = 4
_NO_OBSTACLE_X = 9999.0


class DinoEnv:
    """Single pygame Dino instance with frame-stacked observations."""

    def __init__(
        self,
        render: bool = False,
        seed: int | None = None,
        env_id: int = 0,
        frames_per_step: int = DEFAULT_FRAMES_PER_STEP,
    ):
        self.render = render
        self.seed = seed
        self.env_id = env_id
        self.frames_per_step = max(1, int(frames_per_step))
        self.engine = DinoGameEngine(render=render, seed=seed)
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        self.prev_score = 0
        self._prev_o1_x = None
        self._obstacle_in_danger = False

    def _featurize(self, state):
        o1, o2 = state["o1"], state["o2"]

        def dx_norm(xpos):
            return min(max(0.0, xpos - DINO_X) / DANGER_RANGE, 1.5)

        return np.array(
            [
                state["dinoY"] / CANVAS_H,
                float(state["jumping"]),
                float(state["ducking"]),
                min(state["speed"], SPEED_MAX) / SPEED_MAX,
                dx_norm(o1[0]),
                o1[1] / CANVAS_H,
                o1[2] / CANVAS_W,
                o1[3] / CANVAS_H,
                dx_norm(o2[0]),
                o2[1] / CANVAS_H,
                o2[2] / CANVAS_W,
                o2[3] / CANVAS_H,
            ],
            dtype=np.float32,
        )

    def _stacked_obs(self):
        frames = list(self.frame_buffer)
        while len(frames) < FRAME_STACK:
            frames.insert(0, np.zeros(FEATURES_PER_FRAME, dtype=np.float32))
        return np.concatenate(frames, axis=0)

    def _detect_obstacle_pass(self, prev_x: float | None, cur_x: float) -> bool:
        """Detect when the front obstacle clears after entering the danger zone.

        Pygame obstacles move left (rect.left decreases). After a successful
        dodge they often sit behind the dino at negative x before despawning,
        so a one-frame threshold test misses the event. Latch once the nearest
        obstacle enters the approach window, then reward when it clears.
        """
        if prev_x is not None and prev_x < _NO_OBSTACLE_X and prev_x < CANVAS_W:
            if (DINO_X - 40.0) <= prev_x <= (DINO_X + DANGER_RANGE):
                self._obstacle_in_danger = True

        if not self._obstacle_in_danger:
            return False

        disappeared = cur_x >= _NO_OBSTACLE_X
        respawned_far = cur_x >= CANVAS_W * 0.45
        slot_jumped = (
            prev_x is not None
            and cur_x > prev_x + 80.0
            and prev_x < DINO_X + DANGER_RANGE
        )
        if disappeared or respawned_far or slot_jumped:
            self._obstacle_in_danger = False
            return True
        return False

    def reset(self):
        state = self.engine.reset()
        self.frame_buffer.clear()
        self.prev_score = int(state["score"])
        self._prev_o1_x = float(state["o1"][0])
        self._obstacle_in_danger = False
        self.frame_buffer.append(self._featurize(state))
        return self._stacked_obs()

    def step(self, action: int):
        action = int(action)
        total_reward = 0.0
        done = False
        info = {}
        state = None

        passed_any = False
        for frame_i in range(self.frames_per_step):
            state, done, info = self.engine.step(action)
            self.frame_buffer.append(self._featurize(state))
            score = int(state["score"])
            cur_o1_x = float(state["o1"][0])
            passed = self._detect_obstacle_pass(self._prev_o1_x, cur_o1_x)
            passed_any = passed_any or passed
            self._prev_o1_x = cur_o1_x

            if done:
                total_reward += DEATH_REWARD
            else:
                total_reward += (
                    ALIVE_REWARD
                    + SCORE_REWARD * max(0, score - self.prev_score)
                    - (JUMP_COST if action == 1 and frame_i == 0 else 0.0)
                    + (PASS_BONUS if passed else 0.0)
                )

            self.prev_score = score
            if done:
                break

        obs = self._stacked_obs()
        info.update(
            {
                "score": int(state["score"]),
                "speed": state["speed"],
                "o1_x": float(state["o1"][0]),
                "game_frames": (frame_i + 1) if state else 0,
                "passed": passed_any,
            }
        )
        if done:
            info["death_obstacle"] = state.get("o1type", "")
        return obs, total_reward, done, info

    def prepare_render(self):
        if self.render:
            self.engine.refresh_display()

    def close(self):
        self.engine.close()


def _dino_worker(remote, parent_remote, env_id: int, frames_per_step: int):
    """Subprocess worker — each process owns an isolated headless pygame instance."""
    import os

    parent_remote.close()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    env = DinoEnv(
        env_id=env_id,
        render=False,
        frames_per_step=frames_per_step,
    )
    try:
        while True:
            cmd, *data = remote.recv()
            if cmd == "reset":
                remote.send(env.reset())
            elif cmd == "step":
                remote.send(env.step(int(data[0])))
            elif cmd == "close":
                env.close()
                remote.close()
                break
    except (EOFError, KeyboardInterrupt):
        env.close()


class MPVecDinoEnv:
    """Multiprocess vector env — true parallel headless Dino instances."""

    def __init__(
        self,
        n_envs: int = 4,
        frames_per_step: int = DEFAULT_FRAMES_PER_STEP,
    ):
        import multiprocessing as mp

        self.n_envs = n_envs
        self.closed = False
        self.remotes, self.work_remotes = zip(*[mp.Pipe(duplex=True) for _ in range(n_envs)])
        self.processes = []
        for i, work_remote in enumerate(self.work_remotes):
            proc = mp.Process(
                target=_dino_worker,
                args=(work_remote, self.remotes[i], i, frames_per_step),
                daemon=True,
            )
            proc.start()
            work_remote.close()
            self.processes.append(proc)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset",))
        return np.stack([remote.recv() for remote in self.remotes], axis=0)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, infos = [], [], [], []
        for o, r, d, info in results:
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        reset_idxs = [i for i, d in enumerate(dones) if d]
        for i in reset_idxs:
            infos[i]["terminal_obs"] = obs[i]
            self.remotes[i].send(("reset",))
            obs[i] = self.remotes[i].recv()

        return (
            np.stack(obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def prepare_render(self):
        pass

    def close(self):
        if self.closed:
            return
        self.closed = True
        for remote in self.remotes:
            try:
                remote.send(("close",))
            except (BrokenPipeError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()


class VecDinoEnv:
    """In-process vector env (sequential steps). Use MPVecDinoEnv for parallel training."""

    def __init__(self, n_envs: int = 1, **env_kwargs):
        self.n_envs = n_envs
        self.envs = [DinoEnv(env_id=i, **env_kwargs) for i in range(n_envs)]

    def reset(self):
        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, d, info = env.step(int(action))
            obs.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        reset_idxs = [i for i, d in enumerate(dones) if d]
        for i in reset_idxs:
            infos[i]["terminal_obs"] = obs[i]
            obs[i] = self.envs[i].reset()
        return (
            np.stack(obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def prepare_render(self):
        for env in self.envs:
            env.prepare_render()

    def close(self):
        for env in self.envs:
            env.close()
