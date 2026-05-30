"""
Selenium-driven Chrome Dino environment — DQN variant.

Reward shaping (post-debug-v2):
  - +0.1 per alive step (ALIVE_REWARD)
  - -1.0 on death (DEATH_REWARD)
  - +0.5 per obstacle that passes the dino (PASS_BONUS) — applies to cacti
    and pterodactyls alike; detected by tracking when slot-0 obstacle x
    jumps backwards (= a real obstacle was removed and a further one slid
    into its place)
  - -0.02 per jump OR duck action (ACTION_COST) — penalizes "vertical"
    actions so the agent doesn't reflexively jump on empty screens. Noop
    is free.

The flat +0.1/-1 scheme failed: with uniform replay over mostly non-decisive
frames, the network learned action-indifferent Q-values everywhere. PASS_BONUS
strengthens the credit for "jumped at the right time"; ACTION_COST raises the
floor on wasted jumps. Together they should pull the policy out of its
"jump-on-empty-screen + duck-mid-air" degenerate equilibrium.

Phantom-pass detection:
  t-rex-runner advances obstacles by `currentSpeed * (deltaTime / msPerFrame)`
  pixels per rAF callback. When Chrome briefly throttles a tab (which happens
  frequently to non-foreground windows in our 4-env training setup), the next
  rAF fires with a fat deltaTime and the obstacle teleports forward — often
  jumping clean across the dino's hitbox without the once-per-frame collision
  check ever computing an overlap. The game wrongly treats the dino as alive.

  We catch this: cacti are ground-level and can only be cleared by jumping.
  If a cactus passes (slot-0 x jumps backwards) but the dino was never
  airborne in the recent N-step window, the game-engine missed a collision.
  We override `done=True` with DEATH_REWARD to restore the training signal.
  Birds are not phantom-checked (high birds need no action; classification
  is fuzzier).
  - Per-frame features expanded from 12 to 14: added a one-hot "is_bird" flag
    for the front two obstacles so the agent can learn jump-for-cactus vs
    duck-for-low-bird. Crucial for scores > ~500 once pterodactyls appear.
  - `action_mask()` exposed so the agent can mask the jump action while the
    dino is airborne. Mask is applied at action selection AND inside the
    Bellman target — wiring lives in dqn_agent.py.

Action space (same 3 actions as PPO env):
    0  no-op   — release any held duck key
    1  jump    — tap SPACE
    2  duck    — hold DOWN

The Selenium plumbing is intentionally a near-duplicate of ppo/dino_env.py so
the two implementations can evolve independently.
"""

import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# t-rex-runner lives at ../t-rex-runner relative to this file.
_LOCAL_GAME = "file:///" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "t-rex-runner", "index.html"
).replace("\\", "/")

try:
    from webdriver_manager.chrome import ChromeDriverManager
    _HAS_WDM = True
except ImportError:
    _HAS_WDM = False


# ---------------------------------------------------------------------------
# Keyboard event JS. See ppo/dino_env.py for the keyCode/key/code requirement.
# ---------------------------------------------------------------------------
_JS_SPACE_DOWN = (
    "document.dispatchEvent(new KeyboardEvent('keydown',"
    "{key:' ',code:'Space',keyCode:32,which:32,bubbles:true}));"
)
_JS_SPACE_UP = (
    "document.dispatchEvent(new KeyboardEvent('keyup',"
    "{key:' ',code:'Space',keyCode:32,which:32,bubbles:true}));"
)
_JS_DOWN_DOWN = (
    "document.dispatchEvent(new KeyboardEvent('keydown',"
    "{key:'ArrowDown',code:'ArrowDown',keyCode:40,which:40,bubbles:true}));"
)
_JS_DOWN_UP = (
    "document.dispatchEvent(new KeyboardEvent('keyup',"
    "{key:'ArrowDown',code:'ArrowDown',keyCode:40,which:40,bubbles:true}));"
)


# Feature layout per frame (14 dims):
#   0  dinoY / CANVAS_H
#   1  jumping flag
#   2  ducking flag
#   3  speed / SPEED_MAX
#   4  dx_o1 (normalised, 1.5 = sentinel "very far")
#   5  y_o1 / CANVAS_H
#   6  w_o1 / CANVAS_W
#   7  h_o1 / CANVAS_H
#   8  o1 is_bird (1.0 for pterodactyl, 0.0 for cactus / no obstacle)
#   9  dx_o2
#  10  y_o2 / CANVAS_H
#  11  w_o2 / CANVAS_W
#  12  h_o2 / CANVAS_H
#  13  o2 is_bird
FEATURES_PER_FRAME = 14
FRAME_STACK        = 4
OBS_DIM            = FEATURES_PER_FRAME * FRAME_STACK
N_ACTIONS          = 3

# Indices into a single feature frame (used by action_mask).
FEAT_JUMPING = 1

CANVAS_W   = 600.0
CANVAS_H   = 150.0
SPEED_MAX  = 13.0
DINO_X       = 44.0
DANGER_RANGE = 200.0

# Reward scheme. See module docstring for rationale.
ALIVE_REWARD = 0.1
DEATH_REWARD = -1.0
PASS_BONUS   = 0.5    # awarded when an obstacle (cactus OR bird) slides past
ACTION_COST  = 0.02   # per-step penalty for jump or duck (action != 0)

# Phantom-pass detection. Window of recent steps over which we verify the
# dino was airborne to clear a cactus. At max speed the obstacle is in
# collision range for ~2 steps; at min speed ~4. A length-5 window covers
# both with margin.
JUMP_WINDOW = 5


def action_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    """Compute the action mask from a stacked observation.

    The latest frame is the LAST `FEATURES_PER_FRAME` entries (frame_buffer is
    appended in order, then concatenated). While airborne, jump (index 1) is
    masked; noop and duck stay available (duck mid-air = fast-fall).

    Pure function so it can be called both inside the env and in the training
    loop on stored observations from the replay buffer.
    """
    mask = np.ones(N_ACTIONS, dtype=bool)
    latest = obs[-FEATURES_PER_FRAME:]
    if latest[FEAT_JUMPING] > 0.5:
        mask[1] = False
    return mask


class DinoEnv:
    def __init__(self,
                 chromedriver_path: str = None,
                 headless: bool = False,
                 frames_per_step: int = 4,
                 step_pause: float = 0.0,
                 game_url: str = None,
                 window_size: tuple = (700, 320),
                 window_position: tuple = (0, 0),
                 env_id: int = 0):
        self.env_id = env_id
        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-infobars")
        opts.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        opts.add_argument(f"--window-position={window_position[0]},{window_position[1]}")
        opts.add_argument("--disable-background-timer-throttling")
        opts.add_argument("--disable-backgrounding-occluded-windows")
        opts.add_argument("--disable-renderer-backgrounding")
        if headless:
            opts.add_argument("--headless=new")

        if chromedriver_path:
            service = Service(chromedriver_path)
        elif _HAS_WDM:
            service = Service(ChromeDriverManager().install())
        else:
            service = Service()

        self.driver = webdriver.Chrome(service=service, options=opts)
        self.frame_sleep = frames_per_step / 60.0
        self.step_pause  = step_pause

        url = game_url or _LOCAL_GAME
        print(f"[dqn_env {self.env_id}] Loading: {url}")
        try:
            self.driver.get(url)
        except Exception:
            pass

        time.sleep(1.0)
        self._wait_for_runner_constructor(timeout=10.0)
        self._start_game()
        self._wait_for_runner_instance(timeout=5.0)

        self.body         = self.driver.find_element(By.TAG_NAME, "body")
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        self._held_down   = False
        self.prev_score   = 0
        self._prev_o1_x       = None    # for PASS_BONUS detection
        self._prev_o1_is_bird = False   # for phantom-pass detection
        self._jumping_history = deque(maxlen=JUMP_WINDOW)
        self._phantom_count   = 0       # cumulative for diagnostics

    # ---------------------------------------------------------------- init helpers

    def _wait_for_runner_constructor(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.driver.execute_script("return typeof Runner !== 'undefined';"):
                    print(f"[dqn_env {self.env_id}] Runner constructor ready.")
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"Runner not found after {timeout:.0f}s.")

    def _start_game(self):
        print(f"[dqn_env {self.env_id}] Sending first keypress to start Runner ...")
        self.driver.execute_script(_JS_SPACE_DOWN + _JS_SPACE_UP)
        time.sleep(0.5)

    def _wait_for_runner_instance(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.driver.execute_script("return Runner.instance_ != null;"):
                    print(f"[dqn_env {self.env_id}] Runner.instance_ ready.")
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"Runner.instance_ null after {timeout:.0f}s.")

    # ---------------------------------------------------------------- state

    def _read_state(self):
        return self.driver.execute_script("""
        const r = Runner.instance_;
        if (!r) return null;
        const t   = r.tRex;
        const obs = r.horizon.obstacles || [];
        const o1  = obs[0] || null;
        const o2  = obs[1] || null;
        function pack(o) {
            // Sentinel: dx normalises to ≥ 1.5 (clipped) = "very far away".
            // Bird flag also 0 when there's no obstacle.
            if (!o) return [9999, 0.0, 0.0, 0.0, 0];
            const isBird = (o.typeConfig && o.typeConfig.type === 'PTERODACTYL') ? 1 : 0;
            return [o.xPos, o.yPos, o.width, o.typeConfig.height, isBird];
        }
        return {
            crashed : r.crashed,
            playing : r.playing,
            score   : Math.floor(r.distanceRan * r.distanceMeter.config.COEFFICIENT),
            speed   : r.currentSpeed,
            dinoY   : t.yPos,
            jumping : t.jumping  ? 1 : 0,
            ducking : t.ducking  ? 1 : 0,
            o1      : pack(o1),
            o2      : pack(o2),
        };
        """)

    def _featurize(self, s):
        o1, o2 = s["o1"], s["o2"]

        def dx_norm(xpos):
            return min(max(0.0, xpos - DINO_X) / DANGER_RANGE, 1.5)

        return np.array([
            s["dinoY"] / CANVAS_H,
            float(s["jumping"]),
            float(s["ducking"]),
            min(s["speed"], SPEED_MAX) / SPEED_MAX,
            dx_norm(o1[0]), o1[1] / CANVAS_H, o1[2] / CANVAS_W, o1[3] / CANVAS_H, float(o1[4]),
            dx_norm(o2[0]), o2[1] / CANVAS_H, o2[2] / CANVAS_W, o2[3] / CANVAS_H, float(o2[4]),
        ], dtype=np.float32)

    def _stacked_obs(self):
        frames = list(self.frame_buffer)
        while len(frames) < FRAME_STACK:
            frames.insert(0, np.zeros(FEATURES_PER_FRAME, dtype=np.float32))
        return np.concatenate(frames, axis=0)

    # ---------------------------------------------------------------- actions

    def _release_down(self):
        if self._held_down:
            self.driver.execute_script(_JS_DOWN_UP)
            self._held_down = False

    def _do_action(self, action: int):
        if action == 0:
            self._release_down()
        elif action == 1:
            self._release_down()
            self.driver.execute_script(_JS_SPACE_DOWN + _JS_SPACE_UP)
        elif action == 2:
            if not self._held_down:
                self.driver.execute_script(_JS_DOWN_DOWN)
                self._held_down = True

    # ---------------------------------------------------------------- public API

    def action_mask(self):
        """Bool[N_ACTIONS]. True = action allowed. Mask jump while airborne."""
        if not self.frame_buffer:
            return np.ones(N_ACTIONS, dtype=bool)
        return action_mask_from_obs(self._stacked_obs())

    def reset(self):
        self._release_down()
        self.driver.execute_script(
            "const r = Runner.instance_;"
            "if (r.crashed) { r.restart(); }"
            "else if (!r.playing) {"
            "  " + _JS_SPACE_DOWN +
            "  " + _JS_SPACE_UP +
            "}"
        )
        time.sleep(0.4)

        self.frame_buffer.clear()
        self._jumping_history.clear()
        s = self._read_state()
        self.prev_score = int(s["score"]) if s else 0
        self._prev_o1_x       = float(s["o1"][0]) if s else None
        self._prev_o1_is_bird = bool(s["o1"][4])  if s else False
        if s is not None:
            self.frame_buffer.append(self._featurize(s))
        return self._stacked_obs()

    def step(self, action: int):
        self._do_action(action)
        time.sleep(self.frame_sleep)
        if self.step_pause > 0:
            time.sleep(self.step_pause)

        s = self._read_state()
        if s is None:
            return self._stacked_obs(), DEATH_REWARD, True, {"score": self.prev_score}

        self.frame_buffer.append(self._featurize(s))
        obs   = self._stacked_obs()
        score = int(s["score"])
        done  = bool(s["crashed"])

        # Record dino's airborne state for phantom-pass detection.
        self._jumping_history.append(int(s["jumping"]))

        # Obstacle-pass detection. Slot-0 obstacle x normally decreases each
        # step as it moves toward the dino. When it crosses the dino and gets
        # removed, slot 0 is occupied by the next (further) obstacle, so
        # cur_o1_x > prev_o1_x — that step is when the pass actually happened.
        # The guards filter out two false positives:
        #   - prev_o1_x < CANVAS_W: previous slot 0 was a real obstacle, not
        #     the "no obstacle" sentinel (9999)
        #   - prev_o1_x > DINO_X: it was actually ahead of the dino (not the
        #     edge case where it spawned behind)
        # Works for both cacti and pterodactyls — slot 0 is type-agnostic.
        cur_o1_x       = float(s["o1"][0])
        cur_o1_is_bird = bool(s["o1"][4])
        passed = (
            self._prev_o1_x is not None
            and self._prev_o1_x < CANVAS_W
            and self._prev_o1_x > DINO_X
            and cur_o1_x > self._prev_o1_x
        )

        # Phantom-pass detection (see module docstring).
        phantom = False
        if passed and not self._prev_o1_is_bird:
            # The obstacle that just left slot 0 was a cactus. A cactus can
            # only be cleared by being airborne — `any(history)` is True iff
            # the dino had jumping=1 at some step in the recent window.
            if not any(self._jumping_history):
                phantom = True
                passed  = False    # suppress the false PASS_BONUS
                done    = True     # restore the death the game missed
                self._phantom_count += 1
        if passed:
            # Real pass — reset the jump window so the next obstacle's check
            # doesn't see residual 1s from the jump that just cleared this one.
            self._jumping_history.clear()

        # Advance prev-tracking for the next step.
        self._prev_o1_x       = cur_o1_x
        self._prev_o1_is_bird = cur_o1_is_bird

        if done:
            reward = DEATH_REWARD
        else:
            reward = ALIVE_REWARD
            if passed:
                reward += PASS_BONUS
            # Jump and duck are both deliberate "vertical" actions: charge for
            # either. Noop is free.
            if action == 1 or action == 2:
                reward -= ACTION_COST

        self.prev_score = score
        return obs, reward, done, {
            "score":         score,
            "speed":         s["speed"],
            "passed":        passed,
            "phantom":       phantom,
            "phantom_count": self._phantom_count,
        }

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Vectorised env. Mirrors VecDinoEnv from the PPO env but also returns the
# per-env action mask in a batched form so the training loop can apply it
# before sampling actions.
# ---------------------------------------------------------------------------

def _grid_positions(n: int, w: int, h: int, cols: int = 2):
    return [((i % cols) * w, (i // cols) * h) for i in range(n)]


class VecDinoEnv:
    def __init__(self,
                 n_envs: int = 4,
                 window_size: tuple = (700, 320),
                 grid_cols: int = 2,
                 **env_kwargs):
        self.n_envs = n_envs
        positions = _grid_positions(n_envs, window_size[0], window_size[1], grid_cols)

        self.envs = []
        for i, pos in enumerate(positions):
            print(f"[vec_env] launching env {i} at {pos} ...")
            self.envs.append(DinoEnv(
                env_id=i,
                window_size=window_size,
                window_position=pos,
                **env_kwargs,
            ))

        self.executor = ThreadPoolExecutor(max_workers=n_envs)

    def _parallel(self, fn, args_per_env):
        futures = [self.executor.submit(fn, env, a)
                   for env, a in zip(self.envs, args_per_env)]
        return [f.result() for f in futures]

    def reset(self):
        results = self._parallel(lambda e, _: e.reset(), [None] * self.n_envs)
        return np.stack(results, axis=0)

    def masks(self):
        """Returns (N, N_ACTIONS) bool array for the current state of each env."""
        return np.stack([e.action_mask() for e in self.envs], axis=0)

    def step(self, actions):
        """Returns (obs, rewards, dones, infos, next_masks).

        `next_masks` is the action mask for the *post-step* state — i.e. the
        state the agent's next action will be conditioned on. For envs that
        terminated and auto-reset, this is the mask of the FRESH state.
        `infos[i]["terminal_obs"]` holds the pre-reset obs (we don't bother
        with terminal_mask: at done=True the Bellman target ignores Q(s_next)
        anyway, so the next-state mask we store in the replay buffer is the
        post-reset mask, which is harmless because gamma * mask cancels).
        """
        outs = self._parallel(lambda e, a: e.step(a), list(actions))
        obs, rewards, dones, infos = zip(*outs)
        obs     = list(obs)
        rewards = np.array(rewards, dtype=np.float32)
        dones   = np.array(dones,   dtype=bool)
        infos   = list(infos)

        reset_idxs = [i for i, d in enumerate(dones) if d]
        if reset_idxs:
            reset_futs = {i: self.executor.submit(self.envs[i].reset)
                          for i in reset_idxs}
            for i, fut in reset_futs.items():
                infos[i]["terminal_obs"] = obs[i]
                obs[i] = fut.result()

        next_masks = self.masks()
        return np.stack(obs, axis=0), rewards, dones, infos, next_masks

    def close(self):
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=False)
