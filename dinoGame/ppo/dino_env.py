"""
Selenium-driven Chrome Dino environment — wall-clock frame timing.

State is extracted via JavaScript from Runner.instance_ on every step.
Actions are dispatched as synthetic keyboard events. Timing works as follows:

    step(action):
        1. dispatch the action's key event synchronously
        2. sleep for (frames_per_step / 60) seconds
           — at 60 Hz that is exactly frames_per_step game frames
        3. read state from Runner.instance_

This is simpler and more robust than trying to pause/resume the rAF loop.
The ±1 frame uncertainty is irrelevant for RL purposes.

Action space (Discrete 3):
    0  no-op   — release any held duck key
    1  jump    — tap SPACE (full jump arc plays out over subsequent frames)
    2  duck    — hold DOWN until next non-duck action
"""

import os
import time
from collections import deque

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

# Local t-rex-runner repo sitting next to this file.
# Using file:// avoids chrome:// privilege restrictions on key event injection.
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
# Keyboard event JS snippets.
#
# Chrome 148+ silently ignores `keyCode` in KeyboardEventInit unless `key`,
# `code`, and `which` are ALL also provided. Without them `e.keyCode` is 0
# and Runner.keycodes.JUMP[0] is undefined — the game never reacts.
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


FEATURES_PER_FRAME = 12
FRAME_STACK        = 4
OBS_DIM            = FEATURES_PER_FRAME * FRAME_STACK
N_ACTIONS          = 3

CANVAS_W   = 600.0
CANVAS_H   = 150.0
SPEED_MAX  = 13.0
DEATH_REWARD = -10.0
ALIVE_REWARD =  0.0   # survival bonus removed — was causing credit assignment failure
JUMP_COST    =  0.01  # small penalty per jump — must stay well below PASS_BONUS so
                      # a successful jump is clearly net-positive even when bracketed
                      # by a few wasted attempts
PASS_BONUS   =  1.0   # reward for an obstacle sliding past the dino — must clearly
                      # exceed JUMP_COST so a successful jump is net-positive

# --- Opt-in duck-finetune reward shaping ----------------------------------
# These are ONLY applied when DinoEnv(duck_shaping=True). With the default
# (duck_shaping=False) the reward computed in step() is byte-for-byte identical
# to the original, so the shipped model's training/eval path is unchanged.
DUCK_PASS_BONUS   = 0.5    # extra reward when an obstacle passes while the dino is
                          # DUCKING. Ducking past a bird therefore pays
                          # PASS_BONUS + DUCK_PASS_BONUS = 1.5 — strictly better than
                          # running under it (1.0) or jumping into it (death, -10).
BIRD_JUMP_PENALTY = 0.5    # penalty for JUMPING while a mid/high pterodactyl is in the
                          # danger zone (a jump rises straight into a high bird and
                          # kills you). Kept below PASS_BONUS so normal jumps stay net
                          # positive.
BIRD_DUCK_YMAX    = 80.0   # raw pterodactyl yPos at/below which ducking (not jumping)
                          # is the correct response. Mid bird yPos=75 and high bird
                          # yPos=50 qualify; the low bird (yPos=100) and all cacti do
                          # NOT, so those still want a jump.

# Obstacle-distance normalisation:
#   raw o1.xPos is the obstacle's screen x (0..600). The dino sits at x≈44.
#   The "must decide now" window is roughly 0..200 px ahead of the dino —
#   that's the only range where the agent's choice matters. We map it onto
#   [0.0, 1.0] (clipped at 1.5 for obstacles that are still far away) so the
#   network has full resolution where the timing decision actually lives.
DINO_X       = 44.0
DANGER_RANGE = 200.0


class DinoEnv:
    def __init__(self,
                 chromedriver_path: str = None,
                 headless: bool = False,
                 frames_per_step: int = 4,
                 step_pause: float = 0.0,
                 game_url: str = None,
                 window_size: tuple = (700, 320),
                 window_position: tuple = (0, 0),
                 env_id: int = 0,
                 curriculum_prob: float = 0.0,
                 start_speed_range: tuple = None,
                 start_score: float = 0.0,
                 duck_shaping: bool = False):
        """
        frames_per_step : game frames that elapse per agent step (action repeat).
                          At 60 Hz, 4 frames = 67 ms of game time per step.
        step_pause      : extra sleep after each step — only for slow-motion
                          watching during inference. 0 for training.
        window_size     : (w, h) of the browser window. 700x320 fits 4 in a
                          1920x1080 monitor (2x2 grid).
        window_position : (x, y) screen pixel of top-left corner.
        env_id          : tag for log prints — useful when running many envs.

        --- Opt-in duck-finetune knobs (all default to the original behaviour) ---
        curriculum_prob   : probability that reset() jump-starts the episode into
                            pterodactyl territory (birds need speed >= 8.5). 0.0
                            disables the curriculum entirely (default).
        start_speed_range : (lo, hi) game-speed range sampled on a curriculum start.
        start_score       : in-game score to seed on a curriculum start (the game's
                            distanceRan is set to score / COEFFICIENT so the score
                            readout is consistent). 0 leaves distance untouched.
        duck_shaping      : when True, add duck-targeted reward shaping in step()
                            (DUCK_PASS_BONUS / BIRD_JUMP_PENALTY). Default False =>
                            original reward.
        """
        self.env_id = env_id
        self.curriculum_prob   = curriculum_prob
        self.start_speed_range = start_speed_range
        self.start_score       = start_score
        self.duck_shaping      = duck_shaping
        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-infobars")
        opts.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        opts.add_argument(f"--window-position={window_position[0]},{window_position[1]}")
        # Prevent Chrome from throttling rAF when a window is occluded /
        # backgrounded — important when running multiple instances side by side.
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
        self.frame_sleep = frames_per_step / 60.0   # seconds per step
        self.step_pause  = step_pause

        url = game_url or _LOCAL_GAME
        print(f"[dino_env {self.env_id}] Loading: {url}")
        try:
            self.driver.get(url)
        except Exception:
            # chrome://dino raises WebDriverException — expected and harmless.
            pass

        # Give window.onload time to fire and create the Runner object.
        time.sleep(1.0)

        self._wait_for_runner_constructor(timeout=10.0)
        self._start_game()
        self._wait_for_runner_instance(timeout=5.0)

        self.body         = self.driver.find_element(By.TAG_NAME, "body")
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        self.prev_score   = 0
        self._held_down   = False
        self._prev_o1_x   = None   # for detecting "obstacle passed the dino"

    # ---------------------------------------------------------------- init helpers

    def _wait_for_runner_constructor(self, timeout: float):
        """Wait until the Runner class is defined on the page (no keypress needed)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.driver.execute_script("return typeof Runner !== 'undefined';"):
                    print(f"[dino_env {self.env_id}] Runner constructor ready.")
                    return
            except Exception:
                pass
            time.sleep(0.3)
        raise RuntimeError(
            "Runner constructor not found after {:.0f}s.\n"
            "Fixes to try:\n"
            "  1. Confirm Chrome (not Edge/Firefox) is installed.\n"
            "  2. Open chrome://dino manually and check the game appears.\n"
            "  3. Use a local copy: --game-url file:///path/to/t-rex-runner/index.html\n"
            "     (clone https://github.com/wayou/t-rex-runner)".format(timeout)
        )

    def _start_game(self):
        """Press SPACE once to kick the game loop into existence."""
        print(f"[dino_env {self.env_id}] Sending first keypress to start Runner ...")
        self.driver.execute_script(_JS_SPACE_DOWN + _JS_SPACE_UP)
        time.sleep(0.5)

    def _wait_for_runner_instance(self, timeout: float):
        """Wait until Runner.instance_ is non-null (game loop has ticked once)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.driver.execute_script("return Runner.instance_ != null;"):
                    print(f"[dino_env {self.env_id}] Runner.instance_ ready. Init done.")
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(
            "Runner.instance_ still null after {:.0f}s.\n"
            "The SPACE keypress may not have reached the page.\n"
            "Try the local file fallback:\n"
            "  python train.py --game-url file:///path/to/t-rex-runner/index.html"
            .format(timeout)
        )

    # ---------------------------------------------------------------- state

    def _read_state(self):
        try:
            return self.driver.execute_script("""
        const r = Runner.instance_;
        if (!r) return null;
        const t   = r.tRex;
        const obs = r.horizon.obstacles || [];
        const o1  = obs[0] || null;
        const o2  = obs[1] || null;
        function pack(o) {
            // No obstacle → sentinel that normalises to "very far away" (dx ≥ 1.5),
            // NOT 1.0 which would normalise to "touching the dino" and trigger
            // a panic jump every time the screen is empty.
            if (!o) return [9999, 0.0, 0.0, 0.0];
            return [o.xPos, o.yPos, o.width, o.typeConfig.height];
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
            o1type  : o1 ? o1.typeConfig.type : '',
            o2type  : o2 ? o2.typeConfig.type : '',
        };
        """)
        except Exception:
            return None

    def _read_state_retry(self, attempts=8, pause=0.03):
        """Poll Runner.instance_ — avoids false terminal steps after restart."""
        for i in range(attempts):
            s = self._read_state()
            if s is not None:
                return s
            if i + 1 < attempts:
                time.sleep(pause)
        return None

    def _wait_until_ready(self, timeout=2.5):
        """Block until the game is running and not crashed (post-reset)."""
        end = time.time() + timeout
        last = None
        while time.time() < end:
            s = self._read_state()
            if s is not None:
                last = s
                if not s["crashed"] and s.get("playing"):
                    return s
            time.sleep(0.05)
        return last

    def _is_duckable_bird_near(self, s):
        """True if the nearest obstacle is a mid/high pterodactyl inside the
        danger zone — i.e. a bird you must DUCK (or run) under, not jump into.
        Used only by the opt-in duck_shaping reward."""
        o1 = s["o1"]
        x, ypos = float(o1[0]), float(o1[1])
        return (s.get("o1type", "") == "PTERODACTYL"
                and ypos <= BIRD_DUCK_YMAX
                and DINO_X < x <= DINO_X + DANGER_RANGE)

    def _featurize(self, s):
        o1, o2 = s["o1"], s["o2"]
        # Obstacle x is normalised relative to the dino, not the canvas — see
        # DINO_X / DANGER_RANGE comment above. 0.0 = touching the dino,
        # 1.0 = at the edge of the "must decide now" zone, 1.5 = far away.
        def dx_norm(xpos):
            return min(max(0.0, xpos - DINO_X) / DANGER_RANGE, 1.5)
        return np.array([
            s["dinoY"] / CANVAS_H,
            float(s["jumping"]),
            float(s["ducking"]),
            min(s["speed"], SPEED_MAX) / SPEED_MAX,
            dx_norm(o1[0]), o1[1] / CANVAS_H, o1[2] / CANVAS_W, o1[3] / CANVAS_H,
            dx_norm(o2[0]), o2[1] / CANVAS_H, o2[2] / CANVAS_W, o2[3] / CANVAS_H,
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
        if action == 0:                    # no-op
            self._release_down()
        elif action == 1:                  # jump
            self._release_down()
            self.driver.execute_script(_JS_SPACE_DOWN + _JS_SPACE_UP)
        elif action == 2:                  # duck (hold)
            if not self._held_down:
                self.driver.execute_script(_JS_DOWN_DOWN)
                self._held_down = True

    # ---------------------------------------------------------------- public API

    def reset(self):
        self._release_down()
        s = None
        for attempt in range(3):
            self.driver.execute_script(
                "const r = Runner.instance_;"
                "if (!r) return;"
                "if (r.crashed) { r.restart(); }"
                "else if (!r.playing) {"
                "  " + _JS_SPACE_DOWN +
                "  " + _JS_SPACE_UP +
                "}"
            )
            s = self._wait_until_ready(timeout=2.0)
            if s is not None and not s["crashed"] and s.get("playing"):
                break
            time.sleep(0.15 * (attempt + 1))

        # Opt-in speed curriculum: with probability curriculum_prob, jump-start
        # this episode into pterodactyl territory so the agent actually sees
        # birds (which only spawn once game speed reaches 8.5). Obstacle
        # generation keys off currentSpeed, so setting it is enough; distanceRan
        # is set too only to keep the score readout consistent. With the default
        # curriculum_prob=0 this whole block is skipped and reset() is unchanged.
        if (self.curriculum_prob > 0.0
                and self.start_speed_range is not None
                and np.random.random() < self.curriculum_prob):
            lo, hi = self.start_speed_range
            speed = float(np.random.uniform(lo, hi))
            self.driver.execute_script(
                "const r = Runner.instance_;"
                "if (r) {"
                "  r.setSpeed(arguments[0]);"
                "  if (arguments[1] > 0) {"
                "    r.distanceRan = arguments[1] / r.distanceMeter.config.COEFFICIENT;"
                "  }"
                "}",
                speed, float(self.start_score),
            )
            time.sleep(0.05)

        self.frame_buffer.clear()
        if s is None:
            s = self._read_state_retry()
        self.prev_score = int(s["score"]) if s else 0
        self._prev_o1_x = float(s["o1"][0]) if s else None
        if s is not None:
            self.frame_buffer.append(self._featurize(s))
        return self._stacked_obs()

    def step(self, action: int):
        # 1. dispatch action
        self._do_action(action)

        # 2. let the game run for exactly frames_per_step frames
        time.sleep(self.frame_sleep)

        # 3. optional extra pause (inference / human viewing only)
        if self.step_pause > 0:
            time.sleep(self.step_pause)

        # 4. read state (retry — Runner.instance_ can be briefly null after restart)
        s = self._read_state_retry()
        if s is None:
            return self._stacked_obs(), DEATH_REWARD, True, {
                "score": self.prev_score,
                "read_state_failed": True,
            }

        self.frame_buffer.append(self._featurize(s))
        obs   = self._stacked_obs()
        score = int(s["score"])
        done  = bool(s["crashed"])

        # Obstacle-pass detection: front obstacle's xPos crossed the dino this
        # step (was ahead last frame, behind or gone this frame). Awarding this
        # is what makes a successful jump net-positive — distance reward alone
        # barely covers JUMP_COST.
        # An obstacle "passes" when the front-obstacle x-coordinate jumps
        # backwards (because the old one was removed and the next one is now
        # in slot 0 — further away). Detecting prev > DINO_X gates out the
        # initial frames where there's no obstacle yet.
        cur_o1_x = float(s["o1"][0])
        passed = (
            self._prev_o1_x is not None
            and self._prev_o1_x < CANVAS_W       # there *was* a real obstacle on screen
            and self._prev_o1_x > DINO_X         # ...and it was ahead of us
            and cur_o1_x > self._prev_o1_x       # ...and now slot 0 is a further obstacle (or empty sentinel)
        )
        self._prev_o1_x = cur_o1_x

        if done:
            reward = DEATH_REWARD
        else:
            reward = (
                0.01 * max(0, score - self.prev_score)
                - (JUMP_COST if action == 1 else 0.0)
                + (PASS_BONUS if passed else 0.0)
            )
            # Opt-in duck shaping (no-op unless duck_shaping=True).
            if self.duck_shaping:
                if passed and bool(s["ducking"]):
                    reward += DUCK_PASS_BONUS          # rewarded the correct bird response
                if action == 1 and self._is_duckable_bird_near(s):
                    reward -= BIRD_JUMP_PENALTY        # discouraged jumping into a high bird
        self.prev_score = score

        info = {
            "score": score,
            "speed": s["speed"],
            "o1_type": s.get("o1type", ""),
            "o2_type": s.get("o2type", ""),
            "o1_x": cur_o1_x,
            "o1_y": float(s["o1"][1]),
            "o2_x": float(s["o2"][0]),
            "o2_y": float(s["o2"][1]),
        }
        if done:
            # Tag the obstacle we most likely died on, for death-cause analysis.
            info["death_obstacle"] = s.get("o1type", "")
        return obs, reward, done, info

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Vectorised env — runs N DinoEnv instances in parallel via threads.
# Selenium calls are I/O-bound (waiting on browser round-trips), so a
# ThreadPoolExecutor is enough — the GIL is released during execute_script
# and time.sleep. No need for multiprocessing.
# ---------------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor


def _grid_positions(n: int, w: int, h: int, cols: int = 2):
    """Return [(x, y), ...] for n windows arranged in a `cols`-wide grid."""
    return [((i % cols) * w, (i // cols) * h) for i in range(n)]


class VecDinoEnv:
    """A batched DinoEnv. step() takes a length-N action vector and returns
    arrays of shape (N, ...). Auto-resets terminated envs so the training
    loop always sees a fresh next-obs."""

    def __init__(self,
                 n_envs: int = 4,
                 window_size: tuple = (700, 320),
                 grid_cols: int = 2,
                 **env_kwargs):
        self.n_envs = n_envs
        positions = _grid_positions(n_envs, window_size[0], window_size[1], grid_cols)

        # Construct envs sequentially — Chrome doesn't like a stampede of
        # simultaneous driver starts, and webdriver_manager isn't safe to call
        # from multiple threads at once.
        self.envs = []
        for i, pos in enumerate(positions):
            print(f"[vec_env] launching env {i} at position {pos} ...")
            self.envs.append(DinoEnv(
                env_id=i,
                window_size=window_size,
                window_position=pos,
                **env_kwargs,
            ))

        # One thread per env. Each step submits N calls in parallel; threads
        # block in execute_script and time.sleep, releasing the GIL.
        self.executor = ThreadPoolExecutor(max_workers=n_envs)

    def _parallel(self, fn, args_per_env):
        futures = [self.executor.submit(fn, env, a)
                   for env, a in zip(self.envs, args_per_env)]
        return [f.result() for f in futures]

    def reset(self):
        results = self._parallel(lambda e, _: e.reset(), [None] * self.n_envs)
        return np.stack(results, axis=0)              # (N, obs_dim)

    def step(self, actions):
        """actions: array of length N. Returns (obs, rewards, dones, infos)
        with shapes (N, obs_dim), (N,), (N,), list[N]. Auto-resets any env
        that terminated this step — the returned obs[i] is then the *fresh*
        observation, and info[i]['terminal_obs'] holds the pre-reset one."""
        outs = self._parallel(lambda e, a: e.step(a), list(actions))
        obs, rewards, dones, infos = zip(*outs)
        obs     = list(obs)
        rewards = np.array(rewards, dtype=np.float32)
        dones   = np.array(dones,   dtype=bool)
        infos   = list(infos)

        # Auto-reset any terminated env so the next step has a valid obs.
        reset_idxs = [i for i, d in enumerate(dones) if d]
        if reset_idxs:
            reset_futs = {i: self.executor.submit(self.envs[i].reset)
                          for i in reset_idxs}
            for i, fut in reset_futs.items():
                infos[i]["terminal_obs"] = obs[i]
                obs[i] = fut.result()

        return np.stack(obs, axis=0), rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=False)
