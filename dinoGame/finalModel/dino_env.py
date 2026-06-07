import os
import time
from collections import deque

import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

_LOCAL_GAME = "file:///" + os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "t-rex-runner", "index.html"
).replace("\\", "/")
try:
    from webdriver_manager.chrome import ChromeDriverManager

    _HAS_WDM = True
except ImportError:
    _HAS_WDM = False
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
FRAME_STACK = 4
OBS_DIM = FEATURES_PER_FRAME * FRAME_STACK
N_ACTIONS = 3
CANVAS_W = 600.0
CANVAS_H = 150.0
SPEED_MAX = 13.0
DEATH_REWARD = -10.0
ALIVE_REWARD = 0.0
JUMP_COST = 0.01
PASS_BONUS = 1.0
DUCK_PASS_BONUS = 0.5
BIRD_JUMP_PENALTY = 0.5
BIRD_DUCK_YMAX = 80.0
DINO_X = 44.0
DANGER_RANGE = 200.0


class DinoEnv:
    def __init__(
        self,
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
        duck_shaping: bool = False,
    ):
        self.env_id = env_id
        self.curriculum_prob = curriculum_prob
        self.start_speed_range = start_speed_range
        self.start_score = start_score
        self.duck_shaping = duck_shaping
        opts = Options()
        opts.add_argument("--mute-audio")
        opts.add_argument("--disable-infobars")
        opts.add_argument(f"--window-size={window_size[0]},{window_size[1]}")
        opts.add_argument(
            f"--window-position={window_position[0]},{window_position[1]}"
        )
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
        self.step_pause = step_pause
        url = game_url or _LOCAL_GAME
        print(f"[dino_env {self.env_id}] Loading: {url}")
        try:
            self.driver.get(url)
        except Exception:
            pass
        time.sleep(1.0)
        self._wait_for_runner_constructor(timeout=10.0)
        self._start_game()
        self._wait_for_runner_instance(timeout=5.0)
        self.body = self.driver.find_element(By.TAG_NAME, "body")
        self.frame_buffer = deque(maxlen=FRAME_STACK)
        self.prev_score = 0
        self._held_down = False
        self._prev_o1_x = None

    def _wait_for_runner_constructor(self, timeout: float):
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
        print(f"[dino_env {self.env_id}] Sending first keypress to start Runner ...")
        self.driver.execute_script(_JS_SPACE_DOWN + _JS_SPACE_UP)
        time.sleep(0.5)

    def _wait_for_runner_instance(self, timeout: float):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if self.driver.execute_script("return Runner.instance_ != null;"):
                    print(
                        f"[dino_env {self.env_id}] Runner.instance_ ready. Init done."
                    )
                    return
            except Exception:
                pass
            time.sleep(0.2)
        raise RuntimeError(
            "Runner.instance_ still null after {:.0f}s.\n"
            "The SPACE keypress may not have reached the page.\n"
            "Try the local file fallback:\n"
            "  python train.py --game-url file:///path/to/t-rex-runner/index.html".format(
                timeout
            )
        )

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
        for i in range(attempts):
            s = self._read_state()
            if s is not None:
                return s
            if i + 1 < attempts:
                time.sleep(pause)
        return None

    def _wait_until_ready(self, timeout=2.5):
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
        o1 = s["o1"]
        x, ypos = float(o1[0]), float(o1[1])
        return (
            s.get("o1type", "") == "PTERODACTYL"
            and ypos <= BIRD_DUCK_YMAX
            and DINO_X < x <= DINO_X + DANGER_RANGE
        )

    def _featurize(self, s):
        o1, o2 = s["o1"], s["o2"]

        def dx_norm(xpos):
            return min(max(0.0, xpos - DINO_X) / DANGER_RANGE, 1.5)

        return np.array(
            [
                s["dinoY"] / CANVAS_H,
                float(s["jumping"]),
                float(s["ducking"]),
                min(s["speed"], SPEED_MAX) / SPEED_MAX,
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

    def reset(self):
        self._release_down()
        s = None
        for attempt in range(3):
            self.driver.execute_script(
                "const r = Runner.instance_;"
                "if (!r) return;"
                "if (r.crashed) { r.restart(); }"
                "else if (!r.playing) {"
                "  " + _JS_SPACE_DOWN + "  " + _JS_SPACE_UP + "}"
            )
            s = self._wait_until_ready(timeout=2.0)
            if s is not None and not s["crashed"] and s.get("playing"):
                break
            time.sleep(0.15 * (attempt + 1))
        if (
            self.curriculum_prob > 0.0
            and self.start_speed_range is not None
            and np.random.random() < self.curriculum_prob
        ):
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
                speed,
                float(self.start_score),
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
        self._do_action(action)
        time.sleep(self.frame_sleep)
        if self.step_pause > 0:
            time.sleep(self.step_pause)
        s = self._read_state_retry()
        if s is None:
            return (
                self._stacked_obs(),
                DEATH_REWARD,
                True,
                {
                    "score": self.prev_score,
                    "read_state_failed": True,
                },
            )
        self.frame_buffer.append(self._featurize(s))
        obs = self._stacked_obs()
        score = int(s["score"])
        done = bool(s["crashed"])
        cur_o1_x = float(s["o1"][0])
        passed = (
            self._prev_o1_x is not None
            and self._prev_o1_x < CANVAS_W
            and self._prev_o1_x > DINO_X
            and cur_o1_x > self._prev_o1_x
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
            if self.duck_shaping:
                if passed and bool(s["ducking"]):
                    reward += DUCK_PASS_BONUS
                if action == 1 and self._is_duckable_bird_near(s):
                    reward -= BIRD_JUMP_PENALTY
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
            info["death_obstacle"] = s.get("o1type", "")
        return obs, reward, done, info

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass


from concurrent.futures import ThreadPoolExecutor


def _grid_positions(n: int, w: int, h: int, cols: int = 2):
    return [((i % cols) * w, (i // cols) * h) for i in range(n)]


class VecDinoEnv:
    def __init__(
        self,
        n_envs: int = 4,
        window_size: tuple = (700, 320),
        grid_cols: int = 2,
        **env_kwargs,
    ):
        self.n_envs = n_envs
        positions = _grid_positions(n_envs, window_size[0], window_size[1], grid_cols)
        self.envs = []
        for i, pos in enumerate(positions):
            print(f"[vec_env] launching env {i} at position {pos} ...")
            self.envs.append(
                DinoEnv(
                    env_id=i,
                    window_size=window_size,
                    window_position=pos,
                    **env_kwargs,
                )
            )
        self.executor = ThreadPoolExecutor(max_workers=n_envs)

    def _parallel(self, fn, args_per_env):
        futures = [
            self.executor.submit(fn, env, a) for env, a in zip(self.envs, args_per_env)
        ]
        return [f.result() for f in futures]

    def reset(self):
        results = self._parallel(lambda e, _: e.reset(), [None] * self.n_envs)
        return np.stack(results, axis=0)

    def step(self, actions):
        outs = self._parallel(lambda e, a: e.step(a), list(actions))
        obs, rewards, dones, infos = zip(*outs)
        obs = list(obs)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=bool)
        infos = list(infos)
        reset_idxs = [i for i, d in enumerate(dones) if d]
        if reset_idxs:
            reset_futs = {
                i: self.executor.submit(self.envs[i].reset) for i in reset_idxs
            }
            for i, fut in reset_futs.items():
                infos[i]["terminal_obs"] = obs[i]
                obs[i] = fut.result()
        return np.stack(obs, axis=0), rewards, dones, infos

    def close(self):
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=False)
