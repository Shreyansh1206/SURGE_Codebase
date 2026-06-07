import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
SAVE_DIR = "checkpoints"
LOG = os.path.join(SAVE_DIR, "pipeline.log")
BC_CKPT = os.path.join(SAVE_DIR, "bc_init.pt")
BEST_PPO = os.path.join(SAVE_DIR, "best.pt")
BEST_DUCK = os.path.join(SAVE_DIR, "best_duck.pt")
DONE_MARKER = os.path.join(SAVE_DIR, "pipeline_done.txt")
BC_EPISODES = 30
BC_EPOCHS = 25
PPO_UPDATES = 140
DISTILL_ITERS = 400
BENCH_EPISODES = 30


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def python_cmd() -> list[str]:
    if shutil.which("conda"):
        return ["conda", "run", "--no-capture-output", "-n", "dinoGame", "python", "-u"]
    return [sys.executable, "-u"]


def run(script_args: list[str], retries: int = 1) -> int:
    cmd = python_cmd() + script_args
    for attempt in range(retries + 1):
        log("RUN " + " ".join(cmd) + (f" (attempt {attempt + 1})" if attempt else ""))
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc == 0:
            return 0
        log(f"command failed exit={rc}")
        if attempt < retries:
            log("retrying in 30s ...")
            time.sleep(30)
    return rc


def bc_done() -> bool:
    return os.path.isfile(BC_CKPT) and os.path.getsize(BC_CKPT) > 50_000


def train_resume_path() -> str:
    latest = os.path.join(SAVE_DIR, "latest.pt")
    if os.path.isfile(latest) and os.path.getsize(latest) > 50_000:
        return latest
    return BC_CKPT


def ppo_done() -> bool:
    return os.path.isfile(BEST_PPO) and os.path.getsize(BEST_PPO) > 50_000


def distill_done() -> bool:
    return os.path.isfile(BEST_DUCK) and os.path.getsize(BEST_DUCK) > 50_000


def pipeline_finished() -> bool:
    if os.path.isfile(DONE_MARKER):
        return True
    return os.path.isfile(os.path.join("results", "pipeline_final", "report.md"))


def main():
    if pipeline_finished():
        log("pipeline already complete — nothing to do")
        return
    log("=== finalModel pipeline start (resumable) ===")
    t0 = time.time()
    if not bc_done():
        rc = run(
            [
                "bc_warmup.py",
                "--headless",
                "--episodes",
                str(BC_EPISODES),
                "--epochs",
                str(BC_EPOCHS),
                "--curriculum-prob",
                "0.5",
                "--save",
                BC_CKPT,
            ],
            retries=1,
        )
        if rc != 0:
            sys.exit(rc)
        log("BC warmup done")
    else:
        log(f"BC skip — reusing {BC_CKPT}")
    if not ppo_done():
        resume = train_resume_path()
        log(f"PPO resume from {resume}")
        rc = run(
            [
                "train.py",
                "--save-dir",
                SAVE_DIR,
                "--resume",
                resume,
                "--headless",
                "--n-envs",
                "4",
                "--rollout",
                "256",
                "--updates",
                str(PPO_UPDATES),
                "--curriculum-prob",
                "0.35",
                "--entropy",
                "0.01",
                "--lr",
                "2e-4",
                "--save-every",
                "10",
            ],
            retries=1,
        )
        if rc != 0:
            sys.exit(rc)
        if not os.path.isfile(BEST_PPO):
            shutil.copy(os.path.join(SAVE_DIR, "latest.pt"), BEST_PPO)
        log("PPO train done")
    else:
        log(f"PPO skip — reusing {BEST_PPO}")
    if not distill_done():
        init = (
            BEST_PPO
            if os.path.isfile(BEST_PPO)
            else os.path.join(SAVE_DIR, "scratch_best.pt")
        )
        rc = run(
            [
                "make_duck_distill.py",
                "--init",
                init,
                "--out",
                BEST_DUCK,
                "--iters",
                str(DISTILL_ITERS),
            ],
            retries=1,
        )
        if rc != 0:
            sys.exit(rc)
        log("duck distill done")
    else:
        log(f"distill skip — reusing {BEST_DUCK}")
    rc = run(
        [
            "benchmark.py",
            "--ckpt",
            BEST_DUCK,
            "--episodes",
            str(BENCH_EPISODES),
            "--headless",
            "--out-dir",
            "results/pipeline_final",
        ],
        retries=1,
    )
    if rc != 0:
        sys.exit(rc)
    elapsed = (time.time() - t0) / 3600.0
    with open(DONE_MARKER, "w", encoding="utf-8") as f:
        f.write(f"finished {datetime.now().isoformat()} in {elapsed:.2f}h\n")
    log(f"=== pipeline finished in {elapsed:.2f} h ===")
    log(f"model: {BEST_DUCK} | report: results/pipeline_final/report.md")


if __name__ == "__main__":
    main()
