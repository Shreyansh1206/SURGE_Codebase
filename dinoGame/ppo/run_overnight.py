"""
~8h scratch pipeline with resume: BC -> PPO -> 50-ep benchmark.
Logs to checkpoints_scratch_duck/overnight.log

Checkpoints:
  bc_init.pt   — after BC
  best.pt      — best rollout *mean* score during PPO
  best_max.pt  — best single-episode peak in a rollout
  latest.pt    — last PPO update (resume point)
"""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
SAVE_DIR = "checkpoints_scratch_duck"
LOG = os.path.join(SAVE_DIR, "overnight.log")
BC_CKPT = os.path.join(SAVE_DIR, "bc_init.pt")
DONE_MARKER = os.path.join(SAVE_DIR, "pipeline_done.txt")

# ~8h budget knobs
BC_EPISODES = 30
BC_EPOCHS = 25
PPO_UPDATES = 140
BENCH_EPISODES = 50


def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(args: list[str], retries: int = 1) -> int:
    cmd = ["conda", "run", "--no-capture-output", "-n", "dinoGame", "python", "-u"] + args
    for attempt in range(retries + 1):
        log("RUN " + " ".join(cmd) + (f" (attempt {attempt + 1})" if attempt else ""))
        rc = subprocess.call(cmd)
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


def pipeline_finished() -> bool:
    if os.path.isfile(DONE_MARKER):
        return True
    report = os.path.join("benchmarking", "results", "scratch_duck_final", "report.md")
    return os.path.isfile(report)


def main():
    if pipeline_finished():
        log("pipeline already complete — nothing to do")
        return

    log("=== pipeline start (~8h budget, resumable) ===")
    t0 = time.time()

    if not bc_done():
        rc = run([
            "bc_warmup.py", "--headless",
            "--episodes", str(BC_EPISODES),
            "--epochs", str(BC_EPOCHS),
            "--curriculum-prob", "0.5",
            "--save", BC_CKPT,
        ], retries=1)
        if rc != 0:
            log(f"BC warmup FAILED after retries exit={rc}")
            sys.exit(rc)
        log("BC warmup done")
    else:
        log(f"BC skip — reusing {BC_CKPT}")

    resume = train_resume_path()
    log(f"PPO resume from {resume}")
    rc = run([
        "train.py",
        "--save-dir", SAVE_DIR,
        "--resume", resume,
        "--headless", "--n-envs", "4", "--rollout", "256",
        "--updates", str(PPO_UPDATES),
        "--curriculum-prob", "0.35",
        "--entropy", "0.01", "--lr", "2e-4",
        "--save-every", "10",
    ], retries=1)
    if rc != 0:
        log(f"PPO train FAILED exit={rc}")
        sys.exit(rc)
    log("PPO train done")

    best = os.path.join(SAVE_DIR, "best.pt")
    if not os.path.isfile(best):
        log("best.pt missing — copying latest.pt for benchmark")
        import shutil
        shutil.copy(os.path.join(SAVE_DIR, "latest.pt"), best)

    rc = run([
        "benchmarking/benchmark.py",
        "--ckpt", best,
        "--episodes", str(BENCH_EPISODES),
        "--headless",
        "--out-dir", "benchmarking/results/scratch_duck_final",
    ], retries=1)
    if rc != 0:
        log(f"benchmark FAILED exit={rc}")
        sys.exit(rc)

    elapsed = (time.time() - t0) / 3600.0
    with open(DONE_MARKER, "w") as f:
        f.write(f"finished {datetime.now().isoformat()} in {elapsed:.2f}h\n")
    log(f"=== pipeline finished in {elapsed:.2f} h ===")
    log(f"best: {best} | report: benchmarking/results/scratch_duck_final/report.md")


if __name__ == "__main__":
    main()
