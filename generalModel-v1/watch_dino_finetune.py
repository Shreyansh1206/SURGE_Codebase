
"""
Watchdog: restart Dino fine-tuning if the process crashes.

Keeps resuming from checkpoints_dino_finetune/latest.pt until --updates-total
is reached (read from train_state.json) or you stop this script.

  python watch_dino_finetune.py
  python watch_dino_finetune.py --parallel --updates-per-run 50
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, "checkpoints_dino_finetune")
STATE_PATH = os.path.join(SAVE_DIR, "train_state.json")
LOG_PATH = os.path.join(SAVE_DIR, "watchdog.log")


def wlog(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_update() -> int:
    if not os.path.isfile(STATE_PATH):
        return 0
    with open(STATE_PATH, encoding="utf-8") as f:
        return int(json.load(f).get("update", 0))


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Auto-restart Dino fine-tune on crash")
    p.add_argument("--updates-total", type=int, default=500,
                   help="Stop watchdog when train_state update reaches this")
    p.add_argument("--updates-per-run", type=int, default=25,
                   help="PPO updates per subprocess invocation")
    p.add_argument("--parallel", action="store_true",
                   help="Pass --parallel to train_dino_finetune.py")
    p.add_argument("--sleep", type=float, default=3.0,
                   help="Seconds between restart attempts")
    args = p.parse_args()

    script = os.path.join(ROOT, "train_dino_finetune.py")
    wlog(f"watchdog start | target update {args.updates_total}")

    while True:
        cur = read_update()
        if cur >= args.updates_total:
            wlog(f"target reached (update {cur}) — stopping")
            break

        cmd = [
            sys.executable,
            script,
            "--save-dir",
            "checkpoints_dino_finetune",
            "--updates",
            str(args.updates_per_run),
            "--save-every",
            "5",
        ]
        if args.parallel:
            cmd.extend(["--n-dino-envs", "4"])
        else:
            cmd.append("--no-parallel")

        wlog(f"launching (current update {cur}): {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, cwd=ROOT)
            wlog(f"subprocess exit code {proc.returncode}")
        except Exception as exc:
            wlog(f"subprocess error: {exc}")

        cur = read_update()
        if cur >= args.updates_total:
            wlog(f"target reached after run (update {cur}) — stopping")
            break

        wlog(f"sleep {args.sleep}s then resume")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
