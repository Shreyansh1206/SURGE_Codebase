"""
Watchdog: if the overnight pipeline dies, resume it automatically.
Run in background alongside (or after) run_overnight.py:

    conda run --no-capture-output -n dinoGame python -u watch_pipeline.py
"""
import os
import subprocess
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(ROOT, "checkpoints_scratch_duck")
LOG = os.path.join(SAVE_DIR, "watch.log")
DONE = os.path.join(SAVE_DIR, "pipeline_done.txt")
POLL_SEC = 300   # 5 min


def wlog(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pipeline_running() -> bool:
    try:
        out = subprocess.check_output(
            ["powershell", "-Command",
             "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
             "Select-Object -ExpandProperty CommandLine"],
            text=True, errors="replace",
        )
        return "run_overnight.py" in out
    except Exception:
        return False


def main():
    wlog("watchdog started")
    while True:
        if os.path.isfile(DONE):
            wlog("pipeline_done.txt found — watchdog exit")
            break
        if not pipeline_running():
            wlog("pipeline not running — launching run_overnight.py (resume)")
            subprocess.Popen(
                ["conda", "run", "--no-capture-output", "-n", "dinoGame",
                 "python", "-u", "run_overnight.py"],
                cwd=ROOT,
            )
            time.sleep(60)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
