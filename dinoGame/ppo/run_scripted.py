"""
Benchmark the hand-coded scripted teacher (run before BC warmup).

    conda run --no-capture-output -n dinoGame python run_scripted.py --headless --episodes 20

Target: mean score well above 600 on normal starts; duck > 0 when birds appear.
"""
import argparse
import time

import numpy as np

from dino_env import DinoEnv, N_ACTIONS
from scripted_teacher import scripted_action, new_memory


def run_episodes(env, n_episodes: int):
    scores, speeds = [], []
    actions = np.zeros(N_ACTIONS, dtype=np.int64)
    deaths = {}
    mem = new_memory()

    for ep in range(1, n_episodes + 1):
        obs = env.reset()
        mem = new_memory()
        done = False
        steps = 0
        last_info = {}
        t0 = time.time()

        while not done and steps < 5000:
            a = scripted_action(obs, info=last_info if last_info else None, memory=mem)
            actions[a] += 1
            obs, _, done, last_info = env.step(a)
            steps += 1

        score = int(last_info.get("score", 0))
        spd = float(last_info.get("speed", 0))
        cause = str(last_info.get("death_obstacle", "") or "unknown")
        deaths[cause] = deaths.get(cause, 0) + 1
        scores.append(score)
        speeds.append(spd)
        wall = time.time() - t0
        tot = max(1, actions.sum())
        print(
            f"  ep {ep:3d}/{n_episodes}  score {score:5d}  steps {steps:4d}  "
            f"final_spd {spd:.1f}  death={cause}  "
            f"n/j/d {actions[0]/tot:.2f}/{actions[1]/tot:.2f}/{actions[2]/tot:.2f}  "
            f"{wall:.1f}s"
        )

    return scores, actions, deaths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--chromedriver", type=str, default=None)
    p.add_argument("--game-url", type=str, default=None)
    p.add_argument("--curriculum-prob", type=float, default=0.0)
    p.add_argument("--start-speed-min", type=float, default=8.5)
    p.add_argument("--start-speed-max", type=float, default=9.5)
    p.add_argument("--start-score", type=float, default=450.0)
    args = p.parse_args()

    print(
        f"[scripted] episodes={args.episodes} curriculum_prob={args.curriculum_prob}"
    )
    env = DinoEnv(
        headless=args.headless,
        chromedriver_path=args.chromedriver,
        game_url=args.game_url,
        curriculum_prob=args.curriculum_prob,
        start_speed_range=(args.start_speed_min, args.start_speed_max),
        start_score=args.start_score,
    )
    try:
        scores, actions, deaths = run_episodes(env, args.episodes)
    finally:
        env.close()

    n = len(scores)
    tot = max(1, int(actions.sum()))
    print("\n=== Summary ===")
    print(f"  scores: mean={np.mean(scores):.1f}  std={np.std(scores):.1f}  "
          f"median={np.median(scores):.1f}  max={max(scores)}  min={min(scores)}")
    print(f"  actions: noop={actions[0]} jump={actions[1]} duck={actions[2]}  "
          f"duck%={100*actions[2]/tot:.2f}")
    print(f"  deaths: {deaths}")
    if args.curriculum_prob == 0.0:
        if np.mean(scores) < 500:
            print("  VERDICT: teacher too weak for BC — tune scripted_teacher.py")
        elif np.mean(scores) >= 550:
            print("  VERDICT: teacher is strong enough for BC warmup.")
    if actions[2] == 0 and args.curriculum_prob > 0:
        print("  NOTE: 0 duck actions — try --curriculum-prob 0.5 to hit more birds.")


if __name__ == "__main__":
    main()
