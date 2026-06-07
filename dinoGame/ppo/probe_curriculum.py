"""Quick diagnostic: does the speed curriculum actually produce pterodactyls,
and how many duckable-bird states do we see per N steps? Run headless, ~1 min.

    conda run -n dinoGame python probe_curriculum.py
"""
import numpy as np
from dino_env import DinoEnv, FEATURES_PER_FRAME, BIRD_DUCK_YMAX
from scripted_teacher import scripted_action

def main():
    env = DinoEnv(headless=True, curriculum_prob=1.0,
                  start_speed_range=(8.5, 9.5), start_score=450.0,
                  duck_shaping=True)
    n_steps = 1500
    obs = env.reset()
    types = {}
    bird_steps = 0
    duckable_steps = 0
    speeds = []
    deaths = 0
    for _ in range(n_steps):
        a = scripted_action(obs)
        obs, r, done, info = env.step(a)
        t = info.get("o1_type", "")
        types[t] = types.get(t, 0) + 1
        speeds.append(info.get("speed", 0.0))
        if t == "PTERODACTYL":
            bird_steps += 1
        # duckable = mid/high bird in danger zone (read from latest frame)
        f = obs[-FEATURES_PER_FRAME:]
        if t == "PTERODACTYL" and f[5] * 150.0 <= BIRD_DUCK_YMAX and f[4] < 0.65:
            duckable_steps += 1
        if done:
            deaths += 1
            obs = env.reset()
    env.close()
    print(f"steps={n_steps} deaths={deaths}")
    print(f"speed: min={min(speeds):.1f} mean={np.mean(speeds):.1f} max={max(speeds):.1f}")
    print(f"obstacle-type step counts: {types}")
    print(f"pterodactyl steps = {bird_steps} ({bird_steps/n_steps*100:.1f}%)")
    print(f"duckable-bird steps = {duckable_steps} ({duckable_steps/n_steps*100:.1f}%)")

if __name__ == "__main__":
    main()
