"""Sanity check: evaluate given checkpoints with the SAME VecDinoEnv + run_eval
path the finetune uses, to see whether the ~587 baseline survives in this env
config (i.e. whether the early-game collapse is the env/eval or the training)."""
import argparse
import numpy as np
import torch

from ppo_agent import PPO
from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
from finetune_duck import run_eval

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+",
                   default=["checkpoints/best.pt", "checkpoints_duck/duck_init.pt"])
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    vec = VecDinoEnv(n_envs=args.n_envs, window_size=(700, 320), grid_cols=2,
                     headless=args.headless,
                     curriculum_prob=0.3, start_speed_range=(8.5, 9.0),
                     start_score=450.0, duck_shaping=True)
    try:
        for ck in args.ckpts:
            ppo = PPO(OBS_DIM, N_ACTIONS)
            ppo.load(ck, load_optim=False)
            ev = run_eval(vec, ppo, ppo.device, args.episodes, curriculum_prob=0.0)
            bv = run_eval(vec, ppo, ppo.device, args.episodes, curriculum_prob=1.0)
            print(f"\n=== {ck} ===")
            print(f"  BASE  : mean {ev['mean_score']:.1f} max {ev['max_score']} "
                  f"| duck {ev['duck_frac']*100:.1f}% jump {ev['jump_frac']*100:.1f}% "
                  f"noop {ev['noop_frac']*100:.1f}% | scores {ev['scores']}")
            print(f"  BIRDS : mean {bv['mean_score']:.1f} max {bv['max_score']} "
                  f"| duck {bv['duck_frac']*100:.1f}% jump {bv['jump_frac']*100:.1f}% "
                  f"noop {bv['noop_frac']*100:.1f}% | scores {bv['scores']}")
    finally:
        vec.close()
