"""
Quick diagnostic of a trained checkpoint. Tells you:
- action distribution (is it stuck on no-op?)
- entropy of the policy on real observations
- mean/std of values predicted
- shape of the training log (mean_score, entropy over time)
"""
import argparse, json, os
import numpy as np
import torch

from dino_env import DinoEnv, OBS_DIM, N_ACTIONS
from ppo_agent import PPO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--log", type=str, default="checkpoints/train_log.jsonl")
    args = p.parse_args()

    # ---------- log shape ----------
    if os.path.exists(args.log):
        with open(args.log) as f:
            rows = [json.loads(l) for l in f]
        print(f"\n=== Training log ({len(rows)} updates) ===")
        for r in rows[::max(1, len(rows)//10)]:
            print(f"  upd {r['update']:3d}  score avg {r['mean_score']:5.1f} "
                  f"max {r['max_score']:4d}  H {r['entropy']:.3f}  "
                  f"pi {r['pi_loss']:+.3f}  v {r['v_loss']:.3f}")
        # last few
        print("  ----- last 5 -----")
        for r in rows[-5:]:
            print(f"  upd {r['update']:3d}  score avg {r['mean_score']:5.1f} "
                  f"max {r['max_score']:4d}  H {r['entropy']:.3f}")
    else:
        print(f"  [no log at {args.log}]")

    # ---------- live policy probe ----------
    print("\n=== Live policy probe ===")
    env = DinoEnv()
    ppo = PPO(OBS_DIM, N_ACTIONS); ppo.load(args.ckpt, load_optim=False); ppo.net.eval()

    obs = env.reset()
    action_counts = np.zeros(3, dtype=np.int64)
    all_probs, all_values = [], []
    deaths = 0

    for _ in range(args.steps):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(ppo.device)
        with torch.no_grad():
            logits, value = ppo.net(obs_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        a = int(np.random.choice(3, p=probs))
        action_counts[a] += 1
        all_probs.append(probs)
        all_values.append(float(value.item()))
        obs, _, done, _ = env.step(a)
        if done:
            deaths += 1
            obs = env.reset()

    env.close()

    probs_arr = np.stack(all_probs)         # (steps, 3)
    print(f"  action counts (noop/jump/duck): {action_counts.tolist()}")
    print(f"  action fractions             : {action_counts / action_counts.sum()}")
    print(f"  mean policy probs            : {probs_arr.mean(0)}")
    print(f"  mean entropy (live)          : {(-probs_arr*np.log(probs_arr+1e-8)).sum(-1).mean():.3f}")
    print(f"  value pred mean/std          : {np.mean(all_values):+.2f} / {np.std(all_values):.2f}")
    print(f"  deaths in {args.steps} steps : {deaths}")

    # Verdict
    print()
    if action_counts[1] < args.steps * 0.05:
        print("  [VERDICT] jump action is almost never picked — policy collapsed to no-op.")
        print("            Fix: drop ALIVE_REWARD to 0 and/or raise --entropy to 0.05.")
    elif probs_arr.std(0).mean() < 0.05:
        print("  [VERDICT] policy is ~constant regardless of observation — features may not be informative.")
        print("            Fix: use dist-to-obstacle relative to dino x, not raw xPos.")
    else:
        print("  [VERDICT] policy is varying with obs but not learning — try lr=1e-4 or more rollout.")


if __name__ == "__main__":
    main()
