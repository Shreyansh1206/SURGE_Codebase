"""
Behavior-cloning warmup. Collect (obs, action) pairs from the scripted policy
that we know scores ~400+, then train the actor-critic to imitate it via
cross-entropy on the policy head. The value head is also pre-trained on the
Monte-Carlo returns of the scripted trajectories so PPO doesn't start with a
useless critic.

After this runs, checkpoints/bc_init.pt is a strong starting point for PPO.
Then call train.py with --resume checkpoints/bc_init.pt.
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F

from dino_env import DinoEnv, OBS_DIM, N_ACTIONS, FEATURES_PER_FRAME
from ppo_agent import PPO


def scripted_action(obs):
    """Same rule as test_scripted.py: jump if obstacle close & not jumping."""
    f = obs[-FEATURES_PER_FRAME:]
    dx_o1, jumping = f[4], f[1]
    return 1 if (dx_o1 < 0.30 and jumping < 0.5) else 0


def collect(env, n_episodes: int, gamma: float = 0.99):
    """Run scripted policy, return (obs, actions, returns) arrays."""
    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    scores = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = scripted_action(obs)
            obs_buf.append(obs.copy())
            act_buf.append(a)
            obs, r, done, info = env.step(a)
            rew_buf.append(r)
            done_buf.append(done)
        scores.append(info.get("score", 0))
        print(f"  bc ep {ep+1}/{n_episodes}: score={scores[-1]}")

    # Discounted returns per-episode (backward pass, resetting at done).
    returns = np.zeros(len(rew_buf), dtype=np.float32)
    R = 0.0
    for t in reversed(range(len(rew_buf))):
        if done_buf[t]:
            R = 0.0
        R = rew_buf[t] + gamma * R
        returns[t] = R

    return (np.stack(obs_buf).astype(np.float32),
            np.array(act_buf, dtype=np.int64),
            returns,
            scores)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30,
                   help="Scripted episodes to collect (~30 ≈ 12k samples).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", type=str, default="checkpoints/bc_init.pt")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    print(f"[bc] collecting {args.episodes} scripted episodes ...")
    env = DinoEnv()
    obs_np, act_np, ret_np, scores = collect(env, args.episodes)
    env.close()
    print(f"[bc] collected {len(obs_np)} samples. "
          f"scripted score: mean={np.mean(scores):.1f} max={max(scores)}")
    print(f"[bc] action mix: noop={int((act_np==0).sum())} "
          f"jump={int((act_np==1).sum())} duck={int((act_np==2).sum())}")

    ppo = PPO(OBS_DIM, N_ACTIONS, lr=args.lr)
    dev = ppo.device
    obs_t = torch.tensor(obs_np, device=dev)
    act_t = torch.tensor(act_np, device=dev)
    ret_t = torch.tensor(ret_np, device=dev)

    # Normalise returns for the value-head fit — stabilises learning rate.
    ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)

    N = len(obs_t)
    idx = np.arange(N)
    print(f"[bc] training {args.epochs} epochs over {N} samples ...")
    for epoch in range(args.epochs):
        np.random.shuffle(idx)
        tot_pi, tot_v, n_batches = 0.0, 0.0, 0
        for start in range(0, N, args.batch_size):
            b = torch.as_tensor(idx[start:start + args.batch_size],
                                dtype=torch.long, device=dev)
            logits, value = ppo.net(obs_t[b])
            pi_loss = F.cross_entropy(logits, act_t[b])
            v_loss  = F.mse_loss(value, ret_t[b])
            loss = pi_loss + 0.5 * v_loss
            ppo.optim.zero_grad(); loss.backward(); ppo.optim.step()
            tot_pi += pi_loss.item(); tot_v += v_loss.item(); n_batches += 1

        # Quick policy-accuracy check on the same data.
        with torch.no_grad():
            preds = ppo.net(obs_t)[0].argmax(-1)
            acc = (preds == act_t).float().mean().item()
        print(f"  epoch {epoch+1:2d}: pi_loss={tot_pi/n_batches:.3f} "
              f"v_loss={tot_v/n_batches:.3f}  scripted-accuracy={acc:.3f}")

    ppo.save(args.save)
    print(f"[bc] saved {args.save}")
    print("[bc] next: python train.py --resume " + args.save + " --updates 300 --entropy 0.01")


if __name__ == "__main__":
    main()
