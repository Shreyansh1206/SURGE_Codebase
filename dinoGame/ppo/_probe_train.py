"""Offline (CPU) search over imitation settings on the REAL collected bird/other
data, to find what makes DUCK the argmax on bird states while keeping duck low
on other states (base-preserving)."""
import numpy as np, torch, torch.nn.functional as F
import finetune_duck as fd
from ppo_agent import PPO

d = np.load("_birddata.npz")
bird, other = d["bird"].astype(np.float32), d["other"].astype(np.float32)
print(f"data: bird={len(bird)} other={len(other)}")

def metrics(ppo):
    with torch.no_grad():
        lb = ppo.net(torch.tensor(bird))[0]; pb = F.softmax(lb, -1).mean(0).numpy()
        ab = torch.argmax(lb, -1).numpy()
        lo = ppo.net(torch.tensor(other))[0]; po = F.softmax(lo, -1).mean(0).numpy()
        ao = torch.argmax(lo, -1).numpy()
    return (np.mean(ab == 2) * 100, pb[2], np.mean(ao == 2) * 100, po[2])

def run(target, cap, lr, epochs, other_mult, iters=60):
    ppo = PPO(48, 3, device="cpu"); ppo.load("checkpoints_duck/duck_init.pt", load_optim=False)
    opt = torch.optim.Adam(ppo.net.parameters(), lr=lr)
    for it in range(1, iters + 1):
        k = min(len(other), max(1, int(len(bird) * other_mult)))
        osamp = other[np.random.choice(len(other), k, replace=False)]
        fd.duck_imitation_update(ppo, opt, bird, osamp, coef=2.0,
                                 target_duck=target, duck_cap=cap, epochs=epochs)
    return metrics(ppo)

print("cfg(target,cap,lr,epochs,other_mult) -> birdDuckArg% birdDuckP otherDuckArg% otherDuckP")
for cfg in [
    (0.80, 0.05, 5e-4, 4, 3.0),
    (0.80, 0.05, 1e-3, 6, 1.0),
    (0.85, 0.10, 1e-3, 6, 1.0),
    (0.85, 0.05, 1e-3, 6, 2.0),
    (0.90, 0.05, 1e-3, 8, 1.0),
]:
    bA, bP, oA, oP = run(*cfg)
    print(f"  {cfg} -> birdArg={bA:5.1f}% birdP={bP:.2f} | otherArg={oA:4.1f}% otherP={oP:.3f}")
