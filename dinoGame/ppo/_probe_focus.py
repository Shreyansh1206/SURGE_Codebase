"""Head-only duck row, but FOCUS the suppression on the hard/dangerous boundary:
close LOW obstacles (cacti, o1_y>0.533, dx<0.9) — not harmless empty gaps.
Goal: high bird duck-argmax AND ~0 dangerous (close-cactus) leak."""
import numpy as np, torch, torch.nn.functional as F
from ppo_agent import PPO

d = np.load("_duckdata.npz")
bird = d["bird"].astype(np.float32); other = d["other"].astype(np.float32)
FPF, DUCK, NOOP, JUMP = 12, 2, 0, 1
of = other[:, -FPF:]
# split 'other' into close-low-obstacle (cactus-like, dangerous) vs the rest
cact_mask = (of[:, 5] > 0.533) & (of[:, 4] < 0.9)
cactus = other[cact_mask]; empty = other[~cact_mask]
print(f"bird={len(bird)} cactus(close-low)={len(cactus)} empty/far={len(empty)}")
B = torch.tensor(bird); C = torch.tensor(cactus); E = torch.tensor(empty); O = torch.tensor(other)

def run(target, cap, lr, iters, cact_w):
    ppo = PPO(48, 3, device="cpu"); ppo.load("checkpoints_duck/duck_init.pt", load_optim=False)
    ph = ppo.net.policy_head
    opt = torch.optim.Adam([ph.weight, ph.bias], lr=lr)
    lt, lc = float(np.log(target)), float(np.log(cap))
    nb = len(bird)
    for it in range(iters):
        cs = C[np.random.choice(len(cactus), min(len(cactus), int(nb*cact_w)), replace=False)]
        es = E[np.random.choice(len(empty), min(len(empty), nb), replace=False)]
        lpb = F.log_softmax(ppo.net(B)[0], -1)[:, DUCK]
        lpc = F.log_softmax(ppo.net(cs)[0], -1)[:, DUCK]
        lpe = F.log_softmax(ppo.net(es)[0], -1)[:, DUCK]
        loss = 2.0*torch.clamp(lt-lpb, min=0).mean() \
             + 3.0*torch.clamp(lpc-lc, min=0).mean() \
             + 1.0*torch.clamp(lpe-lc, min=0).mean()
        opt.zero_grad(); loss.backward()
        ph.weight.grad[NOOP].zero_(); ph.weight.grad[JUMP].zero_()
        ph.bias.grad[NOOP].zero_(); ph.bias.grad[JUMP].zero_()
        opt.step()
    with torch.no_grad():
        barg = torch.argmax(ppo.net(B)[0], -1).numpy()
        carg = torch.argmax(ppo.net(C)[0], -1).numpy()
        oarg = torch.argmax(ppo.net(O)[0], -1).numpy()
    dl = of[:, 4]; dy = of[:, 5]
    danger = np.mean((oarg == DUCK) & (dl < 0.35) & (dy > 0.533)) * 100
    print(f"  tgt{target} cap{cap} lr{lr} it{iters} cactW{cact_w} -> "
          f"bird duck-arg {np.mean(barg==DUCK)*100:5.1f}% | "
          f"cactus(close-low) duck-arg {np.mean(carg==DUCK)*100:5.1f}% | "
          f"DANGEROUS leak {danger:.2f}%")

for cfg in [
    (0.9, 0.05, 3e-3, 400, 1.0),
    (0.9, 0.02, 3e-3, 400, 2.0),
    (0.85, 0.02, 3e-3, 600, 3.0),
    (0.8, 0.02, 3e-3, 800, 4.0),
    (0.75, 0.01, 3e-3, 800, 5.0),
]:
    run(*cfg)
