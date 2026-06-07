"""Characterise the 'other' states that leak to duck under head-only training:
are they harmless (far/empty) or dangerous (a close low obstacle = cactus)?"""
import numpy as np, torch, torch.nn.functional as F
from ppo_agent import PPO

d = np.load("_duckdata.npz")
other = d["other"].astype(np.float32); bird = d["bird"].astype(np.float32)
O, Bd = torch.tensor(other), torch.tensor(bird)
FPF, DUCK, NOOP, JUMP = 12, 2, 0, 1

ppo = PPO(48, 3, device="cpu"); ppo.load("checkpoints_duck/duck_init.pt", load_optim=False)
ph = ppo.net.policy_head
opt = torch.optim.Adam([ph.weight, ph.bias], lr=3e-3)
ltgt, lcap = float(np.log(0.9)), float(np.log(0.05))
for it in range(300):
    k = min(len(other), len(bird))
    os_ = O[np.random.choice(len(other), k, replace=False)]
    lb = F.log_softmax(ppo.net(Bd)[0], -1)[:, DUCK]
    lo = F.log_softmax(ppo.net(os_)[0], -1)[:, DUCK]
    loss = 2.0*(torch.clamp(ltgt-lb, min=0).mean() + torch.clamp(lo-lcap, min=0).mean())
    opt.zero_grad(); loss.backward()
    ph.weight.grad[NOOP].zero_(); ph.weight.grad[JUMP].zero_()
    ph.bias.grad[NOOP].zero_(); ph.bias.grad[JUMP].zero_()
    opt.step()

with torch.no_grad():
    arg = torch.argmax(ppo.net(O)[0], -1).numpy()
f = other[:, -FPF:]
o1_dx, o1_y = f[:, 4], f[:, 5]
leak = arg == DUCK
print(f"other states: {len(other)} | leaked to duck: {leak.sum()} ({leak.mean()*100:.1f}%)")
# danger = a LOW obstacle that is CLOSE (would need a jump). dx<0.35 and o1_y>0.533
danger = leak & (o1_dx < 0.35) & (o1_y > 0.533)
near    = leak & (o1_dx < 0.35)
far     = leak & (o1_dx >= 0.9)
print(f"  of leaked: close(<0.35dx)={near.sum()} | close LOW-obstacle(cactus-like)={danger.sum()} | far(>=0.9dx, harmless)={far.sum()}")
print(f"  leaked o1_dx: min {o1_dx[leak].min():.2f} mean {o1_dx[leak].mean():.2f} | "
      f"leaked o1_y mean {o1_y[leak].mean():.2f}")
# danger as fraction of ALL other states (proxy for deadly-duck rate)
print(f"  DANGEROUS leak rate over all other states: {danger.mean()*100:.2f}%")
