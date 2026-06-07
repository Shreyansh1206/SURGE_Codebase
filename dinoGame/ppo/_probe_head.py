"""Offline search on cached real data: can we make DUCK argmax on birds while
keeping no-op/jump behaviour (cactus timing) ~unchanged? Compare:
  (A) head-only (duck row, frozen trunk)  -- safe but maybe too weak
  (B) full-net + distillation anchor on no-op/jump logits to the frozen base
"""
import numpy as np, torch, torch.nn.functional as F, copy
from ppo_agent import PPO

d = np.load("_duckdata.npz")
bird, other = d["bird"].astype(np.float32), d["other"].astype(np.float32)
print(f"data bird={len(bird)} other={len(other)}")
B, O = torch.tensor(bird), torch.tensor(other)
DUCK, NOOP, JUMP = 2, 0, 1

def fresh():
    p = PPO(48, 3, device="cpu"); p.load("checkpoints_duck/duck_init.pt", load_optim=False)
    return p

def report(ppo, ref, tag):
    with torch.no_grad():
        lb = ppo.net(B)[0]; lo = ppo.net(O)[0]
        rb = ref(B)[0];     ro = ref(O)[0]
        bird_arg = torch.argmax(lb, -1).numpy()
        oth_arg  = torch.argmax(lo, -1).numpy()
        oth_ref  = torch.argmax(ro, -1).numpy()
        # how much did the no-op-vs-jump decision drift on 'other'? (cactus timing)
        njo = (lo[:, JUMP] - lo[:, NOOP]); njo_ref = (ro[:, JUMP] - ro[:, NOOP])
        drift = (njo - njo_ref).abs().mean().item()
    print(f"  [{tag}] bird duck-arg {np.mean(bird_arg==DUCK)*100:5.1f}% | "
          f"other duck-arg {np.mean(oth_arg==DUCK)*100:.2f}% | "
          f"other argmax-changed {np.mean(oth_arg!=oth_ref)*100:.2f}% | "
          f"jump-noop drift {drift:.3f}")

def hinge(logp_duck, sign, lim):
    return torch.clamp(sign * (lim - logp_duck), min=0.0)

def train(mode, iters=200, lr=2e-3, target=0.9, cap=0.05, distill=5.0):
    ppo = fresh()
    ref = copy.deepcopy(ppo.net); [p.requires_grad_(False) for p in ref.parameters()]
    ph = ppo.net.policy_head
    params = [ph.weight, ph.bias] if mode == "head" else list(ppo.net.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    ltgt, lcap = float(np.log(target)), float(np.log(cap))
    for it in range(iters):
        k = min(len(other), len(bird))
        os_ = O[np.random.choice(len(other), k, replace=False)]
        lb = ppo.net(B)[0]; lo = ppo.net(os_)[0]
        lpb = F.log_softmax(lb, -1)[:, DUCK]; lpo = F.log_softmax(lo, -1)[:, DUCK]
        loss = 2.0 * (hinge(lpb, +1, ltgt).mean() + hinge(lpo, -1, lcap).mean())
        if mode == "distill":
            # keep jump-vs-noop logit gap close to frozen base on BOTH sets
            with torch.no_grad():
                rb = ref(B)[0]; ro = ref(os_)[0]
            gap_b = (lb[:, JUMP]-lb[:, NOOP]); gap_o = (lo[:, JUMP]-lo[:, NOOP])
            rg_b = (rb[:, JUMP]-rb[:, NOOP]); rg_o = (ro[:, JUMP]-ro[:, NOOP])
            loss = loss + distill * (F.mse_loss(gap_b, rg_b) + F.mse_loss(gap_o, rg_o))
        opt.zero_grad(); loss.backward()
        if mode == "head":
            ph.weight.grad[NOOP].zero_(); ph.weight.grad[JUMP].zero_()
            ph.bias.grad[NOOP].zero_();   ph.bias.grad[JUMP].zero_()
        opt.step()
    report(ppo, ref, f"{mode} lr{lr} it{iters} tgt{target} dist{distill if mode=='distill' else '-'}")
    return ppo

print("baseline:"); report(fresh(), fresh().net, "base")
train("head", iters=300, lr=3e-3, target=0.9)
train("head", iters=300, lr=3e-3, target=0.95)
train("distill", iters=300, lr=2e-3, target=0.9, distill=10.0)
train("distill", iters=300, lr=2e-3, target=0.9, distill=3.0)
train("distill", iters=300, lr=2e-3, target=0.9, distill=1.0)
