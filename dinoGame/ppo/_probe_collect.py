"""Collect a small labelled dataset of real bird vs other states and save it,
plus report the base policy's action distribution on bird states."""
import numpy as np, torch, torch.nn.functional as F
from ppo_agent import PPO
from dino_env import VecDinoEnv, OBS_DIM, N_ACTIONS
from finetune_duck import collect_states, duckable_bird_mask

vec = VecDinoEnv(n_envs=4, window_size=(700, 320), grid_cols=2, headless=True,
                 curriculum_prob=0.8, start_speed_range=(8.5, 9.0),
                 start_score=450.0, duck_shaping=False)
try:
    ppo = PPO(OBS_DIM, N_ACTIONS); ppo.load("checkpoints_duck/duck_init.pt", load_optim=False)
    bird, other = collect_states(vec, ppo, ppo.device, 1200, guide_prob=0.9)
    np.savez("_birddata.npz", bird=bird, other=other)
    with torch.no_grad():
        pb = F.softmax(ppo.net(torch.tensor(bird))[0], -1).mean(0).numpy()
        po = F.softmax(ppo.net(torch.tensor(other))[0], -1).mean(0).numpy()
    print(f"collected bird={len(bird)} other={len(other)}")
    print(f"base policy mean probs on BIRD : noop={pb[0]:.3f} jump={pb[1]:.3f} duck={pb[2]:.3f}")
    print(f"base policy mean probs on OTHER: noop={po[0]:.3f} jump={po[1]:.3f} duck={po[2]:.3f}")
    # how often base argmax == jump on bird states (the thing duck must beat)
    with torch.no_grad():
        am = torch.argmax(ppo.net(torch.tensor(bird))[0], -1).numpy()
    print(f"base argmax on BIRD: noop={np.mean(am==0)*100:.1f}% jump={np.mean(am==1)*100:.1f}% duck={np.mean(am==2)*100:.1f}%")
finally:
    vec.close()
