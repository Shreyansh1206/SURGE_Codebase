"""
Scripted-policy diagnostic. Runs a hand-coded rule that just jumps when the
nearest obstacle is in the danger zone. If THIS can't beat score 200, the
environment itself is broken and no RL fix will help. If it can, then our
PPO learner is the problem, not the env or features.

Also prints what the agent actually "sees" each step so we can sanity-check
that the feature vector matches the screen.
"""
import time
import numpy as np
from dino_env import DinoEnv, FEATURES_PER_FRAME, FRAME_STACK


def main():
    env = DinoEnv()
    obs = env.reset()
    print(f"obs shape: {obs.shape}  (expected {FEATURES_PER_FRAME * FRAME_STACK},)")

    # The most recent frame's features are the LAST FEATURES_PER_FRAME values.
    def latest_frame(o):
        return o[-FEATURES_PER_FRAME:]

    scores = []
    for ep in range(5):
        obs = env.reset()
        done = False
        steps = 0
        sample_log = []
        action_counts = np.zeros(3, dtype=np.int64)

        while not done and steps < 2000:
            f = latest_frame(obs)
            # Feature layout (last frame): [dinoY, jumping, ducking, speed,
            #                               o1_dx, o1_y, o1_w, o1_h,
            #                               o2_dx, o2_y, o2_w, o2_h]
            dx_o1   = f[4]
            jumping = f[1]

            # Scripted rule: jump when obstacle is in the "decide now" window
            # and we're not already mid-jump. Tuned crudely — meant only to
            # show the env can be played, not optimally.
            if dx_o1 < 0.30 and jumping < 0.5:
                action = 1  # jump
            else:
                action = 0  # no-op

            action_counts[action] += 1
            obs, r, done, info = env.step(action)
            steps += 1

            if steps in (1, 5, 10, 20, 40, 80, 160):
                sample_log.append(
                    f"  step {steps:4d}: dx_o1={f[4]:.3f} jump_flag={f[1]:.0f} "
                    f"speed={f[3]:.2f} action={action} r={r:+.3f}"
                )

        score = info.get("score", 0)
        scores.append(score)
        print(f"\nEp {ep+1}: score={score}, steps={steps}, "
              f"actions noop/jump/duck={action_counts.tolist()}")
        for line in sample_log:
            print(line)

    env.close()
    print(f"\n=== Scripted policy scores: {scores} ===")
    print(f"  mean={np.mean(scores):.1f}  max={max(scores)}")
    if max(scores) < 100:
        print("  VERDICT: env or features are broken. RL can't help until this passes.")
    elif np.mean(scores) > 300:
        print("  VERDICT: env works. The bug is in PPO / training, not the environment.")
    else:
        print("  VERDICT: env marginally works. Features are noisy or jump timing is off.")


if __name__ == "__main__":
    main()
