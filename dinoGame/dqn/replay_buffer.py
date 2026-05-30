"""
Uniform replay buffer + n-step collector.

`ReplayBuffer` is a flat ring buffer of (obs, action, R^(k), next_obs,
terminal, gamma_pow, next_mask) tuples. `gamma_pow = gamma^k` where k is the
actual number of steps the n-step return spans — usually n, but smaller when
truncated by an episode boundary.

`NStepCollector` is the glue between the vec-env step loop and the buffer.
It owns one in-flight FIFO per env so that an episode end in env j cannot
contaminate the n-step return computed for env i (i ≠ j) — and crucially,
within a single env, transitions from episode k+1 do NOT bleed into the
n-step return of any transition from episode k.

Invariants enforced:
  - Every transition put into the collector eventually appears in the buffer
    exactly once.
  - The n-step return for a transition stops accumulating at the first done
    inside its window. Beyond that point, the next-obs/next-mask stored are
    those of the step where the done occurred, but they are gated out by
    `terminal=True` (the agent's Bellman target multiplies by (1 - terminal)).
  - When n=1 the collector behaves exactly like a single-step buffer.

Sanity-test the boundary handling by running `python -m dqn.replay_buffer`.
"""

from collections import deque
from typing import Tuple

import numpy as np


class ReplayBuffer:
    """Flat uniform replay over n-step transitions."""

    def __init__(self, capacity: int, obs_dim: int, n_actions: int):
        self.capacity = int(capacity)
        self.obs       = np.zeros((self.capacity, obs_dim),  dtype=np.float32)
        self.next_obs  = np.zeros((self.capacity, obs_dim),  dtype=np.float32)
        self.actions   = np.zeros(self.capacity,             dtype=np.int64)
        self.rewards   = np.zeros(self.capacity,             dtype=np.float32)
        self.terminals = np.zeros(self.capacity,             dtype=np.bool_)
        self.gamma_pow = np.zeros(self.capacity,             dtype=np.float32)
        self.next_mask = np.zeros((self.capacity, n_actions), dtype=np.bool_)
        self.idx  = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self,
             obs: np.ndarray,
             action: int,
             reward: float,
             next_obs: np.ndarray,
             terminal: bool,
             gamma_pow: float,
             next_mask: np.ndarray):
        i = self.idx
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.next_obs[i]  = next_obs
        self.terminals[i] = terminal
        self.gamma_pow[i] = gamma_pow
        self.next_mask[i] = next_mask
        self.idx  = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.obs[idxs],
                self.actions[idxs],
                self.rewards[idxs],
                self.next_obs[idxs],
                self.terminals[idxs],
                self.gamma_pow[idxs],
                self.next_mask[idxs])


class NStepCollector:
    """Per-env n-step return aggregator.

    Usage:
        coll = NStepCollector(n_envs=4, n=3, gamma=0.99)
        for env_i in range(N):
            coll.add(env_i, obs, action, reward, next_obs, done, next_mask, buffer)

    With n=1 this reduces to single-step transitions: every `add` emits one
    record to the buffer immediately.
    """

    def __init__(self, n_envs: int, n: int, gamma: float):
        assert n >= 1
        self.n     = int(n)
        self.gamma = float(gamma)
        self.queues = [deque() for _ in range(n_envs)]

    def reset_env(self, env_i: int):
        """Force-drop any in-flight transitions for env_i.
        Call this if you bypass the normal step loop (e.g. eval interlude)."""
        self.queues[env_i].clear()

    def add(self,
            env_i: int,
            obs: np.ndarray,
            action: int,
            reward: float,
            next_obs: np.ndarray,
            done: bool,
            next_mask: np.ndarray,
            buffer: ReplayBuffer):
        q = self.queues[env_i]
        q.append((obs.copy(), int(action), float(reward),
                  next_obs.copy(), bool(done), next_mask.copy()))

        if done:
            # Flush everything: each remaining transition emits an n-step
            # return that necessarily terminates inside the window.
            while q:
                self._emit_front(env_i, buffer)
        elif len(q) >= self.n:
            self._emit_front(env_i, buffer)

    def _emit_front(self, env_i: int, buffer: ReplayBuffer):
        q = self.queues[env_i]
        first_obs, first_act, _, _, _, _ = q[0]

        R = 0.0
        gamma_acc = 1.0
        terminal  = False
        last_next_obs  = None
        last_next_mask = None
        for k, (_, _, r, no, d, m) in enumerate(q):
            R += gamma_acc * r
            gamma_acc *= self.gamma
            last_next_obs  = no
            last_next_mask = m
            if d:
                terminal = True
                break
            if k + 1 >= self.n:
                break

        buffer.push(
            obs       = first_obs,
            action    = first_act,
            reward    = R,
            next_obs  = last_next_obs,
            terminal  = terminal,
            gamma_pow = gamma_acc,
            next_mask = last_next_mask,
        )
        q.popleft()


# ---------------------------------------------------------------------------
# Self-test: synthetic episodes, verify n-step returns are correct AND that
# transitions from one episode never bleed into the next.
# ---------------------------------------------------------------------------

def _self_test():
    obs_dim = 2
    n_actions = 3
    gamma = 0.9

    # n=1: should behave like a single-step buffer.
    buf = ReplayBuffer(capacity=100, obs_dim=obs_dim, n_actions=n_actions)
    coll = NStepCollector(n_envs=1, n=1, gamma=gamma)
    for i, (r, done) in enumerate([(1.0, False), (2.0, False), (3.0, True)]):
        o  = np.array([i, 0], dtype=np.float32)
        no = np.array([i + 1, 0], dtype=np.float32)
        m  = np.ones(n_actions, dtype=bool)
        coll.add(0, o, 0, r, no, done, m, buf)
    assert len(buf) == 3, f"n=1: expected 3 records, got {len(buf)}"
    assert np.allclose(buf.rewards[:3], [1.0, 2.0, 3.0])
    assert list(buf.terminals[:3]) == [False, False, True]
    assert np.allclose(buf.gamma_pow[:3], [gamma, gamma, gamma])

    # n=3: episode of length 5 ending in done.
    # Rewards: 1, 2, 3, 4, 5  with done on the 5th step.
    # Expected first record (t=0): R = 1 + γ·2 + γ²·3, gamma_pow = γ³,  terminal=False
    # t=1: R = 2 + γ·3 + γ²·4, gamma_pow = γ³, terminal=False
    # t=2: R = 3 + γ·4 + γ²·5, gamma_pow = γ³, terminal=True
    # t=3: R = 4 + γ·5,        gamma_pow = γ², terminal=True
    # t=4: R = 5,              gamma_pow = γ,  terminal=True
    buf2  = ReplayBuffer(capacity=100, obs_dim=obs_dim, n_actions=n_actions)
    coll2 = NStepCollector(n_envs=1, n=3, gamma=gamma)
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    dones   = [False, False, False, False, True]
    for i, (r, d) in enumerate(zip(rewards, dones)):
        o  = np.array([i, 0], dtype=np.float32)
        no = np.array([i + 1, 0], dtype=np.float32)
        m  = np.ones(n_actions, dtype=bool)
        coll2.add(0, o, 0, r, no, d, m, buf2)

    assert len(buf2) == 5, f"n=3: expected 5 records, got {len(buf2)}"
    expected_R = [
        1 + gamma*2 + gamma**2 * 3,
        2 + gamma*3 + gamma**2 * 4,
        3 + gamma*4 + gamma**2 * 5,
        4 + gamma*5,
        5,
    ]
    expected_term = [False, False, True, True, True]
    expected_gp   = [gamma**3, gamma**3, gamma**3, gamma**2, gamma]
    for i in range(5):
        assert abs(buf2.rewards[i] - expected_R[i]) < 1e-5, \
            f"n=3 t={i}: R={buf2.rewards[i]} expected {expected_R[i]}"
        assert bool(buf2.terminals[i]) == expected_term[i], \
            f"n=3 t={i}: terminal mismatch"
        assert abs(buf2.gamma_pow[i] - expected_gp[i]) < 1e-5, \
            f"n=3 t={i}: gamma_pow={buf2.gamma_pow[i]} expected {expected_gp[i]}"

    # Episode 2 right after — make sure FIFO was cleared on the done.
    # First step of new episode with rewards 10, then done.
    coll2.add(0, np.array([100, 0], dtype=np.float32), 0, 10.0,
              np.array([101, 0], dtype=np.float32), True,
              np.ones(n_actions, dtype=bool), buf2)
    assert len(buf2) == 6, f"after ep2 step1: {len(buf2)}"
    # The new record's reward must be 10 (NOT contaminated with ep1's 5).
    assert abs(buf2.rewards[5] - 10.0) < 1e-5
    assert bool(buf2.terminals[5]) is True
    assert abs(buf2.gamma_pow[5] - gamma) < 1e-5
    # next_obs of the new record must be from ep2, not the trailing ep1 state.
    assert buf2.next_obs[5][0] == 101.0

    # Multi-env independence: env 0 dies; env 1's in-flight is untouched.
    coll3 = NStepCollector(n_envs=2, n=3, gamma=gamma)
    buf3  = ReplayBuffer(capacity=100, obs_dim=obs_dim, n_actions=n_actions)
    # env 1 fills two transitions (queue length 2, no emit yet at n=3).
    for i in range(2):
        coll3.add(1, np.array([i, 1], dtype=np.float32), 0, float(i + 1),
                  np.array([i + 1, 1], dtype=np.float32), False,
                  np.ones(n_actions, dtype=bool), buf3)
    assert len(buf3) == 0, "env 1 should not have emitted yet"
    # env 0 fills one then dies.
    coll3.add(0, np.array([0, 0], dtype=np.float32), 0, 7.0,
              np.array([1, 0], dtype=np.float32), True,
              np.ones(n_actions, dtype=bool), buf3)
    assert len(buf3) == 1, "env 0 should have flushed 1 record on done"
    assert abs(buf3.rewards[0] - 7.0) < 1e-5
    # env 1's queue length unchanged after env 0's done.
    assert len(coll3.queues[1]) == 2, "env 1 queue size leaked"

    print("[replay_buffer] self-test passed.")


if __name__ == "__main__":
    _self_test()
