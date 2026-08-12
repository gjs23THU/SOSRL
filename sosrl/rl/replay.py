"""Replay buffers for one-step and n-step DQN updates."""

from collections import deque
import random

import numpy as np


class ReplayBuffer:
    """One-step replay used by the four-rule Scheduler DQN."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, obs, action, reward, next_obs, done, next_mask) -> None:
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, int(batch_size))
        obs, action, reward, next_obs, done, next_mask = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(next_mask, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class FlatReplayBuffer:
    """One-step replay for the assignment-level IntDQN baseline."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, obs, action, reward, next_obs, done, next_mask) -> None:
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
                np.asarray(next_mask, dtype=bool).copy(),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, int(batch_size))
        obs, action, reward, next_obs, done, next_mask = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(next_mask, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class ArchitectureReplayBuffer:
    """Replay containing discounted n-step architecture transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        next_mask,
        discount,
    ) -> None:
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32),
                bool(done),
                np.asarray(next_mask, dtype=np.float32),
                float(discount),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, int(batch_size))
        obs, action, reward, next_obs, done, next_mask, discount = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(next_mask, dtype=np.float32),
            np.asarray(discount, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class NStepAccumulator:
    """Convert primitive architecture transitions into n-step transitions."""

    def __init__(self, n_step: int, gamma: float):
        if int(n_step) <= 0:
            raise ValueError("n_step must be positive.")
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.pending = deque()

    def append(self, transition):
        self.pending.append(transition)
        emitted = []
        if len(self.pending) >= self.n_step:
            emitted.append(self._emit_one())
        if bool(transition[4]):
            while self.pending:
                emitted.append(self._emit_one())
        return emitted

    def _emit_one(self):
        first = self.pending[0]
        reward = 0.0
        steps = 0
        last = first
        for transition in list(self.pending)[: self.n_step]:
            reward += (self.gamma**steps) * float(transition[2])
            steps += 1
            last = transition
            if bool(transition[4]):
                break
        self.pending.popleft()
        return (
            first[0],
            first[1],
            reward,
            last[3],
            bool(last[4]),
            last[5],
            self.gamma**steps,
        )

    def clear(self) -> None:
        self.pending.clear()
