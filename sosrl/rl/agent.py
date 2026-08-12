"""Neural policies used by the scheduler, architecture, and flat baseline."""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DQNConfig, HRLConfig, IntDQNConfig
from .replay import ArchitectureReplayBuffer, FlatReplayBuffer, ReplayBuffer


class QNetwork(nn.Module):
    """Two-hidden-layer MLP shared by all DQN policies."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class DQNAgent:
    """Four-rule Scheduler DQN."""

    checkpoint_kind = "scheduler"

    def __init__(self, obs_dim: int, action_dim: int, config: DQNConfig):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(config.device)
        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def save_checkpoint(self, path, training_state=None):
        from .checkpoint import save_agent_checkpoint

        return save_agent_checkpoint(self, path, training_state)

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        from .checkpoint import load_scheduler_checkpoint

        return load_scheduler_checkpoint(path, device, load_optimizer)

    def select_action(self, obs, action_mask, epsilon: float) -> int:
        valid_actions = np.flatnonzero(np.asarray(action_mask) > 0)
        if len(valid_actions) == 0:
            raise ValueError("No valid rule action.")
        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.tensor(
            action_mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            q_values = q_values.masked_fill(mask_tensor <= 0, -1e9)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        required = max(self.config.min_buffer_size, self.config.batch_size)
        if len(self.replay) < required:
            return None
        obs, action, reward, next_obs, done, next_mask = self.replay.sample(
            self.config.batch_size
        )
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(
            action,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        reward = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.tensor(next_mask, dtype=torch.float32, device=self.device)

        q_value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_obs)
            next_q = next_q.masked_fill(next_mask <= 0, -1e9)
            next_q = next_q.max(dim=1, keepdim=True).values
            has_next_action = (next_mask.sum(dim=1, keepdim=True) > 0).float()
            target = (
                reward
                + self.config.gamma
                * (1.0 - done)
                * has_next_action
                * next_q
            )

        loss = nn.functional.mse_loss(q_value, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


class ArchitectureDQNAgent:
    """Single-headed six-rule Architecture DQN."""

    checkpoint_kind = "architecture"
    ACTION_DIM = 6

    def __init__(self, obs_dim: int, config: HRLConfig):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = self.ACTION_DIM
        self.device = torch.device(config.device)
        self.q_net = QNetwork(
            self.obs_dim,
            self.action_dim,
            config.hidden_dim,
        ).to(self.device)
        self.target_net = QNetwork(
            self.obs_dim,
            self.action_dim,
            config.hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(
            self.q_net.parameters(),
            lr=config.architecture_lr,
        )
        self.replay = ArchitectureReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def save_checkpoint(self, path, training_state=None):
        from .checkpoint import save_agent_checkpoint

        return save_agent_checkpoint(self, path, training_state)

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        from .checkpoint import load_architecture_checkpoint

        return load_architecture_checkpoint(path, device, load_optimizer)

    def select_action(self, obs, action_mask, epsilon: float = 0.0) -> int:
        valid = np.flatnonzero(np.asarray(action_mask) > 0)
        if not valid.size:
            raise ValueError("No valid architecture action.")
        if epsilon > 0 and random.random() < epsilon:
            return int(random.choice(valid.tolist()))
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            values = self.q_net(obs_tensor)
            values = values.masked_fill(mask_tensor <= 0, -1e9)
        return int(values.argmax(dim=1).item())

    def learn(self):
        required = max(self.config.batch_size, self.config.min_buffer_size)
        if len(self.replay) < required:
            return None
        obs, action, reward, next_obs, done, next_mask, discount = (
            self.replay.sample(self.config.batch_size)
        )
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        reward = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.as_tensor(next_mask, dtype=torch.float32, device=self.device)
        discount = torch.as_tensor(
            discount,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            next_value = self.target_net(next_obs)
            next_value = next_value.masked_fill(next_mask <= 0, -1e9)
            next_value = next_value.max(dim=1, keepdim=True).values
            has_next = (next_mask.sum(dim=1, keepdim=True) > 0).float()
            target = reward + discount * (1.0 - done) * has_next * next_value
        loss = nn.functional.mse_loss(value, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())


class IntDQNAgent:
    """Assignment-level flat DQN retained as a baseline."""

    checkpoint_kind = "flat"

    def __init__(self, obs_dim: int, action_dim: int, config: IntDQNConfig):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(config.device)
        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = FlatReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def save_checkpoint(self, path, training_state=None):
        from .checkpoint import save_agent_checkpoint

        return save_agent_checkpoint(self, path, training_state)

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        from .checkpoint import load_flat_checkpoint

        return load_flat_checkpoint(path, device, load_optimizer)

    def select_action(self, obs, action_mask, epsilon: float) -> int:
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (self.action_dim,):
            raise ValueError("action mask has the wrong shape.")
        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            raise ValueError("no valid assignment action.")
        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        required = max(self.config.batch_size, self.config.min_buffer_size)
        if len(self.replay) < required:
            return None
        obs, action, reward, next_obs, done, next_mask = self.replay.sample(
            self.config.batch_size
        )
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(
            action,
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        reward = torch.as_tensor(
            reward,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.as_tensor(next_mask, dtype=torch.bool, device=self.device)

        q_value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_q_values = next_q_values.masked_fill(~next_mask, -torch.inf)
            next_q = next_q_values.max(dim=1, keepdim=True).values
            has_next_action = next_mask.any(dim=1, keepdim=True)
            next_q = torch.where(has_next_action, next_q, torch.zeros_like(next_q))
            target = reward + self.config.gamma * (1.0 - done) * next_q

        loss = nn.functional.smooth_l1_loss(q_value, target)
        if not torch.isfinite(loss):
            raise RuntimeError("DQN loss became non-finite.")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())
