"""Training configuration objects for all SOSRL policies."""

from dataclasses import dataclass

import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class DQNConfig:
    episodes: int = 200
    scenario_pool_size: int = 20
    scenario_order: str = "random"
    shared_mission: bool = False
    rule_set: str = "standard"
    selected_system_num: int | tuple[int, int] | None = None
    min_system_num: int = 3
    max_system_num: int = 22
    cost_limit: float | None = 8000
    gamma: float = 0.99
    lr: float = 1e-4
    lr_end: float | None = None
    lr_decay: float = 1.0
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size: int = 500
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 1024
    seed: int = 1
    device: str = default_device()

    def learning_rate_at_episode(self, episode: int) -> float:
        """Return the exponentially decayed scheduler learning rate."""

        if int(episode) < 0:
            raise ValueError("episode must be non-negative.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive.")
        if not 0.0 < float(self.lr_decay) <= 1.0:
            raise ValueError("lr_decay must be in (0, 1].")
        floor = 0.0 if self.lr_end is None else float(self.lr_end)
        if floor < 0.0 or floor > float(self.lr):
            raise ValueError("lr_end must be between 0 and lr.")
        return max(floor, float(self.lr) * float(self.lr_decay) ** int(episode))


@dataclass
class HRLConfig:
    episodes: int = 500
    scenario_pool_size: int = 50
    budget: float = 8000.0
    refund_rate: float = 0.8
    gamma: float = 0.99
    n_step: int = 5
    architecture_lr: float = 1e-4
    scheduler_finetune_lr: float = 1e-5
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 500
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 256
    seed: int = 1
    device: str = default_device()


@dataclass
class IntDQNConfig:
    episodes: int = 1000
    fixed_mission: bool = False
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 1000
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 512
    seed: int = 1
    device: str = default_device()
    log_interval: int = 10


@dataclass
class BranchingDQNConfig:
    """Training configuration for the constrained additive branching scheduler."""

    episodes: int = 2000
    max_env_steps: int | None = 240000
    scenario_pool_size: int = 100
    budget: float = 8000.0
    refund_rate: float = 0.8
    gamma: float = 0.99
    lr: float = 1e-4
    lr_end: float | None = None
    lr_decay: float = 1.0
    batch_size: int = 64
    buffer_size: int = 50000
    min_buffer_size: int = 1000
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 1
    device: str = default_device()
    log_interval: int = 10

    def learning_rate_at_episode(self, episode: int) -> float:
        """Return the exponentially decayed learning rate for one episode."""

        if int(episode) < 0:
            raise ValueError("episode must be non-negative.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be positive.")
        if not 0.0 < float(self.lr_decay) <= 1.0:
            raise ValueError("lr_decay must be in (0, 1].")
        floor = 0.0 if self.lr_end is None else float(self.lr_end)
        if floor < 0.0 or floor > float(self.lr):
            raise ValueError("lr_end must be between 0 and lr.")
        return max(floor, float(self.lr) * float(self.lr_decay) ** int(episode))
