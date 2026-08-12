"""Checkpoint I/O with compatibility for pre-refactor SOSRL files."""

from dataclasses import asdict
from pathlib import Path

import torch

from .agent import ArchitectureDQNAgent, DQNAgent, IntDQNAgent
from .config import DQNConfig, HRLConfig, IntDQNConfig, default_device


def _device_name(device=None) -> str:
    return str(device or default_device())


def _agent_payload(agent) -> dict:
    return {
        "obs_dim": agent.obs_dim,
        "action_dim": agent.action_dim,
        "config": asdict(agent.config),
        "q_net_state_dict": agent.q_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learn_step": agent.learn_step,
    }


def save_agent_checkpoint(agent, path, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _agent_payload(agent)
    checkpoint["training_state"] = training_state or {}
    torch.save(checkpoint, path)
    return path


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def _restore(agent, checkpoint, load_optimizer: bool):
    agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    if load_optimizer:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.learn_step = int(checkpoint["learn_step"])
    return agent, checkpoint


def load_scheduler_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)
    values = dict(checkpoint["config"])
    values["device"] = device
    agent = DQNAgent(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        DQNConfig(**values),
    )
    return _restore(agent, checkpoint, bool(load_optimizer))


def load_architecture_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)
    values = dict(checkpoint["config"])
    values["device"] = device
    agent = ArchitectureDQNAgent(
        int(checkpoint["obs_dim"]),
        HRLConfig(**values),
    )
    if int(checkpoint["action_dim"]) != agent.action_dim:
        raise ValueError("architecture checkpoint action dimension must be six.")
    return _restore(agent, checkpoint, bool(load_optimizer))


def load_flat_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)
    values = dict(checkpoint["config"])
    values["device"] = device
    agent = IntDQNAgent(
        int(checkpoint["obs_dim"]),
        int(checkpoint["action_dim"]),
        IntDQNConfig(**values),
    )
    return _restore(agent, checkpoint, bool(load_optimizer))


def save_combined_checkpoint(
    path,
    architecture_agent,
    scheduler_agent,
    training_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": _agent_payload(architecture_agent),
            "scheduler": _agent_payload(scheduler_agent),
            "training_state": training_state or {},
        },
        path,
    )
    return path


def load_combined_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)

    arch_data = checkpoint["architecture"]
    arch_values = dict(arch_data["config"])
    arch_values["device"] = device
    architecture_agent = ArchitectureDQNAgent(
        int(arch_data["obs_dim"]),
        HRLConfig(**arch_values),
    )
    if int(arch_data["action_dim"]) != architecture_agent.action_dim:
        raise ValueError("combined checkpoint architecture action dimension must be six.")
    _restore(architecture_agent, arch_data, bool(load_optimizer))

    scheduler_data = checkpoint["scheduler"]
    scheduler_values = dict(scheduler_data["config"])
    scheduler_values["device"] = device
    scheduler_agent = DQNAgent(
        int(scheduler_data["obs_dim"]),
        int(scheduler_data["action_dim"]),
        DQNConfig(**scheduler_values),
    )
    _restore(scheduler_agent, scheduler_data, bool(load_optimizer))
    return architecture_agent, scheduler_agent, checkpoint
