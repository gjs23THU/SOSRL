"""Checkpoint I/O with compatibility for pre-refactor SOSRL files."""

from dataclasses import asdict
from pathlib import Path

import torch

from .agent import ArchitectureDQNAgent, DQNAgent, FlatRuleDQNAgent, IntDQNAgent
from .branching import (
    FEATURE_SCHEMA_VERSION,
    GLOBAL_FEATURE_DIM,
    GLOBAL_FEATURE_NAMES,
    SYSTEM_FEATURE_DIM,
    SYSTEM_FEATURE_NAMES,
    TASK_FEATURE_DIM,
    TASK_FEATURE_NAMES,
    BranchingDQNAgent,
)
from .config import (
    BranchingDQNConfig,
    DQNConfig,
    HRLConfig,
    IntDQNConfig,
    default_device,
)


CHECKPOINT_SCHEMA_VERSION = 1


def _device_name(device=None) -> str:
    return str(device or default_device())


def _agent_payload(agent) -> dict:
    return {
        "checkpoint_kind": getattr(agent, "checkpoint_kind", None),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "obs_dim": agent.obs_dim,
        "action_dim": agent.action_dim,
        "config": asdict(agent.config),
        "q_net_state_dict": agent.q_net.state_dict(),
        "target_net_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "learn_step": agent.learn_step,
    }


def _branching_payload(agent: BranchingDQNAgent) -> dict:
    return {
        "checkpoint_kind": "branching_scheduler",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "global_features": list(GLOBAL_FEATURE_NAMES),
            "task_features": list(TASK_FEATURE_NAMES),
            "system_features": list(SYSTEM_FEATURE_NAMES),
            "dimensions": {
                "global": GLOBAL_FEATURE_DIM,
                "task": TASK_FEATURE_DIM,
                "system": SYSTEM_FEATURE_DIM,
            },
            "time_normalization": "mission_total_processing_time",
            "includes_system_cost": False,
        },
        "network_config": {
            "task_encoder": [TASK_FEATURE_DIM, 128, 64],
            "system_encoder": [SYSTEM_FEATURE_DIM, 128, 64],
            "global_encoder": [GLOBAL_FEATURE_DIM, 128, 128],
            "context_encoder": [384, 256, 128],
            "task_advantage_head": [192, 128, 1],
            "system_advantage_head": [192, 128, 1],
            "state_value_head": [128, 128, 1],
            "activation": "relu",
            "joint_value": "V+A_task+A_system",
        },
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


def save_branching_checkpoint(agent, path, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = _branching_payload(agent)
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


def load_flat_rules_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)
    values = dict(checkpoint["config"])
    values["device"] = device
    agent = FlatRuleDQNAgent(
        int(checkpoint["obs_dim"]),
        HRLConfig(**values),
    )
    if int(checkpoint["action_dim"]) != agent.action_dim:
        raise ValueError("flat-rule checkpoint action dimension must be 24.")
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


def _validate_branching_schema(checkpoint):
    if checkpoint.get("checkpoint_kind") != "branching_scheduler":
        raise ValueError("checkpoint is not a branching scheduler checkpoint.")
    feature_schema = checkpoint.get("feature_schema", {})
    if int(feature_schema.get("version", -1)) != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported branching feature schema version.")
    expected_dimensions = {
        "global": GLOBAL_FEATURE_DIM,
        "task": TASK_FEATURE_DIM,
        "system": SYSTEM_FEATURE_DIM,
    }
    if feature_schema.get("dimensions") != expected_dimensions:
        raise ValueError("branching checkpoint feature dimensions do not match.")


def _restore_branching(agent, checkpoint, load_optimizer: bool):
    agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    if load_optimizer:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.learn_step = int(checkpoint["learn_step"])
    return agent, checkpoint


def load_branching_checkpoint(path, device=None, load_optimizer=True):
    device = _device_name(device)
    checkpoint = _load_checkpoint(path, device)
    _validate_branching_schema(checkpoint)
    values = dict(checkpoint["config"])
    values["device"] = device
    agent = BranchingDQNAgent(BranchingDQNConfig(**values))
    return _restore_branching(agent, checkpoint, bool(load_optimizer))


def save_combined_checkpoint(
    path,
    architecture_agent,
    scheduler_agent,
    training_state=None,
    metadata=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_branching = isinstance(scheduler_agent, BranchingDQNAgent)
    scheduler_payload = (
        _branching_payload(scheduler_agent)
        if is_branching
        else _agent_payload(scheduler_agent)
    )
    torch.save(
        {
            "checkpoint_kind": (
                "architecture_branching" if is_branching else "architecture_scheduler"
            ),
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "scheduler_kind": scheduler_payload["checkpoint_kind"],
            "architecture": _agent_payload(architecture_agent),
            "scheduler": scheduler_payload,
            "training_state": training_state or {},
            "metadata": metadata or {},
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
    scheduler_kind = checkpoint.get(
        "scheduler_kind",
        scheduler_data.get("checkpoint_kind", "scheduler"),
    )
    if scheduler_kind == "branching_scheduler":
        _validate_branching_schema(scheduler_data)
        scheduler_values = dict(scheduler_data["config"])
        scheduler_values["device"] = device
        scheduler_agent = BranchingDQNAgent(
            BranchingDQNConfig(**scheduler_values)
        )
        _restore_branching(
            scheduler_agent,
            scheduler_data,
            bool(load_optimizer),
        )
    else:
        scheduler_values = dict(scheduler_data["config"])
        scheduler_values["device"] = device
        scheduler_agent = DQNAgent(
            int(scheduler_data["obs_dim"]),
            int(scheduler_data["action_dim"]),
            DQNConfig(**scheduler_values),
        )
        _restore(scheduler_agent, scheduler_data, bool(load_optimizer))
    return architecture_agent, scheduler_agent, checkpoint
