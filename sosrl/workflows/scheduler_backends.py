"""Interchangeable lower-level schedulers for architecture-policy rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from .. import domain as syn
from .. import environment as env_module
from ..gp.artifact import sha256_file
from ..gp.provider import ACTION_KIND_NAMES, as_architecture_provider
from ..rl.agent import DQNAgent
from ..rl.branching import (
    BranchingAction,
    build_branching_observation,
    collate_branching_observations,
)
from ..rl.checkpoint import load_branching_checkpoint
from ..rules import architecture as architecture_rules
from . import scheduler as scheduler_workflow
from .branching import (
    SS_REPLACE_REASON_NAMES,
    SS_TRIGGER_NAMES,
    freeze_architecture_provider,
    run_branching_episode,
)
from .hierarchical import scheduler_reward


SCHEDULER_BACKEND_KINDS = ("rule-dqn", "branching-dqn")


def freeze_scheduler_agent(agent) -> None:
    """Put either scheduler family in inference-only mode."""

    for network in (agent.q_net, agent.target_net):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


def scheduler_parameter_hash(agent) -> str:
    """Hash frozen scheduler parameters independently of checkpoint packaging."""

    digest = hashlib.sha256()
    for network_name in ("q_net", "target_net"):
        state = getattr(agent, network_name).state_dict()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(network_name.encode("utf-8"))
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _synchronize(agent) -> None:
    if torch.device(agent.device).type == "cuda":
        torch.cuda.synchronize(torch.device(agent.device))


def _rule_terminal_observation(mission_env: env_module.MissionEnv) -> np.ndarray:
    return np.zeros_like(mission_env.schedule_observation(), dtype=np.float32)


def run_rule_dqn_episode(
    mission_env: env_module.MissionEnv,
    architecture_provider,
    scheduler_agent: DQNAgent,
    *,
    scheduler_epsilon: float,
    update_scheduler: bool,
    store_experience: bool = True,
    measure_inference: bool = False,
) -> dict[str, Any]:
    """Run one architecture/scheduling episode with a rule-selection DQN.

    The architecture decision occurs before every lower-level scheduling
    decision, matching :func:`run_branching_episode`.  A non-terminal rule-DQN
    transition remains pending until the next architecture action has been
    applied, so both scheduler families observe the same environment timing.
    """

    architecture_provider = as_architecture_provider(architecture_provider)
    freeze_architecture_provider(architecture_provider)
    rule_class = scheduler_workflow.get_rule_class(scheduler_agent.config.rule_set)
    if int(scheduler_agent.action_dim) != int(rule_class.RULE_NUM):
        raise ValueError("rule-DQN action dimension does not match its rule set.")
    if int(scheduler_agent.obs_dim) != int(mission_env.schedule_observation().shape[0]):
        raise ValueError("rule-DQN observation dimension does not match the environment.")
    rule_policy = rule_class(mission_env)

    architecture_rule_counts = np.zeros(
        architecture_rules.ArchitectureRule.RULE_NUM, dtype=np.int32
    )
    architecture_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int32)
    architecture_add_system_counts = np.zeros(mission_env.N, dtype=np.int32)
    architecture_remove_system_counts = np.zeros(mission_env.N, dtype=np.int32)
    system_action_counts = np.zeros(mission_env.N, dtype=np.int32)
    scheduler_rule_counts = np.zeros(rule_class.RULE_NUM, dtype=np.int32)
    ss_trigger_counts = {name: 0 for name in SS_TRIGGER_NAMES}
    ss_replace_reason_counts = {name: 0 for name in SS_REPLACE_REASON_NAMES}
    ss_target_capability_counts = np.zeros(
        len(syn.func_type2idx), dtype=np.int32
    )
    ss_capacity_delta_total = 0.0
    ss_net_cost_delta_total = 0.0
    ss_diagnostics_observed = False

    pending_scheduler = None
    scheduler_total = 0.0
    last_scheduler_loss = None
    assignment_steps = 0
    invalid_action_count = 0
    provider_invariant_violations = 0
    architecture_inference_seconds = 0.0
    scheduler_inference_seconds = 0.0
    architecture_decisions = 0
    architecture_candidate_total = 0
    architecture_candidate_max = 0
    scheduler_decisions = 0
    info = {"success": False, "dead_end": False}

    for _ in range(mission_env.T * mission_env.O + mission_env.N):
        if measure_inference:
            inference_start = perf_counter()
        decision = architecture_provider.act(mission_env)
        if measure_inference:
            architecture_inference_seconds += perf_counter() - inference_start
        architecture_decisions += 1
        architecture_candidate_total += int(decision.candidate_count)
        architecture_candidate_max = max(
            architecture_candidate_max, int(decision.candidate_count)
        )

        trigger = decision.diagnostics.get("trigger")
        if trigger in ss_trigger_counts:
            ss_diagnostics_observed = True
            ss_trigger_counts[str(trigger)] += 1
            replace_reason = decision.diagnostics.get("replace_reason")
            if replace_reason in ss_replace_reason_counts:
                ss_replace_reason_counts[str(replace_reason)] += 1
            target_capability = int(decision.diagnostics.get("target_capability", -1))
            if 0 <= target_capability < len(ss_target_capability_counts):
                ss_target_capability_counts[target_capability] += 1
            ss_capacity_delta_total += float(
                decision.diagnostics.get("capacity_delta", 0.0)
            )
            ss_net_cost_delta_total += float(
                decision.diagnostics.get("net_cost_delta", 0.0)
            )

        if not decision.valid:
            info = {"success": False, "dead_end": True}
            terminal_observation = _rule_terminal_observation(mission_env)
            if pending_scheduler is not None:
                previous_obs, previous_action, previous_reward = pending_scheduler
                terminal_reward = float(previous_reward - 2.0)
                scheduler_total -= 2.0
                if store_experience:
                    scheduler_agent.replay.add(
                        previous_obs,
                        previous_action,
                        terminal_reward,
                        terminal_observation,
                        True,
                        np.zeros(rule_class.RULE_NUM, dtype=np.float32),
                    )
                    if update_scheduler:
                        last_scheduler_loss = scheduler_agent.learn()
                pending_scheduler = None
            else:
                scheduler_total -= 2.0
            break

        architecture_action_counts[ACTION_KIND_NAMES.index(decision.action.kind)] += 1
        if decision.action.new_system is not None:
            architecture_add_system_counts[int(decision.action.new_system)] += 1
        if decision.action.old_system is not None:
            architecture_remove_system_counts[int(decision.action.old_system)] += 1
        manual_rule_idx = decision.diagnostics.get("manual_rule_index")
        if manual_rule_idx is not None:
            architecture_rule_counts[int(manual_rule_idx)] += 1

        schedule_observation = mission_env.schedule_observation()
        schedule_mask = scheduler_workflow.rule_action_mask(
            mission_env, rule_class.RULE_NUM
        )
        if not np.any(schedule_mask):
            globally_feasible = mission_env.has_feasible_assignment(
                np.ones(mission_env.N, dtype=bool)
            )
            if globally_feasible:
                provider_invariant_violations += 1
            info = {
                "success": False,
                "dead_end": True,
                "provider_invariant_violation": globally_feasible,
            }
            if pending_scheduler is not None:
                previous_obs, previous_action, previous_reward = pending_scheduler
                terminal_reward = float(previous_reward - 2.0)
                scheduler_total -= 2.0
                if store_experience:
                    scheduler_agent.replay.add(
                        previous_obs,
                        previous_action,
                        terminal_reward,
                        _rule_terminal_observation(mission_env),
                        True,
                        np.zeros(rule_class.RULE_NUM, dtype=np.float32),
                    )
                    if update_scheduler:
                        loss = scheduler_agent.learn()
                        if loss is not None:
                            last_scheduler_loss = loss
                pending_scheduler = None
            else:
                scheduler_total -= 2.0
            break

        if pending_scheduler is not None:
            if store_experience:
                scheduler_agent.replay.add(
                    *pending_scheduler,
                    schedule_observation,
                    False,
                    schedule_mask,
                )
                if update_scheduler:
                    loss = scheduler_agent.learn()
                    if loss is not None:
                        last_scheduler_loss = loss
            pending_scheduler = None

        if measure_inference:
            _synchronize(scheduler_agent)
            inference_start = perf_counter()
        rule_action = scheduler_agent.select_action(
            schedule_observation, schedule_mask, scheduler_epsilon
        )
        if measure_inference:
            _synchronize(scheduler_agent)
            scheduler_inference_seconds += perf_counter() - inference_start
        scheduler_decisions += 1
        scheduler_rule_counts[int(rule_action)] += 1
        environment_action = rule_policy.to_env_action(rule_action)
        decoded = mission_env.decode_assignment(environment_action)
        system_action_counts[int(decoded["sys_idx"])] += 1

        next_observation, base_reward, terminated, _, info = mission_env.step(
            environment_action
        )
        if not info.get("valid", False):
            invalid_action_count += 1
            raise RuntimeError("rule-DQN scheduler produced an invalid assignment.")
        assignment_steps += 1
        success = bool(info.get("success", False))
        dead_end = bool(info.get("dead_end", False))
        schedule_reward = scheduler_reward(base_reward, success, dead_end)
        scheduler_total += schedule_reward

        if terminated:
            if store_experience:
                scheduler_agent.replay.add(
                    schedule_observation,
                    rule_action,
                    schedule_reward,
                    np.asarray(next_observation, dtype=np.float32),
                    True,
                    np.zeros(rule_class.RULE_NUM, dtype=np.float32),
                )
                if update_scheduler:
                    loss = scheduler_agent.learn()
                    if loss is not None:
                        last_scheduler_loss = loss
        else:
            pending_scheduler = (
                schedule_observation,
                rule_action,
                schedule_reward,
            )

        if terminated:
            break

    return {
        "scheduler_reward": float(scheduler_total),
        "scheduler_loss": last_scheduler_loss,
        "scheduler_rule_counts": scheduler_rule_counts,
        "architecture_rule_counts": architecture_rule_counts,
        "architecture_action_counts": architecture_action_counts,
        "architecture_add_system_counts": architecture_add_system_counts,
        "architecture_remove_system_counts": architecture_remove_system_counts,
        "system_action_counts": system_action_counts,
        "success": bool(info.get("success", False)),
        "dead_end": bool(info.get("dead_end", False)),
        "provider_invariant_violations": int(provider_invariant_violations),
        "invalid_action_count": int(invalid_action_count),
        "assignment_steps": int(assignment_steps),
        "architecture_inference_seconds": float(architecture_inference_seconds),
        "scheduler_inference_seconds": float(scheduler_inference_seconds),
        "architecture_decisions": int(architecture_decisions),
        "architecture_candidate_total": int(architecture_candidate_total),
        "architecture_candidate_max": int(architecture_candidate_max),
        "scheduler_decisions": int(scheduler_decisions),
        "ss_trigger_counts": ss_trigger_counts,
        "ss_replace_reason_counts": ss_replace_reason_counts,
        "ss_target_capability_counts": ss_target_capability_counts,
        "ss_capacity_delta_total": float(ss_capacity_delta_total),
        "ss_net_cost_delta_total": float(ss_net_cost_delta_total),
        "ss_diagnostics_observed": bool(ss_diagnostics_observed),
        "inference_measured": bool(measure_inference),
    }


class SchedulerBackend(Protocol):
    kind: str
    agent: Any
    checkpoint_path: Path

    @property
    def device(self) -> torch.device: ...

    def run_episode(
        self,
        mission_env,
        architecture_provider,
        *,
        measure_inference: bool = False,
    ) -> dict[str, Any]: ...

    def has_feasible_action(self, mission_env) -> bool: ...

    def select_environment_actions(
        self, mission_envs: Sequence[env_module.MissionEnv]
    ) -> list[int]: ...

    def provenance(self) -> dict[str, Any]: ...


@dataclass
class BranchingDQNSchedulerBackend:
    agent: Any
    checkpoint_path: Path
    kind: str = "branching-dqn"

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str):
        checkpoint_path = Path(path).resolve()
        agent, _ = load_branching_checkpoint(
            checkpoint_path, device=device, load_optimizer=False
        )
        freeze_scheduler_agent(agent)
        return cls(agent=agent, checkpoint_path=checkpoint_path)

    @property
    def device(self) -> torch.device:
        return torch.device(self.agent.device)

    def run_episode(
        self,
        mission_env,
        architecture_provider,
        *,
        measure_inference: bool = False,
    ) -> dict[str, Any]:
        return run_branching_episode(
            mission_env,
            architecture_provider,
            self.agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
            store_experience=False,
            measure_inference=measure_inference,
        )

    def has_feasible_action(self, mission_env) -> bool:
        return bool(np.any(build_branching_observation(mission_env).pair_mask))

    def select_environment_actions(self, mission_envs) -> list[int]:
        observations = [build_branching_observation(item) for item in mission_envs]
        batch = collate_branching_observations(observations)
        tensors = batch.to_torch(self.agent.device)
        with torch.no_grad():
            scores = self.agent.q_net(**tensors).scores
            pair_mask = tensors["pair_mask"]
            flat_indices = (
                scores.masked_fill(~pair_mask, -torch.inf)
                .reshape(scores.shape[0], -1)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
        padded_systems = int(scores.shape[2])
        actions = []
        for mission_env, observation, flat_index in zip(
            mission_envs, observations, flat_indices, strict=True
        ):
            task_idx, sys_idx = divmod(int(flat_index), padded_systems)
            op_idx = int(observation.task_op_indices[task_idx])
            branching_action = BranchingAction(
                task_idx=task_idx,
                sys_idx=sys_idx,
                op_idx=op_idx,
                decision_version=int(observation.decision_version),
            )
            actions.append(
                self.agent.encode_environment_action(mission_env, branching_action)
            )
        return actions

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "parameter_sha256": scheduler_parameter_hash(self.agent),
        }


@dataclass
class RuleDQNSchedulerBackend:
    agent: DQNAgent
    checkpoint_path: Path
    kind: str = "rule-dqn"

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str):
        checkpoint_path = Path(path).resolve()
        agent, _ = DQNAgent.load_checkpoint(
            checkpoint_path, device=device, load_optimizer=False
        )
        rule_class = scheduler_workflow.get_rule_class(agent.config.rule_set)
        if int(agent.action_dim) != int(rule_class.RULE_NUM):
            raise ValueError("rule-DQN checkpoint action dimension is incompatible.")
        freeze_scheduler_agent(agent)
        return cls(agent=agent, checkpoint_path=checkpoint_path)

    @property
    def device(self) -> torch.device:
        return torch.device(self.agent.device)

    def run_episode(
        self,
        mission_env,
        architecture_provider,
        *,
        measure_inference: bool = False,
    ) -> dict[str, Any]:
        return run_rule_dqn_episode(
            mission_env,
            architecture_provider,
            self.agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
            store_experience=False,
            measure_inference=measure_inference,
        )

    def has_feasible_action(self, mission_env) -> bool:
        return bool(np.any(mission_env.valid_assignment_mask()))

    def select_environment_actions(self, mission_envs) -> list[int]:
        rule_class = scheduler_workflow.get_rule_class(self.agent.config.rule_set)
        observations = np.stack(
            [item.schedule_observation() for item in mission_envs]
        ).astype(np.float32)
        if observations.shape[1] != int(self.agent.obs_dim):
            raise ValueError("rule-DQN observation dimension does not match environment.")
        masks = np.stack(
            [
                scheduler_workflow.rule_action_mask(item, rule_class.RULE_NUM)
                for item in mission_envs
            ]
        ).astype(bool)
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.agent.device
        )
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.agent.device)
        with torch.no_grad():
            q_values = self.agent.q_net(observation_tensor)
            rule_actions = (
                q_values.masked_fill(~mask_tensor, -torch.inf)
                .argmax(dim=1)
                .detach()
                .cpu()
                .numpy()
            )
        return [
            rule_class(mission_env).to_env_action(int(rule_action))
            for mission_env, rule_action in zip(
                mission_envs, rule_actions, strict=True
            )
        ]

    def provenance(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path),
            "parameter_sha256": scheduler_parameter_hash(self.agent),
            "rule_set": str(self.agent.config.rule_set),
            "observation_dim": int(self.agent.obs_dim),
            "action_dim": int(self.agent.action_dim),
        }


def load_scheduler_backend(
    kind: str,
    checkpoint_path: str | Path,
    *,
    device: str,
) -> SchedulerBackend:
    normalized = str(kind).lower()
    if normalized == "branching-dqn":
        return BranchingDQNSchedulerBackend.from_checkpoint(
            checkpoint_path, device=device
        )
    if normalized == "rule-dqn":
        return RuleDQNSchedulerBackend.from_checkpoint(
            checkpoint_path, device=device
        )
    raise ValueError(f"unknown scheduler backend: {kind!r}")
