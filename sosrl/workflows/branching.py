"""Training and evaluation workflows for the constrained branching scheduler."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import torch

from .. import environment as env
from ..rl.branching import (
    BranchingDQNAgent,
    BranchingObservation,
    build_branching_observation,
)
from ..rl.config import BranchingDQNConfig
from ..rules import architecture as archrule
from ..gp.provider import (
    ACTION_KIND_NAMES,
    ArchitectureProvider,
    as_architecture_provider,
)
from . import scheduler as dqn
from .hierarchical import AdaptiveScenarioPool, scenario_hash, scheduler_reward


def freeze_architecture_provider(architecture_agent) -> None:
    """Put the architecture policy in inference-only mode in place."""

    policy = getattr(architecture_agent, "agent", architecture_agent)
    for network_name in ("q_net", "target_net"):
        network = getattr(policy, network_name, None)
        if network is None:
            continue
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


def _terminal_observation(
    observation: BranchingObservation,
) -> BranchingObservation:
    """Return a copy that cannot accidentally contribute a TD bootstrap."""

    terminal = observation.copy()
    terminal.pair_mask.fill(False)
    return terminal


def _store_transition(
    branching_agent: BranchingDQNAgent,
    transition,
) -> None:
    branching_agent.replay.add(*transition)


def _learn_once(
    branching_agent: BranchingDQNAgent,
    *,
    update_agent: bool,
):
    if not update_agent:
        return None
    return branching_agent.learn()


def _synchronize(policy) -> None:
    policy = getattr(policy, "agent", policy)
    device = getattr(policy, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _parameter_count(policy) -> int:
    policy = getattr(policy, "agent", policy)
    network = getattr(policy, "q_net", None)
    if network is None:
        return 0
    return int(sum(parameter.numel() for parameter in network.parameters()))


def run_branching_episode(
    mission_env: env.MissionEnv,
    architecture_provider,
    branching_agent: BranchingDQNAgent,
    *,
    scheduler_epsilon: float,
    update_scheduler: bool,
    store_experience: bool = True,
    measure_inference: bool = False,
):
    """Run one episode with a frozen top policy and a branching scheduler.

    A non-terminal lower transition remains pending until the next architecture
    action has been applied.  Consequently its ``next_observation`` is exactly
    the state from which the next lower-level action is selected.
    """

    architecture_provider = as_architecture_provider(architecture_provider)
    freeze_architecture_provider(architecture_provider)
    architecture_rule_counts = np.zeros(
        archrule.ArchitectureRule.RULE_NUM,
        dtype=np.int32,
    )
    architecture_action_counts = np.zeros(len(ACTION_KIND_NAMES), dtype=np.int32)
    architecture_add_system_counts = np.zeros(mission_env.N, dtype=np.int32)
    architecture_remove_system_counts = np.zeros(mission_env.N, dtype=np.int32)
    system_action_counts = np.zeros(mission_env.N, dtype=np.int32)
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
            _synchronize(architecture_provider)
            inference_start = perf_counter()
        decision = architecture_provider.act(mission_env)
        if measure_inference:
            _synchronize(architecture_provider)
            architecture_inference_seconds += perf_counter() - inference_start
        architecture_decisions += 1
        architecture_candidate_total += int(decision.candidate_count)
        architecture_candidate_max = max(
            architecture_candidate_max, int(decision.candidate_count)
        )
        if not decision.valid:
            info = {"success": False, "dead_end": True}
            if pending_scheduler is not None:
                previous_obs, previous_action, previous_reward = pending_scheduler
                terminal_obs = _terminal_observation(
                    build_branching_observation(mission_env)
                )
                terminal_reward = float(previous_reward - 2.0)
                scheduler_total -= 2.0
                if store_experience:
                    _store_transition(
                        branching_agent,
                        (
                            previous_obs,
                            previous_action,
                            terminal_reward,
                            terminal_obs,
                            True,
                        ),
                    )
                    loss = _learn_once(
                        branching_agent,
                        update_agent=update_scheduler,
                    )
                    if loss is not None:
                        last_scheduler_loss = loss
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

        branching_observation = build_branching_observation(mission_env)
        has_pair = bool(np.any(branching_observation.pair_mask))
        if not has_pair:
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
                    _store_transition(
                        branching_agent,
                        (
                            previous_obs,
                            previous_action,
                            terminal_reward,
                            _terminal_observation(branching_observation),
                            True,
                        ),
                    )
                    loss = _learn_once(
                        branching_agent,
                        update_agent=update_scheduler,
                    )
                    if loss is not None:
                        last_scheduler_loss = loss
                pending_scheduler = None
            else:
                scheduler_total -= 2.0
            break

        stored_transition = False
        if pending_scheduler is not None:
            if store_experience:
                _store_transition(
                    branching_agent,
                    (*pending_scheduler, branching_observation, False),
                )
                stored_transition = True
            pending_scheduler = None

        if measure_inference:
            _synchronize(branching_agent)
            inference_start = perf_counter()
        branching_action = branching_agent.select_action(
            branching_observation,
            epsilon=scheduler_epsilon,
        )
        if measure_inference:
            _synchronize(branching_agent)
            scheduler_inference_seconds += perf_counter() - inference_start
        scheduler_decisions += 1
        system_action_counts[branching_action.sys_idx] += 1
        try:
            env_action = branching_agent.encode_environment_action(
                mission_env,
                branching_action,
            )
        except RuntimeError:
            invalid_action_count += 1
            raise

        _, base_reward, terminated, _, info = mission_env.step(env_action)
        if not info.get("valid", False):
            invalid_action_count += 1
            raise RuntimeError("branching scheduler produced an invalid assignment.")
        assignment_steps += 1
        success = bool(info.get("success", False))
        dead_end = bool(info.get("dead_end", False))
        schedule_r = scheduler_reward(base_reward, success, dead_end)
        scheduler_total += schedule_r

        if terminated:
            next_observation = _terminal_observation(
                build_branching_observation(mission_env)
            )
            if store_experience:
                _store_transition(
                    branching_agent,
                    (
                        branching_observation,
                        branching_action,
                        schedule_r,
                        next_observation,
                        True,
                    ),
                )
                stored_transition = True
        else:
            pending_scheduler = (
                branching_observation,
                branching_action,
                schedule_r,
            )

        if stored_transition:
            loss = _learn_once(
                branching_agent,
                update_agent=update_scheduler,
            )
            if loss is not None:
                last_scheduler_loss = loss

        if terminated:
            break

    return {
        "scheduler_reward": float(scheduler_total),
        "scheduler_loss": last_scheduler_loss,
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
        "inference_measured": bool(measure_inference),
    }


def branching_episode_row(
    episode,
    category,
    mission_env,
    result,
    epsilon,
    *,
    total_env_steps=None,
):
    row = {
        "episode": int(episode),
        "category": category,
        "scheduler_reward": result["scheduler_reward"],
        "scheduler_loss": result["scheduler_loss"],
        "success": result["success"],
        "dead_end": result["dead_end"],
        "makespan": float(mission_env.state.current_makespan),
        "net_cost": float(mission_env.net_cost),
        "active_cost": float(mission_env.active_cost),
        "total_refund": float(mission_env.total_refund),
        "architecture_changes": int(mission_env.architecture_change_count),
        "budget_violation": bool(mission_env.net_cost > mission_env.budget),
        "assigned_ops": int(np.sum(mission_env.state.task_op_idx)),
        "epsilon": float(epsilon),
        "assignment_steps": int(result["assignment_steps"]),
        "invalid_action_count": int(result["invalid_action_count"]),
        "provider_invariant_violations": int(
            result["provider_invariant_violations"]
        ),
    }
    if total_env_steps is not None:
        row["total_env_steps"] = int(total_env_steps)
    if result.get("inference_measured", False):
        architecture_decisions = max(int(result["architecture_decisions"]), 1)
        scheduler_decisions = max(int(result["scheduler_decisions"]), 1)
        row["mean_architecture_inference_ms"] = (
            1000.0
            * float(result["architecture_inference_seconds"])
            / architecture_decisions
        )
        row["mean_scheduler_inference_ms"] = (
            1000.0
            * float(result["scheduler_inference_seconds"])
            / scheduler_decisions
        )
    row.update(mission_env.cost_metrics())
    for index, name in enumerate(archrule.ArchitectureRule.RULE_NAMES):
        row[f"arch_{name.lower()}_count"] = int(
            result["architecture_rule_counts"][index]
        )
    for index, name in enumerate(ACTION_KIND_NAMES):
        row[f"arch_{name}_count"] = int(
            result["architecture_action_counts"][index]
        )
    row["mean_architecture_candidate_count"] = (
        float(result["architecture_candidate_total"])
        / max(int(result["architecture_decisions"]), 1)
    )
    row["max_architecture_candidate_count"] = int(
        result["architecture_candidate_max"]
    )
    for sys_idx in range(mission_env.N):
        row[f"system_{sys_idx}_added_count"] = int(
            result["architecture_add_system_counts"][sys_idx]
        )
        row[f"system_{sys_idx}_removed_count"] = int(
            result["architecture_remove_system_counts"][sys_idx]
        )
        row[f"system_{sys_idx}_used_count"] = int(
            result["system_action_counts"][sys_idx]
        )
    return row


def train_branching_scheduler(
    config: BranchingDQNConfig,
    architecture_agent,
    scenario_pool: AdaptiveScenarioPool,
    *,
    branching_agent: BranchingDQNAgent | None = None,
):
    """Train only the lower scheduler against a frozen architecture provider."""

    dqn.set_seed(config.seed)
    freeze_architecture_provider(architecture_agent)
    if branching_agent is None:
        branching_agent = BranchingDQNAgent(config)
    epsilon = float(config.epsilon_start)
    history = []
    total_env_steps = 0

    for episode in range(config.episodes):
        if (
            config.max_env_steps is not None
            and total_env_steps >= config.max_env_steps
        ):
            break
        architecture, mission, category = scenario_pool.sample()
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        result = run_branching_episode(
            mission_env,
            architecture_agent,
            branching_agent,
            scheduler_epsilon=epsilon,
            update_scheduler=True,
        )
        total_env_steps += int(result["assignment_steps"])
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        history.append(
            branching_episode_row(
                episode,
                category,
                mission_env,
                result,
                epsilon,
                total_env_steps=total_env_steps,
            )
        )

    training_state = {
        "episode": len(history),
        "epsilon": float(epsilon),
        "total_env_steps": int(total_env_steps),
    }
    return branching_agent, history, training_state


def evaluate_branching(
    architecture_agent,
    branching_agent,
    scenarios,
    *,
    budget=8000.0,
    refund_rate=0.8,
):
    architecture_agent = as_architecture_provider(architecture_agent)
    freeze_architecture_provider(architecture_agent)
    branching_agent.q_net.eval()
    parameter_count = _parameter_count(architecture_agent) + _parameter_count(
        branching_agent
    )
    device = torch.device(branching_agent.device)
    results = []
    for episode, scenario in enumerate(scenarios):
        if len(scenario) == 3:
            architecture, mission, category = scenario
        else:
            architecture, mission = scenario
            category = "evaluation"
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=budget,
            refund_rate=refund_rate,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        result = run_branching_episode(
            mission_env,
            architecture_agent,
            branching_agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
            store_experience=False,
            measure_inference=True,
        )
        row = branching_episode_row(
            episode,
            category,
            mission_env,
            result,
            0.0,
        )
        row["scenario_hash"] = scenario_hash(architecture, mission)
        row["policy_parameter_count"] = parameter_count
        row["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        )
        results.append(row)
    return results
