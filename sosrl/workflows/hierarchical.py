from __future__ import annotations

import hashlib
import json
import random
from time import perf_counter

import numpy as np
import torch

from .. import domain as syn
from .. import environment as env
from ..rl.agent import ArchitectureDQNAgent
from ..rl.checkpoint import load_combined_checkpoint, save_combined_checkpoint
from ..rl.config import HRLConfig
from ..rl.replay import ArchitectureReplayBuffer, NStepAccumulator
from ..rules import architecture as archrule
from ..rules import scheduling as rule
from . import scheduler as dqn


class AdaptiveScenarioPool:
    CATEGORIES = (
        "feasible_suboptimal",
        "capacity_tight",
        "missing_capability",
        "redundant_overbudget",
    )
    WEIGHTS = (0.50, 0.20, 0.15, 0.15)

    def __init__(self, size: int, config: HRLConfig):
        self.config = config
        self.scenarios = []
        sampler = dqn.ScenarioPool(size=0, cost_limit=config.budget)
        for _ in range(int(size)):
            mission = syn.build_mission_from_config(syn.CONFIG)
            feasible = tuple(sampler.sample_arch(mission))
            category = random.choices(self.CATEGORIES, self.WEIGHTS, k=1)[0]
            architecture = self._make_initial_architecture(feasible, mission, category)
            self.scenarios.append((architecture, mission, category))

    def _make_initial_architecture(self, feasible, mission, category):
        selected = {int(system.index) for system in feasible}
        if category == "missing_capability":
            first_type = int(mission[0].operations[0].func_type)
            selected = {
                index
                for index in selected
                if int(env.FULL_SOS[index].func_type) != first_type
            }
        elif category == "capacity_tight":
            for func_type in syn.func_type2idx.values():
                matching = [
                    index
                    for index in selected
                    if int(env.FULL_SOS[index].func_type) == int(func_type)
                ]
                if len(matching) > 1:
                    keep = min(
                        matching,
                        key=lambda index: (
                            env.FULL_SOS[index].available_until
                            - env.FULL_SOS[index].available_from,
                            index,
                        ),
                    )
                    selected.difference_update(matching)
                    selected.add(keep)
        elif category == "feasible_suboptimal":
            inactive = [
                index for index in range(len(env.FULL_SOS)) if index not in selected
            ]
            if inactive:
                selected.add(random.choice(inactive))
        else:
            inactive = sorted(
                (index for index in range(len(env.FULL_SOS)) if index not in selected),
                key=lambda index: (-env.FULL_SOS[index].cost, index),
            )
            for index in inactive[:2]:
                selected.add(index)
        return tuple(env.FULL_SOS[index] for index in sorted(selected))

    def get(self, index):
        return self.scenarios[int(index) % len(self.scenarios)]

    def sample(self):
        return random.choice(self.scenarios)


def budget_potential(cost: float, budget: float) -> float:
    ratio = float(cost) / max(float(budget), 1.0)
    return 20.0 * max(0.0, ratio - 1.0) ** 2


def architecture_reward(
    mission_env: env.MissionEnv,
    old_makespan: float,
    old_cost: float,
    changed: bool,
    success: bool,
    dead_end: bool,
) -> float:
    makespan_delta = float(mission_env.state.current_makespan) - float(old_makespan)
    cost_delta = float(mission_env.net_cost) - float(old_cost)
    potential_delta = budget_potential(
        mission_env.net_cost,
        mission_env.budget,
    ) - budget_potential(old_cost, mission_env.budget)
    terminal = 1.0 if success else (-2.0 if dead_end else 0.0)
    return float(
        -10.0 * makespan_delta / mission_env.state.M
        - cost_delta / mission_env.budget
        - potential_delta
        - 0.01 * float(changed)
        + terminal
    )


def scheduler_reward(base_reward: float, success: bool, dead_end: bool) -> float:
    terminal = 1.0 if success else (-2.0 if dead_end else 0.0)
    return float(base_reward + terminal)


def _add_n_step(agent, accumulator, transition):
    for emitted in accumulator.append(transition):
        agent.replay.add(*emitted)


def _synchronize_policy(policy) -> None:
    device = getattr(policy, "device", None)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _policy_parameter_count(policy) -> int:
    network = getattr(policy, "q_net", None)
    if network is None:
        return 0
    return int(sum(parameter.numel() for parameter in network.parameters()))


def run_episode(
    mission_env: env.MissionEnv,
    architecture_agent: ArchitectureDQNAgent,
    scheduler_agent: dqn.DQNAgent,
    *,
    architecture_epsilon: float,
    scheduler_epsilon: float,
    update_architecture: bool,
    update_scheduler: bool,
    store_experience: bool = True,
    measure_inference: bool = False,
):
    architecture_policy = archrule.ArchitectureRule(mission_env)
    scheduler_policy = rule.Rule(mission_env)
    accumulator = (
        NStepAccumulator(
            architecture_agent.config.n_step,
            architecture_agent.config.gamma,
        )
        if store_experience
        else None
    )
    rule_counts = np.zeros(archrule.ArchitectureRule.RULE_NUM, dtype=np.int32)
    schedule_rule_counts = np.zeros(rule.Rule.RULE_NUM, dtype=np.int32)
    pending_scheduler = None
    architecture_total = 0.0
    scheduler_total = 0.0
    last_arch_loss = None
    last_scheduler_loss = None
    info = {"success": False, "dead_end": False}
    architecture_inference_seconds = 0.0
    scheduler_inference_seconds = 0.0
    architecture_decisions = 0
    scheduler_decisions = 0

    for _ in range(mission_env.T * mission_env.O + mission_env.N):
        arch_obs = mission_env.architecture_observation()
        arch_mask = architecture_policy.action_mask()
        if not np.any(arch_mask):
            info = {"success": False, "dead_end": True}
            if pending_scheduler is not None and store_experience:
                scheduler_agent.replay.add(
                    *pending_scheduler,
                    mission_env.schedule_observation(),
                    True,
                    np.zeros(rule.Rule.RULE_NUM, dtype=np.float32),
                )
                pending_scheduler = None
            break

        if measure_inference:
            _synchronize_policy(architecture_agent)
            inference_start = perf_counter()
        arch_action = architecture_agent.select_action(
            arch_obs,
            arch_mask,
            architecture_epsilon,
        )
        if measure_inference:
            _synchronize_policy(architecture_agent)
            architecture_inference_seconds += perf_counter() - inference_start
        architecture_decisions += 1
        rule_counts[arch_action] += 1
        old_makespan = float(mission_env.state.current_makespan)
        old_cost = float(mission_env.net_cost)
        arch_info = architecture_policy.apply(arch_action)
        if not arch_info.get("valid", False):
            raise RuntimeError("architecture agent selected a masked action.")

        schedule_obs = mission_env.schedule_observation()
        schedule_mask = dqn.rule_action_mask(mission_env, rule.Rule.RULE_NUM)
        if pending_scheduler is not None and store_experience:
            scheduler_agent.replay.add(
                *pending_scheduler,
                schedule_obs,
                False,
                schedule_mask,
            )
            if update_scheduler:
                loss = scheduler_agent.learn()
                if loss is not None:
                    last_scheduler_loss = loss
        pending_scheduler = None
        if not np.any(schedule_mask):
            info = {"success": False, "dead_end": True}
            arch_next = mission_env.architecture_observation()
            arch_next_mask = np.zeros_like(arch_mask)
            reward = architecture_reward(
                mission_env,
                old_makespan,
                old_cost,
                bool(arch_info["changed"]),
                False,
                True,
            )
            architecture_total += reward
            if store_experience:
                _add_n_step(
                    architecture_agent,
                    accumulator,
                    (arch_obs, arch_action, reward, arch_next, True, arch_next_mask),
                )
            break


        if measure_inference:
            _synchronize_policy(scheduler_agent)
            inference_start = perf_counter()
        schedule_action = scheduler_agent.select_action(
            schedule_obs,
            schedule_mask,
            scheduler_epsilon,
        )
        if measure_inference:
            _synchronize_policy(scheduler_agent)
            scheduler_inference_seconds += perf_counter() - inference_start
        scheduler_decisions += 1
        schedule_rule_counts[schedule_action] += 1
        env_action = scheduler_policy.to_env_action(schedule_action)
        next_schedule_obs, base_reward, terminated, _, info = mission_env.step(env_action)
        success = bool(info.get("success", False))
        dead_end = bool(info.get("dead_end", False))
        schedule_r = scheduler_reward(base_reward, success, dead_end)
        scheduler_total += schedule_r

        if terminated:
            if store_experience:
                scheduler_agent.replay.add(
                    schedule_obs,
                    schedule_action,
                    schedule_r,
                    next_schedule_obs,
                    True,
                    np.zeros(rule.Rule.RULE_NUM, dtype=np.float32),
                )
            if update_scheduler:
                loss = scheduler_agent.learn()
                if loss is not None:
                    last_scheduler_loss = loss
        else:
            # The next lower-level state is only known after the next upper action.
            pending_scheduler = (schedule_obs, schedule_action, schedule_r)

        arch_next = mission_env.architecture_observation()
        next_arch_mask = (
            np.zeros(archrule.ArchitectureRule.RULE_NUM, dtype=np.float32)
            if terminated
            else architecture_policy.action_mask()
        )
        arch_r = architecture_reward(
            mission_env,
            old_makespan,
            old_cost,
            bool(arch_info["changed"]),
            success,
            dead_end,
        )
        architecture_total += arch_r
        if store_experience:
            _add_n_step(
                architecture_agent,
                accumulator,
                (arch_obs, arch_action, arch_r, arch_next, terminated, next_arch_mask),
            )
        if update_architecture:
            loss = architecture_agent.learn()
            if loss is not None:
                last_arch_loss = loss
        if terminated:
            break

    return {
        "architecture_reward": float(architecture_total),
        "scheduler_reward": float(scheduler_total),
        "architecture_loss": last_arch_loss,
        "scheduler_loss": last_scheduler_loss,
        "architecture_rule_counts": rule_counts,
        "scheduler_rule_counts": schedule_rule_counts,
        "success": bool(info.get("success", False)),
        "dead_end": bool(info.get("dead_end", False)),
        "architecture_inference_seconds": float(architecture_inference_seconds),
        "scheduler_inference_seconds": float(scheduler_inference_seconds),
        "architecture_decisions": int(architecture_decisions),
        "scheduler_decisions": int(scheduler_decisions),
        "inference_measured": bool(measure_inference),
    }


def train_architecture(
    config: HRLConfig,
    scheduler_agent: dqn.DQNAgent,
    scenario_pool: AdaptiveScenarioPool,
):
    dqn.set_seed(config.seed)
    architecture, mission, _ = scenario_pool.get(0)
    probe = env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=config.budget,
        refund_rate=config.refund_rate,
    )
    architecture_agent = ArchitectureDQNAgent(
        probe.architecture_observation_space.shape[0],
        config,
    )
    epsilon = config.epsilon_start
    history = []
    for episode in range(config.episodes):
        architecture, mission, category = scenario_pool.sample()
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        result = run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=epsilon,
            scheduler_epsilon=0.0,
            update_architecture=True,
            update_scheduler=False,
        )
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        history.append(
            episode_row(episode, category, mission_env, result, epsilon)
        )
    return architecture_agent, history


def finetune(
    config: HRLConfig,
    architecture_agent: ArchitectureDQNAgent,
    scheduler_agent: dqn.DQNAgent,
    scenario_pool: AdaptiveScenarioPool,
):
    for group in scheduler_agent.optimizer.param_groups:
        group["lr"] = config.scheduler_finetune_lr
    history = []
    epsilon = config.epsilon_end
    for episode in range(config.episodes):
        architecture, mission, category = scenario_pool.sample()
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        update_architecture = episode % 10 < 8
        result = run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=epsilon,
            scheduler_epsilon=epsilon,
            update_architecture=update_architecture,
            update_scheduler=not update_architecture,
        )
        row = episode_row(episode, category, mission_env, result, epsilon)
        row["updated_policy"] = (
            "architecture" if update_architecture else "scheduler"
        )
        history.append(row)
    return history


def episode_row(episode, category, mission_env, result, epsilon):
    row = {
        "episode": int(episode),
        "category": category,
        "architecture_reward": result["architecture_reward"],
        "scheduler_reward": result["scheduler_reward"],
        "architecture_loss": result["architecture_loss"],
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
    }
    row.update(mission_env.cost_metrics())
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
    for index, name in enumerate(archrule.ArchitectureRule.RULE_NAMES):
        row[f"arch_{name.lower()}_count"] = int(
            result["architecture_rule_counts"][index]
        )
    for index, name in enumerate(rule.Rule.RULE_NAMES):
        row[f"schedule_{name.lower()}_count"] = int(
            result["scheduler_rule_counts"][index]
        )
    return row


def evaluate_hrl(
    architecture_agent,
    scheduler_agent,
    scenarios,
    budget=8000.0,
    refund_rate=0.8,
):
    parameter_count = _policy_parameter_count(
        architecture_agent
    ) + _policy_parameter_count(scheduler_agent)
    device = torch.device(scheduler_agent.device)
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
        result = run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=0.0,
            scheduler_epsilon=0.0,
            update_architecture=False,
            update_scheduler=False,
            store_experience=False,
            measure_inference=True,
        )
        row = episode_row(episode, category, mission_env, result, 0.0)
        row["scenario_hash"] = scenario_hash(architecture, mission)
        row["policy_parameter_count"] = parameter_count
        row["peak_gpu_memory_mb"] = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        )
        results.append(row)
    return results


class FixedArchitectureRulePolicy:
    """Evaluation baseline: KEEP when possible, otherwise first rescue rule."""

    def select_action(self, obs, action_mask, epsilon=0.0):
        valid = np.flatnonzero(np.asarray(action_mask) > 0)
        if not valid.size:
            raise ValueError("No valid architecture action.")
        return 0 if 0 in valid else int(valid[0])


class RandomArchitectureRulePolicy:
    def select_action(self, obs, action_mask, epsilon=0.0):
        valid = np.flatnonzero(np.asarray(action_mask) > 0)
        if not valid.size:
            raise ValueError("No valid architecture action.")
        return int(random.choice(valid.tolist()))


def evaluate_architecture_baseline(
    architecture_policy,
    scheduler_agent,
    scenarios,
    *,
    label,
    budget=8000.0,
    refund_rate=0.8,
):
    results = []
    for episode, (architecture, mission, category) in enumerate(scenarios):
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=budget,
            refund_rate=refund_rate,
        )
        result = run_episode(
            mission_env,
            architecture_policy,
            scheduler_agent,
            architecture_epsilon=0.0,
            scheduler_epsilon=0.0,
            update_architecture=False,
            update_scheduler=False,
            store_experience=False,
        )
        row = episode_row(episode, category, mission_env, result, 0.0)
        row["model"] = label
        row["scenario_hash"] = scenario_hash(architecture, mission)
        results.append(row)
    return results


def evaluate_static_scheduler(
    scheduler_agent,
    scenarios,
    *,
    label,
    full_systems=False,
    budget=8000.0,
):
    results = []
    for episode, (initial_architecture, mission, category) in enumerate(scenarios):
        architecture = env.FULL_SOS if full_systems else initial_architecture
        mission_env = env.MissionEnv(architecture, mission)
        scheduler_policy = rule.Rule(mission_env)
        total_reward = 0.0
        info = {"success": False, "dead_end": False}
        for _ in range(mission_env.T * mission_env.O):
            mask = dqn.rule_action_mask(mission_env, rule.Rule.RULE_NUM)
            if not np.any(mask):
                info = {"success": False, "dead_end": True}
                total_reward -= 2.0
                break
            action = scheduler_agent.select_action(
                mission_env.schedule_observation(),
                mask,
                epsilon=0.0,
            )
            env_action = scheduler_policy.to_env_action(action)
            _, base_reward, terminated, _, info = mission_env.step(env_action)
            total_reward += scheduler_reward(
                base_reward,
                bool(info.get("success", False)),
                bool(info.get("dead_end", False)),
            )
            if terminated:
                break
        results.append(
            {
                "episode": episode,
                "model": label,
                "category": category,
                "scenario_hash": scenario_hash(initial_architecture, mission),
                "success": bool(info.get("success", False)),
                "dead_end": bool(info.get("dead_end", False)),
                "scheduler_reward": float(total_reward),
                "makespan": float(mission_env.state.current_makespan),
                "net_cost": float(sum(system.cost for system in architecture)),
                "active_cost": float(sum(system.cost for system in architecture)),
                "total_refund": 0.0,
                "architecture_changes": 0,
                "budget_violation": bool(
                    sum(system.cost for system in architecture) > budget
                ),
                "assigned_ops": int(np.sum(mission_env.state.task_op_idx)),
                "initial_net_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "final_net_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "peak_net_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "initial_active_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "final_active_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "peak_active_cost": float(
                    sum(system.cost for system in architecture)
                ),
                "gross_charge": float(
                    sum(system.cost for system in architecture)
                ),
                "ever_over_budget": bool(
                    sum(system.cost for system in architecture) > budget
                ),
                "final_over_budget": bool(
                    sum(system.cost for system in architecture) > budget
                ),
            }
        )
    return results


def evaluate_flat_intdqn(flat_agent, scenarios, *, budget=8000.0):
    import intenv

    results = []
    for episode, (initial_architecture, mission, category) in enumerate(scenarios):
        mission_env = intenv.IntEnv(mission)
        obs, _ = mission_env.reset()
        info = {"success": False, "dead_end": False}
        total_reward = 0.0
        for _ in range(mission_env.T * mission_env.O):
            mask = mission_env.valid_action_mask()
            if not np.any(mask):
                info = {"success": False, "dead_end": True}
                break
            action = flat_agent.select_action(obs, mask, epsilon=0.0)
            obs, reward, terminated, _, info = mission_env.step(action)
            total_reward += reward
            if terminated:
                break
        results.append(
            {
                "episode": episode,
                "model": "flat_intdqn",
                "category": category,
                "scenario_hash": scenario_hash(initial_architecture, mission),
                "success": bool(info.get("success", False)),
                "dead_end": bool(info.get("dead_end", False)),
                "scheduler_reward": float(total_reward),
                "makespan": float(mission_env.state.cur_makespan),
                "net_cost": float(mission_env.state.cur_cost),
                "active_cost": float(mission_env.state.cur_cost),
                "total_refund": 0.0,
                "architecture_changes": int(
                    mission_env.state.select_sys_mask.sum()
                ),
                "budget_violation": bool(mission_env.state.cur_cost > budget),
                "assigned_ops": int(np.sum(mission_env.state.task_op_idx)),
                "initial_net_cost": 0.0,
                "final_net_cost": float(mission_env.state.cur_cost),
                "peak_net_cost": float(mission_env.state.cur_cost),
                "initial_active_cost": 0.0,
                "final_active_cost": float(mission_env.state.cur_cost),
                "peak_active_cost": float(mission_env.state.cur_cost),
                "gross_charge": float(mission_env.state.cur_cost),
                "ever_over_budget": bool(mission_env.state.cur_cost > budget),
                "final_over_budget": bool(mission_env.state.cur_cost > budget),
            }
        )
    return results


def scenario_hash(architecture, mission):
    payload = {
        "architecture": sorted(int(system.index) for system in architecture),
        "mission": [
            {
                "release_time": int(task.release_time),
                "due_time": int(task.due_time),
                "operations": [
                    (
                        int(operation.func_type),
                        int(operation.duration),
                        int(operation.release_time),
                    )
                    for operation in task.operations
                ],
            }
            for task in mission
        ],
    }
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
