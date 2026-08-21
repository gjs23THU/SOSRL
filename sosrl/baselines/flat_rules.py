"""Single-policy DQN over the Cartesian product of existing rule actions."""

from __future__ import annotations

import numpy as np

from .. import environment as env
from ..rl.agent import FlatRuleDQNAgent
from ..rl.config import HRLConfig
from ..rl.replay import NStepAccumulator
from ..rules import architecture as archrule
from ..rules import scheduling as rule
from ..workflows import hierarchical
from ..workflows import scheduler


ARCHITECTURE_ACTION_SIZE = archrule.ArchitectureRule.RULE_NUM
SCHEDULING_ACTION_SIZE = rule.Rule.RULE_NUM
JOINT_ACTION_SIZE = ARCHITECTURE_ACTION_SIZE * SCHEDULING_ACTION_SIZE


def encode_joint_action(architecture_action: int, scheduling_action: int) -> int:
    architecture_action = int(architecture_action)
    scheduling_action = int(scheduling_action)
    if not 0 <= architecture_action < ARCHITECTURE_ACTION_SIZE:
        raise ValueError(f"architecture action out of range: {architecture_action}")
    if not 0 <= scheduling_action < SCHEDULING_ACTION_SIZE:
        raise ValueError(f"scheduling action out of range: {scheduling_action}")
    return architecture_action * SCHEDULING_ACTION_SIZE + scheduling_action


def decode_joint_action(joint_action: int) -> tuple[int, int]:
    joint_action = int(joint_action)
    if not 0 <= joint_action < JOINT_ACTION_SIZE:
        raise ValueError(f"joint action out of range: {joint_action}")
    return divmod(joint_action, SCHEDULING_ACTION_SIZE)


def joint_action_mask(architecture_policy: archrule.ArchitectureRule) -> np.ndarray:
    architecture_mask = np.asarray(
        architecture_policy.action_mask(),
        dtype=np.float32,
    )
    if architecture_mask.shape != (ARCHITECTURE_ACTION_SIZE,):
        raise ValueError("architecture action mask has the wrong shape.")
    return np.repeat(architecture_mask, SCHEDULING_ACTION_SIZE)


def step_joint_action(
    mission_env: env.MissionEnv,
    architecture_policy: archrule.ArchitectureRule,
    scheduling_policy: rule.Rule,
    joint_action: int,
):
    architecture_action, scheduling_action = decode_joint_action(joint_action)
    action_mask = joint_action_mask(architecture_policy)
    if action_mask[int(joint_action)] <= 0:
        raise ValueError("joint action is masked in the current state.")

    old_makespan = float(mission_env.state.current_makespan)
    old_cost = float(mission_env.net_cost)
    architecture_info = architecture_policy.apply(architecture_action)
    if not architecture_info.get("valid", False):
        raise RuntimeError("a valid joint action resolved to an invalid architecture action.")

    scheduling_mask = scheduler.rule_action_mask(
        mission_env,
        SCHEDULING_ACTION_SIZE,
    )
    if scheduling_mask[scheduling_action] <= 0:
        raise RuntimeError(
            "a valid architecture action did not leave a schedulable operation."
        )
    environment_action = scheduling_policy.to_env_action(scheduling_action)
    _, _, terminated, truncated, info = mission_env.step(environment_action)
    success = bool(info.get("success", False))
    dead_end = bool(info.get("dead_end", False))
    reward = hierarchical.architecture_reward(
        mission_env,
        old_makespan,
        old_cost,
        bool(architecture_info["changed"]),
        success,
        dead_end,
    )
    next_observation = mission_env.architecture_observation()
    combined_info = dict(info)
    combined_info.update(
        {
            "architecture_action": architecture_action,
            "architecture_rule": architecture_info["rule_name"],
            "architecture_changed": bool(architecture_info["changed"]),
            "scheduling_action": scheduling_action,
            "scheduling_rule": rule.Rule.RULE_NAMES[scheduling_action],
        }
    )
    return next_observation, reward, terminated, truncated, combined_info


def _store_n_step(agent, accumulator, transition) -> None:
    for emitted in accumulator.append(transition):
        agent.replay.add(*emitted)


def run_flat_rule_episode(
    mission_env: env.MissionEnv,
    agent: FlatRuleDQNAgent,
    *,
    epsilon: float,
    update_agent: bool,
    store_experience: bool = True,
):
    architecture_policy = archrule.ArchitectureRule(mission_env)
    scheduling_policy = rule.Rule(mission_env)
    accumulator = (
        NStepAccumulator(agent.config.n_step, agent.config.gamma)
        if store_experience
        else None
    )
    architecture_counts = np.zeros(ARCHITECTURE_ACTION_SIZE, dtype=np.int32)
    scheduling_counts = np.zeros(SCHEDULING_ACTION_SIZE, dtype=np.int32)
    joint_total = 0.0
    last_loss = None
    environment_steps = 0
    info = {"success": False, "dead_end": False}

    for _ in range(mission_env.T * mission_env.O + mission_env.N):
        observation = mission_env.architecture_observation()
        action_mask = joint_action_mask(architecture_policy)
        if not np.any(action_mask):
            info = {"success": False, "dead_end": True}
            break

        joint_action = agent.select_action(observation, action_mask, epsilon)
        architecture_action, scheduling_action = decode_joint_action(joint_action)
        architecture_counts[architecture_action] += 1
        scheduling_counts[scheduling_action] += 1

        next_observation, reward, terminated, _, info = step_joint_action(
            mission_env,
            architecture_policy,
            scheduling_policy,
            joint_action,
        )
        environment_steps += 1
        joint_total += float(reward)
        next_mask = (
            np.zeros(JOINT_ACTION_SIZE, dtype=np.float32)
            if terminated
            else joint_action_mask(architecture_policy)
        )
        if store_experience:
            _store_n_step(
                agent,
                accumulator,
                (
                    observation,
                    joint_action,
                    reward,
                    next_observation,
                    terminated,
                    next_mask,
                ),
            )
        if update_agent:
            loss = agent.learn()
            if loss is not None:
                last_loss = loss
        if terminated:
            break

    return {
        "joint_reward": float(joint_total),
        "joint_loss": last_loss,
        "architecture_rule_counts": architecture_counts,
        "scheduler_rule_counts": scheduling_counts,
        "success": bool(info.get("success", False)),
        "dead_end": bool(info.get("dead_end", False)),
        "environment_steps": int(environment_steps),
    }


def flat_rule_episode_row(
    episode: int,
    category: str,
    mission_env: env.MissionEnv,
    result: dict,
    epsilon: float,
    replay_size: int,
    cumulative_environment_steps: int,
) -> dict:
    row = {
        "episode": int(episode),
        "category": category,
        "joint_reward": result["joint_reward"],
        "joint_loss": result["joint_loss"],
        "success": result["success"],
        "dead_end": result["dead_end"],
        "makespan": float(mission_env.state.current_makespan),
        "net_cost": float(mission_env.net_cost),
        "active_cost": float(mission_env.active_cost),
        "total_refund": float(mission_env.total_refund),
        "architecture_changes": int(mission_env.architecture_change_count),
        "budget_violation": bool(mission_env.net_cost > mission_env.budget),
        "assigned_ops": int(np.sum(mission_env.state.task_op_idx)),
        "episode_environment_steps": int(result["environment_steps"]),
        "cumulative_environment_steps": int(cumulative_environment_steps),
        "epsilon": float(epsilon),
        "replay_size": int(replay_size),
    }
    row.update(mission_env.cost_metrics())
    for index, name in enumerate(archrule.ArchitectureRule.RULE_NAMES):
        row[f"arch_{name.lower()}_count"] = int(
            result["architecture_rule_counts"][index]
        )
    for index, name in enumerate(rule.Rule.RULE_NAMES):
        row[f"schedule_{name.lower()}_count"] = int(
            result["scheduler_rule_counts"][index]
        )
    return row


def train_flat_rules(
    config: HRLConfig,
    scenario_pool: hierarchical.AdaptiveScenarioPool,
    *,
    max_env_steps: int | None = None,
):
    if max_env_steps is not None and int(max_env_steps) <= 0:
        raise ValueError("max_env_steps must be positive when provided.")
    if max_env_steps is None and int(config.episodes) <= 0:
        raise ValueError("episodes must be positive when max_env_steps is omitted.")

    scheduler.set_seed(config.seed)
    architecture, mission, _ = scenario_pool.get(0)
    probe = env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=config.budget,
        refund_rate=config.refund_rate,
    )
    agent = FlatRuleDQNAgent(
        probe.architecture_observation_space.shape[0],
        config,
    )
    epsilon = float(config.epsilon_start)
    history = []
    cumulative_environment_steps = 0
    episode = 0

    while True:
        if max_env_steps is None:
            if episode >= int(config.episodes):
                break
        elif cumulative_environment_steps >= int(max_env_steps):
            break

        architecture, mission, category = scenario_pool.sample()
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=config.budget,
            refund_rate=config.refund_rate,
        )
        result = run_flat_rule_episode(
            mission_env,
            agent,
            epsilon=epsilon,
            update_agent=True,
        )
        if result["environment_steps"] <= 0 and max_env_steps is not None:
            raise RuntimeError(
                "cannot satisfy max_env_steps because an episode made no progress."
            )
        cumulative_environment_steps += int(result["environment_steps"])
        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        history.append(
            flat_rule_episode_row(
                episode,
                category,
                mission_env,
                result,
                epsilon,
                len(agent.replay),
                cumulative_environment_steps,
            )
        )
        episode += 1

    return agent, history


def evaluate_flat_rules(
    agent: FlatRuleDQNAgent,
    scenarios,
    *,
    label: str = "flat_rule_dqn",
    budget: float = 8000.0,
    refund_rate: float = 0.8,
):
    if agent.action_dim != JOINT_ACTION_SIZE:
        raise ValueError("flat-rule agent must contain 24 joint actions.")
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
        if mission_env.architecture_observation_space.shape[0] != agent.obs_dim:
            raise ValueError(
                "flat-rule checkpoint observation dimension does not match the environment."
            )
        result = run_flat_rule_episode(
            mission_env,
            agent,
            epsilon=0.0,
            update_agent=False,
            store_experience=False,
        )
        row = flat_rule_episode_row(
            episode,
            category,
            mission_env,
            result,
            epsilon=0.0,
            replay_size=len(agent.replay),
            cumulative_environment_steps=result["environment_steps"],
        )
        row["model"] = label
        row["scenario_hash"] = hierarchical.scenario_hash(architecture, mission)
        results.append(row)
    return results
