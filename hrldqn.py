from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import archrule
import dqn
import env
import rule
import syn


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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ArchitectureReplayBuffer:
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
    ):
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

    def __len__(self):
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

    def clear(self):
        self.pending.clear()


class ArchitectureDQNAgent:
    """The single-headed upper policy. No trigger or termination network."""

    def __init__(self, obs_dim: int, config: HRLConfig):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = archrule.ArchitectureRule.RULE_NUM
        self.device = torch.device(config.device)
        self.q_net = dqn.QNetwork(
            self.obs_dim,
            self.action_dim,
            config.hidden_dim,
        ).to(self.device)
        self.target_net = dqn.QNetwork(
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
        obs, action, reward, next_obs, done, next_mask, discount = self.replay.sample(
            self.config.batch_size
        )
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
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

    def save_checkpoint(self, path, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
                "q_net_state_dict": self.q_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "learn_step": self.learn_step,
                "training_state": training_state or {},
            },
            path,
        )
        return path

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        values = dict(checkpoint["config"])
        values["device"] = str(device)
        agent = cls(int(checkpoint["obs_dim"]), HRLConfig(**values))
        if int(checkpoint["action_dim"]) != agent.action_dim:
            raise ValueError("architecture checkpoint action dimension must be six.")
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.learn_step = int(checkpoint["learn_step"])
        return agent, checkpoint


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

        arch_action = architecture_agent.select_action(
            arch_obs,
            arch_mask,
            architecture_epsilon,
        )
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


        schedule_action = scheduler_agent.select_action(
            schedule_obs,
            schedule_mask,
            scheduler_epsilon,
        )
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
        result = run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=0.0,
            scheduler_epsilon=0.0,
            update_architecture=False,
            update_scheduler=False,
            store_experience=False,
        )
        row = episode_row(episode, category, mission_env, result, 0.0)
        row["scenario_hash"] = scenario_hash(architecture, mission)
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


def save_combined_checkpoint(
    path,
    architecture_agent,
    scheduler_agent,
    training_state=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": {
            "obs_dim": architecture_agent.obs_dim,
            "action_dim": architecture_agent.action_dim,
            "config": asdict(architecture_agent.config),
            "q_net_state_dict": architecture_agent.q_net.state_dict(),
            "target_net_state_dict": architecture_agent.target_net.state_dict(),
            "optimizer_state_dict": architecture_agent.optimizer.state_dict(),
            "learn_step": architecture_agent.learn_step,
        },
        "scheduler": {
            "obs_dim": scheduler_agent.obs_dim,
            "action_dim": scheduler_agent.action_dim,
            "config": asdict(scheduler_agent.config),
            "q_net_state_dict": scheduler_agent.q_net.state_dict(),
            "target_net_state_dict": scheduler_agent.target_net.state_dict(),
            "optimizer_state_dict": scheduler_agent.optimizer.state_dict(),
            "learn_step": scheduler_agent.learn_step,
        },
        "training_state": training_state or {},
    }
    torch.save(checkpoint, path)
    return path


def load_combined_checkpoint(path, device=None, load_optimizer=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    arch_data = checkpoint["architecture"]
    arch_values = dict(arch_data["config"])
    arch_values["device"] = str(device)
    architecture_agent = ArchitectureDQNAgent(
        int(arch_data["obs_dim"]),
        HRLConfig(**arch_values),
    )
    if int(arch_data["action_dim"]) != architecture_agent.action_dim:
        raise ValueError("combined checkpoint architecture action dimension must be six.")
    architecture_agent.q_net.load_state_dict(arch_data["q_net_state_dict"])
    architecture_agent.target_net.load_state_dict(arch_data["target_net_state_dict"])
    architecture_agent.learn_step = int(arch_data["learn_step"])
    if load_optimizer:
        architecture_agent.optimizer.load_state_dict(arch_data["optimizer_state_dict"])

    scheduler_data = checkpoint["scheduler"]
    scheduler_values = dict(scheduler_data["config"])
    scheduler_values["device"] = str(device)
    scheduler_agent = dqn.DQNAgent(
        int(scheduler_data["obs_dim"]),
        int(scheduler_data["action_dim"]),
        dqn.DQNConfig(**scheduler_values),
    )
    scheduler_agent.q_net.load_state_dict(scheduler_data["q_net_state_dict"])
    scheduler_agent.target_net.load_state_dict(scheduler_data["target_net_state_dict"])
    scheduler_agent.learn_step = int(scheduler_data["learn_step"])
    if load_optimizer:
        scheduler_agent.optimizer.load_state_dict(scheduler_data["optimizer_state_dict"])
    return architecture_agent, scheduler_agent, checkpoint
