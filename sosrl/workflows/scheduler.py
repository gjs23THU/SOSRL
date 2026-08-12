import random

import numpy as np
import torch
from tqdm.auto import tqdm

from .. import domain as syn
from .. import environment as env
from ..rl.agent import DQNAgent, QNetwork
from ..rl.config import DQNConfig
from ..rl.replay import ReplayBuffer
from ..rules import huang as hrule
from ..rules import scheduling as rule


def get_rule_class(rule_set):
    if rule_set == "standard":
        return rule.Rule
    if rule_set == "huang":
        return hrule.HRule
    raise ValueError(f"Unknown rule set: {rule_set}")


class ScenarioPool:
    def __init__(
        self,
        size,
        selected_system_num=None,
        min_system_num=3,
        max_system_num=None,
        cost_limit=None,
        max_attempts=1000,
        shared_mission=False,
        mission=None,
    ):
        self.selected_system_num = selected_system_num
        self.min_system_num = max(min_system_num, len(syn.CONFIG.get("funcs", {})))
        self.max_system_num = len(env.FULL_SOS) if max_system_num is None else max_system_num
        self.cost_limit = syn.CONFIG.get("cost_limit") if cost_limit is None else cost_limit
        self.min_coverage_until = float(syn.CONFIG.get("min_coverage_until", 600))
        self.max_attempts = max_attempts
        self.mission = mission
        if self.mission is None and shared_mission:
            self.mission = syn.build_mission_from_config(syn.CONFIG)
        self.scenarios = []

        for _ in range(size):
            scenario_mission = self.mission
            if scenario_mission is None:
                scenario_mission = syn.build_mission_from_config(syn.CONFIG)
            arch = self.sample_arch(scenario_mission)
            self.scenarios.append((arch, scenario_mission))

    def sample_system_num(self):
        if self.selected_system_num is None:
            return random.randint(self.min_system_num, self.max_system_num)
        if isinstance(self.selected_system_num, tuple):
            low, high = self.selected_system_num
            return random.randint(max(low, self.min_system_num), min(high, self.max_system_num))
        return int(self.selected_system_num)

    def sample_arch(self, mission):
        for _ in range(self.max_attempts):
            arch = syn.random_select_sos(self.sample_system_num())
            if self.cost_limit is not None and sum(s.cost for s in arch) > self.cost_limit:
                continue
            if self.arch_can_cover_mission(arch, mission):
                return arch
        raise ValueError("Cannot sample feasible arch.")

    def arch_can_cover_mission(self, arch, mission):
        for func_name in syn.CONFIG.get("funcs", {}):
            func_type = syn.func_type2idx[func_name]
            matching_systems = [s for s in arch if s.func_type == func_type]
            if not matching_systems:
                return False
            if max(s.available_until for s in matching_systems) < self.min_coverage_until:
                return False
            capacity = sum(s.available_until - s.available_from for s in matching_systems)
            demand = sum(op.duration for task in mission for op in task.operations if op.func_type == func_type)
            if capacity < demand:
                return False
        return True

    def get(self, index):
        index = index % len(self.scenarios)
        arch, mission = self.scenarios[index]
        return index, arch, mission

    def sample(self):
        index = random.randrange(len(self.scenarios))
        arch, mission = self.scenarios[index]
        return index, arch, mission


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rule_action_mask(mission_env, rule_num):
    if np.any(mission_env.valid_assignment_mask()):
        return np.ones(rule_num, dtype=np.float32)
    return np.zeros(rule_num, dtype=np.float32)


def step_rule_action(mission_env, rule_policy, rule_action):
    env_action = rule_policy.to_env_action(rule_action)
    return mission_env.step(env_action)


def train_dqn(config, scenario_pool):
    set_seed(config.seed)
    rule_class = get_rule_class(config.rule_set)

    _, arch, mission = scenario_pool.get(0)
    mission_env = env.MissionEnv(arch, mission)
    obs_dim = mission_env.observation_space.shape[0]
    action_dim = rule_class.RULE_NUM
    agent = DQNAgent(obs_dim, action_dim, config)

    epsilon = config.epsilon_start
    history = []

    progress = tqdm(range(config.episodes), desc="Training", unit="episode")
    for episode in progress:
        if config.scenario_order == "random":
            scenario_idx, arch, mission = scenario_pool.sample()
        elif config.scenario_order == "sequential":
            scenario_idx, arch, mission = scenario_pool.get(episode)
        else:
            raise ValueError(f"Unknown scenario_order: {config.scenario_order}")

        mission_env = env.MissionEnv(arch, mission)
        rule_policy = rule_class(mission_env)
        obs, _ = mission_env.reset()
        total_reward = 0.0
        rule_counts = np.zeros(rule_class.RULE_NUM, dtype=np.int32)
        last_loss = None
        terminated = False
        info = {"success": False, "dead_end": False}

        for _ in range(mission_env.T * mission_env.O):
            mask = rule_action_mask(mission_env, rule_class.RULE_NUM)
            rule_action = agent.select_action(obs, mask, epsilon)
            rule_counts[rule_action] += 1
            next_obs, reward, terminated, _, info = step_rule_action(
                mission_env,
                rule_policy,
                rule_action,
            )

            next_mask = rule_action_mask(mission_env, rule_class.RULE_NUM)
            agent.replay.add(obs, rule_action, reward, next_obs, terminated, next_mask)

            loss = agent.learn()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            total_reward += reward
            if terminated:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row = {
            "episode": episode,
            "scenario_idx": scenario_idx,
            "scenario_sos": len(arch),
            "scenario_cost": sum(s.cost for s in arch),
            "reward": float(total_reward),
            "dead_end": bool(info.get("dead_end", False)),
            "success": bool(info.get("success", False)),
            "makespan": float(mission_env.state.current_makespan),
            "assigned_ops": int(mission_env.state.task_op_idx.sum()),
            "epsilon": float(epsilon),
            "loss": last_loss,
            "replay_size": len(agent.replay),
        }
        for rule_idx, rule_name in enumerate(rule_class.RULE_NAMES):
            row[f"rule_{rule_name.lower()}_count"] = int(rule_counts[rule_idx])
        history.append(row)

        progress.set_postfix(
            reward=f"{total_reward:.3f}",
            makespan=f"{mission_env.state.current_makespan:.1f}",
            assigned_ops=int(mission_env.state.task_op_idx.sum()),
            epsilon=f"{epsilon:.3f}",
            loss="-" if last_loss is None else f"{last_loss:.4f}",
        )

    return agent, history


def evaluate_dqn(agent, scenario_pool, episodes, collect_schedule=False):
    rule_class = get_rule_class(agent.config.rule_set)
    if agent.action_dim != rule_class.RULE_NUM:
        raise ValueError(
            f"Checkpoint action dimension {agent.action_dim} does not match "
            f"rule set '{agent.config.rule_set}' with {rule_class.RULE_NUM} actions."
        )

    results = []
    for episode in range(episodes):
        scenario_idx, arch, mission = scenario_pool.get(episode)
        mission_env = env.MissionEnv(arch, mission)
        rule_policy = rule_class(mission_env)
        obs, _ = mission_env.reset()
        total_reward = 0.0
        rule_counts = np.zeros(rule_class.RULE_NUM, dtype=np.int32)
        terminated = False
        info = {"success": False, "dead_end": False}

        for _ in range(mission_env.T * mission_env.O):
            mask = rule_action_mask(mission_env, rule_class.RULE_NUM)
            rule_action = agent.select_action(obs, mask, epsilon=0.0)
            rule_counts[rule_action] += 1
            obs, reward, terminated, _, info = step_rule_action(
                mission_env,
                rule_policy,
                rule_action,
            )

            total_reward += reward
            if terminated:
                break

        result = {
            "episode": episode,
            "scenario_idx": scenario_idx,
            "reward": float(total_reward),
            "makespan": float(mission_env.state.current_makespan),
            "assigned_ops": int(mission_env.state.task_op_idx.sum()),
            "dead_end": bool(info.get("dead_end", False)),
            "success": bool(info.get("success", False)),
        }
        for rule_idx, rule_name in enumerate(rule_class.RULE_NAMES):
            result[f"rule_{rule_name.lower()}_count"] = int(rule_counts[rule_idx])
        if collect_schedule:
            result["schedule"] = schedule_rows(mission_env, episode, scenario_idx)
        results.append(result)

    return results


def schedule_rows(mission_env, episode, scenario_idx):
    rows = []
    for task_idx in range(mission_env.T):
        task = mission_env.mission[task_idx]
        for op_idx in range(mission_env.O):
            sys_idx = int(mission_env.state.op_assign_sys[task_idx, op_idx])
            if sys_idx < 0:
                continue
            rows.append(
                {
                    "episode": int(episode),
                    "scenario_idx": int(scenario_idx),
                    "task_idx": int(task_idx),
                    "task_name": task.name,
                    "op_idx": int(op_idx),
                    "op_name": task.operations[op_idx].name,
                    "func_type": int(task.operations[op_idx].func_type),
                    "sys_idx": sys_idx,
                    "sys_name": env.FULL_SOS[sys_idx].name,
                    "start_time": float(mission_env.state.op_start_time[task_idx, op_idx]),
                    "finish_time": float(mission_env.state.op_finish_time[task_idx, op_idx]),
                    "duration": float(mission_env.state.op_finish_time[task_idx, op_idx] - mission_env.state.op_start_time[task_idx, op_idx]),
                }
            )
    return sorted(rows, key=lambda row: (row["start_time"], row["task_idx"], row["op_idx"]))
