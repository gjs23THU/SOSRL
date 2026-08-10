from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

import env
import hrule
import rule
import syn


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
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size: int = 500
    target_update_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 1024
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_rule_class(rule_set):
    if rule_set == "standard":
        return rule.Rule
    if rule_set == "huang":
        return hrule.HRule
    raise ValueError(f"Unknown rule set: {rule_set}")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, next_mask = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(next_mask, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
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
    def __init__(self, obs_dim, action_dim, config):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(config.device)
        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def save_checkpoint(self, path, training_state=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "config": asdict(self.config),
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "learn_step": self.learn_step,
            "training_state": training_state or {},
        }
        torch.save(checkpoint, path)
        return path

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        config_values = dict(checkpoint["config"])
        config_values["device"] = str(device)
        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            config=DQNConfig(**config_values),
        )
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.learn_step = int(checkpoint["learn_step"])
        return agent, checkpoint

    def select_action(self, obs, action_mask, epsilon):
        valid_actions = np.flatnonzero(action_mask > 0)
        if len(valid_actions) == 0:
            raise ValueError("No valid rule action.")

        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            q_values = q_values.masked_fill(mask_tensor <= 0, -1e9)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        if len(self.replay) < max(self.config.min_buffer_size, self.config.batch_size):
            return None

        obs, action, reward, next_obs, done, next_mask = self.replay.sample(self.config.batch_size)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.tensor(next_mask, dtype=torch.float32, device=self.device)

        q_value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_obs)
            next_q = next_q.masked_fill(next_mask <= 0, -1e9)
            next_q = next_q.max(dim=1, keepdim=True).values
            has_next_action = (next_mask.sum(dim=1, keepdim=True) > 0).float()
            target = reward + self.config.gamma * (1.0 - done) * has_next_action * next_q

        loss = nn.functional.mse_loss(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


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
