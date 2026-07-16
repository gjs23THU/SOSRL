from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import env
import syn


@dataclass
class DQNConfig:
    episodes: int = 200
    max_steps: int = 300
    scenario_pool_size: int = 20
    scenario_order: str = "random"
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
        self.device = torch.device(config.device)
        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def select_action(self, obs, action_mask, epsilon):
        valid_actions = np.flatnonzero(action_mask > 0)
        if len(valid_actions) == 0:
            raise ValueError("No valid op action.")

        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            q_values = q_values.masked_fill(mask_tensor <= 0, -1e9)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        if len(self.replay) < self.config.min_buffer_size:
            return None
        if len(self.replay) < self.config.batch_size:
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
    ):
        self.selected_system_num = selected_system_num
        self.min_system_num = max(min_system_num, len(syn.CONFIG.get("funcs", {})))
        self.max_system_num = len(env.FULL_SOS) if max_system_num is None else max_system_num
        self.cost_limit = syn.CONFIG.get("cost_limit") if cost_limit is None else cost_limit
        self.max_attempts = max_attempts
        self.scenarios = []

        for _ in range(size):
            mission = syn.build_mission_from_config(syn.CONFIG)
            arch = self.sample_arch(mission)
            self.scenarios.append((arch, mission))

    def sample_system_num(self):
        if self.selected_system_num is None:
            return random.randint(self.min_system_num, self.max_system_num)
        if isinstance(self.selected_system_num, tuple):
            low, high = self.selected_system_num
            return random.randint(max(low, self.min_system_num), min(high, self.max_system_num))
        return int(self.selected_system_num)

    def sample_arch(self, mission):
        for _ in range(self.max_attempts):
            arch = syn.random_select_sos(self.sample_system_num(), syn.CONFIG)
            if self.cost_limit is not None and sum(s.cost for s in arch) > self.cost_limit:
                continue
            if self.arch_can_cover_mission(arch, mission):
                return arch
        raise ValueError("Cannot sample feasible arch.")

    def arch_can_cover_mission(self, arch, mission):
        for func_name in syn.CONFIG.get("funcs", {}):
            func_type = syn.func_type2idx.get(func_name, 0)
            capacity = sum(s.available_until - s.available_from for s in arch if s.func_type == func_type)
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


def op_action_mask(mission_env):
    assign_mask = mission_env.mask_invalid_assign()
    return assign_mask.max(axis=2).reshape(-1).astype(np.float32)


def cssa(mission_env, task_idx, op_idx, sys_indices):
    op = mission_env.state.mission[task_idx].operations[op_idx]
    candidates = []
    for sys_idx in sys_indices:
        start_time = max(
            float(mission_env.state.current_time),
            float(mission_env.state.sys_availble_time[sys_idx]),
            float(mission_env.state.op_release_time[task_idx, op_idx]),
        )
        finish_time = start_time + float(op.duration)
        if finish_time > float(env.FULL_SOS[sys_idx].available_until):
            continue
        candidates.append(
            (
                start_time,
                finish_time,
                float(mission_env.state.sys_busy_time[sys_idx]),
                int(sys_idx),
            )
        )

    if not candidates:
        raise ValueError(f"No feasible system for task={task_idx}, op={op_idx}.")

    _, _, _, sys_idx = min(candidates)
    return sys_idx


def op_action_to_env_action(mission_env, op_action):
    task_idx, op_idx = np.unravel_index(int(op_action), (mission_env.T, mission_env.O))
    assign_mask = mission_env.mask_invalid_assign()
    sys_indices = np.flatnonzero(assign_mask[task_idx, op_idx] > 0)
    if len(sys_indices) == 0:
        raise ValueError(f"No feasible system for task={task_idx}, op={op_idx}.")

    sys_idx = cssa(mission_env, int(task_idx), int(op_idx), sys_indices)
    env_action = mission_env.encode_action(
        "assign_task",
        task_idx=int(task_idx),
        op_idx=int(op_idx),
        sys_idx=int(sys_idx),
    )
    return int(env_action), int(task_idx), int(op_idx), int(sys_idx)


def step_op_action(mission_env, op_action):
    env_action, task_idx, op_idx, sys_idx = op_action_to_env_action(mission_env, op_action)
    next_obs, reward, terminated, truncated, info = mission_env.step(env_action)
    info["op_action"] = int(op_action)
    info["env_action"] = int(env_action)
    info["task_idx"] = task_idx
    info["op_idx"] = op_idx
    info["sys_idx"] = sys_idx
    return next_obs, float(reward), bool(terminated), bool(truncated), info


def train_dqn(config, scenario_pool):
    set_seed(config.seed)

    _, arch, mission = scenario_pool.get(0)
    mission_env = env.MissionEnv(arch, mission)
    obs_dim = mission_env.observation_space.shape[0]
    action_dim = mission_env.T * mission_env.O
    agent = DQNAgent(obs_dim, action_dim, config)

    epsilon = config.epsilon_start
    history = []

    for episode in range(config.episodes):
        if config.scenario_order == "random":
            scenario_idx, arch, mission = scenario_pool.sample()
        elif config.scenario_order == "sequential":
            scenario_idx, arch, mission = scenario_pool.get(episode)
        else:
            raise ValueError(f"Unknown scenario_order: {config.scenario_order}")

        mission_env = env.MissionEnv(arch, mission)
        obs, _ = mission_env.reset()
        total_reward = 0.0
        last_loss = None
        terminated = False
        truncated = False
        info = {"dead_end": False, "valid": True}

        for step in range(config.max_steps):
            mask = op_action_mask(mission_env)
            if not np.any(mask):
                next_obs = mission_env.state.to_obs()
                reward = -1.0
                terminated = True
                truncated = False
                info = {"dead_end": True, "valid": False, "info": "No valid op action."}
                op_action = 0
            else:
                op_action = agent.select_action(obs, mask, epsilon)
                next_obs, reward, terminated, truncated, info = step_op_action(mission_env, op_action)

            done = terminated or truncated
            next_mask = op_action_mask(mission_env)
            agent.replay.add(obs, op_action, reward, next_obs, done, next_mask)

            loss = agent.learn()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            total_reward += reward
            if done:
                break

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row = {
            "episode": episode,
            "scenario_idx": scenario_idx,
            "scenario_sos": len(arch),
            "scenario_cost": sum(s.cost for s in arch),
            "reward": float(total_reward),
            "steps": step + 1,
            "done": bool(terminated or truncated),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "dead_end": bool(info.get("dead_end", False)),
            "makespan": float(mission_env.state.current_makespan),
            "done_ops": int(mission_env.state.task_op_idx.sum()),
            "epsilon": float(epsilon),
            "loss": last_loss,
            "replay_size": len(agent.replay),
        }
        history.append(row)

        if episode == 0 or (episode + 1) % 10 == 0:
            print(row)

    return agent, history


def evaluate_dqn(agent, scenario_pool, episodes, max_steps, collect_schedule=False):
    results = []
    for episode in range(episodes):
        scenario_idx, arch, mission = scenario_pool.get(episode)
        mission_env = env.MissionEnv(arch, mission)
        obs, _ = mission_env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {"dead_end": False, "valid": True}

        for step in range(max_steps):
            mask = op_action_mask(mission_env)
            if not np.any(mask):
                reward = -1.0
                terminated = True
                truncated = False
                info = {"dead_end": True, "valid": False, "info": "No valid op action."}
            else:
                op_action = agent.select_action(obs, mask, epsilon=0.0)
                obs, reward, terminated, truncated, info = step_op_action(mission_env, op_action)

            total_reward += reward
            if terminated or truncated:
                break

        result = {
            "episode": episode,
            "scenario_idx": scenario_idx,
            "reward": float(total_reward),
            "steps": step + 1,
            "makespan": float(mission_env.state.current_makespan),
            "done_ops": int(mission_env.state.task_op_idx.sum()),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "dead_end": bool(info.get("dead_end", False)),
        }
        if collect_schedule:
            result["schedule"] = schedule_rows(mission_env, episode, scenario_idx)
        results.append(result)

    return results


def schedule_rows(mission_env, episode, scenario_idx):
    rows = []
    for task_idx in range(mission_env.T):
        task = mission_env.state.mission[task_idx]
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
                    "func_type": float(task.operations[op_idx].func_type),
                    "sys_idx": sys_idx,
                    "sys_name": env.FULL_SOS[sys_idx].name,
                    "start_time": float(mission_env.state.op_start_time[task_idx, op_idx]),
                    "finish_time": float(mission_env.state.op_finish_time[task_idx, op_idx]),
                    "duration": float(mission_env.state.op_finish_time[task_idx, op_idx] - mission_env.state.op_start_time[task_idx, op_idx]),
                }
            )
    return sorted(rows, key=lambda row: (row["start_time"], row["task_idx"], row["op_idx"]))
