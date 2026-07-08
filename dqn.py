from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import env
import rule
import syn


@dataclass
class DQNConfig:
    action_mode: str = "rule"
    episodes: int = 200
    max_steps: int = 300
    scenario_pool_size: int = 20
    scenario_order: str = "random"
    selected_system_num: int | tuple[int, int] | None = None
    min_system_num: int = 3
    max_system_num: int = 22
    cost_limit: float | None = 8000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 10000
    min_buffer_size: int = 500
    target_update_interval: int = 20
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    n_step: int = 1
    hidden_dim: int = 128
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, next_mask, n_step=1):
        self.data.append((obs, action, reward, next_obs, done, next_mask, n_step))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        obs, action, reward, next_obs, done, next_mask, n_step = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done, dtype=np.float32),
            np.array(next_mask, dtype=np.float32),
            np.array(n_step, dtype=np.float32),
        )

    def __len__(self):
        return len(self.data)


class ScenarioPool:
    def __init__(
        self,
        size=1,
        selected_system_num=None,
        min_system_num=3,
        max_system_num=None,
        cost_limit=None,
        config=syn.CONFIG,
        scenarios=None,
        use_full_sos=False,
        max_attempts=1000,
    ):
        self.scenarios = []
        self.config = config
        self.selected_system_num = selected_system_num
        self.min_system_num = max(min_system_num, len(config.get("funcs", {})))
        self.max_system_num = len(env.FULL_SOS) if max_system_num is None else max_system_num
        self.cost_limit = cost_limit
        self.max_attempts = max_attempts

        if self.cost_limit is None:
            self.cost_limit = config.get("cost_limit")

        if scenarios is not None:
            self.scenarios.extend(scenarios)
            return

        for _ in range(size):
            mission = syn.build_mission_from_config(config)
            arch = env.FULL_SOS if use_full_sos else self.random_arch(mission)
            self.scenarios.append((arch, mission))

    def sample_system_num(self):
        if self.selected_system_num is None:
            return random.randint(self.min_system_num, self.max_system_num)
        if isinstance(self.selected_system_num, tuple):
            low, high = self.selected_system_num
            low = max(int(low), self.min_system_num)
            high = min(int(high), self.max_system_num)
            return random.randint(low, high)
        if isinstance(self.selected_system_num, list):
            return int(random.choice(self.selected_system_num))
        return int(self.selected_system_num)

    def random_arch(self, mission):
        for _ in range(self.max_attempts):
            arch = syn.random_select_sos(self.sample_system_num(), self.config)
            cost = sum(system.cost for system in arch)
            if self.cost_limit is not None and cost > self.cost_limit:
                continue

            feasible = True
            for func_type in self.config.get("funcs", {}):
                func_idx = syn.func_type2idx.get(func_type, 0)
                func_limit = sum((system.available_until - system.available_from) for system in arch if system.func_type == func_idx)
                func_req = sum(op.duration for task in mission for op in task.operations if op.func_type == func_idx)
                if not any(system.func_type == func_idx for system in arch) or func_limit < func_req:
                    feasible = False
                    break

            if feasible:
                return arch
        raise ValueError(
            "Cannot sample a feasible architecture within cost limit. "
            f"selected_system_num={self.selected_system_num}, "
            f"min_system_num={self.min_system_num}, "
            f"max_system_num={self.max_system_num}, "
            f"cost_limit={self.cost_limit}"
        )

    def sample(self):
        scenario_idx = random.randrange(len(self.scenarios))
        arch, mission = self.scenarios[scenario_idx]
        return scenario_idx, arch, mission

    def get(self, scenario_idx):
        real_idx = scenario_idx % len(self.scenarios)
        arch, mission = self.scenarios[real_idx]
        return real_idx, arch, mission

    def __len__(self):
        return len(self.scenarios)


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
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device)
        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.train_steps = 0

    def select_action(self, obs, epsilon, mask):
        valid_actions = np.flatnonzero(mask > 0)
        if len(valid_actions) == 0:
            raise ValueError("No valid DQN action.")

        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor).masked_fill(mask_tensor <= 0, -1e9)
        return int(torch.argmax(q_values, dim=1).item())

    def learn(self):
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        obs, action, reward, next_obs, done, next_mask, n_step = self.replay_buffer.sample(self.config.batch_size)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask = torch.tensor(next_mask, dtype=torch.float32, device=self.device)
        n_step = torch.tensor(n_step, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            discount = torch.pow(torch.tensor(self.config.gamma, dtype=torch.float32, device=self.device), n_step)
            next_target_q = self.target_net(next_obs).masked_fill(next_mask <= 0, -1e9).max(dim=1, keepdim=True).values
            has_next_action = (next_mask.sum(dim=1, keepdim=True) > 0).float()
            target = reward + discount * (1.0 - done) * has_next_action * next_target_q

        loss = nn.functional.smooth_l1_loss(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_n_step_transition(replay_buffer, n_step_buffer, gamma):
    reward = 0.0
    next_obs = n_step_buffer[-1][3]
    done = n_step_buffer[-1][4]
    next_mask = n_step_buffer[-1][5]
    actual_n = len(n_step_buffer)

    for i, transition in enumerate(n_step_buffer):
        reward += (gamma ** i) * transition[2]
        if transition[4]:
            next_obs = transition[3]
            done = transition[4]
            next_mask = transition[5]
            actual_n = i + 1
            break

    replay_buffer.add(
        n_step_buffer[0][0],
        n_step_buffer[0][1],
        reward,
        next_obs,
        done,
        next_mask,
        actual_n,
    )


def make_mission_env(arch=None, mission=None):
    if arch is None:
        arch = env.FULL_SOS
    if mission is None:
        mission = syn.build_mission_from_config(syn.CONFIG)
    return env.MissionEnv(arch=arch, mission=mission)


def make_scenario_pool(config, arch=None, mission=None):
    if arch is not None or mission is not None:
        if arch is None:
            arch = env.FULL_SOS
        if mission is None:
            mission = syn.build_mission_from_config(syn.CONFIG)
        return ScenarioPool(scenarios=[(arch, mission)])

    return ScenarioPool(
        size=config.scenario_pool_size,
        selected_system_num=config.selected_system_num,
        min_system_num=config.min_system_num,
        max_system_num=config.max_system_num,
        cost_limit=config.cost_limit,
        config=syn.CONFIG,
    )


def get_action_dim(mission_env, action_mode):
    if action_mode == "rule":
        return rule.Rule.RULE_NUM
    if action_mode == "op":
        return mission_env.T * mission_env.O
    raise ValueError(f"Unknown action_mode: {action_mode}")


def get_action_mask(mission_env, action_mode):
    assign_mask = mission_env.mask_invalid_assign()
    if action_mode == "rule":
        if np.any(assign_mask):
            return np.ones(rule.Rule.RULE_NUM, dtype=np.float32)
        return np.zeros(rule.Rule.RULE_NUM, dtype=np.float32)
    if action_mode == "op":
        return np.max(assign_mask, axis=2).reshape(-1).astype(np.float32)
    raise ValueError(f"Unknown action_mode: {action_mode}")


def op_to_env_action(mission_env, rule_policy, op_action):
    task_idx, op_idx = np.unravel_index(int(op_action), (mission_env.T, mission_env.O))
    assign_mask = mission_env.mask_invalid_assign()
    sys_indices = np.flatnonzero(assign_mask[task_idx, op_idx] > 0)
    if len(sys_indices) == 0:
        raise ValueError(f"No feasible system for task {task_idx}, operation {op_idx}.")

    sys_idx = rule_policy.cssa(mission_env, int(task_idx), int(op_idx), sys_indices)
    env_action = mission_env.encode_action(
        "assign_task",
        task_idx=int(task_idx),
        op_idx=int(op_idx),
        sys_idx=int(sys_idx),
    )
    return int(env_action), {"task_idx": int(task_idx), "op_idx": int(op_idx), "sys_idx": int(sys_idx)}


def dqn_action_to_env_action(mission_env, rule_policy, dqn_action, action_mode):
    if action_mode == "rule":
        env_action = rule_policy.to_env_action(dqn_action)
        return int(env_action), {"rule_action": int(dqn_action)}
    if action_mode == "op":
        env_action, info = op_to_env_action(mission_env, rule_policy, dqn_action)
        info["op_action"] = int(dqn_action)
        return env_action, info
    raise ValueError(f"Unknown action_mode: {action_mode}")


def run_env_step(mission_env, rule_policy, dqn_action, action_mode):
    try:
        env_action, action_info = dqn_action_to_env_action(mission_env, rule_policy, dqn_action, action_mode)
    except ValueError as exc:
        return mission_env.state.to_obs(), -10.0, False, True, {"valid": False, "dead_end": True, "info": str(exc)}

    obs, reward, terminated, truncated, info = mission_env.step(env_action)
    info.update(action_info)
    info["dqn_action"] = int(dqn_action)
    info["env_action"] = int(env_action)

    return obs, float(reward), bool(terminated), bool(truncated), info


def schedule_rows(mission_env, episode, scenario_idx, action_mode):
    rows = []
    for task_idx in range(mission_env.T):
        task = mission_env.state.mission[task_idx]
        for op_idx in range(mission_env.O):
            sys_idx = int(mission_env.state.op_assign_sys[task_idx, op_idx])
            if sys_idx < 0:
                continue

            op = task.operations[op_idx]
            system = env.FULL_SOS[sys_idx]
            start_time = float(mission_env.state.op_start_time[task_idx, op_idx])
            finish_time = float(mission_env.state.op_finish_time[task_idx, op_idx])
            rows.append(
                {
                    "episode": int(episode),
                    "scenario_idx": int(scenario_idx),
                    "action_mode": action_mode,
                    "task_idx": int(task_idx),
                    "task_name": task.name,
                    "op_idx": int(op_idx),
                    "op_name": op.name,
                    "func_type": op.func_type,
                    "sys_idx": sys_idx,
                    "sys_name": system.name,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "duration": finish_time - start_time,
                }
            )
    return sorted(rows, key=lambda item: (item["episode"], item["start_time"], item["task_idx"], item["op_idx"]))


def train_dqn(config=None, arch=None, mission=None, scenario_pool=None):
    if config is None:
        config = DQNConfig()

    set_seed(config.seed)
    if scenario_pool is None:
        scenario_pool = make_scenario_pool(config, arch=arch, mission=mission)

    _, init_arch, init_mission = scenario_pool.get(0)
    mission_env = make_mission_env(init_arch, init_mission)
    obs_dim = mission_env.observation_space.shape[0]
    action_dim = get_action_dim(mission_env, config.action_mode)
    agent = DQNAgent(obs_dim, action_dim, config)
    epsilon = config.epsilon_start
    history = []

    for episode in range(config.episodes):
        if config.scenario_order == "sequential":
            scenario_idx, arch, mission = scenario_pool.get(episode)
        elif config.scenario_order == "random":
            scenario_idx, arch, mission = scenario_pool.sample()
        else:
            raise ValueError(f"Unknown scenario_order: {config.scenario_order}")
        mission_env = make_mission_env(arch, mission)
        obs, _ = mission_env.reset()
        rule_policy = rule.Rule(mission_env)
        total_reward = 0.0
        last_loss = None
        terminated = False
        truncated = False
        info = {"valid": True}
        n_step_buffer = deque()

        for step in range(config.max_steps):
            mask = get_action_mask(mission_env, config.action_mode)
            if not np.any(mask):
                next_obs = mission_env.state.to_obs()
                reward = -10.0
                terminated = False
                truncated = False
                info = {"valid": False, "dead_end": True, "info": "No valid DQN action."}
                dqn_action = 0
            else:
                dqn_action = agent.select_action(obs, epsilon, mask)
                next_obs, reward, terminated, truncated, info = run_env_step(
                    mission_env,
                    rule_policy,
                    dqn_action,
                    config.action_mode,
                )

            done = terminated or truncated or info.get("dead_end", False) or not info.get("valid", True)
            next_mask = get_action_mask(mission_env, config.action_mode)
            n_step_buffer.append((obs, dqn_action, reward, next_obs, done, next_mask))
            if len(n_step_buffer) >= config.n_step:
                add_n_step_transition(agent.replay_buffer, n_step_buffer, config.gamma)
                n_step_buffer.popleft()
                loss = agent.learn()
                if loss is not None:
                    last_loss = loss

            obs = next_obs
            total_reward += reward
            if done:
                break

        while n_step_buffer:
            add_n_step_transition(agent.replay_buffer, n_step_buffer, config.gamma)
            n_step_buffer.popleft()
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        history.append(
            {
                "episode": episode,
                "scenario_idx": scenario_idx,
                "scenario_sos": len(arch),
                "scenario_cost": sum(system.cost for system in arch),
                "action_mode": config.action_mode,
                "reward": total_reward,
                "steps": step + 1,
                "done": terminated or truncated or info.get("dead_end", False) or not info.get("valid", True),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "dead_end": bool(info.get("dead_end", False)),
                "makespan": float(mission_env.state.current_makespan),
                "done_ops": int(mission_env.state.task_op_idx.sum()),
                "epsilon": epsilon,
                "loss": last_loss,
                "replay_size": len(agent.replay_buffer),
            }
        )

        if episode == 0 or (episode + 1) % 10 == 0:
            print(history[-1])

    return agent, history


def evaluate_dqn(agent, episodes=5, arch=None, mission=None, scenario_pool=None, max_steps=300, collect_schedule=False):
    if scenario_pool is None:
        if arch is not None or mission is not None:
            scenario_pool = ScenarioPool(
                scenarios=[(
                    env.FULL_SOS if arch is None else arch,
                    syn.build_mission_from_config(syn.CONFIG) if mission is None else mission,
                )]
            )
        else:
            scenario_pool = ScenarioPool(
                size=episodes,
                selected_system_num=agent.config.selected_system_num,
                min_system_num=agent.config.min_system_num,
                max_system_num=agent.config.max_system_num,
                cost_limit=agent.config.cost_limit,
            )

    results = []
    for episode in range(episodes):
        scenario_idx, arch, mission = scenario_pool.get(episode)
        mission_env = make_mission_env(arch, mission)
        obs, _ = mission_env.reset()
        rule_policy = rule.Rule(mission_env)
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {"valid": True}

        for step in range(max_steps):
            mask = get_action_mask(mission_env, agent.config.action_mode)
            if not np.any(mask):
                reward = -10.0
                terminated = False
                truncated = False
                info = {"valid": False, "dead_end": True, "info": "No valid DQN action."}
            else:
                dqn_action = agent.select_action(obs, epsilon=0.0, mask=mask)
                obs, reward, terminated, truncated, info = run_env_step(
                    mission_env,
                    rule_policy,
                    dqn_action,
                    agent.config.action_mode,
                )

            total_reward += reward
            if terminated or truncated or info.get("dead_end", False) or not info.get("valid", True):
                break

        result = {
            "episode": episode,
            "scenario_idx": scenario_idx,
            "action_mode": agent.config.action_mode,
            "reward": total_reward,
            "steps": step + 1,
            "makespan": float(mission_env.state.current_makespan),
            "done_ops": int(mission_env.state.task_op_idx.sum()),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "dead_end": bool(info.get("dead_end", False)),
        }
        if collect_schedule:
            result["schedule"] = schedule_rows(mission_env, episode, scenario_idx, agent.config.action_mode)
        results.append(result)

    return results


if __name__ == "__main__":
    config = DQNConfig(action_mode="op", episodes=20, min_buffer_size=64, batch_size=32)
    trained_agent, train_history = train_dqn(config)
    eval_results = evaluate_dqn(trained_agent, episodes=3)
    print("eval:", eval_results)
