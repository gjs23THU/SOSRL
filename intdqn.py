from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import intenv
import syn


@dataclass
class IntDQNConfig:
    episodes: int = 1000
    fixed_mission: bool = False
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 1000
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dim: int = 512
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, next_mask):
        self.buffer.append(
            (
                np.asarray(obs, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_obs, dtype=np.float32).copy(),
                bool(done),
                np.asarray(next_mask, dtype=bool).copy(),
            )
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, next_mask = zip(*batch)
        return (
            np.asarray(obs, dtype=np.float32),
            np.asarray(action, dtype=np.int64),
            np.asarray(reward, dtype=np.float32),
            np.asarray(next_obs, dtype=np.float32),
            np.asarray(done, dtype=np.float32),
            np.asarray(next_mask, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
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


class IntDQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, config: IntDQNConfig):
        self.config = config
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch.device(config.device)

        self.q_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_net = QNetwork(obs_dim, action_dim, config.hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = ReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def select_action(self, obs, action_mask, epsilon: float) -> int:
        action_mask = np.asarray(action_mask, dtype=bool)
        if action_mask.shape != (self.action_dim,):
            raise ValueError("action mask has the wrong shape.")

        valid_actions = np.flatnonzero(action_mask)
        if valid_actions.size == 0:
            raise ValueError("no valid assignment action.")

        if epsilon > 0.0 and random.random() < epsilon:
            return int(random.choice(valid_actions))

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(
            action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_tensor)
            q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
        return int(q_values.argmax(dim=1).item())

    def learn(self):
        required_size = max(self.config.batch_size, self.config.min_buffer_size)
        if len(self.replay) < required_size:
            return None

        obs, action, reward, next_obs, done, next_mask = self.replay.sample(
            self.config.batch_size
        )
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.int64, device=self.device)
        action = action.unsqueeze(1)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        reward = reward.unsqueeze(1)
        next_obs = torch.as_tensor(
            next_obs,
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        done = done.unsqueeze(1)
        next_mask = torch.as_tensor(next_mask, dtype=torch.bool, device=self.device)

        q_value = self.q_net(obs).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            next_q_values = next_q_values.masked_fill(~next_mask, -torch.inf)
            next_q = next_q_values.max(dim=1, keepdim=True).values
            has_next_action = next_mask.any(dim=1, keepdim=True)
            next_q = torch.where(has_next_action, next_q, torch.zeros_like(next_q))
            target = reward + self.config.gamma * (1.0 - done) * next_q

        loss = nn.functional.smooth_l1_loss(q_value, target)
        if not torch.isfinite(loss):
            raise RuntimeError("DQN loss became non-finite.")

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
        agent = cls(
            int(checkpoint["obs_dim"]),
            int(checkpoint["action_dim"]),
            IntDQNConfig(**values),
        )
        agent.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        if load_optimizer:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.learn_step = int(checkpoint["learn_step"])
        return agent, checkpoint


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_env(mission_seed: int):
    random_state = random.getstate()
    random.seed(mission_seed)
    mission = syn.build_mission_from_config(syn.CONFIG)
    random.setstate(random_state)
    return intenv.IntEnv(mission)


def train_intdqn(config: IntDQNConfig):
    set_seed(config.seed)
    probe_env = build_env(config.seed)
    obs_dim = int(probe_env.observation_space.shape[0])
    action_dim = int(probe_env.action_space.n)
    agent = IntDQNAgent(obs_dim, action_dim, config)

    epsilon = config.epsilon_start
    history = []
    for episode in range(config.episodes):
        if config.fixed_mission:
            mission_seed = config.seed
        else:
            mission_seed = config.seed + episode
        mission_env = build_env(mission_seed)
        if mission_env.observation_space.shape[0] != obs_dim:
            raise ValueError("mission changed the observation dimension.")
        if mission_env.action_space.n != action_dim:
            raise ValueError("mission changed the action dimension.")

        obs, reset_info = mission_env.reset(seed=mission_seed)
        total_reward = 0.0
        last_loss = None
        terminated = bool(reset_info.get("dead_end", False))
        info = {
            "valid": True,
            "success": False,
            "dead_end": terminated,
        }

        for _ in range(mission_env.T * mission_env.O):
            if terminated:
                break
            action_mask = mission_env.valid_action_mask()
            if not np.any(action_mask):
                info["dead_end"] = True
                terminated = True
                break

            action = agent.select_action(obs, action_mask, epsilon)
            next_obs, reward, terminated, _, info = mission_env.step(action)
            if not info.get("valid", False):
                raise RuntimeError("the agent selected a masked assignment action.")

            next_mask = mission_env.valid_action_mask()
            agent.replay.add(
                obs,
                action,
                reward,
                next_obs,
                terminated,
                next_mask,
            )
            loss = agent.learn()
            if loss is not None:
                last_loss = loss

            obs = next_obs
            total_reward += reward

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        row = {
            "episode": episode,
            "mission_seed": mission_seed,
            "reward": float(total_reward),
            "makespan": float(mission_env.state.cur_makespan),
            "cost": float(mission_env.state.cur_cost),
            "selected_systems": int(mission_env.state.select_sys_mask.sum()),
            "assigned_ops": int(mission_env.state.task_op_idx.sum()),
            "success": bool(info.get("success", False)),
            "dead_end": bool(info.get("dead_end", False)),
            "epsilon": float(epsilon),
            "loss": last_loss,
            "replay_size": len(agent.replay),
        }
        history.append(row)

        if config.log_interval > 0 and (
            (episode + 1) % config.log_interval == 0
            or episode == 0
            or episode + 1 == config.episodes
        ):
            loss_text = "-" if last_loss is None else f"{last_loss:.6f}"
            print(
                f"episode={episode + 1}/{config.episodes} "
                f"reward={total_reward:.6f} "
                f"makespan={mission_env.state.cur_makespan:.1f} "
                f"cost={mission_env.state.cur_cost:.0f} "
                f"success={row['success']} epsilon={epsilon:.3f} "
                f"loss={loss_text}"
            )

    return agent, history


def schedule_rows(mission_env, episode: int, mission_seed: int):
    rows = []
    for task_idx in range(mission_env.T):
        task = mission_env.mission[task_idx]
        for op_idx in range(mission_env.O):
            sys_idx = int(mission_env.state.op_assign_sys[task_idx, op_idx])
            if sys_idx < 0:
                continue
            start_time = float(mission_env.state.op_start_time[task_idx, op_idx])
            finish_time = float(mission_env.state.op_finish_time[task_idx, op_idx])
            rows.append(
                {
                    "episode": int(episode),
                    "mission_seed": int(mission_seed),
                    "action_mode": "intdqn",
                    "task_idx": int(task_idx),
                    "task_name": task.name,
                    "op_idx": int(op_idx),
                    "op_name": task.operations[op_idx].name,
                    "func_type": int(task.operations[op_idx].func_type),
                    "sys_idx": sys_idx,
                    "sys_name": intenv.FULL_SOS[sys_idx].name,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "duration": finish_time - start_time,
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["start_time"], row["task_idx"], row["op_idx"]),
    )


def evaluate_intdqn(
    agent: IntDQNAgent,
    episodes: int,
    eval_seed: int,
    collect_schedule: bool = True,
):
    results = []
    for episode in range(episodes):
        if agent.config.fixed_mission:
            mission_seed = agent.config.seed
        else:
            mission_seed = eval_seed + episode
        mission_env = build_env(mission_seed)
        if mission_env.observation_space.shape[0] != agent.obs_dim:
            raise ValueError("evaluation observation dimension does not match the model.")
        if mission_env.action_space.n != agent.action_dim:
            raise ValueError("evaluation action dimension does not match the model.")

        obs, reset_info = mission_env.reset(seed=mission_seed)
        total_reward = 0.0
        terminated = bool(reset_info.get("dead_end", False))
        info = {
            "valid": True,
            "success": False,
            "dead_end": terminated,
        }

        for _ in range(mission_env.T * mission_env.O):
            if terminated:
                break
            action_mask = mission_env.valid_action_mask()
            if not np.any(action_mask):
                info["dead_end"] = True
                break

            action = agent.select_action(obs, action_mask, epsilon=0.0)
            obs, reward, terminated, _, info = mission_env.step(action)
            if not info.get("valid", False):
                raise RuntimeError("the agent selected a masked evaluation action.")
            total_reward += reward

        result = {
            "episode": episode,
            "mission_seed": mission_seed,
            "reward": float(total_reward),
            "makespan": float(mission_env.state.cur_makespan),
            "cost": float(mission_env.state.cur_cost),
            "selected_systems": int(mission_env.state.select_sys_mask.sum()),
            "assigned_ops": int(mission_env.state.task_op_idx.sum()),
            "success": bool(info.get("success", False)),
            "dead_end": bool(info.get("dead_end", False)),
        }
        if collect_schedule:
            result["schedule"] = schedule_rows(
                mission_env,
                episode,
                mission_seed,
            )
        results.append(result)

    return results
