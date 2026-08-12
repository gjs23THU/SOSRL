import random

import numpy as np
import torch

from .. import domain as syn
from ..rl.agent import IntDQNAgent, QNetwork
from ..rl.config import IntDQNConfig
from ..rl.replay import FlatReplayBuffer as ReplayBuffer
from . import flat_environment as intenv


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
