import gymnasium as gym  
import numpy as np
import syn
import math, json
from pathlib import Path
from typing import Any
from os import PathLike

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def load_config(path: str | PathLike[str] = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)

CONFIG = load_config()

FULL_SOS= syn.build_sos_from_config(CONFIG)
FULL_COST = sum(s.cost for s in FULL_SOS)

O = CONFIG.get("op_per_task", 4)
T = CONFIG.get("total_task", 30)
N = len(FULL_SOS)
M = 1500  # Example value, replace with actual maximum time if available

class State:
    def __init__(self, arch, mission):
        # self.arch = arch
        # self.mission = mission
        self.op_assign_sys = np.full((len(mission), O), -1, dtype=np.int32)  # -1 indicates unassigned
        self.op_release_time = np.array([[op.release_time for op in task.operations] for task in mission], dtype=np.float32)  # -1 indicates not started
        self.op_start_time = np.full((len(mission), O), -1.0, dtype=np.float32)  # -1 indicates not started
        self.op_finish_time = np.full((len(mission), O), -1.0, dtype=np.float32)  # -1 indicates not finished
        self.op_duration = np.array([[op.duration for op in task.operations] for task in mission], dtype=np.float32)
        self.current_makespan = 1e-6
        self.M = sum(op.duration for task in mission for op in task.operations)  # Total duration of all operations in the mission  
        self.sys_sel_idx = np.array([s.func_type if s in arch else 0 for s in FULL_SOS], dtype=np.float32)  # 0 indicates not selected, otherwise the function type of the system
        self.sys_work_idx = np.zeros([len(FULL_SOS)], dtype=np.int32)
        self.sys_availble_time = np.array([s.available_from if s in arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.sys_availble_until = np.array([s.available_until if s in self.arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.sys_release_time = np.array([s.available_from if s in arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.current_time = 0.0
        # self.sos_cost = sum(s.cost for s in arch)
        self.sys_busy_time = np.array([0 for _ in FULL_SOS], dtype=np.float32)
        self.sys_idle_time = np.array([0 for _ in FULL_SOS], dtype=np.float32)
        self.task_op_idx = np.array([0 for _ in mission], dtype=np.int32)
        self.task_current_op_type = np.array([task.operations[0].func_type for task in mission], dtype=np.float32)
        self.task_completion_time = np.array([math.inf for _ in mission], dtype=np.float32)
        self.task_release_time = np.array([task.release_time for task in mission], dtype=np.float32)
        self.task_due_time = np.array([task.due_time for task in mission], dtype=np.float32)
        

    def state_dim(self):
        return ( 5 * N + 2 * T + 2 * T * O + 2)  # 5 features for each system + 2 features for each task + 2 features for each task-operation pair + 2 global features (current_makespan, current_time)

    def to_obs(self):
        obs = np.concatenate([
            self.sys_sel_idx,   # n dimensions, 0 if not selected, otherwise the function type of the system 
            self.sys_work_idx, # n dimensions, 1 if working, 0 if idle
            (self.op_duration.flatten() / self.M), # T*O dimensions, normalized operation duration
            (self.op_release_time.flatten() / self.M), # T*O dimensions, normalized operation release time 
            (self.sys_availble_time / self.M), # n dimensions, normalized system available time
            (self.sys_busy_time / (self.current_time + 1e-6)), # n dimensions, normalized system busy time
            (self.sys_idle_time / (self.current_time + 1e-6)), # n dimensions, normalized system idle time
            self.task_op_idx / O,    # T dimensions, completion rate for each task, assuming max 4 operations per task
            self.task_current_op_type, # T dimensions, current operation type for each task
            # np.array([self.sos_cost / FULL_COST], dtype=np.float32), # 1 dimension, total cost of selected systems normalized by the total cost of all systems
            np.array([self.current_makespan / self.M], dtype=np.float32), # 1 dimension, current makespan normalized by the maximum time
            np.array([self.current_time / self.M], dtype=np.float32) # 1 dimension, current time normalized by the maximum time
        ])
        return np.array(obs, dtype=np.float32)
    
    def reset_state(self):
        self.op_assign_sys.fill(-1)
        self.op_release_time = np.array([[op.release_time for op in task.operations] for task in self.mission], dtype=np.float32)
        self.op_start_time.fill(-1.0)
        self.op_finish_time.fill(-1.0)
        self.current_makespan = 1e-6
        self.sys_sel_idx = np.array([s.func_type if s in self.arch else 0 for s in FULL_SOS], dtype=np.float32)
        self.sys_work_idx.fill(0)
        self.sys_availble_time = np.array([s.available_from if s in self.arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.sys_availble_until = np.array([s.available_until if s in self.arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.sys_release_time = np.array([s.available_from if s in self.arch else self.M for s in FULL_SOS], dtype=np.float32)
        self.task_current_op_type = np.array([task.operations[0].func_type for task in self.mission], dtype=np.float32)
        self.current_time = 0.0
        # self.sos_cost = sum(s.cost for s in self.arch)
        self.sys_busy_time.fill(0.0)
        self.sys_idle_time.fill(0.0)
        self.task_op_idx.fill(0)
        self.task_completion_time.fill(math.inf)

    def alocate_op2sys(self, task_idx:int, op_idx:int, sys_idx:int):
        if self.task_op_idx[task_idx] != op_idx:
            raise ValueError(f"Operation {op_idx} of task {task_idx} is not available for assignment. Current operation index: {self.task_op_idx[task_idx]}")
        
        if self.sys_sel_idx[sys_idx] == 0:
            raise ValueError(f"System {sys_idx} is not selected.")
        
        op = self.mission[task_idx].operations[op_idx]
        if op.func_type != FULL_SOS[sys_idx].func_type:
            raise ValueError(f"Operation {op_idx} of task {task_idx} requires function type '{op.func_type}', but system {sys_idx} has function type '{FULL_SOS[sys_idx].func_type}'.")
        
        sys_available_from = self.sys_availble_time[sys_idx]
        op_release_time = self.op_release_time[task_idx][op_idx]

        start_time = max(sys_available_from, op_release_time, self.current_time)
        finish_time = start_time + op.duration
        if finish_time > FULL_SOS[sys_idx].available_until:
            raise ValueError(f"System {sys_idx} cannot finish operation {op_idx} of task {task_idx} within its available time window.")
        
        self.advance_time_to(start_time)  # Advance the current time to the start time of the operation
        self.op_assign_sys[task_idx][op_idx] = sys_idx # update the assigned system for the operation
        self.op_start_time[task_idx][op_idx] = start_time # update the start time for the operation
        self.op_finish_time[task_idx][op_idx] = finish_time # update the finish time for the operation
        self.sys_work_idx[sys_idx] = 1  # mark the system as working
        self.sys_availble_time[sys_idx] = self.op_finish_time[task_idx][op_idx] # update the system's available time
        self.current_makespan = max(self.current_makespan, self.sys_availble_time[sys_idx])  # Update the current makespan
        

        # Update task state
        self.task_op_idx[task_idx] += 1  # Move to the next operation for the task
        if self.task_op_idx[task_idx] == O:  # If all operations for the task are completed
            self.task_completion_time[task_idx] = self.op_finish_time[task_idx][O - 1]  # Update the task's completion time
        else:
            self.task_current_op_type[task_idx] = self.mission[task_idx].operations[op_idx + 1].func_type  # Update the current operation type for the task
            self.op_release_time[task_idx][op_idx + 1] = self.op_finish_time[task_idx][op_idx] # update the release time for the operation

    def add_system(self, sys_idx:int):
        if self.sys_sel_idx[sys_idx] != 0:
            raise ValueError(f"System {sys_idx} is already selected.")
        self.sys_sel_idx[sys_idx] = FULL_SOS[sys_idx].func_type
        self.sys_availble_time[sys_idx] = max(self.current_time, FULL_SOS[sys_idx].available_from)
        self.sys_release_time[sys_idx] = self.sys_availble_time[sys_idx]  # Set the system's release time to its available time
        self.sys_availble_until[sys_idx] = FULL_SOS[sys_idx].available_until
        self.sys_work_idx[sys_idx] = 0  # Mark the system as not working
        self.sos_cost += FULL_SOS[sys_idx].cost
    
    def remove_system(self, sys_idx:int):
        if self.sys_sel_idx[sys_idx] == 0:
            raise ValueError(f"System {sys_idx} is not selected.")
        if self.sys_work_idx[sys_idx] == 1:
            raise ValueError(f"System {sys_idx} is currently working and cannot be removed.")
        self.sys_sel_idx[sys_idx] = 0
        self.sys_availble_time[sys_idx] = self.M  # Set the system's available time to the maximum time
        self.sys_work_idx[sys_idx] = 0  # Mark the system as not working
        self.sos_cost -= FULL_SOS[sys_idx].cost / 2 # Update the total cost of selected systems, assuming a penalty for removing a system

    def advance_time_to(self, next_time):
        if next_time < self.current_time:
            raise ValueError(f"Cannot advance time backwards. Current time: {self.current_time}, Next time: {next_time}")
        elif self.current_time == next_time:
            return  # No time advancement needed
        
        old_time = self.current_time
        self.current_time = next_time
        # update the state of other systems
        for sys_idx in range(N):
            if self.sys_sel_idx[sys_idx] == 0:  # If the system is not selected, skip it
                continue
            if self.sys_work_idx[sys_idx] == 1:  # If the system is working
                finish_time = self.sys_availble_time[sys_idx]

                busy_delta = max(0, min(finish_time, next_time) - old_time)
                idle_delta = max(
                    0, 
                    min(next_time, self.sys_availble_until[sys_idx]) 
                    - max(old_time, finish_time, self.sys_release_time[sys_idx]))

                self.sys_busy_time[sys_idx] += busy_delta
                self.sys_idle_time[sys_idx] += idle_delta
            
                if finish_time <= next_time:
                    self.sys_work_idx[sys_idx] = 0  # Mark the system as not working if it has finished its operation
            else:
                idle_delta = max(0, 
                                 min(next_time, self.sys_availble_until[sys_idx]) 
                                 - max(old_time, self.sys_release_time[sys_idx]))
                self.sys_idle_time[sys_idx] += idle_delta  # If the system is idle, increase its idle time by the delta

class MissionEnv(gym.Env):
    def __init__(self, arch: list[syn.ComponentSystem], mission: list[syn.Task]):
        # Initialize the environment with architecture and mission, if no architecture or mission is provided, use the default ones from the configuration
        super(MissionEnv, self).__init__()
        self.arch = arch
        self.mission = mission
        self.state = State(self.arch, self.mission)  # Initialize state with None or appropriate initial 
        self.step_count = 0
        self.max_steps = 1000  # Example maximum number of steps, can be adjusted based on the problem
        self.T = len(self.mission)
        self.O = O
        self.N = N
        self.current_reward = 0.0
        self.action_space = gym.spaces.Discrete((self.T * self.O + 2) * self.N)  # first len(mission) * O * N actions are to assign tasks to systems, last 2*N actions are to select/deselect systems
        # Example for using image as input:
        self.observation_space = gym.spaces.Box(low=-2, high=2, shape=(self.state.state_dim(),), dtype=np.float32)  # Observation space is a vector of floats

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # Reset the state of the environment to an initial state
        self.state = State(self.arch, self.mission)  # Reset state with initial architecture and mission
        self.step_count = 0
        self.current_reward = 0.0
        return self.state.to_obs(), {}  # Return initial observation
    
    def decode_action(self, action:int) -> tuple[int, int]:
        assign_size = self.T * self.O * self.N
        delete_sys = assign_size + self.N

        if action < assign_size:
            task_idx, op_idx, sys_idx = np.unravel_index(action, (self.T, self.O, self.N)) # Decode the action into task index, operation index, and system index, for example (30, 4, 22)
            return {
                "type": "assign_task",
                "task_idx": task_idx,
                "op_idx": op_idx,
                "sys_idx": sys_idx
            }
        elif action < delete_sys:
            sys_idx = action - assign_size
            return {
                "type": "select_system",
                "sys_idx": sys_idx
            }
        else:
            sys_idx = action - delete_sys
            return {
                "type": "deselect_system",
                "sys_idx": sys_idx
            }

    def encode_action(self, act_type:str, task_idx=None, op_idx=None, sys_idx=None) -> int:
        if act_type == "assign_task":
            return np.ravel_multi_index((task_idx, op_idx, sys_idx), (self.T, self.O, self.N))
        elif act_type == "select_system":
            return self.T * self.O * self.N + sys_idx
        elif act_type == "deselect_system":
            return self.T * self.O * self.N + self.N + sys_idx
        else:
            raise ValueError(f"Unknown action type: {act_type}")
    
    def mask_invalid_actions(self):

        # Create a mask for invalid actions based on the current state
        select_mask = np.ones(self.N, dtype=np.float32)  # Start with all actions valid
        deselect_mask = np.ones(self.N, dtype=np.float32)  # Start with all actions valid

        # mask invalid select and deselect actions based on system selection and working status
        for sys_idx in range(self.N):
            if self.state.sys_sel_idx[sys_idx] != 0 :  # If the system is selected
                select_mask[sys_idx] = 0  # Mask out the action to select this system
                if self.state.sys_work_idx[sys_idx] == 1:  # If the system is currently working
                    deselect_mask[sys_idx] = 0  # Mask out the action to deselect this system
            else:
                deselect_mask[sys_idx] = 0  # Mask out the action to deselect this system
        
        assign_mask = self.mask_invalid_assign()
        mask = np.concatenate([assign_mask.reshape(-1), select_mask, deselect_mask])  # Combine the assign mask with the select/deselect mask   
        return mask
    
    def mask_invalid_assign(self):
        # Create a mask for invalid assign actions based on the current state
        assign_mask = np.zeros((self.T, self.O, self.N), dtype=np.float32)  # Start with all actions invalid
        for task_idx in range(self.T):
            for op_idx in range(self.O):
                if self.state.task_op_idx[task_idx] == op_idx:  # If the operation is available for assignment
                    op = self.state.mission[task_idx].operations[op_idx]
                    for sys_idx in range(self.N):
                        if self.state.sys_sel_idx[sys_idx] != 0:  # If the system is selected
                            sys = FULL_SOS[sys_idx]
                            if sys.func_type == op.func_type:  # If the system function type is the type
                                start_time = max(
                                    self.state.current_time,
                                    self.state.sys_availble_time[sys_idx],
                                    self.state.op_release_time[task_idx, op_idx],
                                )
                                finish_time = start_time + op.duration

                                if finish_time <= sys.available_until:  # If the system can finish the operation in its time window
                                    assign_mask[task_idx, op_idx, sys_idx] = 1  # Mark the action as valid
        return assign_mask

    def apply_action(self, decoded_action: dict[str, Any]):
        if decoded_action["type"] == "assign_task":
            task_idx = decoded_action["task_idx"]
            op_idx = decoded_action["op_idx"]
            sys_idx = decoded_action["sys_idx"]
            self.state.alocate_op2sys(task_idx, op_idx, sys_idx)
        elif decoded_action["type"] == "select_system":
            self.state.add_system(decoded_action["sys_idx"])
        elif decoded_action["type"] == "deselect_system":
            self.state.remove_system(decoded_action["sys_idx"])
        # Implement action validation logic here

    def step(self, action):
        self.step_count += 1
        if self.step_count > self.max_steps:  # Example truncation condition
            return self.state.to_obs(), -10.0, False, True, {"valid": False, "dead_end": False, "info": "Step limit exceeded"}
        
        mask = self.mask_invalid_actions()
        if mask[action] == 0: # Check if the action is invalid based on the current state
            return self.state.to_obs(), -1.0, False, False, {"valid": False, "dead_end": False, "info": "Invalid action"}
        
        decoded_action = self.decode_action(action)

        old_makespan = float(self.state.current_makespan)
        self.apply_action(decoded_action)
        obs = self.state.to_obs()
        makespan_delta = max(0.0, float(self.state.current_makespan) - old_makespan)
        reward = -makespan_delta / self.state.M
        info = {
            "valid": True,
            "dead_end": False,
            "decode_action": decoded_action,
            "makespan": self.state.current_makespan,
            "makespan_delta": makespan_delta,
            "step_count": self.step_count,
        }

        terminated = np.all(self.state.task_op_idx == self.O)
        if terminated:
            return obs, float(reward), True, False, info

        truncated = self.step_count >= self.max_steps
        if truncated:
            return obs, float(reward - 10.0), False, True, info

        dead_end = not np.any(self.mask_invalid_assign())
        if dead_end:
            unfinished_ops = self.T * self.O - int(self.state.task_op_idx.sum())
            reward -= unfinished_ops / self.T
            info["dead_end"] = True
            return obs, float(reward), True, False, info

        return obs, float(reward), False, False, info

    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def close(self):
        # Clean up resources
        pass
