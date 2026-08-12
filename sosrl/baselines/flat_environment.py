import gymnasium as gym
import numpy as np

from .. import domain as syn


FULL_SOS = syn.FULL_SOS
N = len(FULL_SOS)


class State:
    def __init__(self, mission: list[syn.Task]):
        task_num = len(mission)
        op_num = len(mission[0].operations)

        self.M = max(
            1.0,
            float(sum(op.duration for task in mission for op in task.operations)),
        )
        self.total_cost = max(
            1.0,
            float(sum(system.cost for system in FULL_SOS)),
        )

        self.op_assign_sys = np.full((task_num, op_num), -1, dtype=np.int32)
        self.op_start_time = np.full((task_num, op_num), -1.0, dtype=np.float32)
        self.op_finish_time = np.full((task_num, op_num), -1.0, dtype=np.float32)
        self.op_duration = np.asarray(
            [[op.duration for op in task.operations] for task in mission],
            dtype=np.float32,
        )
        self.op_ready_time = np.asarray(
            [[op.release_time for op in task.operations] for task in mission],
            dtype=np.float32,
        )

        self.select_sys_mask = np.zeros(N, dtype=bool)
        self.sys_ready_time = np.asarray(
            [system.available_from for system in FULL_SOS],
            dtype=np.float32,
        )
        self.task_op_idx = np.zeros(task_num, dtype=np.int32)
        self.cur_makespan = 0.0
        self.cur_cost = 0.0

    def to_obs(self) -> np.ndarray:
        op_num = self.op_duration.shape[1]
        return np.concatenate(
            [
                self.select_sys_mask.astype(np.float32),
                self.sys_ready_time / self.M,
                self.task_op_idx.astype(np.float32) / op_num,
                self.op_duration.reshape(-1) / self.M,
                self.op_ready_time.reshape(-1) / self.M,
                np.asarray(
                    [
                        self.cur_makespan / self.M,
                        self.cur_cost / self.total_cost,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)


class IntEnv(gym.Env):
    """Joint system-selection and operation-scheduling environment."""

    def __init__(self, mission: list[syn.Task]):
        super().__init__()
        if not mission:
            raise ValueError("mission cannot be empty.")

        operation_counts = {len(task.operations) for task in mission}
        if len(operation_counts) != 1 or 0 in operation_counts:
            raise ValueError("all tasks must have the same non-zero operation count.")

        self.mission = mission
        self.T = len(mission)
        self.O = len(mission[0].operations)
        self.N = N
        self.state = State(mission)

        self.action_space = gym.spaces.Discrete(self.T * self.O * self.N)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.state.to_obs().shape,
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.state = State(self.mission)
        info = {"dead_end": not np.any(self.valid_assignment_mask())}
        return self.state.to_obs(), info

    def encode_asg(self, task_idx: int, op_idx: int, sys_idx: int) -> int:
        if not 0 <= task_idx < self.T:
            raise ValueError(f"task_idx out of range: {task_idx}")
        if not 0 <= op_idx < self.O:
            raise ValueError(f"op_idx out of range: {op_idx}")
        if not 0 <= sys_idx < self.N:
            raise ValueError(f"sys_idx out of range: {sys_idx}")
        return (task_idx * self.O + op_idx) * self.N + sys_idx

    def decode_asg(self, action: int) -> tuple[int, int, int]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"assignment action out of range: {action}")
        task_idx, remainder = divmod(action, self.O * self.N)
        op_idx, sys_idx = divmod(remainder, self.N)
        return task_idx, op_idx, sys_idx

    def assignment_times(
        self,
        task_idx: int,
        op_idx: int,
        sys_idx: int,
    ) -> tuple[float, float] | None:
        if not (
            0 <= task_idx < self.T
            and 0 <= op_idx < self.O
            and 0 <= sys_idx < self.N
        ):
            return None
        if int(self.state.task_op_idx[task_idx]) != op_idx:
            return None

        operation = self.mission[task_idx].operations[op_idx]
        system = FULL_SOS[sys_idx]
        if system.func_type != operation.func_type:
            return None

        start_time = max(
            float(self.state.sys_ready_time[sys_idx]),
            float(self.state.op_ready_time[task_idx, op_idx]),
        )
        finish_time = start_time + float(operation.duration)
        if finish_time > float(system.available_until):
            return None
        return start_time, finish_time

    def valid_assignment_mask(self) -> np.ndarray:
        mask = np.zeros((self.T, self.O, self.N), dtype=bool)
        for task_idx in range(self.T):
            op_idx = int(self.state.task_op_idx[task_idx])
            if op_idx >= self.O:
                continue
            for sys_idx in range(self.N):
                if self.assignment_times(task_idx, op_idx, sys_idx) is not None:
                    mask[task_idx, op_idx, sys_idx] = True
        return mask

    def valid_action_mask(self) -> np.ndarray:
        return self.valid_assignment_mask().reshape(-1)

    def step(self, action: int):
        try:
            task_idx, op_idx, sys_idx = self.decode_asg(action)
        except (TypeError, ValueError) as error:
            info = {
                "valid": False,
                "success": False,
                "dead_end": False,
                "error": str(error),
            }
            return self.state.to_obs(), -1.0, False, False, info

        times = self.assignment_times(task_idx, op_idx, sys_idx)
        if times is None:
            info = {"valid": False, "success": False, "dead_end": False}
            return self.state.to_obs(), -1.0, False, False, info

        start_time, finish_time = times
        state = self.state
        operation = self.mission[task_idx].operations[op_idx]
        old_makespan = float(state.cur_makespan)

        first_use = not bool(state.select_sys_mask[sys_idx])
        cost_delta = 0.0
        if first_use:
            state.select_sys_mask[sys_idx] = True
            cost_delta = float(FULL_SOS[sys_idx].cost)
            state.cur_cost += cost_delta

        state.op_assign_sys[task_idx, op_idx] = sys_idx
        state.op_start_time[task_idx, op_idx] = start_time
        state.op_finish_time[task_idx, op_idx] = finish_time
        state.sys_ready_time[sys_idx] = finish_time
        state.cur_makespan = max(state.cur_makespan, finish_time)
        state.task_op_idx[task_idx] += 1

        next_op_idx = op_idx + 1
        if next_op_idx < self.O:
            state.op_ready_time[task_idx, next_op_idx] = max(
                float(state.op_ready_time[task_idx, next_op_idx]),
                finish_time,
            )

        reward = -(
            (state.cur_makespan - old_makespan) / state.M
            + cost_delta / state.total_cost
        )
        success = bool(np.all(state.task_op_idx == self.O))
        dead_end = not success and not np.any(self.valid_assignment_mask())
        info = {
            "valid": True,
            "success": success,
            "dead_end": dead_end,
            "first_use": first_use,
            "cost_delta": cost_delta,
        }
        return state.to_obs(), float(reward), success or dead_end, False, info
