from typing import Any

import gymnasium as gym
import numpy as np

import syn


FULL_SOS = syn.FULL_SOS
N = len(FULL_SOS)
OBSERVATION_SIZE = 25


class State:
    """Numeric state for an immutable architecture and a partial list schedule."""

    def __init__(self, selected_system_mask: np.ndarray, mission: list[syn.Task]):
        task_num = len(mission)
        op_num = len(mission[0].operations)
        self.op_num = op_num
        self.M = max(
            1.0,
            float(sum(op.duration for task in mission for op in task.operations)),
        )

        self.selected_system_mask = np.array(selected_system_mask, dtype=bool, copy=True)

        self.op_assign_sys = np.full((task_num, op_num), -1, dtype=np.int32)
        self.op_start_time = np.full((task_num, op_num), -1.0, dtype=np.float32)
        self.op_finish_time = np.full((task_num, op_num), -1.0, dtype=np.float32)
        self.op_duration = np.asarray(
            [[op.duration for op in task.operations] for task in mission],
            dtype=np.float32,
        )
        self.operation_ready_time = np.asarray(
            [[op.release_time for op in task.operations] for task in mission],
            dtype=np.float32,
        )

        self.system_ready_time = np.full(N, np.inf, dtype=np.float32)
        for sys_idx in np.flatnonzero(self.selected_system_mask):
            self.system_ready_time[sys_idx] = float(FULL_SOS[sys_idx].available_from)
        self.system_busy_time = np.zeros(N, dtype=np.float32)
        self.system_idle_time = np.zeros(N, dtype=np.float32)

        self.current_makespan = 0.0
        self.task_op_idx = np.zeros(task_num, dtype=np.int32)
        self.task_due_time = np.asarray(
            [task.due_time for task in mission],
            dtype=np.float32,
        )

        self.task_candidate_mask = np.zeros(task_num, dtype=bool)
        self.task_waiting_mask = np.zeros(task_num, dtype=bool)
        self.task_earliest_start = np.zeros(task_num, dtype=np.float32)
        self.task_remaining_time = np.zeros(task_num, dtype=np.float32)
        self.task_ttd = np.zeros(task_num, dtype=np.float32)
        self.task_slack = np.zeros(task_num, dtype=np.float32)
        self.task_next_type_load = np.zeros(task_num, dtype=np.float32)
        self.system_ready_delay = np.zeros(N, dtype=np.float32)

    @staticmethod
    def cv(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.std(values) / np.mean(values)
        value = np.clip(value, -2.0, 2.0)
        return float(np.nan_to_num(value, nan=0.0, posinf=2.0, neginf=-2.0))

    def to_obs(self) -> np.ndarray:
        task_num = len(self.task_op_idx)
        task_denominator = max(task_num, 1)
        active_task_mask = self.task_op_idx < self.op_num
        candidate_mask = self.task_candidate_mask
        waiting_mask = self.task_waiting_mask

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size:
            current_duration = self.op_duration[
                candidate_indices,
                self.task_op_idx[candidate_indices],
            ]
        else:
            current_duration = np.empty(0, dtype=np.float32)

        candidate_remaining = self.task_remaining_time[candidate_mask]
        candidate_next_load = self.task_next_type_load[candidate_mask]
        candidate_ttd = self.task_ttd[candidate_mask]
        candidate_slack = self.task_slack[candidate_mask]
        waiting_slack = self.task_slack[waiting_mask]
        task_completion = self.task_op_idx / max(self.op_num, 1)
        selected_delay = self.system_ready_delay[self.selected_system_mask]

        obs = np.asarray(
            [
                np.count_nonzero(active_task_mask) / task_denominator,
                np.count_nonzero(candidate_mask) / task_denominator,
                np.count_nonzero(waiting_mask) / task_denominator,
                float(np.sum(current_duration)) / self.M if current_duration.size else 0.0,
                float(np.mean(current_duration)) / self.M if current_duration.size else 0.0,
                float(np.min(current_duration)) / self.M if current_duration.size else 0.0,
                float(np.sum(candidate_remaining)) / self.M if candidate_remaining.size else 0.0,
                float(np.mean(candidate_remaining)) / self.M if candidate_remaining.size else 0.0,
                float(np.max(candidate_remaining)) / self.M if candidate_remaining.size else 0.0,
                float(np.mean(candidate_next_load)) / self.M if candidate_next_load.size else 0.0,
                float(np.min(candidate_next_load)) / self.M if candidate_next_load.size else 0.0,
                float(np.mean(candidate_ttd)) / self.M if candidate_ttd.size else 0.0,
                float(np.min(candidate_ttd)) / self.M if candidate_ttd.size else 0.0,
                float(np.mean(candidate_slack)) / self.M if candidate_slack.size else 0.0,
                float(np.min(candidate_slack)) / self.M if candidate_slack.size else 0.0,
                float(np.min(waiting_slack)) / self.M if waiting_slack.size else 0.0,
                float(np.mean(selected_delay)) / self.M if selected_delay.size else 0.0,
                float(np.mean(task_completion)) if task_completion.size else 0.0,
                float(np.mean(candidate_ttd < 0)) if candidate_ttd.size else 0.0,
                float(np.mean(candidate_slack < 0)) if candidate_slack.size else 0.0,
                self.cv(current_duration),
                self.cv(candidate_remaining),
                self.cv(candidate_ttd),
                self.cv(candidate_slack),
                self.cv(candidate_next_load),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


class MissionEnv(gym.Env):
    """Offline tail-append scheduling environment for a fixed architecture."""

    def __init__(
        self,
        architecture: tuple[syn.ComponentSystem, ...] | list[syn.ComponentSystem],
        mission: list[syn.Task],
    ):
        super().__init__()
        if not mission:
            raise ValueError("mission must contain at least one task.")
        operation_counts = {len(task.operations) for task in mission}
        if len(operation_counts) != 1 or 0 in operation_counts:
            raise ValueError("all tasks must contain the same non-zero operation count.")

        architecture = tuple(architecture)
        architecture_indices = [int(system.index) for system in architecture]
        if len(set(architecture_indices)) != len(architecture_indices):
            raise ValueError("architecture contains duplicate system indices.")
        if any(index < 0 or index >= N for index in architecture_indices):
            raise ValueError("architecture contains an unknown system index.")

        self.mission = mission
        self.T = len(mission)
        self.O = len(mission[0].operations)
        self.N = N

        selected_system_mask = np.zeros(self.N, dtype=np.bool_)
        selected_system_mask[architecture_indices] = True
        self.selected_system_mask = selected_system_mask

        self.state = State(self.selected_system_mask, self.mission)
        assignment_shape = (self.T, self.O, self.N)
        self.assignment_mask = np.zeros(assignment_shape, dtype=bool)
        self.assignment_start_time = np.full(
            assignment_shape,
            np.inf,
            dtype=np.float32,
        )
        self.assignment_finish_time = np.full(
            assignment_shape,
            np.inf,
            dtype=np.float32,
        )
        self.system_indices_by_type: dict[int, list[int]] = {}
        for sys_idx in architecture_indices:
            func_type = int(FULL_SOS[sys_idx].func_type)
            self.system_indices_by_type.setdefault(func_type, []).append(sys_idx)

        self.action_space = gym.spaces.Discrete(self.T * self.O * self.N)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.refresh_derived_state()

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self.state = State(self.selected_system_mask, self.mission)
        self.refresh_derived_state()
        return self.state.to_obs(), {}

    def encode_assignment(self, task_idx: int, op_idx: int, sys_idx: int) -> int:
        task_idx = int(task_idx)
        op_idx = int(op_idx)
        sys_idx = int(sys_idx)
        if not 0 <= task_idx < self.T:
            raise ValueError(f"task_idx out of range: {task_idx}")
        if not 0 <= op_idx < self.O:
            raise ValueError(f"op_idx out of range: {op_idx}")
        if not 0 <= sys_idx < self.N:
            raise ValueError(f"sys_idx out of range: {sys_idx}")
        return (task_idx * self.O + op_idx) * self.N + sys_idx

    def decode_assignment(self, action: int) -> dict[str, int]:
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"assignment action out of range: {action}")
        task_idx, remainder = divmod(action, self.O * self.N)
        op_idx, sys_idx = divmod(remainder, self.N)
        return {
            "task_idx": int(task_idx),
            "op_idx": int(op_idx),
            "sys_idx": int(sys_idx),
        }

    def assignment_times(
        self,
        task_idx: int,
        op_idx: int,
        sys_idx: int,
    ) -> tuple[float, float] | None:
        task_idx = int(task_idx)
        op_idx = int(op_idx)
        sys_idx = int(sys_idx)
        if not (
            0 <= task_idx < self.T
            and 0 <= op_idx < self.O
            and 0 <= sys_idx < self.N
        ):
            return None
        if not self.assignment_mask[task_idx, op_idx, sys_idx]:
            return None
        return (
            float(self.assignment_start_time[task_idx, op_idx, sys_idx]),
            float(self.assignment_finish_time[task_idx, op_idx, sys_idx]),
        )

    def refresh_assignment_cache(self) -> np.ndarray:
        assignment_shape = (self.T, self.O, self.N)
        mask = np.zeros(assignment_shape, dtype=bool)
        start_times = np.full(assignment_shape, np.inf, dtype=np.float32)
        finish_times = np.full(assignment_shape, np.inf, dtype=np.float32)

        for task_idx in range(self.T):
            op_idx = int(self.state.task_op_idx[task_idx])
            if op_idx >= self.O:
                continue

            operation = self.mission[task_idx].operations[op_idx]
            operation_ready = float(
                self.state.operation_ready_time[task_idx, op_idx]
            )
            for sys_idx in self.system_indices_by_type.get(
                int(operation.func_type),
                [],
            ):
                start_time = max(
                    float(self.state.system_ready_time[sys_idx]),
                    operation_ready,
                )
                finish_time = start_time + float(operation.duration)
                if finish_time > float(FULL_SOS[sys_idx].available_until):
                    continue
                mask[task_idx, op_idx, sys_idx] = True
                start_times[task_idx, op_idx, sys_idx] = start_time
                finish_times[task_idx, op_idx, sys_idx] = finish_time

        self.assignment_mask = mask
        self.assignment_start_time = start_times
        self.assignment_finish_time = finish_times
        return self.assignment_mask

    def valid_assignment_mask(self) -> np.ndarray:
        return self.assignment_mask

    def valid_action_mask(self) -> np.ndarray:
        return self.valid_assignment_mask().reshape(-1)

    def refresh_derived_state(self) -> None:
        state = self.state
        state.task_candidate_mask.fill(False)
        state.task_waiting_mask.fill(False)
        state.task_earliest_start.fill(0.0)
        state.task_remaining_time.fill(0.0)
        state.task_ttd.fill(0.0)
        state.task_slack.fill(0.0)
        state.task_next_type_load.fill(0.0)
        state.system_ready_delay.fill(0.0)

        assignment_mask = self.refresh_assignment_cache()
        feasible_tasks: list[int] = []
        for task_idx in range(self.T):
            op_idx = int(state.task_op_idx[task_idx])
            if op_idx >= self.O:
                continue

            valid_systems = np.flatnonzero(assignment_mask[task_idx, op_idx] > 0)
            if valid_systems.size == 0:
                continue

            earliest_start = float(
                np.min(
                    self.assignment_start_time[
                        task_idx,
                        op_idx,
                        valid_systems,
                    ]
                )
            )
            estimated_current_finish = earliest_start + float(
                state.op_duration[task_idx, op_idx]
            )
            state.task_remaining_time[task_idx] = float(
                np.sum(state.op_duration[task_idx, op_idx + 1:])
            )
            state.task_candidate_mask[task_idx] = True
            state.task_earliest_start[task_idx] = float(earliest_start)
            state.task_ttd[task_idx] = float(
                state.task_due_time[task_idx] - estimated_current_finish
            )
            state.task_slack[task_idx] = float(
                state.task_ttd[task_idx] - state.task_remaining_time[task_idx]
            )
            feasible_tasks.append(task_idx)

            next_op_idx = op_idx + 1
            if next_op_idx >= self.O:
                continue
            next_type = self.mission[task_idx].operations[next_op_idx].func_type
            matching_ready_times = [
                float(state.system_ready_time[sys_idx])
                for sys_idx in self.system_indices_by_type.get(int(next_type), [])
            ]
            if matching_ready_times:
                state.task_next_type_load[task_idx] = float(
                    np.mean(
                        [
                            max(ready_time - estimated_current_finish, 0.0)
                            for ready_time in matching_ready_times
                        ]
                    )
                )

        if feasible_tasks:
            frontier = min(
                float(state.task_earliest_start[task_idx])
                for task_idx in feasible_tasks
            )
            for task_idx in feasible_tasks:
                if float(state.task_earliest_start[task_idx]) > frontier + 1e-9:
                    state.task_waiting_mask[task_idx] = True
            selected_indices = np.flatnonzero(self.selected_system_mask)
            state.system_ready_delay[selected_indices] = np.maximum(
                state.system_ready_time[selected_indices] - frontier,
                0.0,
            )

    def allocate_assignment(self, task_idx: int, op_idx: int, sys_idx: int) -> None:
        times = self.assignment_times(task_idx, op_idx, sys_idx)
        if times is None:
            raise ValueError(
                f"infeasible assignment: task={task_idx}, op={op_idx}, system={sys_idx}"
            )
        start_time, finish_time = times
        state = self.state
        operation = self.mission[task_idx].operations[op_idx]
        previous_system_ready = float(state.system_ready_time[sys_idx])

        state.op_assign_sys[task_idx, op_idx] = sys_idx
        state.op_start_time[task_idx, op_idx] = start_time
        state.op_finish_time[task_idx, op_idx] = finish_time
        state.system_idle_time[sys_idx] += max(
            0.0,
            start_time - previous_system_ready,
        )
        state.system_busy_time[sys_idx] += float(operation.duration)
        state.system_ready_time[sys_idx] = finish_time
        state.current_makespan = max(float(state.current_makespan), finish_time)

        state.task_op_idx[task_idx] += 1
        if int(state.task_op_idx[task_idx]) == self.O:
            return

        next_op_idx = op_idx + 1
        state.operation_ready_time[task_idx, next_op_idx] = max(
            float(state.operation_ready_time[task_idx, next_op_idx]),
            finish_time,
        )

    def step(self, action: int):
        try:
            assignment = self.decode_assignment(action)
        except (TypeError, ValueError):
            info = {"valid": False, "success": False, "dead_end": False}
            return self.state.to_obs(), -1.0, False, False, info

        if not self.assignment_mask[
            assignment["task_idx"],
            assignment["op_idx"],
            assignment["sys_idx"],
        ]:
            info = {"valid": False, "success": False, "dead_end": False}
            return self.state.to_obs(), -1.0, False, False, info

        old_makespan = float(self.state.current_makespan)
        self.allocate_assignment(**assignment)
        self.refresh_derived_state()
        reward = -(float(self.state.current_makespan) - old_makespan) / self.state.M

        success = bool(np.all(self.state.task_op_idx == self.O))
        dead_end = not success and not np.any(self.assignment_mask)
        info = {"valid": True, "success": success, "dead_end": dead_end}
        return self.state.to_obs(), float(reward), success or dead_end, False, info
