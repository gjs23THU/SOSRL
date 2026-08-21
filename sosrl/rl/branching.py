"""Constrained additive branching DQN primitives for assignment scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random
from typing import NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .. import domain as syn
from .. import environment as env
from .config import BranchingDQNConfig


GLOBAL_FEATURE_DIM = env.OBSERVATION_SIZE
FUNCTION_TYPE_COUNT = len(syn.func_type2idx)
TASK_FEATURE_DIM = 12 + FUNCTION_TYPE_COUNT
SYSTEM_FEATURE_DIM = 13 + FUNCTION_TYPE_COUNT
FEATURE_SCHEMA_VERSION = 1
GLOBAL_FEATURE_NAMES = (
    "active_task_ratio",
    "task_candidate_ratio",
    "task_waiting_ratio",
    "current_duration_sum",
    "current_duration_mean",
    "current_duration_min",
    "remaining_work_sum",
    "remaining_work_mean",
    "remaining_work_max",
    "next_type_load_mean",
    "next_type_load_min",
    "time_to_due_mean",
    "time_to_due_min",
    "slack_mean",
    "slack_min",
    "waiting_slack_min",
    "selected_system_ready_delay_mean",
    "task_completion_mean",
    "late_candidate_ratio",
    "negative_slack_candidate_ratio",
    "current_duration_cv",
    "remaining_work_cv",
    "time_to_due_cv",
    "slack_cv",
    "next_type_load_cv",
)
TASK_FEATURE_NAMES = (
    "unfinished",
    "operation_progress",
    "current_duration",
    "current_ready_time",
    "remaining_work",
    "time_to_due",
    "slack",
    "next_type_load",
    "earliest_feasible_start",
    "earliest_feasible_finish",
    "feasible_system_ratio",
    "due_time",
    "func_type_s",
    "func_type_d",
    "func_type_i",
)
SYSTEM_FEATURE_NAMES = (
    "active",
    "used",
    "ready_time",
    "available_from",
    "available_until",
    "remaining_window",
    "busy_time",
    "idle_time",
    "utilization",
    "feasible_task_ratio",
    "min_feasible_start",
    "min_feasible_finish",
    "mean_feasible_finish",
    "func_type_s",
    "func_type_d",
    "func_type_i",
)


@dataclass(frozen=True)
class BranchingObservation:
    """Variable-length entity observation consumed by the branching scheduler."""

    global_features: np.ndarray
    task_features: np.ndarray
    system_features: np.ndarray
    task_entity_mask: np.ndarray
    system_entity_mask: np.ndarray
    pair_mask: np.ndarray
    task_op_indices: np.ndarray
    decision_version: int

    def copy(self) -> "BranchingObservation":
        return BranchingObservation(
            global_features=np.asarray(self.global_features, dtype=np.float32).copy(),
            task_features=np.asarray(self.task_features, dtype=np.float32).copy(),
            system_features=np.asarray(self.system_features, dtype=np.float32).copy(),
            task_entity_mask=np.asarray(self.task_entity_mask, dtype=bool).copy(),
            system_entity_mask=np.asarray(self.system_entity_mask, dtype=bool).copy(),
            pair_mask=np.asarray(self.pair_mask, dtype=bool).copy(),
            task_op_indices=np.asarray(self.task_op_indices, dtype=np.int64).copy(),
            decision_version=int(self.decision_version),
        )


@dataclass(frozen=True)
class BranchingBatch:
    """Padded batch of variable-length branching observations."""

    global_features: np.ndarray
    task_features: np.ndarray
    system_features: np.ndarray
    task_entity_mask: np.ndarray
    system_entity_mask: np.ndarray
    pair_mask: np.ndarray
    task_op_indices: np.ndarray
    decision_versions: np.ndarray

    def to_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "global_features": torch.as_tensor(
                self.global_features,
                dtype=torch.float32,
                device=device,
            ),
            "task_features": torch.as_tensor(
                self.task_features,
                dtype=torch.float32,
                device=device,
            ),
            "system_features": torch.as_tensor(
                self.system_features,
                dtype=torch.float32,
                device=device,
            ),
            "task_entity_mask": torch.as_tensor(
                self.task_entity_mask,
                dtype=torch.bool,
                device=device,
            ),
            "system_entity_mask": torch.as_tensor(
                self.system_entity_mask,
                dtype=torch.bool,
                device=device,
            ),
            "pair_mask": torch.as_tensor(
                self.pair_mask,
                dtype=torch.bool,
                device=device,
            ),
        }


class BranchingQOutput(NamedTuple):
    scores: torch.Tensor
    value: torch.Tensor
    task_advantages: torch.Tensor
    system_advantages: torch.Tensor


@dataclass(frozen=True)
class BranchingAction:
    task_idx: int
    sys_idx: int
    op_idx: int
    decision_version: int


@dataclass(frozen=True)
class BranchingTransition:
    observation: BranchingObservation
    action: BranchingAction
    reward: float
    next_observation: BranchingObservation
    done: bool


def _function_one_hot(func_type: int) -> np.ndarray:
    result = np.zeros(FUNCTION_TYPE_COUNT, dtype=np.float32)
    func_type = int(func_type)
    if not 0 <= func_type < FUNCTION_TYPE_COUNT:
        raise ValueError(f"unknown function type: {func_type}")
    result[func_type] = 1.0
    return result


def _finite_time(value: float, scale: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(value) / scale


def build_branching_observation(
    mission_env: env.MissionEnv,
) -> BranchingObservation:
    """Build entity features and the current frontier task-system mask."""

    if FUNCTION_TYPE_COUNT != 3:
        raise ValueError(
            "branching feature schema version 1 requires exactly three function types."
        )

    state = mission_env.state
    scale = max(float(state.M), 1.0)
    full_mask = np.asarray(mission_env.valid_assignment_mask(), dtype=bool)
    pair_mask = np.zeros((mission_env.T, mission_env.N), dtype=bool)
    task_op_indices = np.full(mission_env.T, -1, dtype=np.int64)
    task_entity_mask = np.asarray(state.task_op_idx < mission_env.O, dtype=bool)
    system_entity_mask = np.asarray(mission_env.active_system_mask, dtype=bool).copy()

    for task_idx in range(mission_env.T):
        op_idx = int(state.task_op_idx[task_idx])
        if op_idx >= mission_env.O:
            continue
        task_op_indices[task_idx] = op_idx
        pair_mask[task_idx] = full_mask[task_idx, op_idx]

    active_system_count = max(int(np.count_nonzero(system_entity_mask)), 1)
    task_features = np.zeros(
        (mission_env.T, TASK_FEATURE_DIM),
        dtype=np.float32,
    )
    for task_idx, task in enumerate(mission_env.mission):
        op_idx = int(task_op_indices[task_idx])
        if op_idx < 0:
            continue
        operation = task.operations[op_idx]
        valid_systems = np.flatnonzero(pair_mask[task_idx])
        if valid_systems.size:
            starts = mission_env.assignment_start_time[
                task_idx,
                op_idx,
                valid_systems,
            ]
            finishes = mission_env.assignment_finish_time[
                task_idx,
                op_idx,
                valid_systems,
            ]
            earliest_start = float(np.min(starts))
            earliest_finish = float(np.min(finishes))
        else:
            earliest_start = 0.0
            earliest_finish = 0.0

        task_features[task_idx] = np.concatenate(
            [
                np.asarray(
                    [
                        1.0,
                        op_idx / max(mission_env.O, 1),
                        float(operation.duration) / scale,
                        _finite_time(
                            state.operation_ready_time[task_idx, op_idx],
                            scale,
                        ),
                        float(state.task_remaining_time[task_idx]) / scale,
                        float(state.task_ttd[task_idx]) / scale,
                        float(state.task_slack[task_idx]) / scale,
                        float(state.task_next_type_load[task_idx]) / scale,
                        earliest_start / scale,
                        earliest_finish / scale,
                        valid_systems.size / active_system_count,
                        float(state.task_due_time[task_idx]) / scale,
                    ],
                    dtype=np.float32,
                ),
                _function_one_hot(operation.func_type),
            ]
        )

    unfinished_count = max(int(np.count_nonzero(task_entity_mask)), 1)
    system_features = np.zeros(
        (mission_env.N, SYSTEM_FEATURE_DIM),
        dtype=np.float32,
    )
    for sys_idx, system in enumerate(env.FULL_SOS):
        active = bool(system_entity_mask[sys_idx])
        ready_time = float(state.system_ready_time[sys_idx])
        if not np.isfinite(ready_time):
            ready_time = float(system.available_from)
        window_anchor = max(float(system.available_from), ready_time)
        remaining_window = max(float(system.available_until) - window_anchor, 0.0)
        busy_time = float(state.system_busy_time[sys_idx])
        idle_time = float(state.system_idle_time[sys_idx])
        utilization = busy_time / max(busy_time + idle_time, 1.0)
        feasible_tasks = np.flatnonzero(pair_mask[:, sys_idx])
        if feasible_tasks.size:
            starts = np.asarray(
                [
                    mission_env.assignment_start_time[
                        task_idx,
                        int(task_op_indices[task_idx]),
                        sys_idx,
                    ]
                    for task_idx in feasible_tasks
                ],
                dtype=np.float32,
            )
            finishes = np.asarray(
                [
                    mission_env.assignment_finish_time[
                        task_idx,
                        int(task_op_indices[task_idx]),
                        sys_idx,
                    ]
                    for task_idx in feasible_tasks
                ],
                dtype=np.float32,
            )
            min_start = float(np.min(starts))
            min_finish = float(np.min(finishes))
            mean_finish = float(np.mean(finishes))
        else:
            min_start = 0.0
            min_finish = 0.0
            mean_finish = 0.0

        system_features[sys_idx] = np.concatenate(
            [
                np.asarray(
                    [
                        float(active),
                        float(mission_env.used_system_mask[sys_idx]),
                        ready_time / scale,
                        float(system.available_from) / scale,
                        float(system.available_until) / scale,
                        remaining_window / scale,
                        busy_time / scale,
                        idle_time / scale,
                        utilization,
                        feasible_tasks.size / unfinished_count,
                        min_start / scale,
                        min_finish / scale,
                        mean_finish / scale,
                    ],
                    dtype=np.float32,
                ),
                _function_one_hot(system.func_type),
            ]
        )

    return BranchingObservation(
        global_features=np.nan_to_num(
            mission_env.schedule_observation(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32),
        task_features=np.nan_to_num(
            task_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32),
        system_features=np.nan_to_num(
            system_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32),
        task_entity_mask=task_entity_mask.copy(),
        system_entity_mask=system_entity_mask,
        pair_mask=pair_mask,
        task_op_indices=task_op_indices,
        decision_version=int(mission_env.decision_version),
    )


def collate_branching_observations(
    observations: Sequence[BranchingObservation],
) -> BranchingBatch:
    if not observations:
        raise ValueError("cannot collate an empty observation sequence.")
    batch_size = len(observations)
    max_tasks = max(observation.task_features.shape[0] for observation in observations)
    max_systems = max(
        observation.system_features.shape[0] for observation in observations
    )
    global_features = np.zeros(
        (batch_size, GLOBAL_FEATURE_DIM),
        dtype=np.float32,
    )
    task_features = np.zeros(
        (batch_size, max_tasks, TASK_FEATURE_DIM),
        dtype=np.float32,
    )
    system_features = np.zeros(
        (batch_size, max_systems, SYSTEM_FEATURE_DIM),
        dtype=np.float32,
    )
    task_entity_mask = np.zeros((batch_size, max_tasks), dtype=bool)
    system_entity_mask = np.zeros((batch_size, max_systems), dtype=bool)
    pair_mask = np.zeros((batch_size, max_tasks, max_systems), dtype=bool)
    task_op_indices = np.full((batch_size, max_tasks), -1, dtype=np.int64)
    decision_versions = np.zeros(batch_size, dtype=np.int64)

    for batch_idx, observation in enumerate(observations):
        task_num = observation.task_features.shape[0]
        system_num = observation.system_features.shape[0]
        if observation.global_features.shape != (GLOBAL_FEATURE_DIM,):
            raise ValueError("global feature dimension mismatch.")
        if observation.task_features.shape[1] != TASK_FEATURE_DIM:
            raise ValueError("task feature dimension mismatch.")
        if observation.system_features.shape[1] != SYSTEM_FEATURE_DIM:
            raise ValueError("system feature dimension mismatch.")
        if observation.pair_mask.shape != (task_num, system_num):
            raise ValueError("pair mask shape mismatch.")
        global_features[batch_idx] = observation.global_features
        task_features[batch_idx, :task_num] = observation.task_features
        system_features[batch_idx, :system_num] = observation.system_features
        task_entity_mask[batch_idx, :task_num] = observation.task_entity_mask
        system_entity_mask[batch_idx, :system_num] = observation.system_entity_mask
        pair_mask[batch_idx, :task_num, :system_num] = observation.pair_mask
        task_op_indices[batch_idx, :task_num] = observation.task_op_indices
        decision_versions[batch_idx] = observation.decision_version

    return BranchingBatch(
        global_features=global_features,
        task_features=task_features,
        system_features=system_features,
        task_entity_mask=task_entity_mask,
        system_entity_mask=system_entity_mask,
        pair_mask=pair_mask,
        task_op_indices=task_op_indices,
        decision_versions=decision_versions,
    )


def masked_argmax(scores: np.ndarray, pair_mask: np.ndarray) -> tuple[int, int]:
    scores = np.asarray(scores, dtype=np.float32)
    pair_mask = np.asarray(pair_mask, dtype=bool)
    if scores.shape != pair_mask.shape or scores.ndim != 2:
        raise ValueError("scores and pair mask must be matching two-dimensional arrays.")
    if not np.any(pair_mask):
        raise ValueError("no valid task-system pair.")
    masked_scores = np.where(pair_mask, scores, -np.inf)
    flat_index = int(np.argmax(masked_scores.reshape(-1)))
    return tuple(int(value) for value in np.unravel_index(flat_index, scores.shape))


class BranchingQNetwork(nn.Module):
    """Entity encoders and additive dueling heads for variable T/N."""

    def __init__(self):
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(TASK_FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.system_encoder = nn.Sequential(
            nn.Linear(SYSTEM_FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.task_advantage_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.system_advantage_head = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _masked_pool(
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask_float = mask.unsqueeze(-1).to(embeddings.dtype)
        count = mask_float.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * mask_float).sum(dim=1) / count
        masked = embeddings.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = masked.max(dim=1).values
        has_entity = mask.any(dim=1, keepdim=True)
        maximum = torch.where(has_entity, maximum, torch.zeros_like(maximum))
        return mean, maximum

    @staticmethod
    def _center_advantages(
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.to(advantages.dtype)
        mean = (advantages * mask_float).sum(dim=1, keepdim=True) / (
            mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        )
        centered = advantages - mean
        return torch.where(mask, centered, torch.zeros_like(centered))

    def forward(
        self,
        global_features: torch.Tensor,
        task_features: torch.Tensor,
        system_features: torch.Tensor,
        task_entity_mask: torch.Tensor,
        system_entity_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> BranchingQOutput:
        task_embeddings = self.task_encoder(task_features)
        system_embeddings = self.system_encoder(system_features)
        global_embedding = self.global_encoder(global_features)
        task_mean, task_max = self._masked_pool(
            task_embeddings,
            task_entity_mask,
        )
        system_mean, system_max = self._masked_pool(
            system_embeddings,
            system_entity_mask,
        )
        context = self.context_encoder(
            torch.cat(
                [
                    global_embedding,
                    task_mean,
                    task_max,
                    system_mean,
                    system_max,
                ],
                dim=1,
            )
        )
        task_context = context.unsqueeze(1).expand(-1, task_embeddings.shape[1], -1)
        system_context = context.unsqueeze(1).expand(
            -1,
            system_embeddings.shape[1],
            -1,
        )
        raw_task_advantages = self.task_advantage_head(
            torch.cat([task_embeddings, task_context], dim=-1)
        ).squeeze(-1)
        raw_system_advantages = self.system_advantage_head(
            torch.cat([system_embeddings, system_context], dim=-1)
        ).squeeze(-1)
        valid_tasks = pair_mask.any(dim=2) & task_entity_mask
        valid_systems = pair_mask.any(dim=1) & system_entity_mask
        task_advantages = self._center_advantages(
            raw_task_advantages,
            valid_tasks,
        )
        system_advantages = self._center_advantages(
            raw_system_advantages,
            valid_systems,
        )
        value = self.value_head(context)
        scores = (
            value.unsqueeze(2)
            + task_advantages.unsqueeze(2)
            + system_advantages.unsqueeze(1)
        )
        return BranchingQOutput(
            scores=scores,
            value=value,
            task_advantages=task_advantages,
            system_advantages=system_advantages,
        )


class BranchingReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[BranchingTransition] = deque(maxlen=int(capacity))

    def add(
        self,
        observation: BranchingObservation,
        action: BranchingAction,
        reward: float,
        next_observation: BranchingObservation,
        done: bool,
    ) -> None:
        self.buffer.append(
            BranchingTransition(
                observation=observation.copy(),
                action=BranchingAction(
                    int(action.task_idx),
                    int(action.sys_idx),
                    int(action.op_idx),
                    int(action.decision_version),
                ),
                reward=float(reward),
                next_observation=next_observation.copy(),
                done=bool(done),
            )
        )

    def sample(self, batch_size: int) -> list[BranchingTransition]:
        return random.sample(self.buffer, int(batch_size))

    def __len__(self) -> int:
        return len(self.buffer)


class BranchingDQNAgent:
    """Double DQN over an additive, pair-masked task-system value function."""

    checkpoint_kind = "branching_scheduler"

    def __init__(self, config: BranchingDQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.q_net = BranchingQNetwork().to(self.device)
        self.target_net = BranchingQNetwork().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.replay = BranchingReplayBuffer(config.buffer_size)
        self.learn_step = 0

    def save_checkpoint(self, path, training_state=None):
        from .checkpoint import save_branching_checkpoint

        return save_branching_checkpoint(self, path, training_state)

    @classmethod
    def load_checkpoint(cls, path, device=None, load_optimizer=True):
        from .checkpoint import load_branching_checkpoint

        return load_branching_checkpoint(path, device, load_optimizer)

    def select_action(
        self,
        observation: BranchingObservation,
        epsilon: float,
    ) -> BranchingAction:
        pair_mask = np.asarray(observation.pair_mask, dtype=bool)
        valid_pairs = np.argwhere(pair_mask)
        if valid_pairs.size == 0:
            raise ValueError("no valid task-system pair.")
        if epsilon > 0.0 and random.random() < epsilon:
            task_idx, sys_idx = random.choice(valid_pairs.tolist())
        else:
            batch = collate_branching_observations([observation])
            with torch.no_grad():
                scores = self.q_net(**batch.to_torch(self.device)).scores[0]
                mask_tensor = torch.as_tensor(
                    pair_mask,
                    dtype=torch.bool,
                    device=self.device,
                )
                masked_scores = scores.masked_fill(~mask_tensor, -torch.inf)
                flat_index = int(masked_scores.reshape(-1).argmax().item())
            task_idx, sys_idx = np.unravel_index(flat_index, pair_mask.shape)
        op_idx = int(observation.task_op_indices[int(task_idx)])
        if op_idx < 0:
            raise RuntimeError("selected task has no frontier operation.")
        return BranchingAction(
            task_idx=int(task_idx),
            sys_idx=int(sys_idx),
            op_idx=op_idx,
            decision_version=int(observation.decision_version),
        )

    @staticmethod
    def encode_environment_action(
        mission_env: env.MissionEnv,
        action: BranchingAction,
    ) -> int:
        if int(mission_env.decision_version) != int(action.decision_version):
            raise RuntimeError("stale branching action: environment version changed.")
        current_op_idx = int(mission_env.state.task_op_idx[action.task_idx])
        if current_op_idx != int(action.op_idx):
            raise RuntimeError("stale branching action: task frontier changed.")
        if not mission_env.valid_assignment_mask()[
            action.task_idx,
            action.op_idx,
            action.sys_idx,
        ]:
            raise RuntimeError("branching action is no longer feasible.")
        return mission_env.encode_assignment(
            action.task_idx,
            action.op_idx,
            action.sys_idx,
        )

    def learn(self):
        required = max(self.config.batch_size, self.config.min_buffer_size)
        if len(self.replay) < required:
            return None
        transitions = self.replay.sample(self.config.batch_size)
        observations = collate_branching_observations(
            [transition.observation for transition in transitions]
        )
        next_observations = collate_branching_observations(
            [transition.next_observation for transition in transitions]
        )
        observation_tensors = observations.to_torch(self.device)
        next_tensors = next_observations.to_torch(self.device)
        task_actions = torch.as_tensor(
            [transition.action.task_idx for transition in transitions],
            dtype=torch.int64,
            device=self.device,
        )
        system_actions = torch.as_tensor(
            [transition.action.sys_idx for transition in transitions],
            dtype=torch.int64,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [transition.reward for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        done = torch.as_tensor(
            [transition.done for transition in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        batch_indices = torch.arange(len(transitions), device=self.device)

        current_scores = self.q_net(**observation_tensors).scores
        current_values = current_scores[
            batch_indices,
            task_actions,
            system_actions,
        ]
        with torch.no_grad():
            next_pair_mask = next_tensors["pair_mask"]
            online_next_scores = self.q_net(**next_tensors).scores
            online_next_scores = online_next_scores.masked_fill(
                ~next_pair_mask,
                -torch.inf,
            )
            has_next_action = next_pair_mask.flatten(1).any(dim=1)
            next_flat_actions = online_next_scores.flatten(1).argmax(dim=1)
            max_systems = online_next_scores.shape[2]
            next_task_actions = torch.div(
                next_flat_actions,
                max_systems,
                rounding_mode="floor",
            )
            next_system_actions = next_flat_actions % max_systems
            target_next_scores = self.target_net(**next_tensors).scores
            next_values = target_next_scores[
                batch_indices,
                next_task_actions,
                next_system_actions,
            ]
            next_values = torch.where(
                has_next_action,
                next_values,
                torch.zeros_like(next_values),
            )
            target = rewards + self.config.gamma * (1.0 - done) * next_values

        loss = nn.functional.smooth_l1_loss(current_values, target)
        if not torch.isfinite(loss):
            raise RuntimeError("branching DQN loss became non-finite.")
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())
