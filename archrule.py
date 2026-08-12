from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import env
import syn


@dataclass(frozen=True)
class ArchitectureResolution:
    action: int
    kind: str
    old_system: int | None = None
    new_system: int | None = None
    score: float = 0.0


class ArchitectureRule:
    """Resolve six abstract architecture actions deterministically."""

    RULE_NAMES = (
        "KEEP",
        "ADD_CAPABILITY",
        "ADD_CAPACITY",
        "ADD_WINDOW",
        "REMOVE_REDUNDANT",
        "REPLACE_INEFFICIENT",
    )
    RULE_NUM = len(RULE_NAMES)

    def __init__(self, mission_env: env.MissionEnv):
        if not mission_env.adaptive:
            raise ValueError("ArchitectureRule requires MissionEnv(adaptive=True).")
        self.mission_env = mission_env
        self._cache_version: int | None = None
        self._resolution_cache: dict[
            int,
            ArchitectureResolution | None,
        ] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def action_mask(self) -> np.ndarray:
        return np.asarray(
            [self.resolve(action) is not None for action in range(self.RULE_NUM)],
            dtype=np.float32,
        )

    def resolve(self, action: int) -> ArchitectureResolution | None:
        action = int(action)
        version = int(self.mission_env.decision_version)
        if version != self._cache_version:
            self._cache_version = version
            self._resolution_cache.clear()
        if action in self._resolution_cache:
            self.cache_hits += 1
            return self._resolution_cache[action]
        self.cache_misses += 1
        resolution = self._resolve_uncached(action)
        self._resolution_cache[action] = resolution
        return resolution

    def _resolve_uncached(self, action: int) -> ArchitectureResolution | None:
        if action == 0:
            return self._keep()
        if action == 1:
            return self._add_capability()
        if action == 2:
            return self._add_capacity()
        if action == 3:
            return self._add_window()
        if action == 4:
            return self._remove_redundant()
        if action == 5:
            return self._replace_inefficient()
        return None

    def apply(self, action: int) -> dict[str, Any]:
        resolution = self.resolve(action)
        if resolution is None:
            return {
                "valid": False,
                "action": int(action),
                "rule_name": self._rule_name(action),
                "cost_delta": 0.0,
                "changed": False,
            }

        if resolution.kind == "keep":
            result: dict[str, Any] = {
                "valid": True,
                "kind": "keep",
                "cost_delta": 0.0,
                "refund": 0.0,
            }
        elif resolution.kind == "add":
            result = self.mission_env.add_system(resolution.new_system)
        elif resolution.kind == "remove":
            result = self.mission_env.remove_system(resolution.old_system)
        else:
            result = self.mission_env.replace_system(
                resolution.old_system,
                resolution.new_system,
            )
        result.update(
            {
                "action": int(action),
                "rule_name": self.RULE_NAMES[action],
                "changed": resolution.kind != "keep",
                "score": float(resolution.score),
            }
        )
        return result

    def _rule_name(self, action: int) -> str:
        if 0 <= int(action) < self.RULE_NUM:
            return self.RULE_NAMES[int(action)]
        return "UNKNOWN"

    def _keep(self) -> ArchitectureResolution | None:
        if np.any(self.mission_env.valid_assignment_mask()):
            return ArchitectureResolution(0, "keep")
        return None

    def _ready_operations(self):
        mission_env = self.mission_env
        for task_idx, task in enumerate(mission_env.mission):
            op_idx = int(mission_env.state.task_op_idx[task_idx])
            if op_idx < mission_env.O:
                yield task_idx, op_idx, task.operations[op_idx]

    def _candidate_finish(
        self,
        task_idx: int,
        op_idx: int,
        sys_idx: int,
    ) -> float | None:
        mission_env = self.mission_env
        system = env.FULL_SOS[sys_idx]
        operation = mission_env.mission[task_idx].operations[op_idx]
        if int(system.func_type) != int(operation.func_type):
            return None
        ready = float(mission_env.state.system_ready_time[sys_idx])
        if not np.isfinite(ready):
            ready = float(system.available_from)
        start = max(
            ready,
            float(mission_env.state.operation_ready_time[task_idx, op_idx]),
        )
        finish = start + float(operation.duration)
        if finish > float(system.available_until):
            return None
        return finish

    def _would_enable_assignment(self, active_mask: np.ndarray) -> bool:
        return bool(np.any(self.mission_env.hypothetical_assignment_mask(active_mask)))

    def _add_capability(self) -> ArchitectureResolution | None:
        mission_env = self.mission_env
        candidates = []
        active_types = {
            int(env.FULL_SOS[index].func_type)
            for index in np.flatnonzero(mission_env.active_system_mask)
        }
        for task_idx, op_idx, operation in self._ready_operations():
            func_type = int(operation.func_type)
            if func_type in active_types:
                continue
            for sys_idx, system in enumerate(env.FULL_SOS):
                if mission_env.active_system_mask[sys_idx]:
                    continue
                finish = self._candidate_finish(task_idx, op_idx, sys_idx)
                if finish is None:
                    continue
                candidates.append(
                    (finish, float(system.cost), sys_idx)
                )
        if not candidates:
            return None
        finish, _, sys_idx = min(candidates)
        mask = mission_env.active_system_mask.copy()
        mask[sys_idx] = True
        if not self._would_enable_assignment(mask):
            return None
        return ArchitectureResolution(1, "add", new_system=sys_idx, score=-finish)

    def _remaining_capacity(self, sys_idx: int) -> float:
        mission_env = self.mission_env
        system = env.FULL_SOS[sys_idx]
        ready = float(mission_env.state.system_ready_time[sys_idx])
        if not np.isfinite(ready):
            ready = float(system.available_from)
        return max(float(system.available_until) - max(ready, system.available_from), 0.0)

    def _add_capacity(self) -> ArchitectureResolution | None:
        mission_env = self.mission_env
        demand = mission_env.remaining_demand_by_type()
        type_candidates = []
        for func_type in sorted(syn.func_type2idx.values()):
            active = [
                index
                for index in np.flatnonzero(mission_env.active_system_mask)
                if int(env.FULL_SOS[int(index)].func_type) == int(func_type)
            ]
            inactive = [
                index
                for index, system in enumerate(env.FULL_SOS)
                if not mission_env.active_system_mask[index]
                and int(system.func_type) == int(func_type)
                and self._remaining_capacity(index) > 0
            ]
            if not active or not inactive or demand[int(func_type)] <= 0:
                continue
            capacity = sum(self._remaining_capacity(int(index)) for index in active)
            pressure = float(demand[int(func_type)]) / max(capacity, 1.0)
            best_system = min(
                inactive,
                key=lambda index: (
                    -self._remaining_capacity(index),
                    float(env.FULL_SOS[index].cost),
                    index,
                ),
            )
            type_candidates.append((-pressure, best_system))
        if not type_candidates:
            return None
        negative_pressure, sys_idx = min(type_candidates)
        mask = mission_env.active_system_mask.copy()
        mask[sys_idx] = True
        if not np.any(mission_env.valid_assignment_mask()) and not self._would_enable_assignment(mask):
            return None
        return ArchitectureResolution(
            2,
            "add",
            new_system=sys_idx,
            score=-negative_pressure,
        )

    def _add_window(self) -> ArchitectureResolution | None:
        mission_env = self.mission_env
        candidates = []
        for task_idx, op_idx, operation in self._ready_operations():
            if np.any(mission_env.assignment_mask[task_idx, op_idx]):
                continue
            func_type = int(operation.func_type)
            if not any(
                int(env.FULL_SOS[int(index)].func_type) == func_type
                for index in np.flatnonzero(mission_env.active_system_mask)
            ):
                continue
            for sys_idx, system in enumerate(env.FULL_SOS):
                if mission_env.active_system_mask[sys_idx]:
                    continue
                finish = self._candidate_finish(task_idx, op_idx, sys_idx)
                if finish is not None:
                    candidates.append((finish, float(system.cost), sys_idx))
        if not candidates:
            return None
        finish, _, sys_idx = min(candidates)
        mask = mission_env.active_system_mask.copy()
        mask[sys_idx] = True
        if not self._would_enable_assignment(mask):
            return None
        return ArchitectureResolution(3, "add", new_system=sys_idx, score=-finish)

    def _covers_remaining_demand(self, active_mask: np.ndarray) -> bool:
        mission_env = self.mission_env
        demand = mission_env.remaining_demand_by_type()
        for func_type in sorted(syn.func_type2idx.values()):
            if demand[int(func_type)] <= 0:
                continue
            systems = [
                index
                for index in np.flatnonzero(active_mask)
                if int(env.FULL_SOS[int(index)].func_type) == int(func_type)
            ]
            if not systems:
                return False
            if sum(self._remaining_capacity(int(index)) for index in systems) < float(
                demand[int(func_type)]
            ):
                return False
        return True

    def _remove_redundant(self) -> ArchitectureResolution | None:
        mission_env = self.mission_env
        candidates = []
        for sys_idx in np.flatnonzero(mission_env.active_system_mask):
            sys_idx = int(sys_idx)
            mask = mission_env.active_system_mask.copy()
            mask[sys_idx] = False
            if not self._covers_remaining_demand(mask):
                continue
            if not self._would_enable_assignment(mask):
                continue
            refund = mission_env.refund_rate * float(env.FULL_SOS[sys_idx].cost)
            candidates.append((-refund, bool(mission_env.used_system_mask[sys_idx]), sys_idx))
        if not candidates:
            return None
        negative_refund, _, sys_idx = min(candidates)
        return ArchitectureResolution(
            4,
            "remove",
            old_system=sys_idx,
            score=-negative_refund,
        )

    def _best_ready_finish(self, active_mask: np.ndarray) -> float:
        mission_env = self.mission_env
        assignment_mask = mission_env.hypothetical_assignment_mask(active_mask)
        finishes = []
        for task_idx, op_idx, _ in self._ready_operations():
            systems = np.flatnonzero(assignment_mask[task_idx, op_idx])
            if not systems.size:
                continue
            finishes.append(
                min(
                    self._candidate_finish(task_idx, op_idx, int(sys_idx))
                    for sys_idx in systems
                )
            )
        return float(np.mean(finishes)) if finishes else float("inf")

    def _replace_inefficient(self) -> ArchitectureResolution | None:
        mission_env = self.mission_env
        current_finish = self._best_ready_finish(mission_env.active_system_mask)
        candidates = []
        for old_idx in np.flatnonzero(mission_env.active_system_mask):
            old_idx = int(old_idx)
            old_system = env.FULL_SOS[old_idx]
            for new_idx, new_system in enumerate(env.FULL_SOS):
                if mission_env.active_system_mask[new_idx]:
                    continue
                if int(new_system.func_type) != int(old_system.func_type):
                    continue
                mask = mission_env.active_system_mask.copy()
                mask[old_idx] = False
                mask[new_idx] = True
                if not self._would_enable_assignment(mask):
                    continue
                new_finish = self._best_ready_finish(mask)
                cost_delta = float(new_system.cost) - (
                    mission_env.refund_rate * float(old_system.cost)
                )
                time_gain = (
                    0.0
                    if not np.isfinite(current_finish)
                    else (current_finish - new_finish) / mission_env.state.M
                )
                value = time_gain - cost_delta / mission_env.budget
                if value > 1e-9:
                    candidates.append((-value, cost_delta, old_idx, new_idx))
        if not candidates:
            return None
        negative_value, _, old_idx, new_idx = min(candidates)
        return ArchitectureResolution(
            5,
            "replace",
            old_system=old_idx,
            new_system=new_idx,
            score=-negative_value,
        )
