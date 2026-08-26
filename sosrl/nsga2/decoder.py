"""GP-action-aligned Baldwinian decoder for dynamic schedules."""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from .. import domain as syn
from .. import environment as env
from ..gp.architecture import (
    ArchitectureAction,
    apply_architecture_action,
    architecture_action_id,
    hypothetical_active_mask,
    legal_architecture_actions,
    raw_architecture_actions,
)
from ..objectives import (
    DEFAULT_ARCHITECTURE_CHANGE_WEIGHT,
    DEFAULT_PEAK_BUDGET_PENALTY,
    gp_cost_breakdown,
)
from .model import Chromosome, DecodeResult, ProblemLayout, phenotype_digest


class DynamicScheduleDecoder:
    """Decode OS/MS/AA using the same concrete actions as direct GP."""

    def __init__(
        self,
        architecture: Sequence[syn.ComponentSystem],
        mission: Sequence[syn.Task],
        *,
        budget: float = 8000.0,
        refund_rate: float = 0.8,
        architecture_change_weight: float = DEFAULT_ARCHITECTURE_CHANGE_WEIGHT,
        peak_budget_penalty: float = DEFAULT_PEAK_BUDGET_PENALTY,
    ) -> None:
        self.architecture = tuple(architecture)
        self.mission = list(mission)
        self.layout = ProblemLayout.from_mission(self.mission)
        if len(set(self.layout.operation_counts)) != 1:
            raise ValueError("MissionEnv currently requires equal operation counts.")
        self.budget = float(budget)
        self.refund_rate = float(refund_rate)
        self.architecture_change_weight = float(architecture_change_weight)
        self.peak_budget_penalty = float(peak_budget_penalty)
        if self.architecture_change_weight < 0.0:
            raise ValueError("architecture_change_weight cannot be negative.")
        if self.peak_budget_penalty < 0.0:
            raise ValueError("peak_budget_penalty cannot be negative.")
        self.actions = raw_architecture_actions()

    def _new_environment(self) -> env.MissionEnv:
        return env.MissionEnv(
            self.architecture,
            self.mission,
            adaptive=True,
            budget=self.budget,
            refund_rate=self.refund_rate,
        )

    @staticmethod
    def _has_global_candidate(mission_env: env.MissionEnv, task_idx: int) -> bool:
        op_idx = int(mission_env.state.task_op_idx[int(task_idx)])
        if op_idx >= mission_env.O:
            return False
        return bool(
            np.any(
                np.isfinite(
                    mission_env.current_candidate_finish_times()[int(task_idx)]
                )
            )
        )

    @staticmethod
    def _active_feasible_systems(
        mission_env: env.MissionEnv,
        task_idx: int,
        action: ArchitectureAction,
    ) -> np.ndarray:
        post_mask = hypothetical_active_mask(mission_env, action)
        if post_mask is None:
            return np.empty(0, dtype=np.int32)
        finishes = mission_env.current_candidate_finish_times()[int(task_idx)]
        return np.flatnonzero(post_mask & np.isfinite(finishes)).astype(np.int32)

    def _operation_legal_actions(
        self,
        mission_env: env.MissionEnv,
        task_idx: int,
        candidate_actions: Sequence[ArchitectureAction] | None = None,
    ) -> tuple[ArchitectureAction, ...]:
        actions = (
            legal_architecture_actions(mission_env)
            if candidate_actions is None
            else candidate_actions
        )
        return tuple(
            action
            for action in actions
            if self._active_feasible_systems(mission_env, task_idx, action).size
        )

    def _select_post_action_system(
        self,
        mission_env: env.MissionEnv,
        task_idx: int,
        preferred_system: int,
        action: ArchitectureAction,
    ) -> tuple[int | None, bool]:
        feasible = self._active_feasible_systems(mission_env, task_idx, action)
        if feasible.size == 0:
            return None, False
        preferred_system = int(preferred_system)
        if preferred_system in feasible:
            return preferred_system, False
        finishes = mission_env.current_candidate_finish_times()[int(task_idx)]
        selected = min(
            (int(value) for value in feasible),
            key=lambda sys_idx: (
                float(finishes[sys_idx]),
                float(env.FULL_SOS[sys_idx].cost),
                sys_idx,
            ),
        )
        return selected, True

    def _cost_breakdown(
        self,
        mission_env: env.MissionEnv,
        *,
        net_cost: float | None = None,
        peak_net_cost: float | None = None,
        architecture_changes: int | None = None,
    ):
        return gp_cost_breakdown(
            final_net_cost=(
                mission_env.net_cost if net_cost is None else float(net_cost)
            ),
            peak_net_cost=(
                mission_env.peak_net_cost
                if peak_net_cost is None
                else float(peak_net_cost)
            ),
            budget=mission_env.budget,
            architecture_changes=(
                mission_env.architecture_change_count
                if architecture_changes is None
                else int(architecture_changes)
            ),
            architecture_change_weight=self.architecture_change_weight,
            peak_budget_penalty_weight=self.peak_budget_penalty,
        )

    def _post_action_cost(
        self,
        mission_env: env.MissionEnv,
        action: ArchitectureAction,
    ):
        net_cost = float(mission_env.net_cost)
        if action.kind in {"remove", "replace"}:
            old_cost = float(env.FULL_SOS[int(action.old_system)].cost)
            net_cost = max(0.0, net_cost - self.refund_rate * old_cost)
        if action.kind in {"add", "replace"}:
            net_cost += float(env.FULL_SOS[int(action.new_system)].cost)
        peak_cost = max(float(mission_env.peak_net_cost), net_cost)
        changes = int(mission_env.architecture_change_count) + int(
            action.kind != "keep"
        )
        return self._cost_breakdown(
            mission_env,
            net_cost=net_cost,
            peak_net_cost=peak_cost,
            architecture_changes=changes,
        )

    def _effective_cost_delta(
        self,
        mission_env: env.MissionEnv,
        action: ArchitectureAction,
    ) -> float:
        before = self._cost_breakdown(mission_env)
        after = self._post_action_cost(mission_env, action)
        return float(after.effective_cost - before.effective_cost)

    def _resolve_action_and_system(
        self,
        mission_env: env.MissionEnv,
        task_idx: int,
        preferred_system: int,
        requested_action_id: int,
    ) -> tuple[ArchitectureAction | None, int | None, bool, bool]:
        legal = self._operation_legal_actions(mission_env, task_idx)
        if not legal:
            return None, None, False, False

        requested = self.actions[int(requested_action_id)]
        if requested in legal:
            system, repaired_ms = self._select_post_action_system(
                mission_env,
                task_idx,
                preferred_system,
                requested,
            )
            return requested, system, False, repaired_ms

        keep = self.actions[0]
        if keep in legal:
            system, repaired_ms = self._select_post_action_system(
                mission_env,
                task_idx,
                preferred_system,
                keep,
            )
            return keep, system, True, repaired_ms

        finishes = mission_env.current_candidate_finish_times()[int(task_idx)]
        candidates = []
        for action in legal:
            system, repaired_ms = self._select_post_action_system(
                mission_env,
                task_idx,
                preferred_system,
                action,
            )
            if system is None:
                continue
            candidates.append(
                (
                    int(system != int(preferred_system)),
                    float(finishes[system]),
                    self._effective_cost_delta(mission_env, action),
                    action.tie_break_key,
                    int(system),
                    action,
                    bool(repaired_ms),
                )
            )
        if not candidates:
            return None, None, False, False
        selected = min(candidates, key=lambda item: item[:5])
        return selected[5], selected[4], True, selected[6]

    def decode(self, chromosome: Chromosome) -> DecodeResult:
        chromosome, repairs = self.layout.repair(chromosome)
        mission_env = self._new_environment()
        os_work = chromosome.os.astype(np.int32, copy=True).tolist()
        effective_os: list[int] = []
        effective_ms = np.full(self.layout.operation_count, -1, dtype=np.int32)
        effective_aa = np.zeros(self.layout.operation_count, dtype=np.int32)
        architecture_trace: list[dict[str, Any]] = []
        schedule: list[dict[str, Any]] = []
        action_counts: Counter[str] = Counter()
        dead_end = False

        for position in range(self.layout.operation_count):
            selected_position = None
            seen_tasks: set[int] = set()
            for candidate_position in range(position, self.layout.operation_count):
                task_idx = int(os_work[candidate_position])
                if task_idx in seen_tasks:
                    continue
                seen_tasks.add(task_idx)
                if self._has_global_candidate(mission_env, task_idx):
                    selected_position = candidate_position
                    break
            if selected_position is None:
                dead_end = True
                break
            if selected_position != position:
                task_idx = os_work.pop(selected_position)
                os_work.insert(position, task_idx)
                repairs["os_repair_count"] += 1

            task_idx = int(os_work[position])
            op_idx = int(mission_env.state.task_op_idx[task_idx])
            canonical_idx = self.layout.operation_index(task_idx, op_idx)
            preferred_system = int(chromosome.ms[canonical_idx])
            requested_action_id = int(chromosome.aa[canonical_idx])
            requested_action = self.actions[requested_action_id]
            action, sys_idx, repaired_aa, repaired_ms = (
                self._resolve_action_and_system(
                    mission_env,
                    task_idx,
                    preferred_system,
                    requested_action_id,
                )
            )
            if action is None or sys_idx is None:
                dead_end = True
                break
            repairs["aa_repair_count"] += int(repaired_aa)
            repairs["ms_repair_count"] += int(repaired_ms)

            active_before = [
                int(value) for value in np.flatnonzero(mission_env.active_system_mask)
            ]
            before_net = float(mission_env.net_cost)
            before_active = float(mission_env.active_cost)
            action_result = apply_architecture_action(mission_env, action)
            if not action_result.get("valid", False):
                raise RuntimeError("GP-aligned decoder selected an invalid action.")
            action_kind = str(action_result["kind"])
            action_counts[action_kind] += 1
            effective_action_id = architecture_action_id(action)
            active_after = [
                int(value) for value in np.flatnonzero(mission_env.active_system_mask)
            ]

            env_action = mission_env.encode_assignment(task_idx, op_idx, sys_idx)
            _, _, terminated, _, info = mission_env.step(env_action)
            if not info.get("valid", False):
                raise RuntimeError("post-action system became infeasible.")
            start_time = float(mission_env.state.op_start_time[task_idx, op_idx])
            finish_time = float(mission_env.state.op_finish_time[task_idx, op_idx])
            effective_os.append(task_idx)
            effective_ms[canonical_idx] = int(sys_idx)
            effective_aa[canonical_idx] = int(effective_action_id)
            schedule.append(
                {
                    "step": int(position),
                    "task_idx": task_idx,
                    "task_name": self.mission[task_idx].name,
                    "op_idx": op_idx,
                    "op_name": self.mission[task_idx].operations[op_idx].name,
                    "canonical_op_idx": canonical_idx,
                    "func_type": int(
                        self.mission[task_idx].operations[op_idx].func_type
                    ),
                    "preferred_sys_idx": preferred_system,
                    "sys_idx": int(sys_idx),
                    "sys_name": env.FULL_SOS[int(sys_idx)].name,
                    "start_time": start_time,
                    "finish_time": finish_time,
                    "duration": finish_time - start_time,
                }
            )
            architecture_trace.append(
                {
                    "step": int(position),
                    "task_idx": task_idx,
                    "op_idx": op_idx,
                    "canonical_op_idx": canonical_idx,
                    "requested_action_id": requested_action_id,
                    "requested_kind": requested_action.kind,
                    "requested_old_system": requested_action.old_system,
                    "requested_new_system": requested_action.new_system,
                    "effective_action_id": int(effective_action_id),
                    "effective_kind": action.kind,
                    "effective_old_system": action.old_system,
                    "effective_new_system": action.new_system,
                    "aa_repaired": bool(repaired_aa),
                    "requested_system": preferred_system,
                    "actual_system": int(sys_idx),
                    "kind": action_kind,
                    "old_system": action.old_system,
                    "new_system": action.new_system,
                    "cost_delta": float(action_result.get("cost_delta", 0.0)),
                    "refund": float(action_result.get("refund", 0.0)),
                    "net_cost_before": before_net,
                    "net_cost_after": float(mission_env.net_cost),
                    "active_cost_before": before_active,
                    "active_cost_after": float(mission_env.active_cost),
                    "active_systems_before": active_before,
                    "active_systems_after": active_after,
                }
            )
            if terminated:
                dead_end = bool(info.get("dead_end", False))
                break

        completed = int(mission_env.state.task_op_idx.sum())
        total = int(self.layout.operation_count)
        success = completed == total
        remaining_duration = float(
            sum(
                operation.duration
                for task_idx, task in enumerate(self.mission)
                for operation in task.operations[
                    int(mission_env.state.task_op_idx[task_idx]):
                ]
            )
        )
        makespan = float(mission_env.state.current_makespan)
        objective_makespan = makespan if success else makespan + remaining_duration
        constraint_violation = (total - completed) / max(total, 1)
        cost_metrics = mission_env.cost_metrics()
        cost = self._cost_breakdown(mission_env)
        metrics: dict[str, Any] = {
            **cost_metrics,
            "architecture_changes": int(mission_env.architecture_change_count),
            "keep_count": int(action_counts["keep"]),
            "add_count": int(action_counts["add"]),
            "remove_count": int(action_counts["remove"]),
            "replace_count": int(action_counts["replace"]),
            "budget": float(mission_env.budget),
            "refund_rate": float(mission_env.refund_rate),
            "architecture_change_weight": self.architecture_change_weight,
            "peak_budget_penalty_weight": self.peak_budget_penalty,
            "peak_budget_excess_ratio": cost.peak_budget_excess_ratio,
        }
        effective_ms_tuple = tuple(int(value) for value in effective_ms)
        effective_aa_tuple = tuple(int(value) for value in effective_aa)
        phenotype_hash = phenotype_digest(
            effective_os,
            effective_ms_tuple,
            effective_aa_tuple,
            success,
        )
        return DecodeResult(
            chromosome=chromosome,
            chromosome_hash=chromosome.digest,
            phenotype_hash=phenotype_hash,
            success=bool(success),
            dead_end=bool(dead_end and not success),
            completed_operations=completed,
            total_operations=total,
            makespan=makespan,
            final_net_cost=float(mission_env.net_cost),
            effective_cost=cost.effective_cost,
            gp_cost_score=cost.gp_cost_score,
            architecture_change_penalty=cost.architecture_change_penalty,
            peak_budget_penalty=cost.peak_budget_penalty,
            objective_makespan=float(objective_makespan),
            constraint_violation=float(constraint_violation),
            schedule=tuple(schedule),
            architecture_trace=tuple(architecture_trace),
            metrics=metrics,
            repair_counts={key: int(value) for key, value in repairs.items()},
            effective_os=tuple(effective_os),
            effective_ms=effective_ms_tuple,
            effective_aa=effective_aa_tuple,
        )
