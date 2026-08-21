"""Versioned state-action features for direct GP architecture scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .. import domain as syn
from .. import environment as env_module
from .architecture import (
    ArchitectureAction,
    effective_ready_time,
    hypothetical_active_mask,
    remaining_window,
)


ARCH_FEATURE_SCHEMA_VERSION = 2
EPSILON = 1e-6
MISSING_FINISH_SENTINEL = 2.0

CURRENT_SYSTEM_FEATURES = (
    "progress",
    "makespan_norm",
    "active_ratio",
    "used_ratio",
    "active_cost_ratio",
    "net_cost_ratio",
    "budget_excess_ratio",
    "steps_since_change_ratio",
    "mean_active_utilization",
)
ACTION_TARGET_FEATURES = (
    "add_flag",
    "remove_flag",
    "added_cost_ratio",
    "added_ready_time_norm",
    "added_remaining_window_norm",
    "added_used_flag",
    "added_utilization",
    "removed_cost_ratio",
    "removed_ready_time_norm",
    "removed_remaining_window_norm",
    "removed_used_flag",
    "removed_utilization",
)
BASE_SYSTEM_FEATURES = CURRENT_SYSTEM_FEATURES + ACTION_TARGET_FEATURES
DEMAND_CONTEXT_FEATURES = (
    "capability_coverage_ratio",
    "feasible_pair_ratio",
    "blocked_frontier_ratio",
    "target_type_remaining_demand_norm",
    "target_type_active_capacity_norm",
    "target_type_pressure",
    "target_type_blocked_ratio",
    "target_type_active_ratio",
)
COUNTERFACTUAL_DELTA_FEATURES = (
    "delta_net_cost_ratio",
    "budget_excess_after_ratio",
    "delta_capability_coverage",
    "delta_feasible_pair_ratio",
    "delta_blocked_frontier_ratio",
    "delta_target_type_capacity_norm",
    "best_frontier_finish_after_norm",
    "delta_best_frontier_finish_norm",
    "mean_frontier_finish_after_norm",
    "delta_mean_frontier_finish_norm",
)
SYSTEM_DELTA_FEATURES = (
    BASE_SYSTEM_FEATURES + DEMAND_CONTEXT_FEATURES + COUNTERFACTUAL_DELTA_FEATURES
)
_SYSTEM_DELTA_INDEX = {
    name: index for index, name in enumerate(SYSTEM_DELTA_FEATURES)
}


def _op_context_names() -> tuple[str, ...]:
    from ..rl.branching import GLOBAL_FEATURE_NAMES

    return tuple(f"op_{name}" for name in GLOBAL_FEATURE_NAMES)


FEATURE_PRESETS = {
    "system": BASE_SYSTEM_FEATURES,
    "system_demand": BASE_SYSTEM_FEATURES + DEMAND_CONTEXT_FEATURES,
    "system_delta": SYSTEM_DELTA_FEATURES,
    "op_context": _op_context_names() + ACTION_TARGET_FEATURES,
}

_FLAG_FEATURES = frozenset(
    {"add_flag", "remove_flag", "added_used_flag", "removed_used_flag"}
)


@dataclass(frozen=True)
class _MaskMetrics:
    coverage: float
    feasible_pair_ratio: float
    blocked_frontier_ratio: float
    target_demand: float
    target_capacity: float
    target_pressure: float
    target_blocked_ratio: float
    target_active_ratio: float
    best_finish_norm: float
    mean_finish_norm: float


@dataclass
class ArchitectureFeatureContext:
    """Decision-local arrays shared by all concrete action candidates."""

    decision_version: int
    scale: float
    budget: float
    finish_times: np.ndarray
    finite_finish: np.ndarray
    unfinished: np.ndarray
    unfinished_count: int
    demand: np.ndarray
    system_types: np.ndarray
    remaining_windows: np.ndarray
    frontier_types: np.ndarray
    pool_counts: np.ndarray
    active_pair_counts: np.ndarray
    system_finish_counts: np.ndarray
    system_finish_sums: np.ndarray
    system_finish_minima: np.ndarray
    active_capacity: np.ndarray
    active_counts_by_type: np.ndarray
    current_metrics: dict[int | None, _MaskMetrics]


def _clip(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    if value == math.inf:
        return 10.0
    if value == -math.inf:
        return -10.0
    return min(max(value, -10.0), 10.0)


def _system_utilization(mission_env: env_module.MissionEnv, sys_idx: int) -> float:
    busy = float(mission_env.state.system_busy_time[int(sys_idx)])
    idle = float(mission_env.state.system_idle_time[int(sys_idx)])
    return busy / max(busy + idle, 1.0)


def _target_type(action: ArchitectureAction) -> int | None:
    if action.kind in {"add", "replace"}:
        return int(env_module.FULL_SOS[int(action.new_system)].func_type)
    if action.kind == "remove":
        return int(env_module.FULL_SOS[int(action.old_system)].func_type)
    return None


def build_architecture_feature_context(
    mission_env: env_module.MissionEnv,
) -> ArchitectureFeatureContext:
    function_count = len(syn.func_type2idx)
    system_types = np.asarray(
        [int(system.func_type) for system in env_module.FULL_SOS],
        dtype=np.int32,
    )
    remaining_windows = np.asarray(
        [remaining_window(mission_env, index) for index in range(mission_env.N)],
        dtype=np.float64,
    )
    frontier_types = np.full(mission_env.T, -1, dtype=np.int32)
    for task_idx, task in enumerate(mission_env.mission):
        op_idx = int(mission_env.state.task_op_idx[task_idx])
        if op_idx < mission_env.O:
            frontier_types[task_idx] = int(task.operations[op_idx].func_type)
    finish_times = mission_env.current_candidate_finish_times()
    finite_finish = np.isfinite(finish_times)
    active_mask = np.asarray(mission_env.active_system_mask, dtype=bool)
    finite_values = np.where(finite_finish, finish_times, 0.0)
    system_finish_counts = np.sum(finite_finish, axis=0, dtype=np.int32)
    system_finish_minima = np.min(
        np.where(finite_finish, finish_times, np.inf), axis=0
    )
    return ArchitectureFeatureContext(
        decision_version=int(mission_env.decision_version),
        scale=max(float(mission_env.state.M), 1.0),
        budget=max(float(mission_env.budget), 1.0),
        finish_times=finish_times,
        finite_finish=finite_finish,
        unfinished=np.asarray(
            mission_env.state.task_op_idx < mission_env.O, dtype=bool
        ),
        unfinished_count=int(
            np.count_nonzero(mission_env.state.task_op_idx < mission_env.O)
        ),
        demand=np.asarray(
            mission_env.remaining_demand_by_type(), dtype=np.float64
        ),
        system_types=system_types,
        remaining_windows=remaining_windows,
        frontier_types=frontier_types,
        pool_counts=np.bincount(
            system_types, minlength=function_count
        ).astype(np.int32),
        active_pair_counts=np.sum(
            finite_finish[:, active_mask], axis=1, dtype=np.int32
        ),
        system_finish_counts=system_finish_counts,
        system_finish_sums=np.sum(
            finite_values, axis=0, dtype=np.float32
        ),
        system_finish_minima=system_finish_minima,
        active_capacity=np.bincount(
            system_types[active_mask],
            weights=remaining_windows[active_mask],
            minlength=function_count,
        ),
        active_counts_by_type=np.bincount(
            system_types[active_mask], minlength=function_count
        ).astype(np.int32),
        current_metrics={},
    )


def _mask_metrics(
    mission_env: env_module.MissionEnv,
    active_mask: np.ndarray,
    target_type: int | None,
    context: ArchitectureFeatureContext,
) -> _MaskMetrics:
    if int(mission_env.decision_version) != context.decision_version:
        raise RuntimeError("stale GP feature context.")
    finish_times = context.finish_times
    current_mask = np.asarray(mission_env.active_system_mask, dtype=bool)
    added = np.flatnonzero(active_mask & ~current_mask)
    removed = np.flatnonzero(current_mask & ~active_mask)
    pair_counts = context.active_pair_counts.copy()
    for sys_idx in added:
        pair_counts += context.finite_finish[:, int(sys_idx)]
    for sys_idx in removed:
        pair_counts -= context.finite_finish[:, int(sys_idx)]
    unfinished = context.unfinished
    unfinished_count = context.unfinished_count
    feasible_count = int(np.sum(pair_counts[unfinished], dtype=np.int64))
    feasible_pair_ratio = feasible_count / max(unfinished_count * mission_env.N, 1)
    blocked = unfinished & (pair_counts == 0)
    blocked_ratio = int(np.count_nonzero(blocked)) / max(unfinished_count, 1)

    demand = context.demand
    capacity = context.active_capacity.copy()
    for sys_idx in added:
        capacity[context.system_types[int(sys_idx)]] += context.remaining_windows[
            int(sys_idx)
        ]
    for sys_idx in removed:
        capacity[context.system_types[int(sys_idx)]] -= context.remaining_windows[
            int(sys_idx)
        ]
    demanded_types = demand > 0.0
    demanded_count = int(np.count_nonzero(demanded_types))
    coverage = (
        int(np.count_nonzero((capacity > 0.0) & demanded_types))
        / max(demanded_count, 1)
        if demanded_count
        else 1.0
    )

    active_systems = np.flatnonzero(active_mask)
    finite_count = int(
        np.sum(context.system_finish_counts[active_systems], dtype=np.int64)
    )
    scale = context.scale
    if finite_count:
        best_finish = float(
            np.min(context.system_finish_minima[active_systems])
        ) / scale
        finish_sum = float(
            np.sum(
                context.system_finish_sums[active_systems], dtype=np.float32
            )
        )
        mean_finish = float(np.float32(finish_sum / finite_count)) / scale
    else:
        best_finish = MISSING_FINISH_SENTINEL
        mean_finish = MISSING_FINISH_SENTINEL

    if target_type is None:
        return _MaskMetrics(
            coverage=coverage,
            feasible_pair_ratio=feasible_pair_ratio,
            blocked_frontier_ratio=blocked_ratio,
            target_demand=0.0,
            target_capacity=0.0,
            target_pressure=0.0,
            target_blocked_ratio=0.0,
            target_active_ratio=0.0,
            best_finish_norm=best_finish,
            mean_finish_norm=mean_finish,
        )

    type_frontier = context.frontier_types == int(target_type)
    target_frontier_count = int(np.count_nonzero(type_frontier))
    target_blocked = type_frontier & (pair_counts == 0)
    pool_count = int(context.pool_counts[target_type])
    active_count = int(context.active_counts_by_type[target_type])
    active_count += int(
        np.count_nonzero(context.system_types[added] == int(target_type))
    )
    active_count -= int(
        np.count_nonzero(context.system_types[removed] == int(target_type))
    )
    return _MaskMetrics(
        coverage=coverage,
        feasible_pair_ratio=feasible_pair_ratio,
        blocked_frontier_ratio=blocked_ratio,
        target_demand=float(demand[target_type]),
        target_capacity=float(capacity[target_type]),
        target_pressure=float(demand[target_type]) / max(float(capacity[target_type]), EPSILON),
        target_blocked_ratio=int(np.count_nonzero(target_blocked))
        / max(target_frontier_count, 1),
        target_active_ratio=active_count / max(pool_count, 1),
        best_finish_norm=best_finish,
        mean_finish_norm=mean_finish,
    )


def _post_net_cost(
    mission_env: env_module.MissionEnv, action: ArchitectureAction
) -> float:
    value = float(mission_env.net_cost)
    if action.kind in {"remove", "replace"}:
        old_cost = float(env_module.FULL_SOS[int(action.old_system)].cost)
        value = max(0.0, value - mission_env.refund_rate * old_cost)
    if action.kind in {"add", "replace"}:
        value += float(env_module.FULL_SOS[int(action.new_system)].cost)
    return value


def _target_action_features(
    mission_env: env_module.MissionEnv, action: ArchitectureAction
) -> dict[str, float]:
    scale = max(float(mission_env.state.M), 1.0)
    budget = max(float(mission_env.budget), 1.0)
    values = {name: 0.0 for name in ACTION_TARGET_FEATURES}
    if action.kind in {"add", "replace"}:
        sys_idx = int(action.new_system)
        system = env_module.FULL_SOS[sys_idx]
        values.update(
            {
                "add_flag": 1.0,
                "added_cost_ratio": float(system.cost) / budget,
                "added_ready_time_norm": effective_ready_time(mission_env, sys_idx) / scale,
                "added_remaining_window_norm": remaining_window(mission_env, sys_idx) / scale,
                "added_used_flag": float(mission_env.used_system_mask[sys_idx]),
                "added_utilization": _system_utilization(mission_env, sys_idx),
            }
        )
    if action.kind in {"remove", "replace"}:
        sys_idx = int(action.old_system)
        system = env_module.FULL_SOS[sys_idx]
        values.update(
            {
                "remove_flag": 1.0,
                "removed_cost_ratio": float(system.cost) / budget,
                "removed_ready_time_norm": effective_ready_time(mission_env, sys_idx) / scale,
                "removed_remaining_window_norm": remaining_window(mission_env, sys_idx) / scale,
                "removed_used_flag": float(mission_env.used_system_mask[sys_idx]),
                "removed_utilization": _system_utilization(mission_env, sys_idx),
            }
        )
    return values


def extract_architecture_features(
    mission_env: env_module.MissionEnv,
    action: ArchitectureAction,
    *,
    context: ArchitectureFeatureContext | None = None,
) -> dict[str, float]:
    """Compute all 39 version-2 features without changing environment state."""
    post_mask = hypothetical_active_mask(mission_env, action)
    if post_mask is None:
        raise ValueError("action physical preconditions are not satisfied.")
    context = context or build_architecture_feature_context(mission_env)
    if int(mission_env.decision_version) != context.decision_version:
        raise RuntimeError("stale GP feature context.")
    scale = context.scale
    budget = context.budget
    total_operations = max(mission_env.T * mission_env.O, 1)
    target_type = _target_type(action)
    current_metrics = context.current_metrics.get(target_type)
    if current_metrics is None:
        current_metrics = _mask_metrics(
            mission_env,
            mission_env.active_system_mask,
            target_type,
            context,
        )
        context.current_metrics[target_type] = current_metrics
    post_metrics = (
        current_metrics
        if np.array_equal(post_mask, mission_env.active_system_mask)
        else _mask_metrics(
            mission_env,
            post_mask,
            target_type,
            context,
        )
    )
    active_indices = np.flatnonzero(mission_env.active_system_mask)
    mean_utilization = (
        float(np.mean([_system_utilization(mission_env, int(i)) for i in active_indices]))
        if active_indices.size
        else 0.0
    )
    post_cost = _post_net_cost(mission_env, action)

    values: dict[str, float] = {
        "progress": float(np.sum(mission_env.state.task_op_idx)) / total_operations,
        "makespan_norm": float(mission_env.state.current_makespan) / scale,
        "active_ratio": float(np.mean(mission_env.active_system_mask)),
        "used_ratio": float(np.mean(mission_env.used_system_mask)),
        "active_cost_ratio": float(mission_env.active_cost) / budget,
        "net_cost_ratio": float(mission_env.net_cost) / budget,
        "budget_excess_ratio": max(float(mission_env.net_cost) - budget, 0.0) / budget,
        "steps_since_change_ratio": float(mission_env.steps_since_change) / total_operations,
        "mean_active_utilization": mean_utilization,
        **_target_action_features(mission_env, action),
        "capability_coverage_ratio": current_metrics.coverage,
        "feasible_pair_ratio": current_metrics.feasible_pair_ratio,
        "blocked_frontier_ratio": current_metrics.blocked_frontier_ratio,
        "target_type_remaining_demand_norm": current_metrics.target_demand / scale,
        "target_type_active_capacity_norm": current_metrics.target_capacity / scale,
        "target_type_pressure": current_metrics.target_pressure,
        "target_type_blocked_ratio": current_metrics.target_blocked_ratio,
        "target_type_active_ratio": current_metrics.target_active_ratio,
        "delta_net_cost_ratio": (post_cost - float(mission_env.net_cost)) / budget,
        "budget_excess_after_ratio": max(post_cost - budget, 0.0) / budget,
        "delta_capability_coverage": post_metrics.coverage - current_metrics.coverage,
        "delta_feasible_pair_ratio": post_metrics.feasible_pair_ratio - current_metrics.feasible_pair_ratio,
        "delta_blocked_frontier_ratio": post_metrics.blocked_frontier_ratio - current_metrics.blocked_frontier_ratio,
        "delta_target_type_capacity_norm": (post_metrics.target_capacity - current_metrics.target_capacity) / scale,
        "best_frontier_finish_after_norm": post_metrics.best_finish_norm,
        "delta_best_frontier_finish_norm": post_metrics.best_finish_norm - current_metrics.best_finish_norm,
        "mean_frontier_finish_after_norm": post_metrics.mean_finish_norm,
        "delta_mean_frontier_finish_norm": post_metrics.mean_finish_norm - current_metrics.mean_finish_norm,
    }
    normalized: dict[str, float] = {}
    for name in SYSTEM_DELTA_FEATURES:
        value = float(values[name])
        normalized[name] = min(max(value, 0.0), 1.0) if name in _FLAG_FEATURES else _clip(value)
    return normalized


def feature_names_for_preset(preset: str) -> tuple[str, ...]:
    try:
        return tuple(FEATURE_PRESETS[preset])
    except KeyError as exc:
        raise ValueError(f"unknown GP feature preset: {preset}") from exc


def architecture_feature_vector(
    mission_env: env_module.MissionEnv,
    action: ArchitectureAction,
    preset: str = "system_delta",
    *,
    context: ArchitectureFeatureContext | None = None,
) -> np.ndarray:
    names = feature_names_for_preset(preset)
    action_features = extract_architecture_features(
        mission_env, action, context=context
    )
    if preset == "op_context":
        schedule = np.asarray(mission_env.schedule_observation(), dtype=np.float64)
        op_values = {
            name: _clip(float(value))
            for name, value in zip(_op_context_names(), schedule, strict=True)
        }
        action_features = {**op_values, **action_features}
    vector = np.asarray([action_features[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise RuntimeError("non-finite GP architecture feature generated.")
    return vector


def architecture_feature_matrix(
    mission_env: env_module.MissionEnv,
    actions: tuple[ArchitectureAction, ...] | list[ArchitectureAction],
    preset: str = "system_delta",
    *,
    context: ArchitectureFeatureContext | None = None,
) -> np.ndarray:
    """Compute one decision's candidate matrix while sharing state-only work."""
    names = feature_names_for_preset(preset)
    if not actions:
        return np.empty((0, len(names)), dtype=np.float64)
    context = context or build_architecture_feature_context(mission_env)
    if int(mission_env.decision_version) != context.decision_version:
        raise RuntimeError("stale GP feature context.")

    row_count = len(actions)
    matrix = np.zeros((row_count, len(SYSTEM_DELTA_FEATURES)), dtype=np.float64)
    column = _SYSTEM_DELTA_INDEX
    scale = context.scale
    budget = context.budget
    total_operations = max(mission_env.T * mission_env.O, 1)
    active_mask = np.asarray(mission_env.active_system_mask, dtype=bool)
    active_indices = np.flatnonzero(active_mask)
    mean_utilization = (
        float(
            np.mean(
                [
                    _system_utilization(mission_env, int(index))
                    for index in active_indices
                ]
            )
        )
        if active_indices.size
        else 0.0
    )
    current_values = {
        "progress": float(np.sum(mission_env.state.task_op_idx)) / total_operations,
        "makespan_norm": float(mission_env.state.current_makespan) / scale,
        "active_ratio": float(np.mean(active_mask)),
        "used_ratio": float(np.mean(mission_env.used_system_mask)),
        "active_cost_ratio": float(mission_env.active_cost) / budget,
        "net_cost_ratio": float(mission_env.net_cost) / budget,
        "budget_excess_ratio": max(float(mission_env.net_cost) - budget, 0.0)
        / budget,
        "steps_since_change_ratio": float(mission_env.steps_since_change)
        / total_operations,
        "mean_active_utilization": mean_utilization,
    }
    for name, value in current_values.items():
        matrix[:, column[name]] = value

    post_masks = np.empty((row_count, mission_env.N), dtype=bool)
    target_types = np.full(row_count, -1, dtype=np.int32)
    post_costs = np.empty(row_count, dtype=np.float64)
    for row_index, action in enumerate(actions):
        post_mask = hypothetical_active_mask(mission_env, action)
        if post_mask is None:
            raise ValueError("action physical preconditions are not satisfied.")
        post_masks[row_index] = post_mask
        target_type = _target_type(action)
        if target_type is not None:
            target_types[row_index] = int(target_type)

        if action.kind in {"add", "replace"}:
            sys_idx = int(action.new_system)
            system = env_module.FULL_SOS[sys_idx]
            matrix[row_index, column["add_flag"]] = 1.0
            matrix[row_index, column["added_cost_ratio"]] = (
                float(system.cost) / budget
            )
            matrix[row_index, column["added_ready_time_norm"]] = (
                effective_ready_time(mission_env, sys_idx) / scale
            )
            matrix[row_index, column["added_remaining_window_norm"]] = (
                context.remaining_windows[sys_idx] / scale
            )
            matrix[row_index, column["added_used_flag"]] = float(
                mission_env.used_system_mask[sys_idx]
            )
            matrix[row_index, column["added_utilization"]] = _system_utilization(
                mission_env, sys_idx
            )
        if action.kind in {"remove", "replace"}:
            sys_idx = int(action.old_system)
            system = env_module.FULL_SOS[sys_idx]
            matrix[row_index, column["remove_flag"]] = 1.0
            matrix[row_index, column["removed_cost_ratio"]] = (
                float(system.cost) / budget
            )
            matrix[row_index, column["removed_ready_time_norm"]] = (
                effective_ready_time(mission_env, sys_idx) / scale
            )
            matrix[row_index, column["removed_remaining_window_norm"]] = (
                context.remaining_windows[sys_idx] / scale
            )
            matrix[row_index, column["removed_used_flag"]] = float(
                mission_env.used_system_mask[sys_idx]
            )
            matrix[row_index, column["removed_utilization"]] = _system_utilization(
                mission_env, sys_idx
            )

        post_costs[row_index] = _post_net_cost(mission_env, action)

    finite_finish = context.finite_finish.astype(np.int16, copy=False)
    pair_counts = finite_finish @ post_masks.astype(np.int16).T
    unfinished_pair_counts = pair_counts[context.unfinished]
    feasible_counts = np.sum(unfinished_pair_counts, axis=0, dtype=np.int64)
    post_feasible_ratio = feasible_counts / max(
        context.unfinished_count * mission_env.N, 1
    )
    blocked = context.unfinished[:, np.newaxis] & (pair_counts == 0)
    post_blocked_ratio = np.sum(blocked, axis=0, dtype=np.int64) / max(
        context.unfinished_count, 1
    )

    function_count = len(syn.func_type2idx)
    capacity_basis = np.zeros(
        (mission_env.N, function_count), dtype=np.float64
    )
    capacity_basis[
        np.arange(mission_env.N), context.system_types
    ] = context.remaining_windows
    post_capacity = post_masks.astype(np.float64) @ capacity_basis
    demanded_types = context.demand > 0.0
    demanded_count = int(np.count_nonzero(demanded_types))
    post_coverage = (
        np.count_nonzero(
            (post_capacity > 0.0) & demanded_types[np.newaxis, :], axis=1
        )
        / max(demanded_count, 1)
        if demanded_count
        else np.ones(row_count, dtype=np.float64)
    )

    post_finish_counts = (
        post_masks.astype(np.int32) @ context.system_finish_counts
    )
    post_mean_finish = np.full(
        row_count, MISSING_FINISH_SENTINEL, dtype=np.float64
    )
    has_finish = post_finish_counts > 0
    post_best_finish = np.full(
        row_count, MISSING_FINISH_SENTINEL, dtype=np.float64
    )
    for row_index in np.flatnonzero(has_finish):
        active_systems = np.flatnonzero(post_masks[int(row_index)])
        post_best_finish[row_index] = float(
            np.min(context.system_finish_minima[active_systems])
        ) / scale
        finish_sum = float(
            np.sum(
                context.system_finish_sums[active_systems], dtype=np.float32
            )
        )
        post_mean_finish[row_index] = float(
            np.float32(finish_sum / post_finish_counts[row_index])
        ) / scale

    current_feasible_ratio = float(
        np.sum(
            context.active_pair_counts[context.unfinished], dtype=np.int64
        )
    ) / max(context.unfinished_count * mission_env.N, 1)
    current_blocked = context.unfinished & (context.active_pair_counts == 0)
    current_blocked_ratio = int(np.count_nonzero(current_blocked)) / max(
        context.unfinished_count, 1
    )
    current_coverage = (
        int(
            np.count_nonzero(
                (context.active_capacity > 0.0) & demanded_types
            )
        )
        / max(demanded_count, 1)
        if demanded_count
        else 1.0
    )
    active_finish_count = int(
        np.sum(context.system_finish_counts[active_indices], dtype=np.int64)
    )
    if active_finish_count:
        current_best_finish = float(
            np.min(context.system_finish_minima[active_indices])
        ) / scale
        current_finish_sum = float(
            np.sum(
                context.system_finish_sums[active_indices], dtype=np.float32
            )
        )
        current_mean_finish = float(
            np.float32(current_finish_sum / active_finish_count)
        ) / scale
    else:
        current_best_finish = MISSING_FINISH_SENTINEL
        current_mean_finish = MISSING_FINISH_SENTINEL

    target_demand = np.zeros(row_count, dtype=np.float64)
    current_target_capacity = np.zeros(row_count, dtype=np.float64)
    current_target_pressure = np.zeros(row_count, dtype=np.float64)
    current_target_blocked_ratio = np.zeros(row_count, dtype=np.float64)
    current_target_active_ratio = np.zeros(row_count, dtype=np.float64)
    post_target_capacity = np.zeros(row_count, dtype=np.float64)
    target_valid = target_types >= 0
    valid_rows = np.flatnonzero(target_valid)
    if valid_rows.size:
        valid_types = target_types[valid_rows]
        target_demand[valid_rows] = context.demand[valid_types]
        current_target_capacity[valid_rows] = context.active_capacity[valid_types]
        current_target_pressure[valid_rows] = target_demand[valid_rows] / np.maximum(
            current_target_capacity[valid_rows], EPSILON
        )
        post_target_capacity[valid_rows] = post_capacity[
            valid_rows, valid_types
        ]
        current_target_active_ratio[valid_rows] = (
            context.active_counts_by_type[valid_types]
            / np.maximum(context.pool_counts[valid_types], 1)
        )
        frontier_match = (
            context.frontier_types[:, np.newaxis]
            == target_types[np.newaxis, :]
        ) & target_valid[np.newaxis, :]
        frontier_counts = np.sum(frontier_match, axis=0, dtype=np.int64)
        current_target_blocked_ratio[valid_rows] = (
            np.sum(
                current_blocked[:, np.newaxis] & frontier_match,
                axis=0,
                dtype=np.int64,
            )[valid_rows]
            / np.maximum(frontier_counts[valid_rows], 1)
        )

    metric_columns = {
        "capability_coverage_ratio": np.full(row_count, current_coverage),
        "feasible_pair_ratio": np.full(row_count, current_feasible_ratio),
        "blocked_frontier_ratio": np.full(row_count, current_blocked_ratio),
        "target_type_remaining_demand_norm": target_demand / scale,
        "target_type_active_capacity_norm": current_target_capacity / scale,
        "target_type_pressure": current_target_pressure,
        "target_type_blocked_ratio": current_target_blocked_ratio,
        "target_type_active_ratio": current_target_active_ratio,
        "delta_net_cost_ratio": (
            post_costs - float(mission_env.net_cost)
        )
        / budget,
        "budget_excess_after_ratio": np.maximum(post_costs - budget, 0.0)
        / budget,
        "delta_capability_coverage": post_coverage - current_coverage,
        "delta_feasible_pair_ratio": post_feasible_ratio
        - current_feasible_ratio,
        "delta_blocked_frontier_ratio": post_blocked_ratio
        - current_blocked_ratio,
        "delta_target_type_capacity_norm": (
            post_target_capacity - current_target_capacity
        )
        / scale,
        "best_frontier_finish_after_norm": post_best_finish,
        "delta_best_frontier_finish_norm": post_best_finish
        - current_best_finish,
        "mean_frontier_finish_after_norm": post_mean_finish,
        "delta_mean_frontier_finish_norm": post_mean_finish
        - current_mean_finish,
    }
    for name, values in metric_columns.items():
        matrix[:, column[name]] = values

    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(matrix, -10.0, 10.0, out=matrix)
    for name in _FLAG_FEATURES:
        np.clip(matrix[:, column[name]], 0.0, 1.0, out=matrix[:, column[name]])

    if preset == "system_delta":
        return matrix
    if preset in {"system", "system_demand"}:
        indices = [column[name] for name in names]
        return matrix[:, indices]
    schedule = np.asarray(mission_env.schedule_observation(), dtype=np.float64)
    np.nan_to_num(schedule, copy=False, nan=0.0, posinf=10.0, neginf=-10.0)
    np.clip(schedule, -10.0, 10.0, out=schedule)
    action_indices = [column[name] for name in ACTION_TARGET_FEATURES]
    return np.concatenate(
        [np.broadcast_to(schedule, (row_count, schedule.size)), matrix[:, action_indices]],
        axis=1,
    )
