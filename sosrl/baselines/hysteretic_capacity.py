"""S/s-inspired hysteretic capability-management architecture baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .. import domain as syn
from .. import environment as env_module
from ..gp.architecture import (
    ArchitectureAction,
    apply_architecture_action,
    hypothetical_active_mask,
    legal_architecture_actions,
    remaining_window,
)
from ..gp.provider import ArchitectureDecision


SS_HCM_POLICY_VERSION = "ss_hcm_v1"
_EPSILON = 1e-9


@dataclass(frozen=True)
class HystereticCapacityConfig:
    """Immutable expert settings for the S/s-HCM baseline."""

    lower_threshold: float = 0.40
    upper_threshold: float = 0.90
    budget_mode: str = "soft"
    replace_mode: str = "hierarchical"
    policy_version: str = SS_HCM_POLICY_VERSION

    def __post_init__(self) -> None:
        lower = float(self.lower_threshold)
        upper = float(self.upper_threshold)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("S/s-HCM thresholds must be finite.")
        if lower < 0.0 or lower >= upper:
            raise ValueError("S/s-HCM requires 0 <= lower_threshold < upper_threshold.")
        if self.budget_mode != "soft":
            raise ValueError("S/s-HCM currently supports only the shared soft budget mode.")
        if self.replace_mode != "hierarchical":
            raise ValueError("S/s-HCM replacement must use hierarchical mode.")
        if self.policy_version != SS_HCM_POLICY_VERSION:
            raise ValueError("unsupported S/s-HCM policy version.")

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class _ActionMetrics:
    action: ArchitectureAction
    post_pressure: float
    capacity_delta: float
    net_cost_delta: float
    frontier_count: int
    earliest_frontier_finish: float


def _remaining_capacity_by_type(
    mission_env: env_module.MissionEnv,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = (
        np.asarray(mission_env.active_system_mask, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if mask.shape != (mission_env.N,):
        raise ValueError("active_mask has the wrong shape.")
    capacity = np.zeros(len(syn.func_type2idx), dtype=np.float64)
    for sys_idx in np.flatnonzero(mask):
        func_type = int(env_module.FULL_SOS[int(sys_idx)].func_type)
        capacity[func_type] += remaining_window(mission_env, int(sys_idx))
    return capacity


def _pressure(demand: float, capacity: float) -> float:
    if float(demand) <= 0.0:
        return 0.0
    if float(capacity) <= 0.0:
        return float("inf")
    return float(demand) / float(capacity)


def capability_pressures(mission_env: env_module.MissionEnv) -> np.ndarray:
    """Return remaining demand divided by active remaining window by capability."""

    demand = np.asarray(mission_env.remaining_demand_by_type(), dtype=np.float64)
    capacity = _remaining_capacity_by_type(mission_env)
    return np.asarray(
        [_pressure(demand[index], capacity[index]) for index in range(len(demand))],
        dtype=np.float64,
    )


def _post_net_cost(
    mission_env: env_module.MissionEnv,
    action: ArchitectureAction,
) -> float:
    value = float(mission_env.net_cost)
    if action.kind in {"remove", "replace"}:
        old_cost = float(env_module.FULL_SOS[int(action.old_system)].cost)
        value = max(0.0, value - float(mission_env.refund_rate) * old_cost)
    if action.kind in {"add", "replace"}:
        value += float(env_module.FULL_SOS[int(action.new_system)].cost)
    return value


def _target_type(action: ArchitectureAction) -> int | None:
    if action.kind in {"add", "replace"}:
        return int(env_module.FULL_SOS[int(action.new_system)].func_type)
    if action.kind == "remove":
        return int(env_module.FULL_SOS[int(action.old_system)].func_type)
    return None


def _frontier_metrics(
    mission_env: env_module.MissionEnv,
    active_mask: np.ndarray,
) -> tuple[int, float]:
    finish_times = mission_env.current_candidate_finish_times()
    masked = np.where(np.asarray(active_mask, dtype=bool)[None, :], finish_times, np.inf)
    best = np.min(masked, axis=1)
    feasible = np.isfinite(best)
    return (
        int(np.count_nonzero(feasible)),
        float(np.min(best[feasible])) if np.any(feasible) else float("inf"),
    )


class HystereticCapacityProvider:
    """Deterministic ADD/DEL-first capability pressure controller."""

    provider_id = "ss"
    policy_version = SS_HCM_POLICY_VERSION

    def __init__(self, config: HystereticCapacityConfig | None = None):
        self.config = config or HystereticCapacityConfig()

    def _metrics(
        self,
        mission_env: env_module.MissionEnv,
        action: ArchitectureAction,
        *,
        demand: np.ndarray,
        current_capacity: np.ndarray,
    ) -> _ActionMetrics:
        post_mask = hypothetical_active_mask(mission_env, action)
        if post_mask is None:
            raise ValueError("S/s-HCM received an action with failed physical preconditions.")
        post_capacity = _remaining_capacity_by_type(mission_env, post_mask)
        target = _target_type(action)
        if target is None:
            post_pressure = 0.0
            capacity_delta = 0.0
        else:
            post_pressure = _pressure(demand[target], post_capacity[target])
            capacity_delta = float(post_capacity[target] - current_capacity[target])
        frontier_count, earliest = _frontier_metrics(mission_env, post_mask)
        return _ActionMetrics(
            action=action,
            post_pressure=float(post_pressure),
            capacity_delta=capacity_delta,
            net_cost_delta=float(_post_net_cost(mission_env, action) - mission_env.net_cost),
            frontier_count=frontier_count,
            earliest_frontier_finish=earliest,
        )

    @staticmethod
    def _invalid(
        mission_env: env_module.MissionEnv,
        candidate_count: int,
    ) -> ArchitectureDecision:
        return ArchitectureDecision(
            action=ArchitectureAction("keep"),
            score=float("inf"),
            candidate_count=int(candidate_count),
            changed=False,
            valid=False,
            diagnostics={
                "trigger": "emergency",
                "target_capability": -1,
                "pre_pressure": float("inf"),
                "post_pressure": float("inf"),
                "capacity_delta": 0.0,
                "net_cost_delta": 0.0,
                "replace_reason": "no_rescue",
                "decision_version": int(mission_env.decision_version),
            },
        )

    def _decision(
        self,
        *,
        mission_env: env_module.MissionEnv,
        action_metrics: _ActionMetrics | None,
        candidate_count: int,
        trigger: str,
        target_capability: int,
        pre_pressure: float,
        replace_reason: str,
    ) -> ArchitectureDecision:
        action = (
            ArchitectureAction("keep")
            if action_metrics is None
            else action_metrics.action
        )
        post_pressure = (
            float(pre_pressure)
            if action_metrics is None
            else float(action_metrics.post_pressure)
        )
        return ArchitectureDecision(
            action=action,
            score=post_pressure,
            candidate_count=int(candidate_count),
            changed=action.kind != "keep",
            valid=True,
            diagnostics={
                "trigger": trigger,
                "target_capability": int(target_capability),
                "pre_pressure": float(pre_pressure),
                "post_pressure": post_pressure,
                "capacity_delta": (
                    0.0 if action_metrics is None else float(action_metrics.capacity_delta)
                ),
                "net_cost_delta": (
                    0.0 if action_metrics is None else float(action_metrics.net_cost_delta)
                ),
                "replace_reason": replace_reason,
                "decision_version": int(mission_env.decision_version),
            },
        )

    def _emergency_decision(
        self,
        mission_env: env_module.MissionEnv,
        legal: tuple[ArchitectureAction, ...],
        *,
        demand: np.ndarray,
        capacity: np.ndarray,
        pressures: np.ndarray,
    ) -> ArchitectureDecision:
        metrics = [
            self._metrics(
                mission_env,
                action,
                demand=demand,
                current_capacity=capacity,
            )
            for action in legal
            if action.kind in {"add", "replace"}
        ]

        def emergency_key(item: _ActionMetrics):
            return (
                -item.frontier_count,
                item.earliest_frontier_finish,
                item.net_cost_delta,
                item.action.tie_break_key,
            )

        additions = [item for item in metrics if item.action.kind == "add"]
        replacements = [item for item in metrics if item.action.kind == "replace"]
        best_add = min(additions, key=emergency_key) if additions else None
        best_replace = min(replacements, key=emergency_key) if replacements else None
        selected = best_add
        reason = "primary_add"
        if best_add is None:
            selected = best_replace
            reason = "no_add"
        elif best_replace is not None and (
            best_replace.frontier_count >= best_add.frontier_count
            and best_replace.earliest_frontier_finish
            <= best_add.earliest_frontier_finish + _EPSILON
            and best_replace.net_cost_delta < best_add.net_cost_delta - _EPSILON
        ):
            selected = best_replace
            reason = "dominates_add"
        if selected is None:
            return self._invalid(mission_env, len(legal))
        target = int(_target_type(selected.action))
        return self._decision(
            mission_env=mission_env,
            action_metrics=selected,
            candidate_count=len(legal),
            trigger="emergency",
            target_capability=target,
            pre_pressure=float(pressures[target]),
            replace_reason=reason,
        )

    def _best_expansion(
        self,
        candidates: list[_ActionMetrics],
    ) -> _ActionMetrics | None:
        if not candidates:
            return None

        def expansion_key(item: _ActionMetrics):
            crosses = item.post_pressure <= self.config.upper_threshold + _EPSILON
            unit_cost = item.net_cost_delta / max(item.capacity_delta, _EPSILON)
            pressure_term = (
                self.config.upper_threshold - item.post_pressure
                if crosses
                else item.post_pressure
            )
            return (
                0 if crosses else 1,
                unit_cost,
                pressure_term,
                item.action.tie_break_key,
            )

        return min(candidates, key=expansion_key)

    def _expansion_decision(
        self,
        mission_env: env_module.MissionEnv,
        legal: tuple[ArchitectureAction, ...],
        *,
        target: int,
        demand: np.ndarray,
        capacity: np.ndarray,
        pressures: np.ndarray,
    ) -> ArchitectureDecision:
        metrics = []
        for action in legal:
            if action.kind not in {"add", "replace"} or _target_type(action) != target:
                continue
            item = self._metrics(
                mission_env,
                action,
                demand=demand,
                current_capacity=capacity,
            )
            if item.capacity_delta > _EPSILON:
                metrics.append(item)
        best_add = self._best_expansion(
            [item for item in metrics if item.action.kind == "add"]
        )
        best_replace = self._best_expansion(
            [item for item in metrics if item.action.kind == "replace"]
        )
        selected = best_add
        reason = "primary_add"
        if best_add is None:
            selected = best_replace
            reason = "no_add" if best_replace is not None else "no_feasible_expansion"
        elif best_replace is not None:
            pre_pressure = float(pressures[target])
            add_relief = (
                best_add.capacity_delta
                if not math.isfinite(pre_pressure)
                else pre_pressure - best_add.post_pressure
            )
            replace_relief = (
                best_replace.capacity_delta
                if not math.isfinite(pre_pressure)
                else pre_pressure - best_replace.post_pressure
            )
            if (
                replace_relief >= add_relief - _EPSILON
                and best_replace.net_cost_delta < best_add.net_cost_delta - _EPSILON
            ):
                selected = best_replace
                reason = "dominates_add"
        return self._decision(
            mission_env=mission_env,
            action_metrics=selected,
            candidate_count=len(legal),
            trigger="expand",
            target_capability=target,
            pre_pressure=float(pressures[target]),
            replace_reason=reason,
        )

    def _contraction_for_type(
        self,
        mission_env: env_module.MissionEnv,
        legal: tuple[ArchitectureAction, ...],
        *,
        target: int,
        demand: np.ndarray,
        capacity: np.ndarray,
    ) -> tuple[_ActionMetrics | None, str]:
        removals = []
        for action in legal:
            if action.kind != "remove" or _target_type(action) != target:
                continue
            item = self._metrics(
                mission_env,
                action,
                demand=demand,
                current_capacity=capacity,
            )
            if item.post_pressure <= self.config.upper_threshold + _EPSILON:
                removals.append(item)
        if removals:
            selected = min(
                removals,
                key=lambda item: (
                    item.net_cost_delta,
                    -item.capacity_delta,
                    bool(mission_env.used_system_mask[int(item.action.old_system)]),
                    item.action.tie_break_key,
                ),
            )
            return selected, "safe_delete"

        replacements = []
        for action in legal:
            if action.kind != "replace" or _target_type(action) != target:
                continue
            item = self._metrics(
                mission_env,
                action,
                demand=demand,
                current_capacity=capacity,
            )
            if (
                item.capacity_delta < -_EPSILON
                and item.post_pressure <= self.config.upper_threshold + _EPSILON
                and item.net_cost_delta < -_EPSILON
            ):
                replacements.append(item)
        if replacements:
            return (
                min(
                    replacements,
                    key=lambda item: (
                        item.net_cost_delta,
                        -item.capacity_delta,
                        item.action.tie_break_key,
                    ),
                ),
                "delete_unsafe",
            )
        return None, "no_safe_contraction"

    def decide(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        version = int(mission_env.decision_version)
        legal = legal_architecture_actions(mission_env)
        if not legal:
            return self._invalid(mission_env, 0)
        demand = np.asarray(mission_env.remaining_demand_by_type(), dtype=np.float64)
        capacity = _remaining_capacity_by_type(mission_env)
        pressures = np.asarray(
            [_pressure(demand[index], capacity[index]) for index in range(len(demand))],
            dtype=np.float64,
        )
        if not mission_env.has_feasible_assignment(mission_env.active_system_mask):
            return self._emergency_decision(
                mission_env,
                legal,
                demand=demand,
                capacity=capacity,
                pressures=pressures,
            )

        overloaded = [
            index
            for index, pressure in enumerate(pressures)
            if pressure >= self.config.upper_threshold - _EPSILON
        ]
        if overloaded:
            target = max(
                overloaded,
                key=lambda index: (pressures[index], demand[index], -index),
            )
            return self._expansion_decision(
                mission_env,
                legal,
                target=int(target),
                demand=demand,
                capacity=capacity,
                pressures=pressures,
            )

        low_types = sorted(
            (
                index
                for index, pressure in enumerate(pressures)
                if pressure <= self.config.lower_threshold + _EPSILON
                and capacity[index] > _EPSILON
            ),
            key=lambda index: (pressures[index], -demand[index], index),
        )
        for target in low_types:
            selected, reason = self._contraction_for_type(
                mission_env,
                legal,
                target=int(target),
                demand=demand,
                capacity=capacity,
            )
            if selected is not None:
                return self._decision(
                    mission_env=mission_env,
                    action_metrics=selected,
                    candidate_count=len(legal),
                    trigger="contract",
                    target_capability=int(target),
                    pre_pressure=float(pressures[target]),
                    replace_reason=reason,
                )

        if low_types:
            target = int(low_types[0])
            return self._decision(
                mission_env=mission_env,
                action_metrics=None,
                candidate_count=len(legal),
                trigger="contract",
                target_capability=target,
                pre_pressure=float(pressures[target]),
                replace_reason="no_safe_contraction",
            )

        max_pressure_type = int(np.argmax(pressures))
        return self._decision(
            mission_env=mission_env,
            action_metrics=None,
            candidate_count=len(legal),
            trigger="band_keep",
            target_capability=-1,
            pre_pressure=float(pressures[max_pressure_type]),
            replace_reason="not_applicable",
        )

    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        decision = self.decide(mission_env)
        if not decision.valid:
            return decision
        before_cost = float(mission_env.net_cost)
        result = apply_architecture_action(
            mission_env,
            decision.action,
            expected_decision_version=int(decision.diagnostics["decision_version"]),
        )
        diagnostics: dict[str, Any] = dict(decision.diagnostics)
        diagnostics["net_cost_delta"] = float(mission_env.net_cost - before_cost)
        return ArchitectureDecision(
            action=decision.action,
            score=decision.score,
            candidate_count=decision.candidate_count,
            changed=decision.changed,
            valid=bool(result.get("valid", False)),
            diagnostics=diagnostics,
        )
