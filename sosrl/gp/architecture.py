"""Concrete architecture actions and side-effect-free candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Any, Literal

import numpy as np

from .. import environment as env_module


ActionKind = Literal["keep", "add", "remove", "replace"]
_KIND_ORDER = {"keep": 0, "add": 1, "remove": 2, "replace": 3}


@dataclass(frozen=True)
class ArchitectureAction:
    """A concrete architecture change evaluated directly by one GP tree."""

    kind: ActionKind
    old_system: int | None = None
    new_system: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KIND_ORDER:
            raise ValueError(f"unknown architecture action kind: {self.kind}")
        old_present = self.old_system is not None
        new_present = self.new_system is not None
        expected = {
            "keep": (False, False),
            "add": (False, True),
            "remove": (True, False),
            "replace": (True, True),
        }[self.kind]
        if (old_present, new_present) != expected:
            raise ValueError(f"invalid fields for {self.kind} action.")
        if old_present and not 0 <= int(self.old_system) < env_module.N:
            raise ValueError("old_system is outside the system pool.")
        if new_present and not 0 <= int(self.new_system) < env_module.N:
            raise ValueError("new_system is outside the system pool.")
        if self.kind == "replace" and self.old_system == self.new_system:
            raise ValueError("replacement systems must be different.")

    @property
    def changed_system_count(self) -> int:
        return {"keep": 0, "add": 1, "remove": 1, "replace": 2}[self.kind]

    @property
    def tie_break_key(self) -> tuple[int, int, int, int]:
        sentinel = env_module.N
        return (
            self.changed_system_count,
            _KIND_ORDER[self.kind],
            sentinel if self.old_system is None else int(self.old_system),
            sentinel if self.new_system is None else int(self.new_system),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "kind": self.kind,
            "old_system": self.old_system,
            "new_system": self.new_system,
        }


@lru_cache(maxsize=1)
def raw_architecture_actions() -> tuple[ArchitectureAction, ...]:
    """Return the fixed 203-action universe for the current 22-system pool."""
    actions: list[ArchitectureAction] = [ArchitectureAction("keep")]
    actions.extend(ArchitectureAction("add", new_system=i) for i in range(env_module.N))
    actions.extend(ArchitectureAction("remove", old_system=i) for i in range(env_module.N))
    for old_idx in range(env_module.N):
        old_type = int(env_module.FULL_SOS[old_idx].func_type)
        for new_idx in range(env_module.N):
            if old_idx == new_idx:
                continue
            if int(env_module.FULL_SOS[new_idx].func_type) == old_type:
                actions.append(
                    ArchitectureAction(
                        "replace", old_system=old_idx, new_system=new_idx
                    )
                )
    return tuple(actions)


@lru_cache(maxsize=1)
def architecture_action_ids() -> dict[ArchitectureAction, int]:
    """Return the stable canonical ID for every raw GP action."""
    return {
        action: action_id
        for action_id, action in enumerate(raw_architecture_actions())
    }


def architecture_action_id(action: ArchitectureAction) -> int:
    return int(architecture_action_ids()[action])


@lru_cache(maxsize=1)
def architecture_action_table_hash() -> str:
    payload = json.dumps(
        [action.to_dict() for action in raw_architecture_actions()],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def effective_ready_time(mission_env: env_module.MissionEnv, sys_idx: int) -> float:
    system = env_module.FULL_SOS[int(sys_idx)]
    ready = float(mission_env.state.system_ready_time[int(sys_idx)])
    if not np.isfinite(ready):
        ready = float(system.available_from)
    return max(float(system.available_from), ready)


def remaining_window(mission_env: env_module.MissionEnv, sys_idx: int) -> float:
    system = env_module.FULL_SOS[int(sys_idx)]
    return max(float(system.available_until) - effective_ready_time(mission_env, sys_idx), 0.0)


def hypothetical_active_mask(
    mission_env: env_module.MissionEnv,
    action: ArchitectureAction,
) -> np.ndarray | None:
    """Return the action's post-mask, or None when its physical preconditions fail."""
    mask = np.asarray(mission_env.active_system_mask, dtype=bool).copy()
    if action.kind == "keep":
        return mask
    if action.kind == "add":
        new_idx = int(action.new_system)
        if mask[new_idx]:
            return None
        mask[new_idx] = True
        return mask
    if action.kind == "remove":
        old_idx = int(action.old_system)
        if not mask[old_idx]:
            return None
        mask[old_idx] = False
        return mask

    old_idx = int(action.old_system)
    new_idx = int(action.new_system)
    if not mask[old_idx] or mask[new_idx]:
        return None
    if (
        int(env_module.FULL_SOS[old_idx].func_type)
        != int(env_module.FULL_SOS[new_idx].func_type)
    ):
        return None
    mask[old_idx] = False
    mask[new_idx] = True
    return mask


def legal_architecture_actions(
    mission_env: env_module.MissionEnv,
) -> tuple[ArchitectureAction, ...]:
    """Enumerate immediately executable actions without mutating ``mission_env``."""
    if bool(np.all(mission_env.state.task_op_idx >= mission_env.O)):
        return ()
    active_mask = np.asarray(mission_env.active_system_mask, dtype=bool)
    finite_finish = np.isfinite(mission_env.current_candidate_finish_times())
    active_pair_counts = np.sum(
        finite_finish[:, active_mask], axis=1, dtype=np.int32
    )
    legal: list[ArchitectureAction] = []
    for action in raw_architecture_actions():
        mask = hypothetical_active_mask(mission_env, action)
        if mask is None:
            continue
        if action.kind in {"add", "replace"} and remaining_window(
            mission_env, int(action.new_system)
        ) <= 0.0:
            continue
        pair_counts = active_pair_counts
        if action.kind in {"remove", "replace"}:
            pair_counts = pair_counts - finite_finish[:, int(action.old_system)]
        if action.kind in {"add", "replace"}:
            pair_counts = pair_counts + finite_finish[:, int(action.new_system)]
        if np.any(pair_counts > 0):
            legal.append(action)
    return tuple(legal)


def apply_architecture_action(
    mission_env: env_module.MissionEnv,
    action: ArchitectureAction,
    *,
    expected_decision_version: int | None = None,
) -> dict[str, Any]:
    """Validate against the current decision version, then execute one action."""
    if (
        expected_decision_version is not None
        and int(expected_decision_version) != int(mission_env.decision_version)
    ):
        raise RuntimeError("stale architecture decision; regenerate candidates.")
    if action not in legal_architecture_actions(mission_env):
        raise ValueError("architecture action is not legal in the current state.")
    if action.kind == "keep":
        return {
            "valid": True,
            "kind": "keep",
            "cost_delta": 0.0,
            "refund": 0.0,
        }
    if action.kind == "add":
        return mission_env.add_system(int(action.new_system))
    if action.kind == "remove":
        return mission_env.remove_system(int(action.old_system))
    return mission_env.replace_system(
        int(action.old_system), int(action.new_system)
    )
