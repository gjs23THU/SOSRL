"""Unified architecture providers used before every lower-level BDQN decision."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .. import environment as env_module
from ..rules.architecture import ArchitectureRule
from .architecture import (
    ArchitectureAction,
    apply_architecture_action,
    legal_architecture_actions,
)
from .features import (
    architecture_feature_matrix,
    build_architecture_feature_context,
)


ACTION_KIND_NAMES = ("keep", "add", "remove", "replace")


@dataclass(frozen=True)
class ArchitectureDecision:
    action: ArchitectureAction
    score: float
    candidate_count: int
    changed: bool
    valid: bool
    diagnostics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class ArchitectureProvider(Protocol):
    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        ...


def _freeze_networks(policy) -> None:
    for network_name in ("q_net", "target_net"):
        network = getattr(policy, network_name, None)
        if network is None:
            continue
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


def _invalid_decision(candidate_count: int = 0) -> ArchitectureDecision:
    return ArchitectureDecision(
        action=ArchitectureAction("keep"),
        score=float("inf"),
        candidate_count=int(candidate_count),
        changed=False,
        valid=False,
    )


class GPArchitectureProvider:
    def __init__(
        self,
        score_function: Callable[..., float],
        *,
        feature_preset: str = "system_delta",
    ):
        self.score_function = score_function
        self.feature_preset = feature_preset

    @classmethod
    def from_artifact(cls, policy: Any | str | Path):
        if isinstance(policy, (str, Path)):
            from .artifact import load_gp_policy

            loaded = load_gp_policy(policy)
        else:
            loaded = policy
        return cls(
            loaded.score_function,
            feature_preset=loaded.artifact.feature_preset,
        )

    def decide(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        """Select a concrete action without changing the environment."""
        version = int(mission_env.decision_version)
        candidates = legal_architecture_actions(mission_env)
        if not candidates:
            return _invalid_decision()
        context = build_architecture_feature_context(mission_env)
        feature_matrix = architecture_feature_matrix(
            mission_env,
            candidates,
            self.feature_preset,
            context=context,
        )
        ranked = []
        for action, vector in zip(candidates, feature_matrix, strict=True):
            try:
                score = float(self.score_function(*vector.tolist()))
            except (ArithmeticError, OverflowError, ValueError):
                score = float("inf")
            if not math.isfinite(score):
                score = float("inf")
            ranked.append((score, action.tie_break_key, action, vector))
        score, _, selected, vector = min(ranked, key=lambda item: (item[0], item[1]))
        return ArchitectureDecision(
            action=selected,
            score=score,
            candidate_count=len(candidates),
            changed=selected.kind != "keep",
            valid=True,
            diagnostics={
                "feature_l1": float(np.sum(np.abs(vector))),
                "decision_version": float(version),
            },
        )

    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        decision = self.decide(mission_env)
        if not decision.valid:
            return decision
        result = apply_architecture_action(
            mission_env,
            decision.action,
            expected_decision_version=int(decision.diagnostics["decision_version"]),
        )
        return ArchitectureDecision(
            action=decision.action,
            score=decision.score,
            candidate_count=decision.candidate_count,
            changed=decision.changed,
            valid=bool(result.get("valid", False)),
            diagnostics=decision.diagnostics,
        )


class FixedArchitectureProvider:
    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        candidates = legal_architecture_actions(mission_env)
        keep = ArchitectureAction("keep")
        if keep not in candidates:
            return _invalid_decision(len(candidates))
        apply_architecture_action(
            mission_env,
            keep,
            expected_decision_version=mission_env.decision_version,
        )
        return ArchitectureDecision(keep, 0.0, len(candidates), False, True)


class RandomConcreteArchitectureProvider:
    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        candidates = legal_architecture_actions(mission_env)
        if not candidates:
            return _invalid_decision()
        action = candidates[int(self.rng.integers(0, len(candidates)))]
        result = apply_architecture_action(
            mission_env,
            action,
            expected_decision_version=mission_env.decision_version,
        )
        return ArchitectureDecision(
            action,
            0.0,
            len(candidates),
            action.kind != "keep",
            bool(result.get("valid", False)),
        )


class ManualRuleDQNProvider:
    """Compatibility adapter for the current six-rule architecture DQN."""

    def __init__(self, architecture_agent):
        self.agent = architecture_agent
        _freeze_networks(self.agent)

    def act(self, mission_env: env_module.MissionEnv) -> ArchitectureDecision:
        _freeze_networks(self.agent)
        policy = ArchitectureRule(mission_env)
        action_mask = policy.action_mask()
        candidate_count = int(np.count_nonzero(action_mask))
        if not np.any(action_mask):
            return _invalid_decision()
        rule_idx = int(
            self.agent.select_action(
                mission_env.architecture_observation(),
                action_mask,
                epsilon=0.0,
            )
        )
        resolution = policy.resolve(rule_idx)
        if resolution is None:
            raise RuntimeError("manual-rule DQN selected a masked action.")
        result = policy.apply(rule_idx)
        concrete = ArchitectureAction(
            resolution.kind,
            old_system=resolution.old_system,
            new_system=resolution.new_system,
        )
        return ArchitectureDecision(
            action=concrete,
            score=float(resolution.score),
            candidate_count=candidate_count,
            changed=concrete.kind != "keep",
            valid=bool(result.get("valid", False)),
            diagnostics={"manual_rule_index": float(rule_idx)},
        )


def as_architecture_provider(value) -> ArchitectureProvider:
    if isinstance(value, ArchitectureProvider):
        return value
    return ManualRuleDQNProvider(value)
