"""Validated configuration for the dynamic MOPSO baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MOPSOConfig:
    """Search parameters aligned with the NSGA-II fast profile."""

    swarm_size: int = 50
    max_evaluations: int = 5_000
    independent_runs: int = 3
    inertia_weight: float = 0.729844
    cognitive_coefficient: float = 1.49618
    social_coefficient: float = 1.49618
    max_velocity_rate: float = 0.20
    archive_size: int = 200
    architecture_change_weight: float = 0.01
    peak_budget_penalty: float = 20.0
    evaluation_milestones: tuple[int, ...] = ()
    random_fraction: float = 0.50
    makespan_fraction: float = 0.20
    cost_fraction: float = 0.20
    balanced_fraction: float = 0.10
    base_seed: int = 20260825
    workers: int = 1

    def __post_init__(self) -> None:
        milestones = tuple(
            sorted({int(value) for value in self.evaluation_milestones})
        )
        object.__setattr__(self, "evaluation_milestones", milestones)
        if self.swarm_size < 4:
            raise ValueError("swarm_size must be at least four.")
        if self.max_evaluations < self.swarm_size:
            raise ValueError("max_evaluations must cover the initial swarm.")
        if self.max_evaluations % self.swarm_size != 0:
            raise ValueError("max_evaluations must be a multiple of swarm_size.")
        if self.independent_runs <= 0:
            raise ValueError("independent_runs must be positive.")
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        if self.archive_size < 2:
            raise ValueError("archive_size must be at least two.")
        if not 0.0 <= float(self.inertia_weight) <= 1.0:
            raise ValueError("inertia_weight must be between zero and one.")
        for name in ("cognitive_coefficient", "social_coefficient"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative.")
        if not 0.0 < float(self.max_velocity_rate) <= 1.0:
            raise ValueError("max_velocity_rate must be in (0, 1].")
        if self.architecture_change_weight < 0.0:
            raise ValueError("architecture_change_weight cannot be negative.")
        if self.peak_budget_penalty < 0.0:
            raise ValueError("peak_budget_penalty cannot be negative.")
        if any(value <= 0 for value in milestones):
            raise ValueError("evaluation milestones must be positive.")
        if any(value > self.max_evaluations for value in milestones):
            raise ValueError(
                "evaluation milestones cannot exceed max_evaluations."
            )
        if any(value % self.swarm_size != 0 for value in milestones):
            raise ValueError(
                "evaluation milestones must be multiples of swarm_size."
            )
        fractions = (
            self.random_fraction,
            self.makespan_fraction,
            self.cost_fraction,
            self.balanced_fraction,
        )
        if any(value < 0.0 for value in fractions):
            raise ValueError("initialization fractions cannot be negative.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("initialization fractions must sum to one.")

    @property
    def gp_aligned_cost_defaults(self) -> bool:
        return bool(
            abs(float(self.architecture_change_weight) - 0.01) <= 1e-12
            and abs(float(self.peak_budget_penalty) - 20.0) <= 1e-12
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

