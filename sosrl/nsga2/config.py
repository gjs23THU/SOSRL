"""Validated configuration for the dynamic NSGA-II solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NSGA2Config:
    """Search parameters with the approved fast profile as defaults."""

    population_size: int = 50
    max_evaluations: int = 5_000
    independent_runs: int = 3
    crossover_probability: float = 0.90
    os_mutation_probability: float = 0.20
    ms_gene_mutation_probability: float | None = None
    aa_gene_mutation_probability: float | None = None
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
        if self.population_size < 4:
            raise ValueError("population_size must be at least four.")
        if self.max_evaluations < self.population_size:
            raise ValueError("max_evaluations must cover the initial population.")
        if self.independent_runs <= 0:
            raise ValueError("independent_runs must be positive.")
        if any(value <= 0 for value in milestones):
            raise ValueError("evaluation milestones must be positive.")
        if any(value > self.max_evaluations for value in milestones):
            raise ValueError(
                "evaluation milestones cannot exceed max_evaluations."
            )
        if any(value % self.population_size != 0 for value in milestones):
            raise ValueError(
                "evaluation milestones must be multiples of population_size."
            )
        if self.workers <= 0:
            raise ValueError("workers must be positive.")
        for name in (
            "crossover_probability",
            "os_mutation_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        for name in (
            "ms_gene_mutation_probability",
            "aa_gene_mutation_probability",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if self.architecture_change_weight < 0.0:
            raise ValueError("architecture_change_weight cannot be negative.")
        if self.peak_budget_penalty < 0.0:
            raise ValueError("peak_budget_penalty cannot be negative.")
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

    def ms_mutation_probability(self, operation_count: int) -> float:
        if self.ms_gene_mutation_probability is not None:
            return float(self.ms_gene_mutation_probability)
        return 1.0 / max(int(operation_count), 1)

    def aa_mutation_probability(self, operation_count: int) -> float:
        if self.aa_gene_mutation_probability is not None:
            return float(self.aa_gene_mutation_probability)
        return 1.0 / max(int(operation_count), 1)

    @property
    def gp_aligned_cost_defaults(self) -> bool:
        return bool(
            abs(float(self.architecture_change_weight) - 0.01) <= 1e-12
            and abs(float(self.peak_budget_penalty) - 20.0) <= 1e-12
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
