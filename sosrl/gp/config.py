"""Configuration for direct GP architecture-policy evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


FEATURE_SET_NAMES = frozenset(
    {"system", "system_demand", "system_delta", "op_context"}
)


@dataclass(frozen=True)
class GPArchitectureConfig:
    """Validated defaults for the standard direct-GP experiment."""

    population_size: int = 200
    generations: int = 80
    independent_runs: int = 10
    tournament_size: int = 5
    elite_count: int = 2
    crossover_probability: float = 0.75
    mutation_probability: float = 0.20
    reproduction_probability: float = 0.05
    subtree_mutation_probability: float = 0.50
    node_mutation_probability: float = 0.25
    constant_mutation_probability: float = 0.25
    init_min_depth: int = 2
    init_max_depth: int = 4
    mutation_min_depth: int = 0
    mutation_max_depth: int = 2
    max_height: int = 6
    max_nodes: int = 40
    train_batch_size: int = 16
    anchor_size: int = 64
    anchor_interval: int = 10
    anchor_top_k: int = 10
    convergence_interval: int = 0
    convergence_threshold: float = 0.01
    convergence_patience: int = 2
    convergence_confirmation_windows: int = 1
    min_generations: int = 0
    parent_population_fraction: float = 0.30
    parsimony_coefficient: float = 0.001
    base_seed: int = 20260820
    workers: int = 1
    feature_set: str = "system_delta"

    def __post_init__(self) -> None:
        positive_ints = {
            "population_size": self.population_size,
            "generations": self.generations,
            "independent_runs": self.independent_runs,
            "tournament_size": self.tournament_size,
            "train_batch_size": self.train_batch_size,
            "anchor_size": self.anchor_size,
            "anchor_interval": self.anchor_interval,
            "anchor_top_k": self.anchor_top_k,
            "convergence_patience": self.convergence_patience,
            "convergence_confirmation_windows": self.convergence_confirmation_windows,
            "max_nodes": self.max_nodes,
            "workers": self.workers,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.elite_count < 0 or self.elite_count >= self.population_size:
            raise ValueError("elite_count must be in [0, population_size).")
        if self.init_min_depth < 0 or self.init_min_depth > self.init_max_depth:
            raise ValueError("invalid initialization depth range.")
        if self.mutation_min_depth < 0 or self.mutation_min_depth > self.mutation_max_depth:
            raise ValueError("invalid mutation depth range.")
        if self.max_height <= 0 or self.init_max_depth > self.max_height:
            raise ValueError("max_height must cover the initialization depth.")
        if self.max_nodes < 3:
            raise ValueError("max_nodes must be at least three.")
        if self.parsimony_coefficient < 0.0:
            raise ValueError("parsimony_coefficient cannot be negative.")
        if not 0.0 < float(self.convergence_threshold) < 1.0:
            raise ValueError("convergence_threshold must be in (0, 1).")
        if int(self.convergence_interval) not in {0, int(self.anchor_interval)}:
            raise ValueError("convergence_interval must equal anchor_interval.")
        if int(self.min_generations) < 0 or int(self.min_generations) > int(self.generations):
            raise ValueError("min_generations cannot exceed generations.")
        if not 0.0 <= float(self.parent_population_fraction) <= 1.0:
            raise ValueError("parent_population_fraction must be in [0, 1].")
        self._validate_probability_group(
            "variation",
            self.crossover_probability,
            self.mutation_probability,
            self.reproduction_probability,
        )
        self._validate_probability_group(
            "mutation subtype",
            self.subtree_mutation_probability,
            self.node_mutation_probability,
            self.constant_mutation_probability,
        )
        if self.feature_set not in FEATURE_SET_NAMES:
            raise ValueError(
                f"feature_set must be one of {sorted(FEATURE_SET_NAMES)}."
            )

    @staticmethod
    def _validate_probability_group(name: str, *values: float) -> None:
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(f"{name} probabilities must be in [0, 1].")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{name} probabilities must sum to one.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
