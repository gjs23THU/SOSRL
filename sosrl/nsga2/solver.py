"""pymoo-backed NSGA-II execution, archives, and representative selection."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from .. import domain as syn
from .config import NSGA2Config
from .decoder import DynamicScheduleDecoder
from .model import Chromosome, DecodeResult
from .operators import DynamicCrossover, DynamicMutation, DynamicSampling


def dominates(first: DecodeResult, second: DecodeResult) -> bool:
    if first.constraint_violation <= 0.0 < second.constraint_violation:
        return True
    if second.constraint_violation <= 0.0 < first.constraint_violation:
        return False
    if first.constraint_violation > 0.0 or second.constraint_violation > 0.0:
        if first.constraint_violation != second.constraint_violation:
            return first.constraint_violation < second.constraint_violation
    left = np.asarray(first.objectives, dtype=np.float64)
    right = np.asarray(second.objectives, dtype=np.float64)
    return bool(np.all(left <= right) and np.any(left < right))


def nondominated_results(results: Sequence[DecodeResult]) -> list[DecodeResult]:
    unique: dict[str, DecodeResult] = {}
    for result in results:
        existing = unique.get(result.phenotype_hash)
        if existing is None or result.objectives < existing.objectives:
            unique[result.phenotype_hash] = result
    values = list(unique.values())
    return sorted(
        [
            candidate
            for index, candidate in enumerate(values)
            if not any(
                dominates(other, candidate)
                for other_idx, other in enumerate(values)
                if other_idx != index
            )
        ],
        key=lambda result: (result.objectives, result.phenotype_hash),
    )


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=np.float64)
    if objectives.ndim != 2:
        raise ValueError("objectives must be a two-dimensional matrix.")
    count = objectives.shape[0]
    distance = np.zeros(count, dtype=np.float64)
    if count == 0:
        return distance
    if count <= 2:
        distance.fill(np.inf)
        return distance
    for objective_idx in range(objectives.shape[1]):
        order = np.argsort(objectives[:, objective_idx], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        low = float(objectives[order[0], objective_idx])
        high = float(objectives[order[-1], objective_idx])
        if high <= low + 1e-12:
            continue
        for position in range(1, count - 1):
            index = int(order[position])
            if np.isinf(distance[index]):
                continue
            previous_value = float(objectives[order[position - 1], objective_idx])
            next_value = float(objectives[order[position + 1], objective_idx])
            distance[index] += (next_value - previous_value) / (high - low)
    return distance


def select_representatives(front: Sequence[DecodeResult]) -> dict[str, DecodeResult]:
    feasible = [result for result in front if result.success]
    if not feasible:
        return {}
    min_makespan = min(
        feasible,
        key=lambda result: (
            result.makespan,
            result.effective_cost,
            result.final_net_cost,
            result.phenotype_hash,
        ),
    )
    min_cost = min(
        feasible,
        key=lambda result: (
            result.effective_cost,
            result.makespan,
            result.final_net_cost,
            result.phenotype_hash,
        ),
    )
    objectives = np.asarray(
        [[result.makespan, result.effective_cost] for result in feasible],
        dtype=np.float64,
    )
    ideal = np.min(objectives, axis=0)
    nadir = np.max(objectives, axis=0)
    normalized = (objectives - ideal) / np.maximum(nadir - ideal, 1e-12)
    asf = np.max(normalized / 0.5, axis=1) + 1e-6 * np.sum(normalized, axis=1)
    compromise_idx = min(
        range(len(feasible)),
        key=lambda index: (
            float(asf[index]),
            float(np.sum(normalized[index])),
            feasible[index].makespan,
            feasible[index].effective_cost,
            feasible[index].final_net_cost,
            feasible[index].phenotype_hash,
        ),
    )
    return {
        "min_makespan": min_makespan,
        "min_cost": min_cost,
        "compromise": feasible[compromise_idx],
    }


class DynamicNSGA2Problem(ElementwiseProblem):
    def __init__(self, decoder: DynamicScheduleDecoder) -> None:
        self.decoder = decoder
        self.layout = decoder.layout
        self.cache: dict[str, DecodeResult] = {}
        self.feasible_archive: dict[str, DecodeResult] = {}
        k = self.layout.operation_count
        super().__init__(
            n_var=3 * k,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.zeros(3 * k, dtype=np.int32),
            xu=np.concatenate(
                (
                    np.full(k, self.layout.task_count - 1, dtype=np.int32),
                    np.full(k, self.layout.system_count - 1, dtype=np.int32),
                    np.full(k, self.layout.action_count - 1, dtype=np.int32),
                )
            ),
            vtype=np.int32,
        )

    def evaluate_chromosome(self, chromosome: Chromosome) -> DecodeResult:
        repaired, _ = self.layout.repair(chromosome)
        result = self.cache.get(repaired.digest)
        if result is None:
            # Decode the submitted genes so structural repair statistics are
            # retained; use the repaired digest only as the phenotype cache key.
            result = self.decoder.decode(chromosome)
            self.cache[repaired.digest] = result
            if result.success:
                existing = self.feasible_archive.get(result.phenotype_hash)
                if existing is None or result.objectives < existing.objectives:
                    self.feasible_archive[result.phenotype_hash] = result
        return result

    def _evaluate(self, X, out, *args, **kwargs):
        result = self.evaluate_chromosome(
            Chromosome.from_flat(X, self.layout.operation_count)
        )
        out["F"] = np.asarray(result.objectives, dtype=np.float64)
        out["G"] = np.asarray([result.constraint_violation], dtype=np.float64)


class EvolutionHistory(Callback):
    def __init__(
        self,
        problem: DynamicNSGA2Problem,
        milestones: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.problem = problem
        self.milestones = tuple(sorted({int(value) for value in milestones}))
        self.milestone_fronts: dict[int, tuple[DecodeResult, ...]] = {}
        self.rows: list[dict[str, Any]] = []

    def notify(self, algorithm) -> None:
        evaluations = int(algorithm.evaluator.n_eval)
        objectives = np.asarray(algorithm.pop.get("F"), dtype=np.float64)
        cv = np.asarray(algorithm.pop.get("CV"), dtype=np.float64).reshape(-1)
        feasible = cv <= 0.0
        ranks = algorithm.pop.get("rank")
        front_mask = feasible if ranks is None else feasible & (np.asarray(ranks) == 0)
        front = objectives[front_mask]
        self.rows.append(
            {
                "generation": int(algorithm.n_gen),
                "evaluations": evaluations,
                "population_size": int(len(algorithm.pop)),
                "feasible_count": int(np.count_nonzero(feasible)),
                "feasible_rate": float(np.mean(feasible)),
                "front_size": int(front.shape[0]),
                "ideal_makespan": None if not front.size else float(np.min(front[:, 0])),
                "ideal_effective_cost": (
                    None if not front.size else float(np.min(front[:, 1]))
                ),
                "nadir_makespan": None if not front.size else float(np.max(front[:, 0])),
                "nadir_effective_cost": (
                    None if not front.size else float(np.max(front[:, 1]))
                ),
                "archive_size": int(len(self.problem.feasible_archive)),
            }
        )
        for milestone in self.milestones:
            if milestone > evaluations or milestone in self.milestone_fronts:
                continue
            if milestone != evaluations:
                raise RuntimeError(
                    "evaluation milestone was crossed between population batches; "
                    "use multiples of population_size."
                )
            self.milestone_fronts[milestone] = tuple(
                nondominated_results(
                    list(self.problem.feasible_archive.values())
                )
            )


@dataclass(frozen=True)
class NSGA2RunResult:
    seed: int
    evaluations: int
    wall_seconds: float
    front: tuple[DecodeResult, ...]
    milestone_fronts: dict[int, tuple[DecodeResult, ...]]
    history: tuple[dict[str, Any], ...]
    pymoo_version: str


@dataclass(frozen=True)
class NSGA2ScenarioResult:
    runs: tuple[NSGA2RunResult, ...]
    combined_front: tuple[DecodeResult, ...]
    milestone_fronts: dict[int, tuple[DecodeResult, ...]]
    representatives: dict[str, DecodeResult]


def run_nsga2(
    decoder: DynamicScheduleDecoder,
    config: NSGA2Config,
    *,
    seed: int,
) -> NSGA2RunResult:
    problem = DynamicNSGA2Problem(decoder)
    callback = EvolutionHistory(problem, config.evaluation_milestones)
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=DynamicSampling(decoder, config),
        crossover=DynamicCrossover(decoder.layout, config.crossover_probability),
        mutation=DynamicMutation(decoder.layout, config),
        eliminate_duplicates=True,
    )
    algorithm.tournament_type = "comp_by_rank_and_crowding"
    started = perf_counter()
    result = minimize(
        problem,
        algorithm,
        termination=("n_eval", config.max_evaluations),
        seed=int(seed),
        callback=callback,
        verbose=False,
        save_history=False,
    )
    wall_seconds = perf_counter() - started
    front = nondominated_results(list(problem.feasible_archive.values()))
    missing_milestones = set(config.evaluation_milestones) - set(
        callback.milestone_fronts
    )
    if missing_milestones:
        raise RuntimeError(
            f"NSGA-II did not capture milestones: {sorted(missing_milestones)}"
        )
    return NSGA2RunResult(
        seed=int(seed),
        evaluations=int(result.algorithm.evaluator.n_eval),
        wall_seconds=float(wall_seconds),
        front=tuple(front),
        milestone_fronts=dict(callback.milestone_fronts),
        history=tuple(callback.rows),
        pymoo_version=str(pymoo.__version__),
    )


def solve_scenario_nsga2(
    architecture: Sequence[syn.ComponentSystem],
    mission: Sequence[syn.Task],
    *,
    budget: float = 8000.0,
    refund_rate: float = 0.8,
    config: NSGA2Config | None = None,
) -> NSGA2ScenarioResult:
    config = config or NSGA2Config()
    decoder = DynamicScheduleDecoder(
        architecture,
        mission,
        budget=budget,
        refund_rate=refund_rate,
        architecture_change_weight=config.architecture_change_weight,
        peak_budget_penalty=config.peak_budget_penalty,
    )
    runs = tuple(
        run_nsga2(decoder, config, seed=config.base_seed + run_idx)
        for run_idx in range(config.independent_runs)
    )
    combined = nondominated_results(
        [result for run in runs for result in run.front]
    )
    milestone_fronts = {
        milestone: tuple(
            nondominated_results(
                [
                    result
                    for run in runs
                    for result in run.milestone_fronts[milestone]
                ]
            )
        )
        for milestone in config.evaluation_milestones
    }
    return NSGA2ScenarioResult(
        runs=runs,
        combined_front=tuple(combined),
        milestone_fronts=milestone_fronts,
        representatives=select_representatives(combined),
    )
