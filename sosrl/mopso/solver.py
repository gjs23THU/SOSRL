"""pymoo-backed mixed random-key MOPSO execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pymoo
from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
from pymoo.core.callback import Callback
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.optimize import minimize
from pymoo.operators.survival.rank_and_crowding.metrics import (
    get_crowding_function,
)
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from .. import domain as syn
from ..nsga2.decoder import DynamicScheduleDecoder
from ..nsga2.model import DecodeResult
from ..nsga2.operators import InitializationMode, heuristic_chromosome
from ..nsga2.solver import nondominated_results, select_representatives
from .codec import RandomKeyCodec
from .config import MOPSOConfig


class RandomKeySampling(Sampling):
    """Reuse NSGA-II's state-aware initialization in random-key space."""

    def __init__(
        self,
        decoder: DynamicScheduleDecoder,
        codec: RandomKeyCodec,
        config: MOPSOConfig,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.codec = codec
        self.config = config

    def _modes(self, n_samples: int) -> list[InitializationMode]:
        fractions: Sequence[tuple[InitializationMode, float]] = (
            ("random", self.config.random_fraction),
            ("makespan", self.config.makespan_fraction),
            ("cost", self.config.cost_fraction),
            ("balanced", self.config.balanced_fraction),
        )
        counts = {
            mode: int(math.floor(n_samples * fraction))
            for mode, fraction in fractions
        }
        counts["random"] += n_samples - sum(counts.values())
        return [mode for mode, _ in fractions for _ in range(counts[mode])]

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        modes = self._modes(int(n_samples))
        random_state.shuffle(modes)
        return np.asarray(
            [
                self.codec.encode(
                    heuristic_chromosome(self.decoder, mode, random_state),
                    random_state=random_state,
                    jitter=True,
                )
                for mode in modes
            ],
            dtype=np.float64,
        )


class DynamicMOPSOProblem(ElementwiseProblem):
    """Evaluate random-key particles through the shared dynamic decoder."""

    def __init__(self, decoder: DynamicScheduleDecoder) -> None:
        self.decoder = decoder
        self.codec = RandomKeyCodec(decoder.layout)
        self.cache: dict[str, DecodeResult] = {}
        self.position_results: dict[str, DecodeResult] = {}
        self.position_values: dict[str, np.ndarray] = {}
        self.feasible_archive: dict[str, DecodeResult] = {}
        self.position_by_phenotype: dict[str, np.ndarray] = {}
        self.position_digest_by_phenotype: dict[str, str] = {}
        super().__init__(
            n_var=self.codec.dimension,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.zeros(self.codec.dimension, dtype=np.float64),
            xu=np.ones(self.codec.dimension, dtype=np.float64),
            vtype=float,
        )

    def evaluate_position(self, position: np.ndarray) -> DecodeResult:
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        position_digest = self.codec.digest(position)
        existing_position = self.position_results.get(position_digest)
        if existing_position is not None:
            return existing_position
        chromosome = self.codec.decode(position)
        result = self.cache.get(chromosome.digest)
        if result is None:
            result = self.decoder.decode(chromosome)
            self.cache[chromosome.digest] = result
        self.position_results[position_digest] = result
        self.position_values[position_digest] = position.copy()
        if result.success:
            existing = self.feasible_archive.get(result.phenotype_hash)
            current_digest = self.position_digest_by_phenotype.get(
                result.phenotype_hash
            )
            if (
                existing is None
                or result.objectives < existing.objectives
                or (
                    result.objectives == existing.objectives
                    and (current_digest is None or position_digest < current_digest)
                )
            ):
                self.feasible_archive[result.phenotype_hash] = result
                self.position_by_phenotype[result.phenotype_hash] = position.copy()
                self.position_digest_by_phenotype[
                    result.phenotype_hash
                ] = position_digest
        return result

    def result_for_position(self, position: np.ndarray) -> DecodeResult:
        digest = self.codec.digest(position)
        if digest not in self.position_results:
            raise KeyError("particle position has not been evaluated.")
        return self.position_results[digest]

    def archive_key(self, position: np.ndarray) -> str:
        return self.result_for_position(position).phenotype_hash

    def _evaluate(self, X, out, *args, **kwargs):
        result = self.evaluate_position(X)
        out["F"] = np.asarray(result.objectives, dtype=np.float64)
        out["G"] = np.asarray([result.constraint_violation], dtype=np.float64)


class ConstraintAwareMOPSOCD(MOPSO_CD):
    """MOPSO-CD with exact evaluation accounting and Deb constraints."""

    def _setup(self, problem, **kwargs):
        # The upstream implementation evaluates a disposable swarm here and
        # samples again in _initialize_infill.  Setup only the state required by
        # the standard infill path so n_eval equals submitted particles.
        self.archive = None
        self.leader_archive = Population.empty()
        xl, xu = problem.bounds()
        self.v_max = self.max_velocity_rate * (xu - xl)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.leader_archive = self._update_leader_archive(infills)
        self.pbest = infills.copy()
        self.pbest_f = infills.get("F").copy()
        self.pbest_cv = infills.get("CV").reshape(-1).copy()

    def _advance(self, infills=None, **kwargs):
        if infills is None:
            return
        combined = Population.merge(self.pop, infills)
        self.leader_archive = self._update_leader_archive(combined)
        self._update_pbest(infills)
        self.pop = infills

    def _archive_key(self, individual) -> str:
        problem = getattr(self, "problem", None)
        if problem is not None and hasattr(problem, "archive_key"):
            return str(problem.archive_key(individual.X))
        return np.asarray(individual.X, dtype="<f8").tobytes().hex()

    def _update_leader_archive(self, population: Population) -> Population:
        if len(population) == 0:
            return Population.empty()
        combined = (
            Population.merge(self.leader_archive, population)
            if len(self.leader_archive)
            else population
        )
        unique: dict[str, Any] = {}
        for individual in combined:
            key = self._archive_key(individual)
            existing = unique.get(key)
            if existing is None:
                unique[key] = individual
                continue
            relation = Dominator.get_relation(
                individual.F,
                existing.F,
                float(individual.CV[0]),
                float(existing.CV[0]),
            )
            if relation == 1:
                unique[key] = individual
        candidates = Population(list(unique.values()))
        cv = candidates.get("CV").reshape(-1)
        feasible = cv <= 0.0
        if np.any(feasible):
            candidates = candidates[feasible]
        else:
            candidates = candidates[np.isclose(cv, np.min(cv))]
        indices = NonDominatedSorting().do(
            candidates.get("F"), only_non_dominated_front=True
        )
        front = candidates[indices]
        if len(front) <= self.archive_size:
            return front
        crowding = get_crowding_function("cd").do(front.get("F"))
        selected: list[int] = []
        remaining = list(range(len(front)))
        while len(selected) < self.archive_size:
            size = min(3, len(remaining))
            tournament = self.random_state.choice(
                remaining, size=size, replace=False
            )
            winner = int(tournament[np.argmax(crowding[tournament])])
            selected.append(winner)
            remaining.remove(winner)
        return front[selected]

    def _select_diverse_leaders(self):
        archive = self.leader_archive
        if len(archive) == 0:
            return [
                self.pop[int(self.random_state.integers(len(self.pop)))]
                for _ in range(self.pop_size)
            ]
        if len(archive) == 1:
            return [archive[0] for _ in range(self.pop_size)]
        crowding = get_crowding_function("cd").do(archive.get("F"))
        leaders = []
        for _ in range(self.pop_size):
            first, second = self.random_state.integers(0, len(archive), size=2)
            if crowding[first] > crowding[second]:
                leaders.append(archive[int(first)])
            elif crowding[second] > crowding[first]:
                leaders.append(archive[int(second)])
            else:
                leaders.append(archive[int(first if first <= second else second)])
        return leaders

    def _update_pbest(self, new_pop):
        for idx, current in enumerate(new_pop):
            previous = self.pbest[idx]
            relation = Dominator.get_relation(
                current.F,
                previous.F,
                float(current.CV[0]),
                float(previous.CV[0]),
            )
            replace = relation == 1
            if relation == 0 and self._archive_key(current) != self._archive_key(
                previous
            ):
                replace = bool(self.random_state.random() < 0.5)
            if replace:
                self.pbest[idx] = current.copy()
                self.pbest_f[idx] = current.F.copy()
                self.pbest_cv[idx] = float(current.CV[0])

    def _set_optimum(self, **kwargs):
        self.opt = (
            self.leader_archive.copy()
            if len(self.leader_archive)
            else Population.empty()
        )


class MOPSOHistory(Callback):
    def __init__(
        self,
        problem: DynamicMOPSOProblem,
        milestones: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.problem = problem
        self.milestones = tuple(sorted({int(value) for value in milestones}))
        self.milestone_fronts: dict[int, tuple[DecodeResult, ...]] = {}
        self.rows: list[dict[str, Any]] = []

    def notify(self, algorithm) -> None:
        evaluations = int(algorithm.evaluator.n_eval)
        cv = np.asarray(algorithm.pop.get("CV"), dtype=np.float64).reshape(-1)
        front = nondominated_results(list(self.problem.feasible_archive.values()))
        objectives = np.asarray(
            [result.objectives for result in front], dtype=np.float64
        )
        self.rows.append(
            {
                "iteration": int(algorithm.n_gen),
                "evaluations": evaluations,
                "swarm_size": int(len(algorithm.pop)),
                "feasible_count": int(np.count_nonzero(cv <= 0.0)),
                "feasible_rate": float(np.mean(cv <= 0.0)),
                "front_size": len(front),
                "ideal_makespan": (
                    None if not len(front) else float(np.min(objectives[:, 0]))
                ),
                "ideal_effective_cost": (
                    None if not len(front) else float(np.min(objectives[:, 1]))
                ),
                "nadir_makespan": (
                    None if not len(front) else float(np.max(objectives[:, 0]))
                ),
                "nadir_effective_cost": (
                    None if not len(front) else float(np.max(objectives[:, 1]))
                ),
                "leader_archive_size": int(len(algorithm.leader_archive)),
                "report_archive_size": int(len(self.problem.feasible_archive)),
                "unique_chromosomes": int(len(self.problem.cache)),
                "cache_hits": int(max(0, evaluations - len(self.problem.cache))),
            }
        )
        for milestone in self.milestones:
            if milestone > evaluations or milestone in self.milestone_fronts:
                continue
            if milestone != evaluations:
                raise RuntimeError(
                    "evaluation milestone was crossed between swarm batches; "
                    "use multiples of swarm_size."
                )
            self.milestone_fronts[milestone] = tuple(front)


@dataclass(frozen=True)
class MOPSORunResult:
    seed: int
    evaluations: int
    wall_seconds: float
    front: tuple[DecodeResult, ...]
    positions: dict[str, np.ndarray]
    milestone_fronts: dict[int, tuple[DecodeResult, ...]]
    history: tuple[dict[str, Any], ...]
    pymoo_version: str


@dataclass(frozen=True)
class MOPSOScenarioResult:
    runs: tuple[MOPSORunResult, ...]
    combined_front: tuple[DecodeResult, ...]
    positions: dict[str, np.ndarray]
    milestone_fronts: dict[int, tuple[DecodeResult, ...]]
    representatives: dict[str, DecodeResult]


def run_mopso(
    decoder: DynamicScheduleDecoder,
    config: MOPSOConfig,
    *,
    seed: int,
) -> MOPSORunResult:
    problem = DynamicMOPSOProblem(decoder)
    callback = MOPSOHistory(problem, config.evaluation_milestones)
    algorithm = ConstraintAwareMOPSOCD(
        pop_size=config.swarm_size,
        w=config.inertia_weight,
        c1=config.cognitive_coefficient,
        c2=config.social_coefficient,
        max_velocity_rate=config.max_velocity_rate,
        archive_size=config.archive_size,
        sampling=RandomKeySampling(decoder, problem.codec, config),
        seed=int(seed),
    )
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
    missing = set(config.evaluation_milestones) - set(callback.milestone_fronts)
    if missing:
        raise RuntimeError(f"MOPSO did not capture milestones: {sorted(missing)}")
    positions = {
        phenotype_hash: position.copy()
        for phenotype_hash, position in problem.position_by_phenotype.items()
    }
    return MOPSORunResult(
        seed=int(seed),
        evaluations=int(result.algorithm.evaluator.n_eval),
        wall_seconds=float(wall_seconds),
        front=tuple(front),
        positions=positions,
        milestone_fronts=dict(callback.milestone_fronts),
        history=tuple(callback.rows),
        pymoo_version=str(pymoo.__version__),
    )


def solve_scenario_mopso(
    architecture: Sequence[syn.ComponentSystem],
    mission: Sequence[syn.Task],
    *,
    budget: float = 8000.0,
    refund_rate: float = 0.8,
    config: MOPSOConfig | None = None,
) -> MOPSOScenarioResult:
    config = config or MOPSOConfig()
    decoder = DynamicScheduleDecoder(
        architecture,
        mission,
        budget=budget,
        refund_rate=refund_rate,
        architecture_change_weight=config.architecture_change_weight,
        peak_budget_penalty=config.peak_budget_penalty,
    )
    runs = tuple(
        run_mopso(decoder, config, seed=config.base_seed + run_idx)
        for run_idx in range(config.independent_runs)
    )
    combined = nondominated_results([item for run in runs for item in run.front])
    positions: dict[str, np.ndarray] = {}
    for run in runs:
        for phenotype_hash, position in run.positions.items():
            positions.setdefault(phenotype_hash, position.copy())
    milestone_fronts = {
        milestone: tuple(
            nondominated_results(
                [
                    item
                    for run in runs
                    for item in run.milestone_fronts[milestone]
                ]
            )
        )
        for milestone in config.evaluation_milestones
    }
    return MOPSOScenarioResult(
        runs=runs,
        combined_front=tuple(combined),
        positions=positions,
        milestone_fronts=milestone_fronts,
        representatives=select_representatives(combined),
    )

