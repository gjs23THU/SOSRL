"""Fitness accounting and bounded DEAP evolution for architecture policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from functools import partial
import operator
from pathlib import Path
import pickle
import random
from typing import Any, Callable, Sequence

from deap import base, gp, tools
import numpy as np

from ..objectives import gp_cost_breakdown
from .config import GPArchitectureConfig
from .primitives import build_primitive_set, ensure_deap_types, tree_within_limits


EVOLUTION_STATE_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class EpisodeOutcome:
    success: bool
    completed_operations: int
    total_operations: int
    makespan: float
    scale: float
    final_net_cost: float
    peak_net_cost: float
    budget: float
    architecture_changes: int
    dead_end: bool = False


@dataclass(frozen=True)
class FitnessEvaluation:
    failure_rate: float
    regularized_j: float
    raw_mean_j: float

    @property
    def values(self) -> tuple[float, float]:
        return (self.failure_rate, self.regularized_j)


@dataclass(frozen=True)
class EvolutionRunResult:
    run_seed: int
    population: list[Any]
    generation_history: list[dict[str, Any]]
    anchor_history: list[dict[str, Any]]
    candidates: list[dict[str, Any]]


def episode_objective(outcome: EpisodeOutcome) -> float:
    scale = max(float(outcome.scale), 1.0)
    remaining_penalty = (
        0.0
        if outcome.success
        else 1.0
        - float(outcome.completed_operations) / max(int(outcome.total_operations), 1)
    )
    cost = gp_cost_breakdown(
        final_net_cost=outcome.final_net_cost,
        peak_net_cost=outcome.peak_net_cost,
        budget=outcome.budget,
        architecture_changes=outcome.architecture_changes,
    )
    return float(
        10.0 * float(outcome.makespan) / scale
        + cost.gp_cost_score
        + 10.0 * remaining_penalty
    )


def aggregate_fitness(
    outcomes: Sequence[EpisodeOutcome],
    *,
    node_count: int,
    parsimony_coefficient: float = 0.001,
) -> FitnessEvaluation:
    if not outcomes:
        raise ValueError("fitness requires at least one scenario outcome.")
    failure_rate = sum(not outcome.success for outcome in outcomes) / len(outcomes)
    raw_mean_j = float(np.mean([episode_objective(item) for item in outcomes]))
    return FitnessEvaluation(
        failure_rate=float(failure_rate),
        regularized_j=raw_mean_j + float(parsimony_coefficient) * int(node_count),
        raw_mean_j=raw_mean_j,
    )


def _mutate_constant(individual, pset):
    constant_indices = [
        index
        for index, node in enumerate(individual)
        if isinstance(type(node), gp.MetaEphemeral)
    ]
    if not constant_indices:
        return gp.mutNodeReplacement(individual, pset=pset)
    index = random.choice(constant_indices)
    individual[index] = type(individual[index])()
    return (individual,)


def _bounded_mutation(individual, pset, config: GPArchitectureConfig):
    draw = random.random()
    if draw < config.subtree_mutation_probability:
        expression = partial(
            gp.genFull,
            min_=config.mutation_min_depth,
            max_=config.mutation_max_depth,
        )
        return gp.mutUniform(individual, expr=expression, pset=pset)
    if draw < (
        config.subtree_mutation_probability + config.node_mutation_probability
    ):
        return gp.mutNodeReplacement(individual, pset=pset)
    return _mutate_constant(individual, pset)


def build_toolbox(feature_names: Sequence[str], config: GPArchitectureConfig):
    _, individual_type = ensure_deap_types()
    pset = build_primitive_set(feature_names)
    toolbox = base.Toolbox()
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=config.init_min_depth,
        max_=config.init_max_depth,
    )
    toolbox.register("individual", tools.initIterate, individual_type, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", _bounded_mutation, pset=pset, config=config)
    height_limit = gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.max_height
    )
    node_limit = gp.staticLimit(key=len, max_value=config.max_nodes)
    toolbox.decorate("mate", height_limit)
    toolbox.decorate("mate", node_limit)
    toolbox.decorate("mutate", height_limit)
    toolbox.decorate("mutate", node_limit)
    return toolbox, pset


def _variation(population, toolbox, config: GPArchitectureConfig):
    elites = [copy.deepcopy(item) for item in tools.selBest(population, config.elite_count)]
    offspring = list(elites)
    while len(offspring) < config.population_size:
        draw = random.random()
        if draw < config.crossover_probability:
            parents = toolbox.select(population, 2)
            first, second = map(copy.deepcopy, parents)
            children = toolbox.mate(first, second)
            for child in children:
                if len(offspring) >= config.population_size:
                    break
                offspring.append(child)
        elif draw < config.crossover_probability + config.mutation_probability:
            parent = copy.deepcopy(toolbox.select(population, 1)[0])
            child, = toolbox.mutate(parent)
            offspring.append(child)
        else:
            offspring.append(copy.deepcopy(toolbox.select(population, 1)[0]))
    for individual in offspring:
        if individual.fitness.valid:
            del individual.fitness.values
    return offspring


def evaluate_population(
    population,
    scenarios: Sequence[Any],
    individual_evaluator: Callable[[Any, Sequence[Any]], Sequence[EpisodeOutcome]],
    config: GPArchitectureConfig,
    population_evaluator: Callable[
        [Sequence[Any], Sequence[Any]], dict[str, Sequence[EpisodeOutcome]]
    ]
    | None = None,
) -> dict[str, FitnessEvaluation]:
    cache: dict[str, FitnessEvaluation] = {}
    population_outcomes = None
    if population_evaluator is not None:
        unique = {}
        for individual in population:
            unique.setdefault(str(individual), individual)
        population_outcomes = population_evaluator(
            list(unique.values()), scenarios
        )
    for individual in population:
        expression = str(individual)
        evaluation = cache.get(expression)
        if evaluation is None:
            outcomes = (
                population_outcomes[expression]
                if population_outcomes is not None
                else individual_evaluator(individual, scenarios)
            )
            evaluation = aggregate_fitness(
                outcomes,
                node_count=len(individual),
                parsimony_coefficient=config.parsimony_coefficient,
            )
            cache[expression] = evaluation
        individual.fitness.values = evaluation.values
        individual.raw_mean_j = evaluation.raw_mean_j
    return cache


def _candidate_row(individual, generation: int, source: str, run_seed: int):
    return {
        "run_seed": int(run_seed),
        "generation": int(generation),
        "source": source,
        "expression": str(individual),
        "failure_rate": float(individual.fitness.values[0]),
        "regularized_j": float(individual.fitness.values[1]),
        "raw_mean_j": float(getattr(individual, "raw_mean_j", np.nan)),
        "node_count": len(individual),
        "height": int(individual.height),
    }


def _atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def evolve_architecture_policy(
    *,
    feature_names: Sequence[str],
    config: GPArchitectureConfig,
    run_seed: int,
    batch_sampler: Callable[[int, int], Sequence[Any]],
    anchor_scenarios: Sequence[Any],
    individual_evaluator: Callable[[Any, Sequence[Any]], Sequence[EpisodeOutcome]],
    checkpoint_path: str | Path | None = None,
    resume_state: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    population_evaluator: Callable[
        [Sequence[Any], Sequence[Any]], dict[str, Sequence[EpisodeOutcome]]
    ]
    | None = None,
) -> EvolutionRunResult:
    """Run one independent, fully bounded GP evolution."""
    random.seed(run_seed)
    np.random.seed(run_seed % (2**32))
    toolbox, _ = build_toolbox(feature_names, config)
    generation_history: list[dict[str, Any]] = []
    anchor_history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    start_generation = 0

    if resume_state is not None:
        with Path(resume_state).open("rb") as file:
            state = pickle.load(file)
        if int(state.get("schema_version", -1)) != EVOLUTION_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported GP evolution-state schema version.")
        if tuple(state.get("feature_names", ())) != tuple(feature_names):
            raise ValueError("resume state feature registry does not match the requested evolution.")
        if state["config"] != asdict(config) or state["run_seed"] != int(run_seed):
            raise ValueError("resume state does not match the requested evolution.")
        population = state["population"]
        generation_history = state["generation_history"]
        anchor_history = state["anchor_history"]
        candidates = state["candidates"]
        start_generation = int(state["next_generation"])
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
    else:
        population = toolbox.population(n=config.population_size)

    for generation in range(start_generation, config.generations):
        if generation > 0 or start_generation > 0:
            population = _variation(population, toolbox, config)
        scenarios = batch_sampler(run_seed, generation)
        evaluate_population(
            population,
            scenarios,
            individual_evaluator,
            config,
            population_evaluator=population_evaluator,
        )
        if not all(
            tree_within_limits(
                individual,
                max_height=config.max_height,
                max_nodes=config.max_nodes,
            )
            for individual in population
        ):
            raise RuntimeError("variation produced a GP tree outside configured limits.")
        champion = tools.selBest(population, 1)[0]
        row = _candidate_row(champion, generation, "generation_champion", run_seed)
        generation_history.append(row)
        candidates.append(dict(row))
        if progress_callback is not None:
            progress_callback(dict(row))

        if (generation + 1) % config.anchor_interval == 0:
            top = tools.selBest(population, min(config.anchor_top_k, len(population)))
            anchor_copies = [copy.deepcopy(item) for item in top]
            evaluate_population(
                anchor_copies,
                anchor_scenarios,
                individual_evaluator,
                config,
                population_evaluator=population_evaluator,
            )
            for rank, individual in enumerate(tools.selBest(anchor_copies, len(anchor_copies)), 1):
                anchor_row = _candidate_row(
                    individual, generation, "anchor", run_seed
                )
                anchor_row["anchor_rank"] = rank
                anchor_history.append(anchor_row)
                candidates.append(dict(anchor_row))

        if checkpoint_path is not None:
            _atomic_pickle(
                Path(checkpoint_path),
                {
                    "schema_version": EVOLUTION_STATE_SCHEMA_VERSION,
                    "feature_names": list(feature_names),
                    "config": asdict(config),
                    "run_seed": int(run_seed),
                    "next_generation": generation + 1,
                    "population": population,
                    "generation_history": generation_history,
                    "anchor_history": anchor_history,
                    "candidates": candidates,
                    "random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                },
            )

    final_top = tools.selBest(population, min(10, len(population)))
    for individual in final_top:
        candidates.append(
            _candidate_row(
                individual,
                config.generations - 1,
                "final_top10",
                run_seed,
            )
        )
    return EvolutionRunResult(
        run_seed=int(run_seed),
        population=population,
        generation_history=generation_history,
        anchor_history=anchor_history,
        candidates=candidates,
    )
