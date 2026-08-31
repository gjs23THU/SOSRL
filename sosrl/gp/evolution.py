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
from .primitives import (
    build_primitive_set,
    ensure_deap_types,
    individual_from_expression,
    tree_within_limits,
)


EVOLUTION_STATE_SCHEMA_VERSION = 3


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
    actual_generations: int
    stop_reason: str
    convergence: dict[str, Any]


def initial_gp_convergence_state() -> dict[str, Any]:
    return {
        "best_failure_count": None,
        "best_failure_rate": None,
        "best_raw_mean_j": None,
        "stable_windows": 0,
        "provisional_generation": None,
        "confirmed_generation": None,
        "observations": [],
    }


def update_gp_convergence(
    state: dict[str, Any],
    anchor_winner: dict[str, Any],
    *,
    completed_generations: int,
    config: GPArchitectureConfig,
) -> tuple[dict[str, Any], bool]:
    """Update best-so-far anchor convergence and return whether it is confirmed."""

    updated = copy.deepcopy(state)
    failure = float(anchor_winner["failure_rate"])
    failure_count = int(round(failure * int(config.anchor_size)))
    raw_j = float(anchor_winner["raw_mean_j"])
    previous_failure = updated["best_failure_rate"]
    previous_j = updated["best_raw_mean_j"]
    relative_improvement = None
    failure_improved = previous_failure is not None and failure < float(previous_failure)

    if previous_failure is None:
        updated["best_failure_count"] = failure_count
        updated["best_failure_rate"] = failure
        updated["best_raw_mean_j"] = raw_j
        updated["stable_windows"] = 0
    elif failure_improved:
        updated["best_failure_count"] = failure_count
        updated["best_failure_rate"] = failure
        updated["best_raw_mean_j"] = raw_j
        updated["stable_windows"] = 0
        updated["provisional_generation"] = None
    elif failure == float(previous_failure):
        best_j = float(previous_j)
        if raw_j < best_j:
            relative_improvement = (best_j - raw_j) / max(abs(best_j), 1e-12)
            updated["best_raw_mean_j"] = raw_j
        else:
            relative_improvement = 0.0
        if relative_improvement < float(config.convergence_threshold):
            updated["stable_windows"] = int(updated["stable_windows"]) + 1
        else:
            updated["stable_windows"] = 0
            updated["provisional_generation"] = None
    else:
        # A worse current population does not change best-so-far performance.
        relative_improvement = 0.0
        updated["stable_windows"] = int(updated["stable_windows"]) + 1

    stable_windows = int(updated["stable_windows"])
    if (
        stable_windows >= int(config.convergence_patience)
        and updated["provisional_generation"] is None
    ):
        updated["provisional_generation"] = int(completed_generations)
    required = int(config.convergence_patience) + int(
        config.convergence_confirmation_windows
    )
    confirmed = (
        stable_windows >= required
        and int(completed_generations) >= int(config.min_generations)
    )
    if confirmed:
        updated["confirmed_generation"] = int(completed_generations)
    updated["observations"].append(
        {
            "generation": int(completed_generations),
            "anchor_failure_rate": failure,
            "anchor_failure_count": failure_count,
            "best_failure_count": int(updated["best_failure_count"]),
            "anchor_raw_mean_j": raw_j,
            "best_failure_rate": float(updated["best_failure_rate"]),
            "best_raw_mean_j": float(updated["best_raw_mean_j"]),
            "relative_best_j_improvement": relative_improvement,
            "failure_improved": bool(failure_improved),
            "stable_windows": stable_windows,
            "provisional": updated["provisional_generation"] is not None,
            "confirmed": bool(confirmed),
        }
    )
    return updated, bool(confirmed)


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
    try:
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
    except (IndexError, ValueError):
        # DEAP's typed subtree generator cannot replace a literal-only legacy
        # policy because this primitive set has no registered float terminal
        # class beyond arguments and ephemeral constants.  Wrap the policy in
        # a valid primitive so such a parent can still enter ordinary
        # evolution; the static height/node decorators retain the original if
        # the wrapper would exceed the configured bounds.
        fallback = individual_from_expression(f"negative({individual})", pset)
        individual[0 : len(individual)] = fallback
        return (individual,)


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


def mixed_parent_population(
    *,
    toolbox,
    pset,
    config: GPArchitectureConfig,
    parent_expression: str | None,
) -> tuple[list[Any], dict[str, int]]:
    """Create one exact parent, bounded parent mutants, and random diversity."""

    if parent_expression is None:
        population = toolbox.population(n=config.population_size)
        return population, {
            "parent": 0,
            "parent_mutants": 0,
            "random": int(config.population_size),
        }
    inherited = max(
        1,
        min(
            int(config.population_size),
            int(round(config.population_size * config.parent_population_fraction)),
        ),
    )
    parent = individual_from_expression(parent_expression, pset)
    if not tree_within_limits(
        parent,
        max_height=config.max_height,
        max_nodes=config.max_nodes,
    ):
        raise ValueError("parent GP policy exceeds configured tree limits.")
    population = [copy.deepcopy(parent)]
    seen = {str(parent)}
    target_mutants = inherited - 1
    attempts = 0
    max_attempts = max(100, target_mutants * 20)
    while len(population) - 1 < target_mutants and attempts < max_attempts:
        attempts += 1
        try:
            child, = toolbox.mutate(copy.deepcopy(parent))
        except (IndexError, ValueError):
            # A literal-only legacy policy may expose a DEAP return type with no
            # compatible generated terminal.  The unfilled inherited quota is
            # deliberately replaced by random diversity below.
            continue
        expression = str(child)
        if expression in seen:
            continue
        seen.add(expression)
        population.append(child)
    actual_mutants = len(population) - 1
    while len(population) < int(config.population_size):
        population.append(toolbox.individual())
    return population, {
        "parent": 1,
        "parent_mutants": int(actual_mutants),
        "random": int(config.population_size) - 1 - int(actual_mutants),
    }


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
    parent_expression: str | None = None,
) -> EvolutionRunResult:
    """Run one independent, fully bounded GP evolution."""
    random.seed(run_seed)
    np.random.seed(run_seed % (2**32))
    toolbox, pset = build_toolbox(feature_names, config)
    generation_history: list[dict[str, Any]] = []
    anchor_history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    start_generation = 0
    convergence = initial_gp_convergence_state()
    stop_reason: str | None = None
    population_sources = {"parent": 0, "parent_mutants": 0, "random": 0}

    if resume_state is not None:
        with Path(resume_state).open("rb") as file:
            state = pickle.load(file)
        if int(state.get("schema_version", -1)) != EVOLUTION_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported GP evolution-state schema version.")
        if tuple(state.get("feature_names", ())) != tuple(feature_names):
            raise ValueError("resume state feature registry does not match the requested evolution.")
        saved_config = dict(state["config"])
        requested_config = asdict(config)
        saved_generations = int(saved_config.pop("generations"))
        requested_generations = int(requested_config.pop("generations"))
        compatible_extension = (
            saved_config == requested_config
            and saved_generations <= requested_generations
        )
        if not compatible_extension or state["run_seed"] != int(run_seed):
            raise ValueError("resume state does not match the requested evolution.")
        population = state["population"]
        generation_history = state["generation_history"]
        anchor_history = state["anchor_history"]
        candidates = state["candidates"]
        start_generation = int(state["next_generation"])
        convergence = copy.deepcopy(
            state.get("convergence", initial_gp_convergence_state())
        )
        stop_reason = state.get("stop_reason")
        population_sources = dict(state.get("population_sources", population_sources))
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_random_state"])
    else:
        population, population_sources = mixed_parent_population(
            toolbox=toolbox,
            pset=pset,
            config=config,
            parent_expression=parent_expression,
        )

    last_generation = start_generation - 1
    for generation in (
        ()
        if stop_reason == "converged"
        else range(start_generation, config.generations)
    ):
        last_generation = generation
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

        if generation == 0 and parent_expression is not None:
            parent = next(
                item for item in population if str(item) == str(parent_expression)
            )
            candidates.append(
                _candidate_row(parent, generation, "parent_seed", run_seed)
            )

        confirmed = False
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
            ranked_anchors = tools.selBest(anchor_copies, len(anchor_copies))
            for rank, individual in enumerate(ranked_anchors, 1):
                anchor_row = _candidate_row(
                    individual, generation, "anchor", run_seed
                )
                anchor_row["anchor_rank"] = rank
                anchor_history.append(anchor_row)
                candidates.append(dict(anchor_row))
            convergence, confirmed = update_gp_convergence(
                convergence,
                anchor_history[-len(ranked_anchors)],
                completed_generations=generation + 1,
                config=config,
            )
            if confirmed:
                stop_reason = "converged"

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
                    "convergence": convergence,
                    "stop_reason": stop_reason,
                    "population_sources": population_sources,
                    "random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                },
            )
        if confirmed:
            break

    actual_generations = max(last_generation + 1, start_generation)
    if stop_reason is None:
        stop_reason = "max_generations_reached"
    final_top = tools.selBest(population, min(10, len(population)))
    for individual in final_top:
        candidates.append(
            _candidate_row(
                individual,
                max(actual_generations - 1, 0),
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
        actual_generations=int(actual_generations),
        stop_reason=str(stop_reason),
        convergence={
            **convergence,
            "stop_reason": str(stop_reason),
            "actual_generations": int(actual_generations),
            "max_generations": int(config.generations),
            "population_sources": population_sources,
        },
    )
