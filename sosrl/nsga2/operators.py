"""GP-action-aware initialization and variation operators for NSGA-II."""

from __future__ import annotations

import math
from typing import Literal, Sequence

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from .. import environment as env
from ..gp.architecture import (
    apply_architecture_action,
    architecture_action_id,
    legal_architecture_actions,
)
from .config import NSGA2Config
from .decoder import DynamicScheduleDecoder
from .model import Chromosome, ProblemLayout


InitializationMode = Literal["random", "makespan", "cost", "balanced"]


def _initial_gene_arrays(
    decoder: DynamicScheduleDecoder,
    random_state: np.random.Generator,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    layout = decoder.layout
    os_values = random_state.permutation(layout.base_os).astype(np.int32).tolist()
    ms_values = np.asarray(
        [random_state.choice(candidates) for candidates in layout.eligible_systems],
        dtype=np.int32,
    )
    aa_values = np.zeros(layout.operation_count, dtype=np.int32)
    return os_values, ms_values, aa_values


def _next_constructible_position(
    decoder: DynamicScheduleDecoder,
    mission_env: env.MissionEnv,
    os_values: list[int],
    position: int,
) -> int | None:
    seen: set[int] = set()
    for candidate_position in range(position, len(os_values)):
        task_idx = int(os_values[candidate_position])
        if task_idx in seen:
            continue
        seen.add(task_idx)
        if decoder._operation_legal_actions(mission_env, task_idx):
            return candidate_position
    return None


def random_chromosome(
    decoder: DynamicScheduleDecoder,
    random_state: np.random.Generator,
) -> Chromosome:
    """Construct a random chromosome through state-valid GP actions."""
    layout = decoder.layout
    mission_env = decoder._new_environment()
    os_values, ms_values, aa_values = _initial_gene_arrays(decoder, random_state)
    for position in range(layout.operation_count):
        candidate_position = _next_constructible_position(
            decoder, mission_env, os_values, position
        )
        if candidate_position is None:
            break
        if candidate_position != position:
            task_idx = os_values.pop(candidate_position)
            os_values.insert(position, task_idx)
        task_idx = int(os_values[position])
        op_idx = int(mission_env.state.task_op_idx[task_idx])
        canonical_idx = layout.operation_index(task_idx, op_idx)
        legal = decoder._operation_legal_actions(mission_env, task_idx)
        action = legal[int(random_state.integers(len(legal)))]
        feasible = decoder._active_feasible_systems(mission_env, task_idx, action)
        sys_idx = int(random_state.choice(feasible))
        aa_values[canonical_idx] = architecture_action_id(action)
        ms_values[canonical_idx] = sys_idx
        apply_architecture_action(mission_env, action)
        _, _, terminated, _, _ = mission_env.step(
            mission_env.encode_assignment(task_idx, op_idx, sys_idx)
        )
        if terminated:
            break
    return Chromosome(
        np.asarray(os_values, dtype=np.int32),
        ms_values,
        aa_values,
    )


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low = float(np.min(values))
    high = float(np.max(values))
    if high <= low + 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def heuristic_chromosome(
    decoder: DynamicScheduleDecoder,
    mode: InitializationMode,
    random_state: np.random.Generator,
) -> Chromosome:
    if mode == "random":
        return random_chromosome(decoder, random_state)
    layout = decoder.layout
    mission_env = decoder._new_environment()
    os_values: list[int] = []
    ms_values = np.asarray(
        [random_state.choice(candidates) for candidates in layout.eligible_systems],
        dtype=np.int32,
    )
    aa_values = np.zeros(layout.operation_count, dtype=np.int32)

    for _ in range(layout.operation_count):
        candidates: list[tuple[int, int, int, object, float, float]] = []
        finishes = mission_env.current_candidate_finish_times()
        legal_actions = legal_architecture_actions(mission_env)
        for task_idx in range(layout.task_count):
            op_idx = int(mission_env.state.task_op_idx[task_idx])
            if op_idx >= layout.operation_counts[task_idx]:
                continue
            canonical_idx = layout.operation_index(task_idx, op_idx)
            for action in decoder._operation_legal_actions(
                mission_env, task_idx, legal_actions
            ):
                feasible = decoder._active_feasible_systems(
                    mission_env, task_idx, action
                )
                sys_idx = min(
                    (int(value) for value in feasible),
                    key=lambda value: (
                        float(finishes[task_idx, value]),
                        float(env.FULL_SOS[value].cost),
                        value,
                    ),
                )
                candidates.append(
                    (
                        task_idx,
                        canonical_idx,
                        sys_idx,
                        action,
                        float(finishes[task_idx, sys_idx]),
                        decoder._effective_cost_delta(mission_env, action),
                    )
                )
        if not candidates:
            break
        normalized_finishes = _normalize(
            np.asarray([value[4] for value in candidates])
        )
        normalized_costs = _normalize(
            np.asarray([value[5] for value in candidates])
        )
        if mode == "makespan":
            scores = normalized_finishes
        elif mode == "cost":
            scores = normalized_costs
        else:
            scores = 0.5 * normalized_finishes + 0.5 * normalized_costs
        order = np.argsort(scores, kind="stable")
        rcl_size = max(1, int(math.ceil(0.2 * len(candidates))))
        selected_idx = int(random_state.choice(order[:rcl_size]))
        task_idx, canonical_idx, sys_idx, action, _, _ = candidates[selected_idx]
        op_idx = int(mission_env.state.task_op_idx[task_idx])
        os_values.append(task_idx)
        ms_values[canonical_idx] = sys_idx
        aa_values[canonical_idx] = architecture_action_id(action)
        apply_architecture_action(mission_env, action)
        _, _, terminated, _, _ = mission_env.step(
            mission_env.encode_assignment(task_idx, op_idx, sys_idx)
        )
        if terminated:
            break

    scheduled_counts = np.bincount(
        np.asarray(os_values, dtype=np.int32),
        minlength=layout.task_count,
    )
    missing = [
        task_idx
        for task_idx, required in enumerate(layout.operation_counts)
        for _ in range(required - int(scheduled_counts[task_idx]))
    ]
    if missing:
        os_values.extend(random_state.permutation(np.asarray(missing)).tolist())
    return Chromosome(
        np.asarray(os_values, dtype=np.int32),
        ms_values,
        aa_values,
    )


def pox_pair(
    first: np.ndarray,
    second: np.ndarray,
    task_count: int,
    random_state: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.int32)
    second = np.asarray(second, dtype=np.int32)
    if first.shape != second.shape:
        raise ValueError("POX parents must have the same shape.")
    if task_count <= 1:
        return first.copy(), second.copy()
    subset_size = int(random_state.integers(1, task_count))
    selected = set(
        int(value)
        for value in random_state.choice(
            np.arange(task_count), size=subset_size, replace=False
        )
    )

    def child(preserve: np.ndarray, fill: np.ndarray) -> np.ndarray:
        result = np.full(preserve.size, -1, dtype=np.int32)
        mask = np.asarray([int(value) in selected for value in preserve], dtype=bool)
        result[mask] = preserve[mask]
        filler = [int(value) for value in fill if int(value) not in selected]
        result[~mask] = np.asarray(filler, dtype=np.int32)
        return result

    return child(first, second), child(second, first)


def mutate_chromosome(
    chromosome: Chromosome,
    layout: ProblemLayout,
    config: NSGA2Config,
    random_state: np.random.Generator,
) -> Chromosome:
    os_values = chromosome.os.copy()
    ms_values = chromosome.ms.copy()
    aa_values = chromosome.aa.copy()

    if random_state.random() < config.os_mutation_probability and os_values.size > 1:
        if random_state.random() < 0.5:
            differing = np.argwhere(os_values[:, None] != os_values[None, :])
            if differing.size:
                first, second = differing[int(random_state.integers(len(differing)))]
                os_values[first], os_values[second] = os_values[second], os_values[first]
        else:
            source, target = random_state.choice(os_values.size, size=2, replace=False)
            value = int(os_values[source])
            os_values = np.delete(os_values, source)
            os_values = np.insert(os_values, target, value).astype(np.int32)

    ms_probability = config.ms_mutation_probability(layout.operation_count)
    for op_idx, candidates in enumerate(layout.eligible_systems):
        if random_state.random() >= ms_probability or len(candidates) <= 1:
            continue
        alternatives = [value for value in candidates if value != int(ms_values[op_idx])]
        ms_values[op_idx] = int(random_state.choice(alternatives))

    aa_probability = config.aa_mutation_probability(layout.operation_count)
    for op_idx in range(layout.operation_count):
        if random_state.random() >= aa_probability:
            continue
        current = int(aa_values[op_idx])
        draw = int(random_state.integers(layout.action_count - 1))
        aa_values[op_idx] = draw if draw < current else draw + 1
    mutated, _ = layout.repair(Chromosome(os_values, ms_values, aa_values))
    return mutated


class DynamicSampling(Sampling):
    def __init__(self, decoder: DynamicScheduleDecoder, config: NSGA2Config) -> None:
        super().__init__()
        self.decoder = decoder
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
                heuristic_chromosome(self.decoder, mode, random_state).flat
                for mode in modes
            ],
            dtype=np.int32,
        )


class DynamicCrossover(Crossover):
    def __init__(self, layout: ProblemLayout, probability: float) -> None:
        super().__init__(2, 2, prob=float(probability), vtype=np.int32)
        self.layout = layout

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        k = self.layout.operation_count
        children = np.empty((2, X.shape[1], X.shape[2]), dtype=np.int32)
        for mating_idx in range(X.shape[1]):
            first = Chromosome.from_flat(X[0, mating_idx], k)
            second = Chromosome.from_flat(X[1, mating_idx], k)
            first_os, second_os = pox_pair(
                first.os, second.os, self.layout.task_count, random_state
            )
            ms_mask = random_state.random(k) < 0.5
            first_ms = np.where(ms_mask, first.ms, second.ms)
            second_ms = np.where(ms_mask, second.ms, first.ms)
            aa_mask = random_state.random(k) < 0.5
            first_aa = np.where(aa_mask, first.aa, second.aa)
            second_aa = np.where(aa_mask, second.aa, first.aa)
            children[0, mating_idx] = Chromosome(
                first_os, first_ms, first_aa
            ).flat
            children[1, mating_idx] = Chromosome(
                second_os, second_ms, second_aa
            ).flat
        return children


class DynamicMutation(Mutation):
    def __init__(self, layout: ProblemLayout, config: NSGA2Config) -> None:
        super().__init__(prob=1.0, vtype=np.int32)
        self.layout = layout
        self.config = config

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        return np.asarray(
            [
                mutate_chromosome(
                    Chromosome.from_flat(values, self.layout.operation_count),
                    self.layout,
                    self.config,
                    random_state,
                ).flat
                for values in X
            ],
            dtype=np.int32,
        )
