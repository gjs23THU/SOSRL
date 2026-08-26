"""Chromosome layout and decoded result records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np

from .. import domain as syn
from .. import environment as env
from ..gp.architecture import raw_architecture_actions


@dataclass(frozen=True)
class ProblemLayout:
    """Canonical operation indexing shared by MS and AA genes."""

    operation_counts: tuple[int, ...]
    offsets: tuple[int, ...]
    operation_tasks: tuple[int, ...]
    operation_locals: tuple[int, ...]
    eligible_systems: tuple[tuple[int, ...], ...]
    system_count: int
    action_count: int

    @classmethod
    def from_mission(cls, mission: Sequence[syn.Task]) -> "ProblemLayout":
        if not mission:
            raise ValueError("mission must contain at least one task.")
        counts = tuple(len(task.operations) for task in mission)
        if any(count <= 0 for count in counts):
            raise ValueError("each task must contain at least one operation.")
        offsets: list[int] = []
        operation_tasks: list[int] = []
        operation_locals: list[int] = []
        eligible: list[tuple[int, ...]] = []
        offset = 0
        for task_idx, task in enumerate(mission):
            offsets.append(offset)
            for op_idx, operation in enumerate(task.operations):
                candidates = tuple(
                    int(system.index)
                    for system in env.FULL_SOS
                    if int(system.func_type) == int(operation.func_type)
                )
                if not candidates:
                    raise ValueError(
                        f"operation ({task_idx}, {op_idx}) has no capable system."
                    )
                operation_tasks.append(task_idx)
                operation_locals.append(op_idx)
                eligible.append(candidates)
            offset += len(task.operations)
        return cls(
            operation_counts=counts,
            offsets=tuple(offsets),
            operation_tasks=tuple(operation_tasks),
            operation_locals=tuple(operation_locals),
            eligible_systems=tuple(eligible),
            system_count=len(env.FULL_SOS),
            action_count=len(raw_architecture_actions()),
        )

    @property
    def task_count(self) -> int:
        return len(self.operation_counts)

    @property
    def operation_count(self) -> int:
        return len(self.operation_tasks)

    @property
    def base_os(self) -> np.ndarray:
        return np.repeat(
            np.arange(self.task_count, dtype=np.int32),
            np.asarray(self.operation_counts, dtype=np.int32),
        )

    def operation_index(self, task_idx: int, op_idx: int) -> int:
        task_idx = int(task_idx)
        op_idx = int(op_idx)
        if not 0 <= task_idx < self.task_count:
            raise ValueError("task index is outside the layout.")
        if not 0 <= op_idx < self.operation_counts[task_idx]:
            raise ValueError("operation index is outside the task.")
        return int(self.offsets[task_idx] + op_idx)

    def repair(self, chromosome: "Chromosome") -> tuple["Chromosome", dict[str, int]]:
        """Deterministically restore structural chromosome invariants."""
        k = self.operation_count
        counts = np.zeros(self.task_count, dtype=np.int32)
        repaired_os = np.full(k, -1, dtype=np.int32)
        os_repairs = 0
        source_os = np.asarray(chromosome.os, dtype=np.int32).reshape(-1)
        for position in range(k):
            value = int(source_os[position]) if position < source_os.size else -1
            if (
                0 <= value < self.task_count
                and counts[value] < self.operation_counts[value]
            ):
                repaired_os[position] = value
                counts[value] += 1
            else:
                os_repairs += 1
        missing = [
            task_idx
            for task_idx, required in enumerate(self.operation_counts)
            for _ in range(required - int(counts[task_idx]))
        ]
        for position, value in zip(
            np.flatnonzero(repaired_os < 0),
            missing,
            strict=True,
        ):
            repaired_os[int(position)] = int(value)

        source_ms = np.asarray(chromosome.ms, dtype=np.int32).reshape(-1)
        repaired_ms = np.empty(k, dtype=np.int32)
        ms_repairs = 0
        for op_idx, candidates in enumerate(self.eligible_systems):
            value = int(source_ms[op_idx]) if op_idx < source_ms.size else -1
            if value not in candidates:
                value = candidates[abs(value) % len(candidates)]
                ms_repairs += 1
            repaired_ms[op_idx] = value

        source_aa = np.asarray(chromosome.aa, dtype=np.int32).reshape(-1)
        repaired_aa = np.zeros(k, dtype=np.int32)
        aa_repairs = 0
        for op_idx in range(k):
            value = int(source_aa[op_idx]) if op_idx < source_aa.size else 0
            if not 0 <= value < self.action_count:
                value = 0
                aa_repairs += 1
            repaired_aa[op_idx] = value
        return (
            Chromosome(repaired_os, repaired_ms, repaired_aa),
            {
                "os_repair_count": int(os_repairs),
                "ms_repair_count": int(ms_repairs),
                "aa_repair_count": int(aa_repairs),
            },
        )

    def validate(self, chromosome: "Chromosome") -> None:
        repaired, repairs = self.repair(chromosome)
        if any(repairs.values()) or not np.array_equal(repaired.flat, chromosome.flat):
            raise ValueError("chromosome violates the problem layout.")


@dataclass(frozen=True)
class Chromosome:
    os: np.ndarray
    ms: np.ndarray
    aa: np.ndarray

    def __post_init__(self) -> None:
        lengths = {np.asarray(value).size for value in (self.os, self.ms, self.aa)}
        if len(lengths) != 1:
            raise ValueError("OS, MS, and AA must have the same length.")
        for name in ("os", "ms", "aa"):
            value = np.asarray(getattr(self, name), dtype=np.int32).reshape(-1).copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def operation_count(self) -> int:
        return int(self.os.size)

    @property
    def flat(self) -> np.ndarray:
        return np.concatenate((self.os, self.ms, self.aa)).astype(np.int32, copy=False)

    @classmethod
    def from_flat(cls, values: np.ndarray, operation_count: int) -> "Chromosome":
        values = np.asarray(values, dtype=np.int32).reshape(-1)
        k = int(operation_count)
        if values.size != 3 * k:
            raise ValueError(f"flat chromosome must contain {3 * k} genes.")
        return cls(values[:k], values[k:2 * k], values[2 * k:])

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.flat.astype("<i4", copy=False).tobytes()).hexdigest()


@dataclass(frozen=True)
class DecodeResult:
    chromosome: Chromosome
    chromosome_hash: str
    phenotype_hash: str
    success: bool
    dead_end: bool
    completed_operations: int
    total_operations: int
    makespan: float
    final_net_cost: float
    effective_cost: float
    gp_cost_score: float
    architecture_change_penalty: float
    peak_budget_penalty: float
    objective_makespan: float
    constraint_violation: float
    schedule: tuple[dict[str, Any], ...]
    architecture_trace: tuple[dict[str, Any], ...]
    metrics: dict[str, Any]
    repair_counts: dict[str, int]
    effective_os: tuple[int, ...]
    effective_ms: tuple[int, ...]
    effective_aa: tuple[int, ...]

    @property
    def objectives(self) -> tuple[float, float]:
        return (float(self.objective_makespan), float(self.effective_cost))

    def summary(self) -> dict[str, Any]:
        row = {
            "chromosome_hash": self.chromosome_hash,
            "phenotype_hash": self.phenotype_hash,
            "success": self.success,
            "dead_end": self.dead_end,
            "completed_operations": self.completed_operations,
            "total_operations": self.total_operations,
            "constraint_violation": self.constraint_violation,
            "makespan": self.makespan,
            "objective_makespan": self.objective_makespan,
            "final_net_cost": self.final_net_cost,
            "effective_cost": self.effective_cost,
            "gp_cost_score": self.gp_cost_score,
            "architecture_change_penalty": self.architecture_change_penalty,
            "peak_budget_penalty": self.peak_budget_penalty,
        }
        row.update(self.metrics)
        row.update(self.repair_counts)
        return row


def phenotype_digest(
    effective_os: Sequence[int],
    effective_ms: Sequence[int],
    effective_aa: Sequence[int],
    success: bool,
) -> str:
    payload = json.dumps(
        {
            "os": [int(value) for value in effective_os],
            "ms": [int(value) for value in effective_ms],
            "aa": [int(value) for value in effective_aa],
            "success": bool(success),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
