"""Mixed random-key encoding for OS/MS/AA chromosomes."""

from __future__ import annotations

import hashlib

import numpy as np

from ..nsga2.model import Chromosome, ProblemLayout


class RandomKeyCodec:
    """Map bounded continuous particle positions to valid chromosomes."""

    def __init__(self, layout: ProblemLayout) -> None:
        self.layout = layout

    @property
    def dimension(self) -> int:
        return 3 * self.layout.operation_count

    def _position(self, values: np.ndarray) -> np.ndarray:
        position = np.asarray(values, dtype=np.float64).reshape(-1)
        if position.size != self.dimension:
            raise ValueError(
                f"particle position must contain {self.dimension} values."
            )
        if not np.all(np.isfinite(position)):
            raise ValueError("particle position must contain finite values.")
        return np.clip(position, 0.0, 1.0)

    @staticmethod
    def _bin_index(value: float, count: int) -> int:
        return min(int(np.floor(float(value) * int(count))), int(count) - 1)

    def decode(self, values: np.ndarray) -> Chromosome:
        position = self._position(values)
        k = self.layout.operation_count
        os_keys = position[:k]
        ms_keys = position[k : 2 * k]
        aa_keys = position[2 * k :]
        operation_tasks = np.asarray(
            self.layout.operation_tasks, dtype=np.int32
        )
        order = np.argsort(os_keys, kind="stable")
        os_values = operation_tasks[order]
        ms_values = np.asarray(
            [
                candidates[self._bin_index(ms_keys[idx], len(candidates))]
                for idx, candidates in enumerate(self.layout.eligible_systems)
            ],
            dtype=np.int32,
        )
        aa_values = np.asarray(
            [
                self._bin_index(value, self.layout.action_count)
                for value in aa_keys
            ],
            dtype=np.int32,
        )
        chromosome = Chromosome(os_values, ms_values, aa_values)
        self.layout.validate(chromosome)
        return chromosome

    def encode(
        self,
        chromosome: Chromosome,
        *,
        random_state: np.random.Generator | None = None,
        jitter: bool = False,
    ) -> np.ndarray:
        chromosome, _ = self.layout.repair(chromosome)
        k = self.layout.operation_count
        os_keys = np.empty(k, dtype=np.float64)
        occurrences = np.zeros(self.layout.task_count, dtype=np.int32)
        for rank, task_idx in enumerate(chromosome.os):
            task_idx = int(task_idx)
            op_idx = int(occurrences[task_idx])
            canonical_idx = self.layout.operation_index(task_idx, op_idx)
            os_keys[canonical_idx] = (rank + 0.5) / k
            occurrences[task_idx] += 1

        ms_keys = np.empty(k, dtype=np.float64)
        for idx, candidates in enumerate(self.layout.eligible_systems):
            candidate_idx = candidates.index(int(chromosome.ms[idx]))
            ms_keys[idx] = (candidate_idx + 0.5) / len(candidates)

        aa_keys = (
            chromosome.aa.astype(np.float64) + 0.5
        ) / self.layout.action_count
        position = np.concatenate((os_keys, ms_keys, aa_keys))
        if jitter:
            if random_state is None:
                raise ValueError("random_state is required when jitter is enabled.")
            position[:k] += random_state.uniform(-0.2 / k, 0.2 / k, k)
            for idx, candidates in enumerate(self.layout.eligible_systems):
                position[k + idx] += random_state.uniform(
                    -0.2 / len(candidates), 0.2 / len(candidates)
                )
            position[2 * k :] += random_state.uniform(
                -0.2 / self.layout.action_count,
                0.2 / self.layout.action_count,
                k,
            )
        upper = np.nextafter(1.0, 0.0)
        position = np.clip(position, 0.0, upper)
        if not np.array_equal(self.decode(position).flat, chromosome.flat):
            raise RuntimeError("random-key encoding did not preserve chromosome.")
        return position

    def split(self, values: np.ndarray) -> dict[str, tuple[float, ...]]:
        position = self._position(values)
        k = self.layout.operation_count
        return {
            "os_keys": tuple(float(value) for value in position[:k]),
            "ms_keys": tuple(float(value) for value in position[k : 2 * k]),
            "aa_keys": tuple(float(value) for value in position[2 * k :]),
        }

    def digest(self, values: np.ndarray) -> str:
        position = self._position(values)
        return hashlib.sha256(
            position.astype("<f8", copy=False).tobytes()
        ).hexdigest()

