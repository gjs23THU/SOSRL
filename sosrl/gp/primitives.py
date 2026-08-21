"""Protected DEAP primitive set and deterministic GP tree execution."""

from __future__ import annotations

from functools import partial
import math
import random
from typing import Callable, Sequence

from deap import base, creator, gp
import numpy as np


PRIMITIVE_SET_VERSION = 1
EPSILON = 1e-6
SCORE_LIMIT = 10.0
PRIMITIVE_ARITIES = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "pdiv": 2,
    "minimum": 2,
    "maximum": 2,
    "negative": 1,
    "absolute": 1,
}


def _clip(value: float) -> float:
    value = float(value)
    if math.isnan(value):
        return 0.0
    if value == math.inf:
        return SCORE_LIMIT
    if value == -math.inf:
        return -SCORE_LIMIT
    return min(max(value, -SCORE_LIMIT), SCORE_LIMIT)


def add(x: float, y: float) -> float:
    return _clip(float(x) + float(y))


def sub(x: float, y: float) -> float:
    return _clip(float(x) - float(y))


def mul(x: float, y: float) -> float:
    return _clip(float(x) * float(y))


def pdiv(x: float, y: float) -> float:
    return float(x) if abs(float(y)) < EPSILON else _clip(float(x) / float(y))


def minimum(x: float, y: float) -> float:
    return min(float(x), float(y))


def maximum(x: float, y: float) -> float:
    return max(float(x), float(y))


def negative(x: float) -> float:
    return _clip(-float(x))


def absolute(x: float) -> float:
    return _clip(abs(float(x)))


def ephemeral_constant() -> float:
    return round(random.uniform(-1.0, 1.0), 6)


def ensure_deap_types():
    """Register creator types once, including across repeated test/workflow calls."""
    if not hasattr(creator, "GPArchitectureFitness"):
        creator.create(
            "GPArchitectureFitness",
            base.Fitness,
            weights=(-1.0, -1.0),
        )
    if not hasattr(creator, "GPArchitectureIndividual"):
        creator.create(
            "GPArchitectureIndividual",
            gp.PrimitiveTree,
            fitness=creator.GPArchitectureFitness,
        )
    return creator.GPArchitectureFitness, creator.GPArchitectureIndividual


def build_primitive_set(feature_names: Sequence[str]) -> gp.PrimitiveSet:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique.")
    reserved = set(PRIMITIVE_ARITIES) | {"erc"}
    if any(name in reserved or not name.isidentifier() for name in names):
        raise ValueError("feature names must be identifiers distinct from primitives.")
    pset = gp.PrimitiveSet("MAIN", len(names))
    pset.addPrimitive(add, 2, name="add")
    pset.addPrimitive(sub, 2, name="sub")
    pset.addPrimitive(mul, 2, name="mul")
    pset.addPrimitive(pdiv, 2, name="pdiv")
    pset.addPrimitive(minimum, 2, name="minimum")
    pset.addPrimitive(maximum, 2, name="maximum")
    pset.addPrimitive(negative, 1, name="negative")
    pset.addPrimitive(absolute, 1, name="absolute")
    pset.addEphemeralConstant("erc", ephemeral_constant)
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(names)})
    return pset


def individual_from_expression(expression: str, pset: gp.PrimitiveSet):
    ensure_deap_types()
    tree = gp.PrimitiveTree.from_string(str(expression), pset)
    return creator.GPArchitectureIndividual(tree)


def compile_individual(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
) -> Callable[..., float]:
    return gp.compile(expr=individual, pset=pset)


def score_feature_matrix(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    feature_matrix: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(pset.arguments):
        raise ValueError("feature_matrix has the wrong shape.")
    function = compile_individual(individual, pset)
    scores = []
    for row in matrix:
        try:
            value = float(function(*row.tolist()))
        except (ArithmeticError, OverflowError, ValueError):
            value = float("inf")
        scores.append(value if math.isfinite(value) else float("inf"))
    return np.asarray(scores, dtype=np.float64)


def tree_within_limits(
    individual: gp.PrimitiveTree,
    *,
    max_height: int = 6,
    max_nodes: int = 40,
) -> bool:
    return int(individual.height) <= int(max_height) and len(individual) <= int(max_nodes)
