"""Safe JSON deployment artifacts for evolved GP architecture policies."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from deap import gp

from .. import environment as env_module
from .features import ARCH_FEATURE_SCHEMA_VERSION, feature_names_for_preset
from .primitives import (
    PRIMITIVE_ARITIES,
    PRIMITIVE_SET_VERSION,
    absolute,
    add,
    build_primitive_set,
    compile_individual,
    individual_from_expression,
    maximum,
    minimum,
    mul,
    negative,
    pdiv,
    sub,
)


GP_POLICY_SCHEMA_VERSION = 1
SCORE_DIRECTION = "minimize"
TIE_BREAK_RULE = (
    "score, changed_system_count, kind_keep_add_remove_replace, "
    "old_system, new_system"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def system_pool_hash() -> str:
    payload = [
        {
            "index": int(system.index),
            "name": system.name,
            "cost": int(system.cost),
            "available_from": int(system.available_from),
            "available_until": int(system.available_until),
            "func_type": int(system.func_type),
        }
        for system in env_module.FULL_SOS
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_expression_node(
    node: ast.AST,
    feature_names: frozenset[str],
) -> None:
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in PRIMITIVE_ARITIES:
            raise ValueError("GP expression contains an unknown primitive.")
        if node.keywords or len(node.args) != PRIMITIVE_ARITIES[node.func.id]:
            raise ValueError("GP primitive arity does not match the registry.")
        for child in node.args:
            _validate_expression_node(child, feature_names)
        return
    if isinstance(node, ast.Name):
        if node.id not in feature_names:
            raise ValueError("GP expression contains an unknown feature.")
        return
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numeric GP constants are allowed.")
        if not math.isfinite(float(node.value)):
            raise ValueError("GP constants must be finite.")
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        if not isinstance(node.operand, ast.Constant):
            raise ValueError("unary syntax is only allowed for numeric constants.")
        _validate_expression_node(node.operand, feature_names)
        return
    raise ValueError("unsupported syntax in GP expression.")


def validate_expression(expression: str, feature_names: Sequence[str]) -> ast.Expression:
    try:
        parsed = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        raise ValueError("invalid GP expression syntax.") from exc
    _validate_expression_node(parsed.body, frozenset(feature_names))
    return parsed


_PRIMITIVE_FUNCTIONS = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "pdiv": pdiv,
    "minimum": minimum,
    "maximum": maximum,
    "negative": negative,
    "absolute": absolute,
}


def simplify_expression(expression: str, feature_names: Sequence[str]) -> str:
    """Constant-fold protected primitive subtrees without algebraic assumptions."""
    parsed = validate_expression(expression, feature_names)

    def fold(node: ast.AST) -> ast.AST:
        if not isinstance(node, ast.Call):
            return node
        arguments = [fold(child) for child in node.args]
        rebuilt = ast.Call(func=node.func, args=arguments, keywords=[])
        constant_values = []
        for child in arguments:
            try:
                value = ast.literal_eval(child)
            except (ValueError, TypeError):
                return rebuilt
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return rebuilt
            constant_values.append(float(value))
        value = _PRIMITIVE_FUNCTIONS[node.func.id](*constant_values)
        return ast.Constant(value=float(value))

    folded = ast.fix_missing_locations(fold(parsed.body))
    simplified = ast.unparse(folded)
    validate_expression(simplified, feature_names)
    return simplified


def _prefix_nodes(node: ast.AST) -> list[dict[str, Any]]:
    if isinstance(node, ast.Call):
        result = [{"kind": "primitive", "name": node.func.id, "arity": len(node.args)}]
        for child in node.args:
            result.extend(_prefix_nodes(child))
        return result
    if isinstance(node, ast.Name):
        return [{"kind": "feature", "name": node.id, "arity": 0}]
    value = ast.literal_eval(node)
    return [{"kind": "constant", "value": round(float(value), 6), "arity": 0}]


@dataclass(frozen=True)
class GPPolicyArtifact:
    schema_version: int
    feature_schema_version: int
    feature_preset: str
    feature_names: list[str]
    primitive_set_version: int
    expression: str
    prefix_tree: list[dict[str, Any]]
    node_count: int
    height: int
    score_direction: str
    tie_break_rule: str
    validation_fitness: dict[str, float]
    evolution_config: dict[str, Any]
    system_pool_hash: str
    bdqn_checkpoint_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadedGPPolicy:
    artifact: GPPolicyArtifact
    primitive_set: gp.PrimitiveSet
    individual: gp.PrimitiveTree
    score_function: Any


def create_policy_artifact(
    individual: gp.PrimitiveTree,
    *,
    feature_preset: str,
    validation_fitness: Mapping[str, float] | None = None,
    evolution_config: Mapping[str, Any] | None = None,
    bdqn_checkpoint_sha256: str = "",
) -> GPPolicyArtifact:
    feature_names = feature_names_for_preset(feature_preset)
    expression = str(individual)
    parsed = validate_expression(expression, feature_names)
    return GPPolicyArtifact(
        schema_version=GP_POLICY_SCHEMA_VERSION,
        feature_schema_version=ARCH_FEATURE_SCHEMA_VERSION,
        feature_preset=feature_preset,
        feature_names=list(feature_names),
        primitive_set_version=PRIMITIVE_SET_VERSION,
        expression=expression,
        prefix_tree=_prefix_nodes(parsed.body),
        node_count=len(individual),
        height=int(individual.height),
        score_direction=SCORE_DIRECTION,
        tie_break_rule=TIE_BREAK_RULE,
        validation_fitness={
            str(key): float(value) for key, value in (validation_fitness or {}).items()
        },
        evolution_config=dict(evolution_config or {}),
        system_pool_hash=system_pool_hash(),
        bdqn_checkpoint_sha256=str(bdqn_checkpoint_sha256),
    )


def save_gp_policy(path: str | Path, artifact: GPPolicyArtifact) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(artifact.to_dict(), file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(destination)
    return destination


def load_gp_policy(path: str | Path, *, verify_system_pool: bool = True) -> LoadedGPPolicy:
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    artifact = GPPolicyArtifact(**payload)
    if artifact.schema_version != GP_POLICY_SCHEMA_VERSION:
        raise ValueError("unsupported GP policy schema version.")
    if artifact.feature_schema_version != ARCH_FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported architecture feature schema version.")
    if artifact.primitive_set_version != PRIMITIVE_SET_VERSION:
        raise ValueError("unsupported GP primitive-set version.")
    registered_features = feature_names_for_preset(artifact.feature_preset)
    if tuple(artifact.feature_names) != tuple(registered_features):
        raise ValueError("GP policy feature registry does not match its preset.")
    if artifact.score_direction != SCORE_DIRECTION or artifact.tie_break_rule != TIE_BREAK_RULE:
        raise ValueError("GP policy decision semantics do not match this runtime.")
    if verify_system_pool and artifact.system_pool_hash != system_pool_hash():
        raise ValueError("GP policy was evolved for a different system pool.")
    validate_expression(artifact.expression, registered_features)
    pset = build_primitive_set(registered_features)
    individual = individual_from_expression(artifact.expression, pset)
    if len(individual) != artifact.node_count or int(individual.height) != artifact.height:
        raise ValueError("GP policy tree metadata is inconsistent.")
    expected_prefix = _prefix_nodes(ast.parse(artifact.expression, mode="eval").body)
    if expected_prefix != artifact.prefix_tree:
        raise ValueError("GP policy prefix tree is inconsistent with its expression.")
    return LoadedGPPolicy(
        artifact=artifact,
        primitive_set=pset,
        individual=individual,
        score_function=compile_individual(individual, pset),
    )
