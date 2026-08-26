"""Scenario generation, evolution, selection, and evaluation for direct GP."""

from __future__ import annotations

from dataclasses import asdict
import copy
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .. import domain as syn
from .. import environment as env
from ..baselines.hysteretic_capacity import (
    HystereticCapacityConfig,
    HystereticCapacityProvider,
)
from ..gp.artifact import (
    create_policy_artifact,
    load_gp_policy,
    save_gp_policy,
    sha256_file,
    simplify_expression,
)
from ..gp.config import GPArchitectureConfig
from ..gp.architecture import (
    legal_architecture_actions,
)
from ..gp.evolution import (
    EpisodeOutcome,
    aggregate_fitness,
    evolve_architecture_policy,
)
from ..gp.features import (
    architecture_feature_matrix,
    build_architecture_feature_context,
    feature_names_for_preset,
)
from ..gp.primitives import (
    build_primitive_set,
    compile_individual,
    individual_from_expression,
)
from ..gp.provider import (
    FixedArchitectureProvider,
    GPArchitectureProvider,
    ManualRuleDQNProvider,
    RandomConcreteArchitectureProvider,
)
from ..rl.agent import ArchitectureDQNAgent
from ..workflows import scheduler
from . import evaluation
from .branching import branching_episode_row
from .hierarchical import AdaptiveScenarioPool
from .scheduler_backends import (
    SCHEDULER_BACKEND_KINDS,
    SchedulerBackend,
    load_scheduler_backend,
    scheduler_parameter_hash,
)


GP_SCENARIO_SCHEMA_VERSION = 2
SUPPORTED_GP_SCENARIO_SCHEMA_VERSIONS = (1, 2)
SCENARIO_CATEGORIES = AdaptiveScenarioPool.CATEGORIES
DEFAULT_REFUND_RATE = 0.8
_WORKER_BACKEND = None
_WORKER_PRESET = None
_WORKER_PSET = None
_WORKER_FUNCTIONS: dict[str, Any] = {}


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_scenario_manifest(
    path: str | Path,
    *,
    split: str,
    seed: int,
    scenarios: Sequence[dict[str, Any]],
) -> Path:
    payload = {
        "schema_version": GP_SCENARIO_SCHEMA_VERSION,
        "split": str(split),
        "seed": int(seed),
        "size": len(scenarios),
        "categories": list(SCENARIO_CATEGORIES),
        "scenarios": list(scenarios),
    }
    payload["manifest_hash"] = _canonical_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_scenario_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.pop("manifest_hash", None)
    actual = _canonical_hash(payload)
    payload["manifest_hash"] = expected
    if payload.get("schema_version") not in SUPPORTED_GP_SCENARIO_SCHEMA_VERSIONS:
        raise ValueError("unsupported GP scenario manifest schema.")
    if expected != actual:
        raise ValueError("GP scenario manifest hash mismatch.")
    if int(payload.get("size", -1)) != len(payload.get("scenarios", [])):
        raise ValueError("GP scenario manifest size mismatch.")
    for scenario in payload["scenarios"]:
        evaluation.verify_scenario_payload(scenario)
    return payload


def _make_scenario(
    scenario_idx: int,
    *,
    split: str,
    category: str,
    budget: float,
    mission_config: dict[str, Any],
) -> dict[str, Any]:
    mission = syn.build_mission_from_config(mission_config)
    sampler = scheduler.ScenarioPool(size=0, cost_limit=budget)
    feasible = tuple(sampler.sample_arch(mission))
    builder = object.__new__(AdaptiveScenarioPool)
    architecture = builder._make_initial_architecture(
        feasible, mission, category
    )
    return evaluation.scenario_payload(
        scenario_idx,
        architecture,
        mission,
        category=category,
        budget=budget,
        refund_rate=DEFAULT_REFUND_RATE,
        split=split,
        static_feasible_architecture=feasible,
    )


def _generate_split(
    *,
    split: str,
    size: int,
    seed: int,
    ood: bool,
) -> list[dict[str, Any]]:
    if int(size) <= 0 or int(size) % len(SCENARIO_CATEGORIES) != 0:
        raise ValueError("each GP scenario split size must be positive and divisible by four.")
    scheduler.set_seed(seed)
    mission_config = json.loads(json.dumps(syn.CONFIG))
    if ood:
        mission_config["total_task"] = 40
        mission_config["op_per_task"] = 4
        mission_config["op_duration"] = [15, 35]
    scenarios = []
    for index in range(size):
        category = SCENARIO_CATEGORIES[index % len(SCENARIO_CATEGORIES)]
        budget = (
            6400.0 if ood and index < size // 2 else 9600.0
            if ood
            else 8000.0
        )
        scenarios.append(
            _make_scenario(
                index,
                split=split,
                category=category,
                budget=budget,
                mission_config=mission_config,
            )
        )
    return scenarios


def generate_gp_scenario_manifests(
    output_dir: str | Path,
    *,
    base_seed: int = 20260820,
    train_size: int = 256,
    validation_size: int = 128,
    test_size: int = 500,
    ood_size: int = 200,
) -> dict[str, Path]:
    destination = Path(output_dir)
    specifications = (
        ("train", train_size, base_seed, False),
        ("validation", validation_size, base_seed + 1, False),
        ("test_iid", test_size, base_seed + 2, False),
        ("test_ood", ood_size, base_seed + 3, True),
    )
    paths = {}
    for split, size, seed, ood in specifications:
        scenarios = _generate_split(
            split=split,
            size=int(size),
            seed=int(seed),
            ood=ood,
        )
        paths[split] = save_scenario_manifest(
            destination / f"{split}.json",
            split=split,
            seed=seed,
            scenarios=scenarios,
        )
    return paths


def stratified_generation_batch(
    scenarios: Sequence[dict[str, Any]],
    *,
    run_seed: int,
    generation: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size % len(SCENARIO_CATEGORIES) != 0:
        raise ValueError("training batch size must be divisible by four.")
    per_category = batch_size // len(SCENARIO_CATEGORIES)
    rng = random.Random(int(run_seed) + int(generation))
    selected = []
    for category in SCENARIO_CATEGORIES:
        group = [item for item in scenarios if item.get("category") == category]
        if len(group) < per_category:
            raise ValueError(f"not enough training scenarios in category {category}.")
        selected.extend(rng.sample(group, per_category))
    return selected


def fixed_anchor_scenarios(
    scenarios: Sequence[dict[str, Any]],
    anchor_size: int,
) -> list[dict[str, Any]]:
    if anchor_size % len(SCENARIO_CATEGORIES) != 0:
        raise ValueError("anchor size must be divisible by four.")
    per_category = anchor_size // len(SCENARIO_CATEGORIES)
    anchor = []
    for category in SCENARIO_CATEGORIES:
        group = [item for item in scenarios if item.get("category") == category]
        if len(group) < per_category:
            raise ValueError(f"not enough anchor scenarios in category {category}.")
        anchor.extend(group[:per_category])
    return anchor


def freeze_branching_agent(agent) -> None:
    for network in (agent.q_net, agent.target_net):
        network.eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)


def branching_parameter_hash(agent) -> str:
    digest = hashlib.sha256()
    for network_name in ("q_net", "target_net"):
        state = getattr(agent, network_name).state_dict()
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(network_name.encode())
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _episode_outcome(
    backend: SchedulerBackend,
    score_function,
    feature_preset: str,
    scenario: dict[str, Any],
) -> EpisodeOutcome:
    architecture, mission = evaluation.scenario_from_payload(scenario)
    mission_env = env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=float(scenario.get("budget", 8000.0)),
        refund_rate=float(scenario.get("refund_rate", DEFAULT_REFUND_RATE)),
    )
    provider = GPArchitectureProvider(
        score_function,
        feature_preset=feature_preset,
    )
    result = backend.run_episode(mission_env, provider)
    return EpisodeOutcome(
        success=bool(result["success"]),
        dead_end=bool(result["dead_end"]),
        completed_operations=int(np.sum(mission_env.state.task_op_idx)),
        total_operations=mission_env.T * mission_env.O,
        makespan=float(mission_env.state.current_makespan),
        scale=float(mission_env.state.M),
        final_net_cost=float(mission_env.net_cost),
        peak_net_cost=float(mission_env.peak_net_cost),
        budget=float(mission_env.budget),
        architecture_changes=int(mission_env.architecture_change_count),
    )


def _worker_initialize(
    backend_kind: str,
    checkpoint_path: str,
    feature_preset: str,
) -> None:
    global _WORKER_BACKEND, _WORKER_PRESET, _WORKER_PSET, _WORKER_FUNCTIONS
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_BACKEND = load_scheduler_backend(
        backend_kind,
        checkpoint_path,
        device="cpu",
    )
    _WORKER_PRESET = feature_preset
    _WORKER_PSET = build_primitive_set(feature_names_for_preset(feature_preset))
    _WORKER_FUNCTIONS = {}


def _worker_evaluate(task):
    expression, scenario = task
    function = _WORKER_FUNCTIONS.get(expression)
    if function is None:
        individual = individual_from_expression(expression, _WORKER_PSET)
        function = compile_individual(individual, _WORKER_PSET)
        _WORKER_FUNCTIONS[expression] = function
    return _episode_outcome(
        _WORKER_BACKEND,
        function,
        _WORKER_PRESET,
        scenario,
    )


def _outcome_from_environment(mission_env, *, success: bool, dead_end: bool):
    return EpisodeOutcome(
        success=bool(success),
        dead_end=bool(dead_end),
        completed_operations=int(np.sum(mission_env.state.task_op_idx)),
        total_operations=mission_env.T * mission_env.O,
        makespan=float(mission_env.state.current_makespan),
        scale=float(mission_env.state.M),
        final_net_cost=float(mission_env.net_cost),
        peak_net_cost=float(mission_env.peak_net_cost),
        budget=float(mission_env.budget),
        architecture_changes=int(mission_env.architecture_change_count),
    )


def _apply_prevalidated_architecture_action(mission_env, action):
    """Apply a candidate generated for the environment's current version."""
    if action.kind == "keep":
        return {"valid": True, "kind": "keep", "cost_delta": 0.0, "refund": 0.0}
    if action.kind == "add":
        return mission_env.add_system(int(action.new_system))
    if action.kind == "remove":
        return mission_env.remove_system(int(action.old_system))
    return mission_env.replace_system(
        int(action.old_system), int(action.new_system)
    )


def _shared_population_scenario_outcomes(
    backend: SchedulerBackend,
    expressions: Sequence[str],
    score_functions: Sequence[Any],
    feature_preset: str,
    scenario: dict[str, Any],
) -> list[EpisodeOutcome]:
    """Evaluate policies together until their concrete action trajectories split."""
    architecture, mission = evaluation.scenario_from_payload(scenario)
    root_env = env.MissionEnv(
        architecture,
        mission,
        adaptive=True,
        budget=float(scenario.get("budget", 8000.0)),
        refund_rate=float(scenario.get("refund_rate", DEFAULT_REFUND_RATE)),
    )
    outcomes: list[EpisodeOutcome | None] = [None] * len(expressions)
    groups = [(root_env, list(range(len(expressions))), 0)]
    max_steps = root_env.T * root_env.O + root_env.N

    while groups:
        pending = []
        for mission_env, policy_indices, step_count in groups:
            if step_count >= max_steps:
                outcome = _outcome_from_environment(
                    mission_env, success=False, dead_end=True
                )
                for policy_idx in policy_indices:
                    outcomes[policy_idx] = outcome
                continue
            candidates = legal_architecture_actions(mission_env)
            if not candidates:
                outcome = _outcome_from_environment(
                    mission_env, success=False, dead_end=True
                )
                for policy_idx in policy_indices:
                    outcomes[policy_idx] = outcome
                continue

            context = build_architecture_feature_context(mission_env)
            feature_matrix = architecture_feature_matrix(
                mission_env,
                candidates,
                feature_preset,
                context=context,
            )
            selected_groups: dict[Any, list[int]] = {}
            candidate_rows = [row.tolist() for row in feature_matrix]
            for policy_idx in policy_indices:
                function = score_functions[policy_idx]
                ranked = []
                for values, action in zip(candidate_rows, candidates, strict=True):
                    try:
                        score = float(function(*values))
                    except (ArithmeticError, OverflowError, ValueError):
                        score = float("inf")
                    if not np.isfinite(score):
                        score = float("inf")
                    ranked.append((score, action.tie_break_key, action))
                selected = min(ranked, key=lambda item: (item[0], item[1]))[2]
                selected_groups.setdefault(selected, []).append(policy_idx)

            branches = list(selected_groups.items())
            branch_envs = [mission_env]
            branch_envs.extend(copy.deepcopy(mission_env) for _ in branches[1:])
            for (action, branch_indices), branch_env in zip(
                branches, branch_envs, strict=True
            ):
                result = _apply_prevalidated_architecture_action(branch_env, action)
                if not result.get("valid", False):
                    raise RuntimeError("shared GP evaluator selected an illegal action.")
                if not backend.has_feasible_action(branch_env):
                    outcome = _outcome_from_environment(
                        branch_env, success=False, dead_end=True
                    )
                    for policy_idx in branch_indices:
                        outcomes[policy_idx] = outcome
                    continue
                pending.append((branch_env, branch_indices, step_count))

        if not pending:
            groups = []
            continue
        environment_actions = backend.select_environment_actions(
            [item[0] for item in pending]
        )
        next_groups = []
        for pending_item, environment_action in zip(
            pending, environment_actions, strict=True
        ):
            branch_env, branch_indices, step_count = pending_item
            _, _, terminated, _, info = branch_env.step(environment_action)
            if terminated:
                outcome = _outcome_from_environment(
                    branch_env,
                    success=bool(info.get("success", False)),
                    dead_end=bool(info.get("dead_end", False)),
                )
                for policy_idx in branch_indices:
                    outcomes[policy_idx] = outcome
            else:
                next_groups.append((branch_env, branch_indices, step_count + 1))
        groups = next_groups

    if any(outcome is None for outcome in outcomes):
        raise RuntimeError("shared GP evaluator did not finish every policy.")
    return list(outcomes)


def _worker_evaluate_population(task):
    expressions, scenario = task
    functions = []
    for expression in expressions:
        function = _WORKER_FUNCTIONS.get(expression)
        if function is None:
            individual = individual_from_expression(expression, _WORKER_PSET)
            function = compile_individual(individual, _WORKER_PSET)
            _WORKER_FUNCTIONS[expression] = function
        functions.append(function)
    return _shared_population_scenario_outcomes(
        _WORKER_BACKEND,
        expressions,
        functions,
        _WORKER_PRESET,
        scenario,
    )


class ScenarioEvaluator:
    """Evaluate individuals with an interchangeable frozen scheduler backend."""

    def __init__(
        self,
        *,
        backend: SchedulerBackend,
        scheduler_backend: str,
        scheduler_checkpoint: str | Path,
        feature_preset: str,
        workers: int,
    ):
        self.backend = backend
        self.feature_preset = feature_preset
        self.pset = build_primitive_set(feature_names_for_preset(feature_preset))
        self.outcome_cache: dict[tuple[str, str], EpisodeOutcome] = {}
        self.pool = None
        if int(workers) > 1:
            if backend.device.type != "cpu":
                raise ValueError("workers>1 requires a CPU scheduler backend.")
            context = mp.get_context("spawn")
            self.pool = context.Pool(
                processes=int(workers),
                initializer=_worker_initialize,
                initargs=(
                    str(scheduler_backend),
                    str(Path(scheduler_checkpoint).resolve()),
                    feature_preset,
                ),
            )

    def evaluate(self, individual, scenarios):
        expression = str(individual)
        cached = [
            self.outcome_cache.get((expression, scenario["scenario_hash"]))
            for scenario in scenarios
        ]
        missing_indices = [
            index for index, outcome in enumerate(cached) if outcome is None
        ]
        if not missing_indices:
            return cached
        if self.pool is not None:
            computed = self.pool.map(
                _worker_evaluate,
                [(expression, scenarios[index]) for index in missing_indices],
            )
        else:
            function = compile_individual(individual, self.pset)
            computed = [
                _episode_outcome(
                    self.backend,
                    function,
                    self.feature_preset,
                    scenarios[index],
                )
                for index in missing_indices
            ]
        for index, outcome in zip(missing_indices, computed, strict=True):
            scenario_hash = scenarios[index]["scenario_hash"]
            self.outcome_cache[(expression, scenario_hash)] = outcome
            cached[index] = outcome
        return cached

    def evaluate_population(self, individuals, scenarios):
        expressions = [str(individual) for individual in individuals]
        results = {expression: [None] * len(scenarios) for expression in expressions}
        tasks = []
        task_metadata = []
        for scenario_index, scenario in enumerate(scenarios):
            missing_expressions = []
            missing_positions = []
            for expression_index, expression in enumerate(expressions):
                cached = self.outcome_cache.get(
                    (expression, scenario["scenario_hash"])
                )
                if cached is None:
                    missing_expressions.append(expression)
                    missing_positions.append(expression_index)
                else:
                    results[expression][scenario_index] = cached
            if missing_expressions:
                tasks.append((missing_expressions, scenario))
                task_metadata.append(
                    (scenario_index, missing_expressions, missing_positions)
                )
        if tasks:
            if self.pool is not None:
                computed_tasks = self.pool.map(
                    _worker_evaluate_population, tasks
                )
            else:
                computed_tasks = []
                for expressions_for_scenario, scenario in tasks:
                    individuals_for_scenario = [
                        individual_from_expression(expression, self.pset)
                        for expression in expressions_for_scenario
                    ]
                    functions = [
                        compile_individual(individual, self.pset)
                        for individual in individuals_for_scenario
                    ]
                    computed_tasks.append(
                        _shared_population_scenario_outcomes(
                            self.backend,
                            expressions_for_scenario,
                            functions,
                            self.feature_preset,
                            scenario,
                        )
                    )
            for metadata, outcomes in zip(
                task_metadata, computed_tasks, strict=True
            ):
                scenario_index, missing_expressions, _ = metadata
                scenario_hash = scenarios[scenario_index]["scenario_hash"]
                for expression, outcome in zip(
                    missing_expressions, outcomes, strict=True
                ):
                    self.outcome_cache[(expression, scenario_hash)] = outcome
                    results[expression][scenario_index] = outcome
        if any(
            outcome is None
            for expression_results in results.values()
            for outcome in expression_results
        ):
            raise RuntimeError("population evaluator produced incomplete outcomes.")
        return results

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def validate_policy_action_equivalence(
    backend: SchedulerBackend,
    original_score_function,
    deployed_score_function,
    feature_preset: str,
    scenarios: Sequence[dict[str, Any]],
) -> int:
    """Check both policies at every validation decision point on one trajectory."""
    checked = 0

    class ComparingProvider:
        def __init__(self):
            self.original = GPArchitectureProvider(
                original_score_function, feature_preset=feature_preset
            )
            self.deployed = GPArchitectureProvider(
                deployed_score_function, feature_preset=feature_preset
            )

        def act(self, mission_env):
            nonlocal checked
            original = self.original.decide(mission_env)
            deployed = self.deployed.decide(mission_env)
            if original.valid != deployed.valid or (
                original.valid and original.action != deployed.action
            ):
                raise RuntimeError(
                    "deployed GP policy changed a validation concrete action."
                )
            checked += 1
            return self.original.act(mission_env)

    provider = ComparingProvider()
    for scenario in scenarios:
        architecture, mission = evaluation.scenario_from_payload(scenario)
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=float(scenario.get("budget", 8000.0)),
            refund_rate=float(scenario.get("refund_rate", DEFAULT_REFUND_RATE)),
        )
        backend.run_episode(mission_env, provider)
    return checked


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train_gp_architecture(
    *,
    scheduler_checkpoint: str | Path,
    scheduler_backend: str = "branching-dqn",
    scenario_dir: str | Path,
    output_dir: str | Path,
    config: GPArchitectureConfig,
    device: str = "cpu",
    resume_state: str | Path | None = None,
) -> dict[str, Path]:
    """Evolve independent runs, validate all candidates, and lock one JSON rule."""
    if config.workers > 1 and torch.device(device).type != "cpu":
        raise ValueError("multi-process GP evolution requires --device cpu.")
    scheduler_checkpoint = Path(scheduler_checkpoint)
    scenario_dir = Path(scenario_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = load_scenario_manifest(scenario_dir / "train.json")
    validation_manifest = load_scenario_manifest(scenario_dir / "validation.json")
    train_scenarios = train_manifest["scenarios"]
    validation_scenarios = validation_manifest["scenarios"]
    anchor = fixed_anchor_scenarios(train_scenarios, config.anchor_size)

    if scheduler_backend not in SCHEDULER_BACKEND_KINDS:
        raise ValueError(f"unknown scheduler backend: {scheduler_backend!r}")
    backend = load_scheduler_backend(
        scheduler_backend,
        scheduler_checkpoint,
        device=device,
    )
    before_hash = scheduler_parameter_hash(backend.agent)
    evaluator = ScenarioEvaluator(
        backend=backend,
        scheduler_backend=scheduler_backend,
        scheduler_checkpoint=scheduler_checkpoint,
        feature_preset=config.feature_set,
        workers=config.workers,
    )
    generation_history = []
    anchor_history = []
    candidate_rows = []
    try:
        for run_index in range(config.independent_runs):
            evaluator.outcome_cache.clear()
            run_seed = config.base_seed + 1000 + run_index
            run_started = time.perf_counter()
            previous_report = run_started
            print(
                json.dumps(
                    {
                        "event": "gp_run_start",
                        "run_index": run_index,
                        "run_seed": run_seed,
                        "runs": config.independent_runs,
                    }
                ),
                flush=True,
            )

            def batch_sampler(seed, generation):
                return stratified_generation_batch(
                    train_scenarios,
                    run_seed=seed,
                    generation=generation,
                    batch_size=config.train_batch_size,
                )

            run_dir = output_dir / f"run_{run_index:02d}"

            def report_generation(row):
                nonlocal previous_report
                reported_at = time.perf_counter()
                print(
                    json.dumps(
                        {
                            "event": "gp_generation_complete",
                            "run_index": run_index,
                            "run_elapsed_seconds": reported_at - run_started,
                            "since_previous_report_seconds": reported_at
                            - previous_report,
                            **row,
                        }
                    ),
                    flush=True,
                )
                previous_report = reported_at

            result = evolve_architecture_policy(
                feature_names=feature_names_for_preset(config.feature_set),
                config=config,
                run_seed=run_seed,
                batch_sampler=batch_sampler,
                anchor_scenarios=anchor,
                individual_evaluator=evaluator.evaluate,
                checkpoint_path=run_dir / "evolution_state.pkl",
                resume_state=(
                    resume_state
                    if run_index == 0 and resume_state is not None
                    else (run_dir / "evolution_state.pkl")
                    if (run_dir / "evolution_state.pkl").is_file()
                    else None
                ),
                progress_callback=report_generation,
                population_evaluator=evaluator.evaluate_population,
            )
            generation_history.extend(result.generation_history)
            anchor_history.extend(result.anchor_history)
            candidate_rows.extend(result.candidates)

        evaluator.outcome_cache.clear()
        expressions = sorted({row["expression"] for row in candidate_rows})
        validation_rows = []
        pset = build_primitive_set(feature_names_for_preset(config.feature_set))
        individuals = {
            expression: individual_from_expression(expression, pset)
            for expression in expressions
        }
        validation_batch_size = 200
        for validation_start in range(0, len(expressions), validation_batch_size):
            batch_expressions = expressions[
                validation_start : validation_start + validation_batch_size
            ]
            batch_individuals = [
                individuals[expression] for expression in batch_expressions
            ]
            outcome_map = evaluator.evaluate_population(
                batch_individuals, validation_scenarios
            )
            for expression, individual in zip(
                batch_expressions, batch_individuals, strict=True
            ):
                fitness = aggregate_fitness(
                    outcome_map[expression],
                    node_count=len(individual),
                    parsimony_coefficient=config.parsimony_coefficient,
                )
                validation_rows.append(
                    {
                        "expression": expression,
                        "failure_rate": fitness.failure_rate,
                        "regularized_j": fitness.regularized_j,
                        "raw_mean_j": fitness.raw_mean_j,
                        "node_count": len(individual),
                        "height": int(individual.height),
                    }
                )
            print(
                json.dumps(
                    {
                        "event": "gp_validation_progress",
                        "completed": min(
                            validation_start + validation_batch_size,
                            len(expressions),
                        ),
                        "total": len(expressions),
                    }
                ),
                flush=True,
            )
    finally:
        evaluator.close()

    minimum_failure = min(row["failure_rate"] for row in validation_rows)
    failure_best = [
        row for row in validation_rows if row["failure_rate"] == minimum_failure
    ]
    best_raw = min(row["raw_mean_j"] for row in failure_best)
    threshold = best_raw * 1.01
    eligible = [row for row in failure_best if row["raw_mean_j"] <= threshold]
    winner = min(
        eligible,
        key=lambda row: (row["node_count"], row["height"], row["expression"]),
    )
    selected = individuals[winner["expression"]]
    selected_function = compile_individual(selected, pset)
    simplified_expression = simplify_expression(
        str(selected), feature_names_for_preset(config.feature_set)
    )
    deployment_individual = individual_from_expression(
        simplified_expression, pset
    )
    simplification_used = len(deployment_individual) < len(selected)
    if not simplification_used:
        deployment_individual = selected
    scheduler_provenance = backend.provenance()
    scheduler_file_hash = str(scheduler_provenance["checkpoint_sha256"])
    policy_path = output_dir / "gp_policy.json"

    def write_deployment(individual, simplified):
        artifact = create_policy_artifact(
            individual,
            feature_preset=config.feature_set,
            validation_fitness={
                "failure_rate": winner["failure_rate"],
                "regularized_j": winner["regularized_j"],
                "raw_mean_j": winner["raw_mean_j"],
            },
            evolution_config={
                **asdict(config),
                "simplification_attempted": True,
                "simplification_used": bool(simplified),
            },
            training_scheduler=scheduler_provenance,
            bdqn_checkpoint_sha256=(
                scheduler_file_hash
                if scheduler_backend == "branching-dqn"
                else ""
            ),
        )
        save_gp_policy(policy_path, artifact)
        return load_gp_policy(policy_path)

    loaded = write_deployment(deployment_individual, simplification_used)
    try:
        equivalence_points = validate_policy_action_equivalence(
            backend,
            selected_function,
            loaded.score_function,
            config.feature_set,
            validation_scenarios,
        )
    except RuntimeError:
        if not simplification_used:
            raise
        deployment_individual = selected
        simplification_used = False
        loaded = write_deployment(deployment_individual, False)
        equivalence_points = validate_policy_action_equivalence(
            backend,
            selected_function,
            loaded.score_function,
            config.feature_set,
            validation_scenarios,
        )

    after_hash = scheduler_parameter_hash(backend.agent)
    if after_hash != before_hash:
        raise RuntimeError("frozen scheduler parameters changed during GP evolution.")
    scheduler_file_hash_after = sha256_file(scheduler_checkpoint)
    if scheduler_file_hash_after != scheduler_file_hash:
        raise RuntimeError("frozen scheduler checkpoint changed during GP evolution.")
    _write_csv(output_dir / "generation_history.csv", generation_history)
    _write_csv(output_dir / "anchor_history.csv", anchor_history)
    _write_jsonl(output_dir / "candidate_rules.jsonl", candidate_rows)
    _write_csv(output_dir / "validation_results.csv", validation_rows)

    scenario_hashes = {}
    for name in ("train", "validation", "test_iid", "test_ood"):
        path = scenario_dir / f"{name}.json"
        if path.exists():
            scenario_hashes[name] = load_scenario_manifest(path)["manifest_hash"]
    stack = {
        "schema_version": 2,
        "gp_policy": str(policy_path.resolve()),
        "gp_policy_sha256": sha256_file(policy_path),
        "training_scheduler": scheduler_provenance,
        "scheduler_checkpoint_sha256_before": scheduler_file_hash,
        "scheduler_checkpoint_sha256_after": scheduler_file_hash_after,
        "scheduler_parameter_sha256_before": before_hash,
        "scheduler_parameter_sha256_after": after_hash,
        "environment_config_sha256": sha256_file(syn.CONFIG_PATH),
        "scenario_manifest_hashes": scenario_hashes,
        "code_commit": _git_commit(),
        "base_seed": config.base_seed,
        "validation_action_equivalence_points": int(equivalence_points),
        "simplification_used": bool(simplification_used),
    }
    if scheduler_backend == "branching-dqn":
        stack.update(
            {
                "branching_scheduler": str(scheduler_checkpoint.resolve()),
                "branching_scheduler_sha256": scheduler_file_hash,
                "branching_parameter_sha256_before": before_hash,
                "branching_parameter_sha256_after": after_hash,
            }
        )
    stack_path = output_dir / "gp_stack_manifest.json"
    stack_path.write_text(
        json.dumps(stack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "gp_policy": policy_path,
        "gp_stack_manifest": stack_path,
        "generation_history": output_dir / "generation_history.csv",
        "anchor_history": output_dir / "anchor_history.csv",
        "candidate_rules": output_dir / "candidate_rules.jsonl",
        "validation_results": output_dir / "validation_results.csv",
    }


def _evaluate_provider_records(
    provider,
    scheduler_backend: SchedulerBackend,
    scenarios: Sequence[dict[str, Any]],
    *,
    model: str,
    collect_schedule: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    schedules = []
    for episode, scenario in enumerate(scenarios):
        if model == "fixed":
            architecture, mission = evaluation.static_scenario_from_payload(scenario)
        else:
            architecture, mission = evaluation.scenario_from_payload(scenario)
        mission_env = env.MissionEnv(
            architecture,
            mission,
            adaptive=True,
            budget=float(scenario.get("budget", 8000.0)),
            refund_rate=float(scenario.get("refund_rate", DEFAULT_REFUND_RATE)),
        )
        result = scheduler_backend.run_episode(
            mission_env,
            provider,
            measure_inference=True,
        )
        row = branching_episode_row(
            episode,
            scenario.get("category", "evaluation"),
            mission_env,
            result,
            0.0,
        )
        row.update(
            {
                "model": model,
                "scenario_hash": scenario["scenario_hash"],
                "budget": float(mission_env.budget),
            }
        )
        rows.append(row)
        if collect_schedule:
            for task_idx in range(mission_env.T):
                for op_idx in range(mission_env.O):
                    sys_idx = int(mission_env.state.op_assign_sys[task_idx, op_idx])
                    if sys_idx < 0:
                        continue
                    schedules.append(
                        {
                            "model": model,
                            "scenario_hash": scenario["scenario_hash"],
                            "task_idx": task_idx,
                            "op_idx": op_idx,
                            "system_idx": sys_idx,
                            "start_time": float(
                                mission_env.state.op_start_time[task_idx, op_idx]
                            ),
                            "finish_time": float(
                                mission_env.state.op_finish_time[task_idx, op_idx]
                            ),
                        }
                    )
    return rows, schedules


def _bootstrap_interval(values, *, seed=20260820, samples=2000):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return (None, None)
    rng = np.random.default_rng(seed)
    estimates = np.mean(
        rng.choice(array, size=(samples, array.size), replace=True), axis=1
    )
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def paired_gp_comparisons(rows: Sequence[dict[str, Any]], reference="gp"):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], {})[row["scenario_hash"]] = row
    if reference not in grouped:
        raise ValueError("paired comparison reference is missing.")
    comparisons = []
    continuous = (
        "makespan",
        "final_net_cost",
        "peak_net_cost",
        "gross_charge",
        "total_refund",
        "architecture_changes",
    )
    for model, model_rows in grouped.items():
        if model == reference:
            continue
        hashes = sorted(grouped[reference])
        if hashes != sorted(model_rows):
            raise ValueError("models were not evaluated on matching scenario hashes.")
        row = {"reference_model": reference, "candidate_model": model, "paired_scenarios": len(hashes)}
        for metric in continuous:
            differences = [
                float(model_rows[key][metric]) - float(grouped[reference][key][metric])
                for key in hashes
            ]
            low, high = _bootstrap_interval(differences)
            row[f"mean_delta_{metric}"] = float(np.mean(differences))
            row[f"ci95_low_delta_{metric}"] = low
            row[f"ci95_high_delta_{metric}"] = high
        for metric in ("success", "dead_end"):
            differences = [
                float(bool(model_rows[key][metric]))
                - float(bool(grouped[reference][key][metric]))
                for key in hashes
            ]
            low, high = _bootstrap_interval(differences)
            row[f"paired_proportion_delta_{metric}"] = float(np.mean(differences))
            row[f"ci95_low_delta_{metric}"] = low
            row[f"ci95_high_delta_{metric}"] = high
        comparisons.append(row)
    return comparisons


def summarize_gp_evaluation(rows: Sequence[dict[str, Any]]):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    summary = []
    for model, model_rows in grouped.items():
        summary.append(
            {
                "model": model,
                "episodes": len(model_rows),
                "success_rate": float(np.mean([row["success"] for row in model_rows])),
                "dead_end_rate": float(np.mean([row["dead_end"] for row in model_rows])),
                "mean_makespan": float(np.mean([row["makespan"] for row in model_rows])),
                "mean_final_net_cost": float(np.mean([row["final_net_cost"] for row in model_rows])),
                "mean_peak_net_cost": float(np.mean([row["peak_net_cost"] for row in model_rows])),
                "ever_budget_violation_rate": float(np.mean([row["ever_over_budget"] for row in model_rows])),
                "final_budget_violation_rate": float(np.mean([row["final_over_budget"] for row in model_rows])),
                "mean_architecture_changes": float(np.mean([row["architecture_changes"] for row in model_rows])),
            }
        )
    return summary


def evaluate_gp_stack(
    *,
    gp_policy: str | Path,
    scheduler_checkpoint: str | Path,
    scheduler_backend: str = "branching-dqn",
    scenario_manifest: str | Path,
    output_dir: str | Path,
    baselines: Sequence[str] = ("fixed", "ss", "random_concrete", "gp"),
    manual_architecture_checkpoint: str | Path | None = None,
    ss_config: HystereticCapacityConfig | None = None,
    device: str = "cpu",
    collect_schedule: bool = False,
) -> dict[str, Path]:
    manifest = load_scenario_manifest(scenario_manifest)
    scenarios = manifest["scenarios"]
    backend = load_scheduler_backend(
        scheduler_backend,
        scheduler_checkpoint,
        device=device,
    )
    loaded_policy = load_gp_policy(gp_policy)
    training_scheduler = dict(loaded_policy.artifact.training_scheduler)
    actual_scheduler = backend.provenance()
    checkpoint_binding = (
        "matched"
        if training_scheduler.get("checkpoint_sha256")
        == actual_scheduler.get("checkpoint_sha256")
        else "diagnostic_crossed"
    )
    providers = {}
    if "fixed" in baselines:
        providers["fixed"] = FixedArchitectureProvider()
    if "ss" in baselines:
        providers["ss"] = HystereticCapacityProvider(ss_config)
    if "random_concrete" in baselines:
        providers["random_concrete"] = RandomConcreteArchitectureProvider(seed=20260820)
    if "gp" in baselines:
        providers["gp"] = GPArchitectureProvider.from_artifact(loaded_policy)
    if "manual6_dqn" in baselines:
        if manual_architecture_checkpoint is None:
            raise ValueError("manual6_dqn requires an architecture checkpoint.")
        manual_agent, _ = ArchitectureDQNAgent.load_checkpoint(
            manual_architecture_checkpoint,
            device=device,
            load_optimizer=False,
        )
        providers["manual6_dqn"] = ManualRuleDQNProvider(manual_agent)
    unknown = set(baselines) - set(providers)
    if unknown:
        raise ValueError(f"unknown GP evaluation baselines: {sorted(unknown)}")

    rows = []
    schedules = []
    for label in baselines:
        model_rows, model_schedules = _evaluate_provider_records(
            providers[label],
            backend,
            scenarios,
            model=label,
            collect_schedule=collect_schedule,
        )
        for row in model_rows:
            row["gp_tree_node_count"] = loaded_policy.artifact.node_count if label == "gp" else 0
            row["gp_tree_height"] = loaded_policy.artifact.height if label == "gp" else 0
            row["scheduler_backend"] = str(scheduler_backend)
            row["scheduler_checkpoint_sha256"] = actual_scheduler[
                "checkpoint_sha256"
            ]
            row["gp_training_scheduler_backend"] = training_scheduler.get(
                "kind", "unknown"
            )
            row["checkpoint_binding"] = checkpoint_binding
        rows.extend(model_rows)
        schedules.extend(model_schedules)
    comparisons = paired_gp_comparisons(rows, reference="gp") if "gp" in baselines else []
    summary = summarize_gp_evaluation(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.csv"
    comparisons_path = output_dir / "paired_comparisons.csv"
    summary_path = output_dir / "evaluation_summary.csv"
    _write_csv(results_path, rows)
    _write_csv(comparisons_path, comparisons)
    _write_csv(summary_path, summary)
    outputs = {
        "results": results_path,
        "summary": summary_path,
        "paired_comparisons": comparisons_path,
    }
    if collect_schedule:
        schedule_path = output_dir / "schedules.csv"
        _write_csv(schedule_path, schedules)
        outputs["schedules"] = schedule_path
    return outputs
