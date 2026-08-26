import csv
import json
from pathlib import Path
import tempfile

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.artifact import sha256_file
from sosrl.gp.primitives import build_primitive_set, individual_from_expression
from sosrl.rl.agent import DQNAgent
from sosrl.rl.config import DQNConfig
from sosrl.rules.scheduling import Rule
from sosrl.workflows import evaluation
from sosrl.workflows.fixed_rule_scheduler import train_fixed_rule_scheduler
from sosrl.workflows.gp_architecture import ScenarioEvaluator, save_scenario_manifest
from sosrl.workflows.round1_study import _scenario_environment
from sosrl.workflows.scheduler_backends import (
    RuleDQNSchedulerBackend,
    run_rule_dqn_episode,
    scheduler_parameter_hash,
)
from sosrl.gp.provider import FixedArchitectureProvider


CATEGORIES = (
    "feasible_suboptimal",
    "capacity_tight",
    "missing_capability",
    "redundant_overbudget",
)


def _mission():
    func_type = int(env.FULL_SOS[0].func_type)
    return [
        syn.Task(
            0,
            "task",
            [syn.Operation(0, "op", func_type, 10, 0)],
            due_time=100,
        )
    ]


def _manifest(path: Path, split: str):
    scenarios = [
        evaluation.scenario_payload(
            index,
            (env.FULL_SOS[0],),
            _mission(),
            category=category,
            budget=8000.0,
            refund_rate=0.8,
            split=split,
            static_feasible_architecture=(env.FULL_SOS[0],),
        )
        for index, category in enumerate(CATEGORIES)
    ]
    return save_scenario_manifest(
        path,
        split=split,
        seed=7,
        scenarios=scenarios,
    )


def _config():
    return DQNConfig(
        episodes=0,
        scenario_pool_size=0,
        scenario_order="sequential",
        shared_mission=False,
        rule_set="standard",
        gamma=0.99,
        lr=1e-3,
        batch_size=1,
        buffer_size=10,
        min_buffer_size=1,
        target_update_interval=2,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        hidden_dim=16,
        seed=7,
        device="cpu",
    )


def _random_checkpoint(path: Path, scenario: dict):
    mission_env = _scenario_environment(scenario, "fixed")
    agent = DQNAgent(
        mission_env.schedule_observation().shape[0], Rule.RULE_NUM, _config()
    )
    return agent.save_checkpoint(path)


def test_fixed_rule_training_resumes_and_selects_without_architecture_changes():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        train = _manifest(root / "train.json", "b_train")
        validation = _manifest(root / "validation.json", "b_validation")
        common = {
            "train_manifest": train,
            "validation_manifest": validation,
            "config": _config(),
            "max_env_steps": 4,
            "checkpoint_steps": (0, 2, 4),
        }
        train_fixed_rule_scheduler(
            output_dir=root / "run",
            stop_after_checkpoint=2,
            **common,
        )
        outputs = train_fixed_rule_scheduler(output_dir=root / "run", **common)

        manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
        with Path(outputs["training_history"]).open(
            encoding="utf-8", newline=""
        ) as file:
            history = list(csv.DictReader(file))

        assert manifest["status"] == "complete"
        assert manifest["selection"]["checkpoint_sha256"]
        assert Path(outputs["selected_checkpoint"]).is_file()
        assert all(int(row["architecture_changes"]) == 0 for row in history)
        assert all(int(row["invalid_action_count"]) == 0 for row in history)
        assert all(
            int(row["provider_invariant_violations"]) == 0 for row in history
        )


def test_rule_backend_is_frozen_and_single_multi_process_outcomes_match():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        manifest_path = _manifest(root / "train.json", "g_train")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario = manifest["scenarios"][0]
        checkpoint = _random_checkpoint(root / "rule.pt", scenario)
        checkpoint_hash = sha256_file(checkpoint)
        backend = RuleDQNSchedulerBackend.from_checkpoint(
            checkpoint, device="cpu"
        )
        before_hash = scheduler_parameter_hash(backend.agent)
        mission_env = _scenario_environment(scenario, "fixed")
        result = run_rule_dqn_episode(
            mission_env,
            FixedArchitectureProvider(),
            backend.agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
            store_experience=False,
        )

        assert result["success"]
        assert mission_env.architecture_change_count == 0
        assert result["invalid_action_count"] == 0
        assert scheduler_parameter_hash(backend.agent) == before_hash
        assert len(backend.agent.replay) == 0

        feature_preset = "system_delta"
        pset = build_primitive_set(feature_names_for_preset(feature_preset))
        individuals = [
            individual_from_expression("0.0", pset),
            individual_from_expression("progress", pset),
        ]
        single = ScenarioEvaluator(
            backend=backend,
            scheduler_backend="rule-dqn",
            scheduler_checkpoint=checkpoint,
            feature_preset=feature_preset,
            workers=1,
        )
        multi = ScenarioEvaluator(
            backend=backend,
            scheduler_backend="rule-dqn",
            scheduler_checkpoint=checkpoint,
            feature_preset=feature_preset,
            workers=2,
        )
        try:
            single_outcomes = single.evaluate_population(individuals, [scenario])
            multi_outcomes = multi.evaluate_population(individuals, [scenario])
        finally:
            single.close()
            multi.close()

        assert single_outcomes == multi_outcomes
        assert scheduler_parameter_hash(backend.agent) == before_hash
        assert sha256_file(checkpoint) == checkpoint_hash
