import csv
import json
from pathlib import Path
import tempfile

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.gp.artifact import create_policy_artifact, save_gp_policy, sha256_file
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.primitives import build_primitive_set, individual_from_expression
from sosrl.rl.branching import BranchingDQNAgent
from sosrl.rl.config import BranchingDQNConfig
from sosrl.workflows import evaluation
from sosrl.workflows.gp_architecture import (
    SCENARIO_CATEGORIES,
    load_scenario_manifest,
    save_scenario_manifest,
)
from sosrl.workflows.gp_bdqn_tuning import (
    _bdqn_plateau_comparison,
    _initial_bdqn_plateau,
    _update_bdqn_plateau,
    create_gp_bdqn_tuning_spec,
    load_tuning_spec,
    run_gp_bdqn_tuning,
)


def _scenario(index: int, category: str, split: str):
    system = env.FULL_SOS[0]
    mission = [
        syn.Task(
            index,
            f"task-{index}",
            [syn.Operation(0, "op", int(system.func_type), 10, 0)],
            due_time=100,
        )
    ]
    return evaluation.scenario_payload(
        index,
        (system,),
        mission,
        category=category,
        budget=8000.0,
        refund_rate=0.8,
        split=split,
        static_feasible_architecture=(system,),
    )


def _save_split(path: Path, split: str):
    return save_scenario_manifest(
        path,
        split=split,
        seed=71,
        scenarios=[
            _scenario(index, category, split)
            for index, category in enumerate(SCENARIO_CATEGORIES)
        ],
    )


def _write_tiny_spec(root: Path) -> Path:
    b_scenarios = root / "b"
    g_scenarios = root / "g"
    b_scenarios.mkdir()
    g_scenarios.mkdir()
    for directory, prefix in ((b_scenarios, "b"), (g_scenarios, "g")):
        _save_split(directory / "train.json", f"{prefix}_train")
        _save_split(directory / "validation.json", f"{prefix}_validation")

    # R0 is reused when the paired four-step packages tie, so only its frozen
    # identity is needed by this deliberately tiny workflow.
    rule_checkpoint = root / "rule_r0.pt"
    rule_checkpoint.write_bytes(b"frozen-rule-checkpoint")
    names = feature_names_for_preset("system_delta")
    individual = individual_from_expression("0.0", build_primitive_set(names))
    g0 = save_gp_policy(
        root / "g0.json",
        create_policy_artifact(individual, feature_preset="system_delta"),
    )
    b0_checkpoints = []
    for seed in (1, 2, 3):
        agent = BranchingDQNAgent(
            BranchingDQNConfig(seed=seed, device="cpu")
        )
        b0_checkpoints.append(agent.save_checkpoint(root / f"b0_seed{seed}.pt"))

    spec_path = create_gp_bdqn_tuning_spec(
        root / "tuning_spec.json",
        b_scenario_dir=b_scenarios,
        g_scenario_dir=g_scenarios,
        base_rule_checkpoint=rule_checkpoint,
        base_gp_policy=g0,
        base_bdqn_checkpoints=b0_checkpoints,
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["rule_lr"].update(
        {"max_env_steps": 4, "checkpoint_steps": [0, 2, 4]}
    )
    spec["gp"].update(
        {
            "population_size": 8,
            "screen_generations": 1,
            "max_generations": 2,
            "runs": 3,
            "workers": 1,
            "train_batch_size": 4,
            "anchor_size": 4,
            "anchor_interval": 1,
            "anchor_top_k": 2,
            "min_generations": 1,
            "validation_candidates_per_run": 2,
            "base_seed": 103,
        }
    )
    spec["bdqn"].update(
        {
            "screen_steps": 2,
            "max_env_steps": 4,
            "checkpoint_interval": 2,
            "parallel_jobs": 1,
            "batch_size": 1,
            "buffer_size": 32,
            "min_buffer_size": 1,
        }
    )
    spec["evaluation"].update(
        {
            "screen_validation_size": 4,
            "bootstrap_samples": 100,
            "rule_gate_iid": {"size": 4, "seed": 20261001, "ood": False},
            "rule_gate_ood": {"size": 4, "seed": 20261002, "ood": False},
            "gate_iid": {"size": 4, "seed": 20261003, "ood": False},
            "gate_ood": {"size": 4, "seed": 20261004, "ood": False},
            "final_iid": {"size": 4, "seed": 20261005, "ood": False},
            "final_ood": {"size": 4, "seed": 20261006, "ood": False},
        }
    )
    spec_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return spec_path


def _seed_tiny_evaluation_registry(output_root: Path) -> None:
    scenario_root = output_root / "scenarios"
    scenario_root.mkdir(parents=True)
    generated = {}
    for index, name in enumerate(
        (
            "rule_gate_iid",
            "rule_gate_ood",
            "gate_iid",
            "gate_ood",
            "final_iid",
            "final_ood",
        )
    ):
        path = _save_split(scenario_root / f"{name}.json", f"{name}_{index}")
        generated[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    (scenario_root / "scenario_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registered": [],
                "generated": generated,
                "all_hashes_unique": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_tiny_pareto_tuning_workflow_is_resumable_and_reports_repeats():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        spec_path = _write_tiny_spec(root)
        spec = load_tuning_spec(spec_path)
        _seed_tiny_evaluation_registry(root / "tuning")
        partial = run_gp_bdqn_tuning(
            spec_path=spec_path,
            output_dir=root / "tuning",
            rule_device="cpu",
            bdqn_device="cpu",
            stop_after_stage="bdqn_screen",
        )
        assert json.loads(partial["manifest"].read_text(encoding="utf-8"))[
            "status"
        ] == "running"
        outputs = run_gp_bdqn_tuning(
            spec_path=spec_path,
            output_dir=root / "tuning",
            rule_device="cpu",
            bdqn_device="cpu",
        )
        resumed = run_gp_bdqn_tuning(
            spec_path=spec_path,
            output_dir=root / "tuning",
            rule_device="cpu",
            bdqn_device="cpu",
        )

        assert outputs.keys() == resumed.keys()
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["spec_sha256"] == sha256_file(spec_path)
        assert set(manifest["completed_stages"]) == {
            "scenarios",
            "rule_lr",
            "s0_prefix",
            "gp_screen",
            "bdqn_screen",
            "crossplay",
            "g2",
            "b2",
            "final",
        }
        assert spec["rule_lr"]["configs"][0]["name"] == "R0"

        rule_selection = json.loads(
            outputs["rule_lr_selection"].read_text(encoding="utf-8")
        )
        assert rule_selection["decision"]["winner"] == "R0"
        assert rule_selection["performance_statement"] == (
            "aggregate means and 95% confidence intervals across three independent repeats"
        )
        assert rule_selection["selected_deployment_seed"] == 4
        assert all(
            record["shared_initial_weights"]
            and record["shared_scenario_order"]
            and record["shared_epsilon_random_stream"]
            for record in rule_selection["matched_pair_audit"].values()
        )

        pareto = json.loads(
            outputs["pareto_front"].read_text(encoding="utf-8")
        )
        assert len(pareto["all_stacks"]) == 9
        assert pareto["selection_basis"] == (
            "aggregate means and paired 95% confidence intervals across preregistered repeats"
        )

        with outputs["final_metrics"].open(newline="", encoding="utf-8-sig") as file:
            metrics = list(csv.DictReader(file))
        assert {row["stage"] for row in metrics} == {
            "S0",
            "SG1",
            "S1",
            "SG2",
            "S2",
        }
        assert all(int(row["repeat_count"]) == 3 for row in metrics)
        assert all(
            row["mean_success_makespan_ci_low"]
            and row["mean_success_makespan_ci_high"]
            for row in metrics
        )

        registry = json.loads(
            (root / "tuning" / "scenarios" / "scenario_registry.json").read_text(
                encoding="utf-8"
            )
        )
        assert registry["all_hashes_unique"]
        all_hashes = []
        for record in registry["generated"].values():
            all_hashes.extend(
                row["scenario_hash"]
                for row in load_scenario_manifest(record["path"])["scenarios"]
            )
        assert len(all_hashes) == len(set(all_hashes))


def _plateau_rows(makespan: float, final_cost: float = 3000.0):
    return [
        {
            "repeat": repeat,
            "split": "validation",
            "scenario_hash": f"scenario-{scenario}",
            "success": True,
            "makespan": makespan,
            "final_net_cost": final_cost,
            "peak_net_cost": final_cost,
            "failure_aware_j": 1.0,
            "ever_over_budget": False,
            "invalid_action_count": 0,
            "provider_invariant_violations": 0,
            "architecture_changes": 0,
        }
        for repeat in (1, 2, 3)
        for scenario in range(4)
    ]


def test_bdqn_plateau_is_relative_to_parent_and_requires_full_confirmation():
    parent = _plateau_rows(100.0)
    unchanged = _plateau_rows(100.0)
    state, candidate = _update_bdqn_plateau(
        _initial_bdqn_plateau(),
        previous_rows=parent,
        current_rows=unchanged,
        current_step=15000,
        min_steps=20000,
        seed=31,
    )
    assert not candidate
    state, candidate = _update_bdqn_plateau(
        state,
        previous_rows=parent,
        current_rows=unchanged,
        current_step=20000,
        min_steps=20000,
        seed=32,
    )
    assert candidate
    assert state["provisional_step"] == 20000
    assert state["confirmed_step"] is None
    confirmation = _bdqn_plateau_comparison(
        parent, unchanged, samples=100, seed=33
    )
    assert confirmation["stable"]

    improved, candidate = _update_bdqn_plateau(
        state,
        previous_rows=parent,
        current_rows=_plateau_rows(95.0),
        current_step=25000,
        min_steps=20000,
        seed=34,
    )
    assert not candidate
    assert improved["stable_windows"] == 0
    assert improved["provisional_step"] is None
