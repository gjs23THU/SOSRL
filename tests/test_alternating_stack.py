import csv
import json
from pathlib import Path
import tempfile

from sosrl import environment as env
from sosrl import domain as syn
from sosrl.gp.artifact import create_policy_artifact, save_gp_policy
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.primitives import build_primitive_set, individual_from_expression
from sosrl.rl.branching import BranchingDQNAgent
from sosrl.rl.checkpoint import load_branching_checkpoint
from sosrl.rl.config import BranchingDQNConfig
from sosrl.workflows import evaluation
from sosrl.workflows.alternating_stack import (
    _bdqn_config,
    generate_alternation_evaluation_scenarios,
    initial_bdqn_convergence_state,
    run_gp_bdqn_alternation,
    update_bdqn_convergence,
)
from sosrl.workflows.gp_architecture import (
    SCENARIO_CATEGORIES,
    load_scenario_manifest,
    save_scenario_manifest,
)


def _summary(step, *, failure=0, j=2.0, makespan=100.0):
    return {
        "target_environment_steps": step,
        "failure_count": failure,
        "mean_j": j,
        "mean_success_makespan": makespan,
        "invalid_action_count": 0,
        "provider_invariant_violations": 0,
    }


def test_bdqn_convergence_requires_confirmation_and_resets_on_improvement():
    state = initial_bdqn_convergence_state()
    confirmed_values = []
    values = (
        (_summary(5000, j=2.0), _summary(10000, j=1.99)),
        (_summary(10000, j=1.99), _summary(15000, j=1.985)),
        (_summary(15000, j=1.985), _summary(20000, j=1.98)),
    )
    for previous, current in values:
        state, confirmed = update_bdqn_convergence(
            state,
            previous=previous,
            current=current,
            delta_j_ci=(-0.01, 0.01),
        )
        confirmed_values.append(confirmed)
    assert confirmed_values == [False, False, True]
    assert state["provisional_step"] == 15000
    assert state["confirmed_step"] == 20000

    reset, confirmed = update_bdqn_convergence(
        state,
        previous=_summary(20000, j=1.98),
        current=_summary(25000, j=1.8),
        delta_j_ci=(-0.3, -0.1),
    )
    assert not confirmed
    assert reset["stable_windows"] == 0
    assert reset["provisional_step"] is None


def test_bdqn_confirmation_window_does_not_require_ci_to_cross_zero():
    state = initial_bdqn_convergence_state()
    for previous, current in (
        (_summary(5000), _summary(10000, j=1.99)),
        (_summary(10000, j=1.99), _summary(15000, j=1.985)),
    ):
        state, confirmed = update_bdqn_convergence(
            state,
            previous=previous,
            current=current,
            delta_j_ci=(-0.01, 0.01),
        )
        assert not confirmed

    state, confirmed = update_bdqn_convergence(
        state,
        previous=_summary(15000, j=1.985),
        current=_summary(20000, j=1.98),
        delta_j_ci=(-0.02, -0.001),
    )

    assert confirmed
    assert state["confirmed_step"] == 20000
    assert state["comparisons"][-1]["confirmation_window"]


def test_each_bdqn_round_restarts_learning_rate_at_one_e_minus_four():
    first = _bdqn_config(seed=4, max_env_steps=40000, device="cpu")
    second = _bdqn_config(seed=7, max_env_steps=40000, device="cpu")

    assert first.learning_rate_at_episode(0) == 1e-4
    assert second.learning_rate_at_episode(0) == 1e-4
    assert first.learning_rate_at_episode(1) == 1e-4 * 0.9975
    assert first.learning_rate_at_episode(100000) == 1e-5


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


def _save_split(path: Path, split: str, per_category: int):
    rows = []
    index = 0
    for category in SCENARIO_CATEGORIES:
        for _ in range(per_category):
            rows.append(_scenario(index, category, split))
            index += 1
    return save_scenario_manifest(
        path,
        split=split,
        seed=100 + per_category,
        scenarios=rows,
    )


def test_alternation_scenario_generator_is_disjoint_from_existing_manifest():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        existing = _save_split(root / "existing.json", "existing", 1)
        outputs = generate_alternation_evaluation_scenarios(
            root / "generated",
            existing_manifests=[existing],
            gate_iid_size=4,
            gate_ood_size=4,
            final_iid_size=4,
            final_ood_size=4,
            gate_iid_seed=20261010,
            gate_ood_seed=20261011,
            final_iid_seed=20261012,
            final_ood_seed=20261013,
        )
        hashes = set(
            row["scenario_hash"]
            for row in load_scenario_manifest(existing)["scenarios"]
        )
        for name in ("gate_iid", "gate_ood", "final_iid", "final_ood"):
            generated = load_scenario_manifest(outputs[name])
            generated_hashes = {row["scenario_hash"] for row in generated["scenarios"]}
            assert not hashes & generated_hashes
            hashes.update(generated_hashes)
        assert load_scenario_manifest(outputs["gate_iid"])["seed"] == 20261010
        assert load_scenario_manifest(outputs["gate_ood"])["seed"] == 20261011
        assert load_scenario_manifest(outputs["final_iid"])["seed"] == 20261012
        assert load_scenario_manifest(outputs["final_ood"])["seed"] == 20261013


def test_full_two_round_smoke_and_resume_preserve_learning_rate_reset():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        scenario_dir = root / "scenarios"
        scenario_dir.mkdir()
        _save_split(scenario_dir / "train.json", "train", 64)
        _save_split(scenario_dir / "validation.json", "validation", 1)
        gate_iid = _save_split(root / "gate_iid.json", "gate_iid", 1)
        gate_ood = _save_split(root / "gate_ood.json", "gate_ood", 1)
        final_iid = _save_split(root / "final_iid.json", "final_iid", 1)
        final_ood = _save_split(root / "final_ood.json", "final_ood", 1)
        b0_agent = BranchingDQNAgent(BranchingDQNConfig(device="cpu"))
        b0_agent.learn_step = 73
        b0 = b0_agent.save_checkpoint(root / "b0.pt")
        names = feature_names_for_preset("system_delta")
        individual = individual_from_expression(
            "0.0", build_primitive_set(names)
        )
        g0 = save_gp_policy(
            root / "g0.json",
            create_policy_artifact(individual, feature_preset="system_delta"),
        )
        arguments = dict(
            base_gp_policy=g0,
            base_scheduler_checkpoint=b0,
            scenario_dir=scenario_dir,
            gate_iid_manifest=gate_iid,
            gate_ood_manifest=gate_ood,
            final_iid_manifest=final_iid,
            final_ood_manifest=final_ood,
            output_dir=root / "alternation",
            gp_population_size=8,
            gp_max_generations=2,
            gp_runs=1,
            gp_workers=1,
            gp_min_generations=2,
            gp_convergence_interval=1,
            bdqn_max_env_steps=4,
            bdqn_checkpoint_interval=2,
            bdqn_min_convergence_steps=4,
            bdqn_parallel_jobs=1,
            gp_device="cpu",
            bdqn_device="cpu",
        )

        outputs = run_gp_bdqn_alternation(**arguments)
        resumed = run_gp_bdqn_alternation(**arguments)

        assert outputs.keys() == resumed.keys()
        manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert set(manifest["lineage"]) == {"S0", "SG1", "S1", "SG2", "S2"}
        final_summary = json.loads(
            outputs["final_summary"].read_text(encoding="utf-8")
        )
        assert set(final_summary) == {"S0", "SG1", "S1", "SG2", "S2"}
        with (
            root / "alternation" / "crossplay" / "summary.csv"
        ).open(newline="", encoding="utf-8-sig") as file:
            assert len(list(csv.DictReader(file))) == 9
        for round_index in (1, 2):
            convergence = json.loads(
                (
                    root
                    / "alternation"
                    / f"bdqn_round_{round_index}"
                    / "convergence.json"
                ).read_text(encoding="utf-8")
            )
            assert convergence["status"] == "max_env_steps_reached"
            assert convergence["actual_environment_steps"] == 4
            for seed in ((4, 5, 6) if round_index == 1 else (7, 8, 9)):
                history_path = (
                    root
                    / "alternation"
                    / f"bdqn_round_{round_index}"
                    / "cells"
                    / f"seed_{seed}"
                    / "training_history.csv"
                )
                with history_path.open(newline="", encoding="utf-8-sig") as file:
                    history = list(csv.DictReader(file))
                first_row = history[0]
                assert float(first_row["learning_rate"]) == 1e-4
                assert [int(row["episode"]) for row in history] == list(
                    range(len(history))
                )
                zero_checkpoint = (
                    history_path.parent / "checkpoints" / "checkpoint_0.pt"
                )
                zero_agent, _ = load_branching_checkpoint(
                    zero_checkpoint, device="cpu", load_optimizer=True
                )
                assert zero_agent.learn_step == 0
                assert not zero_agent.optimizer.state
