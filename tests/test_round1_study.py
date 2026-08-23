from pathlib import Path
import tempfile

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.rl.config import BranchingDQNConfig
from sosrl.rl.checkpoint import load_branching_checkpoint
from sosrl.workflows import evaluation
from sosrl.workflows.gp_architecture import save_scenario_manifest
from sosrl.workflows.round1_study import (
    PROVIDER_KINDS,
    bdqn_hyperparameter_matrix,
    ensure_initial_checkpoint,
    generate_round1_scenarios,
    gp_discovery_matrix,
    provider_cross_matrix_jobs,
    train_bdqn_provider_cell,
)


def one_operation_mission():
    func_type = int(env.FULL_SOS[0].func_type)
    return [
        syn.Task(
            0,
            "task",
            [syn.Operation(0, "op", func_type, 10, 0)],
            due_time=100,
        )
    ]


def tiny_manifest(path: Path, split: str):
    mission = one_operation_mission()
    scenarios = [
        evaluation.scenario_payload(
            index,
            (env.FULL_SOS[0],),
            mission,
            category=category,
            budget=8000.0,
            refund_rate=0.8,
            split=split,
            static_feasible_architecture=(env.FULL_SOS[0],),
        )
        for index, category in enumerate(
            (
                "feasible_suboptimal",
                "capacity_tight",
                "missing_capability",
                "redundant_overbudget",
            )
        )
    ]
    return save_scenario_manifest(
        path,
        split=split,
        seed=1,
        scenarios=scenarios,
    )


def test_round1_scenarios_are_disjoint_and_register_static_architecture():
    with tempfile.TemporaryDirectory() as temp_dir:
        paths = generate_round1_scenarios(
            temp_dir,
            base_seed=91,
            b_train_size=4,
            b_validation_size=4,
            g_train_size=4,
            g_validation_size=4,
            test_iid_size=4,
            test_ood_size=4,
        )
        registry = Path(paths["registry"]).read_text(encoding="utf-8")

    assert '"test_locked": true' in registry
    assert len(paths) == 7


def test_preregistered_matrices_are_exact_and_deduplicated():
    hyperparameters = bdqn_hyperparameter_matrix()
    gp_matrix = gp_discovery_matrix()

    assert [row["name"] for row in hyperparameters] == [
        "H0",
        "H1",
        "H2",
        "H3",
        "H4",
        "H5",
        "H6",
        "H7",
        "H8",
        "H9",
        "H10",
    ]
    assert len(gp_matrix) == len(
        {(row["population_size"], row["generations"]) for row in gp_matrix}
    )
    overlap = [
        row
        for row in gp_matrix
        if row["population_size"] == 120 and row["generations"] == 50
    ]
    assert overlap[0]["families"] == "equal_budget+generation_axis+population_axis"


def test_cross_matrix_has_every_training_and_evaluation_provider():
    checkpoints = {
        provider: {1: f"{provider}.pt"} for provider in PROVIDER_KINDS
    }
    jobs = provider_cross_matrix_jobs(
        seeds=[1], checkpoint_by_training_provider=checkpoints
    )

    assert len(jobs) == 9
    assert {
        (job["training_provider"], job["evaluation_provider"]) for job in jobs
    } == {(left, right) for left in PROVIDER_KINDS for right in PROVIDER_KINDS}


def test_fixed_provider_cell_uses_shared_initial_checkpoint_and_validates():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        train = tiny_manifest(root / "train.json", "b_train")
        validation = tiny_manifest(root / "validation.json", "b_validation")
        config = BranchingDQNConfig(
            episodes=10,
            max_env_steps=1,
            scenario_pool_size=4,
            batch_size=2,
            buffer_size=10,
            min_buffer_size=10,
            seed=3,
            device="cpu",
        )
        initial = ensure_initial_checkpoint(root, seed=3, config=config)
        outputs = train_bdqn_provider_cell(
            output_dir=root / "cell",
            provider_kind="fixed",
            source_checkpoint=initial,
            train_manifest=train,
            validation_manifest=validation,
            config=config,
            checkpoint_steps=(0, 1),
        )

        assert Path(outputs["manifest"]).is_file()
        assert Path(outputs["summary"]).is_file()
        assert "fixed_3_1" in Path(outputs["summary"]).read_text(encoding="utf-8")


def test_interrupted_cell_resume_matches_continuous_training():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        train = tiny_manifest(root / "train.json", "b_train")
        validation = tiny_manifest(root / "validation.json", "b_validation")
        config = BranchingDQNConfig(
            episodes=10,
            max_env_steps=4,
            scenario_pool_size=4,
            batch_size=1,
            buffer_size=10,
            min_buffer_size=1,
            target_update_interval=2,
            seed=7,
            device="cpu",
        )
        initial = ensure_initial_checkpoint(root, seed=7, config=config)
        common = {
            "provider_kind": "fixed",
            "source_checkpoint": initial,
            "train_manifest": train,
            "validation_manifest": validation,
            "config": config,
            "checkpoint_steps": (0, 2, 4),
        }
        train_bdqn_provider_cell(
            output_dir=root / "resumed",
            stop_after_checkpoint=2,
            **common,
        )
        resumed_outputs = train_bdqn_provider_cell(
            output_dir=root / "resumed",
            **common,
        )
        continuous_outputs = train_bdqn_provider_cell(
            output_dir=root / "continuous",
            **common,
        )

        resumed, _ = load_branching_checkpoint(
            root / "resumed" / "checkpoints" / "checkpoint_4.pt",
            device="cpu",
            load_optimizer=True,
        )
        continuous, _ = load_branching_checkpoint(
            root / "continuous" / "checkpoints" / "checkpoint_4.pt",
            device="cpu",
            load_optimizer=True,
        )
        for name, tensor in resumed.q_net.state_dict().items():
            assert tensor.equal(continuous.q_net.state_dict()[name])
        assert resumed.learn_step == continuous.learn_step
        assert Path(resumed_outputs["summary"]).read_text(encoding="utf-8") == Path(
            continuous_outputs["summary"]
        ).read_text(encoding="utf-8")
