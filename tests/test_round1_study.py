import csv
import json
from pathlib import Path
import tempfile

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.baselines.hysteretic_capacity import HystereticCapacityConfig
from sosrl.rl.config import BranchingDQNConfig
from sosrl.rl.checkpoint import load_branching_checkpoint
from sosrl.gp.artifact import sha256_file
from sosrl.workflows import evaluation
from sosrl.workflows.gp_architecture import save_scenario_manifest
from sosrl.workflows.round1_study import (
    LEGACY_PROVIDER_KINDS,
    PROVIDER_KINDS,
    ROUND1_MANIFEST_SCHEMA_VERSION,
    _convergence_cell_directories,
    _load_study,
    _provider_input_hash,
    _study_provider_kinds,
    baseline_bdqn_config,
    bdqn_hyperparameter_matrix,
    ensure_initial_checkpoint,
    generate_round1_scenarios,
    gp_discovery_matrix,
    initialize_round1_study,
    migration_path_jobs,
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

    assert len(jobs) == 16
    assert {
        (job["training_provider"], job["evaluation_provider"]) for job in jobs
    } == {(left, right) for left in PROVIDER_KINDS for right in PROVIDER_KINDS}


def test_study_schema_controls_provider_membership():
    assert _study_provider_kinds({"schema_version": 1}) == LEGACY_PROVIDER_KINDS
    assert _study_provider_kinds(
        {
            "schema_version": ROUND1_MANIFEST_SCHEMA_VERSION,
            "providers": list(PROVIDER_KINDS),
        }
    ) == PROVIDER_KINDS
    try:
        _study_provider_kinds(
            {
                "schema_version": ROUND1_MANIFEST_SCHEMA_VERSION,
                "providers": ["fixed", "arch", "ss", "g0"],
            }
        )
    except ValueError as error:
        assert "provider order" in str(error)
    else:
        raise AssertionError("schema-v2 accepted a reordered provider list")


def test_migration_paths_add_ss_self_and_g0_routes():
    checkpoints = {
        provider: {1: f"{provider}.pt"} for provider in PROVIDER_KINDS
    }
    jobs = migration_path_jobs(seeds=[1], t0_checkpoints=checkpoints)

    assert {job["name"] for job in jobs} == {
        "fixed_to_fixed",
        "fixed_to_g0",
        "ss_to_ss",
        "ss_to_g0",
        "arch_to_arch",
        "arch_to_g0",
        "g0_to_g0",
    }


def test_convergence_report_resolves_imported_v1_cells_from_t0_registry():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        external = root / "source-v1" / "fixed" / "seed_1"
        external.mkdir(parents=True)
        selection = root / "augmented-v2" / "bdqn" / "t0_selection.json"
        selection.parent.mkdir(parents=True)
        selection.write_text(
            json.dumps(
                {"cells": {"fixed": {"1": str(external.resolve())}}}
            ),
            encoding="utf-8",
        )

        resolved = _convergence_cell_directories(root / "augmented-v2")

        assert resolved == {"fixed": [external.resolve()]}


def test_completed_v1_convergence_can_be_imported_without_modifying_source():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source_root = root / "source-v1"
        source_root.mkdir()
        architecture = source_root / "architecture.pt"
        gp_policy = source_root / "gp_policy.json"
        architecture.write_bytes(b"architecture")
        gp_policy.write_text("{}", encoding="utf-8")
        scenarios = generate_round1_scenarios(
            source_root / "scenarios",
            base_seed=77,
            b_train_size=4,
            b_validation_size=4,
            g_train_size=4,
            g_validation_size=4,
            test_iid_size=4,
            test_ood_size=4,
        )
        config = baseline_bdqn_config(seed=1, max_env_steps=1, device="cpu")
        initial = ensure_initial_checkpoint(source_root, seed=1, config=config)
        initial_hash = sha256_file(initial)
        for provider in LEGACY_PROVIDER_KINDS:
            cell_dir = source_root / "bdqn" / "convergence" / provider / "seed_1"
            cell_dir.mkdir(parents=True)
            cell = {
                "schema_version": 1,
                "status": "complete",
                "provider": provider,
                "seed": 1,
                "checkpoint_steps": [0, 1],
                "inputs": {
                    "source_checkpoint": {
                        "path": str(initial.resolve()),
                        "sha256": initial_hash,
                    },
                    "provider_sha256": _provider_input_hash(
                        provider,
                        architecture_checkpoint=architecture,
                        gp_policy=gp_policy,
                    ),
                    "train_manifest": {
                        "path": str(Path(scenarios["b_train"]).resolve()),
                        "sha256": sha256_file(scenarios["b_train"]),
                    },
                    "validation_manifest": {
                        "path": str(Path(scenarios["b_validation"]).resolve()),
                        "sha256": sha256_file(scenarios["b_validation"]),
                    },
                },
                "checkpoints": {
                    str(step): {"path": str(initial), "sha256": initial_hash}
                    for step in (0, 1)
                },
            }
            (cell_dir / "cell_manifest.json").write_text(
                json.dumps(cell), encoding="utf-8"
            )
        source_manifest = source_root / "study_manifest.json"
        source_study = {
            "schema_version": 1,
            "output_dir": str(source_root.resolve()),
            "base_seed": 77,
            "device": "cpu",
            "inputs": {
                "architecture_checkpoint": {
                    "path": str(architecture.resolve()),
                    "sha256": sha256_file(architecture),
                },
                "g0_policy": {
                    "path": str(gp_policy.resolve()),
                    "sha256": sha256_file(gp_policy),
                },
            },
            "scenarios": {
                name: str(Path(path).resolve())
                for name, path in scenarios.items()
                if name != "registry"
            },
            "scenario_registry": str(Path(scenarios["registry"]).resolve()),
            "seeds": [1],
            "discovery_seeds": [1],
            "confirmation_seeds": [1],
            "convergence_steps": [0, 1],
            "transfer_steps": [0, 1],
            "hyperparameter_matrix": bdqn_hyperparameter_matrix(),
            "gp_discovery_matrix": gp_discovery_matrix(),
            "test_locked": True,
            "stages": {
                "preflight": {"status": "complete"},
                "bdqn_convergence": {"status": "complete"},
            },
        }
        source_manifest.write_text(json.dumps(source_study), encoding="utf-8")
        source_hash = sha256_file(source_manifest)

        destination_manifest = initialize_round1_study(
            output_dir=root / "augmented-v2",
            augment_from=source_manifest,
            device="auto",
        )
        augmented = json.loads(destination_manifest.read_text(encoding="utf-8"))

        assert sha256_file(source_manifest) == source_hash
        assert augmented["schema_version"] == ROUND1_MANIFEST_SCHEMA_VERSION
        assert tuple(augmented["providers"]) == PROVIDER_KINDS
        assert augmented["ss_config_sha256"] == _provider_input_hash(
            "ss",
            architecture_checkpoint=None,
            gp_policy=None,
        )
        assert set(augmented["imported_convergence_cells"]) == set(
            LEGACY_PROVIDER_KINDS
        )
        assert augmented["shared_initial_checkpoints"]["1"]["sha256"] == initial_hash
        assert augmented["stages"]["preflight"]["status"] == "complete"

        running_source = json.loads(source_manifest.read_text(encoding="utf-8"))
        running_source["stages"]["bdqn_convergence"]["status"] = "running"
        running_source_manifest = source_root / "running_study_manifest.json"
        running_source_manifest.write_text(
            json.dumps(running_source), encoding="utf-8"
        )
        subset_manifest = initialize_round1_study(
            output_dir=root / "augmented-subset-v2",
            augment_from=running_source_manifest,
            augment_seeds=[1],
            device="auto",
        )
        subset = json.loads(subset_manifest.read_text(encoding="utf-8"))
        assert subset["seeds"] == [1]
        assert subset["imported_from"]["seed_subset"] == [1]
        assert set(subset["imported_convergence_cells"]) == set(
            LEGACY_PROVIDER_KINDS
        )

        try:
            initialize_round1_study(
                output_dir=root / "augmented-v2",
                augment_from=source_manifest,
                ss_config=HystereticCapacityConfig(
                    lower_threshold=0.35,
                    upper_threshold=0.90,
                ),
            )
        except ValueError as error:
            assert "configuration changed" in str(error)
        else:
            raise AssertionError("resuming v2 accepted changed S/s-HCM thresholds")

        tampered = dict(augmented)
        tampered["ss_config"] = {
            **tampered["ss_config"],
            "lower_threshold": 0.35,
        }
        destination_manifest.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            _load_study(destination_manifest)
        except ValueError as error:
            assert "configuration hash changed" in str(error)
        else:
            raise AssertionError("loading v2 accepted a tampered S/s-HCM config")


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


def test_ss_provider_cell_uses_the_shared_initial_checkpoint_and_validates():
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
            provider_kind="ss",
            source_checkpoint=initial,
            train_manifest=train,
            validation_manifest=validation,
            config=config,
            checkpoint_steps=(0, 1),
        )

        assert Path(outputs["manifest"]).is_file()
        assert Path(outputs["summary"]).is_file()
        assert "ss_3_1" in Path(outputs["summary"]).read_text(encoding="utf-8")


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
            lr=1e-4,
            lr_end=1e-5,
            lr_decay=0.5,
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
        with (root / "continuous" / "training_history.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            history = list(csv.DictReader(file))
        assert [float(row["learning_rate"]) for row in history] == [
            1e-4,
            5e-5,
            2.5e-5,
            1.25e-5,
        ]
        assert float(history[-1]["next_learning_rate"]) == 1e-5
