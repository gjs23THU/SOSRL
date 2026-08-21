from pathlib import Path
import csv
import json
import tempfile
import unittest

import torch

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.gp.artifact import (
    create_policy_artifact,
    save_gp_policy,
    sha256_file,
)
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.primitives import build_primitive_set, individual_from_expression
from sosrl.rl.branching import BranchingDQNAgent
from sosrl.rl.config import BranchingDQNConfig
from sosrl.workflows import evaluation
from sosrl.workflows.branching_gp_finetune import (
    StratifiedManifestSampler,
    classify_adaptation,
    compare_paired_results,
    finetune_branching_with_frozen_gp,
    initialize_run_directory,
    prepare_finetune_agent,
)
from sosrl.workflows.gp_architecture import save_scenario_manifest


class BranchingGPFinetuneUnitTest(unittest.TestCase):
    def test_stratified_sampler_is_balanced_reproducible_and_cycles(self):
        scenarios = [
            {"category": category, "scenario_hash": f"{category}-{index}"}
            for category in (
                "feasible_suboptimal",
                "capacity_tight",
                "missing_capability",
                "redundant_overbudget",
            )
            for index in range(2)
        ]
        first = StratifiedManifestSampler(scenarios, seed=4)
        second = StratifiedManifestSampler(scenarios, seed=4)

        first_eight = [first.next_payload()["scenario_hash"] for _ in range(8)]
        second_eight = [second.next_payload()["scenario_hash"] for _ in range(8)]
        next_eight = [first.next_payload()["category"] for _ in range(8)]

        self.assertEqual(first_eight, second_eight)
        self.assertEqual(len(set(first_eight)), 8)
        self.assertEqual(
            {category: next_eight.count(category) for category in set(next_eight)},
            {
                "feasible_suboptimal": 2,
                "capacity_tight": 2,
                "missing_capability": 2,
                "redundant_overbudget": 2,
            },
        )

    def test_prepare_agent_copies_networks_but_resets_optimizer_and_replay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "b0.pt"
            base_config = BranchingDQNConfig(device="cpu", lr=1e-4)
            base = BranchingDQNAgent(base_config)
            with torch.no_grad():
                next(base.q_net.parameters()).fill_(0.25)
                next(base.target_net.parameters()).fill_(0.5)
            base.learn_step = 123
            base.save_checkpoint(checkpoint_path)

            finetune_config = BranchingDQNConfig(
                device="cpu",
                lr=1e-5,
                buffer_size=50_000,
                min_buffer_size=1_000,
            )
            adapted, source = prepare_finetune_agent(
                checkpoint_path,
                finetune_config,
            )

        self.assertEqual(source["learn_step"], 123)
        self.assertEqual(adapted.learn_step, 123)
        self.assertEqual(len(adapted.replay), 0)
        self.assertEqual(adapted.optimizer.state, {})
        self.assertEqual(adapted.optimizer.param_groups[0]["lr"], 1e-5)
        self.assertTrue(
            torch.equal(
                next(base.q_net.parameters()).cpu(),
                next(adapted.q_net.parameters()).cpu(),
            )
        )
        self.assertTrue(
            torch.equal(
                next(base.target_net.parameters()).cpu(),
                next(adapted.target_net.parameters()).cpu(),
            )
        )

    def test_initialize_run_directory_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "run"
            config = {"seed": 4, "steps": 40_000}
            initialize_run_directory(destination, config=config, inputs={})
            with self.assertRaises(FileExistsError):
                initialize_run_directory(destination, config=config, inputs={})
            resumed = initialize_run_directory(
                destination,
                config=config,
                inputs={},
                resume=True,
            )

        self.assertEqual(resumed["config"], config)

    def test_sha256_source_files_remain_stable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.bin"
            path.write_bytes(b"baseline")
            before = sha256_file(path)
            initialize_run_directory(
                Path(temp_dir) / "run",
                config={"seed": 4},
                inputs={"source": {"path": str(path), "sha256": before}},
            )
            after = sha256_file(path)

        self.assertEqual(before, after)

    def test_adaptation_classification_covers_all_outcomes(self):
        base = {
            "failure_rate": 0.0,
            "mean_j": 2.0,
            "mean_success_makespan": 100.0,
            "budget_violation_rate": 0.10,
            "mean_architecture_changes": 10.0,
        }
        safe = {
            **base,
            "mean_j": 1.95,
            "mean_success_makespan": 99.0,
            "invalid_action_count": 0,
            "provider_invariant_violations": 0,
        }
        mixed = {
            **safe,
            "mean_j": 2.02,
            "mean_success_makespan": 95.0,
        }
        failed = {**safe, "failure_rate": 0.10}
        flat = {
            **safe,
            "mean_j": 1.995,
            "mean_success_makespan": 99.5,
        }

        self.assertEqual(
            classify_adaptation(base, safe, delta_j_ci=(-0.08, -0.01)),
            "accept_b1_no_gp_tuning",
        )
        self.assertEqual(
            classify_adaptation(base, mixed, delta_j_ci=(-0.01, 0.05)),
            "accept_b1_consider_gp_tuning",
        )
        self.assertEqual(
            classify_adaptation(base, failed, delta_j_ci=(-0.20, 0.10)),
            "reject_b1_revisit_scheduler",
        )
        self.assertEqual(
            classify_adaptation(base, flat, delta_j_ci=(-0.02, 0.02)),
            "inconclusive",
        )

    def test_paired_makespan_excludes_any_failed_episode(self):
        common = {
            "category": "feasible_suboptimal",
            "failure_aware_j": 1.0,
            "makespan": 10.0,
            "final_net_cost": 100.0,
            "peak_net_cost": 100.0,
            "architecture_changes": 0,
            "dead_end": False,
        }
        baseline = [
            {**common, "scenario_hash": "a", "success": True},
            {**common, "scenario_hash": "b", "success": True},
        ]
        candidate = [
            {
                **common,
                "scenario_hash": "a",
                "success": True,
                "makespan": 8.0,
            },
            {
                **common,
                "scenario_hash": "b",
                "success": False,
                "dead_end": True,
                "makespan": 20.0,
            },
        ]

        paired = compare_paired_results(baseline, candidate, seed=4)

        self.assertEqual(paired["delta_successful_makespan"]["count"], 1)
        self.assertEqual(paired["delta_successful_makespan"]["mean"], -2.0)


class BranchingGPFinetuneIntegrationTest(unittest.TestCase):
    categories = (
        "feasible_suboptimal",
        "capacity_tight",
        "missing_capability",
        "redundant_overbudget",
    )

    def _scenario(self, index, category, split):
        func_type = int(env.FULL_SOS[0].func_type)
        mission = [
            syn.Task(
                0,
                "task",
                [syn.Operation(0, "op", func_type, 1, 0)],
                due_time=10,
            )
        ]
        return evaluation.scenario_payload(
            index,
            (env.FULL_SOS[0],),
            mission,
            category=category,
            budget=8000.0,
            refund_rate=0.8,
            split=split,
        )

    def _fixture(self, root: Path, *, bound_hash=None):
        scenario_dir = root / "scenarios"
        train = [
            self._scenario(index, category, "train")
            for category in self.categories
            for index in range(64)
        ]
        validation = [
            self._scenario(index, category, "validation")
            for index, category in enumerate(self.categories)
        ]
        for filename, split, scenarios in (
            ("train.json", "train", train),
            ("validation.json", "validation", validation),
            ("test_iid.json", "test_iid", validation),
            ("test_ood.json", "test_ood", validation),
        ):
            save_scenario_manifest(
                scenario_dir / filename,
                split=split,
                seed=4,
                scenarios=scenarios,
            )
        scheduler_path = root / "b0.pt"
        BranchingDQNAgent(
            BranchingDQNConfig(device="cpu")
        ).save_checkpoint(scheduler_path)
        scheduler_hash = sha256_file(scheduler_path)
        feature_names = feature_names_for_preset("system_delta")
        individual = individual_from_expression(
            "0.0", build_primitive_set(feature_names)
        )
        artifact = create_policy_artifact(
            individual,
            feature_preset="system_delta",
            bdqn_checkpoint_sha256=(
                scheduler_hash if bound_hash is None else bound_hash
            ),
        )
        gp_path = save_gp_policy(root / "g0.json", artifact)
        return scheduler_path, gp_path, scenario_dir

    def test_full_smoke_and_resume_preserve_inputs_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scheduler, gp_policy, scenario_dir = self._fixture(root)
            output_dir = root / "run"
            hashes_before = {
                "b0": sha256_file(scheduler),
                "g0": sha256_file(gp_policy),
            }
            arguments = dict(
                scheduler_checkpoint=scheduler,
                gp_policy=gp_policy,
                scenario_dir=scenario_dir,
                output_dir=output_dir,
                extra_env_steps=8,
                checkpoint_interval_steps=2,
                seed=4,
                device="cpu",
                skip_historical_test=True,
            )

            outputs = finetune_branching_with_frozen_gp(**arguments)
            resumed = finetune_branching_with_frozen_gp(**arguments, resume=True)

            self.assertEqual(outputs.keys(), resumed.keys())
            self.assertEqual(hashes_before["b0"], sha256_file(scheduler))
            self.assertEqual(hashes_before["g0"], sha256_file(gp_policy))
            for threshold in (2, 4, 6, 8):
                self.assertTrue(
                    (output_dir / "training" / f"checkpoint_{threshold}steps.pt").is_file()
                )
            for relative in (
                "baseline/validation_results.csv",
                "training/training_history.csv",
                "validation/checkpoint_comparison.csv",
                "validation/paired_results.csv",
                "validation/selection.json",
                "report/b0_b1_summary.md",
                "report/stage_curves.png",
                "report/paired_effects.png",
            ):
                self.assertTrue((output_dir / relative).is_file(), relative)
            selection = json.loads(
                (output_dir / "validation" / "selection.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(selection["selected_model"], "B0")
            self.assertFalse(selection["adaptation_accepted"])
            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                manifest["stages"]["training"][
                    "replay_reinitialized_on_resume"
                ]
            )
            with (output_dir / "training" / "training_history.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                history = list(csv.DictReader(file))
            self.assertEqual(len(history), 8)
            self.assertEqual(
                {category: sum(row["category"] == category for row in history)
                 for category in self.categories},
                {category: 2 for category in self.categories},
            )
            with (output_dir / "validation" / "all_results.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                validation_rows = list(csv.DictReader(file))
            crossed = [row for row in validation_rows if row["model"] != "B0"]
            self.assertTrue(crossed)
            self.assertTrue(
                all(row["checkpoint_binding"] == "diagnostic_crossed" for row in crossed)
            )
            self.assertFalse(any(output_dir.rglob("*evolution*")))

    def test_hash_mismatch_rejects_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scheduler, gp_policy, scenario_dir = self._fixture(
                root, bound_hash="0" * 64
            )
            output_dir = root / "run"

            with self.assertRaisesRegex(ValueError, "does not match"):
                finetune_branching_with_frozen_gp(
                    scheduler_checkpoint=scheduler,
                    gp_policy=gp_policy,
                    scenario_dir=scenario_dir,
                    output_dir=output_dir,
                    extra_env_steps=4,
                    checkpoint_interval_steps=1,
                    device="cpu",
                    skip_historical_test=True,
                )

            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
