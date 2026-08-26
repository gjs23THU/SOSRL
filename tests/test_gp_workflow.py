import csv
import json
from pathlib import Path
import tempfile
import unittest

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.gp.config import GPArchitectureConfig
from sosrl.rl.branching import BranchingDQNAgent
from sosrl.rl.config import BranchingDQNConfig
from sosrl.workflows import evaluation
from sosrl.workflows.gp_architecture import (
    SCENARIO_CATEGORIES,
    branching_parameter_hash,
    evaluate_gp_stack,
    generate_gp_scenario_manifests,
    load_scenario_manifest,
    save_scenario_manifest,
    stratified_generation_batch,
    train_gp_architecture,
)


def one_operation_mission(func_type):
    return [
        syn.Task(
            0,
            "task",
            [syn.Operation(0, "op", int(func_type), 10, 0)],
            due_time=100,
        )
    ]


class GPWorkflowTest(unittest.TestCase):
    def make_tiny_manifest(self, directory: Path, split: str, seed: int):
        func_type = int(env.FULL_SOS[0].func_type)
        static_system = next(
            system
            for system in env.FULL_SOS[1:]
            if int(system.func_type) == func_type
        )
        mission = one_operation_mission(func_type)
        scenarios = [
            evaluation.scenario_payload(
                index,
                (env.FULL_SOS[0],),
                mission,
                category=category,
                budget=8000.0,
                refund_rate=0.8,
                split=split,
                static_feasible_architecture=(static_system,),
            )
            for index, category in enumerate(SCENARIO_CATEGORIES)
        ]
        return save_scenario_manifest(
            directory / f"{split}.json",
            split=split,
            seed=seed,
            scenarios=scenarios,
        )

    def test_generated_manifests_are_balanced_hash_stable_and_ood(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = generate_gp_scenario_manifests(
                temp_dir,
                base_seed=123,
                train_size=4,
                validation_size=4,
                test_size=4,
                ood_size=4,
            )
            first = load_scenario_manifest(paths["train"])
            second = load_scenario_manifest(paths["train"])
            ood = load_scenario_manifest(paths["test_ood"])

        self.assertEqual(first["manifest_hash"], second["manifest_hash"])
        self.assertEqual(first["schema_version"], 2)
        self.assertEqual(
            {item["category"] for item in first["scenarios"]},
            set(SCENARIO_CATEGORIES),
        )
        self.assertTrue(
            all(len(item["mission"]) == 40 for item in ood["scenarios"])
        )
        self.assertEqual(
            sorted(item["budget"] for item in ood["scenarios"]),
            [6400.0, 6400.0, 9600.0, 9600.0],
        )
        for item in first["scenarios"]:
            self.assertIn("static_feasible_system_indices", item)
            static_architecture, _ = evaluation.static_scenario_from_payload(item)
            self.assertLessEqual(
                sum(system.cost for system in static_architecture),
                item["budget"],
            )

    def test_static_scenario_requires_registered_architecture(self):
        func_type = int(env.FULL_SOS[0].func_type)
        payload = evaluation.scenario_payload(
            0,
            (env.FULL_SOS[0],),
            one_operation_mission(func_type),
        )

        with self.assertRaisesRegex(ValueError, "static feasible architecture"):
            evaluation.static_scenario_from_payload(payload)

    def test_manifest_round_trip_reconstructs_full_mission(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_tiny_manifest(Path(temp_dir), "train", 1)
            manifest = load_scenario_manifest(path)
            scenario = manifest["scenarios"][0]
            architecture, mission = evaluation.scenario_from_payload(scenario)

        self.assertEqual(architecture[0].index, 0)
        self.assertEqual(mission[0].operations[0].duration, 10)
        self.assertEqual(
            evaluation.verify_scenario_payload(scenario),
            scenario["scenario_hash"],
        )

    def test_generation_sampling_is_balanced_and_reproducible(self):
        scenarios = [
            {"category": category, "scenario_idx": repeat}
            for category in SCENARIO_CATEGORIES
            for repeat in range(4)
        ]
        first = stratified_generation_batch(
            scenarios,
            run_seed=10,
            generation=2,
            batch_size=8,
        )
        second = stratified_generation_batch(
            scenarios,
            run_seed=10,
            generation=2,
            batch_size=8,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {category: sum(item["category"] == category for item in first) for category in SCENARIO_CATEGORIES},
            {category: 2 for category in SCENARIO_CATEGORIES},
        )

    def test_two_generation_training_locks_json_without_changing_bdqn(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scenario_dir = root / "scenarios"
            scenario_dir.mkdir()
            self.make_tiny_manifest(scenario_dir, "train", 1)
            self.make_tiny_manifest(scenario_dir, "validation", 2)
            self.make_tiny_manifest(scenario_dir, "test_iid", 3)
            self.make_tiny_manifest(scenario_dir, "test_ood", 4)
            agent = BranchingDQNAgent(BranchingDQNConfig(device="cpu"))
            before = branching_parameter_hash(agent)
            checkpoint = agent.save_checkpoint(root / "branching_scheduler.pt")
            config = GPArchitectureConfig(
                population_size=8,
                generations=2,
                independent_runs=1,
                elite_count=1,
                train_batch_size=4,
                anchor_size=4,
                anchor_interval=1,
                anchor_top_k=2,
                workers=1,
                base_seed=17,
            )
            outputs = train_gp_architecture(
                scheduler_checkpoint=checkpoint,
                scenario_dir=scenario_dir,
                output_dir=root / "gp",
                config=config,
                device="cpu",
            )
            restored, _ = BranchingDQNAgent.load_checkpoint(
                checkpoint, device="cpu", load_optimizer=False
            )
            evaluation_outputs = evaluate_gp_stack(
                gp_policy=outputs["gp_policy"],
                scheduler_checkpoint=checkpoint,
                scenario_manifest=scenario_dir / "test_iid.json",
                output_dir=root / "evaluation",
                baselines=("fixed", "ss", "gp"),
                device="cpu",
            )

            expected = {
                "gp_policy",
                "gp_stack_manifest",
                "generation_history",
                "anchor_history",
                "candidate_rules",
                "validation_results",
            }
            self.assertEqual(set(outputs), expected)
            self.assertTrue(all(path.exists() for path in outputs.values()))
            self.assertEqual(before, branching_parameter_hash(restored))
            stack = json.loads(
                outputs["gp_stack_manifest"].read_text(encoding="utf-8")
            )
            self.assertEqual(
                stack["scheduler_checkpoint_sha256_before"],
                stack["scheduler_checkpoint_sha256_after"],
            )
            self.assertEqual(
                stack["scheduler_parameter_sha256_before"],
                stack["scheduler_parameter_sha256_after"],
            )
            self.assertTrue(all(path.exists() for path in evaluation_outputs.values()))
            with evaluation_outputs["results"].open(encoding="utf-8-sig") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["model"] for row in rows}, {"fixed", "ss", "gp"})
            self.assertIn("ss_emergency_count", rows[0])
            self.assertIn("ss_replace_dominates_add_count", rows[0])
            self.assertIn("ss_net_cost_delta_total", rows[0])
            scenarios = {
                item["scenario_hash"]: item
                for item in load_scenario_manifest(
                    scenario_dir / "test_iid.json"
                )["scenarios"]
            }
            for row in rows:
                scenario = scenarios[row["scenario_hash"]]
                indices_key = (
                    "static_feasible_system_indices"
                    if row["model"] == "fixed"
                    else "architecture_system_indices"
                )
                expected_cost = sum(
                    env.FULL_SOS[int(index)].cost
                    for index in scenario[indices_key]
                )
                self.assertEqual(float(row["initial_net_cost"]), expected_cost)


if __name__ == "__main__":
    unittest.main()
