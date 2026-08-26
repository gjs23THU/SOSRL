import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from pymoo.core.population import Population

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.nsga2 import Chromosome, DynamicScheduleDecoder
from sosrl.workflows import evaluation


class MOPSOCoreTests(unittest.TestCase):
    def make_task(self, index, function_types, durations=None):
        durations = durations or [10] * len(function_types)
        return syn.Task(
            index=index,
            name=f"task-{index}",
            operations=[
                syn.Operation(
                    index=op_idx,
                    name=f"op-{index}-{op_idx}",
                    func_type=int(func_type),
                    duration=int(duration),
                    release_time=0,
                )
                for op_idx, (func_type, duration) in enumerate(
                    zip(function_types, durations, strict=True)
                )
            ],
            due_time=500,
        )

    def test_config_validates_milestones_and_defaults(self):
        from sosrl.mopso import MOPSOConfig

        config = MOPSOConfig(evaluation_milestones=(200, 50, 100, 100))

        self.assertEqual(config.evaluation_milestones, (50, 100, 200))
        self.assertEqual(config.swarm_size, 50)
        self.assertEqual(config.archive_size, 200)
        self.assertTrue(config.gp_aligned_cost_defaults)
        with self.assertRaisesRegex(ValueError, "multiples of swarm_size"):
            MOPSOConfig(evaluation_milestones=(75,))

    def test_random_key_codec_round_trip_and_boundaries(self):
        from sosrl.mopso import RandomKeyCodec

        s_type = syn.func_type2idx["S"]
        d_type = syn.func_type2idx["D"]
        mission = [
            self.make_task(0, [s_type, d_type]),
            self.make_task(1, [d_type, s_type]),
        ]
        decoder = DynamicScheduleDecoder([env.FULL_SOS[0]], mission)
        layout = decoder.layout
        chromosome = Chromosome(
            np.asarray([1, 0, 1, 0], dtype=np.int32),
            np.asarray(
                [candidates[-1] for candidates in layout.eligible_systems],
                dtype=np.int32,
            ),
            np.asarray([0, 1, 202, 17], dtype=np.int32),
        )
        codec = RandomKeyCodec(layout)

        position = codec.encode(chromosome)
        decoded = codec.decode(position)

        self.assertTrue(np.array_equal(decoded.flat, chromosome.flat))
        boundary = codec.decode(
            np.concatenate(
                (
                    np.linspace(0.0, 1.0, layout.operation_count),
                    np.ones(layout.operation_count),
                    np.ones(layout.operation_count),
                )
            )
        )
        layout.validate(boundary)
        self.assertTrue(np.all(boundary.aa == layout.action_count - 1))
        self.assertTrue(
            all(
                int(boundary.ms[idx]) in layout.eligible_systems[idx]
                for idx in range(layout.operation_count)
            )
        )

    def test_jittered_encoding_preserves_chromosome(self):
        from sosrl.mopso import RandomKeyCodec

        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(idx, [s_type, s_type]) for idx in range(3)]
        decoder = DynamicScheduleDecoder([env.FULL_SOS[0]], mission)
        chromosome = Chromosome(
            np.asarray([2, 0, 1, 2, 1, 0], dtype=np.int32),
            np.asarray(
                [candidates[0] for candidates in decoder.layout.eligible_systems],
                dtype=np.int32,
            ),
            np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int32),
        )
        codec = RandomKeyCodec(decoder.layout)

        position = codec.encode(
            chromosome,
            random_state=np.random.default_rng(19),
            jitter=True,
        )

        self.assertTrue(np.array_equal(codec.decode(position).flat, chromosome.flat))

    def test_constraint_aware_archive_prefers_feasible_particle(self):
        from sosrl.mopso.solver import ConstraintAwareMOPSOCD

        algorithm = ConstraintAwareMOPSOCD(
            pop_size=4,
            archive_size=4,
            seed=7,
        )
        algorithm.random_state = np.random.default_rng(7)
        algorithm.leader_archive = Population.empty()
        population = Population.new(
            "X",
            np.asarray([[0.0], [1.0]]),
            "F",
            np.asarray([[0.0, 0.0], [10.0, 10.0]]),
            "G",
            np.asarray([[1.0], [0.0]]),
        )

        archive = algorithm._update_leader_archive(population)

        self.assertEqual(len(archive), 1)
        self.assertEqual(float(archive[0].X[0]), 1.0)
        self.assertEqual(float(archive[0].CV[0]), 0.0)

    def test_constraint_archive_uses_minimum_cv_and_deduplicates(self):
        from sosrl.mopso.solver import ConstraintAwareMOPSOCD

        algorithm = ConstraintAwareMOPSOCD(pop_size=4, archive_size=4, seed=8)
        algorithm.random_state = np.random.default_rng(8)
        algorithm.leader_archive = Population.empty()
        infeasible = Population.new(
            "X",
            np.asarray([[0.0], [1.0], [2.0]]),
            "F",
            np.asarray([[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]]),
            "G",
            np.asarray([[0.4], [0.2], [0.3]]),
        )

        minimum_cv = algorithm._update_leader_archive(infeasible)

        self.assertEqual(len(minimum_cv), 1)
        self.assertEqual(float(minimum_cv[0].X[0]), 1.0)
        algorithm.leader_archive = Population.empty()
        duplicates = Population.new(
            "X",
            np.asarray([[3.0], [3.0]]),
            "F",
            np.asarray([[2.0, 2.0], [1.0, 1.0]]),
            "G",
            np.asarray([[0.0], [0.0]]),
        )

        deduplicated = algorithm._update_leader_archive(duplicates)

        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0].F.tolist(), [1.0, 1.0])

    def test_constraint_aware_pbest_prefers_lower_cv(self):
        from sosrl.mopso.solver import ConstraintAwareMOPSOCD

        algorithm = ConstraintAwareMOPSOCD(pop_size=1, archive_size=4, seed=11)
        algorithm.random_state = np.random.default_rng(11)
        algorithm.pbest = Population.new(
            "X",
            np.asarray([[0.0]]),
            "F",
            np.asarray([[0.0, 0.0]]),
            "G",
            np.asarray([[1.0]]),
        )
        algorithm.pbest_f = algorithm.pbest.get("F").copy()
        algorithm.pbest_cv = algorithm.pbest.get("CV").reshape(-1).copy()
        current = Population.new(
            "X",
            np.asarray([[1.0]]),
            "F",
            np.asarray([[100.0, 100.0]]),
            "G",
            np.asarray([[0.0]]),
        )

        algorithm._update_pbest(current)

        self.assertEqual(float(algorithm.pbest[0].X[0]), 1.0)
        self.assertEqual(float(algorithm.pbest_cv[0]), 0.0)

    def test_solver_uses_exact_budget_and_is_reproducible(self):
        from sosrl.mopso import MOPSOConfig, RandomKeyCodec, solve_scenario_mopso

        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(idx, [s_type, s_type]) for idx in range(2)]
        architecture = [env.FULL_SOS[0], env.FULL_SOS[2]]
        config = MOPSOConfig(
            swarm_size=10,
            max_evaluations=40,
            independent_runs=1,
            archive_size=20,
            base_seed=91,
            evaluation_milestones=(10, 20, 40),
        )

        first = solve_scenario_mopso(architecture, mission, config=config)
        second = solve_scenario_mopso(architecture, mission, config=config)

        self.assertEqual(first.runs[0].evaluations, 40)
        self.assertEqual(first.runs[0].history[0]["evaluations"], 10)
        self.assertEqual(sorted(first.milestone_fronts), [10, 20, 40])
        self.assertTrue(first.combined_front)
        self.assertEqual(
            [item.phenotype_hash for item in first.combined_front],
            [item.phenotype_hash for item in second.combined_front],
        )
        self.assertEqual(
            {
                role: item.phenotype_hash
                for role, item in first.representatives.items()
            },
            {
                role: item.phenotype_hash
                for role, item in second.representatives.items()
            },
        )
        self.assertEqual(
            {
                milestone: [item.phenotype_hash for item in front]
                for milestone, front in first.milestone_fronts.items()
            },
            {
                milestone: [item.phenotype_hash for item in front]
                for milestone, front in second.milestone_fronts.items()
            },
        )
        codec = RandomKeyCodec(DynamicScheduleDecoder(architecture, mission).layout)
        for item in first.combined_front:
            position = first.positions[item.phenotype_hash]
            self.assertTrue(
                np.array_equal(codec.decode(position).flat, item.chromosome.flat)
            )

    def test_manifest_artifacts_preserve_keys_chromosome_and_replay(self):
        from sosrl.mopso import (
            MOPSOConfig,
            RandomKeyCodec,
            solve_manifest_mopso,
        )

        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(idx, [s_type, s_type]) for idx in range(4)]
        architecture = [env.FULL_SOS[0], env.FULL_SOS[2]]
        scenario = evaluation.scenario_payload(
            0, architecture, mission, category="unit", budget=8000.0
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "scenarios.json"
            manifest_path.write_text(
                json.dumps({"scenarios": [scenario]}), encoding="utf-8"
            )
            output = root / "output"
            result = solve_manifest_mopso(
                manifest_path,
                output,
                config=MOPSOConfig(
                    swarm_size=20,
                    max_evaluations=300,
                    independent_runs=2,
                    archive_size=200,
                    base_seed=123,
                    evaluation_milestones=(20, 100, 300),
                ),
            )

            self.assertEqual(result["schema_version"], 2)
            self.assertEqual(
                result["algorithm"],
                "dynamic_architecture_scheduling_mopso_cd",
            )
            self.assertTrue((output / "mopso_manifest.json").is_file())
            scenario_dir = Path(result["scenarios"][0]["output_dir"])
            expected = {
                "run_manifest.json",
                "iteration_history.csv",
                "pareto_front.csv",
                "selected_solutions.json",
                "schedule.csv",
                "architecture_trace.csv",
                "scenario_summary.csv",
                "milestone_summary.csv",
            }
            self.assertTrue(all((scenario_dir / name).is_file() for name in expected))
            self.assertTrue(
                (
                    scenario_dir
                    / "milestones"
                    / "eval_000020"
                    / "pareto_front.csv"
                ).is_file()
            )
            with (scenario_dir / "milestone_summary.csv").open(
                encoding="utf-8"
            ) as file:
                milestone_rows = list(csv.DictReader(file))
            self.assertEqual(
                [int(row["evaluation_budget_per_run"]) for row in milestone_rows],
                [20, 100, 300],
            )
            run_manifest = json.loads(
                (scenario_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                run_manifest["encoding_version"],
                "random-key-os-ms-aa-gp203-v1",
            )
            self.assertEqual(run_manifest["run_evaluations"], [300, 300])
            selected = json.loads(
                (scenario_dir / "selected_solutions.json").read_text(
                    encoding="utf-8"
                )
            )["compromise"]
            self.assertIn("particle_position", selected)
            self.assertIn("random_keys", selected)
            decoder = DynamicScheduleDecoder(architecture, mission)
            codec = RandomKeyCodec(decoder.layout)
            chromosome = Chromosome(
                np.asarray(selected["chromosome"]["os"], dtype=np.int32),
                np.asarray(selected["chromosome"]["ms"], dtype=np.int32),
                np.asarray(selected["chromosome"]["aa"], dtype=np.int32),
            )
            position = np.asarray(selected["particle_position"], dtype=float)
            self.assertTrue(np.array_equal(codec.decode(position).flat, chromosome.flat))
            replayed = decoder.decode(chromosome)
            self.assertEqual(replayed.phenotype_hash, selected["phenotype_hash"])
            self.assertAlmostEqual(replayed.makespan, selected["makespan"])
            self.assertAlmostEqual(
                replayed.effective_cost, selected["effective_cost"]
            )


if __name__ == "__main__":
    unittest.main()
