import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.gp.architecture import (
    ArchitectureAction,
    apply_architecture_action,
    architecture_action_id,
    architecture_action_table_hash,
    raw_architecture_actions,
)
from sosrl.gp.evolution import EpisodeOutcome, episode_objective
from sosrl.nsga2 import (
    Chromosome,
    DynamicScheduleDecoder,
    NSGA2Config,
    ProblemLayout,
    crowding_distance,
    solve_manifest_nsga2,
    solve_scenario_nsga2,
)
from sosrl.nsga2.operators import (
    mutate_chromosome,
    pox_pair,
    random_chromosome,
)
from sosrl.objectives import gp_cost_breakdown
from sosrl.workflows import evaluation


class NSGA2CoreTests(unittest.TestCase):
    def make_task(self, index, function_types, durations=None):
        durations = durations or [10] * len(function_types)
        operations = [
            syn.Operation(
                index=op_idx,
                name=f"op-{index}-{op_idx}",
                func_type=int(func_type),
                duration=int(durations[op_idx]),
                release_time=0,
            )
            for op_idx, func_type in enumerate(function_types)
        ]
        return syn.Task(index, f"task-{index}", operations, due_time=1000)

    @staticmethod
    def action_id(kind, old=None, new=None):
        return architecture_action_id(
            ArchitectureAction(kind, old_system=old, new_system=new)
        )

    def test_gp_action_table_is_stable_and_complete(self):
        actions = raw_architecture_actions()

        self.assertEqual(len(actions), 203)
        self.assertEqual(actions[0], ArchitectureAction("keep"))
        self.assertEqual(
            [architecture_action_id(action) for action in actions],
            list(range(203)),
        )
        self.assertEqual(
            architecture_action_table_hash(),
            "f2465be320569282023699ea6675a6841cdb1b305cf1323a5ebdb53f39f90dc6",
        )

    def test_layout_repairs_os_ms_and_aa_structure(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type, s_type]),
            self.make_task(1, [s_type, s_type]),
        ]
        layout = ProblemLayout.from_mission(mission)
        chromosome = Chromosome(
            np.array([0, 0, 0, 99]),
            np.array([-3, -3, -3, -3]),
            np.array([-2, 0, 100, 999]),
        )

        repaired, counts = layout.repair(chromosome)

        self.assertEqual(sorted(repaired.os.tolist()), [0, 0, 1, 1])
        self.assertEqual(counts["os_repair_count"], 2)
        self.assertEqual(counts["aa_repair_count"], 2)
        self.assertEqual(repaired.aa.tolist(), [0, 0, 100, 0])
        self.assertTrue(
            all(
                int(repaired.ms[idx]) in layout.eligible_systems[idx]
                for idx in range(layout.operation_count)
            )
        )

    def test_decoder_executes_direct_gp_replace_and_shared_cost_formula(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type, s_type]),
            self.make_task(1, [s_type, s_type]),
        ]
        old_system = env.FULL_SOS[0]
        new_system = env.FULL_SOS[2]
        replace_id = self.action_id("replace", old_system.index, new_system.index)
        chromosome = Chromosome(
            np.array([0, 1, 0, 1]),
            np.array([new_system.index] * 4),
            np.array([replace_id, 0, 0, 0]),
        )

        result = DynamicScheduleDecoder([old_system], mission).decode(chromosome)

        self.assertTrue(result.success)
        self.assertEqual(result.completed_operations, 4)
        self.assertEqual(result.architecture_trace[0]["kind"], "replace")
        expected_net = (
            float(old_system.cost)
            + float(new_system.cost)
            - 0.8 * float(old_system.cost)
        )
        self.assertAlmostEqual(result.final_net_cost, expected_net)
        self.assertAlmostEqual(result.architecture_change_penalty, 80.0)
        expected = gp_cost_breakdown(
            final_net_cost=result.final_net_cost,
            peak_net_cost=result.metrics["peak_net_cost"],
            budget=8000.0,
            architecture_changes=1,
        )
        self.assertAlmostEqual(result.effective_cost, expected.effective_cost)
        self.assertAlmostEqual(result.gp_cost_score, expected.gp_cost_score)
        outcome = EpisodeOutcome(
            success=True,
            completed_operations=4,
            total_operations=4,
            makespan=result.makespan,
            scale=100.0,
            final_net_cost=result.final_net_cost,
            peak_net_cost=result.metrics["peak_net_cost"],
            budget=8000.0,
            architecture_changes=1,
        )
        self.assertAlmostEqual(
            episode_objective(outcome),
            10.0 * result.makespan / 100.0 + result.gp_cost_score,
        )

    def test_illegal_action_repairs_to_keep_when_keep_is_operation_feasible(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        system = env.FULL_SOS[0]
        illegal_add = self.action_id("add", new=system.index)

        result = DynamicScheduleDecoder([system], mission).decode(
            Chromosome(
                np.array([0]),
                np.array([system.index]),
                np.array([illegal_add]),
            )
        )

        trace = result.architecture_trace[0]
        self.assertTrue(result.success)
        self.assertEqual(trace["requested_kind"], "add")
        self.assertEqual(trace["effective_kind"], "keep")
        self.assertTrue(trace["aa_repaired"])
        self.assertEqual(result.repair_counts["aa_repair_count"], 1)

    def test_keep_infeasible_repairs_to_action_that_enables_preferred_system(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        system = env.FULL_SOS[0]

        result = DynamicScheduleDecoder([], mission).decode(
            Chromosome(
                np.array([0]),
                np.array([system.index]),
                np.array([0]),
            )
        )

        trace = result.architecture_trace[0]
        self.assertTrue(result.success)
        self.assertEqual(trace["requested_kind"], "keep")
        self.assertEqual(trace["effective_kind"], "add")
        self.assertEqual(trace["effective_new_system"], system.index)
        self.assertEqual(result.schedule[0]["sys_idx"], system.index)
        self.assertEqual(result.repair_counts["aa_repair_count"], 1)

    def test_post_action_machine_repair_never_assigns_removed_system(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        first = env.FULL_SOS[0]
        second = env.FULL_SOS[2]
        remove_second = self.action_id("remove", old=second.index)

        result = DynamicScheduleDecoder([first, second], mission).decode(
            Chromosome(
                np.array([0]),
                np.array([second.index]),
                np.array([remove_second]),
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.architecture_trace[0]["kind"], "remove")
        self.assertEqual(result.schedule[0]["sys_idx"], first.index)
        self.assertEqual(result.repair_counts["ms_repair_count"], 1)

    def test_decoder_repairs_operation_order_and_retains_dead_end_individual(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type], durations=[2000]),
            self.make_task(1, [s_type], durations=[10]),
        ]
        capable = ProblemLayout.from_mission(mission).eligible_systems[0][0]
        chromosome = Chromosome(
            np.array([0, 1]),
            np.array([capable, capable]),
            np.array([0, 0]),
        )

        result = DynamicScheduleDecoder([], mission).decode(chromosome)

        self.assertFalse(result.success)
        self.assertEqual(result.effective_os[0], 1)
        self.assertEqual(result.completed_operations, 1)
        self.assertEqual(result.constraint_violation, 0.5)
        self.assertGreaterEqual(result.repair_counts["os_repair_count"], 1)

    def test_all_four_gp_action_kinds_are_executable(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        first = env.FULL_SOS[0]
        second = env.FULL_SOS[2]

        cases = [
            ([first], first.index, 0, "keep"),
            ([], first.index, self.action_id("add", new=first.index), "add"),
            (
                [first, second],
                second.index,
                self.action_id("remove", old=first.index),
                "remove",
            ),
            (
                [first],
                second.index,
                self.action_id("replace", first.index, second.index),
                "replace",
            ),
        ]
        observed = []
        for architecture, system_idx, action_id, expected_kind in cases:
            result = DynamicScheduleDecoder(architecture, mission).decode(
                Chromosome(
                    np.array([0]),
                    np.array([system_idx]),
                    np.array([action_id]),
                )
            )
            self.assertTrue(result.success)
            observed.append(result.architecture_trace[0]["kind"])
            self.assertEqual(observed[-1], expected_kind)
        self.assertEqual(observed, ["keep", "add", "remove", "replace"])

    def test_schedule_respects_precedence_nonoverlap_and_windows(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type, s_type], durations=[11, 13]),
            self.make_task(1, [s_type, s_type], durations=[7, 9]),
        ]
        system = env.FULL_SOS[0]
        result = DynamicScheduleDecoder([system], mission).decode(
            Chromosome(
                np.array([0, 1, 0, 1]),
                np.full(4, system.index),
                np.zeros(4),
            )
        )

        self.assertTrue(result.success)
        for task_idx in range(2):
            rows = sorted(
                (row for row in result.schedule if row["task_idx"] == task_idx),
                key=lambda row: row["op_idx"],
            )
            self.assertGreaterEqual(rows[1]["start_time"], rows[0]["finish_time"])
        rows = sorted(result.schedule, key=lambda row: row["start_time"])
        for previous, current in zip(rows, rows[1:]):
            self.assertGreaterEqual(current["start_time"], previous["finish_time"])
        self.assertTrue(
            all(
                row["start_time"] >= system.available_from
                and row["finish_time"] <= system.available_until
                for row in result.schedule
            )
        )

    def test_remove_then_readd_preserves_historical_ready_time(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(index, [s_type]) for index in range(3)]
        first = env.FULL_SOS[0]
        second = env.FULL_SOS[2]
        result = DynamicScheduleDecoder([first], mission).decode(
            Chromosome(
                np.array([0, 1, 2]),
                np.array([first.index, second.index, first.index]),
                np.array(
                    [
                        0,
                        self.action_id("replace", first.index, second.index),
                        self.action_id("replace", second.index, first.index),
                    ]
                ),
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [row["kind"] for row in result.architecture_trace],
            ["keep", "replace", "replace"],
        )
        first_use, _, readded_use = result.schedule
        self.assertGreaterEqual(readded_use["start_time"], first_use["finish_time"])
        self.assertEqual(result.metrics["architecture_changes"], 2)

    def test_peak_budget_penalty_is_soft_and_only_changes_second_objective(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        system = env.FULL_SOS[0]
        result = DynamicScheduleDecoder([system], mission, budget=100.0).decode(
            Chromosome(
                np.array([0]),
                np.array([system.index]),
                np.array([0]),
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.constraint_violation, 0.0)
        self.assertTrue(result.metrics["ever_over_budget"])
        self.assertGreater(result.peak_budget_penalty, 0.0)
        self.assertGreater(result.effective_cost, result.final_net_cost)

    def test_state_valid_random_initialization_and_mutation_preserve_invariants(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(idx, [s_type, s_type]) for idx in range(3)]
        decoder = DynamicScheduleDecoder([env.FULL_SOS[0]], mission)
        layout = decoder.layout
        rng = np.random.default_rng(17)
        initialized = random_chromosome(decoder, rng)
        decoded = decoder.decode(initialized)

        self.assertTrue(decoded.success)
        self.assertEqual(decoded.repair_counts["aa_repair_count"], 0)
        self.assertEqual(decoded.repair_counts["ms_repair_count"], 0)
        first = np.array([0, 1, 2, 0, 1, 2])
        second = np.array([2, 2, 1, 1, 0, 0])
        child_a, child_b = pox_pair(first, second, 3, rng)
        self.assertEqual(sorted(child_a.tolist()), sorted(first.tolist()))
        self.assertEqual(sorted(child_b.tolist()), sorted(first.tolist()))
        chromosome = Chromosome(
            child_a,
            np.asarray([values[0] for values in layout.eligible_systems]),
            np.zeros(layout.operation_count),
        )
        mutated = mutate_chromosome(
            chromosome,
            layout,
            NSGA2Config(
                population_size=4,
                max_evaluations=4,
                os_mutation_probability=1.0,
                ms_gene_mutation_probability=1.0,
                aa_gene_mutation_probability=1.0,
            ),
            rng,
        )
        layout.validate(mutated)
        self.assertTrue(np.all((mutated.aa >= 0) & (mutated.aa < 203)))

    def test_gp_action_trace_replays_to_identical_environment_state(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(index, [s_type]) for index in range(3)]
        first = env.FULL_SOS[0]
        second = env.FULL_SOS[2]
        decoder = DynamicScheduleDecoder([first], mission)
        result = decoder.decode(
            Chromosome(
                np.array([0, 1, 2]),
                np.array([first.index, second.index, first.index]),
                np.array(
                    [
                        0,
                        self.action_id("replace", first.index, second.index),
                        self.action_id("replace", second.index, first.index),
                    ]
                ),
            )
        )

        replay = decoder._new_environment()
        for trace, schedule in zip(
            result.architecture_trace, result.schedule, strict=True
        ):
            action = raw_architecture_actions()[trace["effective_action_id"]]
            apply_architecture_action(replay, action)
            _, _, _, _, info = replay.step(
                replay.encode_assignment(
                    schedule["task_idx"],
                    schedule["op_idx"],
                    schedule["sys_idx"],
                )
            )
            self.assertTrue(info["valid"])
            self.assertEqual(
                np.flatnonzero(replay.active_system_mask).tolist(),
                trace["active_systems_after"],
            )
        self.assertEqual(float(replay.state.current_makespan), result.makespan)
        self.assertEqual(float(replay.net_cost), result.final_net_cost)
        self.assertEqual(float(replay.peak_net_cost), result.metrics["peak_net_cost"])
        self.assertEqual(
            replay.architecture_change_count,
            result.metrics["architecture_changes"],
        )

    def test_crowding_distance_matches_standard_normalized_formula(self):
        values = np.asarray([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float)

        distances = crowding_distance(values)

        self.assertTrue(np.isinf(distances[0]))
        self.assertTrue(np.isinf(distances[-1]))
        self.assertAlmostEqual(distances[1], 4.0 / 3.0)
        self.assertAlmostEqual(distances[2], 4.0 / 3.0)

    def test_small_solver_is_reproducible_and_uses_effective_cost(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type, s_type]),
            self.make_task(1, [s_type, s_type]),
        ]
        architecture = [env.FULL_SOS[0], env.FULL_SOS[2]]
        config = NSGA2Config(
            population_size=10,
            max_evaluations=40,
            independent_runs=1,
            base_seed=91,
            evaluation_milestones=(10, 20, 40),
        )

        first = solve_scenario_nsga2(architecture, mission, config=config)
        second = solve_scenario_nsga2(architecture, mission, config=config)

        self.assertTrue(first.combined_front)
        self.assertEqual(sorted(first.milestone_fronts), [10, 20, 40])
        self.assertTrue(all(first.milestone_fronts.values()))
        self.assertEqual(
            [
                result.phenotype_hash
                for result in first.milestone_fronts[20]
            ],
            [
                result.phenotype_hash
                for result in second.milestone_fronts[20]
            ],
        )
        self.assertTrue(all(result.success for result in first.combined_front))
        self.assertEqual(
            [result.phenotype_hash for result in first.combined_front],
            [result.phenotype_hash for result in second.combined_front],
        )
        self.assertIn("compromise", first.representatives)
        self.assertEqual(
            first.representatives["min_cost"].effective_cost,
            min(result.effective_cost for result in first.combined_front),
        )

    def test_v2_manifest_artifacts_include_replayable_aa_compromise(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type, s_type]),
            self.make_task(1, [s_type, s_type]),
        ]
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
            result = solve_manifest_nsga2(
                manifest_path,
                root / "output",
                config=NSGA2Config(
                    population_size=10,
                    max_evaluations=20,
                    independent_runs=1,
                    base_seed=123,
                    evaluation_milestones=(10, 20),
                ),
            )
            self.assertEqual(result["schema_version"], 2)
            scenario_dir = Path(result["scenarios"][0]["output_dir"])
            expected = {
                "run_manifest.json",
                "generation_history.csv",
                "pareto_front.csv",
                "selected_solutions.json",
                "schedule.csv",
                "architecture_trace.csv",
                "scenario_summary.csv",
            }
            self.assertTrue(all((scenario_dir / name).is_file() for name in expected))
            self.assertTrue((scenario_dir / "milestone_summary.csv").is_file())
            self.assertTrue(
                (
                    scenario_dir
                    / "milestones"
                    / "eval_000010"
                    / "pareto_front.csv"
                ).is_file()
            )
            with (scenario_dir / "milestone_summary.csv").open(
                encoding="utf-8"
            ) as file:
                milestone_rows = list(csv.DictReader(file))
            self.assertEqual(
                [int(row["evaluation_budget_per_run"]) for row in milestone_rows],
                [10, 20],
            )
            self.assertTrue(all(row["gp_aligned_j"] for row in milestone_rows))
            run_manifest = json.loads(
                (scenario_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["schema_version"], 2)
            self.assertEqual(run_manifest["encoding_version"], "os-ms-aa-gp203-v1")
            self.assertEqual(
                run_manifest["architecture_action_table_hash"],
                architecture_action_table_hash(),
            )
            with (scenario_dir / "scenario_summary.csv").open(
                encoding="utf-8"
            ) as file:
                summary_rows = list(csv.DictReader(file))
            self.assertEqual(len(summary_rows), 1)
            self.assertIn("effective_cost", summary_rows[0])
            selected = json.loads(
                (scenario_dir / "selected_solutions.json").read_text(
                    encoding="utf-8"
                )
            )["compromise"]
            self.assertIn("aa", selected["chromosome"])
            self.assertNotIn("rt", selected["chromosome"])
            replayed = DynamicScheduleDecoder(architecture, mission).decode(
                Chromosome(
                    np.asarray(selected["chromosome"]["os"], dtype=np.int32),
                    np.asarray(selected["chromosome"]["ms"], dtype=np.int32),
                    np.asarray(selected["chromosome"]["aa"], dtype=np.int32),
                )
            )
            self.assertTrue(replayed.success)
            self.assertAlmostEqual(replayed.makespan, selected["makespan"])
            self.assertAlmostEqual(
                replayed.effective_cost, selected["effective_cost"]
            )

    def test_milestones_must_align_with_population_batches(self):
        with self.assertRaisesRegex(ValueError, "multiples of population_size"):
            NSGA2Config(
                population_size=10,
                max_evaluations=40,
                evaluation_milestones=(15,),
            )


if __name__ == "__main__":
    unittest.main()
