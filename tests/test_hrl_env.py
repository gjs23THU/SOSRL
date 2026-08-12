import unittest
from unittest.mock import patch

import numpy as np

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.rules import architecture as archrule
from sosrl.rules import scheduling as rule


class AdaptiveMissionEnvironmentTests(unittest.TestCase):
    def make_task(self, index, function_types, durations=None):
        durations = durations or [10] * len(function_types)
        operations = [
            syn.Operation(
                index=op_idx,
                name=f"op-{index}-{op_idx}",
                func_type=func_type,
                duration=durations[op_idx],
                release_time=0,
            )
            for op_idx, func_type in enumerate(function_types)
        ]
        return syn.Task(index, f"task-{index}", operations, due_time=1000)

    def system(self, index):
        return env.FULL_SOS[index]

    def test_dynamic_observations_keep_scheduler_dimension(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        mission_env = env.MissionEnv([self.system(0)], mission, adaptive=True)

        self.assertEqual(mission_env.schedule_observation().shape, (25,))
        expected_arch_dim = 25 + 3 * mission_env.N + 5 * len(syn.func_type2idx) + 6
        self.assertEqual(
            mission_env.architecture_observation().shape,
            (expected_arch_dim,),
        )
        self.assertTrue(np.isfinite(mission_env.architecture_observation()).all())

    def test_add_remove_and_readd_use_eighty_percent_refund(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        initial = self.system(0)
        added = self.system(2)
        mission_env = env.MissionEnv([initial], mission, adaptive=True)

        initial_cost = float(initial.cost)
        add_result = mission_env.add_system(added.index)
        self.assertTrue(add_result["valid"])
        self.assertEqual(mission_env.net_cost, initial_cost + float(added.cost))

        remove_result = mission_env.remove_system(added.index)
        self.assertEqual(remove_result["refund"], 0.8 * float(added.cost))
        self.assertEqual(
            mission_env.net_cost,
            initial_cost + 0.2 * float(added.cost),
        )

        mission_env.add_system(added.index)
        self.assertEqual(
            mission_env.net_cost,
            initial_cost + 1.2 * float(added.cost),
        )

    def test_cost_trajectory_records_peak_and_transient_budget_violation(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        initial = self.system(0)
        added = self.system(2)
        budget = float(initial.cost) + 0.5 * float(added.cost)
        mission_env = env.MissionEnv(
            [initial],
            mission,
            adaptive=True,
            budget=budget,
        )

        initial_metrics = mission_env.cost_metrics()
        self.assertEqual(initial_metrics["initial_net_cost"], float(initial.cost))
        self.assertEqual(initial_metrics["gross_charge"], float(initial.cost))
        self.assertFalse(initial_metrics["ever_over_budget"])

        mission_env.add_system(added.index)
        mission_env.remove_system(added.index)
        metrics = mission_env.cost_metrics()

        self.assertEqual(
            metrics["peak_net_cost"],
            float(initial.cost + added.cost),
        )
        self.assertEqual(
            metrics["gross_charge"],
            float(initial.cost + added.cost),
        )
        self.assertEqual(metrics["total_refund"], 0.8 * float(added.cost))
        self.assertTrue(metrics["ever_over_budget"])
        self.assertFalse(metrics["final_over_budget"])
        self.assertEqual(metrics["final_net_cost"], mission_env.net_cost)

    def test_reset_restores_initial_cost_trajectory(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        initial = self.system(0)
        added = self.system(2)
        mission_env = env.MissionEnv([initial], mission, adaptive=True)
        mission_env.add_system(added.index)

        mission_env.reset()
        metrics = mission_env.cost_metrics()

        self.assertEqual(metrics["gross_charge"], float(initial.cost))
        self.assertEqual(metrics["peak_net_cost"], float(initial.cost))
        self.assertEqual(metrics["peak_active_cost"], float(initial.cost))
        self.assertFalse(metrics["ever_over_budget"])

    def test_candidate_finish_matrix_is_cached_by_decision_version(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        mission_env = env.MissionEnv([self.system(0)], mission, adaptive=True)

        first = mission_env.current_candidate_finish_times()
        second = mission_env.current_candidate_finish_times()
        self.assertIs(first, second)

        mission_env.add_system(self.system(2).index)
        third = mission_env.current_candidate_finish_times()
        self.assertIsNot(first, third)

    def test_remove_preserves_past_assignment_and_blocks_future_use(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type]), self.make_task(1, [s_type])]
        mission_env = env.MissionEnv(
            [self.system(0), self.system(2)],
            mission,
            adaptive=True,
        )
        action = mission_env.encode_assignment(0, 0, 0)
        mission_env.step(action)
        old_start = float(mission_env.state.op_start_time[0, 0])
        old_finish = float(mission_env.state.op_finish_time[0, 0])

        mission_env.remove_system(0)

        self.assertEqual(float(mission_env.state.op_start_time[0, 0]), old_start)
        self.assertEqual(float(mission_env.state.op_finish_time[0, 0]), old_finish)
        self.assertEqual(int(mission_env.state.op_assign_sys[0, 0]), 0)
        self.assertFalse(np.any(mission_env.valid_assignment_mask()[:, :, 0]))

        previous_ready = float(mission_env.state.system_ready_time[0])
        mission_env.add_system(0)
        self.assertEqual(float(mission_env.state.system_ready_time[0]), previous_ready)

    def test_replace_is_atomic_and_charges_net_delta(self):
        mission = [self.make_task(0, [syn.func_type2idx["S"]])]
        old_system = self.system(0)
        new_system = self.system(2)
        mission_env = env.MissionEnv([old_system], mission, adaptive=True)

        result = mission_env.replace_system(old_system.index, new_system.index)

        expected_delta = float(new_system.cost) - 0.8 * float(old_system.cost)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["cost_delta"], expected_delta)
        self.assertEqual(mission_env.architecture_change_count, 1)
        self.assertFalse(mission_env.active_system_mask[old_system.index])
        self.assertTrue(mission_env.active_system_mask[new_system.index])

    def test_architecture_rules_restore_missing_capability(self):
        d_type = syn.func_type2idx["D"]
        mission = [self.make_task(0, [d_type])]
        mission_env = env.MissionEnv([self.system(0)], mission, adaptive=True)
        policy = archrule.ArchitectureRule(mission_env)

        mask = policy.action_mask()
        self.assertEqual(mask.shape, (6,))
        self.assertEqual(mask[0], 0.0)
        self.assertEqual(mask[1], 1.0)

        result = policy.apply(1)
        self.assertTrue(result["valid"])
        self.assertEqual(result["rule_name"], "ADD_CAPABILITY")
        self.assertTrue(np.any(mission_env.valid_assignment_mask()))

    def test_adaptive_step_waits_for_architecture_rescue(self):
        s_type = syn.func_type2idx["S"]
        d_type = syn.func_type2idx["D"]
        mission = [self.make_task(0, [s_type, d_type])]
        mission_env = env.MissionEnv([self.system(0)], mission, adaptive=True)

        _, _, terminated, _, info = mission_env.step(
            mission_env.encode_assignment(0, 0, 0)
        )

        self.assertFalse(terminated)
        self.assertFalse(info["dead_end"])
        self.assertTrue(info["needs_architecture_change"])

    def test_scheduler_rules_follow_changed_active_mask(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type])]
        mission_env = env.MissionEnv(
            [self.system(0), self.system(2)], mission, adaptive=True
        )
        mission_env.remove_system(0)
        scheduler = rule.Rule(mission_env)

        for rule_action in range(rule.Rule.RULE_NUM):
            assignment = mission_env.decode_assignment(
                scheduler.to_env_action(rule_action)
            )
            self.assertEqual(assignment["sys_idx"], 2)

    def test_capacity_window_remove_and_replace_rules_have_concrete_targets(self):
        s_type = syn.func_type2idx["S"]

        capacity_env = env.MissionEnv(
            [self.system(0)],
            [self.make_task(0, [s_type], durations=[100])],
            adaptive=True,
        )
        capacity_resolution = archrule.ArchitectureRule(capacity_env).resolve(2)
        self.assertIsNotNone(capacity_resolution)
        self.assertEqual(capacity_resolution.kind, "add")

        window_env = env.MissionEnv(
            [self.system(16)],
            [self.make_task(0, [s_type], durations=[200])],
            adaptive=True,
        )
        window_resolution = archrule.ArchitectureRule(window_env).resolve(3)
        self.assertIsNotNone(window_resolution)
        self.assertEqual(window_resolution.kind, "add")

        remove_env = env.MissionEnv(
            [self.system(0), self.system(2)],
            [self.make_task(0, [s_type])],
            adaptive=True,
        )
        remove_resolution = archrule.ArchitectureRule(remove_env).resolve(4)
        self.assertIsNotNone(remove_resolution)
        self.assertEqual(remove_resolution.kind, "remove")

        replace_env = env.MissionEnv(
            [self.system(11)],
            [self.make_task(0, [s_type])],
            adaptive=True,
        )
        replace_resolution = archrule.ArchitectureRule(replace_env).resolve(5)
        self.assertIsNotNone(replace_resolution)
        self.assertEqual(replace_resolution.kind, "replace")

    def test_full_candidate_pool_dead_end_has_no_architecture_action(self):
        s_type = syn.func_type2idx["S"]
        mission_env = env.MissionEnv(
            [],
            [self.make_task(0, [s_type], durations=[2000])],
            adaptive=True,
        )

        self.assertFalse(np.any(mission_env.global_assignment_mask()))
        self.assertFalse(
            np.any(archrule.ArchitectureRule(mission_env).action_mask())
        )

    def test_architecture_resolution_cache_reuses_and_invalidates_by_version(self):
        s_type = syn.func_type2idx["S"]
        mission_env = env.MissionEnv(
            [self.system(0)],
            [self.make_task(0, [s_type], durations=[100])],
            adaptive=True,
        )
        policy = archrule.ArchitectureRule(mission_env)

        with patch.object(
            policy,
            "_add_capacity",
            wraps=policy._add_capacity,
        ) as resolver:
            first = policy.resolve(2)
            second = policy.resolve(2)
            self.assertEqual(first, second)
            self.assertEqual(resolver.call_count, 1)
            self.assertEqual(policy.cache_hits, 1)

            mission_env.step(
                rule.Rule(mission_env).to_env_action(0)
            )
            policy.resolve(2)
            self.assertEqual(resolver.call_count, 2)

    def test_action_mask_and_apply_share_cached_resolution(self):
        d_type = syn.func_type2idx["D"]
        mission_env = env.MissionEnv(
            [self.system(0)],
            [self.make_task(0, [d_type])],
            adaptive=True,
        )
        policy = archrule.ArchitectureRule(mission_env)

        with patch.object(
            policy,
            "_add_capability",
            wraps=policy._add_capability,
        ) as resolver:
            mask = policy.action_mask()
            self.assertEqual(mask[1], 1.0)
            policy.apply(1)
            self.assertEqual(resolver.call_count, 1)


if __name__ == "__main__":
    unittest.main()
