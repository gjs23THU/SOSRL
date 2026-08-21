import unittest

import numpy as np

from sosrl import domain as syn
from sosrl.environment import FULL_SOS, MissionEnv
from sosrl.gp.architecture import ArchitectureAction, legal_architecture_actions
from sosrl.gp.features import (
    ACTION_TARGET_FEATURES,
    ARCH_FEATURE_SCHEMA_VERSION,
    BASE_SYSTEM_FEATURES,
    DEMAND_CONTEXT_FEATURES,
    FEATURE_PRESETS,
    SYSTEM_DELTA_FEATURES,
    architecture_feature_matrix,
    architecture_feature_vector,
    extract_architecture_features,
)


def systems_of_type(func_type: int) -> list[int]:
    return [int(system.index) for system in FULL_SOS if int(system.func_type) == func_type]


def build_mission(func_types: list[int]) -> list[syn.Task]:
    return [
        syn.Task(
            index=0,
            name="task-0",
            operations=[
                syn.Operation(
                    index=index,
                    name=f"op-{index}",
                    func_type=func_type,
                    duration=10 + index,
                    release_time=index * 10,
                )
                for index, func_type in enumerate(func_types)
            ],
            due_time=1000,
        )
    ]


class ArchitectureFeatureTest(unittest.TestCase):
    def setUp(self):
        self.types = sorted({int(system.func_type) for system in FULL_SOS})

    def make_env(self, active, mission_types, budget=8000.0):
        return MissionEnv(
            [FULL_SOS[index] for index in active],
            build_mission(list(mission_types)),
            adaptive=True,
            budget=budget,
            refund_rate=0.8,
        )

    def test_schema_and_preset_widths_are_fixed(self):
        self.assertEqual(ARCH_FEATURE_SCHEMA_VERSION, 2)
        self.assertEqual(len(BASE_SYSTEM_FEATURES), 21)
        self.assertEqual(len(DEMAND_CONTEXT_FEATURES), 8)
        self.assertEqual(len(SYSTEM_DELTA_FEATURES), 39)
        self.assertEqual(len(FEATURE_PRESETS["system"]), 21)
        self.assertEqual(len(FEATURE_PRESETS["system_demand"]), 29)
        self.assertEqual(len(FEATURE_PRESETS["system_delta"]), 39)
        self.assertEqual(len(FEATURE_PRESETS["op_context"]), 37)

    def test_redundant_version_one_features_are_absent(self):
        removed = {
            "removed_refund_ratio",
            "net_cost_after_ratio",
            "capability_coverage_after",
            "feasible_pair_ratio_after",
            "blocked_frontier_ratio_after",
            "target_type_capacity_after_norm",
            "target_type_capacity_margin_after_norm",
        }
        self.assertTrue(removed.isdisjoint(SYSTEM_DELTA_FEATURES))

    def test_keep_zeros_all_action_target_and_target_type_features(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], [func_type])

        features = extract_architecture_features(env, ArchitectureAction("keep"))

        for name in ACTION_TARGET_FEATURES:
            self.assertEqual(features[name], 0.0, name)
        for name in (
            "target_type_remaining_demand_norm",
            "target_type_active_capacity_norm",
            "target_type_pressure",
            "target_type_blocked_ratio",
            "target_type_active_ratio",
        ):
            self.assertEqual(features[name], 0.0, name)

    def test_add_cost_and_counterfactual_delta_are_exact(self):
        func_type = self.types[0]
        old_idx, new_idx = systems_of_type(func_type)[:2]
        env = self.make_env([old_idx], [func_type], budget=8000.0)
        action = ArchitectureAction("add", new_system=new_idx)

        features = extract_architecture_features(env, action)
        expected = float(FULL_SOS[new_idx].cost) / env.budget

        self.assertEqual(features["add_flag"], 1.0)
        self.assertAlmostEqual(features["added_cost_ratio"], expected)
        self.assertAlmostEqual(features["delta_net_cost_ratio"], expected)
        self.assertNotIn("net_cost_after_ratio", features)

    def test_replace_cost_delta_matches_environment_semantics(self):
        func_type = self.types[0]
        old_idx, new_idx = systems_of_type(func_type)[:2]
        env = self.make_env([old_idx], [func_type])
        action = ArchitectureAction(
            "replace", old_system=old_idx, new_system=new_idx
        )

        features = extract_architecture_features(env, action)
        expected_delta = (
            float(FULL_SOS[new_idx].cost)
            - env.refund_rate * float(FULL_SOS[old_idx].cost)
        ) / env.budget

        self.assertEqual(features["add_flag"], 1.0)
        self.assertEqual(features["remove_flag"], 1.0)
        self.assertAlmostEqual(features["delta_net_cost_ratio"], expected_delta)
        self.assertNotIn("removed_refund_ratio", features)

    def test_rescue_add_has_finite_sentinel_deltas(self):
        required_type, other_type = self.types[:2]
        required_idx = systems_of_type(required_type)[0]
        other_idx = systems_of_type(other_type)[0]
        env = self.make_env([other_idx], [required_type])
        action = ArchitectureAction("add", new_system=required_idx)
        self.assertIn(action, legal_architecture_actions(env))

        vector = architecture_feature_vector(env, action)
        features = extract_architecture_features(env, action)
        expected_best = float(
            np.min(
                env.current_candidate_finish_times()[
                    :, np.asarray(
                        [
                            index == required_idx or env.active_system_mask[index]
                            for index in range(env.N)
                        ],
                        dtype=bool,
                    )
                ]
            )
        ) / env.state.M

        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(
            features["best_frontier_finish_after_norm"], expected_best
        )
        self.assertGreater(features["delta_best_frontier_finish_norm"], -10.0)

    def test_every_action_vector_is_finite_and_clipped(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[:2]
        env = self.make_env(active, [func_type])

        for action in legal_architecture_actions(env):
            vector = architecture_feature_vector(env, action, "system_delta")
            self.assertEqual(vector.shape, (39,))
            self.assertTrue(np.all(np.isfinite(vector)), action)
            self.assertTrue(np.all(vector >= -10.0), action)
            self.assertTrue(np.all(vector <= 10.0), action)

    def test_feature_extraction_does_not_mutate_environment(self):
        func_type = self.types[0]
        old_idx, new_idx = systems_of_type(func_type)[:2]
        env = self.make_env([old_idx], [func_type])
        action = ArchitectureAction("add", new_system=new_idx)
        snapshot = (
            env.active_system_mask.copy(),
            env.state.system_ready_time.copy(),
            env.net_cost,
            env.active_cost,
            env.decision_version,
        )

        extract_architecture_features(env, action)

        np.testing.assert_array_equal(env.active_system_mask, snapshot[0])
        np.testing.assert_array_equal(env.state.system_ready_time, snapshot[1])
        self.assertEqual(env.net_cost, snapshot[2])
        self.assertEqual(env.active_cost, snapshot[3])
        self.assertEqual(env.decision_version, snapshot[4])

    def test_op_context_vector_uses_25_schedule_features(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], [func_type])
        vector = architecture_feature_vector(
            env, ArchitectureAction("keep"), "op_context"
        )

        self.assertEqual(vector.shape, (37,))
        np.testing.assert_allclose(
            vector[:25], np.clip(env.schedule_observation(), -10.0, 10.0)
        )

    def test_vectorized_matrix_matches_scalar_features_for_every_preset(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[:2]
        env = self.make_env(active, [func_type])
        actions = legal_architecture_actions(env)

        for preset in FEATURE_PRESETS:
            matrix = architecture_feature_matrix(env, actions, preset)
            scalar = np.vstack(
                [architecture_feature_vector(env, action, preset) for action in actions]
            )
            np.testing.assert_array_equal(matrix, scalar, err_msg=preset)


if __name__ == "__main__":
    unittest.main()
