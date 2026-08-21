import unittest

import numpy as np

from sosrl import domain as syn
from sosrl.environment import FULL_SOS, MissionEnv
from sosrl.gp.architecture import (
    ArchitectureAction,
    apply_architecture_action,
    legal_architecture_actions,
    raw_architecture_actions,
)


def system_indices(func_type: int) -> list[int]:
    return [int(system.index) for system in FULL_SOS if int(system.func_type) == func_type]


def mission_with_types(*func_types: int) -> list[syn.Task]:
    operations = [
        syn.Operation(
            index=index,
            name=f"op-{index}",
            func_type=int(func_type),
            duration=10,
            release_time=index * 10,
        )
        for index, func_type in enumerate(func_types)
    ]
    return [
        syn.Task(
            index=0,
            name="task-0",
            operations=operations,
            due_time=1000,
        )
    ]


class ArchitectureActionTest(unittest.TestCase):
    def setUp(self):
        self.types = sorted({int(system.func_type) for system in FULL_SOS})

    def make_env(self, active_indices, mission_types):
        return MissionEnv(
            [FULL_SOS[index] for index in active_indices],
            mission_with_types(*mission_types),
            adaptive=True,
        )

    def test_fixed_universe_has_203_typed_actions(self):
        actions = raw_architecture_actions()

        self.assertEqual(len(actions), 203)
        self.assertEqual(sum(a.kind == "keep" for a in actions), 1)
        self.assertEqual(sum(a.kind == "add" for a in actions), 22)
        self.assertEqual(sum(a.kind == "remove" for a in actions), 22)
        self.assertEqual(sum(a.kind == "replace" for a in actions), 158)
        self.assertTrue(
            all(
                int(FULL_SOS[a.old_system].func_type)
                == int(FULL_SOS[a.new_system].func_type)
                for a in actions
                if a.kind == "replace"
            )
        )

    def test_action_field_contract_is_strict(self):
        with self.assertRaises(ValueError):
            ArchitectureAction("add")
        with self.assertRaises(ValueError):
            ArchitectureAction("keep", new_system=0)
        with self.assertRaises(ValueError):
            ArchitectureAction("replace", old_system=0, new_system=0)

    def test_active_system_cannot_be_added_and_inactive_cannot_be_removed(self):
        func_type = self.types[0]
        old_idx, new_idx = system_indices(func_type)[:2]
        env = self.make_env([old_idx], [func_type])
        actions = set(legal_architecture_actions(env))

        self.assertNotIn(ArchitectureAction("add", new_system=old_idx), actions)
        self.assertNotIn(ArchitectureAction("remove", old_system=new_idx), actions)

    def test_cross_function_replacement_is_not_legal(self):
        old_idx = system_indices(self.types[0])[0]
        new_idx = system_indices(self.types[1])[0]
        env = self.make_env([old_idx], [self.types[0]])

        action = ArchitectureAction("replace", old_system=old_idx, new_system=new_idx)
        self.assertNotIn(action, legal_architecture_actions(env))

    def test_missing_capability_removes_keep_but_retains_rescue_add(self):
        required_type, other_type = self.types[:2]
        inactive_required = system_indices(required_type)[0]
        active_other = system_indices(other_type)[0]
        env = self.make_env([active_other], [required_type])
        actions = set(legal_architecture_actions(env))

        self.assertNotIn(ArchitectureAction("keep"), actions)
        self.assertIn(
            ArchitectureAction("add", new_system=inactive_required), actions
        )

    def test_remove_checks_immediate_pair_not_future_coverage(self):
        current_type, future_type = self.types[:2]
        current_idx = system_indices(current_type)[0]
        future_idx = system_indices(future_type)[0]
        env = self.make_env([current_idx, future_idx], [current_type, future_type])

        self.assertIn(
            ArchitectureAction("remove", old_system=future_idx),
            legal_architecture_actions(env),
        )

    def test_enumeration_does_not_change_environment_state(self):
        func_type = self.types[0]
        active = system_indices(func_type)[:2]
        env = self.make_env(active, [func_type])
        snapshot = {
            "active": env.active_system_mask.copy(),
            "selected": env.state.selected_system_mask.copy(),
            "ready": env.state.system_ready_time.copy(),
            "net": env.net_cost,
            "active_cost": env.active_cost,
            "version": env.decision_version,
            "changes": env.architecture_change_count,
        }

        legal_architecture_actions(env)

        np.testing.assert_array_equal(env.active_system_mask, snapshot["active"])
        np.testing.assert_array_equal(env.state.selected_system_mask, snapshot["selected"])
        np.testing.assert_array_equal(env.state.system_ready_time, snapshot["ready"])
        self.assertEqual(env.net_cost, snapshot["net"])
        self.assertEqual(env.active_cost, snapshot["active_cost"])
        self.assertEqual(env.decision_version, snapshot["version"])
        self.assertEqual(env.architecture_change_count, snapshot["changes"])

    def test_apply_revalidates_decision_version(self):
        func_type = self.types[0]
        old_idx, new_idx = system_indices(func_type)[:2]
        env = self.make_env([old_idx], [func_type])
        action = ArchitectureAction("add", new_system=new_idx)

        with self.assertRaises(RuntimeError):
            apply_architecture_action(
                env, action, expected_decision_version=env.decision_version - 1
            )
        result = apply_architecture_action(
            env, action, expected_decision_version=env.decision_version
        )
        self.assertTrue(result["valid"])
        self.assertTrue(env.active_system_mask[new_idx])


if __name__ == "__main__":
    unittest.main()
