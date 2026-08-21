import unittest

import numpy as np

from sosrl import domain as syn
from sosrl.environment import FULL_SOS, MissionEnv
from sosrl.gp.architecture import ArchitectureAction, legal_architecture_actions
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.provider import (
    FixedArchitectureProvider,
    GPArchitectureProvider,
    RandomConcreteArchitectureProvider,
)


def systems_of_type(func_type):
    return [int(system.index) for system in FULL_SOS if int(system.func_type) == int(func_type)]


def mission(func_type):
    return [
        syn.Task(
            0,
            "task",
            [syn.Operation(0, "op", int(func_type), 10, 0)],
            due_time=100,
        )
    ]


class GPArchitectureProviderTest(unittest.TestCase):
    def setUp(self):
        self.types = sorted({int(system.func_type) for system in FULL_SOS})

    def make_env(self, active, required):
        return MissionEnv(
            [FULL_SOS[index] for index in active],
            mission(required),
            adaptive=True,
        )

    def test_constant_score_uses_keep_tie_break(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type)
        version = env.decision_version
        provider = GPArchitectureProvider(lambda *values: 0.0)

        decision = provider.act(env)

        self.assertTrue(decision.valid)
        self.assertEqual(decision.action, ArchitectureAction("keep"))
        self.assertFalse(decision.changed)
        self.assertEqual(env.decision_version, version)

    def test_score_selects_concrete_add_and_executes_before_return(self):
        required_type, other_type = self.types[:2]
        active_other = systems_of_type(other_type)[0]
        env = self.make_env([active_other], required_type)
        names = feature_names_for_preset("system_delta")
        add_index = names.index("added_cost_ratio")
        provider = GPArchitectureProvider(
            lambda *values: values[add_index]
        )
        before = env.active_system_mask.copy()

        decision = provider.act(env)

        self.assertTrue(decision.valid)
        self.assertEqual(decision.action.kind, "add")
        self.assertFalse(before[decision.action.new_system])
        self.assertTrue(env.active_system_mask[decision.action.new_system])
        self.assertGreater(env.decision_version, 0)

    def test_fixed_provider_dead_ends_when_keep_is_illegal(self):
        required_type, other_type = self.types[:2]
        env = self.make_env(
            [systems_of_type(other_type)[0]], required_type
        )

        decision = FixedArchitectureProvider().act(env)

        self.assertFalse(decision.valid)
        self.assertGreater(decision.candidate_count, 0)

    def test_random_provider_executes_only_a_legal_candidate(self):
        func_type = self.types[0]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type)
        legal = set(legal_architecture_actions(env))

        decision = RandomConcreteArchitectureProvider(seed=7).act(env)

        self.assertTrue(decision.valid)
        self.assertIn(decision.action, legal)
        self.assertEqual(decision.candidate_count, len(legal))


if __name__ == "__main__":
    unittest.main()
