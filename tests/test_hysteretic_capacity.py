import unittest

from sosrl import domain as syn
from sosrl.baselines.hysteretic_capacity import (
    HystereticCapacityConfig,
    HystereticCapacityProvider,
    capability_pressures,
)
from sosrl.environment import FULL_SOS, MissionEnv
from sosrl.gp.architecture import legal_architecture_actions


def systems_of_type(func_type: int) -> list[int]:
    return [
        int(system.index)
        for system in FULL_SOS
        if int(system.func_type) == int(func_type)
    ]


def one_operation_tasks(func_type: int, durations: list[int]) -> list[syn.Task]:
    return [
        syn.Task(
            index=index,
            name=f"task-{index}",
            operations=[
                syn.Operation(
                    index=0,
                    name=f"op-{index}",
                    func_type=int(func_type),
                    duration=int(duration),
                    release_time=0,
                )
            ],
            due_time=5000,
        )
        for index, duration in enumerate(durations)
    ]


class HystereticCapacityProviderTest(unittest.TestCase):
    def setUp(self):
        self.types = sorted({int(system.func_type) for system in FULL_SOS})
        self.provider = HystereticCapacityProvider()

    def make_env(
        self,
        active: list[int],
        func_type: int,
        durations: list[int],
        *,
        budget: float = 8000.0,
    ):
        return MissionEnv(
            [FULL_SOS[index] for index in active],
            one_operation_tasks(func_type, durations),
            adaptive=True,
            budget=budget,
            refund_rate=0.8,
        )

    def test_config_requires_a_strict_hysteresis_band(self):
        with self.assertRaises(ValueError):
            HystereticCapacityConfig(lower_threshold=0.9, upper_threshold=0.9)
        with self.assertRaises(ValueError):
            HystereticCapacityConfig(lower_threshold=-0.1, upper_threshold=0.9)
        with self.assertRaises(ValueError):
            HystereticCapacityConfig(budget_mode="hard")

    def test_pressure_uses_remaining_demand_over_remaining_window(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type, [25] * 6)

        pressures = capability_pressures(env)

        remaining_window = (
            float(FULL_SOS[active].available_until)
            - float(FULL_SOS[active].available_from)
        )
        self.assertAlmostEqual(
            pressures[func_type],
            150.0 / remaining_window,
        )

    def test_band_state_keeps_the_architecture(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type, [25] * 6)

        decision = self.provider.act(env)

        self.assertEqual(decision.action.kind, "keep")
        self.assertEqual(decision.diagnostics["trigger"], "band_keep")

    def test_threshold_boundaries_are_inclusive(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]

        low = self.make_env([active], func_type, [100])
        low_decision = self.provider.decide(low)
        self.assertEqual(capability_pressures(low)[func_type], 0.40)
        self.assertEqual(low_decision.action.kind, "keep")
        self.assertEqual(low_decision.diagnostics["trigger"], "contract")

        high = self.make_env([active], func_type, [225])
        high_decision = self.provider.decide(high)
        self.assertEqual(capability_pressures(high)[func_type], 0.90)
        self.assertEqual(high_decision.action.kind, "add")
        self.assertEqual(high_decision.diagnostics["trigger"], "expand")

    def test_upper_threshold_prefers_add_over_non_dominating_replace(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type, [25] * 9)

        decision = self.provider.act(env)

        self.assertEqual(decision.action.kind, "add")
        self.assertEqual(decision.diagnostics["trigger"], "expand")
        self.assertEqual(decision.diagnostics["replace_reason"], "primary_add")

    def test_low_pressure_uses_safe_delete_before_downgrade_replace(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[:2]
        env = self.make_env(active, func_type, [25] * 4)

        decision = self.provider.act(env)

        self.assertEqual(decision.action.kind, "remove")
        self.assertEqual(decision.diagnostics["trigger"], "contract")
        self.assertEqual(decision.diagnostics["replace_reason"], "safe_delete")

    def test_low_pressure_downgrades_when_deleting_the_only_system_is_unsafe(self):
        func_type = self.types[1]
        candidates = systems_of_type(func_type)
        active = max(
            candidates,
            key=lambda index: (
                float(FULL_SOS[index].available_until)
                - float(FULL_SOS[index].available_from)
            ),
        )
        env = self.make_env([active], func_type, [25] * 8)

        decision = self.provider.act(env)

        self.assertEqual(decision.action.kind, "replace")
        self.assertEqual(decision.diagnostics["trigger"], "contract")
        self.assertEqual(decision.diagnostics["replace_reason"], "delete_unsafe")
        self.assertLess(decision.diagnostics["capacity_delta"], 0.0)
        self.assertLess(decision.diagnostics["net_cost_delta"], 0.0)

    def test_missing_capability_uses_emergency_add(self):
        required_type, other_type = self.types[:2]
        active_other = systems_of_type(other_type)[0]
        env = self.make_env([active_other], required_type, [10])

        pressures = capability_pressures(env)
        self.assertEqual(pressures[required_type], float("inf"))
        unused_type = next(
            value for value in self.types if value not in {required_type, other_type}
        )
        self.assertEqual(pressures[unused_type], 0.0)

        legal = legal_architecture_actions(env)
        decision = self.provider.decide(env)

        self.assertTrue(decision.valid)
        self.assertIn(decision.action, legal)
        self.assertEqual(decision.action.kind, "add")
        self.assertEqual(decision.diagnostics["trigger"], "emergency")
        self.assertEqual(decision.diagnostics["target_capability"], required_type)
        self.assertEqual(
            {
                "trigger",
                "target_capability",
                "pre_pressure",
                "post_pressure",
                "capacity_delta",
                "net_cost_delta",
                "replace_reason",
            }
            - set(decision.diagnostics),
            set(),
        )

    def test_emergency_replace_must_dominate_an_available_add(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type, [10])
        env.state.system_ready_time[active] = (
            float(FULL_SOS[active].available_until) + 1.0
        )
        env.decision_version += 1
        legal = legal_architecture_actions(env)
        self.assertTrue(any(action.kind == "add" for action in legal))

        decision = self.provider.decide(env)

        self.assertEqual(decision.action.kind, "replace")
        self.assertIn(decision.action, legal)
        self.assertEqual(decision.diagnostics["trigger"], "emergency")
        self.assertEqual(decision.diagnostics["replace_reason"], "dominates_add")

    def test_soft_budget_does_not_screen_expansion_and_tracks_over_budget(self):
        func_type = self.types[1]
        active = systems_of_type(func_type)[0]
        env = self.make_env([active], func_type, [225], budget=300.0)
        self.assertFalse(env.ever_over_budget)

        decision = self.provider.act(env)

        self.assertEqual(decision.action.kind, "add")
        self.assertTrue(env.ever_over_budget)
        self.assertGreater(env.peak_net_cost, env.budget)


if __name__ == "__main__":
    unittest.main()
