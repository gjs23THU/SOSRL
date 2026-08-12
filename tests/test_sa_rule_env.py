import unittest
from unittest.mock import patch

import numpy as np

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.rules import scheduling as rule
from sosrl.workflows import scheduler as dqn


class SARuleEnvironmentTests(unittest.TestCase):
    def setUp(self):
        s_type = syn.func_type2idx["S"]
        d_type = syn.func_type2idx["D"]
        i_type = syn.func_type2idx["I"]
        operation_types = [
            [s_type, s_type, d_type, i_type],
            [s_type, d_type, s_type, i_type],
            [s_type, i_type, d_type, s_type],
        ]
        durations = [
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [30, 10, 5, 5],
        ]
        due_times = [100, 60, 55]
        self.mission = []
        for task_idx in range(3):
            operations = [
                syn.Operation(
                    index=op_idx,
                    name=f"Op_{task_idx}_{op_idx}",
                    func_type=operation_types[task_idx][op_idx],
                    duration=durations[task_idx][op_idx],
                    release_time=0,
                )
                for op_idx in range(4)
            ]
            self.mission.append(
                syn.Task(
                    index=task_idx,
                    name=f"Task_{task_idx}",
                    operations=operations,
                    release_time=0,
                    due_time=due_times[task_idx],
                )
            )

        self.arch = (
            next(system for system in env.FULL_SOS if system.index == 0),
            next(system for system in env.FULL_SOS if system.index == 1),
            next(system for system in env.FULL_SOS if system.index == 5),
        )
        self.mission_env = env.MissionEnv(self.arch, self.mission)
        self.obs, _ = self.mission_env.reset()
        self.rule_policy = rule.Rule(self.mission_env)

    def test_state_is_numeric_only_and_observation_is_25_dimensional(self):
        self.assertFalse(hasattr(self.mission_env.state, "arch"))
        self.assertFalse(hasattr(self.mission_env.state, "mission"))
        self.assertFalse(hasattr(self.mission_env.state, "current_time"))
        self.assertFalse(hasattr(self.mission_env.state, "sys_work_idx"))
        self.assertFalse(hasattr(self.mission_env.state, "sos_cost"))
        self.assertEqual(self.obs.shape, (25,))
        self.assertEqual(self.obs.dtype, np.float32)
        self.assertTrue(np.isfinite(self.obs).all())

    def test_cv_preserves_sign_clips_and_maps_nan_to_zero(self):
        self.assertAlmostEqual(
            env.State.cv(np.array([1.0, 3.0], dtype=np.float32)),
            0.5,
            places=6,
        )
        self.assertAlmostEqual(
            env.State.cv(np.array([-1.0, -3.0], dtype=np.float32)),
            -0.5,
            places=6,
        )
        self.assertEqual(
            env.State.cv(np.array([-1.0, 1.0], dtype=np.float32)),
            2.0,
        )
        self.assertEqual(
            env.State.cv(np.array([-1.0, 0.999999], dtype=np.float32)),
            -2.0,
        )
        self.assertEqual(
            env.State.cv(np.array([0.0, 0.0], dtype=np.float32)),
            0.0,
        )

    def test_next_type_load_uses_estimated_current_finish(self):
        np.testing.assert_allclose(
            self.mission_env.state.task_next_type_load,
            np.array([0.0, 10.0, 50.0], dtype=np.float32),
        )

        self.mission_env.state.task_op_idx[0] = 3
        self.mission_env.refresh_derived_state()
        self.assertEqual(float(self.mission_env.state.task_next_type_load[0]), 0.0)

    def test_urgency_state_uses_current_operation_finish(self):
        np.testing.assert_allclose(
            self.mission_env.state.task_earliest_start,
            np.array([20.0, 20.0, 20.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            self.mission_env.state.task_remaining_time,
            np.array([30.0, 60.0, 20.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            self.mission_env.state.task_ttd,
            np.array([70.0, 20.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            self.mission_env.state.task_slack,
            np.array([40.0, -40.0, -15.0], dtype=np.float32),
        )

    def test_assignment_only_action_space_and_round_trip(self):
        self.assertEqual(
            self.mission_env.action_space.n,
            self.mission_env.T * self.mission_env.O * self.mission_env.N,
        )
        action = self.mission_env.encode_assignment(1, 0, 0)
        self.assertEqual(
            self.mission_env.decode_assignment(action),
            {"task_idx": 1, "op_idx": 0, "sys_idx": 0},
        )
        self.assertEqual(
            self.mission_env.valid_action_mask().shape,
            (self.mission_env.action_space.n,),
        )
        self.assertFalse(hasattr(self.mission_env.state, "add_system"))
        self.assertFalse(hasattr(self.mission_env.state, "remove_system"))

    def test_assignment_candidates_are_cached(self):
        cached_mask = self.mission_env.valid_assignment_mask()
        self.assertIs(cached_mask, self.mission_env.assignment_mask)
        self.assertEqual(
            self.mission_env.assignment_start_time.shape,
            cached_mask.shape,
        )
        self.assertEqual(
            self.mission_env.assignment_finish_time.shape,
            cached_mask.shape,
        )

        task_idx, op_idx, sys_idx = 0, 0, 0
        self.assertTrue(cached_mask[task_idx, op_idx, sys_idx])
        self.assertEqual(
            self.mission_env.assignment_times(task_idx, op_idx, sys_idx),
            (
                float(self.mission_env.assignment_start_time[task_idx, op_idx, sys_idx]),
                float(self.mission_env.assignment_finish_time[task_idx, op_idx, sys_idx]),
            ),
        )

        with patch.object(
            self.mission_env,
            "assignment_times",
            side_effect=AssertionError("cache refresh must calculate candidates directly"),
        ) as assignment_times:
            self.mission_env.valid_assignment_mask()
            self.mission_env.valid_action_mask()
            self.mission_env.refresh_assignment_cache()

        assignment_times.assert_not_called()

    def test_allocate_assignment_uses_cached_times(self):
        task_idx, op_idx, sys_idx = 0, 0, 0
        start_time = float(
            self.mission_env.assignment_start_time[task_idx, op_idx, sys_idx]
        )
        finish_time = float(
            self.mission_env.assignment_finish_time[task_idx, op_idx, sys_idx]
        )

        action = self.mission_env.encode_assignment(task_idx, op_idx, sys_idx)
        _, _, _, _, info = self.mission_env.step(action)

        self.assertTrue(info["valid"])
        self.assertEqual(
            float(self.mission_env.state.op_start_time[task_idx, op_idx]),
            start_time,
        )
        self.assertEqual(
            float(self.mission_env.state.op_finish_time[task_idx, op_idx]),
            finish_time,
        )

    def test_four_rules_encode_to_assignment_space(self):
        expected_tasks = {
            "SPT": 0,
            "WINQ": 0,
            "CR": 2,
            "MS": 1,
        }
        for rule_action, rule_name in enumerate(rule.Rule.RULE_NAMES):
            env_action = self.rule_policy.to_env_action(rule_action)
            self.assertTrue(self.mission_env.action_space.contains(env_action))
            decoded = self.mission_env.decode_assignment(env_action)
            self.assertEqual(int(decoded["task_idx"]), expected_tasks[rule_name])
            self.assertEqual(int(decoded["op_idx"]), 0)
            self.assertEqual(int(decoded["sys_idx"]), 0)

    def test_rule_step_refreshes_state_and_preserves_makespan_reward(self):
        old_makespan = float(self.mission_env.state.current_makespan)
        next_obs, reward, terminated, truncated, info = dqn.step_rule_action(
            self.mission_env,
            self.rule_policy,
            0,
        )
        expected_delta = float(self.mission_env.state.current_makespan) - old_makespan
        self.assertAlmostEqual(reward, -expected_delta / self.mission_env.state.M)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(int(self.mission_env.state.task_op_idx[0]), 1)
        self.assertEqual(next_obs.shape, (25,))
        self.assertTrue(np.isfinite(next_obs).all())
        self.assertAlmostEqual(float(next_obs[17]), 1.0 / (self.mission_env.T * self.mission_env.O))

    def test_rule_action_mask_and_dqn_boundary(self):
        np.testing.assert_array_equal(
            dqn.rule_action_mask(self.mission_env, rule.Rule.RULE_NUM),
            np.ones(4, dtype=np.float32),
        )
        self.assertFalse(hasattr(dqn, "cssa"))
        self.assertEqual(rule.Rule.RULE_NUM, 4)

    def test_later_decision_can_schedule_earlier_on_another_system(self):
        mission = [
            syn.Task(
                0,
                "future-S",
                [syn.Operation(0, "future-S-op", syn.func_type2idx["S"], 10, 200)],
                due_time=500,
            ),
            syn.Task(
                1,
                "early-D",
                [syn.Operation(0, "early-D-op", syn.func_type2idx["D"], 10, 0)],
                due_time=500,
            ),
        ]
        architecture = (
            next(system for system in env.FULL_SOS if system.index == 0),
            next(system for system in env.FULL_SOS if system.index == 1),
        )
        mission_env = env.MissionEnv(architecture, mission)

        mission_env.step(mission_env.encode_assignment(0, 0, 0))
        self.assertEqual(float(mission_env.state.op_start_time[0, 0]), 200.0)
        self.assertEqual(
            float(mission_env.valid_assignment_mask()[1, 0, 1]),
            1.0,
        )

        _, _, terminated, truncated, info = mission_env.step(
            mission_env.encode_assignment(1, 0, 1)
        )
        self.assertEqual(float(mission_env.state.op_start_time[1, 0]), 50.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])

    def test_same_system_appends_at_tail_without_overlap(self):
        mission = [
            syn.Task(
                task_idx,
                f"task-{task_idx}",
                [syn.Operation(0, f"op-{task_idx}", syn.func_type2idx["S"], 10, 0)],
                due_time=500,
            )
            for task_idx in range(2)
        ]
        architecture = (next(system for system in env.FULL_SOS if system.index == 0),)
        mission_env = env.MissionEnv(architecture, mission)

        mission_env.step(mission_env.encode_assignment(0, 0, 0))
        mission_env.step(mission_env.encode_assignment(1, 0, 0))

        self.assertEqual(float(mission_env.state.op_start_time[0, 0]), 20.0)
        self.assertEqual(float(mission_env.state.op_finish_time[0, 0]), 30.0)
        self.assertEqual(float(mission_env.state.op_start_time[1, 0]), 30.0)
        self.assertEqual(float(mission_env.state.system_busy_time[0]), 20.0)

    def test_successor_ready_time_preserves_original_release(self):
        mission = [
            syn.Task(
                0,
                "release",
                [
                    syn.Operation(0, "first", syn.func_type2idx["S"], 10, 0),
                    syn.Operation(1, "second", syn.func_type2idx["S"], 10, 100),
                ],
                due_time=500,
            )
        ]
        architecture = (next(system for system in env.FULL_SOS if system.index == 0),)
        mission_env = env.MissionEnv(architecture, mission)

        mission_env.step(mission_env.encode_assignment(0, 0, 0))
        self.assertEqual(float(mission_env.state.operation_ready_time[0, 1]), 100.0)
        mission_env.step(mission_env.encode_assignment(0, 1, 0))
        self.assertEqual(float(mission_env.state.op_start_time[0, 1]), 100.0)

    def test_assignment_outside_system_window_is_invalid(self):
        mission = [
            syn.Task(
                0,
                "outside-window",
                [syn.Operation(0, "late", syn.func_type2idx["D"], 10, 295)],
                due_time=500,
            )
        ]
        architecture = (next(system for system in env.FULL_SOS if system.index == 1),)
        mission_env = env.MissionEnv(architecture, mission)

        self.assertIsNone(mission_env.assignment_times(0, 0, 1))
        self.assertFalse(np.any(mission_env.valid_assignment_mask()))

    def test_dead_end_only_when_no_assignment_remains(self):
        mission = [
            syn.Task(
                0,
                "S-first",
                [syn.Operation(0, "S0", syn.func_type2idx["S"], 100, 0)],
                due_time=500,
            ),
            syn.Task(
                1,
                "S-doomed",
                [syn.Operation(0, "S1", syn.func_type2idx["S"], 100, 0)],
                due_time=500,
            ),
            syn.Task(
                2,
                "D-still-valid",
                [syn.Operation(0, "D0", syn.func_type2idx["D"], 10, 0)],
                due_time=500,
            ),
        ]
        architecture = (
            next(system for system in env.FULL_SOS if system.index == 4),
            next(system for system in env.FULL_SOS if system.index == 1),
        )
        mission_env = env.MissionEnv(architecture, mission)

        _, first_reward, terminated, truncated, info = mission_env.step(
            mission_env.encode_assignment(0, 0, 4)
        )

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertFalse(info["dead_end"])
        self.assertIsNotNone(mission_env.assignment_times(2, 0, 1))

        _, second_reward, terminated, truncated, info = mission_env.step(
            mission_env.encode_assignment(2, 0, 1)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info["success"])
        self.assertTrue(info["dead_end"])
        self.assertAlmostEqual(
            first_reward + second_reward,
            -float(mission_env.state.current_makespan) / mission_env.state.M,
        )

    def test_episode_reward_telescopes_to_normalized_makespan(self):
        mission_env = env.MissionEnv(self.arch, self.mission)
        rule_policy = rule.Rule(mission_env)
        total_reward = 0.0
        for _ in range(mission_env.T * mission_env.O):
            action = rule_policy.to_env_action(3)
            _, reward, terminated, truncated, info = mission_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        self.assertTrue(info["success"])
        self.assertFalse(info["dead_end"])
        self.assertAlmostEqual(
            total_reward,
            -float(mission_env.state.current_makespan) / mission_env.state.M,
        )

    def test_each_fixed_rule_can_run_a_complete_rollout(self):
        for rule_action in range(rule.Rule.RULE_NUM):
            mission_env = env.MissionEnv(self.arch, self.mission)
            mission_env.reset()
            rule_policy = rule.Rule(mission_env)
            terminated = False
            truncated = False
            for _ in range(mission_env.T * mission_env.O):
                env_action = rule_policy.to_env_action(rule_action)
                _, _, terminated, truncated, _ = mission_env.step(env_action)
                if terminated or truncated:
                    break

            self.assertTrue(terminated, msg=rule.Rule.RULE_NAMES[rule_action])
            self.assertFalse(truncated, msg=rule.Rule.RULE_NAMES[rule_action])
            self.assertEqual(int(mission_env.state.task_op_idx.sum()), mission_env.T * mission_env.O)

    def test_due_time_uses_release_time_once(self):
        task = syn.Task(index=0, name="due", operations=[], release_time=5)
        task.operations = [
            syn.Operation(0, "op", syn.func_type2idx["S"], 40, release_time=5)
        ]
        task.set_due_time(tightness=3)
        self.assertEqual(task.due_time, 125)

    def test_mission_samples_task_tightness_from_one_to_configured_upper_bound(self):
        config = dict(syn.CONFIG)
        config["total_task"] = 2
        config["due_time_tightness"] = 3.0

        with patch.object(syn.random, "uniform", side_effect=[1.25, 2.75]) as uniform:
            mission = syn.build_mission_from_config(config)

        self.assertEqual(
            [call.args for call in uniform.call_args_list],
            [(1.0, 3.0), (1.0, 3.0)],
        )
        self.assertEqual(
            mission[0].due_time,
            int(sum(op.duration for op in mission[0].operations) * 1.25),
        )
        self.assertEqual(
            mission[1].due_time,
            int(sum(op.duration for op in mission[1].operations) * 2.75),
        )


class SingleScenarioV2IntegrationTests(unittest.TestCase):
    def test_seed_4_fixed_rules_complete_without_dead_end(self):
        dqn.set_seed(4)
        pool = dqn.ScenarioPool(
            size=1,
            selected_system_num=15,
            min_system_num=3,
            max_system_num=22,
            cost_limit=8000,
            shared_mission=True,
        )
        _, architecture, mission = pool.get(0)
        makespans = {}

        for rule_action, rule_name in enumerate(rule.Rule.RULE_NAMES):
            mission_env = env.MissionEnv(architecture, mission)
            rule_policy = rule.Rule(mission_env)
            info = {"success": False, "dead_end": False}
            for _ in range(mission_env.T * mission_env.O):
                action = rule_policy.to_env_action(rule_action)
                _, _, terminated, truncated, info = mission_env.step(action)
                if terminated or truncated:
                    break

            self.assertTrue(info["success"], msg=rule_name)
            self.assertFalse(info["dead_end"], msg=rule_name)
            self.assertEqual(
                int(mission_env.state.task_op_idx.sum()),
                mission_env.T * mission_env.O,
            )
            makespans[rule_name] = float(mission_env.state.current_makespan)

        self.assertEqual(
            makespans,
            {"SPT": 525.0, "WINQ": 680.0, "CR": 870.0, "MS": 872.0},
        )


if __name__ == "__main__":
    unittest.main()
