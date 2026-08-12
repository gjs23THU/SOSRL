import unittest

import numpy as np

import intenv
import syn


class IntegratedEnvironmentTests(unittest.TestCase):
    def make_task(self, index, operation_types, durations):
        operations = []
        for op_idx, func_type in enumerate(operation_types):
            operations.append(
                syn.Operation(
                    index=op_idx,
                    name=f"op-{index}-{op_idx}",
                    func_type=func_type,
                    duration=durations[op_idx],
                    release_time=0,
                )
            )
        return syn.Task(
            index=index,
            name=f"task-{index}",
            operations=operations,
            due_time=500,
        )

    def test_encode_decode_round_trip(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type], [10])]
        mission_env = intenv.IntEnv(mission)

        action = mission_env.encode_asg(0, 0, 0)

        self.assertEqual(mission_env.decode_asg(action), (0, 0, 0))

    def test_first_use_cost_is_charged_once(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type], [10]),
            self.make_task(1, [s_type], [10]),
        ]
        system = intenv.FULL_SOS[0]
        mission_env = intenv.IntEnv(mission)

        _, first_reward, _, _, first_info = mission_env.step(
            mission_env.encode_asg(0, 0, 0)
        )
        _, second_reward, terminated, _, second_info = mission_env.step(
            mission_env.encode_asg(1, 0, 0)
        )

        self.assertTrue(first_info["first_use"])
        self.assertEqual(first_info["cost_delta"], float(system.cost))
        self.assertFalse(second_info["first_use"])
        self.assertEqual(second_info["cost_delta"], 0.0)
        self.assertEqual(mission_env.state.cur_cost, float(system.cost))
        self.assertTrue(terminated)
        expected_total = -(
            mission_env.state.cur_makespan / mission_env.state.M
            + mission_env.state.cur_cost / mission_env.state.total_cost
        )
        self.assertAlmostEqual(first_reward + second_reward, expected_total)

    def test_operation_order_and_ready_time(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type, s_type], [10, 10])]
        mission_env = intenv.IntEnv(mission)

        mask = mission_env.valid_assignment_mask()
        self.assertTrue(mask[0, 0, 0])
        self.assertFalse(mask[0, 1, 0])

        mission_env.step(mission_env.encode_asg(0, 0, 0))
        first_finish = float(mission_env.state.op_finish_time[0, 0])

        mask = mission_env.valid_assignment_mask()
        self.assertFalse(mask[0, 0, 0])
        self.assertTrue(mask[0, 1, 0])
        mission_env.step(mission_env.encode_asg(0, 1, 0))
        self.assertGreaterEqual(
            float(mission_env.state.op_start_time[0, 1]),
            first_finish,
        )

    def test_system_window_is_checked(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type], [200])]
        short_window_system = intenv.FULL_SOS[16]
        mission_env = intenv.IntEnv(mission)
        self.assertIsNone(
            mission_env.assignment_times(
                0,
                0,
                short_window_system.index,
            )
        )

    def test_reset_clears_schedule_and_selected_architecture(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task(0, [s_type], [10])]
        mission_env = intenv.IntEnv(mission)
        mission_env.step(mission_env.encode_asg(0, 0, 0))

        obs, info = mission_env.reset()

        self.assertEqual(obs.shape, mission_env.observation_space.shape)
        self.assertEqual(obs.dtype, np.float32)
        self.assertFalse(np.any(mission_env.state.select_sys_mask))
        self.assertEqual(mission_env.state.cur_cost, 0.0)
        self.assertEqual(int(mission_env.state.task_op_idx[0]), 0)
        self.assertFalse(info["dead_end"])


if __name__ == "__main__":
    unittest.main()
