import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import archrule
import dqn
import env
import hrldqn
import intdqn
import rule
import syn


class NStepReplayTests(unittest.TestCase):
    def transition(self, index, reward, done=False):
        return (
            np.array([index], dtype=np.float32),
            index % 2,
            reward,
            np.array([index + 1], dtype=np.float32),
            done,
            np.ones(2, dtype=np.float32) if not done else np.zeros(2, dtype=np.float32),
        )

    def test_five_step_accumulator_uses_discounted_return(self):
        accumulator = hrldqn.NStepAccumulator(n_step=3, gamma=0.5)
        self.assertEqual(accumulator.append(self.transition(0, 1.0)), [])
        self.assertEqual(accumulator.append(self.transition(1, 2.0)), [])
        emitted = accumulator.append(self.transition(2, 3.0))

        self.assertEqual(len(emitted), 1)
        self.assertAlmostEqual(emitted[0][2], 1.0 + 0.5 * 2.0 + 0.25 * 3.0)
        self.assertAlmostEqual(emitted[0][6], 0.5**3)

    def test_terminal_flushes_all_short_transitions(self):
        accumulator = hrldqn.NStepAccumulator(n_step=5, gamma=0.9)
        accumulator.append(self.transition(0, 1.0))
        emitted = accumulator.append(self.transition(1, 2.0, done=True))

        self.assertEqual(len(emitted), 2)
        self.assertTrue(all(item[4] for item in emitted))
        self.assertAlmostEqual(emitted[0][2], 1.0 + 0.9 * 2.0)
        self.assertAlmostEqual(emitted[1][2], 2.0)

    def test_architecture_target_handles_empty_next_mask(self):
        config = hrldqn.HRLConfig(
            hidden_dim=8,
            batch_size=1,
            min_buffer_size=1,
            buffer_size=4,
            device="cpu",
        )
        agent = hrldqn.ArchitectureDQNAgent(obs_dim=3, config=config)
        agent.replay.add(
            np.zeros(3),
            0,
            1.0,
            np.ones(3),
            False,
            np.zeros(archrule.ArchitectureRule.RULE_NUM),
            0.9,
        )

        loss = agent.learn()

        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))


class HRLIntegrationTests(unittest.TestCase):
    def make_task(self, function_type):
        return syn.Task(
            index=0,
            name="task",
            operations=[
                syn.Operation(0, "op", function_type, duration=10, release_time=0)
            ],
            due_time=1000,
        )

    def make_agents(self, mission_env):
        hrl_config = hrldqn.HRLConfig(
            hidden_dim=8,
            batch_size=1,
            min_buffer_size=1,
            buffer_size=32,
            n_step=2,
            device="cpu",
        )
        architecture_agent = hrldqn.ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            hrl_config,
        )
        scheduler_config = dqn.DQNConfig(
            hidden_dim=8,
            batch_size=1,
            min_buffer_size=1,
            buffer_size=32,
            device="cpu",
        )
        scheduler_agent = dqn.DQNAgent(25, rule.Rule.RULE_NUM, scheduler_config)
        for network in (architecture_agent.q_net, scheduler_agent.q_net):
            for parameter in network.parameters():
                torch.nn.init.zeros_(parameter)
        architecture_agent.target_net.load_state_dict(architecture_agent.q_net.state_dict())
        scheduler_agent.target_net.load_state_dict(scheduler_agent.q_net.state_dict())
        return architecture_agent, scheduler_agent

    def test_two_policy_cycle_rescues_missing_capability(self):
        mission = [self.make_task(syn.func_type2idx["D"])]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        architecture_agent, scheduler_agent = self.make_agents(mission_env)

        result = hrldqn.run_episode(
            mission_env,
            architecture_agent,
            scheduler_agent,
            architecture_epsilon=0.0,
            scheduler_epsilon=0.0,
            update_architecture=False,
            update_scheduler=False,
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["dead_end"])
        self.assertEqual(int(mission_env.state.task_op_idx.sum()), 1)
        self.assertEqual(architecture_agent.action_dim, 6)
        self.assertEqual(scheduler_agent.action_dim, 4)

    def test_architecture_reward_uses_ten_times_makespan_delta(self):
        mission = [self.make_task(syn.func_type2idx["S"])]
        mission_env = env.MissionEnv(
            [env.FULL_SOS[0]],
            mission,
            adaptive=True,
        )
        mission_env.state.current_makespan = 5.0
        reward = hrldqn.architecture_reward(
            mission_env,
            old_makespan=0.0,
            old_cost=mission_env.net_cost,
            changed=False,
            success=False,
            dead_end=False,
        )

        self.assertAlmostEqual(reward, -5.0)

    def test_dead_end_penalty_is_shared_by_both_reward_functions(self):
        mission = [self.make_task(syn.func_type2idx["S"])]
        mission_env = env.MissionEnv(
            [env.FULL_SOS[0]],
            mission,
            adaptive=True,
        )
        architecture_value = hrldqn.architecture_reward(
            mission_env,
            old_makespan=0.0,
            old_cost=mission_env.net_cost,
            changed=False,
            success=False,
            dead_end=True,
        )

        self.assertEqual(architecture_value, -2.0)
        self.assertEqual(
            hrldqn.scheduler_reward(0.0, success=False, dead_end=True),
            -2.0,
        )

    def test_combined_checkpoint_restores_both_policy_networks(self):
        mission = [self.make_task(syn.func_type2idx["S"])]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        architecture_agent, scheduler_agent = self.make_agents(mission_env)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = hrldqn.save_combined_checkpoint(
                Path(temp_dir) / "hrl.pt",
                architecture_agent,
                scheduler_agent,
                training_state={"stage": "finetune", "episode": 3},
            )
            loaded_arch, loaded_scheduler, checkpoint = (
                hrldqn.load_combined_checkpoint(path, device="cpu")
            )

        self.assertEqual(loaded_arch.action_dim, 6)
        self.assertEqual(loaded_scheduler.action_dim, 4)
        self.assertEqual(checkpoint["training_state"]["stage"], "finetune")
        for expected, actual in zip(
            architecture_agent.q_net.parameters(),
            loaded_arch.q_net.parameters(),
        ):
            torch.testing.assert_close(expected, actual)

    def test_flat_intdqn_checkpoint_can_be_loaded_for_paired_baseline(self):
        config = intdqn.IntDQNConfig(
            hidden_dim=8,
            buffer_size=8,
            device="cpu",
        )
        agent = intdqn.IntDQNAgent(obs_dim=5, action_dim=7, config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = agent.save_checkpoint(Path(temp_dir) / "flat.pt")
            loaded, _ = intdqn.IntDQNAgent.load_checkpoint(path, device="cpu")

        self.assertEqual(loaded.obs_dim, 5)
        self.assertEqual(loaded.action_dim, 7)


if __name__ == "__main__":
    unittest.main()
