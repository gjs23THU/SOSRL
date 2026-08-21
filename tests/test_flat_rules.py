import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sosrl import domain as syn
from sosrl import environment as env
from sosrl.baselines import flat_rules
from sosrl.rl.agent import ArchitectureDQNAgent, DQNAgent, FlatRuleDQNAgent
from sosrl.rl.config import DQNConfig, HRLConfig
from sosrl.rules import architecture as archrule
from sosrl.rules import scheduling as rule
from sosrl.workflows import evaluation, hierarchical


class ScenarioPoolStub:
    def __init__(self, scenario):
        self.scenarios = [scenario]

    def get(self, index):
        return self.scenarios[int(index) % len(self.scenarios)]

    def sample(self):
        return self.scenarios[0]


class FlatRuleDQNTests(unittest.TestCase):
    def make_task(self, function_types, durations=None):
        durations = durations or [10] * len(function_types)
        return syn.Task(
            index=0,
            name="task",
            operations=[
                syn.Operation(
                    index,
                    f"op-{index}",
                    function_type,
                    duration=durations[index],
                    release_time=0,
                )
                for index, function_type in enumerate(function_types)
            ],
            due_time=1000,
        )

    def make_flat_agent(self, mission_env, **overrides):
        values = {
            "hidden_dim": 8,
            "batch_size": 1,
            "min_buffer_size": 1,
            "buffer_size": 32,
            "n_step": 2,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "device": "cpu",
        }
        values.update(overrides)
        config = HRLConfig(**values)
        return FlatRuleDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            config,
        )

    def zero_network(self, agent):
        for parameter in agent.q_net.parameters():
            torch.nn.init.zeros_(parameter)
        agent.target_net.load_state_dict(agent.q_net.state_dict())

    def test_joint_action_encode_decode_covers_all_rule_pairs(self):
        actions = set()
        for architecture_action in range(6):
            for scheduling_action in range(4):
                joint_action = flat_rules.encode_joint_action(
                    architecture_action,
                    scheduling_action,
                )
                actions.add(joint_action)
                self.assertEqual(
                    flat_rules.decode_joint_action(joint_action),
                    (architecture_action, scheduling_action),
                )

        self.assertEqual(actions, set(range(24)))
        self.assertEqual(FlatRuleDQNAgent.ACTION_DIM, 24)

    def test_joint_mask_repeats_each_architecture_entry_four_times(self):
        mission = [self.make_task([syn.func_type2idx["D"]])]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        architecture_policy = archrule.ArchitectureRule(mission_env)

        architecture_mask = architecture_policy.action_mask()
        joint_mask = flat_rules.joint_action_mask(architecture_policy)

        np.testing.assert_array_equal(joint_mask, np.repeat(architecture_mask, 4))
        self.assertEqual(joint_mask.shape, (24,))

    def test_joint_step_matches_manual_rule_execution(self):
        mission = [self.make_task([syn.func_type2idx["S"]])]
        joint_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        manual_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        action = flat_rules.encode_joint_action(0, 3)

        next_obs, reward, terminated, _, _ = flat_rules.step_joint_action(
            joint_env,
            archrule.ArchitectureRule(joint_env),
            rule.Rule(joint_env),
            action,
        )

        old_makespan = manual_env.state.current_makespan
        old_cost = manual_env.net_cost
        arch_info = archrule.ArchitectureRule(manual_env).apply(0)
        env_action = rule.Rule(manual_env).to_env_action(3)
        _, _, manual_terminated, _, manual_info = manual_env.step(env_action)
        expected_reward = hierarchical.architecture_reward(
            manual_env,
            old_makespan,
            old_cost,
            arch_info["changed"],
            manual_info["success"],
            manual_info["dead_end"],
        )

        self.assertEqual(terminated, manual_terminated)
        self.assertEqual(reward, expected_reward)
        np.testing.assert_array_equal(
            joint_env.state.op_assign_sys,
            manual_env.state.op_assign_sys,
        )
        np.testing.assert_allclose(next_obs, manual_env.architecture_observation())
        self.assertEqual(joint_env.net_cost, manual_env.net_cost)
        self.assertEqual(
            joint_env.state.current_makespan,
            manual_env.state.current_makespan,
        )

    def test_missing_capability_is_rescued_by_joint_action(self):
        mission = [self.make_task([syn.func_type2idx["D"]])]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        action = flat_rules.encode_joint_action(1, 0)

        _, _, terminated, _, info = flat_rules.step_joint_action(
            mission_env,
            archrule.ArchitectureRule(mission_env),
            rule.Rule(mission_env),
            action,
        )

        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertEqual(info["architecture_rule"], "ADD_CAPABILITY")
        self.assertEqual(int(mission_env.state.task_op_idx.sum()), 1)

    def test_joint_remove_rule_preserves_refund_and_net_cost_semantics(self):
        mission = [self.make_task([syn.func_type2idx["S"]])]
        systems = [env.FULL_SOS[0], env.FULL_SOS[2]]
        mission_env = env.MissionEnv(systems, mission, adaptive=True)
        initial_cost = mission_env.net_cost
        action = flat_rules.encode_joint_action(4, 0)

        _, _, terminated, _, info = flat_rules.step_joint_action(
            mission_env,
            archrule.ArchitectureRule(mission_env),
            rule.Rule(mission_env),
            action,
        )

        expected_refund = 0.8 * env.FULL_SOS[2].cost
        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertEqual(info["architecture_rule"], "REMOVE_REDUNDANT")
        self.assertAlmostEqual(mission_env.total_refund, expected_refund)
        self.assertAlmostEqual(mission_env.net_cost, initial_cost - expected_refund)

    def test_unrecoverable_joint_state_reports_dead_end_without_a_step(self):
        mission = [
            self.make_task(
                [syn.func_type2idx["S"]],
                durations=[2000],
            )
        ]
        mission_env = env.MissionEnv([], mission, adaptive=True)
        agent = self.make_flat_agent(mission_env)

        result = flat_rules.run_flat_rule_episode(
            mission_env,
            agent,
            epsilon=0.0,
            update_agent=False,
        )

        self.assertTrue(result["dead_end"])
        self.assertFalse(result["success"])
        self.assertEqual(result["environment_steps"], 0)

    def test_flat_rule_target_handles_empty_next_mask(self):
        config = HRLConfig(
            hidden_dim=8,
            batch_size=1,
            min_buffer_size=1,
            buffer_size=4,
            device="cpu",
        )
        agent = FlatRuleDQNAgent(obs_dim=3, config=config)
        agent.replay.add(
            np.zeros(3),
            0,
            1.0,
            np.ones(3),
            False,
            np.zeros(24),
            0.9,
        )

        loss = agent.learn()

        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))

    def test_flat_rule_checkpoint_round_trip_preserves_24_actions(self):
        config = HRLConfig(hidden_dim=8, buffer_size=8, device="cpu")
        agent = FlatRuleDQNAgent(obs_dim=112, config=config)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = agent.save_checkpoint(
                Path(temp_dir) / "flat_rules.pt",
                training_state={"actual_environment_steps": 120},
            )
            loaded, checkpoint = FlatRuleDQNAgent.load_checkpoint(
                path,
                device="cpu",
            )

        self.assertEqual(loaded.obs_dim, 112)
        self.assertEqual(loaded.action_dim, 24)
        self.assertEqual(checkpoint["training_state"]["actual_environment_steps"], 120)
        for expected, actual in zip(agent.q_net.parameters(), loaded.q_net.parameters()):
            torch.testing.assert_close(expected, actual)

    def test_environment_step_budget_stops_after_complete_episode(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task([s_type, s_type])]
        scenario = ((env.FULL_SOS[0],), mission, "test")
        config = HRLConfig(
            episodes=1,
            hidden_dim=8,
            batch_size=64,
            min_buffer_size=64,
            buffer_size=64,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
        )

        _, history = flat_rules.train_flat_rules(
            config,
            ScenarioPoolStub(scenario),
            max_env_steps=3,
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]["cumulative_environment_steps"], 4)
        self.assertTrue(all(row["success"] for row in history))
        self.assertTrue(all(row["assigned_ops"] == 2 for row in history))

    def test_flat_and_hrl_evaluation_use_identical_scenario_hashes(self):
        s_type = syn.func_type2idx["S"]
        mission = [self.make_task([s_type])]
        scenarios = [((env.FULL_SOS[0],), mission, "evaluation")]
        probe = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        config = HRLConfig(hidden_dim=8, device="cpu")
        flat_agent = FlatRuleDQNAgent(
            probe.architecture_observation_space.shape[0],
            config,
        )
        architecture_agent = ArchitectureDQNAgent(
            probe.architecture_observation_space.shape[0],
            config,
        )
        scheduler_agent = DQNAgent(
            25,
            4,
            DQNConfig(hidden_dim=8, device="cpu"),
        )
        for agent in (flat_agent, architecture_agent, scheduler_agent):
            self.zero_network(agent)

        flat_rows = flat_rules.evaluate_flat_rules(flat_agent, scenarios)
        hrl_rows = hierarchical.evaluate_hrl(
            architecture_agent,
            scheduler_agent,
            scenarios,
        )
        hrl_rows[0]["model"] = "hrl"

        self.assertEqual(flat_rows[0]["scenario_hash"], hrl_rows[0]["scenario_hash"])
        comparisons = evaluation.paired_adaptive_comparisons(
            hrl_rows + flat_rows,
            candidate_labels=["flat_rule_dqn"],
        )
        self.assertEqual(comparisons[0]["paired_scenarios"], 1)


if __name__ == "__main__":
    unittest.main()
