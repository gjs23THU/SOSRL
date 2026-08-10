import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import dqn
import evaluate_independent
import hrule
import main
import rule


class TrainingCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_model_and_training_state(self):
        config = dqn.DQNConfig(hidden_dim=8, buffer_size=16, device="cpu")
        agent = dqn.DQNAgent(obs_dim=5, action_dim=4, config=config)
        observation = np.arange(5, dtype=np.float32)
        action_mask = np.ones(4, dtype=np.float32)

        with torch.no_grad():
            expected = agent.q_net(torch.from_numpy(observation).unsqueeze(0)).clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = agent.save_checkpoint(
                Path(temp_dir) / "model.pt",
                training_state={"episodes_completed": 3, "epsilon": 0.5},
            )
            loaded_agent, checkpoint = dqn.DQNAgent.load_checkpoint(
                checkpoint_path,
                device="cpu",
            )

        with torch.no_grad():
            actual = loaded_agent.q_net(torch.from_numpy(observation).unsqueeze(0))

        torch.testing.assert_close(actual, expected)
        self.assertEqual(loaded_agent.obs_dim, 5)
        self.assertEqual(loaded_agent.action_dim, 4)
        self.assertEqual(checkpoint["training_state"]["episodes_completed"], 3)

    def test_huang_checkpoint_preserves_rule_set_and_five_actions(self):
        config = dqn.DQNConfig(
            hidden_dim=8,
            buffer_size=16,
            rule_set="huang",
            device="cpu",
        )
        agent = dqn.DQNAgent(obs_dim=5, action_dim=5, config=config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = agent.save_checkpoint(Path(temp_dir) / "model.pt")
            loaded_agent, _ = dqn.DQNAgent.load_checkpoint(
                checkpoint_path,
                device="cpu",
            )

        self.assertEqual(loaded_agent.config.rule_set, "huang")
        self.assertEqual(loaded_agent.action_dim, 5)

    def test_checkpoint_without_rule_set_defaults_to_standard(self):
        config = dqn.DQNConfig(hidden_dim=8, buffer_size=16, device="cpu")
        agent = dqn.DQNAgent(obs_dim=5, action_dim=4, config=config)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = agent.save_checkpoint(Path(temp_dir) / "model.pt")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            checkpoint["config"].pop("rule_set")
            torch.save(checkpoint, checkpoint_path)
            loaded_agent, _ = dqn.DQNAgent.load_checkpoint(
                checkpoint_path,
                device="cpu",
            )

        self.assertEqual(loaded_agent.config.rule_set, "standard")
        self.assertEqual(loaded_agent.action_dim, 4)


class RuleSetCompatibilityTests(unittest.TestCase):
    def test_rule_set_names_resolve_to_expected_classes(self):
        self.assertIs(dqn.get_rule_class("standard"), rule.Rule)
        self.assertIs(dqn.get_rule_class("huang"), hrule.HRule)
        with self.assertRaisesRegex(ValueError, "Unknown rule set"):
            dqn.get_rule_class("unknown")

    def test_action_mask_uses_selected_rule_count(self):
        mission_env = type(
            "MissionEnvStub",
            (),
            {"valid_assignment_mask": lambda self: np.ones(1, dtype=bool)},
        )()

        self.assertEqual(dqn.rule_action_mask(mission_env, 4).shape, (4,))
        self.assertEqual(dqn.rule_action_mask(mission_env, 5).shape, (5,))

    def test_evaluation_rejects_checkpoint_rule_dimension_mismatch(self):
        agent = type(
            "AgentStub",
            (),
            {
                "config": dqn.DQNConfig(rule_set="huang", device="cpu"),
                "action_dim": 4,
            },
        )()

        with self.assertRaisesRegex(ValueError, "action dimension"):
            dqn.evaluate_dqn(agent, scenario_pool=None, episodes=0)

    def test_main_accepts_huang_rule_set(self):
        with patch("sys.argv", ["main.py", "--rule-set", "huang"]):
            args = main.parse_args()

        self.assertEqual(args.rule_set, "huang")

    def test_main_uses_a_separate_evaluation_seed(self):
        with patch("sys.argv", ["main.py", "--eval-seed", "99"]):
            args = main.parse_args()

        self.assertEqual(args.eval_seed, 99)


class IndependentEvaluationTests(unittest.TestCase):
    def test_independent_pool_never_reuses_a_shared_training_mission(self):
        config = dqn.DQNConfig(
            shared_mission=True,
            selected_system_num=15,
            min_system_num=3,
            max_system_num=22,
            cost_limit=8000,
        )

        with patch.object(
            evaluate_independent.dqn,
            "set_seed",
        ) as set_seed, patch.object(
            evaluate_independent.dqn,
            "ScenarioPool",
            return_value="pool",
        ) as scenario_pool:
            actual = evaluate_independent.build_independent_pool(
                config,
                episodes=25,
                eval_seed=123,
            )

        self.assertEqual(actual, "pool")
        set_seed.assert_called_once_with(123)
        scenario_pool.assert_called_once_with(
            size=25,
            selected_system_num=15,
            min_system_num=3,
            max_system_num=22,
            cost_limit=8000,
            shared_mission=False,
            mission=None,
        )

    def test_paired_comparison_requires_matching_scenarios(self):
        left = {
            "scenario_hash": "left",
            "success": True,
            "makespan": 10.0,
        }
        right = {
            "scenario_hash": "right",
            "success": True,
            "makespan": 11.0,
        }

        with self.assertRaisesRegex(ValueError, "matching scenarios"):
            evaluate_independent.paired_rows({"SIG": [left], "MIG": [right]})


class SharedMissionScenarioPoolTests(unittest.TestCase):
    def test_shared_mission_builds_once_and_varies_architecture(self):
        mission = object()
        architectures = [("arch-0",), ("arch-1",), ("arch-2",)]

        with patch.object(
            dqn.syn,
            "build_mission_from_config",
            return_value=mission,
        ) as build_mission, patch.object(
            dqn.ScenarioPool,
            "sample_arch",
            side_effect=architectures,
        ):
            pool = dqn.ScenarioPool(size=3, shared_mission=True)

        build_mission.assert_called_once_with(dqn.syn.CONFIG)
        self.assertIs(pool.mission, mission)
        self.assertEqual([pool.get(i)[1] for i in range(3)], architectures)
        self.assertTrue(all(pool.get(i)[2] is mission for i in range(3)))

    def test_injected_mission_is_reused_for_new_evaluation_architectures(self):
        mission = object()
        architectures = [("eval-arch-0",), ("eval-arch-1",)]

        with patch.object(
            dqn.syn,
            "build_mission_from_config",
        ) as build_mission, patch.object(
            dqn.ScenarioPool,
            "sample_arch",
            side_effect=architectures,
        ):
            pool = dqn.ScenarioPool(size=2, mission=mission)

        build_mission.assert_not_called()
        self.assertIs(pool.mission, mission)
        self.assertTrue(all(pool.get(i)[2] is mission for i in range(2)))

    def test_architecture_requires_each_function_type_to_cover_threshold(self):
        pool = dqn.ScenarioPool(size=0)
        mission = [
            type(
                "TaskStub",
                (),
                {
                    "operations": [
                        type("OperationStub", (), {"func_type": func_type, "duration": 10})()
                        for func_type in dqn.syn.func_type2idx.values()
                    ]
                },
            )()
        ]

        def system(func_name, available_until):
            return type(
                "SystemStub",
                (),
                {
                    "func_type": dqn.syn.func_type2idx[func_name],
                    "available_from": 0,
                    "available_until": available_until,
                },
            )()

        threshold = pool.min_coverage_until
        valid_arch = [
            system("S", threshold),
            system("D", threshold + 50),
            system("I", threshold + 100),
        ]
        short_arch = [
            system("S", threshold),
            system("D", threshold - 1),
            system("I", threshold + 100),
        ]

        self.assertTrue(pool.arch_can_cover_mission(valid_arch, mission))
        self.assertFalse(pool.arch_can_cover_mission(short_arch, mission))


if __name__ == "__main__":
    unittest.main()
