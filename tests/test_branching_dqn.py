import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from sosrl import domain as syn
from sosrl import environment as env
from sosrl import cli
from sosrl.rl.branching import (
    BranchingAction,
    BranchingDQNAgent,
    BranchingObservation,
    BranchingQNetwork,
    BranchingQOutput,
    build_branching_observation,
    collate_branching_observations,
    masked_argmax,
)
from sosrl.rl.config import BranchingDQNConfig
from sosrl.rl.agent import ArchitectureDQNAgent, DQNAgent
from sosrl.rl.config import DQNConfig, HRLConfig
from sosrl.rl.checkpoint import (
    load_combined_checkpoint,
    save_combined_checkpoint,
)
from sosrl.workflows.branching import (
    run_branching_episode,
    train_branching_scheduler,
)


class BranchingObservationTests(unittest.TestCase):
    def make_task(self, index, operation_types, durations):
        operations = [
            syn.Operation(
                index=op_idx,
                name=f"op-{index}-{op_idx}",
                func_type=func_type,
                duration=durations[op_idx],
                release_time=0,
            )
            for op_idx, func_type in enumerate(operation_types)
        ]
        return syn.Task(
            index=index,
            name=f"task-{index}",
            operations=operations,
            due_time=500,
        )

    def test_builder_extracts_frontier_pair_mask_and_finite_features(self):
        s_type = syn.func_type2idx["S"]
        d_type = syn.func_type2idx["D"]
        mission = [
            self.make_task(0, [s_type, d_type], [10, 12]),
            self.make_task(1, [d_type, s_type], [11, 13]),
        ]
        mission_env = env.MissionEnv(
            [env.FULL_SOS[0], env.FULL_SOS[1]],
            mission,
            adaptive=True,
        )

        observation = build_branching_observation(mission_env)
        expected_pair_mask = np.zeros((mission_env.T, mission_env.N), dtype=bool)
        full_mask = mission_env.valid_assignment_mask()
        for task_idx, op_idx in enumerate(mission_env.state.task_op_idx):
            expected_pair_mask[task_idx] = full_mask[task_idx, int(op_idx)]

        self.assertEqual(observation.global_features.shape, (25,))
        self.assertEqual(observation.task_features.shape, (2, 15))
        self.assertEqual(observation.system_features.shape, (mission_env.N, 16))
        np.testing.assert_array_equal(observation.pair_mask, expected_pair_mask)
        np.testing.assert_array_equal(observation.task_op_indices, [0, 0])
        np.testing.assert_array_equal(
            observation.system_entity_mask,
            mission_env.active_system_mask,
        )
        self.assertTrue(np.isfinite(observation.task_features).all())
        self.assertTrue(np.isfinite(observation.system_features).all())

    def test_completed_task_is_masked_and_has_no_valid_pair(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            self.make_task(0, [s_type], [10]),
            self.make_task(1, [s_type], [10]),
        ]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        mission_env.step(mission_env.encode_assignment(0, 0, 0))

        observation = build_branching_observation(mission_env)

        self.assertFalse(observation.task_entity_mask[0])
        self.assertEqual(observation.task_op_indices[0], -1)
        self.assertFalse(np.any(observation.pair_mask[0]))


class BranchingBatchTests(unittest.TestCase):
    @staticmethod
    def observation(task_num, system_num):
        pair_mask = np.zeros((task_num, system_num), dtype=bool)
        pair_mask[0, 0] = True
        return BranchingObservation(
            global_features=np.zeros(25, dtype=np.float32),
            task_features=np.ones((task_num, 15), dtype=np.float32),
            system_features=np.ones((system_num, 16), dtype=np.float32),
            task_entity_mask=np.ones(task_num, dtype=bool),
            system_entity_mask=np.ones(system_num, dtype=bool),
            pair_mask=pair_mask,
            task_op_indices=np.zeros(task_num, dtype=np.int64),
            decision_version=0,
        )

    def test_collator_pads_variable_task_and_system_counts(self):
        batch = collate_branching_observations(
            [self.observation(2, 3), self.observation(4, 2)]
        )

        self.assertEqual(batch.global_features.shape, (2, 25))
        self.assertEqual(batch.task_features.shape, (2, 4, 15))
        self.assertEqual(batch.system_features.shape, (2, 3, 16))
        self.assertEqual(batch.pair_mask.shape, (2, 4, 3))
        self.assertFalse(batch.task_entity_mask[0, 3])
        self.assertFalse(batch.system_entity_mask[1, 2])
        self.assertEqual(batch.task_op_indices[0, 3], -1)

    def test_network_accepts_padded_variable_entities(self):
        batch = collate_branching_observations(
            [self.observation(2, 3), self.observation(4, 2)]
        )
        network = BranchingQNetwork()

        with torch.no_grad():
            output = network(**batch.to_torch(torch.device("cpu")))

        self.assertEqual(output.scores.shape, (2, 4, 3))
        self.assertEqual(output.value.shape, (2, 1))
        self.assertTrue(torch.isfinite(output.scores).all())


class BranchingActionSelectionTests(unittest.TestCase):
    def test_masked_argmax_uses_legal_pair_and_row_major_tie_break(self):
        scores = np.asarray(
            [
                [10.0, 5.0, 4.0],
                [5.0, 5.0, 0.0],
            ],
            dtype=np.float32,
        )
        pair_mask = np.asarray(
            [
                [False, True, False],
                [True, True, False],
            ],
            dtype=bool,
        )

        self.assertEqual(masked_argmax(scores, pair_mask), (0, 1))

    def test_uniform_exploration_source_contains_only_valid_pairs(self):
        random.seed(7)
        pair_mask = np.asarray(
            [[False, True, False], [True, False, False]],
            dtype=bool,
        )
        valid = {tuple(pair) for pair in np.argwhere(pair_mask)}

        sampled = {tuple(random.choice(np.argwhere(pair_mask))) for _ in range(100)}

        self.assertTrue(sampled)
        self.assertTrue(sampled.issubset(valid))

    def test_agent_exploration_and_greedy_selection_return_valid_pair(self):
        observation = BranchingBatchTests.observation(2, 3)
        observation.pair_mask[1, 2] = True
        agent = BranchingDQNAgent(
            BranchingDQNConfig(
                batch_size=1,
                min_buffer_size=1,
                device="cpu",
            )
        )

        random_action = agent.select_action(observation, epsilon=1.0)
        greedy_action = agent.select_action(observation, epsilon=0.0)

        self.assertTrue(observation.pair_mask[random_action.task_idx, random_action.sys_idx])
        self.assertTrue(observation.pair_mask[greedy_action.task_idx, greedy_action.sys_idx])
        self.assertEqual(random_action.op_idx, 0)
        self.assertEqual(greedy_action.decision_version, 0)

    def test_encode_action_rejects_stale_environment_version(self):
        s_type = syn.func_type2idx["S"]
        operation = syn.Operation(0, "op", s_type, 10, 0)
        mission = [syn.Task(0, "task", [operation], due_time=100)]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        observation = build_branching_observation(mission_env)
        action = BranchingAction(0, 0, 0, observation.decision_version)
        agent = BranchingDQNAgent(BranchingDQNConfig(device="cpu"))
        mission_env.add_system(2)

        with self.assertRaisesRegex(RuntimeError, "stale branching action"):
            agent.encode_environment_action(mission_env, action)


class BranchingLearningTests(unittest.TestCase):
    def test_double_dqn_target_uses_online_argmax_and_target_evaluation(self):
        config = BranchingDQNConfig(
            gamma=1.0,
            batch_size=1,
            min_buffer_size=1,
            target_update_interval=100,
            device="cpu",
        )
        agent = BranchingDQNAgent(config)
        observation = BranchingBatchTests.observation(2, 2)
        observation.pair_mask[:, :] = True
        action = BranchingAction(0, 0, 0, 0)
        agent.replay.add(
            observation,
            action,
            0.0,
            observation,
            False,
        )

        current_scores = torch.zeros((1, 2, 2), requires_grad=True)
        online_next_scores = torch.tensor(
            [[[5.0, 1.0], [0.0, 4.0]]],
            requires_grad=True,
        )
        target_next_scores = torch.tensor(
            [[[2.0, 10.0], [0.0, 3.0]]],
        )

        def output(scores):
            return BranchingQOutput(
                scores=scores,
                value=torch.zeros((1, 1)),
                task_advantages=torch.zeros((1, 2)),
                system_advantages=torch.zeros((1, 2)),
            )

        with mock.patch.object(
            agent.q_net,
            "forward",
            side_effect=[output(current_scores), output(online_next_scores)],
        ), mock.patch.object(
            agent.target_net,
            "forward",
            return_value=output(target_next_scores),
        ):
            loss = agent.learn()

        # Online selects (0, 0), target evaluates it as 2.0. Huber(0, 2)=1.5.
        self.assertAlmostEqual(loss, 1.5, places=6)

    def test_terminal_transition_does_not_bootstrap(self):
        config = BranchingDQNConfig(
            gamma=1.0,
            batch_size=1,
            min_buffer_size=1,
            device="cpu",
        )
        agent = BranchingDQNAgent(config)
        observation = BranchingBatchTests.observation(1, 1)
        terminal_observation = BranchingObservation(
            **{
                **observation.__dict__,
                "pair_mask": np.zeros((1, 1), dtype=bool),
            }
        )
        agent.replay.add(
            observation,
            BranchingAction(0, 0, 0, 0),
            0.25,
            terminal_observation,
            True,
        )

        loss = agent.learn()

        self.assertIsNotNone(loss)
        self.assertTrue(np.isfinite(loss))


class BranchingWorkflowTests(unittest.TestCase):
    @staticmethod
    def make_environment():
        s_type = syn.func_type2idx["S"]
        mission = [
            syn.Task(
                index,
                f"task-{index}",
                [syn.Operation(0, f"op-{index}", s_type, 10, 0)],
                due_time=100,
            )
            for index in range(2)
        ]
        return env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)

    def test_pending_next_observation_matches_next_lower_decision(self):
        mission_env = self.make_environment()
        architecture_agent = ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            HRLConfig(device="cpu"),
        )
        for parameter in architecture_agent.q_net.parameters():
            parameter.data.zero_()
        architecture_agent.target_net.load_state_dict(
            architecture_agent.q_net.state_dict()
        )
        branching_agent = BranchingDQNAgent(
            BranchingDQNConfig(
                batch_size=8,
                min_buffer_size=8,
                device="cpu",
            )
        )
        architecture_before = {
            name: value.detach().clone()
            for name, value in architecture_agent.q_net.state_dict().items()
        }
        architecture_target_before = {
            name: value.detach().clone()
            for name, value in architecture_agent.target_net.state_dict().items()
        }

        with mock.patch.object(
            branching_agent,
            "learn",
            wraps=branching_agent.learn,
        ) as learn:
            result = run_branching_episode(
                mission_env,
                architecture_agent,
                branching_agent,
                scheduler_epsilon=0.0,
                update_scheduler=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["invalid_action_count"], 0)
        self.assertLessEqual(learn.call_count, result["assignment_steps"])
        self.assertEqual(len(branching_agent.replay), 2)
        first, second = list(branching_agent.replay.buffer)
        self.assertEqual(
            first.next_observation.decision_version,
            second.observation.decision_version,
        )
        np.testing.assert_allclose(
            first.next_observation.global_features,
            second.observation.global_features,
        )
        self.assertFalse(
            mission_env.active_system_mask[~np.asarray([True] + [False] * 21)].any()
        )
        for transition in branching_agent.replay.buffer:
            self.assertTrue(mission_env.active_system_mask[transition.action.sys_idx])
        for name, value in architecture_agent.q_net.state_dict().items():
            torch.testing.assert_close(value, architecture_before[name])
        for name, value in architecture_agent.target_net.state_dict().items():
            torch.testing.assert_close(value, architecture_target_before[name])
        self.assertFalse(architecture_agent.q_net.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in architecture_agent.q_net.parameters()
            )
        )

    def test_episode_obeys_operation_frontier_order(self):
        s_type = syn.func_type2idx["S"]
        mission = [
            syn.Task(
                0,
                "two-operation-task",
                [
                    syn.Operation(0, "first", s_type, 10, 0),
                    syn.Operation(1, "second", s_type, 10, 0),
                ],
                due_time=100,
            )
        ]
        mission_env = env.MissionEnv([env.FULL_SOS[0]], mission, adaptive=True)
        architecture_agent = ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            HRLConfig(device="cpu"),
        )
        for parameter in architecture_agent.q_net.parameters():
            parameter.data.zero_()
        branching_agent = BranchingDQNAgent(
            BranchingDQNConfig(
                batch_size=8,
                min_buffer_size=8,
                device="cpu",
            )
        )

        result = run_branching_episode(
            mission_env,
            architecture_agent,
            branching_agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            [transition.action.op_idx for transition in branching_agent.replay.buffer],
            [0, 1],
        )
        self.assertLessEqual(
            mission_env.state.op_finish_time[0, 0],
            mission_env.state.op_start_time[0, 1],
        )

    def test_pending_state_is_captured_after_next_architecture_change(self):
        mission_env = self.make_environment()

        class AddCapacityOnSecondDecision:
            def __init__(self):
                self.calls = 0

            def select_action(self, obs, action_mask, epsilon=0.0):
                self.calls += 1
                if self.calls == 1:
                    return 0
                if action_mask[2] <= 0:
                    raise AssertionError("ADD_CAPACITY must be legal in this fixture")
                return 2

        branching_agent = BranchingDQNAgent(
            BranchingDQNConfig(
                batch_size=8,
                min_buffer_size=8,
                device="cpu",
            )
        )

        result = run_branching_episode(
            mission_env,
            AddCapacityOnSecondDecision(),
            branching_agent,
            scheduler_epsilon=0.0,
            update_scheduler=False,
        )

        self.assertTrue(result["success"])
        first, second = list(branching_agent.replay.buffer)
        self.assertEqual(first.observation.decision_version, 0)
        self.assertEqual(first.next_observation.decision_version, 2)
        self.assertEqual(
            first.next_observation.decision_version,
            second.observation.decision_version,
        )
        self.assertEqual(np.count_nonzero(second.observation.system_entity_mask), 2)

    def test_fixed_seed_training_smoke_is_reproducible_and_finite(self):
        mission_env = self.make_environment()
        scenario = (
            (env.FULL_SOS[0],),
            mission_env.mission,
            "reproducible",
        )
        pool = type(
            "Pool",
            (),
            {"sample": lambda self: scenario},
        )()
        config = BranchingDQNConfig(
            episodes=1,
            max_env_steps=2,
            batch_size=1,
            min_buffer_size=1,
            epsilon_start=0.0,
            epsilon_end=0.0,
            device="cpu",
            seed=19,
        )

        def train_once():
            architecture_agent = ArchitectureDQNAgent(
                mission_env.architecture_observation_space.shape[0],
                HRLConfig(device="cpu"),
            )
            for parameter in architecture_agent.q_net.parameters():
                parameter.data.zero_()
            agent, history, state = train_branching_scheduler(
                config,
                architecture_agent,
                pool,
            )
            return agent, history, state

        first_agent, first_history, first_state = train_once()
        second_agent, second_history, second_state = train_once()

        self.assertEqual(len(first_agent.replay), 2)
        self.assertEqual(len(second_agent.replay), 2)
        self.assertTrue(np.isfinite(first_history[0]["scheduler_loss"]))
        self.assertEqual(
            first_history[0]["scheduler_loss"],
            second_history[0]["scheduler_loss"],
        )
        self.assertEqual(first_state, second_state)
        for name, value in first_agent.q_net.state_dict().items():
            torch.testing.assert_close(value, second_agent.q_net.state_dict()[name])


class BranchingCheckpointTests(unittest.TestCase):
    def test_standalone_checkpoint_restores_values_action_and_optimizer(self):
        config = BranchingDQNConfig(
            batch_size=1,
            min_buffer_size=1,
            device="cpu",
        )
        agent = BranchingDQNAgent(config)
        observation = BranchingBatchTests.observation(2, 2)
        observation.pair_mask[:, :] = True
        agent.replay.add(
            observation,
            BranchingAction(0, 0, 0, 0),
            0.5,
            observation,
            True,
        )
        self.assertIsNotNone(agent.learn())
        expected_action = agent.select_action(observation, epsilon=0.0)
        batch = collate_branching_observations([observation])
        with torch.no_grad():
            expected_scores = agent.q_net(
                **batch.to_torch(torch.device("cpu"))
            ).scores.clone()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = agent.save_checkpoint(
                Path(temp_dir) / "branching_scheduler.pt",
                training_state={"total_env_steps": 12},
            )
            loaded, checkpoint = BranchingDQNAgent.load_checkpoint(
                path,
                device="cpu",
            )

        with torch.no_grad():
            actual_scores = loaded.q_net(
                **batch.to_torch(torch.device("cpu"))
            ).scores
        torch.testing.assert_close(actual_scores, expected_scores)
        self.assertEqual(
            loaded.select_action(observation, epsilon=0.0),
            expected_action,
        )
        self.assertEqual(loaded.learn_step, agent.learn_step)
        self.assertEqual(
            len(loaded.optimizer.state_dict()["state"]),
            len(agent.optimizer.state_dict()["state"]),
        )
        expected_optimizer = agent.optimizer.state_dict()
        actual_optimizer = loaded.optimizer.state_dict()
        self.assertEqual(
            actual_optimizer["param_groups"],
            expected_optimizer["param_groups"],
        )
        for parameter_id, expected_state in expected_optimizer["state"].items():
            actual_state = actual_optimizer["state"][parameter_id]
            self.assertEqual(actual_state.keys(), expected_state.keys())
            for key, expected_value in expected_state.items():
                if torch.is_tensor(expected_value):
                    torch.testing.assert_close(actual_state[key], expected_value)
                else:
                    self.assertEqual(actual_state[key], expected_value)
        self.assertEqual(checkpoint["checkpoint_kind"], "branching_scheduler")
        self.assertEqual(checkpoint["feature_schema"]["dimensions"]["task"], 15)
        self.assertEqual(checkpoint["training_state"]["total_env_steps"], 12)

    def test_combined_checkpoint_dispatches_branching_scheduler(self):
        mission_env = BranchingWorkflowTests.make_environment()
        architecture_agent = ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            HRLConfig(device="cpu"),
        )
        branching_agent = BranchingDQNAgent(BranchingDQNConfig(device="cpu"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_combined_checkpoint(
                Path(temp_dir) / "architecture_branching.pt",
                architecture_agent,
                branching_agent,
                metadata={"architecture_provider": "frozen"},
            )
            loaded_arch, loaded_scheduler, checkpoint = load_combined_checkpoint(
                path,
                device="cpu",
            )

        self.assertIsInstance(loaded_arch, ArchitectureDQNAgent)
        self.assertIsInstance(loaded_scheduler, BranchingDQNAgent)
        self.assertEqual(checkpoint["scheduler_kind"], "branching_scheduler")
        self.assertEqual(
            checkpoint["metadata"]["architecture_provider"],
            "frozen",
        )

    def test_combined_loader_accepts_legacy_checkpoint_without_kind(self):
        mission_env = BranchingWorkflowTests.make_environment()
        architecture_agent = ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            HRLConfig(device="cpu"),
        )
        scheduler_agent = DQNAgent(
            25,
            4,
            DQNConfig(device="cpu"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_combined_checkpoint(
                Path(temp_dir) / "legacy_hrl.pt",
                architecture_agent,
                scheduler_agent,
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload.pop("checkpoint_kind")
            payload.pop("schema_version")
            payload.pop("scheduler_kind")
            payload["scheduler"].pop("checkpoint_kind")
            payload["scheduler"].pop("schema_version")
            torch.save(payload, path)
            _, loaded_scheduler, _ = load_combined_checkpoint(
                path,
                device="cpu",
            )

        self.assertIsInstance(loaded_scheduler, DQNAgent)
        self.assertEqual(loaded_scheduler.action_dim, 4)

    def test_training_cli_writes_branching_artifacts(self):
        mission_env = BranchingWorkflowTests.make_environment()
        scenario = (
            tuple(env.FULL_SOS[index] for index in np.flatnonzero(
                mission_env.active_system_mask
            )),
            mission_env.mission,
            "smoke",
        )
        pool = type(
            "Pool",
            (),
            {
                "scenarios": [scenario],
                "sample": lambda self: scenario,
            },
        )()
        architecture_agent = ArchitectureDQNAgent(
            mission_env.architecture_observation_space.shape[0],
            HRLConfig(device="cpu"),
        )
        for parameter in architecture_agent.q_net.parameters():
            parameter.data.zero_()
        architecture_agent.target_net.load_state_dict(
            architecture_agent.q_net.state_dict()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            architecture_path = architecture_agent.save_checkpoint(
                temp_path / "architecture.pt"
            )
            output_dir = temp_path / "output"
            args = cli.build_parser().parse_args(
                [
                    "train-branching-scheduler",
                    "--architecture-checkpoint",
                    str(architecture_path),
                    "--episodes",
                    "1",
                    "--scenario-pool-size",
                    "1",
                    "--max-env-steps",
                    "2",
                    "--batch-size",
                    "8",
                    "--min-buffer-size",
                    "8",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            with mock.patch.object(cli, "adaptive_pool", return_value=pool):
                args.handler(args)

            expected_files = {
                "branching_scheduler.pt",
                "architecture_branching.pt",
                "branching_history.csv",
                "branching_config.json",
            }
            self.assertTrue(
                expected_files.issubset(
                    {path.name for path in output_dir.iterdir()}
                )
            )
            _, loaded_scheduler, checkpoint = load_combined_checkpoint(
                output_dir / "architecture_branching.pt",
                device="cpu",
            )
            self.assertIsInstance(loaded_scheduler, BranchingDQNAgent)
            self.assertEqual(
                checkpoint["training_state"]["actual_environment_steps"],
                2,
            )

            evaluation_dir = temp_path / "evaluation"
            evaluate_args = cli.build_parser().parse_args(
                [
                    "evaluate",
                    "--checkpoint",
                    str(output_dir / "architecture_branching.pt"),
                    "--eval-episodes",
                    "1",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(evaluation_dir),
                ]
            )
            with mock.patch.object(cli, "adaptive_pool", return_value=pool):
                evaluate_args.handler(evaluate_args)
            with (evaluation_dir / "evaluation_manifest.json").open(
                encoding="utf-8"
            ) as file:
                manifest = json.load(file)
            self.assertEqual(manifest["scheduler_kind"], "branching_scheduler")
            with (evaluation_dir / "results.csv").open(encoding="utf-8") as file:
                results_text = file.read()
            self.assertIn("architecture_branching", results_text)


if __name__ == "__main__":
    unittest.main()
