import argparse
from pathlib import Path
import unittest

from sosrl import cli


class UnifiedCliTests(unittest.TestCase):
    def subcommands(self):
        parser = cli.build_parser()
        action = next(
            item
            for item in parser._actions
            if isinstance(item, argparse._SubParsersAction)
        )
        return parser, set(action.choices)

    def test_exposes_the_existing_workflows_and_flat_rule_training(self):
        _, commands = self.subcommands()
        self.assertEqual(
            commands,
            {
                "train-scheduler",
                "train-architecture",
                "finetune",
                "evaluate",
                "train-flat",
                "train-flat-rules",
                "train-branching-scheduler",
                "compare-schedulers",
                "generate-gp-scenarios",
                "train-gp-architecture",
                "evaluate-gp-stack",
            },
        )

    def test_checkpoint_commands_parse_required_paths(self):
        parser, _ = self.subcommands()
        architecture = parser.parse_args(
            [
                "train-architecture",
                "--scheduler-checkpoint",
                "scheduler.pt",
            ]
        )
        finetune = parser.parse_args(
            [
                "finetune",
                "--scheduler-checkpoint",
                "scheduler.pt",
                "--architecture-checkpoint",
                "architecture.pt",
            ]
        )
        evaluate = parser.parse_args(
            ["evaluate", "--checkpoint", "hrl.pt"]
        )
        branching = parser.parse_args(
            [
                "train-branching-scheduler",
                "--architecture-checkpoint",
                "architecture.pt",
            ]
        )

        self.assertEqual(architecture.command, "train-architecture")
        self.assertEqual(finetune.command, "finetune")
        self.assertEqual(evaluate.command, "evaluate")
        self.assertEqual(branching.command, "train-branching-scheduler")
        self.assertEqual(branching.max_env_steps, 240000)

    def test_gp_commands_parse_standard_and_smoke_parameters(self):
        parser, _ = self.subcommands()
        generate = parser.parse_args(
            ["generate-gp-scenarios", "--train-size", "4"]
        )
        train = parser.parse_args(
            [
                "train-gp-architecture",
                "--scheduler-checkpoint",
                "branching_scheduler.pt",
                "--scenario-dir",
                "gp_scenarios",
                "--population-size",
                "8",
                "--generations",
                "2",
                "--runs",
                "1",
                "--workers",
                "1",
            ]
        )
        evaluate = parser.parse_args(
            [
                "evaluate-gp-stack",
                "--gp-policy",
                "gp_policy.json",
                "--scheduler-checkpoint",
                "branching_scheduler.pt",
                "--scenario-manifest",
                "test_iid.json",
                "--baselines",
                "fixed",
                "gp",
            ]
        )

        self.assertEqual(generate.train_size, 4)
        self.assertEqual(train.population_size, 8)
        self.assertEqual(train.generations, 2)
        self.assertEqual(train.feature_set, "system_delta")
        self.assertEqual(evaluate.baselines, ["fixed", "gp"])

    def test_flat_rule_commands_parse_training_budget_and_models(self):
        parser, _ = self.subcommands()
        training = parser.parse_args(
            ["train-flat-rules", "--max-env-steps", "240000"]
        )
        evaluation = parser.parse_args(
            [
                "evaluate",
                "--checkpoint",
                "hrl.pt",
                "--flat-rule-model",
                "flat128=flat_rules.pt",
            ]
        )

        self.assertEqual(training.hidden_dim, 128)
        self.assertEqual(training.max_env_steps, 240000)
        self.assertEqual(
            evaluation.flat_rule_models,
            [("flat128", Path("flat_rules.pt"))],
        )

    def test_auto_device_resolves_to_a_concrete_torch_device(self):
        self.assertIn(cli.resolve_device("auto"), {"cpu", "cuda"})
        self.assertEqual(cli.resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
