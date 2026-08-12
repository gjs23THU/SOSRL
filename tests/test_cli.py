import argparse
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

    def test_exposes_only_the_six_documented_workflows(self):
        _, commands = self.subcommands()
        self.assertEqual(
            commands,
            {
                "train-scheduler",
                "train-architecture",
                "finetune",
                "evaluate",
                "train-flat",
                "compare-schedulers",
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

        self.assertEqual(architecture.command, "train-architecture")
        self.assertEqual(finetune.command, "finetune")
        self.assertEqual(evaluate.command, "evaluate")

    def test_auto_device_resolves_to_a_concrete_torch_device(self):
        self.assertIn(cli.resolve_device("auto"), {"cpu", "cuda"})
        self.assertEqual(cli.resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
