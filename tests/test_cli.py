import argparse
from pathlib import Path
import unittest
from unittest.mock import patch

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
                "finetune-branching-with-gp",
                "generate-round1-scenarios",
                "train-round1-bdqn-cell",
                "init-round1-study",
                "run-round1-study",
                "smoke-round1-study",
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
                "ss",
                "gp",
                "--ss-low-threshold",
                "0.4",
                "--ss-high-threshold",
                "0.9",
            ]
        )

        self.assertEqual(generate.train_size, 4)
        self.assertEqual(train.population_size, 8)
        self.assertEqual(train.generations, 2)
        self.assertEqual(train.feature_set, "system_delta")
        self.assertEqual(evaluate.baselines, ["fixed", "ss", "gp"])
        self.assertEqual(evaluate.ss_low_threshold, 0.4)
        self.assertEqual(evaluate.ss_high_threshold, 0.9)

        finetune = parser.parse_args(
            [
                "finetune-branching-with-gp",
                "--scheduler-checkpoint",
                "branching_scheduler.pt",
                "--gp-policy",
                "gp_policy.json",
                "--scenario-dir",
                "gp_scenarios",
                "--output-dir",
                "g0_b1",
                "--resume",
                "--skip-historical-test",
            ]
        )
        self.assertEqual(finetune.extra_env_steps, 40000)
        self.assertEqual(finetune.checkpoint_interval_steps, 10000)
        self.assertTrue(finetune.resume)
        self.assertTrue(finetune.skip_historical_test)

        round1_scenarios = parser.parse_args(
            ["generate-round1-scenarios", "--output-dir", "round1"]
        )
        round1_cell = parser.parse_args(
            [
                "train-round1-bdqn-cell",
                "--provider",
                "g0",
                "--mode",
                "finetune",
                "--source-checkpoint",
                "b0.pt",
                "--gp-policy",
                "g0.json",
                "--train-manifest",
                "train.json",
                "--validation-manifest",
                "validation.json",
                "--output-dir",
                "cell",
                "--seed",
                "4",
                "--checkpoint-steps",
                "0",
                "10",
            ]
        )
        self.assertEqual(round1_scenarios.test_iid_size, 1000)
        self.assertEqual(round1_cell.provider, "g0")
        self.assertEqual(round1_cell.checkpoint_steps, [0, 10])
        ss_cell = parser.parse_args(
            [
                "train-round1-bdqn-cell",
                "--provider",
                "ss",
                "--source-checkpoint",
                "b0.pt",
                "--train-manifest",
                "train.json",
                "--validation-manifest",
                "validation.json",
                "--output-dir",
                "cell",
                "--seed",
                "4",
            ]
        )
        self.assertEqual(ss_cell.provider, "ss")

        round1_init = parser.parse_args(
            [
                "init-round1-study",
                "--architecture-checkpoint",
                "arch.pt",
                "--gp-policy",
                "g0.json",
                "--output-dir",
                "round1",
            ]
        )
        round1_run = parser.parse_args(
            [
                "run-round1-study",
                "--study-manifest",
                "round1/study_manifest.json",
                "--stage",
                "gp-discovery",
                "--workers",
                "4",
            ]
        )
        self.assertEqual(round1_init.b_train_size, 256)
        self.assertEqual(round1_run.stage, "gp-discovery")
        self.assertEqual(round1_run.workers, 4)
        round1_augment = parser.parse_args(
            [
                "init-round1-study",
                "--augment-from",
                "round1/study_manifest.json",
                "--output-dir",
                "round1-v2",
            ]
        )
        self.assertEqual(
            round1_augment.augment_from,
            Path("round1/study_manifest.json"),
        )
        with patch(
            "sosrl.workflows.round1_study.initialize_round1_study",
            return_value=Path("round1-v2/study_manifest.json"),
        ) as initialize:
            round1_augment.handler(round1_augment)
        self.assertEqual(
            initialize.call_args.kwargs["augment_from"],
            Path("round1/study_manifest.json"),
        )

        with patch(
            "sosrl.workflows.round1_study.train_bdqn_provider_cell",
            return_value={"manifest": Path("cell/cell_manifest.json")},
        ) as train_cell:
            ss_cell.handler(ss_cell)
        self.assertEqual(train_cell.call_args.kwargs["provider_kind"], "ss")
        self.assertNotIn("augment_from", train_cell.call_args.kwargs)
        round1_smoke = parser.parse_args(
            [
                "smoke-round1-study",
                "--architecture-checkpoint",
                "arch.pt",
                "--gp-policy",
                "g0.json",
                "--output-dir",
                "smoke",
            ]
        )
        self.assertEqual(round1_smoke.seed, 20260824)

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
