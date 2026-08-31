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
                "train-fixed-rule-scheduler",
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
                "generate-alternation-scenarios",
                "train-round1-bdqn-cell",
                "init-round1-study",
                "run-round1-study",
                "smoke-round1-study",
                "run-gp-bdqn-alternation",
                "init-gp-bdqn-tuning",
                "run-gp-bdqn-tuning",
                "solve-nsga2",
                "solve-mopso",
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
                "--parent-population-fraction",
                "0.15",
                "--crossover-probability",
                "0.60",
                "--mutation-probability",
                "0.35",
                "--reproduction-probability",
                "0.05",
                "--parsimony-coefficient",
                "0.005",
                "--validation-candidates-per-run",
                "5",
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
        self.assertEqual(train.scheduler_backend, "branching-dqn")
        self.assertEqual(train.anchor_interval, 5)
        self.assertEqual(train.parent_population_fraction, 0.15)
        self.assertEqual(train.crossover_probability, 0.60)
        self.assertEqual(train.mutation_probability, 0.35)
        self.assertEqual(train.reproduction_probability, 0.05)
        self.assertEqual(train.parsimony_coefficient, 0.005)
        self.assertEqual(train.validation_candidates_per_run, 5)
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

        fixed_rule = parser.parse_args(
            [
                "train-fixed-rule-scheduler",
                "--train-manifest",
                "b/train.json",
                "--validation-manifest",
                "b/validation.json",
                "--output-dir",
                "rule-fixed",
                "--lr-end",
                "0.00001",
                "--lr-decay",
                "0.9975",
            ]
        )
        self.assertEqual(fixed_rule.max_env_steps, 200000)
        self.assertEqual(fixed_rule.seed, 4)
        self.assertEqual(fixed_rule.lr_end, 0.00001)
        self.assertEqual(fixed_rule.lr_decay, 0.9975)

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
                "--lr-end",
                "0.00001",
                "--lr-decay",
                "0.9975",
            ]
        )
        self.assertEqual(round1_scenarios.test_iid_size, 1000)
        self.assertEqual(round1_cell.provider, "g0")
        self.assertEqual(round1_cell.checkpoint_steps, [0, 10])
        self.assertEqual(round1_cell.lr_end, 0.00001)
        self.assertEqual(round1_cell.lr_decay, 0.9975)
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
        alternating_scenarios = parser.parse_args(
            [
                "generate-alternation-scenarios",
                "--existing-manifest",
                "round1/train.json",
                "--gate-iid-seed",
                "20261010",
                "--gate-ood-seed",
                "20261011",
                "--final-iid-seed",
                "20261012",
                "--final-ood-seed",
                "20261013",
                "--output-dir",
                "alternation-scenarios",
            ]
        )
        alternation = parser.parse_args(
            [
                "run-gp-bdqn-alternation",
                "--base-gp-policy",
                "g0.json",
                "--base-scheduler-checkpoint",
                "b0.pt",
                "--scenario-dir",
                "round1/g",
                "--gate-iid-manifest",
                "gate_iid.json",
                "--gate-ood-manifest",
                "gate_ood.json",
                "--final-iid-manifest",
                "final_iid.json",
                "--final-ood-manifest",
                "final_ood.json",
                "--output-dir",
                "alternation",
            ]
        )
        self.assertEqual(alternating_scenarios.gate_iid_size, 512)
        self.assertEqual(alternating_scenarios.gate_iid_seed, 20261010)
        self.assertEqual(alternating_scenarios.gate_ood_seed, 20261011)
        self.assertEqual(alternating_scenarios.final_iid_seed, 20261012)
        self.assertEqual(alternating_scenarios.final_ood_seed, 20261013)
        self.assertEqual(alternation.gp_population_size, 120)
        self.assertEqual(alternation.gp_max_generations, 50)
        self.assertEqual(alternation.bdqn_max_env_steps, 40000)
        self.assertEqual(alternation.bdqn_round1_seeds, (4, 5, 6))
        self.assertEqual(alternation.bdqn_round2_seeds, (7, 8, 9))

        tuning_init = parser.parse_args(
            [
                "init-gp-bdqn-tuning",
                "--b-scenario-dir",
                "round1/b",
                "--g-scenario-dir",
                "round1/g",
                "--base-rule-checkpoint",
                "rule.pt",
                "--base-gp-policy",
                "g0.json",
                "--base-bdqn-checkpoint",
                "b0_seed1.pt",
                "--base-bdqn-checkpoint",
                "b0_seed2.pt",
                "--base-bdqn-checkpoint",
                "b0_seed3.pt",
                "--existing-manifest",
                "round1/b/train.json",
                "--output-spec",
                "tuning_spec.json",
            ]
        )
        tuning_run = parser.parse_args(
            [
                "run-gp-bdqn-tuning",
                "--spec",
                "tuning_spec.json",
                "--output-dir",
                "tuning",
                "--resume",
            ]
        )
        self.assertEqual(len(tuning_init.base_bdqn_checkpoint), 3)
        self.assertEqual(tuning_init.existing_manifest, [Path("round1/b/train.json")])
        self.assertTrue(tuning_run.resume)
        round1_augment = parser.parse_args(
            [
                "init-round1-study",
                "--augment-from",
                "round1/study_manifest.json",
                "--augment-seeds",
                "1",
                "2",
                "3",
                "4",
                "5",
                "--output-dir",
                "round1-v2",
            ]
        )
        self.assertEqual(
            round1_augment.augment_from,
            Path("round1/study_manifest.json"),
        )
        self.assertEqual(round1_augment.augment_seeds, [1, 2, 3, 4, 5])
        with patch(
            "sosrl.workflows.round1_study.initialize_round1_study",
            return_value=Path("round1-v2/study_manifest.json"),
        ) as initialize:
            round1_augment.handler(round1_augment)
        self.assertEqual(
            initialize.call_args.kwargs["augment_from"],
            Path("round1/study_manifest.json"),
        )
        self.assertEqual(
            initialize.call_args.kwargs["augment_seeds"],
            [1, 2, 3, 4, 5],
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

    def test_fixed_rule_handler_only_runs_the_scheduler_workflow(self):
        parser, _ = self.subcommands()
        args = parser.parse_args(
            [
                "train-fixed-rule-scheduler",
                "--train-manifest",
                "b/train.json",
                "--validation-manifest",
                "b/validation.json",
                "--output-dir",
                "rule-fixed",
                "--max-env-steps",
                "1",
                "--checkpoint-steps",
                "0",
                "1",
                "--device",
                "cpu",
            ]
        )
        with patch(
            "sosrl.workflows.fixed_rule_scheduler.train_fixed_rule_scheduler",
            return_value={"manifest": Path("rule-fixed/run_manifest.json")},
        ) as train_fixed:
            args.handler(args)

        train_fixed.assert_called_once()
        self.assertEqual(train_fixed.call_args.kwargs["max_env_steps"], 1)
        self.assertEqual(train_fixed.call_args.kwargs["checkpoint_steps"], (0, 1))

    def test_nsga2_command_parses_profile_and_overrides(self):
        parser, _ = self.subcommands()
        args = parser.parse_args(
            [
                "solve-nsga2",
                "--scenario-manifest",
                "scenarios.json",
                "--scenario-indices",
                "0",
                "3",
                "--profile",
                "custom",
                "--population-size",
                "10",
                "--max-evaluations",
                "100",
                "--evaluation-milestones",
                "10",
                "50",
                "100",
                "--runs",
                "1",
                "--architecture-change-weight",
                "0.02",
                "--peak-budget-penalty",
                "15",
                "--workers",
                "2",
            ]
        )

        self.assertEqual(args.command, "solve-nsga2")
        self.assertEqual(args.scenario_indices, [0, 3])
        self.assertEqual(args.population_size, 10)
        self.assertEqual(args.max_evaluations, 100)
        self.assertEqual(args.evaluation_milestones, [10, 50, 100])
        self.assertEqual(args.runs, 1)
        self.assertEqual(args.architecture_change_weight, 0.02)
        self.assertEqual(args.peak_budget_penalty, 15.0)
        self.assertEqual(args.workers, 2)

    def test_mopso_command_parses_profile_and_overrides(self):
        parser, _ = self.subcommands()
        args = parser.parse_args(
            [
                "solve-mopso",
                "--scenario-manifest",
                "scenarios.json",
                "--scenario-indices",
                "0",
                "3",
                "--profile",
                "custom",
                "--swarm-size",
                "20",
                "--max-evaluations",
                "300",
                "--evaluation-milestones",
                "20",
                "100",
                "300",
                "--runs",
                "2",
                "--inertia-weight",
                "0.7",
                "--cognitive-coefficient",
                "1.4",
                "--social-coefficient",
                "1.6",
                "--max-velocity-rate",
                "0.25",
                "--archive-size",
                "80",
                "--workers",
                "2",
            ]
        )

        self.assertEqual(args.command, "solve-mopso")
        self.assertEqual(args.scenario_indices, [0, 3])
        self.assertEqual(args.swarm_size, 20)
        self.assertEqual(args.max_evaluations, 300)
        self.assertEqual(args.evaluation_milestones, [20, 100, 300])
        self.assertEqual(args.runs, 2)
        self.assertEqual(args.inertia_weight, 0.7)
        self.assertEqual(args.cognitive_coefficient, 1.4)
        self.assertEqual(args.social_coefficient, 1.6)
        self.assertEqual(args.max_velocity_rate, 0.25)
        self.assertEqual(args.archive_size, 80)
        self.assertEqual(args.workers, 2)

    def test_auto_device_resolves_to_a_concrete_torch_device(self):
        self.assertIn(cli.resolve_device("auto"), {"cpu", "cuda"})
        self.assertEqual(cli.resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
