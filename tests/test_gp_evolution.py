import tempfile
from pathlib import Path
import unittest

from sosrl.gp.config import GPArchitectureConfig
from sosrl.gp.evolution import (
    EpisodeOutcome,
    aggregate_fitness,
    build_toolbox,
    episode_objective,
    evolve_architecture_policy,
    initial_gp_convergence_state,
    mixed_parent_population,
    update_gp_convergence,
)
from sosrl.gp.features import feature_names_for_preset


def outcome(*, success=True, completed=4, peak=8000.0, changes=0):
    return EpisodeOutcome(
        success=success,
        completed_operations=completed,
        total_operations=4,
        makespan=100.0,
        scale=200.0,
        final_net_cost=4000.0,
        peak_net_cost=peak,
        budget=8000.0,
        architecture_changes=changes,
        dead_end=not success,
    )


class GPEvolutionTest(unittest.TestCase):
    def test_parent_population_has_exact_parent_mutants_and_random_diversity(self):
        config = GPArchitectureConfig(
            population_size=10,
            generations=2,
            independent_runs=1,
            elite_count=1,
            anchor_interval=1,
            parent_population_fraction=0.30,
        )
        names = feature_names_for_preset("system")
        toolbox, pset = build_toolbox(names, config)
        parent = f"add({names[0]}, {names[1]})"

        population, sources = mixed_parent_population(
            toolbox=toolbox,
            pset=pset,
            config=config,
            parent_expression=parent,
        )

        self.assertEqual(len(population), 10)
        self.assertEqual(sources, {"parent": 1, "parent_mutants": 2, "random": 7})
        self.assertEqual(sum(str(item) == parent for item in population), 1)

    def test_online_convergence_requires_two_windows_and_confirmation(self):
        config = GPArchitectureConfig(
            population_size=8,
            generations=50,
            independent_runs=1,
            elite_count=1,
            anchor_interval=5,
            convergence_interval=5,
            min_generations=20,
        )
        state = initial_gp_convergence_state()
        stopped = []
        for generation, raw_j in ((5, 10.0), (10, 9.95), (15, 9.94), (20, 9.93)):
            state, confirmed = update_gp_convergence(
                state,
                {"failure_rate": 0.0, "raw_mean_j": raw_j},
                completed_generations=generation,
                config=config,
            )
            stopped.append(confirmed)

        self.assertEqual(stopped, [False, False, False, True])
        self.assertEqual(state["provisional_generation"], 15)
        self.assertEqual(state["confirmed_generation"], 20)

    def test_online_convergence_resets_after_meaningful_improvement(self):
        config = GPArchitectureConfig(
            population_size=8,
            generations=50,
            independent_runs=1,
            elite_count=1,
            anchor_interval=5,
            convergence_interval=5,
            min_generations=20,
        )
        state = initial_gp_convergence_state()
        for generation, raw_j in ((5, 10.0), (10, 9.95), (15, 9.0)):
            state, confirmed = update_gp_convergence(
                state,
                {"failure_rate": 0.0, "raw_mean_j": raw_j},
                completed_generations=generation,
                config=config,
            )

        self.assertFalse(confirmed)
        self.assertEqual(state["stable_windows"], 0)
        self.assertIsNone(state["provisional_generation"])

    def test_episode_fitness_formula(self):
        self.assertAlmostEqual(episode_objective(outcome()), 5.5)
        self.assertAlmostEqual(
            episode_objective(outcome(peak=12000.0, changes=2)),
            5.5 + 20.0 * 0.5**2 + 0.02,
        )
        self.assertAlmostEqual(
            episode_objective(outcome(success=False, completed=2)),
            5.5 + 5.0,
        )

    def test_failure_rate_is_first_fitness_objective(self):
        successful = aggregate_fitness([outcome()], node_count=40)
        failed = aggregate_fitness(
            [
                EpisodeOutcome(
                    success=False,
                    completed_operations=3,
                    total_operations=4,
                    makespan=0.0,
                    scale=200.0,
                    final_net_cost=0.0,
                    peak_net_cost=0.0,
                    budget=8000.0,
                    architecture_changes=0,
                    dead_end=True,
                )
            ],
            node_count=1,
        )
        self.assertEqual(successful.failure_rate, 0.0)
        self.assertEqual(failed.failure_rate, 1.0)
        self.assertLess(successful.values, failed.values)

    def test_small_seeded_evolution_is_reproducible_and_bounded(self):
        config = GPArchitectureConfig(
            population_size=8,
            generations=2,
            independent_runs=1,
            elite_count=1,
            train_batch_size=4,
            anchor_size=4,
            anchor_interval=1,
            anchor_top_k=2,
            max_height=6,
            max_nodes=40,
        )
        names = feature_names_for_preset("system")
        scenarios = [0, 1, 2, 3]

        def batch_sampler(run_seed, generation):
            return scenarios

        def evaluator(individual, selected):
            value = (sum(ord(char) for char in str(individual)) % 17) / 100.0
            return [
                EpisodeOutcome(
                    success=True,
                    completed_operations=4,
                    total_operations=4,
                    makespan=100.0 + value + item,
                    scale=200.0,
                    final_net_cost=4000.0,
                    peak_net_cost=4000.0,
                    budget=8000.0,
                    architecture_changes=0,
                )
                for item in selected
            ]

        with tempfile.TemporaryDirectory() as temp_dir:
            first = evolve_architecture_policy(
                feature_names=names,
                config=config,
                run_seed=123,
                batch_sampler=batch_sampler,
                anchor_scenarios=scenarios,
                individual_evaluator=evaluator,
                checkpoint_path=Path(temp_dir) / "first.pkl",
            )
            second = evolve_architecture_policy(
                feature_names=names,
                config=config,
                run_seed=123,
                batch_sampler=batch_sampler,
                anchor_scenarios=scenarios,
                individual_evaluator=evaluator,
                checkpoint_path=Path(temp_dir) / "second.pkl",
            )

        self.assertEqual(first.generation_history, second.generation_history)
        self.assertEqual(first.anchor_history, second.anchor_history)
        self.assertEqual(first.stop_reason, "max_generations_reached")
        self.assertEqual(first.actual_generations, 2)
        self.assertTrue(all(item.height <= 6 for item in first.population))
        self.assertTrue(all(len(item) <= 40 for item in first.population))

    def test_atomic_checkpoint_resume_matches_uninterrupted_run(self):
        config = GPArchitectureConfig(
            population_size=8,
            generations=2,
            independent_runs=1,
            elite_count=1,
            train_batch_size=4,
            anchor_size=4,
            anchor_interval=2,
            anchor_top_k=2,
        )
        names = feature_names_for_preset("system")
        scenarios = [0, 1, 2, 3]

        def evaluator(individual, selected):
            return [outcome() for _ in selected]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            def interrupted_sampler(run_seed, generation):
                if generation == 1:
                    raise RuntimeError("simulated interruption")
                return scenarios

            checkpoint = root / "resume.pkl"
            with self.assertRaises(RuntimeError):
                evolve_architecture_policy(
                    feature_names=names,
                    config=config,
                    run_seed=91,
                    batch_sampler=interrupted_sampler,
                    anchor_scenarios=scenarios,
                    individual_evaluator=evaluator,
                    checkpoint_path=checkpoint,
                )
            with self.assertRaisesRegex(ValueError, "feature registry"):
                evolve_architecture_policy(
                    feature_names=names[:-1],
                    config=config,
                    run_seed=91,
                    batch_sampler=lambda seed, generation: scenarios,
                    anchor_scenarios=scenarios,
                    individual_evaluator=evaluator,
                    checkpoint_path=checkpoint,
                    resume_state=checkpoint,
                )
            resumed = evolve_architecture_policy(
                feature_names=names,
                config=config,
                run_seed=91,
                batch_sampler=lambda seed, generation: scenarios,
                anchor_scenarios=scenarios,
                individual_evaluator=evaluator,
                checkpoint_path=checkpoint,
                resume_state=checkpoint,
            )
            full = evolve_architecture_policy(
                feature_names=names,
                config=config,
                run_seed=91,
                batch_sampler=lambda seed, generation: scenarios,
                anchor_scenarios=scenarios,
                individual_evaluator=evaluator,
                checkpoint_path=root / "full.pkl",
            )

        self.assertEqual(resumed.generation_history, full.generation_history)
        self.assertEqual(resumed.anchor_history, full.anchor_history)


if __name__ == "__main__":
    unittest.main()
