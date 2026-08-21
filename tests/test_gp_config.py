import unittest

from sosrl.gp.config import GPArchitectureConfig


class GPArchitectureConfigTest(unittest.TestCase):
    def test_standard_defaults(self):
        config = GPArchitectureConfig()

        self.assertEqual(config.population_size, 200)
        self.assertEqual(config.generations, 80)
        self.assertEqual(config.independent_runs, 10)
        self.assertEqual(config.max_height, 6)
        self.assertEqual(config.max_nodes, 40)
        self.assertAlmostEqual(
            config.crossover_probability
            + config.mutation_probability
            + config.reproduction_probability,
            1.0,
        )

    def test_invalid_probability_sum_is_rejected(self):
        with self.assertRaises(ValueError):
            GPArchitectureConfig(crossover_probability=0.8)

    def test_non_positive_population_is_rejected(self):
        with self.assertRaises(ValueError):
            GPArchitectureConfig(population_size=0)

    def test_initial_depth_cannot_exceed_limit(self):
        with self.assertRaises(ValueError):
            GPArchitectureConfig(init_max_depth=7, max_height=6)

    def test_unknown_feature_set_is_rejected(self):
        with self.assertRaises(ValueError):
            GPArchitectureConfig(feature_set="unknown")


if __name__ == "__main__":
    unittest.main()
