import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sosrl.gp.artifact import (
    create_policy_artifact,
    load_gp_policy,
    save_gp_policy,
    simplify_expression,
)
from sosrl.gp.features import feature_names_for_preset
from sosrl.gp.primitives import (
    build_primitive_set,
    individual_from_expression,
    mul,
    pdiv,
    score_feature_matrix,
    tree_within_limits,
)


class GPPrimitiveAndArtifactTest(unittest.TestCase):
    def test_protected_math(self):
        self.assertEqual(pdiv(3.0, 0.0), 3.0)
        self.assertEqual(pdiv(3.0, 1e-8), 3.0)
        self.assertEqual(mul(5.0, 5.0), 10.0)
        self.assertEqual(mul(-5.0, 5.0), -10.0)

    def test_fixed_expression_scores_known_features(self):
        names = feature_names_for_preset("system")
        pset = build_primitive_set(names)
        individual = individual_from_expression(
            "add(progress, mul(added_cost_ratio, 0.5))", pset
        )
        matrix = np.zeros((2, len(names)), dtype=np.float64)
        matrix[0, names.index("progress")] = 0.2
        matrix[0, names.index("added_cost_ratio")] = 0.6
        matrix[1, names.index("progress")] = 0.7
        matrix[1, names.index("added_cost_ratio")] = 0.2

        scores = score_feature_matrix(individual, pset, matrix)

        np.testing.assert_allclose(scores, [0.5, 0.8])

    def test_json_round_trip_preserves_expression_scores_and_metadata(self):
        names = feature_names_for_preset("system_delta")
        pset = build_primitive_set(names)
        individual = individual_from_expression(
            "sub(delta_net_cost_ratio, delta_feasible_pair_ratio)", pset
        )
        artifact = create_policy_artifact(
            individual,
            feature_preset="system_delta",
            validation_fitness={"failure_rate": 0.0, "raw_mean_j": 1.25},
            evolution_config={"population_size": 8},
            bdqn_checkpoint_sha256="abc",
        )
        vector = np.linspace(-1.0, 1.0, len(names))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_gp_policy(Path(temp_dir) / "gp_policy.json", artifact)
            loaded = load_gp_policy(path)

        self.assertEqual(loaded.artifact.expression, str(individual))
        self.assertEqual(loaded.artifact.node_count, len(individual))
        self.assertEqual(loaded.artifact.height, individual.height)
        self.assertAlmostEqual(
            float(loaded.score_function(*vector.tolist())),
            float(
                vector[names.index("delta_net_cost_ratio")]
                - vector[names.index("delta_feasible_pair_ratio")]
            ),
        )

    def test_json_loader_rejects_unknown_or_executable_nodes(self):
        names = feature_names_for_preset("system")
        pset = build_primitive_set(names)
        individual = individual_from_expression("progress", pset)
        artifact = create_policy_artifact(
            individual,
            feature_preset="system",
        ).to_dict()
        artifact["expression"] = "__import__('os')"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_gp_policy(path)

    def test_json_loader_rejects_version_one_feature_schema(self):
        names = feature_names_for_preset("system")
        pset = build_primitive_set(names)
        individual = individual_from_expression("progress", pset)
        artifact = create_policy_artifact(
            individual,
            feature_preset="system",
        ).to_dict()
        artifact["feature_schema_version"] = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "old-feature-schema.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema"):
                load_gp_policy(path)

    def test_tree_limits_check_height_and_node_count(self):
        names = feature_names_for_preset("system")
        pset = build_primitive_set(names)
        small = individual_from_expression("add(progress, active_ratio)", pset)
        deep = individual_from_expression(
            "negative(negative(negative(negative(negative(negative(negative(progress)))))))",
            pset,
        )

        self.assertTrue(tree_within_limits(small, max_height=6, max_nodes=40))
        self.assertFalse(tree_within_limits(deep, max_height=6, max_nodes=40))

    def test_simplifier_only_folds_constant_primitive_subtrees(self):
        names = feature_names_for_preset("system")
        simplified = simplify_expression(
            "add(progress, mul(0.5, 0.25))", names
        )
        pset = build_primitive_set(names)
        original = individual_from_expression(
            "add(progress, mul(0.5, 0.25))", pset
        )
        folded = individual_from_expression(simplified, pset)
        vector = np.linspace(-1.0, 1.0, len(names))

        self.assertLess(len(folded), len(original))
        np.testing.assert_allclose(
            score_feature_matrix(original, pset, vector[None, :]),
            score_feature_matrix(folded, pset, vector[None, :]),
        )


if __name__ == "__main__":
    unittest.main()
