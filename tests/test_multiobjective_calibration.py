import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sosrl.multiobjective_calibration import run_calibration, write_csv


class MultiObjectiveCalibrationTests(unittest.TestCase):
    def test_recommends_first_budget_meeting_all_and_split_thresholds(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for scenario_idx, split in enumerate(("b", "g")):
                scenario_dir = root / f"scenario_{scenario_idx:05d}_{split}"
                summary_rows = []
                fronts = {
                    50: [(2.0, 1.0)],
                    100: [(1.0, 1.0), (2.0, 0.0)],
                    500: [(1.0, 1.0), (2.0, 0.0)],
                }
                for budget, points in fronts.items():
                    summary_rows.append(
                        {
                            "scenario_idx": scenario_idx,
                            "scenario_hash": f"hash-{split}",
                            "split": split,
                            "category": f"category-{split}",
                            "evaluation_budget_per_run": budget,
                            "independent_runs": 3,
                            "total_evaluations": 3 * budget,
                            "success": True,
                            "pareto_front_size": len(points),
                            "gp_aligned_j": 2.0 if budget == 50 else 1.0,
                            "gp_aligned_makespan": points[0][0],
                            "gp_aligned_effective_cost": points[0][1],
                        }
                    )
                    write_csv(
                        scenario_dir
                        / "milestones"
                        / f"eval_{budget:06d}"
                        / "pareto_front.csv",
                        [
                            {"makespan": makespan, "effective_cost": cost}
                            for makespan, cost in points
                        ],
                    )
                write_csv(
                    scenario_dir / "milestone_summary.csv", summary_rows
                )
                (scenario_dir / "run_manifest.json").write_text(
                    json.dumps(
                        {
                            "wall_seconds": 3.0,
                            "run_wall_seconds": [1.0, 1.0, 1.0],
                        }
                    ),
                    encoding="utf-8",
                )

            output = root / "calibration"
            summary = run_calibration(
                [root], output, algorithm="test", max_budget=500
            )

            self.assertEqual(summary["recommended_minimum_evaluations"], 100)
            self.assertEqual(summary["scenario_count"], 2)
            self.assertEqual(
                summary["runtime_and_retries"]["scenario_wall_seconds"], 6.0
            )
            self.assertTrue((output / "scenario_budget_metrics.csv").is_file())
            self.assertTrue((output / "budget_curve.csv").is_file())
            self.assertTrue((output / "calibration_summary.json").is_file())
            self.assertTrue((output / "calibration_report.md").is_file())


if __name__ == "__main__":
    unittest.main()

