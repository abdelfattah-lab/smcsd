from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.accuracy_test_gsm8k import (
    aggregate_smc_diagnostics,
    smc_particle_metrics,
)


class TestAccuracyGSM8KDiagnostics(unittest.TestCase):
    def test_particle_metrics_persist_ess_logz_and_selection(self):
        output = {
            "text": "work\n#### 2",
            "smc_particle_texts": ["work\n#### 1", "work\n#### 2"],
            "smc_log_w_tilde": [math.log(0.25), math.log(0.75)],
            "smc_log_Z_hat": -0.4,
        }

        metrics = smc_particle_metrics(output, "2")

        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["particle_ess"], 1.6)
        self.assertAlmostEqual(metrics["correct_weight_mass"], 0.75)
        self.assertEqual(metrics["smc_log_Z_hat"], -0.4)
        self.assertTrue(metrics["selected_matches_max_weight"])
        self.assertEqual(metrics["nonfinite_weight_count"], 0)

    def test_nonfinite_weights_are_reported_without_dropping_row(self):
        output = {
            "text": "#### 1",
            "smc_particle_texts": ["#### 1", "#### 2"],
            "smc_log_w_tilde": [float("inf"), float("nan")],
            "smc_log_Z_hat": float("nan"),
        }

        metrics = smc_particle_metrics(output, "1")

        self.assertIsNotNone(metrics)
        self.assertEqual(metrics["particle_normalized_weights"], [1.0, 0.0])
        self.assertEqual(metrics["nonfinite_weight_count"], 2)
        self.assertEqual(
            metrics["weight_normalization_fallback"],
            "equal_positive_infinity",
        )
        self.assertIsNone(metrics["smc_log_Z_hat"])

    def test_aggregate_reports_terminal_health(self):
        rows = [
            {
                "particle_ess": 2.0,
                "particle_preds": ["1", "2"],
                "max_normalized_weight": 0.6,
                "correct_weight_mass": 0.4,
                "any_particle_correct": True,
                "max_weight_correct": False,
                "selected_matches_max_weight": True,
                "nonfinite_weight_count": 0,
                "weight_normalization_fallback": None,
                "smc_log_Z_hat": -0.2,
            },
            {
                "particle_ess": 1.0,
                "particle_preds": ["1", "2"],
                "max_normalized_weight": 1.0,
                "correct_weight_mass": 1.0,
                "any_particle_correct": True,
                "max_weight_correct": True,
                "selected_matches_max_weight": False,
                "nonfinite_weight_count": 1,
                "weight_normalization_fallback": "equal_positive_infinity",
                "smc_log_Z_hat": None,
            },
        ]

        summary = aggregate_smc_diagnostics(rows)

        self.assertAlmostEqual(summary["mean_terminal_ess"], 1.5)
        self.assertAlmostEqual(summary["mean_terminal_ess_fraction"], 0.75)
        self.assertEqual(summary["nonfinite_weight_count"], 1)
        self.assertEqual(summary["weight_normalization_fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
