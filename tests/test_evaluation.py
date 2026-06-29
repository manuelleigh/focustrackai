from __future__ import annotations

import unittest

import pandas as pd

from focustrack.engine.evaluation import build_labeled_dataset, evaluate_label_predictions


class EvaluationTests(unittest.TestCase):
    def test_build_labeled_dataset_matches_snapshots_with_human_ranges(self) -> None:
        history = pd.DataFrame(
            [
                {
                    "timestamp": "2026-06-01T10:00:00",
                    "session_id": "session-demo",
                    "productivity_label": "Productivo",
                    "productivity_score": 90.0,
                },
                {
                    "timestamp": "2026-06-01T10:02:00",
                    "session_id": "session-demo",
                    "productivity_label": "Regular",
                    "productivity_score": 60.0,
                },
            ]
        )
        human_labels = pd.DataFrame(
            [
                {
                    "session_id": "session-demo",
                    "start_time": "2026-06-01T09:59:00",
                    "end_time": "2026-06-01T10:01:00",
                    "label": "Productivo",
                    "notes": "Consistente",
                }
            ]
        )

        dataset = build_labeled_dataset(history=history, human_labels=human_labels)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.iloc[0]["predicted_label"], "Productivo")
        self.assertEqual(dataset.iloc[0]["human_label"], "Productivo")

    def test_evaluate_label_predictions_returns_accuracy_and_confusion(self) -> None:
        dataset = pd.DataFrame(
            [
                {"predicted_label": "Productivo", "human_label": "Productivo"},
                {"predicted_label": "Regular", "human_label": "Productivo"},
                {"predicted_label": "Regular", "human_label": "Regular"},
            ]
        )

        report = evaluate_label_predictions(dataset)

        self.assertEqual(report["total_samples"], 3)
        self.assertAlmostEqual(float(report["accuracy"]), 2 / 3, places=4)
        self.assertIn("Productivo", report["confusion_matrix"])
        self.assertEqual(report["confusion_matrix"]["Productivo"]["Regular"], 1)
        self.assertIn("Regular", report["per_label"])


if __name__ == "__main__":
    unittest.main()
