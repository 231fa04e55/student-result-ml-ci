
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_score",
]


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        model_path = BASE_DIR / "student_result_model.pkl"
        dataset_path = BASE_DIR / "student_results.csv"
        metrics_path = BASE_DIR / "metrics.json"

        if not all(p.exists() for p in (
            model_path, dataset_path, metrics_path
        )):
            raise FileNotFoundError(
                "Model artifacts are missing. Run train_model.py first."
            )

        cls.model = joblib.load(model_path)
        cls.dataset = pd.read_csv(dataset_path)

        with open(metrics_path, encoding="utf-8") as file:
            cls.metrics = json.load(file)

    def test_dataset_created(self):
        self.assertGreater(len(self.dataset), 0)
        self.assertTrue(set(FEATURES + ["result"]).issubset(
            self.dataset.columns
        ))

    def test_model_created(self):
        self.assertTrue(hasattr(self.model, "predict"))

    def test_metrics_created(self):
        self.assertIn("accuracy", self.metrics)
        self.assertIn("training_records", self.metrics)
        self.assertIn("testing_records", self.metrics)

    def test_accuracy_is_valid(self):
        accuracy = self.metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_features_match(self):
        self.assertEqual(
            list(self.model.feature_names_in_),
            FEATURES,
        )

    def test_model_prediction(self):
        sample = pd.DataFrame(
            [[8, 95, 90, 90, 90]],
            columns=FEATURES,
        )
        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])

    def test_high_performance_student(self):
        sample = pd.DataFrame(
            [[10, 100, 100, 100, 100]],
            columns=FEATURES,
        )
        self.assertEqual(self.model.predict(sample)[0], 1)

    def test_low_performance_student(self):
        sample = pd.DataFrame(
            [[0, 20, 10, 10, 10]],
            columns=FEATURES,
        )
        self.assertEqual(self.model.predict(sample)[0], 0)


if __name__ == "__main__":
    unittest.main()
