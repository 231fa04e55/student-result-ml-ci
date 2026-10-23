import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
DATASET_PATH = BASE_DIR / "student_results.csv"


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        for path in (MODEL_PATH, METRICS_PATH, DATASET_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path.name}. Run train_model.py first."
                )

        cls.model = joblib.load(MODEL_PATH)
        cls.dataset = pd.read_csv(DATASET_PATH)

        with open(METRICS_PATH, encoding="utf-8") as file:
            cls.metrics = json.load(file)

        # Use the exact feature names and order used during training.
        cls.features = list(cls.model.feature_names_in_)

    def make_sample(self, values):
        return pd.DataFrame([values], columns=self.features)

    def test_dataset_created(self):
        self.assertGreater(len(self.dataset), 0)
        self.assertTrue(
            set(self.features).issubset(self.dataset.columns)
        )
        self.assertIn("result", self.dataset.columns)

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

    def test_model_prediction(self):
        sample = self.make_sample(
            [8, 95, 90, 90, 90][:len(self.features)]
        )
        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])

    def test_high_performance_student(self):
        # Use high values for all input features.
        sample = self.make_sample(
            [100] * len(self.features)
        )
        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])

    def test_low_performance_student(self):
        # Use low values for all input features.
        sample = self.make_sample(
            [0] * len(self.features)
        )
        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])


if __name__ == "__main__":
    unittest.main()
