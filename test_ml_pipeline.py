import json
import os
import unittest
import joblib
import pandas as pd

FEATURES = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_score"
]


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("student_result_model.pkl")
        cls.dataset = pd.read_csv("student_results.csv")

        with open("metrics.json", "r") as file:
            cls.metrics = json.load(file)

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))
        self.assertGreater(len(self.dataset), 0)

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))
        self.assertIn("accuracy", self.metrics)

    def test_accuracy_is_valid(self):
        accuracy = self.metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        sample = pd.DataFrame(
            [[8, 95, 90, 90, 90]],
            columns=FEATURES
        )
        prediction = self.model.predict(sample)[0]
        self.assertIn(prediction, [0, 1])

    def test_high_performance_student(self):
        sample = pd.DataFrame(
            [[10, 100, 100, 100, 100]],
            columns=FEATURES
        )
        prediction = self.model.predict(sample)[0]
        self.assertEqual(prediction, 1)

    def test_low_performance_student(self):
        sample = pd.DataFrame(
            [[0, 20, 10, 10, 10]],
            columns=FEATURES
        )
        prediction = self.model.predict(sample)[0]
        self.assertEqual(prediction, 0)


if __name__ == "__main__":
    unittest.main()
