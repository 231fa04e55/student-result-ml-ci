import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

FEATURES = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_score",
]

TARGET = "result"


def generate_dataset(n=1000, seed=42):
    rng = random.Random(seed)
    rows = []

    for _ in range(n):
        study_hours = rng.randint(0, 10)
        attendance = rng.randint(20, 100)
        assignment_marks = rng.randint(0, 100)
        internal_marks = rng.randint(0, 100)
        previous_score = rng.randint(0, 100)

        score = (
            study_hours * 2
            + attendance * 0.20
            + assignment_marks * 0.25
            + internal_marks * 0.25
            + previous_score * 0.20
        )

        rows.append({
            "study_hours": study_hours,
            "attendance": attendance,
            "assignment_marks": assignment_marks,
            "internal_marks": internal_marks,
            "previous_score": previous_score,
            "result": int(score >= 50),
        })

    return pd.DataFrame(rows)


def main():
    df = generate_dataset()

    # Save the dataset
    df.to_csv(BASE_DIR / "student_results.csv", index=False)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Predict using exactly the training feature names and order
    predictions = model.predict(X_test[FEATURES])
    accuracy = accuracy_score(y_test, predictions)

    # Save model and evaluation metrics
    joblib.dump(model, BASE_DIR / "student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "features": FEATURES,
    }

    with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Dataset generated successfully.")
    print(f"Model accuracy: {accuracy:.4f}")
    print("Model saved successfully.")
    print("metrics.json generated successfully.")


if __name__ == "__main__":
    main()
