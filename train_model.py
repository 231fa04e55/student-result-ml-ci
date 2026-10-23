import json
import random
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

FEATURES = [
    "study_hours",
    "attendance",
    "assignment_marks",
    "internal_marks",
    "previous_score"
]

# Generate a reproducible demonstration dataset
rng = random.Random(42)
rows = []

for _ in range(300):
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

    result = int(score >= 50)

    rows.append({
        "study_hours": study_hours,
        "attendance": attendance,
        "assignment_marks": assignment_marks,
        "internal_marks": internal_marks,
        "previous_score": previous_score,
        "result": result
    })

df = pd.DataFrame(rows)

# Save demonstration dataset
df.to_csv("student_results.csv", index=False)

X = df[FEATURES]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model
joblib.dump(model, "student_result_model.pkl")

# Save evaluation metrics
metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Dataset generated successfully")
print(f"Model accuracy: {accuracy:.4f}")
print("Model saved successfully")
print("metrics.json generated successfully")
