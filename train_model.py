import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Generate demonstration student dataset
data = {
    "assignment_marks": [
        45, 60, 70, 80, 90, 35, 55, 65, 75, 85,
        40, 50, 60, 70, 80, 90, 30, 45, 65, 95,
        55, 75, 85, 95, 40, 60, 70, 80, 50, 90
    ],
    "internal_marks": [
        40, 55, 65, 75, 85, 30, 50, 60, 70, 80,
        35, 45, 55, 65, 75, 90, 25, 40, 60, 95,
        50, 70, 80, 90, 35, 55, 65, 75, 45, 85
    ],
    "previous_score": [
        42, 58, 68, 78, 88, 32, 52, 62, 72, 82,
        38, 48, 58, 68, 78, 92, 28, 42, 62, 96,
        52, 72, 82, 92, 38, 58, 68, 78, 48, 88
    ]
}

df = pd.DataFrame(data)

# 2. Generate target labels
df["result"] = (
    (
        df["assignment_marks"]
        + df["internal_marks"]
        + df["previous_score"]
    ) / 3 >= 50
).astype(int)

# 3. Save the dataset required by the tests
df.to_csv("student_results.csv", index=False)

# 4. Select features and target
features = [
    "assignment_marks",
    "internal_marks",
    "previous_score"
]

X = df[features]
y = df["result"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 6. Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 8. Save trained model
joblib.dump(model, "student_result_model.pkl")

# 9. Generate metrics.json
metrics = {
    "accuracy": float(accuracy),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Dataset generated successfully")
print("Model trained successfully")
print("Model accuracy:", round(accuracy, 4))
print("Metrics saved to metrics.json")
print("Training completed")
