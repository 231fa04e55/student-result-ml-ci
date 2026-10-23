import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create demonstration student dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5,
                    3, 6, 7, 1, 4, 8, 2, 6, 5, 7],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90, 52, 72,
                   62, 78, 85, 45, 68, 95, 58, 82, 74, 88],
    "result": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
               0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

import joblib
joblib.dump(model, "student_result_model.pkl")

metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model trained successfully")
print("Model accuracy:", round(accuracy, 4))
print("Metrics saved to metrics.json")
