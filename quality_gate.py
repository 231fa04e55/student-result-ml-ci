import json
import os
import sys


def main():
    metrics_file = "metrics.json"
    model_file = "student_result_model.pkl"
    dataset_file = "student_results.csv"

    required_files = [metrics_file, model_file, dataset_file]

    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"QUALITY GATE FAILED: Missing file {file_path}")
            sys.exit(1)

    try:
        with open(metrics_file, "r") as file:
            metrics = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"QUALITY GATE FAILED: Cannot read metrics.json: {error}")
        sys.exit(1)

    accuracy = metrics.get("accuracy")

    if not isinstance(accuracy, (int, float)) or isinstance(accuracy, bool):
        print("QUALITY GATE FAILED: Accuracy is missing or invalid.")
        sys.exit(1)

    if not 0 <= accuracy <= 1:
        print("QUALITY GATE FAILED: Accuracy must be between 0 and 1.")
        sys.exit(1)

    minimum_accuracy = 0.70

    if accuracy < minimum_accuracy:
        print(
            f"QUALITY GATE FAILED: Accuracy {accuracy:.2f} "
            f"is below {minimum_accuracy:.2f}."
        )
        sys.exit(1)

    print(f"QUALITY GATE PASSED: Model accuracy = {accuracy:.2f}")


if __name__ == "__main__":
    main()
