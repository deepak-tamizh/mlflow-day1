import mlflow
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-Model-Monitoring")

# Load registered champion model
model = mlflow.pyfunc.load_model(
    "models:/WineClassifier@champion"
)

# Load labelled test data
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Original test predictions
baseline_predictions = model.predict(X_test.values)

# Simulate incoming data drift
incoming = X_test.copy()
incoming["alcohol"] += 1.5
incoming["color_intensity"] *= 1.5

incoming_predictions = model.predict(incoming.values)

# Calculate performance metrics
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
incoming_accuracy = accuracy_score(y_test, incoming_predictions)

baseline_f1 = f1_score(y_test, baseline_predictions, average="weighted")
incoming_f1 = f1_score(y_test, incoming_predictions, average="weighted")

# Log monitoring results
with mlflow.start_run(run_name="Performance-Monitoring"):
    mlflow.log_metrics({
        "baseline_accuracy": baseline_accuracy,
        "incoming_accuracy": incoming_accuracy,
        "baseline_f1": baseline_f1,
        "incoming_f1": incoming_f1,
        "accuracy_change": incoming_accuracy - baseline_accuracy,
    })

print(f"Baseline accuracy:  {baseline_accuracy:.4f}")
print(f"Incoming accuracy:  {incoming_accuracy:.4f}")
print(f"Baseline F1-score:  {baseline_f1:.4f}")
print(f"Incoming F1-score:  {incoming_f1:.4f}")
print(f"Accuracy change:    {incoming_accuracy - baseline_accuracy:+.4f}")