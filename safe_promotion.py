
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "WineClassifier"
CANDIDATE_VERSION = "2"

MIN_ACCURACY = 0.95
MIN_F1 = 0.95

client = MlflowClient(tracking_uri=TRACKING_URI)

# Get current champion and candidate
champion = client.get_model_version_by_alias(
    MODEL_NAME, "champion"
)
candidate = client.get_model_version(
    MODEL_NAME, CANDIDATE_VERSION
)

champion_metrics = client.get_run(
    champion.run_id
).data.metrics

candidate_metrics = client.get_run(
    candidate.run_id
).data.metrics

champion_accuracy = champion_metrics.get("accuracy")
champion_f1 = champion_metrics.get("f1_score")

candidate_accuracy = candidate_metrics.get("accuracy")
candidate_f1 = candidate_metrics.get("f1_score")

print("Current champion:", champion.version)
print("Candidate:", candidate.version)
print("Champion accuracy:", champion_accuracy)
print("Candidate accuracy:", candidate_accuracy)
print("Champion F1:", champion_f1)
print("Candidate F1:", candidate_f1)

# Apply promotion rules
checks = {
    "Minimum accuracy": (
        candidate_accuracy is not None
        and candidate_accuracy >= MIN_ACCURACY
    ),
    "Minimum F1": (
        candidate_f1 is not None
        and candidate_f1 >= MIN_F1
    ),
    "Accuracy not reduced": (
        candidate_accuracy is not None
        and champion_accuracy is not None
        and candidate_accuracy >= champion_accuracy
    ),
    "F1 not reduced": (
        candidate_f1 is not None
        and champion_f1 is not None
        and candidate_f1 >= champion_f1
    ),
}

print("\nPromotion checks:")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if all(checks.values()):
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=CANDIDATE_VERSION,
    )
    print("\nCandidate promoted successfully!")
else:
    print("\nCandidate rejected. Champion unchanged.")
