
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "WineClassifier"

client = MlflowClient(tracking_uri=TRACKING_URI)

versions = client.search_model_versions(
    f"name='{MODEL_NAME}'"
)

print(f"\nRegistered versions of {MODEL_NAME}\n")

for version in sorted(versions, key=lambda v: int(v.version)):
    run = client.get_run(version.run_id)
    metrics = run.data.metrics

    print(f"Version: {version.version}")
    print(f"Aliases: {version.aliases}")
    print(f"Accuracy: {metrics.get('accuracy', 'Not logged')}")
    print(f"F1-score: {metrics.get('f1_score', 'Not logged')}")
    print("-" * 35)
