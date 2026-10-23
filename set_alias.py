
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

model_name = "WineClassifier"

# Get registered versions
versions = client.search_model_versions(
    f"name='{model_name}'"
)

# Select the version with the highest recorded accuracy
best_version = max(
    versions,
    key=lambda v: float(
        client.get_run(v.run_id)
        .data.metrics.get("accuracy", float("-inf"))
    )
)

# Assign champion to the best version
client.set_registered_model_alias(
    model_name,
    "champion",
    best_version.version
)

print("Champion version:", best_version.version)

# Assign challenger to another version, if available
other_versions = [
    v for v in versions
    if v.version != best_version.version
]

if other_versions:
    challenger = max(
        other_versions,
        key=lambda v: int(v.version)
    )

    client.set_registered_model_alias(
        model_name,
        "challenger",
        challenger.version
    )

    print("Challenger version:", challenger.version)
