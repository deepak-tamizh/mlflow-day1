
import mlflow
from mlflow import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Wine-MLflow-Capstone"
MODEL_NAME = "WineCapstone"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# Find the best run by test accuracy
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    raise ValueError("Experiment not found. Run train_models.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)

if not runs:
    raise ValueError("No training runs found.")

best_run = runs[0]
run_id = best_run.info.run_id

print("Best run ID:", run_id)
print("Best accuracy:", best_run.data.metrics["accuracy"])

# Register the model artifact from the best run
model_uri = f"runs:/{run_id}/wine_model"

registered = mlflow.register_model(
    model_uri=model_uri,
    name=MODEL_NAME,
)

# Assign the champion alias
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered.version,
)

print("Registered model:", MODEL_NAME)
print("Version:", registered.version)
print("Alias: champion")
