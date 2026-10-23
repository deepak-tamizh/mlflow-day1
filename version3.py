from mlflow import MlflowClient

client = MlflowClient(
    tracking_uri="http://127.0.0.1:5000"
)

version = client.get_model_version_by_alias(
    "WineClassifier", "champion"
)

print("Champion version:", version.version)