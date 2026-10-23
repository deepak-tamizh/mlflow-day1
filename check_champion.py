
from mlflow import MlflowClient

client = MlflowClient(
    tracking_uri="http://127.0.0.1:5000"
)

model_name = "WineClassifier"

try:
    champion = client.get_model_version_by_alias(
        model_name, "champion"
    )
    print("Current champion version:", champion.version)
    print("Run ID:", champion.run_id)
except Exception as e:
    print("Champion alias lookup failed:", e)
