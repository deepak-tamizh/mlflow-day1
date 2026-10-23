
from mlflow import MlflowClient

client = MlflowClient(
    tracking_uri="http://127.0.0.1:5000"
)

model_name = "WineClassifier"

# Move champion alias to version 2
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="2",
)

# Verify the change
champion = client.get_model_version_by_alias(
    model_name, "champion"
)

print("Current champion version:", champion.version)
