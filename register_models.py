
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to your tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Classification")

client = MlflowClient()

# Load and split the dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42,
    stratify=wine.target
)

model_name = "WineClassifier"

# Two model configurations
configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 200, "max_depth": None},
]

for config in configs:
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            **config,
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="wine_classifier",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # Register the logged model
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name
        )

        print("Version:", registered.version)
        print("Accuracy:", accuracy)

print("Model registration completed!")
