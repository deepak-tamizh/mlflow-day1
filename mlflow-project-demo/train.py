
import argparse
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Read parameters from MLflow Projects
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()


# Connect to MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Do not call mlflow.set_experiment() here.
# MLflow Projects assigns the experiment to this run.


# Load dataset
X, y = load_wine(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train and log model
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(
        model,
        name="wine_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

    print(f"Accuracy: {accuracy:.4f}")
    print("Model logged successfully!")
