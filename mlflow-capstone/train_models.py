
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-MLflow-Capstone")

X, y = load_wine(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

configs = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 8},
]

best_accuracy = -1
best_run_id = None

for config in configs:
    with mlflow.start_run(
        run_name=f"RF_{config['n_estimators']}_{config['max_depth']}"
    ):
        model = RandomForestClassifier(
            **config,
            random_state=42,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_params(config)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            name="wine_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        run_id = mlflow.active_run().info.run_id

        print(
            f"Config: {config} | "
            f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = run_id

print("\nTraining completed!")
print("Best accuracy:", best_accuracy)
print("Best run ID:", best_run_id)
