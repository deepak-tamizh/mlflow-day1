
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42,
    stratify=wine.target
)

# Configure experiment
mlflow.set_experiment("Wine Classification")

configs = [
    {"n_estimators": 10, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 8},
    {"n_estimators": 200, "max_depth": None},
]

# Start the parent run
with mlflow.start_run(
    run_name="Hyperparameter Search",
    description="Comparing five Random Forest configurations",
    tags={"project": "wine-classification", "stage": "day-2"}
):
    for config in configs:

        # Start a child run
        with mlflow.start_run(
            run_name=f"Trial {config['n_estimators']}",
            nested=True,
            tags={"model": "RandomForest"}
        ):
            model = RandomForestClassifier(
                **config,
                random_state=42
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            mlflow.log_params(config)
            mlflow.log_metric("accuracy", accuracy)

            print(config, "Accuracy:", accuracy)

print("Hyperparameter search completed!")
