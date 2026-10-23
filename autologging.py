
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
wine = load_wine()
X = wine.data
y = wine.target

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Set the experiment
mlflow.set_experiment("Wine Classification")

# 4. Enable autologging
mlflow.sklearn.autolog()

# 5. Run five experiments
configs = [
    {"n_estimators": 10, "max_depth": 2},
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 8},
    {"n_estimators": 200, "max_depth": None},
]

for config in configs:
    with mlflow.start_run():
        # Create the model
        model = RandomForestClassifier(
            **config,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log the test accuracy manually
        mlflow.log_metric("test_accuracy", accuracy)

        print(
            f"Parameters: {config} | "
            f"Test Accuracy: {accuracy:.4f}"
        )
