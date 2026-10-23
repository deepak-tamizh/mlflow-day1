
import mlflow
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Connect to your tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Classification")

# 2. Load the dataset
wine = load_wine()

X = pd.DataFrame(
    wine.data,
    columns=wine.feature_names
)
y = wine.target

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Prepare evaluation data
eval_data = X_test.copy()
eval_data["target"] = y_test

# 6. Evaluate with MLflow
with mlflow.start_run(run_name="Wine Model Evaluation"):

    results = mlflow.models.evaluate(
    model=model.predict,
    data=eval_data,
    targets="target",
    model_type="classifier"
    )

    print("Evaluation completed!")
    print(results.metrics)
