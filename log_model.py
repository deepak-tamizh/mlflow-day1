
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Connect to your tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Classification")

# Load the dataset
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data,
    wine.target,
    test_size=0.2,
    random_state=42,
    stratify=wine.target
)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Infer the model signature
predictions = model.predict(X_train)
signature = infer_signature(X_train, predictions)

# Log the model with a signature and input example
with mlflow.start_run(run_name="Signed Wine Model"):
    mlflow.sklearn.log_model(
        sk_model=model,
        name="wine_classifier",
        signature=signature,
        input_example=X_train[:3],
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("Model logged with signature and input example!")
