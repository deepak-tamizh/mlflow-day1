import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "WineClassifier"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Wine-Retraining")
client = MlflowClient(tracking_uri=TRACKING_URI)

# 1. Load the current champion
champion = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

# 2. Prepare the same test data for both models
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Evaluate the current champion
champion_predictions = champion.predict(X_test.values)

champion_accuracy = accuracy_score(y_test, champion_predictions)
champion_f1 = f1_score(
    y_test, champion_predictions, average="weighted"
)

print(f"Champion accuracy: {champion_accuracy:.4f}")
print(f"Champion F1:       {champion_f1:.4f}")

# 4. Train and evaluate a candidate
candidate = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=123,
)

with mlflow.start_run(run_name="Retraining-Candidate") as run:
    candidate.fit(X_train, y_train)
    candidate_predictions = candidate.predict(X_test)

    candidate_accuracy = accuracy_score(y_test, candidate_predictions)
    candidate_f1 = f1_score(
        y_test, candidate_predictions, average="weighted"
    )

    mlflow.log_params({
        "n_estimators": 300,
        "max_depth": "None",
        "random_state": 123,
    })

    mlflow.log_metrics({
        "accuracy": candidate_accuracy,
        "f1_score": candidate_f1,
        "champion_accuracy": champion_accuracy,
        "champion_f1_score": champion_f1,
    })

    mlflow.sklearn.log_model(
        candidate,
        name="candidate_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    candidate_run_id = run.info.run_id

print(f"\nCandidate accuracy: {candidate_accuracy:.4f}")
print(f"Candidate F1:       {candidate_f1:.4f}")

# 5. Promote only if the candidate matches or beats the champion on both metrics
if (
    candidate_accuracy >= champion_accuracy
    and candidate_f1 >= champion_f1
):
    model_uri = f"runs:/{candidate_run_id}/candidate_model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print(f"\nCandidate promoted: version {registered.version}")
else:
    print("\nCandidate rejected: champion remains unchanged.")

print("\nRetraining workflow completed.")





