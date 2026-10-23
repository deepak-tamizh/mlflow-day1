
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "WineClassifier"

MIN_ACCURACY = 0.95
MIN_F1 = 0.95

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("Wine-Automated-Pipeline")

client = MlflowClient(tracking_uri=TRACKING_URI)

# 1. Load the current champion
champion = mlflow.pyfunc.load_model(
    f"models:/{MODEL_NAME}@champion"
)

# 2. Prepare a consistent evaluation dataset
wine = load_wine(as_frame=True)
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Evaluate the champion on this test dataset
champion_predictions = champion.predict(X_test)

champion_accuracy = accuracy_score(
    y_test, champion_predictions
)
champion_f1 = f1_score(
    y_test, champion_predictions, average="weighted"
)

print("Champion accuracy:", round(champion_accuracy, 4))
print("Champion F1:", round(champion_f1, 4))

# 4. Train and evaluate a new candidate
candidate = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=7,
)

with mlflow.start_run(run_name="Automated-Candidate") as run:
    candidate.fit(X_train, y_train)
    predictions = candidate.predict(X_test)

    candidate_accuracy = accuracy_score(y_test, predictions)
    candidate_f1 = f1_score(y_test, predictions, average="weighted")

    mlflow.log_params({
        "n_estimators": 300,
        "max_depth": -1,
        "random_state": 7,
    })

    mlflow.log_metrics({
        "accuracy": candidate_accuracy,
        "f1_score": candidate_f1,
        "champion_accuracy": champion_accuracy,
        "champion_f1": champion_f1,
    })

    model_info = mlflow.sklearn.log_model(
        candidate,
        name="candidate_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    candidate_run_id = run.info.run_id
    candidate_model_uri = model_info.model_uri

print("\nCandidate accuracy:", round(candidate_accuracy, 4))
print("Candidate F1:", round(candidate_f1, 4))

# 5. Apply promotion rules
checks = {
    "Minimum accuracy": candidate_accuracy >= MIN_ACCURACY,
    "Minimum F1": candidate_f1 >= MIN_F1,
    "Accuracy not reduced": candidate_accuracy >= champion_accuracy,
    "F1 not reduced": candidate_f1 >= champion_f1,
}

print("\nPromotion checks:")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if all(checks.values()):
    registered = mlflow.register_model(
        model_uri=candidate_model_uri,
        name=MODEL_NAME,
    )

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print(f"\nPromoted version: {registered.version}")
else:
    print("\nCandidate rejected. Champion unchanged.")

print("\nAutomated pipeline completed.")
