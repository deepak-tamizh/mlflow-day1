
import json
import time
import urllib.error
import urllib.request
import subprocess

import mlflow
import mlflow.sklearn

from mlflow import MlflowClient
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_NAME = "WineClassifier"
EXPERIMENT_NAME = "Wine-End-to-End-Verification"
IMAGE_NAME = "wine-classifier"
CONTAINER_NAME = "wine-classifier-e2e"
API_URL = "http://127.0.0.1:5001"

MIN_ACCURACY = 0.95
MIN_F1 = 0.95

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(tracking_uri=TRACKING_URI)


def api_request(path, method="GET", payload=None):
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        API_URL + path,
        data=data,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, body


def verify_docker_api():
    print("\n=== DOCKER API VERIFICATION ===")

    # Reuse the local image. Never build or pull an image here.
    subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        check=True,
        capture_output=True,
        text=True,
    )

    # Remove only a previous container created by this script.
    old = subprocess.run(
        ["docker", "inspect", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )

    if old.returncode == 0:
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            check=True,
            capture_output=True,
            text=True,
        )

    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", "5001:8080",
            IMAGE_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        # Wait for the model server to start.
        for attempt in range(30):
            try:
                status, body = api_request("/ping")
                if status == 200:
                    print("API health check: PASS")
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        else:
            logs = subprocess.run(
                ["docker", "logs", CONTAINER_NAME],
                capture_output=True,
                text=True,
            )
            print("API health check: FAIL")
            print(logs.stdout)
            print(logs.stderr)
            return False

        wine = load_wine(as_frame=True)
        sample = wine.data.iloc[[0]]

        payload = {
            "dataframe_split": {
                "columns": sample.columns.tolist(),
                "data": sample.values.tolist(),
            }
        }

        status, body = api_request(
            "/invocations",
            method="POST",
            payload=payload,
        )

        result = json.loads(body)
        if status == 200 and "predictions" in result:
            print("Prediction API: PASS")
            print("Prediction response:", result)
            print(
                "Note: this prediction came from the existing "
                "Docker image, not necessarily the new champion."
            )
            return True

        print("Prediction API: FAIL")
        print(body)
        return False

    except Exception as exc:
        print("Docker API verification failed:", exc)
        return False

    finally:
        subprocess.run(
            ["docker", "rm", "-f", CONTAINER_NAME],
            capture_output=True,
            text=True,
        )


def main():
    print("=== 1. CHECK CURRENT CHAMPION ===")

    champion_version = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )
    champion_version_number = champion_version.version

    champion = mlflow.pyfunc.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    print("Champion version:", champion_version_number)

    print("\n=== 2. PREPARE DATA ===")

    wine = load_wine(as_frame=True)
    X, y = wine.data, wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    print("\n=== 3. EVALUATE CHAMPION ===")

    champion_predictions = champion.predict(X_test)

    champion_accuracy = accuracy_score(
        y_test, champion_predictions
    )
    champion_f1 = f1_score(
        y_test, champion_predictions, average="weighted"
    )

    print("Champion accuracy:", round(champion_accuracy, 4))
    print("Champion F1:", round(champion_f1, 4))

    print("\n=== 4. TRAIN AND LOG CANDIDATE ===")

    candidate = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=7,
    )

    with mlflow.start_run(run_name="End-to-End-Candidate") as run:
        candidate.fit(X_train, y_train)
        predictions = candidate.predict(X_test)

        candidate_accuracy = accuracy_score(y_test, predictions)
        candidate_f1 = f1_score(
            y_test, predictions, average="weighted"
        )

        mlflow.log_params({
            "n_estimators": 300,
            "max_depth": "None",
            "random_state": 7,
            "champion_version_before_run": champion_version_number,
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

    print("Candidate run ID:", candidate_run_id)
    print("Candidate accuracy:", round(candidate_accuracy, 4))
    print("Candidate F1:", round(candidate_f1, 4))

    print("\n=== 5. APPLY PROMOTION RULES ===")

    checks = {
        "Minimum accuracy": candidate_accuracy >= MIN_ACCURACY,
        "Minimum F1": candidate_f1 >= MIN_F1,
        "Accuracy not reduced": candidate_accuracy >= champion_accuracy,
        "F1 not reduced": candidate_f1 >= champion_f1,
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    promoted = False

    if all(checks.values()):
        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=MODEL_NAME,
        )

        # Wait until the registry version is ready.
        for _ in range(30):
            current = client.get_model_version(
                MODEL_NAME, registered.version
            )
            if current.status == "READY":
                break
            if current.status == "FAILED":
                raise RuntimeError("Candidate registration failed.")
            time.sleep(2)
        else:
            raise TimeoutError("Candidate registration timed out.")

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=registered.version,
        )

        promoted = True
        print("Promotion: PASS")
        print("New champion version:", registered.version)
    else:
        print("Promotion: REJECTED; champion remains unchanged.")

    print("\n=== 6. VERIFY REGISTRY ===")

    final_champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )
    print("Champion alias points to:", final_champion.version)

    if promoted and final_champion.version != registered.version:
        raise RuntimeError("Champion alias verification failed.")

    if not promoted and final_champion.version != champion_version_number:
        raise RuntimeError(
            "Champion changed unexpectedly during this run."
        )

    print("Registry verification: PASS")

    print("\n=== 7. VERIFY EXISTING DOCKER IMAGE ===")
    api_passed = verify_docker_api()

    print("\n=== FINAL RESULT ===")
    print("Training and evaluation: PASS")
    print("Registry verification: PASS")
    print("Docker API verification:", "PASS" if api_passed else "FAIL")
    print(
        "Deployment status: existing image tested; "
        "new champion was not deployed to that image."
    )

    if not api_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
