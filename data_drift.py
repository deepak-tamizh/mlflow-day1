
import mlflow
import pandas as pd

from scipy.stats import ks_2samp
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine-Data-Drift")

# Load the original Wine dataset
wine = load_wine(as_frame=True)
X = wine.data

baseline, incoming = train_test_split(
    X, test_size=0.2, random_state=42
)

# Simulate changes in incoming data
incoming = incoming.copy()
incoming["alcohol"] += 1.5
incoming["color_intensity"] *= 1.5

results = []

with mlflow.start_run(run_name="Data-Drift-Detection"):
    for feature in X.columns:
        statistic, p_value = ks_2samp(
            baseline[feature],
            incoming[feature],
        )

        drift = p_value < 0.05

        results.append({
            "feature": feature,
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": drift,
        })

        mlflow.log_metric(f"{feature}_p_value", float(p_value))
        mlflow.log_metric(
            f"{feature}_drift",
            int(drift),
        )

    report = pd.DataFrame(results)
    report.to_csv("drift_report.csv", index=False)
    mlflow.log_artifact("drift_report.csv")

    print("\nData Drift Report:")
    print(
        report[report["drift_detected"]][
            ["feature", "ks_statistic", "p_value"]
        ].to_string(index=False)
    )

    print(
        "\nFeatures showing drift:",
        int(report["drift_detected"].sum()),
        "out of",
        len(report),
    )
