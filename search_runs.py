
import mlflow

# 1. Search for experiment runs
runs = mlflow.search_runs(
    experiment_names=["Wine Classification"]
)

# 2. Display selected columns
columns = [
    "run_id",
    "status",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy"
]

print(runs[columns].head(10))

# 3. Find the run with the highest accuracy
completed_runs = runs[
    (runs["status"] == "FINISHED") &
    (runs["metrics.accuracy"].notna())
]

best_run = completed_runs.loc[
    completed_runs["metrics.accuracy"].idxmax()
]

high_accuracy_runs = runs[
    runs["metrics.accuracy"] >= 0.95
]

print("\nBest Run ID:", best_run["run_id"])
print("Best Accuracy:", best_run["metrics.accuracy"])
print("Best Parameters:")
print("n_estimators:", best_run["params.n_estimators"])
print("max_depth:", best_run["params.max_depth"])
print(high_accuracy_runs)
