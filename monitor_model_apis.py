
import math
import csv
import json
import time
import urllib.request
from datetime import datetime

APIS = {
    "existing_docker": "http://127.0.0.1:5001",
    "registry_v8": "http://127.0.0.1:5002",
}

BODY = {
    "dataframe_split": {
        "columns": [
            "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
            "magnesium", "total_phenols", "flavanoids",
            "nonflavanoid_phenols", "proanthocyanins",
            "color_intensity", "hue",
            "od280/od315_of_diluted_wines", "proline"
        ],
        "data": [[
            13.2, 2.77, 2.51, 18.5, 96.0, 1.9, 1.5,
            0.5, 1.3, 5.0, 1.0, 2.7, 1000.0
        ]]
    }
}

ROUNDS = 100
WARMUP_ROUNDS = 5

MAX_LATENCY_MS = 200
MAX_FAILURE_RATE = 0.05
MAX_P95_LATENCY_MS = 200
MAX_P99_LATENCY_MS = 200

results = []

for name, base_url in APIS.items():
    print(f"\nMonitoring: {name}")

    successes = 0
    failures = 0
    latencies = []

    # Warm-up requests: excluded from measured results and CSV
    print(f"Running {WARMUP_ROUNDS} warm-up requests...")

    for warmup_i in range(1, WARMUP_ROUNDS + 1):
        started = time.perf_counter()

        try:
            request = urllib.request.Request(
                base_url + "/invocations",
                data=json.dumps(BODY).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                response_body = json.loads(
                    response.read().decode("utf-8")
                )

                if response.status == 200 and "predictions" in response_body:
                    print(f"Warm-up {warmup_i}: PASS")
                else:
                    print(f"Warm-up {warmup_i}: FAIL | Unexpected response")

        except Exception as exc:
            print(f"Warm-up {warmup_i}: FAIL | {exc}")

    print(f"\nStarting {ROUNDS} measured requests...")

    # Measured requests
    for i in range(1, ROUNDS + 1):
        started = time.perf_counter()
        status = "FAIL"
        prediction = ""
        error = ""

        try:
            request = urllib.request.Request(
                base_url + "/invocations",
                data=json.dumps(BODY).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=10) as response:
                response_body = json.loads(
                    response.read().decode("utf-8")
                )

                elapsed_ms = (time.perf_counter() - started) * 1000

                if response.status == 200 and "predictions" in response_body:
                    status = "PASS"
                    prediction = str(response_body["predictions"])
                    successes += 1
                    latencies.append(elapsed_ms)
                else:
                    failures += 1
                    error = "Unexpected response format"

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            failures += 1
            error = str(exc)

        results.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "api": name,
            "request": i,
            "status": status,
            "latency_ms": round(elapsed_ms, 2),
            "prediction": prediction,
            "error": error,
        })

        print(
            f"Request {i:03d}: {status} | "
            f"{elapsed_ms:.2f} ms"
            + (f" | {error}" if error else "")
        )

    # Calculate metrics using successful measured requests only
    avg = sum(latencies) / len(latencies) if latencies else 0
    sorted_latencies = sorted(latencies)

    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    p95_latency = sorted_latencies[p95_index] if sorted_latencies else 0

    p99_index = max(0, math.ceil(0.99 * len(sorted_latencies)) - 1)
    p99_latency = sorted_latencies[p99_index] if sorted_latencies else 0

    print(f"\nSuccessful requests: {successes}/{ROUNDS}")
    print(f"Failed requests: {failures}/{ROUNDS}")
    print(f"Average successful latency: {avg:.2f} ms")
    print(f"P95 latency: {p95_latency:.2f} ms")
    print(f"P99 latency: {p99_latency:.2f} ms")

    failure_rate = failures / ROUNDS

    print("\n--- ALERT CHECK ---")

    if avg > MAX_LATENCY_MS:
        print("WARNING: Average latency exceeded the threshold!")
    else:
        print("OK: Average latency is within the threshold.")

    if p95_latency > MAX_P95_LATENCY_MS:
        print("WARNING: P95 latency exceeded the threshold!")
    else:
        print("OK: P95 latency is within the threshold.")

    if p99_latency > MAX_P99_LATENCY_MS:
        print("WARNING: P99 latency exceeded the threshold!")
    else:
        print("OK: P99 latency is within the threshold.")

    if failure_rate > MAX_FAILURE_RATE:
        print("WARNING: Failure rate exceeded the threshold!")
    else:
        print("OK: Failure rate is within the threshold.")


# Save measured requests only; warm-up requests are excluded
with open("model_monitoring.csv", "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "timestamp", "api", "request", "status",
        "latency_ms", "prediction", "error"
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\nDetailed measured results saved to model_monitoring.csv")
