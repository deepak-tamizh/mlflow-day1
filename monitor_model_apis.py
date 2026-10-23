
import csv
import json
import time
import urllib.request
import urllib.error
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

ROUNDS = 10
results = []

for name, base_url in APIS.items():
    print(f"\nMonitoring: {name}")
    successes = 0
    failures = 0
    latencies = []

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
                elapsed_ms = (time.perf_counter() - started) * 1000
                response_body = json.loads(
                    response.read().decode("utf-8")
                )

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
            f"Request {i:02d}: {status} | "
            f"{elapsed_ms:.2f} ms"
            + (f" | {error}" if error else "")
        )

    avg = sum(latencies) / len(latencies) if latencies else 0
    print(f"Successful requests: {successes}/{ROUNDS}")
    print(f"Failed requests: {failures}/{ROUNDS}")
    print(f"Average successful latency: {avg:.2f} ms")

with open("model_monitoring.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\nDetailed results saved to model_monitoring.csv")
