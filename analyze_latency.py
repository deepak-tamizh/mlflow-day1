
import csv
from collections import defaultdict
from statistics import median

results = defaultdict(list)

with open("model_monitoring.csv", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["status"] == "PASS":
            results[row["api"]].append({
                "request": row["request"],
                "latency": float(row["latency_ms"])
            })

for api, requests in results.items():
    latencies = [item["latency"] for item in requests]
    med = median(latencies)

    print(f"\nAPI: {api}")
    print(f"Successful requests: {len(latencies)}")
    print(f"Median latency: {med:.2f} ms")
    print(f"Slowest latency: {max(latencies):.2f} ms")

    print("\nSlowest 5 requests:")

    for item in sorted(
        requests,
        key=lambda x: x["latency"],
        reverse=True
    )[:5]:
        print(
            f"Request {item['request']}: "
            f"{item['latency']:.2f} ms"
        )

    outliers = [
        item for item in requests
        if item["latency"] > 3 * med
    ]

    print(f"\nPotential outliers (>3x median): {len(outliers)}")

    
    first_10 = [
        item["latency"] for item in requests
        if int(item["request"]) <= 10
    ]

    remaining = [
        item["latency"] for item in requests
        if int(item["request"]) > 10
    ]

    print("\nFirst 10 requests:")
    print(f"Average: {sum(first_10) / len(first_10):.2f} ms")

    print("\nRemaining requests:")
    print(f"Average: {sum(remaining) / len(remaining):.2f} ms")

    print("\nFirst 10 individual request latencies:")

    for item in sorted(
        requests,
        key=lambda x: int(x["request"])
    )[:10]:
        print(
            f"Request {item['request']}: "
            f"{item['latency']:.2f} ms"
        )

