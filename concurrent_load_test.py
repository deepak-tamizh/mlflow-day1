
import math
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

APIS = {
    "existing_docker": "http://127.0.0.1:5001",
    "registry_v8": "http://127.0.0.1:5002",
}

TOTAL_REQUESTS = 100
WORKERS = 5

columns = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids",
    "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue",
    "od280/od315_of_diluted_wines", "proline"
]

sample = [[
    13.2, 2.77, 2.51, 18.5, 96.0, 1.04, 2.1,
    0.35, 1.4, 5.0, 1.05, 3.4, 1050.0
]]

payload = {
    "dataframe_split": {
        "columns": columns,
        "data": sample
    }
}


def send_request(url):
    start = time.perf_counter()

    try:
        response = requests.post(
            f"{url}/invocations",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        success = True

    except requests.RequestException:
        success = False

    latency_ms = (time.perf_counter() - start) * 1000
    return success, latency_ms


for name, url in APIS.items():
    print(f"\nTesting {name} with {WORKERS} workers...")

    # Warm up the API
    for _ in range(5):
        send_request(url)

    successful = 0
    failed = 0
    latencies = []

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(send_request, url)
            for _ in range(TOTAL_REQUESTS)
        ]

        for future in as_completed(futures):
            success, latency = future.result()

            if success:
                successful += 1
                latencies.append(latency)
            else:
                failed += 1

    elapsed = time.perf_counter() - start

    throughput = successful / elapsed if elapsed else 0
    failure_rate = failed / TOTAL_REQUESTS * 100

    print(f"Total time:       {elapsed:.2f} seconds")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Throughput:       {throughput:.2f} requests/sec")
    print(f"Failure rate:     {failure_rate:.2f}%")

    if latencies:
        print(f"Average latency:  {mean(latencies):.2f} ms")
        print(f"Max latency:      {max(latencies):.2f} ms")

        sorted_latencies = sorted(latencies)
        p95 = sorted_latencies[math.ceil(0.95 * len(sorted_latencies)) - 1]
        p99 = sorted_latencies[math.ceil(0.99 * len(sorted_latencies)) - 1]
        print(f"P95 latency:      {p95:.2f} ms")
        print(f"P99 latency:      {p99:.2f} ms")