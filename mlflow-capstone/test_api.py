
import requests
from sklearn.datasets import load_wine

wine = load_wine()
sample = wine.data[0].tolist() #.tolist() converts a NumPy array into a Python list.

# It prepares the input data to be sent to the model server in JSON format. Here, only one sample is being sent.
payload = {
    "inputs": [sample]
}

# requests.post() — Sends an HTTP POST request.
# 127.0.0.1:5001 — The model server running on port 5001 on your computer.
# /invocations — The MLflow model prediction endpoint.
# json=payload — Sends the input in JSON format.
# timeout=30 — The request times out if no response is received within 30 seconds.
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json=payload,
    timeout=30,
)

# If an HTTP error occurs on the server, it raises an exception. If the request is successful, execution continues to the next lines.
response.raise_for_status()

print("Status Code:", response.status_code)
print("Prediction:", response.json())
print("Actual Class:", int(wine.target[0]))




