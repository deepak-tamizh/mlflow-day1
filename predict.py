
import mlflow
from sklearn.datasets import load_wine

# Connect to the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Identify the logged model
model_uri = "runs:/da330ec047a64b209e60bb43efce6348/wine_classifier"

# Load the model
model = mlflow.pyfunc.load_model(model_uri)

# Load example input data
wine = load_wine()
input_data = wine.data[:3]

# Make predictions
predictions = model.predict(input_data)

print("Predicted wine classes:", predictions)
