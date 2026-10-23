
import mlflow
import pandas as pd
from sklearn.datasets import load_wine

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the registered champion model
model = mlflow.pyfunc.load_model(
    "models:/WineCapstone@champion"
)

# Load one sample with the original feature names
wine = load_wine(as_frame=True)
sample = wine.data.iloc[[0]]

# Predict
prediction = model.predict(sample.to_numpy())

print("Input features:")
print(sample)

print("\nPredicted wine class:", int(prediction[0]))
print("Actual wine class:", int(wine.target.iloc[0]))
