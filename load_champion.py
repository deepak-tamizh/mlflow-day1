import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

model = mlflow.pyfunc.load_model(
    "models:/WineClassifier@champion"
)

print("Champion model loaded successfully!")


