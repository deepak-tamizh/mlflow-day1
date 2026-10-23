
import os
import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd

from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# 1. Connect to the tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Wine Classification")


# 2. Load the dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 3. Train the preprocessing step
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train the classifier on scaled features
classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

classifier.fit(X_train_scaled, y_train)


# 5. Save the preprocessing and classifier objects
joblib.dump(scaler, "wine_scaler.joblib")
joblib.dump(classifier, "wine_classifier.joblib")


# 6. Define a custom PyFunc model
class WinePipeline(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.classifier = joblib.load(
            context.artifacts["classifier"]
        )

    def predict(self, context, model_input, params=None):
        scaled_input = self.scaler.transform(model_input)
        return self.classifier.predict(scaled_input)


# 7. Prepare an input example and signature
input_example = X_test.head(3)
example_predictions = classifier.predict(
    scaler.transform(input_example)
)

signature = infer_signature(
    input_example,
    example_predictions
)


# 8. Log the custom model
with mlflow.start_run(run_name="Custom Wine PyFunc") as run:

    mlflow.pyfunc.log_model(
        name="wine_custom_pipeline",
        python_model=WinePipeline(),
        artifacts={
            "scaler": os.path.abspath("wine_scaler.joblib"),
            "classifier": os.path.abspath("wine_classifier.joblib")
        },
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "mlflow==3.17.1",
            "scikit-learn==1.9.1",
            "pandas==3.0.6",
            "joblib"
        ]
    )

    run_id = run.info.run_id


# 9. Load the logged custom model
model_uri = f"runs:/{run_id}/wine_custom_pipeline"
loaded_model = mlflow.pyfunc.load_model(model_uri)


# 10. Predict using RAW input features
predictions = loaded_model.predict(X_test.head(3))

print("Custom PyFunc model logged successfully!")
print("Run ID:", run_id)
print("Model URI:", model_uri)
print("Predictions:", predictions)
