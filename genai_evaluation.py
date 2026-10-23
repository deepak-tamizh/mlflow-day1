
import mlflow
from mlflow.genai import scorer

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("GenAI-Evaluation")


# Simulated AI assistant
@mlflow.trace
def answer_question(question: str) -> str:
    if "mlflow" in question.lower():
        return "MLflow is a platform for managing machine learning workflows."
    return "I don't know the answer."


# Custom evaluation metric: exact match
@scorer
def answer_correctness(outputs: str, expectations: dict) -> bool:
    expected = expectations["expected_response"]
    return expected.lower() in outputs.lower()


# Evaluation dataset
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {
            "expected_response": "MLflow is a platform for managing machine learning workflows."
        },
    },
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {
            "expected_response": "MLflow is a platform for managing machine learning workflows."
        },
    },
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {
            "expected_response": "MLflow is an operating system."
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=answer_question,
    scorers=[answer_correctness],
)

print(results)
