
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("GenAI-Tracing")

@mlflow.trace
def retrieve_context(question):
    return "MLflow is a platform for managing machine learning workflows."

@mlflow.trace
def generate_answer(question, context):
    return f"Question: {question}\nAnswer: {context}"

@mlflow.trace
def ask_assistant(question):
    context = retrieve_context(question)
    answer = generate_answer(question, context)
    return answer

result = ask_assistant("What is MLflow?")
print(result)
