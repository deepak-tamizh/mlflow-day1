
from typing import Literal
from mlflow.genai.judges import make_judge

correctness_judge = make_judge(
    name="answer_correctness",
    instructions=(
        "Evaluate whether the answer in {{ outputs }} "
        "correctly answers the question in {{ inputs }}. "
        "Compare it with the reference in {{ expectations }}. "
        "Return 'correct' or 'incorrect'."
    ),
    feedback_value_type=Literal["correct", "incorrect"],
    model="openai:/gpt-4o-mini",
)

result = correctness_judge(
    inputs={"question": "What is MLflow?"},
    outputs="MLflow manages machine learning experiments and models.",
    expectations={
        "expected_answer": "MLflow manages machine learning workflows."
    },
)

print("Verdict:", result.value)
print("Reason:", result.rationale)
