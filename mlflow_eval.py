import mlflow
import pandas as pd
from mlflow.metrics.genai import make_genai_metric, EvaluationExample
import os

# Set environment variables to point MLflow (via OpenAI client) to our local vLLM
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"

def run_mlflow_evaluation():
    # 1. Prepare a dataset
    eval_df = pd.DataFrame({
        "question": [
            "What is the capital of Italy?",
            "What is 2 + 2?",
            "Who wrote Hamlet?"
        ],
        "answer": [
            "The capital of Italy is Rome.",
            "2 + 2 equals 4.",
            "Hamlet was written by William Shakespeare."
        ],
        "ground_truth": [
            "Rome",
            "4",
            "William Shakespeare"
        ]
    })

    # 2. Define a custom GenAI metric
    # This metric will ask the LLM (our local vLLM) to judge the correctness of the answer.
    
    # Example for the judge to understand what we want
    example = EvaluationExample(
        input="What is the capital of France?",
        output="Paris",
        score=5,
        justification="The answer is correct and concise."
    )

    correctness_metric = make_genai_metric(
        name="correctness",
        definition="Correctness of the answer based on the question and ground truth.",
        grading_prompt=(
            "You are a helpful judge. "
            "Check if the answer provided in 'answer' correctly answers the 'question' "
            "comparable to the 'ground_truth'. "
            "Give a score from 1 to 5, where 5 is perfect."
        ),
        examples=[example],
        model="openai:/Qwen/Qwen2.5-1.5B-Instruct", # Points to our local model via OpenAI provider
        parameters={"temperature": 0.0},
        aggregations=["mean"],
        greater_is_better=True,
    )

    # 3. Run the evaluation
    # We are evaluating the static 'answer' column against 'question'
    
    with mlflow.start_run() as run:
        print(f"Starting MLflow run: {run.info.run_id}")
        
        results = mlflow.evaluate(
            data=eval_df,
            targets="ground_truth", # Optional, used if metric needs it
            predictions="answer",   # The column containing our model/system outputs
            model_type="text",
            extra_metrics=[correctness_metric],
            evaluator_config={
                "col_mapping": {
                    "inputs": "question"
                }
            }
        )
        
        print("\nEvaluation Results:")
        print(results.metrics)
        print("\nEvaluation Table:")
        print(results.tables["eval_results_table"])

if __name__ == "__main__":
    run_mlflow_evaluation()
