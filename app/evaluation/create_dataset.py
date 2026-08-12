# app/evaluation/create_dataset.py
# Creates a LangSmith evaluation dataset with ground truth QA pairs.
#
# Why ground truth matters:
# An evaluator needs something to compare against.
# Without ground truth answers, you can only measure style (is it coherent?)
# not accuracy (is it correct?).
# Ground truth answers are the benchmark your system is measured against.
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()

DATASET_NAME = "langchain-copilot-eval-v1"

# ── Ground truth QA pairs ────────────────────────────────────────────────────
# These are based on your Cambodia economics document.
# Each entry has:
#   input    → the question the system will be asked
#   output   → the correct answer it should produce
#
# Rules for writing good eval questions:
# 1. Cover different question types — factual, comparative, analytical
# 2. Include questions the system should answer AND questions it should decline
# 3. Be specific enough that correctness is unambiguous
# 4. Include questions that require multi-chunk retrieval

QA_PAIRS = [
    # ── Factual questions — single data point ────────────────────────────────
    {
        "input": "What was Cambodia's GDP in 2013?",
        "output": "Cambodia's GDP in 2013 was $15,228.0 million USD."
    },
    {
        "input": "What was Cambodia's GDP in 2023?",
        "output": "Cambodia's GDP in 2023 was $31,940.0 million USD."
    },
    {
        "input": "What was Cambodia's GDP growth rate in 2013?",
        "output": "Cambodia's GDP growth rate in 2013 was 8.35%."
    },
    {
        "input": "What was Cambodia's highest GDP growth rate and when did it occur?",
        "output": "Cambodia's highest GDP growth rate was 19.82%, occurring in 2008."
    },
    {
        "input": "What was Cambodia's GDP in 2000?",
        "output": "Cambodia's GDP in 2000 was $3,666.6 million USD."
    },
    {
        "input": "What happened to Cambodia's GDP growth in 2009?",
        "output": "Cambodia's GDP growth slowed significantly in 2009 due to the global financial crisis."
    },
    {
        "input": "What was Cambodia's GDP growth rate in 2020?",
        "output": "Cambodia's GDP growth rate declined sharply in 2020 due to the COVID-19 pandemic."
    },

    # ── Analytical questions — require interpretation ─────────────────────────
    {
        "input": "What are the main drivers of Cambodia's economic growth?",
        "output": "The main drivers include garments, tourism, construction, agriculture, foreign direct investment, and regional trade integration."
    },
    {
        "input": "What major external shocks affected Cambodia's economy?",
        "output": "The two major external shocks were the global financial crisis in 2009 and the COVID-19 pandemic in 2020, both of which caused significant slowdowns in growth."
    },
    {
        "input": "What is the main topic of the document?",
        "output": "The document analyzes Cambodia's economic development from 2000 to 2023, focusing on GDP data, growth rates, and key economic trends."
    },
    {
        "input": "How did Cambodia's economy perform between 2004 and 2008?",
        "output": "The period 2004 to 2008 was the most dynamic in terms of growth, averaging over 10% annual growth and peaking at 19.82% in 2008."
    },
    {
        "input": "What role did foreign direct investment play in Cambodia's growth?",
        "output": "Foreign direct investment became a crucial income source for Cambodia, enhancing productivity and contributing to economic output."
    },

    # ── Comparative questions — require multiple chunks ───────────────────────
    {
        "input": "How did Cambodia's GDP change from 2000 to 2023?",
        "output": "Cambodia's GDP grew from $3,666.6 million in 2000 to $31,940.0 million in 2023, representing growth of approximately 770%."
    },
    {
        "input": "Compare Cambodia's economic performance before and after the 2009 financial crisis.",
        "output": "Before 2009, Cambodia experienced exceptional growth peaking at 19.82% in 2008. After the crisis in 2009, growth slowed significantly but recovered in subsequent years."
    },
    {
        "input": "How did Cambodia's growth rate in 2022 and 2023 compare to its peak growth?",
        "output": "Cambodia's growth rates of 5.16% in 2022 and 8.25% in 2023 were significantly lower than its peak growth of 19.82% in 2008."
    },

    # ── Questions requiring calculation ──────────────────────────────────────
    {
        "input": "What was the total GDP growth percentage from 2013 to 2023?",
        "output": "Cambodia's GDP grew by approximately 109.7% from $15,228.0 million in 2013 to $31,940.0 million in 2023."
    },

    # ── Boundary questions — system should handle gracefully ─────────────────
    {
        "input": "What is Cambodia's GDP forecast for 2030?",
        "output": "The document does not contain GDP forecasts for 2030. It covers data from 2000 to 2023 only."
    },
    {
        "input": "What is the capital city of Cambodia?",
        "output": "The document does not contain information about Cambodia's capital city. It focuses on economic data only."
    },

    # ── Infrastructure and policy questions ───────────────────────────────────
    {
        "input": "What role did infrastructure development play in Cambodia's economy?",
        "output": "Infrastructure development contributed to economic transformation alongside urbanization and digital services, supporting Cambodia's broader economic growth."
    },
    {
        "input": "What does the document say about Cambodia's economic resilience?",
        "output": "The document states that Cambodia effectively navigated multiple external shocks including the global financial crisis and COVID-19 pandemic, demonstrating economic resilience and adaptability."
    },
]

def create_dataset():
    """
    Creates or updates the evaluation dataset in LangSmith.

    Why LangSmith for datasets?
    It stores inputs, outputs, and evaluation results together.
    You can see exactly which questions your system struggled with
    and track improvement over time across multiple evaluation runs.
    """

    # Check if dataset already exists
    existing_datasets = [d.name for d in client.list_datasets()]

    if DATASET_NAME in existing_datasets:
        print(f"Dataset '{DATASET_NAME}' already exists.")
        print("Deleting and recreating to ensure fresh data...")
        client.delete_dataset(dataset_name=DATASET_NAME)

    # Create fresh dataset
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Evaluation dataset for LangChain Research Copilot — Cambodia economics document"
    )

    # Add all QA pairs as examples
    client.create_examples(
        inputs=[{"question": qa["input"]} for qa in QA_PAIRS],
        outputs=[{"answer": qa["output"]} for qa in QA_PAIRS],
        dataset_id=dataset.id
    )

    print(f"Created dataset '{DATASET_NAME}' with {len(QA_PAIRS)} examples")
    print(f"View at: https://smith.langchain.com")
    return dataset

if __name__ == "__main__":
    create_dataset()