# app/evaluation/run_evaluation.py
# Runs the evaluation dataset through your RAG system and scores results.
#
# Three evaluators:
#
# 1. Correctness (reference-based)
#    Compares system answer to ground truth answer
#    Uses LLM-as-judge to determine if they match
#    Score: 0 (wrong) or 1 (correct)
#
# 2. Faithfulness (reference-free)
#    Checks if the answer is grounded in retrieved context
#    Does not need ground truth — judges against source documents
#    Score: 0.0 to 1.0
#
# 3. Relevance (reference-free)
#    Checks if the answer actually addresses the question asked
#    Score: 0.0 to 1.0

# Add project root to Python path so 'app' module is always findable
# regardless of where the script is run from
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


client = Client()

DATASET_NAME = "langchain-copilot-eval-v1"

def build_rag_pipeline():
    """
    Builds the RAG pipeline for evaluation.
    This must match exactly what your production system uses.
    If you evaluate a different pipeline than what runs in production,
    your scores are meaningless.
    """
    from app.core.ingestion import load_existing_vectorstore
    from app.core.agent import build_research_agent, run_agent

    vectorstore = load_existing_vectorstore()
    agent = build_research_agent(vectorstore)

    def run_pipeline(inputs: dict) -> dict:
        """
        Wrapper function that matches the signature LangSmith expects.

        LangSmith calls this function for each example in the dataset.
        Input:  {"question": "What was Cambodia's GDP in 2013?"}
        Output: {"answer": "Cambodia's GDP in 2013 was $15,228 million."}
        """
        question = inputs["question"]
        answer = run_agent(agent, question, chat_history=[])
        return {"answer": answer}

    return run_pipeline

def build_correctness_evaluator():
    """
    LLM-as-judge evaluator that compares system answer to ground truth.

    Why LLM-as-judge instead of exact string matching?
    Because "Cambodia's GDP in 2013 was $15,228 million" and
    "In 2013, Cambodia had a GDP of 15228.0 million USD" are both correct
    but would fail exact matching. The LLM understands semantic equivalence.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    correctness_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating whether an AI system's answer is correct.

Compare the system answer to the reference answer and determine if they convey
the same information.

Rules:
- Minor wording differences are acceptable
- Numbers must match (within reasonable rounding)
- If the system correctly says information is not available, that counts as correct
- Score 1 if correct, 0 if incorrect

Respond with ONLY a JSON object:
{{"score": 0 or 1, "reasoning": "brief explanation"}}"""),
        ("human", """Question: {question}
Reference answer: {reference}
System answer: {prediction}""")
    ])

    def evaluate_correctness(run, example) -> dict:
        """
        Evaluator function called by LangSmith for each example.

        Args:
            run:     the system's output for this example
            example: the ground truth example from the dataset
        """
        question = example.inputs["question"]
        reference = example.outputs["answer"]
        prediction = run.outputs["answer"]

        chain = correctness_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "reference": reference,
            "prediction": prediction
        })

        import json
        try:
            parsed = json.loads(result)
            score = float(parsed["score"])
            reasoning = parsed.get("reasoning", "")
        except Exception:
            score = 0.0
            reasoning = "Failed to parse evaluator response"

        return {
            "key": "correctness",
            "score": score,
            "comment": reasoning
        }

    return evaluate_correctness

def build_faithfulness_evaluator():
    """
    Evaluates whether the answer is grounded in the source documents.
    A hallucinated answer may be correct but not faithful —
    it states things not supported by the retrieved context.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    faithfulness_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are evaluating whether an AI answer is faithful to its sources.

A faithful answer only contains claims that are directly supported by
the provided context. It does not add information from outside the context.

Score:
1.0 = fully faithful, every claim is supported by context
0.5 = partially faithful, some claims unsupported
0.0 = not faithful, makes claims not in context

Respond with ONLY a JSON object:
{{"score": 0.0 to 1.0, "reasoning": "brief explanation"}}"""),
        ("human", """Question: {question}
Answer: {answer}
Context used: {context}""")
    ])

    def evaluate_faithfulness(run, example) -> dict:
        question = example.inputs["question"]
        answer = run.outputs["answer"]

        # For faithfulness we need the retrieved context
        # We re-run retrieval to get what the system actually used
        from app.core.ingestion import load_existing_vectorstore
        vectorstore = load_existing_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        chain = faithfulness_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context
        })

        import json
        try:
            parsed = json.loads(result)
            score = float(parsed["score"])
            reasoning = parsed.get("reasoning", "")
        except Exception:
            score = 0.0
            reasoning = "Failed to parse evaluator response"

        return {
            "key": "faithfulness",
            "score": score,
            "comment": reasoning
        }

    return evaluate_faithfulness

def run_evaluation():
    """
    Runs the full evaluation pipeline and prints a summary.
    Results are automatically saved to LangSmith.
    """
    print(f"Running evaluation on dataset: {DATASET_NAME}")
    print("This will take 2-3 minutes...\n")

    pipeline = build_rag_pipeline()
    correctness_evaluator = build_correctness_evaluator()
    faithfulness_evaluator = build_faithfulness_evaluator()

    # evaluate() runs every example through the pipeline
    # and scores each result with every evaluator
    results = evaluate(
        pipeline,
        data=DATASET_NAME,
        evaluators=[
            correctness_evaluator,
            faithfulness_evaluator,
        ],
        experiment_prefix="eval",
        metadata={"version": "1.0.0"}
    )

    # Print summary scores
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    scores = {"correctness": [], "faithfulness": []}

    for result in results:
        for feedback in result.get("evaluation_results", {}).get("results", []):
            key = feedback.key
            if key in scores:
                scores[key].append(feedback.score)

    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            print(f"{metric:20} {avg:.2f} ({len(values)} examples)")

    print("\nFull results visible at: https://smith.langchain.com")
    return scores

if __name__ == "__main__":
    run_evaluation()