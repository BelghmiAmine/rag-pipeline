import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from openai import OpenAI
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference, Faithfulness
from dotenv import load_dotenv
load_dotenv()

from caas_RAG import EPFL_BASE_URL, LLM_MODEL_NAME, SYSTEM_PROMPT

DEFAULT_LLM = LLM_MODEL_NAME
DEFAULT_EVALUATOR_LLM = "swiss-ai/Apertus-70B-Instruct-2509"
SEED = 42


def run_evaluation(
    retrieval_results_path: str,
    llm_model_name: str = DEFAULT_LLM,
    evaluator_llm_name: str = DEFAULT_EVALUATOR_LLM,
):
    print(f"Loading retrieval results from '{retrieval_results_path}'...")
    with open(retrieval_results_path, encoding="utf-8") as f:
        data = json.load(f)

    queries: list[str] = data["queries"]
    references: list[str] = data["references"]
    all_docs: list[list[str]] = data["retrieved_docs"]
    metadata: dict = data.get("metadata", {})
    print(f"Loaded {len(queries)} queries. Metadata: {metadata}")

    # --- Step 1: generate answers ---
    import time
    api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(base_url=EPFL_BASE_URL, api_key=api_key, timeout=120.0)

    responses = []
    for query, docs in tqdm(zip(queries, all_docs), total=len(queries), desc="Generating answers"):
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\nQuestion: {query}",
            },
        ]
        for attempt in range(5):
            try:
                completion = client.chat.completions.create(
                    model=llm_model_name,
                    messages=messages,
                    temperature=0,
                    seed=SEED,
                )
                responses.append(completion.choices[0].message.content)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"\nAttempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            print(f"\nAll retries failed for query: {query[:80]}. Appending empty response.")
            responses.append("")

    # --- Step 2: print queries / generated answers / expected answers ---
    print("\n" + "=" * 80)
    for i, (query, response, reference) in enumerate(zip(queries, responses, references)):
        print(f"\n[{i+1}] Query:    {query}")
        print(f"     Generated: {response}")
        print(f"     Expected:  {reference}")
    print("=" * 80 + "\n")

    # --- Step 3: build RAGAS dataset ---
    evaluation_dataset = EvaluationDataset.from_list([
        {
            "user_input": query,
            "retrieved_contexts": docs,
            "response": response,
            "reference": reference,
        }
        for query, docs, response, reference in zip(queries, all_docs, responses, references)
    ])

    # --- Step 4: evaluate ---
    # LLMContextRecall:                 retrieval coverage  — does context contain what's needed?
    # LLMContextPrecisionWithReference: retrieval ranking   — are relevant docs ranked above irrelevant ones?
    # Faithfulness:                     generation quality  — are answer claims grounded in retrieved docs?
    print(f"Configuring evaluator LLM ({evaluator_llm_name})...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=evaluator_llm_name,
            base_url=EPFL_BASE_URL,
            api_key=api_key,
            temperature=0,
            model_kwargs={"seed": SEED},
        )
    )

    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), LLMContextPrecisionWithReference(), Faithfulness()],
        llm=evaluator_llm,
        raise_exceptions=False,
    )
    print(result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="CPU job: generate answers and evaluate with RAGAS from pre-computed retrieval results."
    )
    parser.add_argument(
        "--retrieval-results",
        type=str,
        required=True,
        help="Path to the JSON file produced by retrieve.py",
    )
    parser.add_argument("--llm", type=str, default=DEFAULT_LLM, help="Generator LLM model name (EPFL API)")
    parser.add_argument(
        "--evaluator-llm",
        type=str,
        default=DEFAULT_EVALUATOR_LLM,
        help="Evaluator LLM model name for RAGAS (EPFL API)",
    )
    args = parser.parse_args()

    run_evaluation(
        retrieval_results_path=args.retrieval_results,
        llm_model_name=args.llm,
        evaluator_llm_name=args.evaluator_llm,
    )
