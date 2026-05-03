import os
import json
import datasets
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness
from dotenv import load_dotenv
load_dotenv()

# Import your RAG class
from caas_RAG import RAG

def run_evaluation(
    index_path: str = "/mnt/nlp/scratch/home/belghmi/indexes/hf_index",
    embedding_model_path: str = "/mnt/nlp/scratch/home/belghmi/models/snowflake-arctic-embed-m",
    ):
    # Load test dataset
    print("Loading test dataset...")
    test = datasets.load_dataset('m-ric/huggingface_doc_qa_eval', split="train")
    sample_queries = list(test['question'])
    expected_responses = list(test['answer'])
    print(f"Loaded {len(sample_queries)} test queries")

    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAG(
        index_path=index_path,
        embedding_model_path=embedding_model_path,
    )

    # Collect evaluation data
    print("Running RAG pipeline on test queries...")
    dataset = []
    for i, (query, reference) in enumerate(zip(sample_queries, expected_responses)):
        print(f"Query {i+1}/{len(sample_queries)}: {query[:60]}...")
        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        dataset.append({
            "user_input": query,
            "retrieved_contexts": relevant_docs,
            "response": response,
            "reference": reference,
        })

    # Load into RAGAS
    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # Configure evaluator LLM
    print("Configuring evaluator LLM...")
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="swiss-ai/Apertus-8B-Instruct-2509",
            base_url="https://inference.rcp.epfl.ch/v1",
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )

    # Run evaluation
    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness()],
        llm=evaluator_llm,
    )
    print(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="/mnt/nlp/scratch/home/belghmi/indexes/hf_index")
    parser.add_argument("--embedding-model", type=str, default="/mnt/nlp/scratch/home/belghmi/models/snowflake-arctic-embed-m")
    args = parser.parse_args()

    run_evaluation(
        index_path=args.index,
        embedding_model_path=args.embedding_model,
    )