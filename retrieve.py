import json
import argparse
import os
import faiss
import datasets
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS as LangchainFAISS
from caas_knowledge_base_embeddings import LocalEmbeddings

DEFAULT_EMBEDDING_MODEL = "/mnt/nlp/scratch/home/belghmi/models/bge-m3"


def load_qa_pairs(
    dataset_type: str,
    dataset: str,
    language: str = "en",
    num_queries: str = "all",
) -> tuple[list[str], list[str]]:
    if dataset_type == "hf":
        print(f"Loading HuggingFace dataset '{dataset}'...")
        test = datasets.load_dataset(dataset, split="train")
        queries = list(test["question"])
        references = list(test["answer"])
    elif dataset_type == "json":
        print(f"Loading local JSON dataset from '{dataset}' (language='{language}')...")
        with open(dataset) as f:
            data = json.load(f)
        if language not in data:
            raise ValueError(f"Language '{language}' not found. Available: {list(data.keys())}")
        pairs = data[language]
        queries = [p["query"] for p in pairs]
        references = [p["answer"] for p in pairs]
    else:
        raise ValueError(f"Unknown dataset-type '{dataset_type}'. Use 'hf' or 'json'.")

    if num_queries != "all":
        n = int(num_queries)
        queries = queries[:n]
        references = references[:n]

    print(f"Loaded {len(queries)} QA pairs.")
    return queries, references


def run_retrieve(
    dataset_type: str = "json",
    dataset: str = "mkqa_pairs.json",
    language: str = "en",
    num_queries: str = "all",
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL,
    index_path: str = "",
    retrieval_k: int = 5,
    output_path: str = "retrieval_results.json",
):
    queries, references = load_qa_pairs(dataset_type, dataset, language, num_queries)

    # --- Step 1: embed queries on GPU ---
    embedding_model = LocalEmbeddings(model_path=embedding_model_path)
    print(f"Embedding {len(queries)} queries...")
    embeddings = embedding_model.embed_documents(queries)

    # --- Step 2: load FAISS index and move to GPU ---
    print(f"Loading FAISS index from '{index_path}'...")
    vector_db = LangchainFAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    print("Moving FAISS index to GPU...")
    res = faiss.StandardGpuResources()
    vector_db.index = faiss.index_cpu_to_gpu(res, 0, vector_db.index)

    # --- Step 3: search for each query embedding ---
    print(f"Searching index for {len(embeddings)} queries (k={retrieval_k})...")
    all_docs = []
    for emb in tqdm(embeddings, desc="Searching index"):
        docs = vector_db.similarity_search_by_vector(emb, k=retrieval_k)
        all_docs.append([doc.metadata.get("source", "") + " " + doc.page_content for doc in docs])

    results = {
        "queries": queries,
        "references": references,
        "retrieved_docs": all_docs,
        "metadata": {
            "language": language,
            "num_queries": len(queries),
            "embedding_model_path": embedding_model_path,
            "index_path": index_path,
            "retrieval_k": retrieval_k,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    print(f"Saved retrieval results for {len(queries)} queries to '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU job: embed queries and search FAISS index, save retrieved docs to disk."
    )
    parser.add_argument("--dataset-type", type=str, default="json", choices=["hf", "json"])
    parser.add_argument("--dataset", type=str, default="mkqa_pairs.json")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--num-queries", type=str, default="all", help="Integer or 'all'")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--index", type=str, required=True, help="Path to the FAISS index directory")
    parser.add_argument("--retrieval-k", type=int, default=5, help="Number of documents to retrieve per query")
    parser.add_argument("--output", type=str, required=True, help="Path to save the retrieval results JSON")
    args = parser.parse_args()

    run_retrieve(
        dataset_type=args.dataset_type,
        dataset=args.dataset,
        language=args.language,
        num_queries=args.num_queries,
        embedding_model_path=args.embedding_model,
        index_path=args.index,
        retrieval_k=args.retrieval_k,
        output_path=args.output,
    )
