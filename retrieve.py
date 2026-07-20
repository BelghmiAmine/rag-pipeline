import json
import argparse
import os
import faiss
import datasets
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class LocalEmbeddings(Embeddings):
    def __init__(self, model_path: str, device: str = "cuda"):
        print(f"Loading embedding model from '{model_path}' on {device}...")
        self.model = SentenceTransformer(model_path, device=device)
        print("Embedding model loaded.")

    def embed_documents(self, texts: list[str], batch_size: int = 2048) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

DEFAULT_EMBEDDING_MODEL = "/mnt/nlp/scratch/home/belghmi/models/bge-m3"


INCLUDE_LANG_NAME = {
    "ar": "Arabic", "de": "German", "es": "Spanish", "fr": "French",
    "it": "Italian", "ja": "Japanese", "nl": "Dutch", "pl": "Polish",
    "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "zh": "Chinese",
}


def load_qa_pairs(
    dataset_type: str,
    dataset: str,
    language: str = "en",
    num_queries: str = "all",
) -> tuple[list[str], list, list[list[dict]] | None, dict | None]:
    extra: dict | None = None

    if dataset_type == "hf":
        print(f"Loading HuggingFace dataset '{dataset}'...")
        test = datasets.load_dataset(dataset, split="train")
        queries = list(test["question"])
        references = list(test["answer"])
        answer_entities = None
    elif dataset_type == "json":
        print(f"Loading local JSON dataset from '{dataset}' (language='{language}')...")
        with open(dataset) as f:
            data = json.load(f)
        if language not in data:
            raise ValueError(f"Language '{language}' not found. Available: {list(data.keys())}")
        pairs = data[language]
        queries = [p["query"] for p in pairs]
        references = [p["answer"] for p in pairs]
        answer_entities = [p.get("answer_entities") for p in pairs]
    elif dataset_type == "synthetic":
        print(f"Loading synthetic QA dataset from '{dataset}'...")
        with open(dataset, encoding="utf-8") as f:
            data = json.load(f)
        pairs = data["qa_pairs"]
        queries = [p["question"] for p in pairs]
        references = [p["answer"] for p in pairs]
        answer_entities = None
    elif dataset_type == "include":
        if language not in INCLUDE_LANG_NAME:
            raise ValueError(
                f"Language '{language}' not in INCLUDE map. Available: {list(INCLUDE_LANG_NAME)}"
            )
        lang_name = INCLUDE_LANG_NAME[language]
        print(f"Loading CohereLabs/include-base-44 (config='{lang_name}')...")
        ds = datasets.load_dataset("CohereLabs/include-base-44", lang_name, split="test")
        print(f"  -> {len(ds)} rows for {lang_name}")
        queries = [r["question"] for r in ds]
        references = [r["answer"] for r in ds]  # int 0..3
        answer_entities = None
        extra = {
            "options": [
                [r["option_a"], r["option_b"], r["option_c"], r["option_d"]] for r in ds
            ],
            "subjects": [r["subject"] for r in ds],
            "domains": [r["domain"] for r in ds],
            "countries": [r.get("country", "") for r in ds],
        }
    else:
        raise ValueError(
            f"Unknown dataset-type '{dataset_type}'. Use 'hf', 'json', 'synthetic', or 'include'."
        )

    if num_queries != "all":
        n = int(num_queries)
        queries = queries[:n]
        references = references[:n]
        if answer_entities is not None:
            answer_entities = answer_entities[:n]
        if extra is not None:
            extra = {k: v[:n] for k, v in extra.items()}

    print(f"Loaded {len(queries)} QA pairs.")
    return queries, references, answer_entities, extra


def run_retrieve(
    dataset_type: str = "json",
    dataset: str = "mkqa_pairs.json",
    language: str = "en",
    num_queries: str = "all",
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL,
    index_path: str = "",
    retrieval_k: int = 5,
    output_path: str = "retrieval_results.json",
    faiss_device: str = "gpu",
):
    queries, references, answer_entities, extra = load_qa_pairs(
        dataset_type, dataset, language, num_queries
    )

    # --- Step 1: embed queries ---
    embedding_device = "cuda" if faiss_device == "gpu" else "cpu"
    embedding_model = LocalEmbeddings(model_path=embedding_model_path, device=embedding_device)
    print(f"Embedding {len(queries)} queries...")
    embeddings = embedding_model.embed_documents(queries)

    # --- Step 2: load FAISS index and move to GPU ---
    print(f"Loading FAISS index from '{index_path}'...")
    vector_db = LangchainFAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    if faiss_device == "gpu":
        print("Moving FAISS index to GPU...")
        res = faiss.StandardGpuResources()
        vector_db.index = faiss.index_cpu_to_gpu(res, 0, vector_db.index)
    else:
        print("Using FAISS on CPU.")

    # --- Step 3: search for each query embedding ---
    print(f"Searching index for {len(embeddings)} queries (k={retrieval_k})...")
    all_docs = []
    for emb in tqdm(embeddings, desc="Searching index"):
        docs = vector_db.similarity_search_by_vector(emb, k=retrieval_k)
        all_docs.append([doc.metadata.get("source", "") + " " + doc.page_content for doc in docs])

    results = {
        "queries": queries,
        "references": references,
        "answer_entities": answer_entities,
        "retrieved_docs": all_docs,
        "extra": extra,
        "metadata": {
            "language": language,
            "dataset_type": dataset_type,
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
    parser.add_argument("--dataset-type", type=str, default="json",
                        choices=["hf", "json", "synthetic", "include"])
    parser.add_argument("--dataset", type=str, default="mkqa_pairs.json")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--num-queries", type=str, default="all", help="Integer or 'all'")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--index", type=str, required=True, help="Path to the FAISS index directory")
    parser.add_argument("--retrieval-k", type=int, default=5, help="Number of documents to retrieve per query")
    parser.add_argument("--output", type=str, required=True, help="Path to save the retrieval results JSON")
    parser.add_argument("--faiss-device", type=str, default="gpu", choices=["gpu", "cpu"],
                        help="Where to run FAISS search (default: gpu).")
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
        faiss_device=args.faiss_device,
    )
