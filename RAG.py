import requests
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from knowledge_base_embeddings import EPFLEmbeddings
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

# --- Constants ---
EPFL_BASE_URL = "https://inference.rcp.epfl.ch/v1"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME = "swiss-ai/Apertus-8B-Instruct-2509"

SYSTEM_PROMPT = """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer."""


# --- Retrieval ---
def load_vector_database(index_path: str, embedding_model: EPFLEmbeddings) -> FAISS:
    print(f"Loading vector database from '{index_path}'...")
    return FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )


def retrieve_documents(
    vector_db: FAISS,
    query: str,
    k: int = 10,
) -> list[str]:
    print(f"Retrieving top {k} documents for query: '{query}'")
    retrieved_docs = vector_db.similarity_search(query=query, k=k)
    return [doc.metadata.get("source", "") + " " + doc.page_content for doc in retrieved_docs]


# --- Reranking ---
def rerank_documents(
    query: str,
    documents: list[str],
    api_key: str,
    base_url: str = EPFL_BASE_URL,
    model: str = RERANKER_MODEL_NAME,
    top_n: int = 5,
) -> list[str]:
    print(f"Reranking {len(documents)} documents, keeping top {top_n}...")
    url = f"{base_url}/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "top_n": top_n,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()

    ranked = sorted(result["results"], key=lambda x: x["relevance_score"], reverse=True)
    return [documents[doc["index"]] for doc in ranked]


# --- Generation ---
def generate_answer(
    query: str,
    context_docs: list[str],
    api_key: str,
    base_url: str = EPFL_BASE_URL,
    model: str = LLM_MODEL_NAME,
) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\nQuestion: {query}",
        },
    ]

    completion = client.chat.completions.create(model=model, messages=messages)

    print("-" * 50)
    print(f"completion_tokens: {completion.usage.completion_tokens}")
    print(f"prompt_tokens:     {completion.usage.prompt_tokens}")
    print(f"total_tokens:      {completion.usage.total_tokens}")

    return completion.choices[0].message.content


# --- Main RAG pipeline ---
def run_rag(
    query: str,
    index_path: str = "hugging_face_documentation",
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    llm_model_name: str = LLM_MODEL_NAME,
    reranker_model_name: str = RERANKER_MODEL_NAME,
    retrieval_k: int = 10,
    rerank: bool = True,
    rerank_top_n: int = 5,
):
    api_key = os.environ["OPENAI_API_KEY"]

    embedding_model = EPFLEmbeddings(model_name=embedding_model_name)
    vector_db = load_vector_database(index_path, embedding_model)

    documents = retrieve_documents(vector_db, query, k=retrieval_k)

    if rerank:
        documents = rerank_documents(
            query=query,
            documents=documents,
            api_key=api_key,
            model=reranker_model_name,
            top_n=rerank_top_n,
        )

    print("\nContext documents passed to LLM:")
    for i, doc in enumerate(documents):
        print(f"  [{i+1}] {doc[:100]}...")

    answer = generate_answer(
        query=query,
        context_docs=documents,
        api_key=api_key,
        model=llm_model_name,
    )

    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG query against a FAISS knowledge base.")
    parser.add_argument("--query", type=str, required=True, help="The question to answer")
    parser.add_argument("--index", type=str, default="hugging_face_documentation", help="Path to the FAISS index")
    parser.add_argument("--llm", type=str, default=LLM_MODEL_NAME, help="LLM model name")
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--reranker", type=str, default=RERANKER_MODEL_NAME, help="Reranker model name")
    parser.add_argument("--retrieval-k", type=int, default=10, help="Number of documents to retrieve before reranking")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    parser.add_argument("--rerank-top-n", type=int, default=5, help="Number of documents to keep after reranking")
    args = parser.parse_args()

    run_rag(
        query=args.query,
        index_path=args.index,
        llm_model_name=args.llm,
        embedding_model_name=args.embedding_model,
        reranker_model_name=args.reranker,
        retrieval_k=args.retrieval_k,
        rerank=not args.no_rerank,
        rerank_top_n=args.rerank_top_n,
    )