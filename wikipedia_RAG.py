from openai import OpenAI
from langchain_community.vectorstores import FAISS
from caas_knowledge_base_embeddings import LocalEmbeddings
from tqdm import tqdm
import argparse
import os
from dotenv import load_dotenv
load_dotenv()

# --- Constants ---
EPFL_BASE_URL = "https://inference.rcp.epfl.ch/v1"
DEFAULT_MODEL_PATH = "/scratch/home/belghmi/models/bge-m3"
LLM_MODEL_NAME = "swiss-ai/Apertus-8B-Instruct-2509"


SYSTEM_PROMPT = """You are a precise research assistant for general knowledge questions grounded in Wikipedia.

Answer ONLY using information explicitly present in the provided context documents.
Do NOT add information from your general knowledge, even if it seems relevant.
Every claim in your answer must be traceable to a specific context document.
Always cite your sources when giving an answer.
If the context doesn't contain enough information to answer fully, say so explicitly rather than supplementing with outside knowledge.
Be concise and structured.
"""


# --- Retrieval ---
def load_vector_database(index_path: str, embedding_model: LocalEmbeddings) -> FAISS:
    print(f"Loading vector database from '{index_path}'...")
    return FAISS.load_local(
        index_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(
    vector_db: FAISS,
    query: str,
    k: int = 10,
) -> list[str]:
    retrieved_docs = vector_db.similarity_search(query=query, k=k)
    return [doc.metadata.get("source", "") + " " + doc.page_content for doc in retrieved_docs]


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
    completion = client.chat.completions.create(model=model, messages=messages, temperature=0, seed=42)
    return completion.choices[0].message.content


# --- RAGAS-compatible wrapper class ---
class RAG:
    def __init__(
        self,
        index_path: str = "/scratch/home/belghmi/indexes/hf_index",
        embedding_model_path: str = DEFAULT_MODEL_PATH,
        llm_model_name: str = LLM_MODEL_NAME,
        retrieval_k: int = 10,
    ):
        self.llm_model_name = llm_model_name
        self.retrieval_k = retrieval_k
        self.client = OpenAI(base_url=EPFL_BASE_URL, api_key=os.environ["OPENAI_API_KEY"])

        self.embedding_model = LocalEmbeddings(model_path=embedding_model_path)
        self.vector_db = load_vector_database(index_path, self.embedding_model)

    def get_most_relevant_docs(self, query: str) -> list[str]:
        return retrieve_documents(self.vector_db, query, k=self.retrieval_k)

    def batch_retrieve(self, queries: list[str]) -> list[list[str]]:
        """Embed all queries at once, then retrieve documents for each."""
        print(f"Batch embedding {len(queries)} queries...")
        embeddings = self.embedding_model.embed_documents(queries)
        all_docs = []
        for emb in tqdm(embeddings, desc="Searching index"):
            docs = self.vector_db.similarity_search_by_vector(emb, k=self.retrieval_k)
            all_docs.append([doc.metadata.get("source", "") + " " + doc.page_content for doc in docs])
        return all_docs

    def generate_answer(self, query: str, relevant_docs: list[str]) -> str:
        context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(relevant_docs)])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n---\nNow here is the question you need to answer.\nQuestion: {query}",
            },
        ]
        completion = self.client.chat.completions.create(model=self.llm_model_name, messages=messages, temperature=0, seed=42)
        return completion.choices[0].message.content


# --- Main RAG pipeline ---
def run_rag(
    query: str,
    index_path: str = "/scratch/home/belghmi/indexes/hf_index",
    embedding_model_path: str = DEFAULT_MODEL_PATH,
    llm_model_name: str = LLM_MODEL_NAME,
    retrieval_k: int = 10,
):
    rag = RAG(
        index_path=index_path,
        embedding_model_path=embedding_model_path,
        llm_model_name=llm_model_name,
        retrieval_k=retrieval_k,
    )

    relevant_docs = rag.get_most_relevant_docs(query)

    print("\nContext documents passed to LLM:")
    for i, doc in enumerate(relevant_docs):
        print(f"  [{i+1}] {doc[:200]}...")

    answer = rag.generate_answer(query, relevant_docs)
    print(f"\nAnswer:\n{answer}")
    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a RAG query against a FAISS knowledge base.")
    parser.add_argument("--query", type=str, required=True, help="The question to answer")
    parser.add_argument(
        "--index",
        type=str,
        default="/scratch/home/belghmi/indexes/hf_index",
        help="Path to the FAISS index on NAS3 scratch",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to local embedding model on NAS3 scratch",
    )
    parser.add_argument("--llm", type=str, default=LLM_MODEL_NAME, help="LLM model name (EPFL API)")
    parser.add_argument("--retrieval-k", type=int, default=10, help="Number of documents to retrieve")
    args = parser.parse_args()

    run_rag(
        query=args.query,
        index_path=args.index,
        embedding_model_path=args.embedding_model,
        llm_model_name=args.llm,
        retrieval_k=args.retrieval_k,
    )