import argparse
import datasets
from tqdm import tqdm
from typing import Optional, List
from langchain_core.documents import Document as LangchainDocument
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv
load_dotenv()

# --- Constants ---
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
EPFL_BASE_URL = "https://inference.rcp.epfl.ch/v1"
CHUNK_SIZE = 512  # tokens — more appropriate than MAX_SEQ_LENGTH for retrieval
CHUNK_OVERLAP_RATIO = 0.1

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


# --- Embedding class ---
class EPFLEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, base_url: str = EPFL_BASE_URL):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding documents"):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(model=self.model_name, input=batch)
            embeddings.extend([item.embedding for item in response.data])
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding


# --- Document processing ---
def load_knowledge_base(dataset_name: str) -> List[LangchainDocument]:
    ds = datasets.load_dataset(dataset_name, split="train")
    return [
        LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in tqdm(ds, desc="Loading dataset")
    ]


def split_documents(
    knowledge_base: List[LangchainDocument],
    chunk_size: int = CHUNK_SIZE,
    tokenizer_name: str = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    
    print(f"Downloading tokenizer '{tokenizer_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print("Tokenizer ready. Splitting documents...")
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * CHUNK_OVERLAP_RATIO),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in tqdm(knowledge_base, desc="Splitting documents"):
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    print(f"Deduplication: {len(docs_processed)} → {len(docs_processed_unique)} chunks")
    return docs_processed_unique


def build_vector_database(
    dataset_name: str = "m-ric/huggingface_doc",
    output_path: str = "hugging_face_documentation",
    embedding_model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
):
    print(f"Loading dataset: {dataset_name}")
    knowledge_base = load_knowledge_base(dataset_name)

    print(f"Splitting documents with chunk_size={chunk_size}")
    docs_processed = split_documents(knowledge_base, chunk_size=chunk_size, tokenizer_name=embedding_model_name)
    print(f"Total chunks after splitting and deduplication: {len(docs_processed)}")

    print("Building vector database...")
    embedding_model = EPFLEmbeddings(model_name=embedding_model_name)
    knowledge_vector_database = FAISS.from_documents(
        docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    knowledge_vector_database.save_local(output_path)
    print(f"Vector database saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS vector database from a HuggingFace dataset.")
    parser.add_argument("--dataset", type=str, default="m-ric/huggingface_doc", help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, default="hugging_face_documentation", help="Output path for the FAISS index")
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL_NAME, help="Embedding model name")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Chunk size in tokens")
    args = parser.parse_args()

    build_vector_database(
        dataset_name=args.dataset,
        output_path=args.output,
        embedding_model_name=args.model,
        chunk_size=args.chunk_size,
    )

    