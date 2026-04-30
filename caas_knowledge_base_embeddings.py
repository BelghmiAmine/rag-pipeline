import argparse
import datasets
from tqdm import tqdm
from typing import List
from langchain_core.documents import Document as LangchainDocument
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# --- Constants ---
DEFAULT_MODEL_PATH = "/scratch/home/belghmi/models/snowflake-arctic-embed-m"
CHUNK_SIZE = 512  # tokens
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


# --- Embedding class (local model, no API calls) ---
class LocalEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        print(f"Loading embedding model from '{model_path}'...")
        self.model = SentenceTransformer(model_path)
        print("Embedding model loaded.")

    def embed_documents(self, texts: list[str], batch_size: int = 256) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return embedding.tolist()


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
    tokenizer_path: str = DEFAULT_MODEL_PATH,
) -> List[LangchainDocument]:

    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
    output_path: str = "/scratch/home/belghmi/indexes/hf_index",
    model_path: str = DEFAULT_MODEL_PATH,
    chunk_size: int = CHUNK_SIZE,
):
    print(f"Loading dataset: {dataset_name}")
    knowledge_base = load_knowledge_base(dataset_name)

    print(f"Splitting documents with chunk_size={chunk_size}")
    docs_processed = split_documents(
        knowledge_base,
        chunk_size=chunk_size,
        tokenizer_path=model_path,
    )
    print(f"Total chunks after splitting and deduplication: {len(docs_processed)}")

    print("Building vector database...")
    embedding_model = LocalEmbeddings(model_path=model_path)
    knowledge_vector_database = FAISS.from_documents(
        docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    knowledge_vector_database.save_local(output_path)
    print(f"Vector database saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector database from a HuggingFace dataset using a local embedding model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="m-ric/huggingface_doc",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/scratch/home/belghmi/indexes/hf_index",
        help="Output path for the FAISS index (should be on NAS3 scratch)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to local embedding model on NAS3 scratch",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Chunk size in tokens",
    )
    args = parser.parse_args()

    build_vector_database(
        dataset_name=args.dataset,
        output_path=args.output,
        model_path=args.model,
        chunk_size=args.chunk_size,
    )