import argparse
import re
import os
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
DEFAULT_MODEL_PATH = "/mnt/nlp/scratch/home/belghmi/models/bge-m3"
CHUNK_SIZE = 1024  # tokens
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
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
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
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# --- GESDA PDF loading ---
def load_gesda_txt(
    txt_path: str,
    pdf_name: str = None,
) -> List[LangchainDocument]:
    """
    Parse a .txt file (gesda reports)
    Each page delimited by <<<PAGE_N>>> / <<<END_PAGE_N>>> becomes
    one LangchainDocument with metadata: source, page, pdf_name.
    Empty/image-only pages are skipped.
    """
    if pdf_name is None:
        pdf_name = os.path.splitext(os.path.basename(txt_path))[0]

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse all page blocks
    page_blocks = re.findall(
        r"<<<PAGE_(\d+)>>>(.*?)<<<END_PAGE_\1>>>",
        content,
        re.DOTALL,
    )

    docs = []
    skipped = 0
    for page_num_str, page_text in tqdm(page_blocks, desc="Loading pages"):
        page_num = int(page_num_str)
        text = page_text.strip()

        # Skip blank/image-only pages
        if not text:
            skipped += 1
            continue

        docs.append(
            LangchainDocument(
                page_content=text,
                metadata={
                    "source": pdf_name,
                    "page": page_num,
                    "pdf_name": pdf_name,
                },
            )
        )

    print(f"Loaded {len(docs)} pages ({skipped} empty pages skipped) from '{txt_path}'")
    return docs


def load_multiple_gesda_txts(txt_paths: List[str]) -> List[LangchainDocument]:
    """Load and merge multiple GESDA .txt files (e.g. one per yearly report)."""
    all_docs = []
    for path in txt_paths:
        all_docs.extend(load_gesda_txt(path))
    print(f"Total pages across all files: {len(all_docs)}")
    return all_docs


# --- Chunking (unchanged logic, adapted for GESDA metadata) ---
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
        chunks = text_splitter.split_documents([doc])
        # Propagate page metadata to all chunks (LangChain preserves metadata
        # automatically, but we make sure page is always present)
        for chunk in chunks:
            chunk.metadata.setdefault("page", doc.metadata.get("page"))
            chunk.metadata.setdefault("pdf_name", doc.metadata.get("pdf_name"))
        docs_processed.extend(chunks)

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    print(f"Deduplication: {len(docs_processed)} → {len(docs_processed_unique)} chunks")
    return docs_processed_unique


# --- Main pipeline ---
def build_vector_database(
    txt_paths: List[str],
    output_path: str = "gesda_documentation",
    model_path: str = DEFAULT_MODEL_PATH,
    chunk_size: int = CHUNK_SIZE,
):
    print(f"Loading {len(txt_paths)} GESDA txt file(s)...")
    knowledge_base = load_multiple_gesda_txts(txt_paths)

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
        description="Build a FAISS vector database from GESDA PDF .txt files."
    )
    parser.add_argument(
    "--txt",
    type=str,
    nargs="+",
    default=[
        "/mnt/nlp/scratch/home/belghmi/data/gesda_radars/gesda_radar_2021.txt",
        "/mnt/nlp/scratch/home/belghmi/data/gesda_radars/gesda_radar_2022.txt",
        "/mnt/nlp/scratch/home/belghmi/data/gesda_radars/gesda_radar_2023.txt",
        "/mnt/nlp/scratch/home/belghmi/data/gesda_radars/gesda_radar_2024.txt",
        "/mnt/nlp/scratch/home/belghmi/data/gesda_radars/gesda_radar_2026.txt",
    ],
    help="Path(s) to the extracted .txt file(s). Example: --txt gesda_2026.txt gesda_2025.txt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/mnt/nlp/scratch/home/belghmi/indexes/gesda_bge-m3",
        help="Output path for the FAISS index",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to local embedding model",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Chunk size in tokens",
    )
    args = parser.parse_args()

    build_vector_database(
        txt_paths=args.txt,
        output_path=args.output,
        model_path=args.model,
        chunk_size=args.chunk_size,
    )