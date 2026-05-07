"""
wikipedia_knowledge_base_embeddings.py

Builds a FAISS vector index from a Wikipedia language dump (wikimedia/wikipedia).

Key features:
  - Streaming mode: articles are never fully loaded into RAM
  - Batched incremental indexing: builds mini FAISS indexes per batch and merges
  - Rich metadata: stores title, url, article_id, language, snapshot per chunk

Usage example:
  python wikipedia_knowledge_base_embeddings.py \
    --lang en \
    --snapshot 20231101 \
    --output /mnt/nlp/scratch/home/belghmi/indexes/wikipedia_20231101_en_snowflake-arctic-embed-m \
    --model /mnt/nlp/scratch/home/belghmi/models/snowflake-arctic-embed-m
"""

import argparse
import gc
import os
import time
from typing import List

import datasets
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document as LangchainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "/mnt/nlp/scratch/home/belghmi/models/snowflake-arctic-embed-m"
DEFAULT_SNAPSHOT = "20231101"
CHUNK_SIZE = 512
CHUNK_OVERLAP_RATIO = 0.1
BATCH_SIZE = 50_000

WIKIPEDIA_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class LocalEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        print(f"Loading embedding model from '{model_path}'...")
        self.model = SentenceTransformer(model_path)
        print("Embedding model loaded.")

    def embed_documents(self, texts: List[str], batch_size: int = 2048) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------

def make_text_splitter(tokenizer_path: str, chunk_size: int) -> RecursiveCharacterTextSplitter:
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer ready.")
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * CHUNK_OVERLAP_RATIO),
        add_start_index=True,
        strip_whitespace=True,
        separators=WIKIPEDIA_SEPARATORS,
    )


def split_articles_to_chunks(
    articles: List[dict],
    text_splitter: RecursiveCharacterTextSplitter,
    lang: str,
    snapshot: str,
    seen_texts: set,
) -> List[LangchainDocument]:
    """
    Convert a batch of raw Wikipedia article dicts into deduplicated LangchainDocument chunks.
    seen_texts persists across batches to deduplicate globally.
    """
    chunks = []
    for article in articles:
        doc = LangchainDocument(
            page_content=article["text"],
            metadata={
                "title":      article["title"],
                "url":        article["url"],
                "article_id": article["id"],
                "language":   lang,
                "snapshot":   snapshot,
            },
        )
        for chunk in text_splitter.split_documents([doc]):
            if chunk.page_content not in seen_texts:
                seen_texts.add(chunk.page_content)
                chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Main indexing function
# ---------------------------------------------------------------------------

def build_wikipedia_index(
    lang: str,
    snapshot: str,
    output_path: str,
    model_path: str,
    chunk_size: int,
    batch_size: int,
) -> None:
    start_time = time.time()
    dataset_config = f"{snapshot}.{lang}"
    print("=" * 70)
    print(f"Building Wikipedia index")
    print(f"  Language : {lang}")
    print(f"  Snapshot : {snapshot}")
    print(f"  Dataset  : wikimedia/wikipedia / {dataset_config}")
    print(f"  Output   : {output_path}")
    print(f"  Model    : {model_path}")
    print(f"  Batch    : {batch_size:,} articles")
    print("=" * 70)

    os.makedirs(output_path, exist_ok=True)

    embedding_model = LocalEmbeddings(model_path=model_path)
    text_splitter = make_text_splitter(model_path, chunk_size)

    print(f"\nFetching dataset size for wikimedia/wikipedia / {dataset_config} ...")
    ds_info = datasets.load_dataset_builder("wikimedia/wikipedia", dataset_config)
    total_articles = ds_info.info.splits["train"].num_examples
    print(f"Total articles in dataset: {total_articles:,}")

    print(f"Streaming dataset: wikimedia/wikipedia / {dataset_config} ...")
    ds = datasets.load_dataset(
        "wikimedia/wikipedia",
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    main_index = None
    seen_texts = set()
    batch = []
    total_chunks = 0
    articles_processed = 0

    pbar = tqdm(desc=f"[{lang}] Articles processed", unit="articles", total=total_articles)

    for article in ds:
        batch.append(article)

        if len(batch) < batch_size:
            continue

        chunks = split_articles_to_chunks(batch, text_splitter, lang, snapshot, seen_texts)
        total_chunks += len(chunks)

        if chunks:
            mini_index = FAISS.from_documents(
                chunks,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            if main_index is None:
                main_index = mini_index
            else:
                main_index.merge_from(mini_index)
            del mini_index
            gc.collect()

        articles_processed += len(batch)
        pbar.update(len(batch))
        batch = []

    # Process remaining articles in last partial batch
    if batch:
        chunks = split_articles_to_chunks(batch, text_splitter, lang, snapshot, seen_texts)
        total_chunks += len(chunks)
        if chunks:
            mini_index = FAISS.from_documents(
                chunks,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            if main_index is None:
                main_index = mini_index
            else:
                main_index.merge_from(mini_index)
        articles_processed += len(batch)
        pbar.update(len(batch))

    pbar.close()

    print(f"\nSaving final index to '{output_path}' ...")
    main_index.save_local(output_path)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Done!")
    print(f"  Total articles : {articles_processed:,}")
    print(f"  Total chunks   : {total_chunks:,}")
    print(f"  Elapsed time   : {elapsed/3600:.2f} hours")
    print(f"  Index saved to : {output_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector index from wikimedia/wikipedia for a given language."
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Wikipedia language code (e.g. en, fr, de, ar, zh)",
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=DEFAULT_SNAPSHOT,
        help=f"Wikipedia snapshot date (default: {DEFAULT_SNAPSHOT})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for the FAISS index. "
            "Defaults to /mnt/nlp/scratch/home/belghmi/indexes/wikipedia_{snapshot}_{lang}_{model_name}"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to local embedding model on Scratch",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size in tokens (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of articles per processing batch (default: {BATCH_SIZE})",
    )

    args = parser.parse_args()

    if args.output is None:
        model_name = os.path.basename(args.model.rstrip("/"))
        args.output = (
            f"/mnt/nlp/scratch/home/belghmi/indexes/"
            f"wikipedia_{args.snapshot}_{args.lang}_{model_name}"
        )

    build_wikipedia_index(
        lang=args.lang,
        snapshot=args.snapshot,
        output_path=args.output,
        model_path=args.model,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )