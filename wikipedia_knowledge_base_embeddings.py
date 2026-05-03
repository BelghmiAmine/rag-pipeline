"""
wikipedia_knowledge_base_embeddings.py

Builds a FAISS vector index from a Wikipedia language dump (wikimedia/wikipedia).


Key features vs caas_knowledge_base_embeddings.py:
  - Streaming mode: articles are never fully loaded into RAM
  - Batched incremental indexing: builds mini FAISS indexes per batch and merges
  - Single checkpoint: saves progress every CHECKPOINT_EVERY articles,
    deletes the previous checkpoint to avoid disk bloat
  - Resume support: detects existing checkpoint and skips already-processed articles
  - Rich metadata: stores title, url, article_id, language, snapshot per chunk

Usage example:
  python wikipedia_knowledge_base_embeddings.py \
    --lang en \
    --snapshot 20231101 \
    --output /mnt/nlp/scratch/home/belghmi/indexes/wikipedia_20231101_en_snowflake-arctic-embed-m \
    --model /mnt/nlp/scratch/home/belghmi/models/snowflake-arctic-embed-m

To run all 20 languages in parallel, submit one RunAI job per language.
"""

import argparse
import gc
import json
import os
import shutil
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
CHUNK_SIZE = 512          # tokens
CHUNK_OVERLAP_RATIO = 0.1
BATCH_SIZE = 50_000       # articles per batch (keeps GPU saturated on A100)
CHECKPOINT_EVERY = 200_000  # save checkpoint every N articles

# Wikipedia plain-text separators (no markdown, but keep paragraph breaks)
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
) -> List[LangchainDocument]:
    """
    Convert a batch of raw Wikipedia article dicts into deduplicated LangchainDocument chunks.
    Stores rich metadata per chunk: title, url, article_id, language, snapshot.
    """
    chunks = []
    seen_texts = set()

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
# Checkpoint helpers
# ---------------------------------------------------------------------------

CHECKPOINT_META_FILE = "checkpoint_meta.json"


def checkpoint_path(output_path: str) -> str:
    return output_path + "_checkpoint"


def save_checkpoint(index: FAISS, output_path: str, articles_processed: int) -> None:
    ckpt = checkpoint_path(output_path)
    os.makedirs(ckpt, exist_ok=True)
    index.save_local(ckpt)
    with open(os.path.join(ckpt, CHECKPOINT_META_FILE), "w") as f:
        json.dump({"articles_processed": articles_processed}, f)
    print(f"  → Checkpoint saved at {ckpt} ({articles_processed:,} articles processed)")


def load_checkpoint(output_path: str, embedding_model: LocalEmbeddings):
    """
    Returns (index, articles_processed) if a checkpoint exists, else (None, 0).
    """
    ckpt = checkpoint_path(output_path)
    meta_file = os.path.join(ckpt, CHECKPOINT_META_FILE)
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
        articles_processed = meta["articles_processed"]
        print(f"Checkpoint found: resuming from article {articles_processed:,}")
        index = FAISS.load_local(ckpt, embedding_model, allow_dangerous_deserialization=True)
        return index, articles_processed
    return None, 0


def delete_checkpoint(output_path: str) -> None:
    ckpt = checkpoint_path(output_path)
    if os.path.exists(ckpt):
        shutil.rmtree(ckpt)
        print(f"  → Old checkpoint deleted: {ckpt}")


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
    checkpoint_every: int,
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
    print(f"  Checkpoint every : {checkpoint_every:,} articles")
    print("=" * 70)

    os.makedirs(output_path, exist_ok=True)

    # --- Load embedding model (once, reused throughout) ---
    embedding_model = LocalEmbeddings(model_path=model_path)

    # --- Check for existing checkpoint ---
    main_index, articles_processed = load_checkpoint(output_path, embedding_model)
    if main_index is None:
        print("No checkpoint found. Starting from scratch.")

    # --- Prepare text splitter ---
    text_splitter = make_text_splitter(model_path, chunk_size)

    # --- Stream dataset ---
    print(f"\nStreaming dataset: wikimedia/wikipedia / {dataset_config} ...")
    ds = datasets.load_dataset(
        "wikimedia/wikipedia",
        dataset_config,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Skip already-processed articles if resuming
    if articles_processed > 0:
        print(f"Skipping first {articles_processed:,} articles (already indexed)...")
        ds = ds.skip(articles_processed)

    # --- Batch processing loop ---
    batch = []
    total_chunks = 0
    last_checkpoint_count = articles_processed  # track when to save next checkpoint

    pbar = tqdm(desc=f"[{lang}] Articles processed", unit="articles", initial=articles_processed)

    for article in ds:
        batch.append(article)

        if len(batch) < batch_size:
            continue

        # --- Process full batch ---
        chunks = split_articles_to_chunks(batch, text_splitter, lang, snapshot)
        total_chunks += len(chunks)

        if chunks:
            # Build mini FAISS index from this batch
            mini_index = FAISS.from_documents(
                chunks,
                embedding_model,
                distance_strategy=DistanceStrategy.COSINE,
            )
            # Merge into main index
            if main_index is None:
                main_index = mini_index
            else:
                main_index.merge_from(mini_index)
            del mini_index
            gc.collect()

        articles_processed += len(batch)
        pbar.update(len(batch))
        batch = []

        # --- Checkpoint logic: save every checkpoint_every articles ---
        if articles_processed - last_checkpoint_count >= checkpoint_every:
            print(f"\n[Checkpoint] {articles_processed:,} articles | {total_chunks:,} chunks total")
            delete_checkpoint(output_path)   # delete previous checkpoint first
            save_checkpoint(main_index, output_path, articles_processed)
            last_checkpoint_count = articles_processed

    # --- Process remaining articles in last partial batch ---
    if batch:
        chunks = split_articles_to_chunks(batch, text_splitter, lang, snapshot)
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

    # --- Save final index ---
    print(f"\nSaving final index to '{output_path}' ...")
    main_index.save_local(output_path)

    # --- Clean up checkpoint now that final index is saved ---
    delete_checkpoint(output_path)

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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help=f"Save checkpoint every N articles (default: {CHECKPOINT_EVERY})",
    )

    args = parser.parse_args()

    # Auto-generate output path if not provided
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
        checkpoint_every=args.checkpoint_every,
    )