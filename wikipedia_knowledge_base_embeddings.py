"""
wikipedia_knowledge_base_embeddings.py

Builds a FAISS vector index from a Wikipedia language dump (wikimedia/wikipedia).

Key features:
  - Deterministic dataset iteration (non-streaming HF Arrow dataset)
  - Batched incremental indexing
  - Robust checkpoint/resume
  - Exact resume semantics using stable dataset ordering
  - Rich metadata per chunk

Usage example:
  python wikipedia_knowledge_base_embeddings.py \
    --lang en \
    --snapshot 20231101 \
    --output /mnt/nlp/scratch/home/belghmi/indexes/wikipedia_20231101_en_bge-m3 \
    --model /mnt/nlp/scratch/home/belghmi/models/bge-m3
"""

import argparse
import gc
import json
import os
import shutil
import time
from typing import List, Optional, Tuple

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

DEFAULT_MODEL_PATH = "/mnt/nlp/scratch/home/belghmi/models/bge-m3"
DEFAULT_SNAPSHOT = "20231101"

CHUNK_SIZE = 512
CHUNK_OVERLAP_RATIO = 0.1

BATCH_SIZE = 50_000
CHECKPOINT_EVERY = 200_000

WIKIPEDIA_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]

CHECKPOINT_META_FILENAME = "meta.json"


# ---------------------------------------------------------------------------
# Embedding model wrapper
# ---------------------------------------------------------------------------

class LocalEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        print(f"Loading embedding model from '{model_path}'...")
        self.model = SentenceTransformer(model_path)
        print("Embedding model loaded.")

    def embed_documents(
        self,
        texts: List[str],
        batch_size: int = 2048,
    ) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# ---------------------------------------------------------------------------
# Document splitting
# ---------------------------------------------------------------------------

def make_text_splitter(
    tokenizer_path: str,
    chunk_size: int,
) -> RecursiveCharacterTextSplitter:
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
    Convert raw Wikipedia articles into deduplicated LangchainDocument chunks.

    NOTE:
      Cross-checkpoint duplicate chunk generation is possible but rare.
    """

    chunks = []

    for article in articles:
        text = article.get("text", "")

        if not text or not text.strip():
            continue

        doc = LangchainDocument(
            page_content=text,
            metadata={
                "title": article["title"],
                "url": article["url"],
                "article_id": article["id"],
                "language": lang,
                "snapshot": snapshot,
            },
        )

        for chunk in text_splitter.split_documents([doc]):
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_dir(output_path: str, articles_processed: int) -> str:
    return os.path.join(
        output_path,
        f"checkpoint_{articles_processed:010d}",
    )


def save_checkpoint(
    index: FAISS,
    output_path: str,
    articles_processed: int,
    total_chunks: int,
) -> None:
    """
    Atomically save checkpoint.

    Strategy:
      1. Write new checkpoint fully
      2. Only after success, delete older checkpoints
    """

    ckpt_dir = checkpoint_dir(output_path, articles_processed)

    os.makedirs(ckpt_dir, exist_ok=True)

    # Save FAISS index
    index.save_local(ckpt_dir)

    # Save metadata
    with open(
        os.path.join(ckpt_dir, CHECKPOINT_META_FILENAME),
        "w",
    ) as f:
        json.dump(
            {
                "articles_processed": articles_processed,
                "total_chunks": total_chunks,
            },
            f,
        )

    print(
        f"\n  → Checkpoint saved: "
        f"{ckpt_dir} ({articles_processed:,} articles)"
    )

    # Remove older checkpoints
    for name in os.listdir(output_path):
        if not name.startswith("checkpoint_"):
            continue

        old_path = os.path.join(output_path, name)

        if old_path != ckpt_dir:
            shutil.rmtree(old_path)
            print(f"  → Deleted old checkpoint: {old_path}")


def find_latest_checkpoint(
    output_path: str,
    embedding_model: LocalEmbeddings,
) -> Tuple[Optional[FAISS], int, int]:
    """
    Returns:
      (index, articles_processed, total_chunks)

    If no checkpoint exists:
      (None, 0, 0)
    """

    if not os.path.exists(output_path):
        return None, 0, 0

    checkpoints = []

    for name in os.listdir(output_path):
        if not name.startswith("checkpoint_"):
            continue

        ckpt_path = os.path.join(output_path, name)

        meta_path = os.path.join(
            ckpt_path,
            CHECKPOINT_META_FILENAME,
        )

        faiss_path = os.path.join(ckpt_path, "index.faiss")
        pkl_path = os.path.join(ckpt_path, "index.pkl")

        if (
            os.path.exists(meta_path)
            and os.path.exists(faiss_path)
            and os.path.exists(pkl_path)
        ):
            with open(meta_path) as f:
                meta = json.load(f)

            checkpoints.append(
                (
                    meta["articles_processed"],
                    meta["total_chunks"],
                    ckpt_path,
                )
            )

    if not checkpoints:
        return None, 0, 0

    checkpoints.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    best_articles, best_chunks, best_path = checkpoints[0]

    print(
        f"Found {len(checkpoints)} checkpoint(s). "
        f"Loading latest: {best_path}"
    )

    print(
        f"  → Resuming from "
        f"{best_articles:,} articles, "
        f"{best_chunks:,} chunks"
    )

    index = FAISS.load_local(
        best_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )

    return index, best_articles, best_chunks


def cleanup_checkpoints(output_path: str) -> None:
    """
    Remove checkpoint directories after final successful save.
    """

    for name in os.listdir(output_path):
        if name.startswith("checkpoint_"):
            ckpt_path = os.path.join(output_path, name)
            shutil.rmtree(ckpt_path)
            print(f"  → Removed checkpoint: {ckpt_path}")


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
    print("Building Wikipedia index")
    print(f"  Language         : {lang}")
    print(f"  Snapshot         : {snapshot}")
    print(f"  Dataset          : wikimedia/wikipedia / {dataset_config}")
    print(f"  Output           : {output_path}")
    print(f"  Model            : {model_path}")
    print(f"  Batch size       : {batch_size:,} articles")
    print(f"  Checkpoint every : {checkpoint_every:,} articles")
    print("=" * 70)

    os.makedirs(output_path, exist_ok=True)

    embedding_model = LocalEmbeddings(model_path=model_path)

    text_splitter = make_text_splitter(
        model_path,
        chunk_size,
    )

    # -----------------------------------------------------------------------
    # Load deterministic Arrow dataset (NON-STREAMING)
    # -----------------------------------------------------------------------

    print(
        f"\nLoading dataset "
        f"wikimedia/wikipedia / {dataset_config} ..."
    )

    ds = datasets.load_dataset(
        "wikimedia/wikipedia",
        dataset_config,
        split="train",
        streaming=False,
    )

    total_articles = len(ds)

    print(f"Total articles in dataset: {total_articles:,}")

    # -----------------------------------------------------------------------
    # Resume checkpoint
    # -----------------------------------------------------------------------

    main_index, articles_processed, total_chunks = (
        find_latest_checkpoint(
            output_path,
            embedding_model,
        )
    )

    if main_index is None:
        print("No checkpoint found. Starting from scratch.")

    if articles_processed > total_articles:
        raise RuntimeError(
            f"Checkpoint corruption detected:\n"
            f"articles_processed={articles_processed:,}\n"
            f"dataset_size={total_articles:,}"
        )

    # -----------------------------------------------------------------------
    # Processing
    # -----------------------------------------------------------------------


    batch = []

    last_checkpoint_at = articles_processed

    pbar = tqdm(
        total=total_articles,
        initial=articles_processed,
        desc=f"[{lang}] Articles processed",
        unit="articles",
    )

    # Deterministic indexed iteration
    for idx in range(articles_processed, total_articles):

        article = ds[idx]

        batch.append(article)

        if len(batch) < batch_size:
            continue

        # -------------------------------------------------------------------
        # Split + embed batch
        # -------------------------------------------------------------------

        chunks = split_articles_to_chunks(
            batch,
            text_splitter,
            lang,
            snapshot,
        )

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

        # -------------------------------------------------------------------
        # Save checkpoint
        # -------------------------------------------------------------------

        if (
            articles_processed - last_checkpoint_at
            >= checkpoint_every
        ):
            save_checkpoint(
                main_index,
                output_path,
                articles_processed,
                total_chunks,
            )

            last_checkpoint_at = articles_processed

    # -----------------------------------------------------------------------
    # Final partial batch
    # -----------------------------------------------------------------------

    if batch:

        chunks = split_articles_to_chunks(
            batch,
            text_splitter,
            lang,
            snapshot,
        )

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

    pbar.close()

    # -----------------------------------------------------------------------
    # Final consistency checks
    # -----------------------------------------------------------------------

    if articles_processed != total_articles:
        print(
            "\nWARNING:\n"
            f"Processed articles ({articles_processed:,}) "
            f"!= dataset size ({total_articles:,})"
        )

    # -----------------------------------------------------------------------
    # Save final index
    # -----------------------------------------------------------------------

    print(f"\nSaving final index to '{output_path}' ...")

    main_index.save_local(output_path)

    # -----------------------------------------------------------------------
    # Cleanup checkpoints
    # -----------------------------------------------------------------------

    print("Cleaning up checkpoints...")

    cleanup_checkpoints(output_path)

    elapsed = time.time() - start_time

    print("=" * 70)
    print("Done!")
    print(f"  Total articles : {articles_processed:,}")
    print(f"  Total chunks   : {total_chunks:,}")
    print(f"  Elapsed time   : {elapsed / 3600:.2f} hours")
    print(f"  Index saved to : {output_path}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Build a FAISS vector index from "
            "wikimedia/wikipedia."
        )
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Wikipedia language code (en, fr, de, ar, zh, ja, ...)",
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
            "Output path for the FAISS index.\n"
            "Defaults to:\n"
            "/mnt/nlp/scratch/home/belghmi/indexes/"
            "wikipedia_{snapshot}_{lang}_{model_name}"
        ),
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
        help=f"Chunk size in tokens (default: {CHUNK_SIZE})",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Articles per processing batch (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help=f"Checkpoint every N articles (default: {CHECKPOINT_EVERY})",
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
        checkpoint_every=args.checkpoint_every,
    )