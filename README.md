# RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that runs entirely on the EPFL RCP AI-as-a-Service platform — no local GPU required.

## What it does

Given a question, the pipeline:
1. Searches a local FAISS vector database for the most relevant document chunks
2. Reranks the retrieved chunks using a cross-encoder reranker
3. Passes the best chunks as context to an LLM to generate a grounded answer

## Files

**`RAG.py`** — Runs the RAG pipeline for a single query. It loads the saved FAISS index, retrieves the top-k most similar chunks, optionally reranks them, and sends the context + question to an LLM to generate an answer.

**`gesda_knowledge_base_embeddings.py`** — Builds the GESDA knowledge base. It loads the .txt versions of the gesda radars pdfs, splits them into chunks using a token-aware text splitter, embeds each chunk using the EPFL embedding API, and saves the resulting FAISS index locally.

**`knowledge_base_embeddings.py`** — Same as gesda_knowledge_base_embeddings.py but for datasets from huggingface.

**`ragas_eval.ipynb`** — RAG performance evaluation using ragas.

## How it works

The pipeline has two steps that must be run in order:

1. **Build the index** — `gesda_knowledge_base_embeddings.py` downloads the dataset, chunks it, embeds each chunk via the EPFL API, and saves a FAISS index to disk. This only needs to run once.

2. **Query the index** — `RAG.py` loads the saved FAISS index and answers questions against it.

The FAISS index is stored locally in the folder specified by `--output` (default: `hugging_face_documentation/`).

## Models (all served via EPFL RCP)

- **Embedding**: `Qwen/Qwen3-Embedding-8B`
- **Reranker**: `BAAI/bge-reranker-v2-m3`
- **LLM**: `swiss-ai/Apertus-70B-Instruct-2509`

## Setup

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd rag-pipeline
```

**2. Create the conda environment**
```bash
conda env create -f environment.yml
conda activate embedding
```

**3. Set up your API key** : 

Create a `.env` file in the project root :

```bash
OPENAI_API_KEY=your_epfl_api_key_here
KMP_DUPLICATE_LIB_OK=TRUE
```
## Usage

**Build the knowledge base (only needs to run once)**
```bash
python gesda_knowledge_base_embeddings.py
```

With custom options:
```bash
python gesda_knowledge_base_embeddings.py --output gesda_index --chunk-size 512
```

**Run a query**
```bash
python RAG.py --query "How to create a pipeline object?" --index gesda_index
```

With custom options:
```bash
python RAG.py --query "What is quantum computing and when will it be widely available?" --retrieval-k 50 --rerank-top-n 5 --no-rerank
```

## Notes

- The FAISS index is saved locally and does not need to be rebuilt on every run
- Reranking is enabled by default
