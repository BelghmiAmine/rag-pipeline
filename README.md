# Multilingual Wikipedia CPT-for-RAG — Evaluation Pipeline

Retrieval and evaluation code for the MSc thesis
**_"Does Continual Pretraining Improve Retrieval-Augmented Generation on the Same Knowledge Distribution?"_** (EPFL NLP, 2026).

It builds per-language Wikipedia FAISS indexes, generates a synthetic Wikipedia-grounded
QA benchmark, and evaluates a **matched pair** of Apertus-8B models
(`base → SFT` vs `base → CPT → SFT`) in closed-book and retrieval-augmented settings,
scored by an independent LLM judge.

> **Sibling repositories**
> - Continual-pretraining (CPT) training code: **[apertus-cpt](https://github.com/BelghmiAmine/apertus-cpt)**
> - GESDA Radar Explorer (secondary contribution): **[gesda-app](https://github.com/BelghmiAmine/gesda-app)**

## Pipeline overview

```
build indexes ──▶ generate synthetic QA ──▶ retrieve ──▶ evaluate ──▶ analyze
```

1. **Build indexes** — `wikipedia_knowledge_base_embeddings.py`
   Chunks the `wikimedia/wikipedia` 20231101 snapshot per language, embeds with BGE-M3,
   and writes one FAISS index per language (resumable; persists every 200k articles).
2. **Generate the synthetic benchmark**
   - `generate_synthetic_qa.py` — v1: one factoid QA per sampled article, critic-filtered, top-500/language.
   - `generate_synthetic_qa_v2.py` — v2: harder (multi-hop / reasoning) questions, richer source articles.
3. **Retrieve** — `retrieve.py`
   For each test question, encodes with BGE-M3 and pulls the top-k chunks; writes a
   retrieval-results JSON consumed by the evaluators.
4. **Evaluate**
   - `llm_as_judge_eval.py` — **primary** synthetic eval; generates answers (closed-book or RAG)
     and scores them 1–5 against the gold reference with an independent judge; also reports Hit@5 / Precision@5.
   - `include_eval.py` — deterministic MCQ accuracy on the INCLUDE benchmark.
   - `serve_and_eval.sh` / `serve_and_eval_all.sh` — boot a local vLLM server for a checkpoint
     and run the eval matrix against it (self-contained RunAI jobs).
5. **Analyze**
   - `compare_synthetic.py` — aggregates per-language results into the final table.
   - `analyze_v1_significance.py` — paired t-test, random-effects meta-analysis (DerSimonian–Laird), and per-question pooled test.

## Supporting modules

| File | Role |
|---|---|
| `llm_client.py` | OpenAI-client factory: routes generation to the EPFL API (online) or a local vLLM server; the judge always uses the EPFL API. |
| `wikipedia_RAG.py` | Single-query RAG helper and shared constants (`EPFL_BASE_URL`, system prompt, local embeddings). |
| `Dockerfile` | Main image (FAISS + PyTorch + sentence-transformers + langchain + OpenAI client) for indexing/retrieval/generation jobs. |
| `Dockerfile.eval` | Lean image built on `vllm/vllm-openai` for self-hosting checkpoints and running the evaluators. |
| `requirements.txt` | Python dependencies. |

## Models (served via EPFL RCP AI-as-a-Service)

- **Retriever / embeddings:** `BAAI/bge-m3`
- **Independent judge:** `Qwen/Qwen3-235B-A22B-Instruct-2507`
- **Generators evaluated:** self-hosted Apertus-8B `base-SFT` / `cpt-SFT` (via vLLM), with the
  released `swiss-ai/Apertus-8B-Instruct-2509` and `-70B-` as scale references.

## Setup

```bash
pip install -r requirements.txt      # or build the Docker image
```

Create a `.env` in the project root (never commit it):

```bash
OPENAI_API_KEY=your_epfl_aiaas_key   # EPFL RCP AI-as-a-Service
```

The heavy stages (index builds, generation, evaluation) are designed to run as containerized
jobs on the EPFL RCP **RunAI** cluster, reading/writing the lab Scratch volume
(`/mnt/nlp/scratch`). See the `serve_and_eval*.sh` scripts for the self-hosting + eval pattern.
