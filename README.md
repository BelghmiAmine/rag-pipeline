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
build indexes ──▶ generate synthetic QA ──▶ retrieve ──▶ evaluate
```

1. **Build indexes** — `wikipedia_knowledge_base_embeddings.py`
   Chunks the `wikimedia/wikipedia` 20231101 snapshot per language, embeds with BGE-M3,
   and writes one FAISS index per language (resumable; persists every 200k articles).
2. **Generate the synthetic benchmark** — `generate_synthetic.py`
   Samples articles per language, generates one factoid QA per article with a generator LLM,
   filters them with an independent critic (groundedness / relevance / standaloneness),
   and keeps the top-500 per language.
3. **Retrieve** — `retrieve.py`
   For each test question, encodes with BGE-M3 and pulls the top-k chunks; writes a
   retrieval-results JSON consumed by the evaluators.
4. **Evaluate** — self-host a checkpoint with vLLM (built from `Dockerfile.eval`) and run an
   evaluator against it with `--inference local`:
   - `llm_as_judge_eval.py` — **primary** synthetic eval; generates answers (closed-book or RAG)
     and scores them 1–5 against the gold reference with an independent judge; also reports Hit@5 / Precision@5.
   - `include_eval.py` — deterministic MCQ accuracy on the INCLUDE benchmark.

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

## Prerequisites

Every heavy stage runs as a containerized job on the **EPFL RCP RunAI** cluster. You need:

- **RunAI access** to a project on the RCP cluster (`runai login`, then `runai config project <your-project>`).
- **Harbor registry** access to push images (`registry.rcp.epfl.ch/<lab>/<user>/...`).
- A **Scratch PVC** (`nlp-scratch` → `/mnt/nlp/scratch`) for indexes, datasets, and results.
- An **EPFL AI-as-a-Service** API key (`OPENAI_API_KEY`) for generation/judging, and an `HF_TOKEN`
  for dataset/model downloads. Pass them as env vars (`-e OPENAI_API_KEY=$OPENAI_API_KEY`) — never commit them.

Local, GPU-free experimentation is possible (`pip install -r requirements.txt`), but retrieval,
index builds, and self-hosted evaluation need GPUs and the cluster.

## Build and push the images

Two images: the main one (indexing / retrieval / generation) and a lean vLLM image for
self-hosting checkpoints during evaluation. Replace the LDAP build-args with your own RCP values.

```bash
# Main image
docker build --platform linux/amd64 . \
  --tag registry.rcp.epfl.ch/nlp/<user>/rag-pipeline:v1 \
  --build-arg LDAP_GROUPNAME=<group> --build-arg LDAP_GID=<gid> \
  --build-arg LDAP_USERNAME=<user> --build-arg LDAP_UID=<uid>
docker push registry.rcp.epfl.ch/nlp/<user>/rag-pipeline:v1

# Evaluation image (vLLM + evaluators)
docker build --platform linux/amd64 -f Dockerfile.eval . \
  --tag registry.rcp.epfl.ch/nlp/<user>/rag-eval-vllm:v1 \
  --build-arg LDAP_GROUPNAME=<group> --build-arg LDAP_GID=<gid> \
  --build-arg LDAP_USERNAME=<user> --build-arg LDAP_UID=<uid>
docker push registry.rcp.epfl.ch/nlp/<user>/rag-eval-vllm:v1
```

## Run the pipeline (RunAI job examples)

Below, `IMG=registry.rcp.epfl.ch/nlp/<user>/rag-pipeline:v1`, `EVAL_IMG=…/rag-eval-vllm:v1`,
and `S=/mnt/nlp/scratch/<user>`. Shown for French (`fr`); loop over the 13 languages.

**1. Build a per-language Wikipedia index** (GPU — embeds with BGE-M3):
```bash
runai submit build-index-fr --image $IMG --gpu 1 --node-pools default \
  -e PYTHONUNBUFFERED=1 -e HF_TOKEN=$HF_TOKEN --pvc nlp-scratch:/mnt/nlp/scratch \
  -- python3 -u wikipedia_knowledge_base_embeddings.py --lang fr --output $S/indexes/fr
```

**2. Generate the synthetic benchmark** (CPU — calls the AIaaS API):
```bash
runai submit gen-qa-fr --image $IMG --gpu 0 \
  -e PYTHONUNBUFFERED=1 -e OPENAI_API_KEY=$OPENAI_API_KEY -e HF_TOKEN=$HF_TOKEN \
  --pvc nlp-scratch:/mnt/nlp/scratch \
  -- python3 -u generate_synthetic.py --languages fr --n-articles 10000 --target-qa 500 \
       --output-dir $S/synthetic_qa
```

**3. Retrieve top-k context** (GPU — embeds queries, searches FAISS):
```bash
runai submit retrieve-fr --image $IMG --gpu 1 --node-pools default \
  -e PYTHONUNBUFFERED=1 --pvc nlp-scratch:/mnt/nlp/scratch \
  -- python3 -u retrieve.py --dataset-type synthetic --dataset $S/synthetic_qa/fr.json \
       --language fr --index $S/indexes/fr --retrieval-k 5 --output $S/retrieval/fr.json
```

**4. Evaluate a checkpoint** — self-host it with vLLM, then run the judge in the same job
(add `--closed-book` for the CB condition):
```bash
runai submit eval-cptsft-fr-rag --image $EVAL_IMG --gpu 1 --node-pools default \
  -e PYTHONUNBUFFERED=1 -e OPENAI_API_KEY=$OPENAI_API_KEY --pvc nlp-scratch:/mnt/nlp/scratch \
  -- bash -lc '
     vllm serve '"$S"'/sft_runs/cpt_sft --served-model-name local-model \
       --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 4096 > /tmp/vllm.log 2>&1 &
     until curl -sf localhost:8000/health; do sleep 5; done
     python3 -u llm_as_judge_eval.py --inference local --llm local-model \
       --judge Qwen/Qwen3-235B-A22B-Instruct-2507 \
       --retrieval-results '"$S"'/retrieval/fr.json --output '"$S"'/results/cptsft-fr-rag.json'
```

To evaluate a **released** model instead of a local checkpoint, skip vLLM and use the API directly:
`--inference online --llm swiss-ai/Apertus-8B-Instruct-2509`. For INCLUDE, swap
`llm_as_judge_eval.py` for `include_eval.py` (deterministic MCQ, no judge).

Monitor any job with `runai logs <job>` and list with `runai list`.
