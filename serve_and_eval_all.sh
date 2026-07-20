#!/bin/bash
# serve_and_eval_all.sh — boot vLLM ONCE for a local checkpoint, then run the full
# evaluation matrix (INCLUDE + synthetic, all languages, cb + rag) against it.
#
# Usage (inside a RunAI job, --gpu 1):
#   bash serve_and_eval_all.sh <MODEL_PATH> <MODEL_TAG>
#
# Example:
#   bash serve_and_eval_all.sh /mnt/nlp/scratch/home/belghmi/sft_runs/base_sft basesft
#   bash serve_and_eval_all.sh /mnt/nlp/scratch/home/belghmi/sft_runs/cpt_sft  cptsft
#
# Outputs follow the existing naming convention:
#   include_results/{TAG}-{lang}-{cb|rag}.json
#   synthetic_results/{TAG}-{lang}-{cb|rag}-judge.json
# Existing output files are SKIPPED, so the job is safe to re-run / resume.

set -uo pipefail   # NOTE: not -e, so one failing cell doesn't abort the whole sweep

MODEL_PATH="$1"
MODEL_TAG="$2"
PORT=8000
JUDGE="Qwen/Qwen3-235B-A22B-Instruct-2507"

INCLUDE_RETR=/mnt/nlp/scratch/home/belghmi/include_retrieval
SYNTH_RETR=/mnt/nlp/scratch/home/belghmi/synthetic_retrieval_results
INCLUDE_OUT=/mnt/nlp/scratch/home/belghmi/include_results
SYNTH_OUT=/mnt/nlp/scratch/home/belghmi/synthetic_results

INCLUDE_LANGS="ar de es fr it ja nl pl pt ru tr zh"        # 12 (no en)
SYNTH_LANGS="ar de en es fr it ja nl pl pt ru tr zh"       # 13

mkdir -p "$INCLUDE_OUT" "$SYNTH_OUT"

# ---- Boot vLLM once -----------------------------------------------------------
echo "[serve] starting vLLM for: ${MODEL_PATH}  (tag=${MODEL_TAG})"
vllm serve "${MODEL_PATH}" \
  --served-model-name local-model \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 --max-model-len 4096 \
  > "${VLLM_LOG:-/tmp/vllm.log}" 2>&1 &
VLLM_PID=$!
cleanup() { echo "[serve] stopping vLLM (pid ${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "[serve] waiting for vLLM /health ..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "[serve] vLLM is up after ~$((i*5))s"; break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[serve] vLLM died during startup. Last log lines:"; tail -n 200 "${VLLM_LOG:-/tmp/vllm.log}"; exit 1
  fi
  sleep 5
done
curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 || { echo "[serve] never healthy"; tail -n 200 "${VLLM_LOG:-/tmp/vllm.log}"; exit 1; }

# ---- INCLUDE: 12 langs × {cb, rag} --------------------------------------------
for lang in $INCLUDE_LANGS; do
  for mode in cb rag; do
    flag=""; [ "$mode" = "cb" ] && flag="--closed-book"
    out="${INCLUDE_OUT}/${MODEL_TAG}-${lang}-${mode}.json"
    if [ -f "$out" ]; then echo "[skip] $out"; continue; fi
    echo "[include] ${MODEL_TAG} ${lang} ${mode}"
    python3 -u include_eval.py --inference local --llm local-model $flag \
      --retrieval-results "${INCLUDE_RETR}/${lang}.json" \
      --output "$out"
  done
done

# ---- Synthetic: 13 langs × {cb, rag}, LLM-as-judge ----------------------------
for lang in $SYNTH_LANGS; do
  for mode in cb rag; do
    flag=""; [ "$mode" = "cb" ] && flag="--closed-book"
    out="${SYNTH_OUT}/${MODEL_TAG}-${lang}-${mode}-judge.json"
    if [ -f "$out" ]; then echo "[skip] $out"; continue; fi
    echo "[synthetic] ${MODEL_TAG} ${lang} ${mode}"
    python3 -u llm_as_judge_eval.py --inference local --llm local-model $flag \
      --judge "$JUDGE" \
      --retrieval-results "${SYNTH_RETR}/${lang}.json" \
      --output "$out"
  done
done

echo "[done] full eval matrix complete for ${MODEL_TAG}"
