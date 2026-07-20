#!/bin/bash
# serve_and_eval.sh — boot a local vLLM server for a checkpoint, then run an eval against it.
#
# Usage (inside a RunAI job, --gpu 1):
#   bash serve_and_eval.sh <MODEL_PATH> <eval command...>
#
# Example:
#   bash serve_and_eval.sh /mnt/nlp/scratch/home/belghmi/sft_runs/cpt_sft \
#     python3 -u include_eval.py --inference local --llm local-model \
#       --retrieval-results /mnt/nlp/scratch/home/belghmi/include_retrieval/ar.json \
#       --output /mnt/nlp/scratch/home/belghmi/include_results/cptsft-ar-rag.json
#
# The eval command MUST pass --inference local and --llm local-model (the served name).

set -euo pipefail

MODEL_PATH="$1"
shift   # the rest of "$@" is the eval command

PORT=8000
SERVED_NAME="local-model"

echo "[serve] starting vLLM for: ${MODEL_PATH}"
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len 4096 \
  > "${VLLM_LOG:-"${VLLM_LOG:-/tmp/vllm.log}"}" 2>&1 &
VLLM_PID=$!

# Kill vLLM whenever this script exits (success, error, or signal).
cleanup() { echo "[serve] stopping vLLM (pid ${VLLM_PID})"; kill "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the server to become healthy (model load can take a few minutes for 8B).
echo "[serve] waiting for vLLM /health ..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "[serve] vLLM is up after ~$((i*5))s"
    break
  fi
  # If vLLM died during startup, surface its log and bail.
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[serve] vLLM process died during startup. Last log lines:"
    tail -n 300 "${VLLM_LOG:-/tmp/vllm.log}"
    exit 1
  fi
  sleep 5
done

if ! curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "[serve] vLLM never became healthy. Last log lines:"
  tail -n 300 "${VLLM_LOG:-/tmp/vllm.log}"
  exit 1
fi

echo "[eval] running: $*"
"$@"
EVAL_RC=$?
echo "[eval] finished with rc=${EVAL_RC}"
exit ${EVAL_RC}
