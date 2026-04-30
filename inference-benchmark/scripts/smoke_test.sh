#!/usr/bin/env bash
# Quick smoke test — 5 requests, concurrency 2, profile=chat-singleturn.
# Use this to verify everything works after code changes.
#
# Usage:
#   ./scripts/smoke_test.sh [--backend trtllm] [--url http://host:port/endpoint]
#
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

URL="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BACKEND="vllm"
API_KEY="test"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM="${GPU_MEM:-}"
TP="${TP:-}"
PREFIX_CACHING_STATE="${PREFIX_CACHING_STATE:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-auto}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-}"
CONTEXT_SAFETY_MARGIN_TOKENS="${CONTEXT_SAFETY_MARGIN_TOKENS:-256}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)     URL="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    --tensor-parallel-size) TP="$2"; shift 2 ;;
    --prefix-caching-state) PREFIX_CACHING_STATE="$2"; shift 2 ;;
    --chunked-prefill) CHUNKED_PREFILL="$2"; shift 2 ;;
    --max-context-tokens) MAX_CONTEXT_TOKENS="$2"; shift 2 ;;
    --context-safety-margin-tokens) CONTEXT_SAFETY_MARGIN_TOKENS="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "Smoke test: backend=$BACKEND url=$URL"
cd "$REPO_ROOT"

RUNNER_ARGS=(
  --url "$URL"
  --model "$MODEL"
  --backend "$BACKEND"
  --profile chat-singleturn
  --concurrency 2
  --num-requests 5
  --warmup 2
  --api-key "$API_KEY"
  --prefix-caching-state "$PREFIX_CACHING_STATE"
  --chunked-prefill "$CHUNKED_PREFILL"
  --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS"
  --output results/smoke_test_latest.json
)

if [[ -n "$MAX_MODEL_LEN" ]]; then
  RUNNER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "$GPU_MEM" ]]; then
  RUNNER_ARGS+=(--gpu-memory-utilization "$GPU_MEM")
fi
if [[ -n "$TP" ]]; then
  RUNNER_ARGS+=(--tensor-parallel-size "$TP")
fi
if [[ -n "$MAX_CONTEXT_TOKENS" ]]; then
  RUNNER_ARGS+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
fi

OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
  "${RUNNER_ARGS[@]}"

echo ""
echo "Smoke test passed."
