#!/usr/bin/env bash
# Quick smoke test — 5 requests, concurrency 2, profile=output-short.
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)     URL="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

echo "Smoke test: backend=$BACKEND url=$URL"
cd "$REPO_ROOT"

OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
  --url "$URL" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --profile output-short \
  --concurrency 2 \
  --num-requests 5 \
  --warmup 2 \
  --api-key "$API_KEY" \
  --output results/smoke_test_latest.json

echo ""
echo "Smoke test passed."
