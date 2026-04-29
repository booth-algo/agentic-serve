#!/usr/bin/env bash
# Run a single benchmark profile.
#
# Usage:
#   ./scripts/bench.sh [OPTIONS]
#
# Examples:
#   ./scripts/bench.sh
#   ./scripts/bench.sh --profile decode-heavy --concurrency 20
#   ./scripts/bench.sh --backend trtllm --url http://localhost:8000/generate_stream
#
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Defaults (override via flags)
URL="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BACKEND="vllm"
PROFILE="chat-singleturn"
CONCURRENCY=10
NUM_REQUESTS=100
WARMUP=5
ARRIVAL="steady"
TARGET_RATE=10.0
API_KEY="test"
OUTPUT=""
IGNORE_EOS=""

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)           URL="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --backend)       BACKEND="$2"; shift 2 ;;
    --profile)       PROFILE="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --num-requests)  NUM_REQUESTS="$2"; shift 2 ;;
    --warmup)        WARMUP="$2"; shift 2 ;;
    --arrival)       ARRIVAL="$2"; shift 2 ;;
    --target-rate)   TARGET_RATE="$2"; shift 2 ;;
    --api-key)       API_KEY="$2"; shift 2 ;;
    --output)        OUTPUT="$2"; shift 2 ;;
    --ignore-eos)    IGNORE_EOS="--ignore-eos"; shift ;;
    -h|--help)       usage ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  TS=$(date +%Y%m%d_%H%M%S)
  OUTPUT="results/${BACKEND}_${PROFILE}_conc${CONCURRENCY}_${TS}.json"
fi

echo "Backend:     $BACKEND"
echo "URL:         $URL"
echo "Model:       $MODEL"
echo "Profile:     $PROFILE"
echo "Concurrency: $CONCURRENCY"
echo "Requests:    $NUM_REQUESTS"
echo "Output:      $OUTPUT"
echo ""

cd "$REPO_ROOT"
OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
  --url "$URL" \
  --model "$MODEL" \
  --backend "$BACKEND" \
  --profile "$PROFILE" \
  --concurrency "$CONCURRENCY" \
  --num-requests "$NUM_REQUESTS" \
  --warmup "$WARMUP" \
  --arrival "$ARRIVAL" \
  --target-rate "$TARGET_RATE" \
  --api-key "$API_KEY" \
  --output "$OUTPUT" \
  $IGNORE_EOS
