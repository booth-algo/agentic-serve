#!/usr/bin/env bash
# Sweep across concurrency levels for one or more profiles.
# Saves a timestamped JSON per run, prints a summary table at the end.
#
# Usage:
#   ./scripts/sweep.sh [OPTIONS]
#
# Examples:
#   ./scripts/sweep.sh
#   ./scripts/sweep.sh --profiles "chat-singleturn coding-singleturn" --concurrency "1 5 10 20 40"
#   ./scripts/sweep.sh --backend trtllm --url http://localhost:8000/generate_stream
#
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

URL="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BACKEND="vllm"
PROFILES="chat-singleturn coding-singleturn"
CONCURRENCY_LEVELS="1 5 10 20 40"
NUM_REQUESTS=100
WARMUP=5
API_KEY="test"
MODE=""
MAX_CONTEXT_TOKENS=""
CONTEXT_SAFETY_MARGIN_TOKENS="${CONTEXT_SAFETY_MARGIN_TOKENS:-256}"
PREFIX_CACHING_STATE="${PREFIX_CACHING_STATE:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GPU_MEM="${GPU_MEM:-}"
TP="${TP:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)          URL="$2"; shift 2 ;;
    --model)        MODEL="$2"; shift 2 ;;
    --backend)      BACKEND="$2"; shift 2 ;;
    --profiles)     PROFILES="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY_LEVELS="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --warmup)       WARMUP="$2"; shift 2 ;;
    --api-key)      API_KEY="$2"; shift 2 ;;
    --mode)         MODE="$2"; shift 2 ;;
    --max-context-tokens) MAX_CONTEXT_TOKENS="$2"; shift 2 ;;
    --context-safety-margin-tokens) CONTEXT_SAFETY_MARGIN_TOKENS="$2"; shift 2 ;;
    --prefix-caching-state) PREFIX_CACHING_STATE="$2"; shift 2 ;;
    --chunked-prefill) CHUNKED_PREFILL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEM="$2"; shift 2 ;;
    --tensor-parallel-size) TP="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$REPO_ROOT/results/sweep_${BACKEND}_${TS}"
mkdir -p "$RESULTS_DIR"

echo "=============================="
echo " Concurrency sweep"
echo " Backend:  $BACKEND"
echo " Profiles: $PROFILES"
echo " Levels:   $CONCURRENCY_LEVELS"
echo " Requests: $NUM_REQUESTS minimum per run"
echo "           single-turn concurrency>1 uses at least 2x concurrency"
echo " Output:   $RESULTS_DIR"
echo "=============================="
echo ""

cd "$REPO_ROOT"

for PROFILE in $PROFILES; do
  for CONC in $CONCURRENCY_LEVELS; do
    OUT="$RESULTS_DIR/${PROFILE}_conc${CONC}.json"
    echo "--- $PROFILE | concurrency=$CONC ---"
    profile_mode="$MODE"
    if [[ -z "$profile_mode" && "$PROFILE" == *multiturn* ]]; then
      profile_mode="multi-turn"
    elif [[ -z "$profile_mode" ]]; then
      profile_mode="single-turn"
    fi

    run_num_requests="$NUM_REQUESTS"
    if [[ "$profile_mode" != "multi-turn" && "$CONC" -gt 1 ]]; then
      min_loaded_requests=$(( CONC * 2 ))
      [[ "$run_num_requests" -lt "$min_loaded_requests" ]] && run_num_requests="$min_loaded_requests"
    fi

    if [[ "$run_num_requests" != "$NUM_REQUESTS" ]]; then
      echo "  num_requests=$run_num_requests (raised for loaded single-turn concurrency)"
    fi

    RUNNER_ARGS=(
      --url "$URL"
      --model "$MODEL"
      --backend "$BACKEND"
      --profile "$PROFILE"
      --concurrency "$CONC"
      --num-requests "$run_num_requests"
      --warmup "$WARMUP"
      --api-key "$API_KEY"
      --prefix-caching-state "$PREFIX_CACHING_STATE"
      --chunked-prefill "$CHUNKED_PREFILL"
      --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS"
      --mode "$profile_mode"
      --output "$OUT"
    )

    if [[ -n "$MAX_CONTEXT_TOKENS" ]]; then
      RUNNER_ARGS+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
    elif [[ "$profile_mode" == "multi-turn" && -n "$MAX_MODEL_LEN" ]]; then
      RUNNER_ARGS+=(--max-context-tokens "$MAX_MODEL_LEN")
    fi
    if [[ -n "$MAX_MODEL_LEN" ]]; then
      RUNNER_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
    fi
    if [[ -n "$GPU_MEM" ]]; then
      RUNNER_ARGS+=(--gpu-memory-utilization "$GPU_MEM")
    fi
    if [[ -n "$TP" ]]; then
      RUNNER_ARGS+=(--tensor-parallel-size "$TP")
    fi

    OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
      "${RUNNER_ARGS[@]}"
    echo ""
  done
done

echo "=============================="
echo " Sweep complete. Results in:"
echo " $RESULTS_DIR"
echo ""
echo " Summary (output tok/s | p99 TTFT ms):"
echo "=============================="

# Print a quick summary table from the JSON files
"$PYTHON" - "$RESULTS_DIR" <<'PYEOF'
import json, os, sys, glob

results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(results_dir, "*.json")))

print(f"{'Profile':<20} {'Conc':>6} {'Req/s':>8} {'Out tok/s':>10} {'TTFT p99':>10} {'TPOT p99':>10} {'E2EL p99':>10}")
print("-" * 80)

for f in files:
    try:
        with open(f) as fh:
            d = json.load(fh)
        s = d["summary"]
        print(f"{s['profile']:<20} {s['concurrency']:>6} {s['request_throughput']:>8.2f} "
              f"{s['output_token_throughput']:>10.0f} {s['p99_ttft_ms']:>10.1f} "
              f"{s['p99_tpot_ms']:>10.1f} {s['p99_e2el_ms']:>10.1f}")
    except Exception as e:
        print(f"  [skip {os.path.basename(f)}: {e}]")
PYEOF
