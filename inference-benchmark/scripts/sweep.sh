#!/usr/bin/env bash
# Sweep across concurrency levels for one or more profiles.
# Saves a timestamped JSON per run, prints a summary table at the end.
#
# Usage:
#   ./scripts/sweep.sh [OPTIONS]
#
# Examples:
#   ./scripts/sweep.sh
#   ./scripts/sweep.sh --profiles "prefill-heavy decode-heavy" --concurrency "1 5 10 20 40"
#   ./scripts/sweep.sh --backend trtllm --url http://localhost:8000/generate_stream
#
set -euo pipefail

PYTHON="${PYTHON:-$(which python)}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

URL="http://localhost:8000/v1/chat/completions"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
BACKEND="vllm"
PROFILES="prefill-heavy decode-heavy chat-short"
CONCURRENCY_LEVELS="1 5 10 20 40"
NUM_REQUESTS=100
WARMUP=5
API_KEY="test"

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
echo " Requests: $NUM_REQUESTS per run"
echo " Output:   $RESULTS_DIR"
echo "=============================="
echo ""

cd "$REPO_ROOT"

for PROFILE in $PROFILES; do
  for CONC in $CONCURRENCY_LEVELS; do
    OUT="$RESULTS_DIR/${PROFILE}_conc${CONC}.json"
    echo "--- $PROFILE | concurrency=$CONC ---"
    OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
      --url "$URL" \
      --model "$MODEL" \
      --backend "$BACKEND" \
      --profile "$PROFILE" \
      --concurrency "$CONC" \
      --num-requests "$NUM_REQUESTS" \
      --warmup "$WARMUP" \
      --api-key "$API_KEY" \
      --output "$OUT"
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
