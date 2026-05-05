#!/usr/bin/env bash
# MSE validation: run matched distributional-mse + legacy profile on same server.
# Both use the same ISL filter, same concurrency, same scope, same sessions.
# For the time-limited H100-only path, prefer run_mse_validation_fast.sh.
#
# Usage (on GPU host):
#   bash scripts/run_mse_validation.sh \
#       /data/models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm \
#       swebench 40 /tmp/results/mse \
#       /home/kevinlau/miniconda3/envs/vllm/bin/python 0.85 32768
#
# Runs BOTH swebench-multiturn-mse AND swebench-multiturn-short at the
# same concurrency, back-to-back on the same server. Saves results as:
#   ${OUT_DIR}/${SHORT}_tp${TP}_${BACKEND}/swebench-multiturn-mse_conc${C}.json
#   ${OUT_DIR}/${SHORT}_tp${TP}_${BACKEND}/swebench-multiturn-short_conc${C}.json
#
# Runtime bounds:
#   SESSIONS=40              # caps sampled multi-turn sessions; floored at C
#   MIN_SUCCESS_RATE=0.75
#   SOURCE_SESSION_IDS_FILE=ids.txt  # optional source-locked MSE validation
#   PREFIX_CACHING=on|off    # launch vLLM with APC enabled/disabled
#   CHUNKED_PREFILL=on|off   # keep chunked prefill enabled unless ablation needs off
#   PREFIX_AWARE_SYNTHETIC=on|off      # add shared APC prefix to MSE prompts only
#   SHARED_PREFIX_TOKENS=1024          # content-token target for the shared prefix
#   SHARED_PREFIX_BLOCK_SIZE=16        # align shared prefix for APC block hashing
set -euo pipefail

MODEL_PATH="${1:?model path}"
TP="${2:?tp size}"
SHORT="${3:?short name}"
BACKEND="${4:?backend}"
DATASET="${5:?dataset (swebench|terminalbench|osworld)}"
CONC="${6:?concurrency}"
OUT_DIR="${7:?output dir}"
PY="${8:-python3}"
GPU_MEM="${9:-0.85}"
MAX_LEN="${10:-32768}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"
SESSIONS="${SESSIONS:-40}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.75}"
SOURCE_SESSION_IDS_FILE="${SOURCE_SESSION_IDS_FILE:-}"
PREFIX_CACHING="${PREFIX_CACHING:-on}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-on}"
PREFIX_AWARE_SYNTHETIC="${PREFIX_AWARE_SYNTHETIC:-off}"
SHARED_PREFIX_TOKENS="${SHARED_PREFIX_TOKENS:-1024}"
SHARED_PREFIX_BLOCK_SIZE="${SHARED_PREFIX_BLOCK_SIZE:-16}"
NREQ=$(( CONC * 2 ))
[[ "$NREQ" -lt 20 ]] && NREQ=20

case "$DATASET" in
    swebench|terminalbench)
        MSE_PROFILE="${DATASET}-multiturn-mse-short"
        ;;
    *)
        MSE_PROFILE="${DATASET}-multiturn-mse"
        ;;
esac
LEGACY_PROFILE="${DATASET}-multiturn-short"

SUB_DIR="${SHORT}_tp${TP}_${BACKEND}"
mkdir -p "${OUT_DIR}/${SUB_DIR}"
SOURCE_LOCK_ARGS=()
if [[ -n "$SOURCE_SESSION_IDS_FILE" ]]; then
    SOURCE_LOCK_ARGS=(--source-session-ids-file "$SOURCE_SESSION_IDS_FILE")
fi

PREFIX_ARGS=()
case "$PREFIX_CACHING" in
    on|true|1|yes)
        PREFIX_CACHING_STATE="on"
        PREFIX_ARGS=(--enable-prefix-caching)
        ;;
    off|false|0|no)
        PREFIX_CACHING_STATE="off"
        PREFIX_ARGS=(--no-enable-prefix-caching)
        ;;
    *)
        echo "PREFIX_CACHING must be on or off, got: $PREFIX_CACHING" >&2
        exit 2
        ;;
esac

CHUNKED_ARGS=()
case "$CHUNKED_PREFILL" in
    on|true|1|yes)
        CHUNKED_PREFILL_STATE="on"
        CHUNKED_ARGS=(--enable-chunked-prefill)
        ;;
    off|false|0|no)
        CHUNKED_PREFILL_STATE="off"
        ;;
    *)
        echo "CHUNKED_PREFILL must be on or off, got: $CHUNKED_PREFILL" >&2
        exit 2
        ;;
esac

DIST_ENV=(DISTRIBUTIONAL_PREFIX_AWARE=0)
case "$PREFIX_AWARE_SYNTHETIC" in
    on|true|1|yes)
        PREFIX_AWARE_SYNTHETIC_STATE="on"
        DIST_ENV=(
            DISTRIBUTIONAL_PREFIX_AWARE=1
            DISTRIBUTIONAL_SHARED_PREFIX_TOKENS="$SHARED_PREFIX_TOKENS"
            DISTRIBUTIONAL_PREFIX_BLOCK_SIZE="$SHARED_PREFIX_BLOCK_SIZE"
        )
        ;;
    off|false|0|no)
        PREFIX_AWARE_SYNTHETIC_STATE="off"
        ;;
    *)
        echo "PREFIX_AWARE_SYNTHETIC must be on or off, got: $PREFIX_AWARE_SYNTHETIC" >&2
        exit 2
        ;;
esac

# ── vLLM launch ──────────────────────────────────────────────────
echo "=== Launching vLLM: $SHORT TP=$TP on port $PORT ==="
echo "Prefix caching: $PREFIX_CACHING_STATE"
echo "Chunked prefill: $CHUNKED_PREFILL_STATE"
echo "Synthetic shared APC prefix: $PREFIX_AWARE_SYNTHETIC_STATE"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --port "$PORT" \
    --dtype auto \
    "${PREFIX_ARGS[@]}" \
    "${CHUNKED_ARGS[@]}" \
    --no-enable-log-requests \
    &>/tmp/vllm_$$.log &
VLLM_PID=$!
cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap cleanup EXIT

# Warmup + wait
echo "Waiting for server..."
sleep 15
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready after $(( i * 2 ))s"
        break
    fi
    sleep 2
done

# ── Distributional (MSE-filtered) ──────────────────────────────
echo ""
echo "=== Distributional: $MSE_PROFILE C=$CONC ==="
env "${DIST_ENV[@]}" OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
    --url        "http://localhost:$PORT/v1/chat/completions" \
    --model      "$MODEL_PATH" \
    --backend    "$BACKEND" \
    --profile    "$MSE_PROFILE" \
    --concurrency "$CONC" \
    --num-requests "$NREQ" \
    --multi-turn-sessions "$SESSIONS" \
    "${SOURCE_LOCK_ARGS[@]}" \
    --prefix-caching-state "$PREFIX_CACHING_STATE" \
    --chunked-prefill "$CHUNKED_PREFILL_STATE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --scope      mse \
    --warmup     2 \
    --timeout    300 \
    --min-success-rate "$MIN_SUCCESS_RATE" \
    --api-key    "$API_KEY" \
    --output     "${OUT_DIR}/${SUB_DIR}/${MSE_PROFILE}_conc${CONC}.json"

# ── Legacy (real-trajectory) ────────────────────────────────────
echo ""
echo "=== Legacy: $LEGACY_PROFILE C=$CONC ==="
OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
    --url        "http://localhost:$PORT/v1/chat/completions" \
    --model      "$MODEL_PATH" \
    --backend    "$BACKEND" \
    --profile    "$LEGACY_PROFILE" \
    --concurrency "$CONC" \
    --num-requests "$NREQ" \
    --multi-turn-sessions "$SESSIONS" \
    --prefix-caching-state "$PREFIX_CACHING_STATE" \
    --chunked-prefill "$CHUNKED_PREFILL_STATE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --scope      mse \
    --warmup     2 \
    --timeout    300 \
    --min-success-rate "$MIN_SUCCESS_RATE" \
    --api-key    "$API_KEY" \
    --output     "${OUT_DIR}/${SUB_DIR}/${LEGACY_PROFILE}_conc${CONC}.json"

# ── Teardown ─────────────────────────────────────────────────────
echo ""
cleanup
trap - EXIT
echo "=== Done: ${OUT_DIR}/${SUB_DIR} ==="
