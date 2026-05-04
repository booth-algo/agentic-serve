#!/usr/bin/env bash
# MSE validation: run matched distributional-mse + legacy profile on same server.
# Both use the same ISL filter, same concurrency, same scope, same sessions.
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
NREQ=$(( CONC * 2 ))
[[ "$NREQ" -lt 20 ]] && NREQ=20

MSE_PROFILE="${DATASET}-multiturn-mse"
LEGACY_PROFILE="${DATASET}-multiturn-short"

SUB_DIR="${SHORT}_tp${TP}_${BACKEND}"
mkdir -p "${OUT_DIR}/${SUB_DIR}"

# ── vLLM launch ──────────────────────────────────────────────────
echo "=== Launching vLLM: $SHORT TP=$TP on port $PORT ==="
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --port "$PORT" \
    --dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests \
    &>/tmp/vllm_$$.log &
VLLM_PID=$!

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
OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
    --url        "http://localhost:$PORT/v1/chat/completions" \
    --model      "$MODEL_PATH" \
    --backend    "$BACKEND" \
    --profile    "$MSE_PROFILE" \
    --concurrency "$CONC" \
    --num-requests "$NREQ" \
    --prefix-caching-state on \
    --chunked-prefill on \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --scope      mse \
    --warmup     2 \
    --timeout    300 \
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
    --prefix-caching-state on \
    --chunked-prefill on \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --scope      mse \
    --warmup     2 \
    --timeout    300 \
    --api-key    "$API_KEY" \
    --output     "${OUT_DIR}/${SUB_DIR}/${LEGACY_PROFILE}_conc${CONC}.json"

# ── Teardown ─────────────────────────────────────────────────────
echo ""
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "=== Done: ${OUT_DIR}/${SUB_DIR} ==="
