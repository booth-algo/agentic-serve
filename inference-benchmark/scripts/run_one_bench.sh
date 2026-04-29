#!/usr/bin/env bash
# Launch vLLM server for one model, run a single benchmark profile, tear down.
# Designed to run on any GPU host with vllm installed and the inference-benchmark
# source synced to /tmp/inference-benchmark/.
#
# Usage:
#   bash run_one_bench.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND PROFILE CONC NUM_REQ OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN]
#
# Example:
#   bash run_one_bench.sh /data/models/Llama-3.1-8B-Instruct 1 Llama-3.1-8B vllm chat-singleturn 10 50 \
#       /tmp/results/a100_Llama-8B_tp1_vllm /data/kevinlau/miniconda3/bin/python 0.85 8192
set -euo pipefail

MODEL_PATH="${1:?model path}"
TP="${2:?tp size}"
SHORT="${3:?short name}"
BACKEND="${4:?backend}"
PROFILE="${5:?profile}"
CONC="${6:?concurrency}"
NUM_REQ="${7:?num requests}"
OUT_DIR="${8:?out dir}"
PY="${9:-python}"
GPU_MEM="${10:-0.85}"
MAX_LEN="${11:-8192}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/${SHORT}_tp${TP}_${BACKEND}_${PROFILE}_conc${CONC}.json"

echo "[bench] MODEL=$MODEL_PATH TP=$TP PROFILE=$PROFILE CONC=$CONC -> $OUT_FILE"

"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_LEN" \
    --trust-remote-code \
    > /tmp/vllm_${PORT}.log 2>&1 &
SERVER_PID=$!
echo "[bench] vllm PID=$SERVER_PID (port $PORT)"

trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; true' EXIT

for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[bench] server ready after ${i}×5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[bench] server died; tail log:"
        tail -30 /tmp/vllm_${PORT}.log
        exit 1
    fi
    sleep 5
done

cd /tmp/inference-benchmark
OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
    --url        "http://localhost:$PORT/v1/chat/completions" \
    --model      "$MODEL_PATH" \
    --backend    "$BACKEND" \
    --profile    "$PROFILE" \
    --concurrency "$CONC" \
    --num-requests "$NUM_REQ" \
    --warmup     2 \
    --timeout    300 \
    --api-key    "$API_KEY" \
    --output     "$OUT_FILE"

echo "[bench] done -> $OUT_FILE ($(du -h "$OUT_FILE" | cut -f1))"
