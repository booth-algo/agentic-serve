#!/usr/bin/env bash
# Launch one vLLM server, sweep ALL profiles × concurrencies against it, teardown.
# Matches the H100 baseline sweep structure from run_all_benchmarks.sh.
#
# Usage:
#   bash sweep_all_profiles.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [NREQ_PER_CONC]
#
# Defaults match the H100 canonical sweep (run_all_benchmarks.sh CONC_SWEEP +
# standard production profiles).
set -euo pipefail

# Include CUDA graph memory in vLLM's pre-flight memory profiler.
# Without this, vLLM sizes the KV cache greedily and OOMs during cudagraph
# capture on tight configs (e.g. 70B/72B at TP=4 on 40GB A100). Slightly
# reduces KV cache headroom in exchange for guaranteed startup; will be
# vLLM's default in v0.19+.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1

MODEL_PATH="${1:?model path}"
TP="${2:?tp}"
SHORT="${3:?short}"
BACKEND="${4:?backend}"
OUT_DIR="${5:?out dir}"
PY="${6:-python}"
GPU_MEM="${7:-0.85}"
MAX_LEN="${8:-8192}"
CONCS="${9:-1 10 20 40 80 120 160 200 256 320 500}"
PROFILES="${10:-chat-short chat-medium chat-long coding-agent prefill-heavy decode-heavy random-1k}"
NREQ="${11:-100}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"

mkdir -p "$OUT_DIR"
echo "[sweep] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[sweep] concurrencies: $CONCS"
echo "[sweep] profiles: $PROFILES"

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
echo "[sweep] vllm PID=$SERVER_PID (port $PORT)"

trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; true' EXIT

for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[sweep] server ready after ${i}×5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[sweep] server died; tail log:"
        tail -30 /tmp/vllm_${PORT}.log
        exit 1
    fi
    sleep 5
done

cd /tmp/inference-benchmark

# Capture engine version alongside results so the dashboard can attribute
# each sweep to a specific vllm build. Written once per sweep; applies to
# every result file in the output dir.
VLLM_VERSION=$("$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
echo "backend=vllm version=$VLLM_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[sweep] captured engine version: vllm $VLLM_VERSION"

for PROFILE in $PROFILES; do
    for CONC in $CONCS; do
        OUT_FILE="$OUT_DIR/${SHORT}_tp${TP}_${BACKEND}_${PROFILE}_conc${CONC}.json"
        if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
            echo "[skip] $OUT_FILE exists"
            continue
        fi
        echo ""
        echo "=== profile=$PROFILE conc=$CONC ==="
        local_nreq="$NREQ"
        [[ "$CONC" -eq 1 ]] && local_nreq=30
        OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
            --url        "http://localhost:$PORT/v1/chat/completions" \
            --model      "$MODEL_PATH" \
            --backend    "$BACKEND" \
            --profile    "$PROFILE" \
            --concurrency "$CONC" \
            --num-requests "$local_nreq" \
            --warmup     2 \
            --timeout    300 \
            --api-key    "$API_KEY" \
            --output     "$OUT_FILE" || echo "[warn] bench failed for $PROFILE conc=$CONC (continuing)"
    done
done

echo "[sweep] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -15
