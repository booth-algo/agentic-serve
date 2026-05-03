#!/usr/bin/env bash
# Full MSE validation sweep: all 3 datasets × C=40,80 on one GPU host.
# Runs distributional-mse and legacy-short back-to-back on the same vLLM.
#
# Usage (on GPU host):
#   bash run_mse_sweep.sh /data/models/Llama-3.1-8B-Instruct 1 H100 \
#       /data/kevinlau/miniconda3/bin/python /tmp/results/mse_sweep
#
# DATASETS: swebench terminalbench osworld
# CONCS: 40 80
# Each cell: distributional-mse then legacy-short on same server.
set -euo pipefail

MODEL_PATH="${1:?model path}"
TP="${2:?tensor-parallel size}"
GPU_NAME="${3:?GPU name (H100, A100, RTX3090)}"
PY="${4:-python3}"
OUT_DIR="${5:?output dir}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"
MAX_LEN=32768
GPU_MEM=0.85
NREQ_MULT=2

DATASETS=(swebench terminalbench osworld)
CONCS=(40 80)

mkdir -p "$OUT_DIR"

launch_vllm() {
    echo "=== Launching vLLM: $GPU_NAME TP=$TP on port $PORT ==="
    "$PY" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$TP" \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM" \
        --port "$PORT" \
        --dtype auto \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --disable-log-requests \
        &>/tmp/vllm_mse.log &
    VLLM_PID=$!

    for i in $(seq 1 30); do
        sleep 2
        if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
            echo "Server ready after $((i * 2))s"
            return 0
        fi
    done
    echo "FAIL: vLLM did not start" && return 1
}

kill_vllm() {
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 3
}

run_one() {
    local profile="$1" concurrency="$2" tag="$3"
    local nreq=$(( concurrency * NREQ_MULT ))
    [[ "$nreq" -lt 20 ]] && nreq=20

    echo "  [$tag] C=$concurrency ..."
    "$PY" -m src.benchmark.runner \
        --url "http://localhost:$PORT/v1/chat/completions" \
        --model "$MODEL_PATH" \
        --backend vllm \
        --profile "$profile" \
        --concurrency "$concurrency" \
        --num-requests "$nreq" \
        --prefix-caching-state on \
        --chunked-prefill on \
        --max-model-len "$MAX_LEN" \
        --gpu-memory-utilization "$GPU_MEM" \
        --tensor-parallel-size "$TP" \
        --scope fixed \
        --warmup 2 \
        --timeout 300 \
        --api-key "$API_KEY" \
        --output "${OUT_DIR}/${tag}_conc${concurrency}.json" \
        2>&1 | grep -E "Results saved|Duration|failed|Requests ok"
}

# ── Main loop ─────────────────────────────────────────────────────
launch_vllm || exit 1
trap kill_vllm EXIT

for ds in "${DATASETS[@]}"; do
    echo ""
    echo "=== Dataset: $ds ==="
    mse_profile="${ds}-multiturn-mse"
    legacy_profile="${ds}-multiturn-short"

    for conc in "${CONCS[@]}"; do
        echo "  C=$conc"

        # Distributional (MSE-filtered, synthetic)
        run_one "$mse_profile" "$conc" "${GPU_NAME}_${ds}_mse"

        # Legacy (real traces, ground truth)
        run_one "$legacy_profile" "$conc" "${GPU_NAME}_${ds}_legacy"
    done
done

echo ""
echo "=== Done: $OUT_DIR ==="
