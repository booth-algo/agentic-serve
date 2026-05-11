#!/usr/bin/env bash
# Fast H100-only MSE validation.
#
# Runs one short captured-token distributional profile against one H100 vLLM
# server. This intentionally does not run the legacy REAL profile; the REAL
# baseline should come from the archived H100 trace-replay result used to build
# the captured-token distribution.
#
# Usage:
#   bash scripts/run_mse_validation_fast.sh \
#       /data/models/Llama-3.1-8B-Instruct 1 \
#       /data/kevinlau/miniconda3/envs/vllm/bin/python
#
# Useful overrides:
#   PROFILE=swebench-multiturn-mse-short      # or terminalbench-multiturn-mse-short
#   CONC=5 SESSIONS=40 OUT_DIR=results/archived/mse_validation_fast
#   PORT=8089 GPU_MEM=0.75 MAX_LEN=32768 CUDA_VISIBLE_DEVICES=0
#   SOURCE_SESSION_IDS_FILE=ids.txt            # optional source-locked MSE ablation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

MODEL_PATH="${1:-${MODEL_PATH:-/data/models/Llama-3.1-8B-Instruct}}"
TP="${2:-${TP:-1}}"
PY="${3:-${PY:-python3}}"

PROFILE="${PROFILE:-swebench-multiturn-mse-short}"
CONC="${CONC:-5}"
SESSIONS="${SESSIONS:-40}"
OUT_DIR="${OUT_DIR:-results/archived/mse_validation_fast}"
PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"
GPU_MEM="${GPU_MEM:-0.75}"
MAX_LEN="${MAX_LEN:-32768}"
WARMUP="${WARMUP:-2}"
TIMEOUT="${TIMEOUT:-300}"
MIN_SUCCESS_RATE="${MIN_SUCCESS_RATE:-0.75}"
REQUIRE_H100="${REQUIRE_H100:-1}"
SOURCE_SESSION_IDS_FILE="${SOURCE_SESSION_IDS_FILE:-}"

case "$PROFILE" in
    swebench-multiturn-mse-short|terminalbench-multiturn-mse-short)
        ;;
    *)
        echo "FAIL: fast MSE only supports short MSE profiles; got PROFILE=$PROFILE" >&2
        exit 2
        ;;
esac

if (( CONC > 10 )); then
    echo "FAIL: refusing CONC=$CONC; fast validation cap is 10." >&2
    exit 2
fi

if (( SESSIONS > 60 )); then
    echo "FAIL: refusing SESSIONS=$SESSIONS; fast validation cap is 60." >&2
    exit 2
fi

if [[ "$REQUIRE_H100" == "1" ]]; then
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true)"
    if [[ "$GPU_NAMES" != *H100* ]]; then
        echo "FAIL: this fast validation script is scoped to H100 hosts." >&2
        echo "Detected GPUs: ${GPU_NAMES:-none}" >&2
        echo "Set REQUIRE_H100=0 to override deliberately." >&2
        exit 2
    fi
fi

mkdir -p "$OUT_DIR"
SAFE_PROFILE="${PROFILE//-/_}"
OUT="${OUT_DIR}/h100_${SAFE_PROFILE}_fast_conc${CONC}_sessions${SESSIONS}.json"
LOG="/tmp/vllm_fast_mse_${PORT}.log"
STARTED_SERVER=0
VLLM_PID=""
SOURCE_LOCK_ARGS=()
if [[ -n "$SOURCE_SESSION_IDS_FILE" ]]; then
    SOURCE_LOCK_ARGS=(--source-session-ids-file "$SOURCE_SESSION_IDS_FILE")
fi

cleanup() {
    if [[ "$STARTED_SERVER" == "1" && -n "$VLLM_PID" ]]; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "=== Using existing vLLM server on port $PORT ==="
else
    echo "=== Launching H100 vLLM server on port $PORT ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        "$PY" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_PATH" \
            --tensor-parallel-size "$TP" \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM" \
            --port "$PORT" \
            --dtype auto \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --no-enable-log-requests \
            >"$LOG" 2>&1 &
    VLLM_PID=$!
    STARTED_SERVER=1

    for i in $(seq 1 60); do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "Server ready after $((i * 2))s"
            break
        fi
        sleep 2
    done

    if ! curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        tail -80 "$LOG" || true
        echo "FAIL: vLLM did not become healthy." >&2
        exit 1
    fi
fi

echo "=== Fast MSE validation: $PROFILE C=$CONC sessions=$SESSIONS ==="
OPENAI_API_KEY="$API_KEY" DASHBOARD_SCOPE=archived RESULT_SCOPE=archived/mse "$PY" -m src.benchmark.runner \
    --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --model "$MODEL_PATH" \
    --backend vllm \
    --profile "$PROFILE" \
    --concurrency "$CONC" \
    --multi-turn-sessions "$SESSIONS" \
    "${SOURCE_LOCK_ARGS[@]}" \
    --prefix-caching-state on \
    --chunked-prefill on \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    --tensor-parallel-size "$TP" \
    --scope archived \
    --warmup "$WARMUP" \
    --timeout "$TIMEOUT" \
    --min-success-rate "$MIN_SUCCESS_RATE" \
    --api-key "$API_KEY" \
    --output "$OUT"

echo "=== Done: $OUT ==="
