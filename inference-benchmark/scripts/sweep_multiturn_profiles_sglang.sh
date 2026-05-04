#!/usr/bin/env bash
# SGLang multi-turn variant of sweep_multiturn_profiles.sh.
#
# Same positional-arg shape as the vLLM launcher so bench_orchestrator.sh can
# dispatch backend=sglang multi-turn cells once sweep.yaml enables them.
#
# Usage:
#   bash sweep_multiturn_profiles_sglang.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [WARMUP]
set -uo pipefail

export SGLANG_DISABLE_CUDNN_CHECK=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN

SGLANG_ENV_DIR="$(dirname "$(dirname "${6:-python}")")"
if [[ -x "$SGLANG_ENV_DIR/bin/nvcc" ]]; then
    export CUDA_HOME="$SGLANG_ENV_DIR"
    export PATH="$SGLANG_ENV_DIR/bin:$PATH"
    export LIBRARY_PATH="$SGLANG_ENV_DIR/lib:$SGLANG_ENV_DIR/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$SGLANG_ENV_DIR/lib:$SGLANG_ENV_DIR/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

MODEL_PATH="${1:?model path}"
TP="${2:?tp}"
SHORT="${3:?short}"
BACKEND="${4:?backend}"
OUT_DIR="${5:?out dir}"
PY="${6:-python}"
GPU_MEM="${7:-0.85}"
MAX_LEN="${8:-32768}"
CONCS="${9:-5 20 40 80 160}"
PROFILES="${10:-chat-multiturn swebench-multiturn terminalbench-multiturn osworld-multiturn}"
WARMUP="${11:-3}"
CONTEXT_SAFETY_MARGIN_TOKENS="${CONTEXT_SAFETY_MARGIN_TOKENS:-256}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"
DASHBOARD_SCOPE="${DASHBOARD_SCOPE:-fixed}"

result_scope_matches_expected() {
    local file="$1"
    "$PY" - "$file" "$DASHBOARD_SCOPE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as f:
        raw = json.load(f)
except Exception:
    raise SystemExit(1)

scope = (raw.get("config") or {}).get("dashboard_scope")
raise SystemExit(0 if scope == sys.argv[2] else 1)
PY
}

mkdir -p "$OUT_DIR"
echo "[mt-sweep-sglang] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[mt-sweep-sglang] concurrencies: $CONCS"
echo "[mt-sweep-sglang] profiles: $PROFILES"
echo "[mt-sweep-sglang] dashboard scope: $DASHBOARD_SCOPE"

"$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --tp "$TP" \
    --mem-fraction-static "$GPU_MEM" \
    --context-length "$MAX_LEN" \
    --trust-remote-code \
    > /tmp/vllm_${PORT}.log 2>&1 &
SERVER_PID=$!
echo "[mt-sweep-sglang] sglang PID=$SERVER_PID (port $PORT)"

trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; true' EXIT

for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[mt-sweep-sglang] server ready after ${i}x5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[mt-sweep-sglang] server died; tail log:"
        tail -30 /tmp/vllm_${PORT}.log
        exit 1
    fi
    sleep 5
done

cd /tmp/inference-benchmark

SGLANG_VERSION=$("$PY" -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
echo "backend=sglang version=$SGLANG_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[mt-sweep-sglang] captured engine version: sglang $SGLANG_VERSION"

for PROFILE in $PROFILES; do
    for CONC in $CONCS; do
        OUT_FILE="$OUT_DIR/${PROFILE}_conc${CONC}.json"
        if [ -f "$OUT_FILE" ] && [ -s "$OUT_FILE" ]; then
            if result_scope_matches_expected "$OUT_FILE"; then
                echo "[skip] $OUT_FILE exists with dashboard_scope=$DASHBOARD_SCOPE"
                continue
            fi
            echo "[rerun] $OUT_FILE exists with stale/missing dashboard_scope; overwriting for $DASHBOARD_SCOPE"
        fi
        echo ""
        echo "=== profile=$PROFILE conc=$CONC (sglang multi-turn) ==="
        OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
            --url        "http://localhost:$PORT/v1/chat/completions" \
            --model      "$MODEL_PATH" \
            --backend    "$BACKEND" \
            --profile    "$PROFILE" \
            --concurrency "$CONC" \
            --mode       multi-turn \
            --max-context-tokens "$MAX_LEN" \
            --context-safety-margin-tokens "$CONTEXT_SAFETY_MARGIN_TOKENS" \
            --prefix-caching-state on \
            --chunked-prefill unknown \
            --max-model-len "$MAX_LEN" \
            --gpu-memory-utilization "$GPU_MEM" \
            --tensor-parallel-size "$TP" \
            --warmup     "$WARMUP" \
            --timeout    300 \
            --api-key    "$API_KEY" \
            --scope      "$DASHBOARD_SCOPE" \
            --output     "$OUT_FILE" || echo "[warn] mt-sglang bench failed for $PROFILE conc=$CONC (continuing)"
    done
done

echo "[mt-sweep-sglang] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -15
