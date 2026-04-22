#!/usr/bin/env bash
# sglang variant of sweep_all_profiles.sh.
#
# Launches `python -m sglang.launch_server` (OpenAI-compat on :8089),
# sweeps profiles x concurrencies, teardown. Same positional-arg shape
# as the vllm script so bench_orchestrator.sh can swap launchers by
# backend without reshaping the CMD line.
#
# Usage (matches sweep_all_profiles.sh):
#   bash sweep_all_profiles_sglang.sh \
#       MODEL_PATH TP SHORT_NAME BACKEND OUT_DIR \
#       [PY] [GPU_MEM] [MAX_LEN] [CONC_LIST] [PROFILE_LIST] [NREQ_PER_CONC]
#
# BACKEND is expected to be "sglang" - the 4th arg is preserved for CMD
# symmetry with the vllm script; the output filenames still use it.
set -uo pipefail

# sglang 0.5.9 has a startup check that refuses to run on torch 2.9.1 +
# cudnn < 9.15 due to a nn.Conv3d bug (pytorch#168167). LLM inference
# doesn't touch Conv3d, so the check is a false positive for us.
export SGLANG_DISABLE_CUDNN_CHECK=1

# NCCL init workarounds — sglang 0.5.9 hits "unhandled system error" on
# 3090 tp>1 otherwise. Disabling direct P2P and shared-memory fast paths
# falls back to CUDA IPC which works universally (slightly slower but
# fine for benchmarks where the primary metric is forward-pass latency,
# not NCCL bandwidth).
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# Surface the real NCCL error (not just "unhandled system error") on crash.
export NCCL_DEBUG=WARN

# sglang's CUDA graph runner JIT-compiles flashinfer kernels with nvcc
# then links with libcudart. Point CUDA_HOME at the conda-installed nvcc
# (cuda-nvcc=12.8 from nvidia channel) and make sure the linker finds
# libcudart.so (from nvidia-cuda-runtime-cu12 pip wheel or the env lib).
SGLANG_ENV_DIR="$(dirname "$(dirname "${6:-python}")")"
if [[ -x "$SGLANG_ENV_DIR/bin/nvcc" ]]; then
    export CUDA_HOME="$SGLANG_ENV_DIR"
    export PATH="$SGLANG_ENV_DIR/bin:$PATH"
    # Compile-time lib search (ld -lcudart) and runtime dynamic linker.
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
MAX_LEN="${8:-8192}"
CONCS="${9:-1 10 20 40 80 120 160 200 256 320 500}"
PROFILES="${10:-chat-short chat-medium chat-long coding-agent prefill-heavy decode-heavy random-1k}"
NREQ="${11:-100}"

PORT="${PORT:-8089}"
API_KEY="${API_KEY:-test}"

mkdir -p "$OUT_DIR"
echo "[sweep-sglang] MODEL=$MODEL_PATH TP=$TP OUT=$OUT_DIR"
echo "[sweep-sglang] concurrencies: $CONCS"
echo "[sweep-sglang] profiles: $PROFILES"

# sglang.launch_server flags:
#   --model-path         path to HF model dir
#   --host / --port      bind address
#   --api-key            bearer token (matches OpenAI-compat /v1/*)
#   --tp                 tensor-parallel size
#   --mem-fraction-static  analogous to vllm --gpu-memory-utilization
#   --context-length     analogous to vllm --max-model-len
#   --trust-remote-code  HF trust-remote-code
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
echo "[sweep-sglang] sglang PID=$SERVER_PID (port $PORT)"

trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; true' EXIT

# sglang takes a bit longer to warm up than vllm - same 15-min budget.
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" > /dev/null 2>&1; then
        echo "[sweep-sglang] server ready after ${i}x5s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[sweep-sglang] server died; tail log:"
        tail -30 /tmp/vllm_${PORT}.log
        exit 1
    fi
    sleep 5
done

cd /tmp/inference-benchmark

# Capture engine version for dashboard attribution.
SGLANG_VERSION=$("$PY" -c "import sglang; print(sglang.__version__)" 2>/dev/null || echo "unknown")
echo "backend=sglang version=$SGLANG_VERSION" > "$OUT_DIR/_engine_version.txt"
echo "[sweep-sglang] captured engine version: sglang $SGLANG_VERSION"

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
            --output     "$OUT_FILE" \
            --mode       single-turn \
            2>&1 | tail -8
    done
done

echo "[sweep-sglang] done; results in $OUT_DIR"
ls -la "$OUT_DIR" | tail -20
