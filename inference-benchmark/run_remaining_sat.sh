#!/usr/bin/env bash
set -uo pipefail
PYTHON="${PYTHON:-$(which python)}"
PORT=8000; API_KEY="test"; WARMUP=5; TIMEOUT=300; MAX_WAIT=1200
CONC_HIGH="400 512 640 768"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[SAT]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERR]\033[0m $1"; }

gpu_free() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null; done
    sleep 5
    local g0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    [[ "$g0" -lt 1000 ]] && return 0 || return 1
}

wait_health() {
    local e=0
    while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        # Check if server process died
        if ! kill -0 $1 2>/dev/null; then err "Server process died"; return 1; fi
        sleep 5; e=$((e+5))
        [[ $e -ge $MAX_WAIT ]] && return 1
    done; log "Ready ${e}s"; return 0
}

run_conc() {
    local eng=$1 name=$2 path=$3 tp=$4 dir=$5
    mkdir -p "$dir"
    for C in $CONC_HIGH; do
        local out="$dir/${name}_tp${tp}_${eng}_chat-singleturn_conc${C}.json"
        [[ -f "$out" ]] && { log "SKIP $C"; continue; }
        log "conc=$C"
        OPENAI_API_KEY=$API_KEY $PYTHON -m src.benchmark.runner \
            --url http://localhost:$PORT/v1/chat/completions --model "$path" --backend "$eng" \
            --profile chat-singleturn --concurrency "$C" --num-requests 150 \
            --warmup $WARMUP --timeout $TIMEOUT --api-key $API_KEY --output "$out" 2>&1 || err "FAIL $C"
    done
}

# === Qwen2.5-72B SGLang ===
log "=== Qwen2.5-72B SGLang ==="
gpu_free
$PYTHON -m sglang.launch_server --model-path /workspace/models/Qwen2.5-72B-Instruct --tp 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static 0.95 --context-length 4096 --api-key $API_KEY --trust-remote-code > /tmp/srv_72b_sg.log 2>&1 &
SRV=$!; wait_health $SRV && run_conc sglang Qwen2.5-72B /workspace/models/Qwen2.5-72B-Instruct 2 results/Qwen2.5-72B_tp2_sglang || err "72B SGLang failed"

# === Qwen2.5-72B vLLM ===
log "=== Qwen2.5-72B vLLM ==="
gpu_free
$PYTHON -m vllm.entrypoints.openai.api_server --model /workspace/models/Qwen2.5-72B-Instruct --tensor-parallel-size 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization 0.95 --max-model-len 4096 --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --trust-remote-code > /tmp/srv_72b_vl.log 2>&1 &
SRV=$!; wait_health $SRV && run_conc vllm Qwen2.5-72B /workspace/models/Qwen2.5-72B-Instruct 2 results/Qwen2.5-72B_tp2_vllm || err "72B vLLM failed"

# === Llama-3.1-70B SGLang ===
log "=== Llama-3.1-70B SGLang ==="
gpu_free
$PYTHON -m sglang.launch_server --model-path /workspace/models/Llama-3.1-70B-Instruct --tp 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static 0.95 --context-length 4096 --api-key $API_KEY > /tmp/srv_70b_sg.log 2>&1 &
SRV=$!; wait_health $SRV && run_conc sglang Llama-3.1-70B /workspace/models/Llama-3.1-70B-Instruct 2 results/Llama-3.1-70B_tp2_sglang || err "70B SGLang failed"

# === Llama-3.1-70B vLLM ===
log "=== Llama-3.1-70B vLLM ==="
gpu_free
$PYTHON -m vllm.entrypoints.openai.api_server --model /workspace/models/Llama-3.1-70B-Instruct --tensor-parallel-size 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization 0.95 --max-model-len 4096 --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill > /tmp/srv_70b_vl.log 2>&1 &
SRV=$!; wait_health $SRV && run_conc vllm Llama-3.1-70B /workspace/models/Llama-3.1-70B-Instruct 2 results/Llama-3.1-70B_tp2_vllm || err "70B vLLM failed"

gpu_free
log "DONE"
