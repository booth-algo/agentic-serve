#!/usr/bin/env bash
# High concurrency saturation retry — proper GPU cleanup between runs
set -uo pipefail

PYTHON="${PYTHON:-$(which python)}"
PORT=8000
API_KEY="test"
WARMUP=5
TIMEOUT=300
CONC_HIGH="400 512 640 768"
MAX_SERVER_WAIT=1200

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[SAT]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

kill_and_wait_gpu_free() {
    log "Killing all servers..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "sglang.srt" 2>/dev/null || true
    pkill -9 -f "VLLM::Worker" 2>/dev/null || true
    pkill -9 -f "multiprocessing.resource_tracker" 2>/dev/null || true

    # Also kill any process nvidia-smi reports as using the GPUs
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
    done

    # Wait for GPU memory to actually free
    local attempts=0
    while true; do
        sleep 5
        local gpu0_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
        local gpu1_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
        if [[ "$gpu0_used" -lt 1000 ]] && [[ "$gpu1_used" -lt 1000 ]]; then
            log "GPUs free (GPU0: ${gpu0_used}MiB, GPU1: ${gpu1_used}MiB)"
            break
        fi
        # Retry killing GPU processes
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do
            [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null || true
        done
        attempts=$((attempts + 1))
        if [[ $attempts -ge 24 ]]; then
            err "GPUs not free after 2 min (GPU0: ${gpu0_used}MiB, GPU1: ${gpu1_used}MiB)"
            return 1
        fi
    done
    return 0
}

wait_for_server() {
    local elapsed=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $MAX_SERVER_WAIT ]; then return 1; fi
        if (( elapsed % 30 == 0 )); then echo -n "."; fi
    done
    echo ""; log "Server ready after ${elapsed}s"; return 0
}

run_high_conc() {
    local engine="$1" model_name="$2" model_path="$3" tp="$4" results_dir="$5"
    local url="http://localhost:${PORT}/v1/chat/completions"
    mkdir -p "$results_dir"

    for CONC in $CONC_HIGH; do
        local tag="${model_name}_tp${tp}_${engine}_chat-singleturn_conc${CONC}"
        local out="${results_dir}/${tag}.json"

        if [[ -f "$out" ]] && [[ -s "$out" ]]; then
            log "  SKIP conc=$CONC (exists)"
            continue
        fi

        log "  chat-singleturn conc=$CONC nreq=150"
        OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
            --url "$url" --model "$model_path" --backend "$engine" \
            --profile chat-singleturn --concurrency "$CONC" \
            --num-requests 150 --warmup "$WARMUP" --timeout "$TIMEOUT" \
            --api-key "$API_KEY" --output "$out" 2>&1 || {
                err "  FAILED: conc=$CONC"
            }
    done
}

# Model configs: name|path|tp|sglang_extra|vllm_extra|max_len|gpu_mem
declare -a MODELS=(
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1||--enable-chunked-prefill|32768|0.90"
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|2||--enable-chunked-prefill|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "Qwen3-32B|/workspace/models/Qwen3-32B|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|16384|0.92"
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|16384|0.92"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|4096|0.95"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2||--enable-chunked-prefill|4096|0.95"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path tp sglang_extra vllm_extra max_len gpu_mem <<< "$entry"

    for engine in sglang vllm; do
        results_dir="results/${name}_tp${tp}_${engine}"

        # Check if all 4 high-conc files already exist
        existing=0
        for c in $CONC_HIGH; do
            [[ -f "${results_dir}/${name}_tp${tp}_${engine}_chat-singleturn_conc${c}.json" ]] && existing=$((existing + 1))
        done
        if [[ $existing -eq 4 ]]; then
            log "SKIP $name TP=$tp $engine (all 4 high-conc exist)"
            continue
        fi

        log "══ $engine: $name TP=$tp ══"
        kill_and_wait_gpu_free || continue

        if [[ "$engine" == "sglang" ]]; then
            local_extra="$sglang_extra"
            cmd="$PYTHON -m sglang.launch_server --model-path $path --tp $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
            [[ -n "$local_extra" ]] && cmd="$cmd $local_extra"
        else
            local_extra="$vllm_extra"
            cmd="$PYTHON -m vllm.entrypoints.openai.api_server --model $path --tensor-parallel-size $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization $gpu_mem --max-model-len $max_len --enable-prefix-caching --api-key $API_KEY"
            [[ -n "$local_extra" ]] && cmd="$cmd $local_extra"
        fi

        log "CMD: $cmd"
        eval $cmd > "/tmp/server_sat_${name}_tp${tp}_${engine}.log" 2>&1 &

        if wait_for_server; then
            run_high_conc "$engine" "$name" "$path" "$tp" "$results_dir"
            log "DONE: $name TP=$tp $engine saturation"
        else
            err "Server failed: $name TP=$tp $engine"
            tail -5 /tmp/server_sat_${name}_tp${tp}_${engine}.log
        fi
    done
done

kill_and_wait_gpu_free
log "ALL SATURATION COMPLETE"
find results -name "*conc400*" -o -name "*conc512*" -o -name "*conc640*" -o -name "*conc768*" 2>/dev/null | wc -l
echo "total high-conc files"
