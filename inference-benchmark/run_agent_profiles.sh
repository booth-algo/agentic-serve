#!/usr/bin/env bash
# Run coding-agent profiles across all models (SGLang + vLLM)
set -uo pipefail
PYTHON="${PYTHON:-$(which python)}"
PORT=8000; API_KEY="test"; WARMUP=5; TIMEOUT=300; MAX_WAIT=1200
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
PROFILES="coding-agent"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[AGENT]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERR]\033[0m $1"; }

gpu_free() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do
        [[ -n "$pid" ]] && kill -9 "$pid" 2>/dev/null; done
    lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 5
}

wait_health() {
    local srv_pid=$1 e=0
    while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
        if ! kill -0 $srv_pid 2>/dev/null; then err "Server died"; return 1; fi
        sleep 5; e=$((e+5))
        [[ $e -ge $MAX_WAIT ]] && return 1
    done; log "Ready ${e}s"; return 0
}

run_profiles() {
    local eng=$1 name=$2 path=$3 tp=$4 dir=$5
    mkdir -p "$dir"
    for PROF in $PROFILES; do
        for C in $CONC_SWEEP; do
            NREQ=200; [[ "$C" -eq 1 ]] && NREQ=50; [[ "$C" -ge 200 ]] && NREQ=150
            local out="$dir/${name}_tp${tp}_${eng}_${PROF}_conc${C}.json"
            [[ -f "$out" ]] && [[ -s "$out" ]] && { log "SKIP $PROF conc=$C"; continue; }
            log "$PROF conc=$C nreq=$NREQ"
            OPENAI_API_KEY=$API_KEY $PYTHON -m src.benchmark.runner \
                --url http://localhost:$PORT/v1/chat/completions --model "$path" --backend "$eng" \
                --profile "$PROF" --concurrency "$C" --num-requests "$NREQ" \
                --warmup $WARMUP --timeout $TIMEOUT --api-key $API_KEY --output "$out" 2>&1 || err "FAIL $PROF $C"
        done
    done
}

# Model configs: name|path|tp|sglang_extra|vllm_extra|max_len|gpu_mem
declare -a MODELS=(
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1|||32768|0.90"
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|2|||32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1|--trust-remote-code --disable-overlap-schedule|--trust-remote-code|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|2|--trust-remote-code --disable-overlap-schedule|--trust-remote-code|32768|0.90"
    "Qwen3-32B|/workspace/models/Qwen3-32B|2|--trust-remote-code|--trust-remote-code|16384|0.92"
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--trust-remote-code|16384|0.92"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|--trust-remote-code|4096|0.95"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2|||4096|0.95"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path tp sg_extra vl_extra max_len gpu_mem <<< "$entry"

    # SGLang
    log "══ SGLang: $name TP=$tp ══"
    gpu_free
    cmd="$PYTHON -m sglang.launch_server --model-path $path --tp $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
    [[ -n "$sg_extra" ]] && cmd="$cmd $sg_extra"
    eval $cmd > "/tmp/srv_agent_${name}_tp${tp}_sg.log" 2>&1 &
    SRV=$!
    if wait_health $SRV; then
        run_profiles sglang "$name" "$path" "$tp" "results/${name}_tp${tp}_sglang"
        log "DONE: $name TP=$tp SGLang"
    else
        err "$name TP=$tp SGLang failed"; tail -5 "/tmp/srv_agent_${name}_tp${tp}_sg.log"
    fi

    # vLLM (skip Qwen3.5 on vLLM 0.16)
    if [[ "$name" == *"Qwen3.5"* ]]; then
        log "SKIP vLLM for $name (Qwen3.5 unsupported on vLLM 0.16)"
        continue
    fi
    log "══ vLLM: $name TP=$tp ══"
    gpu_free
    cmd="$PYTHON -m vllm.entrypoints.openai.api_server --model $path --tensor-parallel-size $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization $gpu_mem --max-model-len $max_len --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --max-num-seqs 512"
    [[ -n "$vl_extra" ]] && cmd="$cmd $vl_extra"
    eval $cmd > "/tmp/srv_agent_${name}_tp${tp}_vl.log" 2>&1 &
    SRV=$!
    if wait_health $SRV; then
        run_profiles vllm "$name" "$path" "$tp" "results/${name}_tp${tp}_vllm"
        log "DONE: $name TP=$tp vLLM"
    else
        err "$name TP=$tp vLLM failed"; tail -5 "/tmp/srv_agent_${name}_tp${tp}_vl.log"
    fi
done

gpu_free
log "ALL AGENT PROFILES COMPLETE"
