#!/usr/bin/env bash
# Multi-turn sweep — 1x H100 (GPU0 only) models on SGLang
set -uo pipefail
export CUDA_VISIBLE_DEVICES=0
PYTHON="${PYTHON:-$(which python)}"
PORT=8000; API_KEY="test"; WARMUP=3; TIMEOUT=300; MAX_WAIT=600
MT_PROFILES="multi-turn-short multi-turn-medium multi-turn-long"
CONC_SWEEP="1 10 20 40 80"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[MT]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERR]\033[0m $1"; }

gpu_free() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 2>/dev/null | tr -d ' ' | while read pid; do
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

run_mt() {
    local eng=$1 name=$2 path=$3 dir=$4
    mkdir -p "$dir"
    for PROF in $MT_PROFILES; do
        for C in $CONC_SWEEP; do
            local out="$dir/${name}_tp1_${eng}_${PROF}_conc${C}.json"
            [[ -f "$out" ]] && [[ -s "$out" ]] && { log "SKIP $PROF conc=$C"; continue; }
            log "$PROF conc=$C"
            OPENAI_API_KEY=$API_KEY $PYTHON -m src.benchmark.runner \
                --url http://localhost:$PORT/v1/chat/completions --model "$path" --backend "$eng" \
                --profile "$PROF" --concurrency "$C" --num-requests 50 \
                --warmup $WARMUP --timeout $TIMEOUT --api-key $API_KEY \
                --mode multi-turn --output "$out" 2>&1 || err "FAIL $PROF $C"
        done
    done
}

# Models that fit on 1x H100
declare -a MODELS=(
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct||32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|--trust-remote-code --disable-overlap-schedule|32768|0.90"
    "gpt-oss-20b|/workspace/models/gpt-oss-20b|--trust-remote-code|32768|0.90"
    "gpt-oss-120b|/workspace/models/gpt-oss-120b|--trust-remote-code|32768|0.90"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path extra max_len gpu_mem <<< "$entry"

    log "══ SGLang: $name (multi-turn) ══"
    gpu_free
    cmd="$PYTHON -m sglang.launch_server --model-path $path --tp 1 --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    eval $cmd > "/tmp/srv_mt_${name}.log" 2>&1 &
    SRV=$!
    if wait_health $SRV; then
        run_mt sglang "$name" "$path" "results/${name}_tp1_sglang"
        log "DONE: $name multi-turn"
    else
        err "$name failed to start"; tail -5 "/tmp/srv_mt_${name}.log"
    fi
done

gpu_free
log "ALL MULTI-TURN COMPLETE"
