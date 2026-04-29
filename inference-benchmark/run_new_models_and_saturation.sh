#!/usr/bin/env bash
# =============================================================================
# Phase 1: Qwen3-32B on SGLang + vLLM
# Phase 2: High concurrency saturation sweep (400,512,640,768) on chat-singleturn
# =============================================================================
set -uo pipefail

PYTHON="${PYTHON:-$(which python)}"
PORT=8000
API_KEY="test"
WARMUP=5
TIMEOUT=300
PROFILES="chat-singleturn coding-agent"
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
CONC_HIGH="400 512 640 768"
MAX_SERVER_WAIT=1800

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[BENCH]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

kill_all_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
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

run_suite() {
    local engine="$1" model_name="$2" model_path="$3" tp="$4" results_dir="$5" profiles="$6" conc_list="$7"
    local url="http://localhost:${PORT}/v1/chat/completions"
    mkdir -p "$results_dir"

    for PROFILE in $profiles; do
        log "  Profile: $PROFILE"
        for CONC in $conc_list; do
            NREQ=200
            [[ "$CONC" -eq 1 ]] && NREQ=50
            [[ "$CONC" -ge 200 ]] && NREQ=150

            local tag="${model_name}_tp${tp}_${engine}_${PROFILE}_conc${CONC}"
            local out="${results_dir}/${tag}.json"

            if [[ -f "$out" ]] && [[ -s "$out" ]]; then
                log "    SKIP conc=$CONC (exists)"
                continue
            fi

            log "    conc=$CONC nreq=$NREQ"
            OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
                --url "$url" --model "$model_path" --backend "$engine" \
                --profile "$PROFILE" --concurrency "$CONC" \
                --num-requests "$NREQ" --warmup "$WARMUP" --timeout "$TIMEOUT" \
                --api-key "$API_KEY" --output "$out" 2>&1 || {
                    err "    FAILED: $PROFILE conc=$CONC"
                }
        done
    done
}

start_sglang() {
    local model_path="$1" tp="$2" extra="$3" max_len="$4" gpu_mem="$5" tag="$6"
    kill_all_servers
    local cmd="$PYTHON -m sglang.launch_server --model-path $model_path --tp $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    log "CMD: $cmd"
    eval $cmd > "/tmp/server_${tag}.log" 2>&1 &
    echo $!
}

start_vllm() {
    local model_path="$1" tp="$2" extra="$3" max_len="$4" gpu_mem="$5" tag="$6"
    kill_all_servers
    local cmd="$PYTHON -m vllm.entrypoints.openai.api_server --model $model_path --tensor-parallel-size $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization $gpu_mem --max-model-len $max_len --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --max-num-seqs 512"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    log "CMD: $cmd"
    eval $cmd > "/tmp/server_${tag}.log" 2>&1 &
    echo $!
}

# ═══════════════════════════════════════════════════
# PHASE 1: Qwen3-32B full sweep
# ═══════════════════════════════════════════════════
log "╔══════════════════════════════════════════╗"
log "║  PHASE 1: Qwen3-32B TP=2                ║"
log "╚══════════════════════════════════════════╝"

# SGLang
log "══ SGLang: Qwen3-32B TP=2 ══"
start_sglang "/workspace/models/Qwen3-32B" 2 "--trust-remote-code" 16384 0.92 "Qwen3-32B_tp2_sglang"
if wait_for_server; then
    run_suite "sglang" "Qwen3-32B" "/workspace/models/Qwen3-32B" 2 "results/Qwen3-32B_tp2_sglang" "$PROFILES" "$CONC_SWEEP"
    log "DONE: Qwen3-32B SGLang ($(ls results/Qwen3-32B_tp2_sglang/*.json 2>/dev/null | wc -l) files)"
else
    err "Server failed for Qwen3-32B SGLang"
    tail -20 /tmp/server_Qwen3-32B_tp2_sglang.log
fi

# vLLM
log "══ vLLM: Qwen3-32B TP=2 ══"
start_vllm "/workspace/models/Qwen3-32B" 2 "--trust-remote-code" 16384 0.92 "Qwen3-32B_tp2_vllm"
if wait_for_server; then
    run_suite "vllm" "Qwen3-32B" "/workspace/models/Qwen3-32B" 2 "results/Qwen3-32B_tp2_vllm" "$PROFILES" "$CONC_SWEEP"
    log "DONE: Qwen3-32B vLLM ($(ls results/Qwen3-32B_tp2_vllm/*.json 2>/dev/null | wc -l) files)"
else
    err "Server failed for Qwen3-32B vLLM"
    tail -20 /tmp/server_Qwen3-32B_tp2_vllm.log
fi

# ═══════════════════════════════════════════════════
# PHASE 2: High concurrency saturation (chat-singleturn only)
# ═══════════════════════════════════════════════════
log "╔══════════════════════════════════════════╗"
log "║  PHASE 2: High Concurrency Saturation   ║"
log "╚══════════════════════════════════════════╝"

# Model configs: name|path|tp|sglang_extra|vllm_extra|max_len|gpu_mem
declare -a SAT_MODELS=(
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1||--enable-chunked-prefill|32768|0.90"
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|2||--enable-chunked-prefill|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "Qwen3-32B|/workspace/models/Qwen3-32B|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|16384|0.92"
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|16384|0.92"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|4096|0.95"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2||--enable-chunked-prefill|4096|0.95"
)

for entry in "${SAT_MODELS[@]}"; do
    IFS='|' read -r name path tp sglang_extra vllm_extra max_len gpu_mem <<< "$entry"

    # SGLang high concurrency
    log "══ SGLang saturation: $name TP=$tp ══"
    start_sglang "$path" "$tp" "$sglang_extra" "$max_len" "$gpu_mem" "${name}_tp${tp}_sglang_sat"
    if wait_for_server; then
        run_suite "sglang" "$name" "$path" "$tp" "results/${name}_tp${tp}_sglang" "chat-singleturn" "$CONC_HIGH"
        log "DONE: $name SGLang saturation"
    else
        err "Server failed: $name SGLang saturation"
        tail -10 /tmp/server_${name}_tp${tp}_sglang_sat.log
    fi

    # vLLM high concurrency
    log "══ vLLM saturation: $name TP=$tp ══"
    start_vllm "$path" "$tp" "$vllm_extra" "$max_len" "$gpu_mem" "${name}_tp${tp}_vllm_sat"
    if wait_for_server; then
        run_suite "vllm" "$name" "$path" "$tp" "results/${name}_tp${tp}_vllm" "chat-singleturn" "$CONC_HIGH"
        log "DONE: $name vLLM saturation"
    else
        err "Server failed: $name vLLM saturation"
        tail -10 /tmp/server_${name}_tp${tp}_vllm_sat.log
    fi
done

kill_all_servers
log "╔══════════════════════════════════════════╗"
log "║  ALL COMPLETE                            ║"
log "╚══════════════════════════════════════════╝"

# Summary
log "Results summary:"
for d in results/*/; do
    echo "  $(basename $d): $(ls "$d"/*.json 2>/dev/null | wc -l) files"
done
