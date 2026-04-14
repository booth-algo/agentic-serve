#!/usr/bin/env bash
# =============================================================================
# Overnight Benchmark Run — New Profiles
# Runs all new single-turn + stress-test profiles across all models × engines.
# Multi-turn chat profiles run as a separate pass.
#
# Usage: nohup bash run_overnight.sh > /tmp/overnight.log 2>&1 &
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$(which python)}"
PORT=8000
API_KEY="test"
WARMUP=5
TIMEOUT=300
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
MAX_SERVER_WAIT=1200

# Single-turn + stress-test profiles for the main orchestrator
# Note: prefill-heavy (ISL=8192) and decode-heavy (OSL=4096) need max_model_len >= 8448
# They'll be skipped for 70B/72B models (max_model_len=4096)
PROFILES_STANDARD="chat-short chat-medium coding-agent"
PROFILES_LONG_CTX="chat-long prefill-heavy decode-heavy"
PROFILES_STRESS="random-1k"

# Multi-turn profiles (run separately with --mode multi-turn)
MT_PROFILES_CHAT="chat-multiturn-short chat-multiturn-medium"

log() { echo -e "\033[0;32m[OVERNIGHT]\033[0m $(date '+%H:%M:%S') $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $(date '+%H:%M:%S') $1"; }

# Model registry
declare -a MODELS=(
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1,2||--enable-chunked-prefill|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1,2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|16384|0.92"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|4096|0.95"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2||--enable-chunked-prefill|4096|0.95"
    "gpt-oss-20b|/workspace/models/gpt-oss-20b|1|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|32768|0.90"
    "gpt-oss-120b|/workspace/models/gpt-oss-120b|1|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code|4096|0.95"
)

kill_all_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    local max_wait=$MAX_SERVER_WAIT elapsed=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then return 1; fi
        (( elapsed % 30 == 0 )) && echo -n "."
    done
    echo ""
    log "Server ready after ${elapsed}s"
    return 0
}

model_exists() {
    local path="$1"
    [[ -d "$path" ]] && [[ -f "$path/config.json" ]]
}

start_sglang_server() {
    local model_path="$1" tp="$2" extra="$3" max_len="$4" gpu_mem="$5" tag="$6"
    local cmd="$PYTHON -m sglang.launch_server --model-path $model_path --tp $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    log "CMD: $cmd"
    eval $cmd > "/tmp/server_${tag}.log" 2>&1 &
    echo $!
}

start_vllm_server() {
    local model_path="$1" tp="$2" extra="$3" max_len="$4" gpu_mem="$5" tag="$6"
    local cmd="$PYTHON -m vllm.entrypoints.openai.api_server --model $model_path --tensor-parallel-size $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization $gpu_mem --max-model-len $max_len --enable-prefix-caching --api-key $API_KEY"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    log "CMD: $cmd"
    eval $cmd > "/tmp/server_${tag}.log" 2>&1 &
    echo $!
}

run_profile() {
    local engine="$1" model_name="$2" model_path="$3" tp="$4" results_dir="$5" profile="$6" mode="${7:-}"
    local url="http://localhost:${PORT}/v1/chat/completions"

    for CONC in $CONC_SWEEP; do
        NREQ=200
        [[ "$CONC" -eq 1 ]] && NREQ=50
        [[ "$CONC" -ge 200 ]] && NREQ=150

        local tag="${model_name}_tp${tp}_${engine}_${profile}_conc${CONC}"
        local out="${results_dir}/${tag}.json"

        # Resume support
        if [[ -f "$out" ]] && [[ -s "$out" ]]; then
            log "    SKIP conc=$CONC (exists)"
            continue
        fi

        log "    profile=$profile conc=$CONC nreq=$NREQ"

        local mode_flag=""
        [[ -n "$mode" ]] && mode_flag="--mode $mode"

        OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
            --url        "$url" \
            --model      "$model_path" \
            --backend    "$engine" \
            --profile    "$profile" \
            --concurrency "$CONC" \
            --num-requests "$NREQ" \
            --warmup     "$WARMUP" \
            --timeout    "$TIMEOUT" \
            --api-key    "$API_KEY" \
            --output     "$out" \
            $mode_flag \
            2>&1 || {
                err "    FAILED: $profile conc=$CONC"
            }
    done
}

run_model() {
    local engine="$1" name="$2" path="$3" tp="$4" sglang_extra="$5" vllm_extra="$6" max_len="$7" gpu_mem="$8"

    local tag="${name}_tp${tp}_${engine}"
    local results_dir="results/${tag}"
    mkdir -p "$results_dir"

    log ""
    log "══════════════════════════════════════════════════"
    log "  $engine | $name | TP=$tp | max_len=$max_len"
    log "══════════════════════════════════════════════════"

    kill_all_servers

    # Start server
    local server_pid
    if [[ "$engine" == "sglang" ]]; then
        server_pid=$(start_sglang_server "$path" "$tp" "$sglang_extra" "$max_len" "$gpu_mem" "$tag")
    else
        server_pid=$(start_vllm_server "$path" "$tp" "$vllm_extra" "$max_len" "$gpu_mem" "$tag")
    fi
    log "Server PID: $server_pid"

    if ! wait_for_server; then
        err "Server failed to start for $tag"
        tail -20 "/tmp/server_${tag}.log"
        kill_all_servers
        return
    fi

    # --- Single-turn profiles (always run) ---
    for PROFILE in $PROFILES_STANDARD; do
        log "  ━━━ Profile: $PROFILE ━━━"
        run_profile "$engine" "$name" "$path" "$tp" "$results_dir" "$PROFILE"
    done

    # --- Long-context profiles (skip if max_model_len < 8448) ---
    if [[ "$max_len" -ge 8448 ]]; then
        for PROFILE in $PROFILES_LONG_CTX; do
            log "  ━━━ Profile: $PROFILE (long ctx) ━━━"
            run_profile "$engine" "$name" "$path" "$tp" "$results_dir" "$PROFILE"
        done
    else
        log "  SKIP long-ctx profiles (max_model_len=$max_len < 8448)"
    fi

    # --- Stress-test profiles (run on SGLang only to save time) ---
    if [[ "$engine" == "sglang" ]]; then
        for PROFILE in $PROFILES_STRESS; do
            log "  ━━━ Profile: $PROFILE (stress) ━━━"
            run_profile "$engine" "$name" "$path" "$tp" "$results_dir" "$PROFILE"
        done
    fi

    # --- Multi-turn chat profiles (SGLang only — has better prefix cache) ---
    if [[ "$engine" == "sglang" ]]; then
        for PROFILE in $MT_PROFILES_CHAT; do
            log "  ━━━ Profile: $PROFILE (multi-turn) ━━━"
            run_profile "$engine" "$name" "$path" "$tp" "$results_dir" "$PROFILE" "multi-turn"
        done
    fi

    kill_all_servers
    log "DONE: $tag ($(ls "$results_dir"/*.json 2>/dev/null | wc -l) files)"
}

# =============================================================================
# Main
# =============================================================================
ENGINE="${1:-all}"

log "╔══════════════════════════════════════════════════════════╗"
log "║  OVERNIGHT BENCHMARK RUN — NEW PROFILES                 ║"
log "║  Started: $(date)                      ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
log "Profiles (standard): $PROFILES_STANDARD"
log "Profiles (long-ctx): $PROFILES_LONG_CTX"
log "Profiles (stress):   $PROFILES_STRESS"
log "Profiles (multi-t):  $MT_PROFILES_CHAT"
log "Concurrency sweep:   $CONC_SWEEP"
log ""

for engine_target in sglang vllm; do
    if [[ "$ENGINE" != "all" ]] && [[ "$ENGINE" != "$engine_target" ]]; then
        continue
    fi

    log ""
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  Engine: $(printf '%-47s' "$engine_target")  ║"
    log "╚══════════════════════════════════════════════════════════╝"

    for model_spec in "${MODELS[@]}"; do
        IFS='|' read -r name path tp_list sglang_extra vllm_extra max_len gpu_mem <<< "$model_spec"

        if ! model_exists "$path"; then
            log "SKIP $name — not found at $path"
            continue
        fi

        IFS=',' read -ra tps <<< "$tp_list"
        for tp in "${tps[@]}"; do
            if [[ "$engine_target" == "sglang" ]]; then
                run_model "$engine_target" "$name" "$path" "$tp" "$sglang_extra" "" "$max_len" "$gpu_mem"
            else
                run_model "$engine_target" "$name" "$path" "$tp" "" "$vllm_extra" "$max_len" "$gpu_mem"
            fi
        done
    done
done

log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  ALL DONE — $(date)                    ║"
log "╚══════════════════════════════════════════════════════════╝"
log "Total result files: $(find results/ -name '*.json' -not -path '*/archive/*' | wc -l)"
