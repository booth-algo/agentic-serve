#!/usr/bin/env bash
# SGLang-only high concurrency saturation (chat-singleturn at 400,512,640,768)
set -uo pipefail
PYTHON="${PYTHON:-$(which python)}"
PORT=8000; API_KEY="test"; WARMUP=5; TIMEOUT=300; MAX_SERVER_WAIT=1800
CONC_HIGH="400 512 640 768"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$REPO_ROOT"
log() { echo -e "\033[0;32m[SAT-SG]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

kill_servers() { pkill -f "sglang.launch_server" 2>/dev/null || true; sleep 3; }

wait_for_server() {
    local elapsed=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $MAX_SERVER_WAIT ]; then return 1; fi
        if (( elapsed % 30 == 0 )); then echo -n "."; fi
    done; echo ""; log "Ready after ${elapsed}s"; return 0
}

run_high_conc() {
    local model_name="$1" model_path="$2" tp="$3" results_dir="$4"
    local url="http://localhost:${PORT}/v1/chat/completions"
    mkdir -p "$results_dir"
    for CONC in $CONC_HIGH; do
        local tag="${model_name}_tp${tp}_sglang_chat-singleturn_conc${CONC}"
        local out="${results_dir}/${tag}.json"
        [[ -f "$out" ]] && [[ -s "$out" ]] && { log "  SKIP conc=$CONC"; continue; }
        log "  conc=$CONC nreq=150"
        OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
            --url "$url" --model "$model_path" --backend sglang \
            --profile chat-singleturn --concurrency "$CONC" \
            --num-requests 150 --warmup "$WARMUP" --timeout "$TIMEOUT" \
            --api-key "$API_KEY" --output "$out" 2>&1 || err "  FAILED conc=$CONC"
    done
}

declare -a MODELS=(
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1|--trust-remote-code --disable-overlap-schedule|32768|0.90"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|2|--trust-remote-code --disable-overlap-schedule|32768|0.90"
    "Qwen3-32B|/workspace/models/Qwen3-32B|2|--trust-remote-code|16384|0.92"
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|16384|0.92"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|4096|0.95"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2||4096|0.95"
)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path tp extra max_len gpu_mem <<< "$entry"
    log "══ $name TP=$tp ══"
    kill_servers
    cmd="$PYTHON -m sglang.launch_server --model-path $path --tp $tp --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static $gpu_mem --context-length $max_len --api-key $API_KEY"
    [[ -n "$extra" ]] && cmd="$cmd $extra"
    eval $cmd > "/tmp/server_sat_${name}_tp${tp}_sg.log" 2>&1 &
    if wait_for_server; then
        run_high_conc "$name" "$path" "$tp" "results/${name}_tp${tp}_sglang"
        log "DONE: $name TP=$tp"
    else
        err "Server failed: $name TP=$tp"; tail -5 "/tmp/server_sat_${name}_tp${tp}_sg.log"
    fi
done
kill_servers
log "ALL SGLang SATURATION COMPLETE"
