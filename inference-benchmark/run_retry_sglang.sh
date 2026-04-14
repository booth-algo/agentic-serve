#!/usr/bin/env bash
# Retry failed SGLang benchmarks: Qwen2.5-72B and Llama-3.1-70B
set -uo pipefail

PYTHON="${PYTHON:-$(which python)}"
PORT=8000
API_KEY="test"
WARMUP=5
TIMEOUT=300
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
PROFILES="chatbot-short rag-retrieval rag-heavy coding-assist coding-heavy"
MAX_SERVER_WAIT=1200  # 20 min for large models

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[RETRY]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

kill_all_servers() {
    pkill -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
}

wait_for_server() {
    local elapsed=0
    while ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $MAX_SERVER_WAIT ]; then
            return 1
        fi
        if (( elapsed % 30 == 0 )); then echo -n "."; fi
    done
    echo ""
    log "Server ready after ${elapsed}s"
    return 0
}

run_suite() {
    local model_name="$1" model_path="$2" results_dir="$3"
    local url="http://localhost:${PORT}/v1/chat/completions"
    mkdir -p "$results_dir"

    for PROFILE in $PROFILES; do
        log "  Profile: $PROFILE"
        for CONC in $CONC_SWEEP; do
            NREQ=200
            [[ "$CONC" -eq 1 ]] && NREQ=50
            [[ "$CONC" -ge 200 ]] && NREQ=150

            local tag="${model_name}_tp2_sglang_${PROFILE}_conc${CONC}"
            local out="${results_dir}/${tag}.json"

            if [[ -f "$out" ]] && [[ -s "$out" ]]; then
                log "    SKIP conc=$CONC (exists)"
                continue
            fi

            log "    conc=$CONC nreq=$NREQ"
            OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
                --url "$url" --model "$model_path" --backend sglang \
                --profile "$PROFILE" --concurrency "$CONC" \
                --num-requests "$NREQ" --warmup "$WARMUP" --timeout "$TIMEOUT" \
                --api-key "$API_KEY" --output "$out" 2>&1 || {
                    err "    FAILED: $PROFILE conc=$CONC"
                }
        done
    done
}

# === Qwen2.5-72B TP=2 ===
log "══════ Qwen2.5-72B TP=2 SGLang ══════"
kill_all_servers
$PYTHON -m sglang.launch_server \
    --model-path /workspace/models/Qwen2.5-72B-Instruct \
    --tp 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 \
    --mem-fraction-static 0.95 --context-length 4096 \
    --api-key $API_KEY --trust-remote-code \
    > /tmp/server_retry_72b.log 2>&1 &
SERVER_PID=$!
log "Server PID: $SERVER_PID"

if wait_for_server; then
    run_suite "Qwen2.5-72B" "/workspace/models/Qwen2.5-72B-Instruct" "results/Qwen2.5-72B_tp2_sglang"
    log "DONE: Qwen2.5-72B ($(ls results/Qwen2.5-72B_tp2_sglang/*.json 2>/dev/null | wc -l) files)"
else
    err "Server failed to start for Qwen2.5-72B"
    tail -20 /tmp/server_retry_72b.log
fi

# === Llama-3.1-70B TP=2 ===
log "══════ Llama-3.1-70B TP=2 SGLang ══════"
kill_all_servers
$PYTHON -m sglang.launch_server \
    --model-path /workspace/models/Llama-3.1-70B-Instruct \
    --tp 2 --port $PORT --dtype bfloat16 --host 0.0.0.0 \
    --mem-fraction-static 0.95 --context-length 4096 \
    --api-key $API_KEY \
    > /tmp/server_retry_70b.log 2>&1 &
SERVER_PID=$!
log "Server PID: $SERVER_PID"

if wait_for_server; then
    run_suite "Llama-3.1-70B" "/workspace/models/Llama-3.1-70B-Instruct" "results/Llama-3.1-70B_tp2_sglang"
    log "DONE: Llama-3.1-70B ($(ls results/Llama-3.1-70B_tp2_sglang/*.json 2>/dev/null | wc -l) files)"
else
    err "Server failed to start for Llama-3.1-70B"
    tail -20 /tmp/server_retry_70b.log
fi

kill_all_servers
log "ALL RETRIES COMPLETE"
