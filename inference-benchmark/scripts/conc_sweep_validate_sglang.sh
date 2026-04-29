#!/usr/bin/env bash
# Fine-grained concurrency sweep for TPOT validation
# Runs Qwen3.5-27B TP=2 on two GPU pairs in parallel
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
SGLANG_PYTHON="/root/sglang-venv/bin/python"
MODEL="/workspace/models/Qwen3.5-27B"
API_KEY="test"
PROFILES="chat-singleturn prefill-heavy"
NREQ=200
WARMUP=5
TIMEOUT=300
RESULTS_DIR="results/tpot_validation_Qwen3.5-27B_tp2_sglang"
mkdir -p "$RESULTS_DIR"

# Fine-grained concurrency sweep up to 500
ALL_CONCS=(1 5 10 15 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 180 200 224 256 288 320 352 384 416 448 500)

# Split into two lists for parallel execution
SLOT0_CONCS=()
SLOT1_CONCS=()
for i in "${!ALL_CONCS[@]}"; do
    if (( i % 2 == 0 )); then
        SLOT0_CONCS+=("${ALL_CONCS[$i]}")
    else
        SLOT1_CONCS+=("${ALL_CONCS[$i]}")
    fi
done

log() { echo -e "\033[0;32m[SWEEP]\033[0m $1"; }

start_server() {
    local cuda_devs="$1" port="$2" slot="$3"
    log "Starting server on GPU=$cuda_devs port=$port"
    CUDA_VISIBLE_DEVICES=$cuda_devs setsid "$SGLANG_PYTHON" -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp 2 \
        --port "$port" \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --mem-fraction-static 0.92 \
        --context-length 16384 \
        --trust-remote-code \
        --disable-overlap-schedule \
        --api-key "$API_KEY" \
        > "/tmp/tpot_server_slot${slot}.log" 2>&1 &
    echo $! > "/tmp/tpot_server_slot${slot}.pid"
    log "Server PID=$(cat /tmp/tpot_server_slot${slot}.pid)"
}

wait_for_health() {
    local port="$1" max_wait=600
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            log "FAILED: server on port $port didn't start in ${max_wait}s"
            return 1
        fi
    done
    log "Server on port $port ready (${elapsed}s)"
    return 0
}

run_sweep() {
    local port="$1" slot="$2"
    shift 2
    local concs=("$@")

    local url="http://localhost:${port}/v1/chat/completions"

    for PROFILE in $PROFILES; do
    for CONC in "${concs[@]}"; do
        local out="${RESULTS_DIR}/${PROFILE}_conc${CONC}.json"

        # Skip if exists and valid
        if [[ -f "$out" ]] && [[ -s "$out" ]]; then
            completed=$("$PYTHON" -c "import json; d=json.load(open('$out')); s=d.get('summary',{}); print(s.get('successful_requests',0))" 2>/dev/null || echo "0")
            if [[ "$completed" -gt 0 ]]; then
                log "[SLOT$slot] SKIP $PROFILE conc=$CONC (exists, $completed reqs)"
                continue
            fi
        fi

        # nreq must always exceed concurrency to ensure full load pressure
        local nreq=$(( CONC * 2 ))
        [[ "$nreq" -lt 200 ]] && nreq=200
        [[ "$CONC" -eq 1 ]] && nreq=50

        log "[SLOT$slot] Running $PROFILE conc=$CONC nreq=$nreq"
        OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
            --url "$url" \
            --model "$MODEL" \
            --backend sglang \
            --profile "$PROFILE" \
            --concurrency "$CONC" \
            --num-requests "$nreq" \
            --warmup "$WARMUP" \
            --timeout "$TIMEOUT" \
            --api-key "$API_KEY" \
            --output "$out" \
            2>&1 || log "[SLOT$slot] FAILED $PROFILE conc=$CONC"
    done
    done
    log "[SLOT$slot] Sweep complete"
}

# --- Main ---
log "╔══════════════════════════════════════════════════════════╗"
log "║  TPOT VALIDATION SWEEP — Qwen3.5-27B TP=2 (SGLang)     ║"
log "║  Fine-grained concurrency: ${#ALL_CONCS[@]} levels               ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
log "Slot 0 concs: ${SLOT0_CONCS[*]}"
log "Slot 1 concs: ${SLOT1_CONCS[*]}"

# Start two servers
start_server "0,1" 9000 0
start_server "2,3" 9001 1

# Wait for both
wait_for_health 9000 || exit 1
wait_for_health 9001 || exit 1

log "Both servers ready. Starting parallel sweeps..."

# Run sweeps in parallel
run_sweep 9000 0 "${SLOT0_CONCS[@]}" &
PID0=$!
run_sweep 9001 1 "${SLOT1_CONCS[@]}" &
PID1=$!

wait $PID0
wait $PID1

log "All sweeps done. Killing servers..."
kill -9 -$(cat /tmp/tpot_server_slot0.pid) 2>/dev/null
kill -9 -$(cat /tmp/tpot_server_slot1.pid) 2>/dev/null
sleep 3

log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  RESULTS SUMMARY                                        ║"
log "╚══════════════════════════════════════════════════════════╝"

# Print TPOT table per profile
for PROFILE in $PROFILES; do
log ""
log "=== $PROFILE ==="
log "Conc | TPOT p50 (ms) | Output tok/s | Requests OK"
log "-----|---------------|-------------|------------"
for CONC in "${ALL_CONCS[@]}"; do
    out="${RESULTS_DIR}/${PROFILE}_conc${CONC}.json"
    if [[ -f "$out" ]]; then
        "$PYTHON" -c "
import json
d = json.load(open('$out'))
s = d.get('summary', {})
print(f'{$CONC:>4} | {s.get(\"median_tpot_ms\",0):>13.2f} | {s.get(\"output_token_throughput\",0):>11.0f} | {s.get(\"successful_requests\",0)}')
" 2>/dev/null || echo "  $CONC | ERROR"
    fi
done
done

log "Done. Results in $RESULTS_DIR/"
