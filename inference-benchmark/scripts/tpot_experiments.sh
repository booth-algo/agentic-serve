#!/usr/bin/env bash
# =============================================================================
# TPOT Discontinuity Experiments
# Tests hypotheses for the TPOT drop at conc ~200:
#   1. KV cache capacity (gpu_memory_utilization: 0.80 vs 0.92 vs 0.95)
#   2. max_num_batched_tokens (4096 vs 8192 vs 16384)
#   3. Chunked prefill (on vs off)
#   4. Smaller model (8B) where KV cache is proportionally larger
#
# Each experiment runs chat-singleturn across the transition zone (80-320)
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
API_KEY="test"
PROFILE="chat-singleturn"
WARMUP=5
TIMEOUT=300

# Focused concurrency levels around the transition zone
CONCS=(80 100 120 140 160 180 200 224 256 320)

log() { echo -e "\033[0;32m[EXP]\033[0m $1"; }

wait_for_health() {
    local port="$1" max_wait="${2:-600}"
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

kill_all_servers() {
    for pidfile in /tmp/tpot_exp_server_*.pid; do
        [[ -f "$pidfile" ]] || continue
        local pid=$(cat "$pidfile")
        kill -9 -"$pid" 2>/dev/null || true
        rm -f "$pidfile"
    done
    sleep 5
}

run_experiment() {
    local exp_name="$1"
    local model="$2"
    local tp="$3"
    local cuda_devs="$4"
    local port="$5"
    local gpu_mem="$6"
    local max_model_len="$7"
    local extra_flags="$8"
    local nreq="${9:-200}"

    local results_dir="results/tpot_experiments/${exp_name}"
    mkdir -p "$results_dir"

    log "╔══════════════════════════════════════════════════════════╗"
    log "║  Experiment: $exp_name"
    log "╚══════════════════════════════════════════════════════════╝"

    # Start server
    log "Starting server: model=$(basename $model) TP=$tp gpu_mem=$gpu_mem $extra_flags"
    CUDA_VISIBLE_DEVICES=$cuda_devs setsid "$PYTHON" -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --tensor-parallel-size "$tp" \
        --port "$port" \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --gpu-memory-utilization "$gpu_mem" \
        --max-model-len "$max_model_len" \
        --enable-prefix-caching \
        --api-key "$API_KEY" \
        $extra_flags \
        > "/tmp/tpot_exp_server_${exp_name}.log" 2>&1 &
    echo $! > "/tmp/tpot_exp_server_${exp_name}.pid"

    if ! wait_for_health "$port" 600; then
        log "FAILED to start server for $exp_name"
        kill_all_servers
        return 1
    fi

    local url="http://localhost:${port}/v1/chat/completions"

    for CONC in "${CONCS[@]}"; do
        local out="${results_dir}/${PROFILE}_conc${CONC}.json"

        # Skip if exists and valid
        if [[ -f "$out" ]] && [[ -s "$out" ]]; then
            completed=$("$PYTHON" -c "import json; d=json.load(open('$out')); s=d.get('summary',{}); print(s.get('successful_requests',0))" 2>/dev/null || echo "0")
            if [[ "$completed" -gt 0 ]]; then
                log "  SKIP conc=$CONC (exists)"
                continue
            fi
        fi

        local req=$nreq
        [[ "$CONC" -ge 200 ]] && req=150

        log "  Running conc=$CONC nreq=$req"
        OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
            --url "$url" \
            --model "$model" \
            --backend vllm \
            --profile "$PROFILE" \
            --concurrency "$CONC" \
            --num-requests "$req" \
            --warmup "$WARMUP" \
            --timeout "$TIMEOUT" \
            --api-key "$API_KEY" \
            --output "$out" \
            2>&1 || log "  FAILED conc=$CONC"
    done

    # Print results table
    log ""
    log "=== $exp_name Results ==="
    log "Conc | TPOT p50 (ms) | Output tok/s"
    log "-----|---------------|------------"
    for CONC in "${CONCS[@]}"; do
        out="${results_dir}/${PROFILE}_conc${CONC}.json"
        if [[ -f "$out" ]]; then
            "$PYTHON" -c "
import json
d = json.load(open('$out'))
s = d.get('summary', {})
print(f'{$CONC:>4} | {s.get(\"median_tpot_ms\",0):>13.2f} | {s.get(\"output_token_throughput\",0):>11.0f}')
" 2>/dev/null || echo "  $CONC | ERROR"
        fi
    done

    # Kill server
    log "Stopping server..."
    kill_all_servers
}

# =============================================================================
# Experiments
# =============================================================================

MODEL_27B="/workspace/models/Qwen3.5-27B"
MODEL_8B="/workspace/models/Llama-3.1-8B-Instruct"
TRUST_REMOTE="--trust-remote-code --gdn-prefill-backend triton"

log "╔══════════════════════════════════════════════════════════╗"
log "║  TPOT DISCONTINUITY EXPERIMENTS                         ║"
log "║  Testing: KV cache, batched tokens, chunked prefill     ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""

# --- Experiment 1: KV cache capacity (gpu_memory_utilization) ---
# Hypothesis: lower gpu_mem = less KV cache = cliff shifts left
#             higher gpu_mem = more KV cache = cliff shifts right

run_experiment "kv_cache_low_0.80" \
    "$MODEL_27B" 2 "0,1" 9000 0.80 16384 \
    "--enable-chunked-prefill $TRUST_REMOTE"

run_experiment "kv_cache_high_0.95" \
    "$MODEL_27B" 2 "0,1" 9000 0.95 16384 \
    "--enable-chunked-prefill $TRUST_REMOTE"

# Baseline for comparison (0.92 is what we already ran — skip if exists)
run_experiment "kv_cache_baseline_0.92" \
    "$MODEL_27B" 2 "0,1" 9000 0.92 16384 \
    "--enable-chunked-prefill $TRUST_REMOTE"

# --- Experiment 2: max_num_batched_tokens ---
# Hypothesis: lower budget = smaller decode batches always = earlier cliff
#             higher budget = larger batches = delayed cliff

run_experiment "batched_tokens_4096" \
    "$MODEL_27B" 2 "0,1" 9000 0.92 16384 \
    "--enable-chunked-prefill --max-num-batched-tokens 4096 $TRUST_REMOTE"

run_experiment "batched_tokens_16384" \
    "$MODEL_27B" 2 "0,1" 9000 0.92 16384 \
    "--enable-chunked-prefill --max-num-batched-tokens 16384 $TRUST_REMOTE"

run_experiment "batched_tokens_32768" \
    "$MODEL_27B" 2 "0,1" 9000 0.92 16384 \
    "--enable-chunked-prefill --max-num-batched-tokens 32768 $TRUST_REMOTE"

# --- Experiment 3: Chunked prefill ON vs OFF ---
# Hypothesis: chunked prefill interleaves prefill chunks with decode,
#             stealing decode batch budget. Without it, prefill runs
#             as one big block, decode gets full budget → different curve.

run_experiment "no_chunked_prefill" \
    "$MODEL_27B" 2 "0,1" 9000 0.92 16384 \
    "$TRUST_REMOTE"

# --- Experiment 4: Smaller model (8B TP=1) ---
# Hypothesis: 8B uses ~16GB for weights vs 54GB for 27B, leaving
#             much more KV cache space. Cliff should be at much higher conc.

run_experiment "small_model_8B_tp1" \
    "$MODEL_8B" 1 "0" 9000 0.92 32768 \
    "--enable-chunked-prefill"

# =============================================================================
# Final summary
# =============================================================================
log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  ALL EXPERIMENTS COMPLETE                                ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
log "Results in results/tpot_experiments/"
for d in results/tpot_experiments/*/; do
    [[ -d "$d" ]] || continue
    count=$(ls "$d"/*.json 2>/dev/null | wc -l)
    log "  $(basename "$d"): $count files"
done
