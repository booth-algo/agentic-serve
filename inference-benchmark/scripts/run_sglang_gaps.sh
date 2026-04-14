#!/usr/bin/env bash
# Fill SGLang 70B TP2 conc=80 gaps for coding-agent, prefill-heavy, decode-heavy
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/root/sglang-venv/bin/python}"
CLIENT_PYTHON="/root/miniconda3/bin/python"
API_KEY="test"
WARMUP=5
TIMEOUT=300

log() { echo -e "\033[0;32m[GAP]\033[0m $1"; }

MODELS=(
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|--disable-piecewise-cuda-graph"
    "Llama-3.3-70B|/workspace/models/Llama-3.3-70B-Instruct|--disable-piecewise-cuda-graph"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|--trust-remote-code --disable-piecewise-cuda-graph"
)
PROFILES="coding-agent prefill-heavy decode-heavy"
CONC=80
NREQ=200

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r name path extra <<< "$model_spec"
    tag="${name}_tp2_sglang"

    # Find latest date dir or create today's
    latest=$(ls -d "results/${tag}"/2026-* 2>/dev/null | sort -r | head -1)
    if [[ -z "$latest" ]]; then
        results_dir="results/${tag}/$(date +%Y-%m-%d)"
    else
        results_dir="$latest"
    fi
    mkdir -p "$results_dir"

    log "Starting $name TP=2 on GPU=0,1"
    CUDA_VISIBLE_DEVICES=0,1 setsid "$PYTHON" -m sglang.launch_server \
        --model-path "$path" \
        --tp 2 \
        --port 9000 \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --mem-fraction-static 0.90 \
        --context-length 32768 \
        --api-key "$API_KEY" \
        $extra \
        > "/tmp/sglang_gap_${name}.log" 2>&1 &
    SERVER_PID=$!

    # Wait for health
    elapsed=0
    while ! curl -s http://localhost:9000/health > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge 600 ]; then
            log "FAILED: $name server didn't start"
            kill -9 -$SERVER_PID 2>/dev/null
            continue 2
        fi
    done
    log "$name server ready (${elapsed}s)"

    for PROFILE in $PROFILES; do
        out="${results_dir}/${PROFILE}_conc${CONC}.json"
        if [[ -f "$out" ]] && [[ -s "$out" ]]; then
            completed=$("$CLIENT_PYTHON" -c "import json; d=json.load(open('$out')); s=d.get('summary',{}); print(s.get('successful_requests',0))" 2>/dev/null || echo "0")
            if [[ "$completed" -gt 0 ]]; then
                log "  SKIP $PROFILE conc=$CONC (exists, $completed reqs)"
                continue
            fi
        fi

        log "  Running $PROFILE conc=$CONC nreq=$NREQ"
        OPENAI_API_KEY="$API_KEY" "$CLIENT_PYTHON" -m src.benchmark.runner \
            --url http://localhost:9000/v1/chat/completions \
            --model "$path" \
            --backend sglang \
            --profile "$PROFILE" \
            --concurrency "$CONC" \
            --num-requests "$NREQ" \
            --warmup "$WARMUP" \
            --timeout "$TIMEOUT" \
            --api-key "$API_KEY" \
            --output "$out" \
            2>&1 || log "  FAILED $PROFILE"
    done

    log "Stopping $name server"
    kill -9 -$SERVER_PID 2>/dev/null
    sleep 5
done

log "All SGLang gaps filled"
