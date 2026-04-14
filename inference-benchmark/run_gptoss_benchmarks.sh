#!/usr/bin/env bash
# Benchmark GPT-OSS-20B and GPT-OSS-120B on SGLang + vLLM
set -uo pipefail
PYTHON="${PYTHON:-$(which python)}"
PORT=8000; API_KEY="test"; WARMUP=5; TIMEOUT=300; MAX_WAIT=1200
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
PROFILES="chatbot-short rag-retrieval rag-heavy coding-assist coding-heavy agentic-tool-use coding-agent"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

log() { echo -e "\033[0;32m[GPTOSS]\033[0m $1"; }
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

# === GPT-OSS-20B (dense, TP=1) ===
# SGLang
log "══ SGLang: GPT-OSS-20B TP=1 ══"
gpu_free
$PYTHON -m sglang.launch_server --model-path /workspace/models/gpt-oss-20b --tp 1 --port $PORT --dtype bfloat16 --host 0.0.0.0 --mem-fraction-static 0.90 --context-length 32768 --api-key $API_KEY --trust-remote-code > /tmp/srv_gptoss20b_sg.log 2>&1 &
SRV=$!
if wait_health $SRV; then
    run_profiles sglang gpt-oss-20b /workspace/models/gpt-oss-20b 1 results/gpt-oss-20b_tp1_sglang
    log "DONE: GPT-OSS-20B SGLang"
else
    err "GPT-OSS-20B SGLang failed"; tail -10 /tmp/srv_gptoss20b_sg.log
fi

# vLLM
log "══ vLLM: GPT-OSS-20B TP=1 ══"
gpu_free
$PYTHON -m vllm.entrypoints.openai.api_server --model /workspace/models/gpt-oss-20b --tensor-parallel-size 1 --port $PORT --dtype bfloat16 --host 0.0.0.0 --gpu-memory-utilization 0.90 --max-model-len 32768 --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --max-num-seqs 512 --trust-remote-code > /tmp/srv_gptoss20b_vl.log 2>&1 &
SRV=$!
if wait_health $SRV; then
    run_profiles vllm gpt-oss-20b /workspace/models/gpt-oss-20b 1 results/gpt-oss-20b_tp1_vllm
    log "DONE: GPT-OSS-20B vLLM"
else
    err "GPT-OSS-20B vLLM failed"; tail -10 /tmp/srv_gptoss20b_vl.log
fi

# === GPT-OSS-120B (MoE, TP=1 at native MXFP4 or TP=2 at BF16) ===
# Try TP=1 first (native precision should fit)
log "══ SGLang: GPT-OSS-120B TP=1 ══"
gpu_free
$PYTHON -m sglang.launch_server --model-path /workspace/models/gpt-oss-120b --tp 1 --port $PORT --dtype auto --host 0.0.0.0 --mem-fraction-static 0.90 --context-length 32768 --api-key $API_KEY --trust-remote-code > /tmp/srv_gptoss120b_sg.log 2>&1 &
SRV=$!
if wait_health $SRV; then
    run_profiles sglang gpt-oss-120b /workspace/models/gpt-oss-120b 1 results/gpt-oss-120b_tp1_sglang
    log "DONE: GPT-OSS-120B SGLang TP=1"
else
    err "GPT-OSS-120B SGLang TP=1 failed"; tail -10 /tmp/srv_gptoss120b_sg.log
    # Fallback: try TP=2
    log "Retrying with TP=2..."
    gpu_free
    $PYTHON -m sglang.launch_server --model-path /workspace/models/gpt-oss-120b --tp 2 --port $PORT --dtype auto --host 0.0.0.0 --mem-fraction-static 0.90 --context-length 32768 --api-key $API_KEY --trust-remote-code > /tmp/srv_gptoss120b_tp2_sg.log 2>&1 &
    SRV=$!
    if wait_health $SRV; then
        run_profiles sglang gpt-oss-120b /workspace/models/gpt-oss-120b 2 results/gpt-oss-120b_tp2_sglang
        log "DONE: GPT-OSS-120B SGLang TP=2"
    else
        err "GPT-OSS-120B SGLang TP=2 also failed"; tail -10 /tmp/srv_gptoss120b_tp2_sg.log
    fi
fi

# vLLM
log "══ vLLM: GPT-OSS-120B TP=1 ══"
gpu_free
$PYTHON -m vllm.entrypoints.openai.api_server --model /workspace/models/gpt-oss-120b --tensor-parallel-size 1 --port $PORT --dtype auto --host 0.0.0.0 --gpu-memory-utilization 0.90 --max-model-len 32768 --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --max-num-seqs 512 --trust-remote-code > /tmp/srv_gptoss120b_vl.log 2>&1 &
SRV=$!
if wait_health $SRV; then
    run_profiles vllm gpt-oss-120b /workspace/models/gpt-oss-120b 1 results/gpt-oss-120b_tp1_vllm
    log "DONE: GPT-OSS-120B vLLM TP=1"
else
    err "GPT-OSS-120B vLLM TP=1 failed"; tail -10 /tmp/srv_gptoss120b_vl.log
    log "Retrying with TP=2..."
    gpu_free
    $PYTHON -m vllm.entrypoints.openai.api_server --model /workspace/models/gpt-oss-120b --tensor-parallel-size 2 --port $PORT --dtype auto --host 0.0.0.0 --gpu-memory-utilization 0.90 --max-model-len 32768 --enable-prefix-caching --api-key $API_KEY --enable-chunked-prefill --max-num-seqs 512 --trust-remote-code > /tmp/srv_gptoss120b_tp2_vl.log 2>&1 &
    SRV=$!
    if wait_health $SRV; then
        run_profiles vllm gpt-oss-120b /workspace/models/gpt-oss-120b 2 results/gpt-oss-120b_tp2_vllm
        log "DONE: GPT-OSS-120B vLLM TP=2"
    else
        err "GPT-OSS-120B vLLM TP=2 also failed"; tail -10 /tmp/srv_gptoss120b_tp2_vl.log
    fi
fi

gpu_free
log "ALL GPT-OSS BENCHMARKS COMPLETE"
