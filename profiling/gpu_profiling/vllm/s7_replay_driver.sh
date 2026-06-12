#!/usr/bin/env bash
# L13 S7 probe driver (runs ON the 3090 host): replay the GT chat-multiturn-synth
# concurrency ladder against a fresh GT-protocol vLLM server while a /metrics
# sidecar records the engine's prefix-cache token counters.
#
# GT protocol replicated from inference-benchmark/scripts/sweep_multiturn_profiles.sh
# (the launcher that produced results/synthetic_distributional/3090_Llama-3.1-8B_tp{2,4}_vllm):
#   * VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 exported
#   * one server reused across the ascending concurrency ladder (cache NOT reset
#     between cells - the cross-cell prefix reuse is part of GT)
#   * server flags: --api-key test --enable-prefix-caching --enable-chunked-prefill
#     --tensor-parallel-size TP --gpu-memory-utilization 0.85 --max-model-len 32768
#     --trust-remote-code   (defaults otherwise: vllm 0.19.0 small-device
#     max_num_batched_tokens=2048 / max_num_seqs=256)
#   * runner flags: --mode multi-turn --warmup 3 --seed 42 --num-requests 100
#     --timeout 300 --max-context-tokens 32768 (harness defaults: arrival steady,
#     context-safety-margin 256, prefix block 16)
#
# Usage: bash s7_replay_driver.sh TP GPU_IDS "CONC LIST"
#   e.g. bash s7_replay_driver.sh 4 0,1,2,3 "1 5 10 20 40 80 120"
set -uo pipefail

TP="${1:?tp}"
GPUS="${2:?gpu ids}"
CONCS="${3:?conc list}"
PORT=8793
PY=/home/kevinlau/miniconda3/envs/vllm/bin/python
MODEL=/home/kevinlau/models/Llama-3.1-8B-Instruct
RUN=/home/kevinlau/m3090_run/l13/s7
BENCH=/tmp/inference-benchmark   # the EXACT GT source (sweep_multiturn_profiles.sh cd's here; the
                                 # ~/agentic-serve checkout is older and lacks --max-context-tokens)
API_KEY=test

export TMPDIR=/home/kevinlau/tmp
export XDG_CACHE_HOME=/home/kevinlau/tmp/.cache
export TIKTOKEN_CACHE_DIR=/home/kevinlau/tiktoken_cache
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
mkdir -p "$RUN" "$TMPDIR"

echo "[s7] tp=$TP gpus=$GPUS concs=$CONCS $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$RUN/preflight_tp${TP}.txt"
if [ -s "$RUN/preflight_tp${TP}.txt" ]; then
    echo "[s7] ABORT: GPUs busy"; cat "$RUN/preflight_tp${TP}.txt"; exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --api-key "$API_KEY" \
    --enable-prefix-caching --enable-chunked-prefill \
    --tensor-parallel-size "$TP" --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --trust-remote-code \
    > "$RUN/vllm_s7_tp${TP}.log" 2>&1 &
SPID=$!
echo "[s7] server pid=$SPID"
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; true' EXIT

for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
        echo "[s7] server ready after ${i}x5s"; break
    fi
    if ! kill -0 $SPID 2>/dev/null; then
        echo "[s7] server died"; tail -30 "$RUN/vllm_s7_tp${TP}.log"; exit 1
    fi
    sleep 5
done
grep -E "KV cache size|Available KV cache|Maximum concurrency" "$RUN/vllm_s7_tp${TP}.log" \
    > "$RUN/s7_poolline_tp${TP}.txt" || true
cat "$RUN/s7_poolline_tp${TP}.txt"

"$PY" "$RUN/s7_metrics_sampler.py" --port "$PORT" \
    --out "$RUN/s7_metrics_tp${TP}.jsonl" --interval 0.2 &
MPID=$!
echo "[s7] sampler pid=$MPID"

cd "$BENCH"
for CONC in $CONCS; do
    echo "CELL_START tp${TP} chat-multiturn-synth ${CONC} $(date +%s.%N)" >> "$RUN/s7_cells_tp${TP}.log"
    OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
        --url "http://localhost:$PORT/v1/chat/completions" \
        --model "$MODEL" --backend vllm \
        --profile chat-multiturn-synth --concurrency "$CONC" \
        --mode multi-turn --num-requests 100 --warmup 3 --seed 42 --timeout 300 \
        --api-key "$API_KEY" --max-context-tokens 32768 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --tensor-parallel-size "$TP" --prefix-caching-state on --chunked-prefill on \
        --output "$RUN/s7_chat_conc${CONC}_tp${TP}.json" \
        >> "$RUN/s7_runner_tp${TP}.log" 2>&1 \
        || echo "[s7] RUNNER FAILED conc=$CONC" >> "$RUN/s7_runner_tp${TP}.log"
    echo "CELL_END tp${TP} chat-multiturn-synth ${CONC} $(date +%s.%N)" >> "$RUN/s7_cells_tp${TP}.log"
done

kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
trap - EXIT
sleep 10
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$RUN/postflight_tp${TP}.txt"
echo "[s7] postflight compute apps: $(wc -l < "$RUN/postflight_tp${TP}.txt")"
echo "[s7] DONE tp=$TP $(date -u +%FT%TZ)"
