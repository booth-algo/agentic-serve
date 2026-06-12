#!/usr/bin/env bash
# L13 S8 probe driver (runs ON the 3090 host): replay a GT multi-turn PROFILE
# concurrency ladder (terminalbench/swebench — the big-context profiles S7's
# chat-only probe did not cover) against a fresh GT-protocol vLLM server while
# the /metrics sidecar records the engine's prefix-cache token counters.
#
# Identical GT protocol to s7_replay_driver.sh (one server per profile ladder,
# ascending concs, VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1, vllm 0.19.0
# small-device defaults); the ONLY differences are the profile parameter and
# the s8_* file naming (one run dir per profile so the reducer's per-dir file
# layout is preserved).
#
# Round-3 purpose (L13-3090multi): (1) measure the tp4 BULK prefill drain rate
# on clean prefill-dominated windows (tb/swe outputs are ~10-50 tokens, so the
# burst windows are not decode-contaminated the way chat's 200-300-token decodes
# were — the S7 tp4 "0.45 ms/tok" medians at 2-20k bursts carry decode overlap);
# (2) measure rho / computed-volume truth for the big-context profiles.
#
# Usage: bash s8_replay_driver.sh TP GPU_IDS PROFILE "CONC LIST"
#   e.g. bash s8_replay_driver.sh 4 0,1,2,3 terminalbench-multiturn-synth "1 5 10 20 40 80 120"
set -uo pipefail

TP="${1:?tp}"
GPUS="${2:?gpu ids}"
PROFILE="${3:?profile, e.g. terminalbench-multiturn-synth}"
CONCS="${4:?conc list}"
SHORT="${PROFILE%%-*}"
PORT=8793
PY=/home/kevinlau/miniconda3/envs/vllm/bin/python
MODEL=/home/kevinlau/models/Llama-3.1-8B-Instruct
RUN="/home/kevinlau/m3090_run/l13/s8_${SHORT}_tp${TP}"
BENCH=/tmp/inference-benchmark   # the EXACT GT source (md5-identical to the repo, verified L13 S7)
API_KEY=test

export TMPDIR=/home/kevinlau/tmp
export XDG_CACHE_HOME=/home/kevinlau/tmp/.cache
export TIKTOKEN_CACHE_DIR=/home/kevinlau/tiktoken_cache
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
mkdir -p "$RUN" "$TMPDIR"

echo "[s8] tp=$TP gpus=$GPUS profile=$PROFILE concs=$CONCS $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$RUN/preflight_tp${TP}.txt"
if [ -s "$RUN/preflight_tp${TP}.txt" ]; then
    echo "[s8] ABORT: GPUs busy"; cat "$RUN/preflight_tp${TP}.txt"; exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --api-key "$API_KEY" \
    --enable-prefix-caching --enable-chunked-prefill \
    --tensor-parallel-size "$TP" --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --trust-remote-code \
    > "$RUN/vllm_s8_tp${TP}.log" 2>&1 &
SPID=$!
echo "[s8] server pid=$SPID"
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; true' EXIT

for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/v1/models" -H "Authorization: Bearer $API_KEY" >/dev/null 2>&1; then
        echo "[s8] server ready after ${i}x5s"; break
    fi
    if ! kill -0 $SPID 2>/dev/null; then
        echo "[s8] server died"; tail -30 "$RUN/vllm_s8_tp${TP}.log"; exit 1
    fi
    sleep 5
done
grep -E "KV cache size|Available KV cache|Maximum concurrency" "$RUN/vllm_s8_tp${TP}.log" \
    > "$RUN/s8_poolline_tp${TP}.txt" || true
cat "$RUN/s8_poolline_tp${TP}.txt"

"$PY" "$RUN/s7_metrics_sampler.py" --port "$PORT" \
    --out "$RUN/s8_metrics_tp${TP}.jsonl" --interval 0.2 &
MPID=$!
echo "[s8] sampler pid=$MPID"

cd "$BENCH"
for CONC in $CONCS; do
    echo "CELL_START tp${TP} ${PROFILE} ${CONC} $(date +%s.%N)" >> "$RUN/s8_cells_tp${TP}.log"
    OPENAI_API_KEY="$API_KEY" "$PY" -m src.benchmark.runner \
        --url "http://localhost:$PORT/v1/chat/completions" \
        --model "$MODEL" --backend vllm \
        --profile "$PROFILE" --concurrency "$CONC" \
        --mode multi-turn --num-requests 100 --warmup 3 --seed 42 --timeout 300 \
        --api-key "$API_KEY" --max-context-tokens 32768 \
        --max-model-len 32768 --gpu-memory-utilization 0.85 \
        --tensor-parallel-size "$TP" --prefix-caching-state on --chunked-prefill on \
        --output "$RUN/s8_${SHORT}_conc${CONC}_tp${TP}.json" \
        >> "$RUN/s8_runner_tp${TP}.log" 2>&1 \
        || echo "[s8] RUNNER FAILED conc=$CONC" >> "$RUN/s8_runner_tp${TP}.log"
    echo "CELL_END tp${TP} ${PROFILE} ${CONC} $(date +%s.%N)" >> "$RUN/s8_cells_tp${TP}.log"
done

kill $MPID 2>/dev/null; wait $MPID 2>/dev/null
kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
trap - EXIT
sleep 10
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader > "$RUN/postflight_tp${TP}.txt"
echo "[s8] postflight compute apps: $(wc -l < "$RUN/postflight_tp${TP}.txt")"
echo "[s8] DONE tp=$TP profile=$PROFILE $(date -u +%FT%TZ)"
