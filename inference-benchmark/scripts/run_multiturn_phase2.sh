#!/usr/bin/env bash
# Multi-Turn Phase 2: Fill gaps for xl/long profiles and missing models
# Uses GPU parallelism: TP=1 runs 4 models, TP=2 runs 2
set -uo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
API_KEY="test"
WARMUP=5
TIMEOUT=300
CONC_SWEEP="5 10 20 40 80"
CONC_SWEEP_HIGH="5 10 20 40 80 120 160"

source scripts/gpu_scheduler.sh

log() { echo -e "\033[0;32m[MT2]\033[0m $1"; }

# Multi-turn profiles to run
MT_PROFILES="chat-multiturn-short chat-multiturn-medium chat-multiturn-long chat-multiturn-xl swebench-multiturn-short swebench-multiturn-medium terminalbench-multiturn-short terminalbench-multiturn-medium"

# Models that need multi-turn runs
declare -a MODELS=(
    # TP=1 models needing multi-turn
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1||--enable-chunked-prefill|32768|0.80"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code --gdn-prefill-backend triton|32768|0.80"
    # TP=2 models
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code --gdn-prefill-backend triton|16384|0.92"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|2||--enable-chunked-prefill|4096|0.90"
    "Llama-3.3-70B|/workspace/models/Llama-3.3-70B-Instruct|2||--enable-chunked-prefill|4096|0.90"
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|2|--trust-remote-code|--enable-chunked-prefill --trust-remote-code|4096|0.90"
)

bench_callback() {
    local engine="$1" name="$2" path="$3" tp="$4" port="$5" slot_idx="$6"
    local tag="${name}_tp${tp}_${engine}"
    local run_date=$(date +%Y-%m-%d)
    local results_dir="results/${tag}/${run_date}"
    mkdir -p "$results_dir"
    local url="http://localhost:${port}/v1/chat/completions"

    # Select concurrency sweep
    local conc_list="$CONC_SWEEP_HIGH"
    case "$name" in
        Llama-3.1-70B|Llama-3.3-70B|Qwen2.5-72B|Qwen3.5-27B)
            conc_list="$CONC_SWEEP"
            ;;
    esac

    for PROFILE in $MT_PROFILES; do
        slot_log "$slot_idx" "━━━ $name TP=$tp: $PROFILE ━━━"
        for CONC in $conc_list; do
            # nreq must exceed concurrency
            NREQ=$(( CONC * 2 ))
            [[ "$NREQ" -lt 200 ]] && NREQ=200
            [[ "$CONC" -eq 1 ]] && NREQ=50

            local out="${results_dir}/${PROFILE}_conc${CONC}.json"

            # Resume: check ALL date folders
            local skip_run=false
            for existing_dir in results/${tag}/*/; do
                local existing_file="${existing_dir}${PROFILE}_conc${CONC}.json"
                if [[ -f "$existing_file" ]] && [[ -s "$existing_file" ]]; then
                    completed=$("$PYTHON" -c "import json; d=json.load(open('$existing_file')); s=d.get('summary',{}); print(s.get('successful_requests', d.get('num_requests_completed', d.get('completed_requests',0))))" 2>/dev/null || echo "0")
                    if [[ "$completed" -gt 0 ]]; then
                        slot_log "$slot_idx" "  SKIP $PROFILE conc=$CONC (exists in $(basename "$existing_dir"), $completed reqs)"
                        skip_run=true
                        break
                    fi
                fi
            done
            if [[ "$skip_run" == "true" ]]; then
                continue
            fi

            slot_log "$slot_idx" "  $PROFILE conc=$CONC nreq=$NREQ"
            OPENAI_API_KEY="$API_KEY" "$PYTHON" -m src.benchmark.runner \
                --url "$url" \
                --model "$path" \
                --backend "$engine" \
                --profile "$PROFILE" \
                --concurrency "$CONC" \
                --num-requests "$NREQ" \
                --warmup "$WARMUP" \
                --timeout "$TIMEOUT" \
                --api-key "$API_KEY" \
                --output "$out" \
                2>&1 || slot_log "$slot_idx" "  FAILED $PROFILE conc=$CONC"
        done
    done
    slot_log "$slot_idx" "Complete: $tag"
}

log "╔══════════════════════════════════════════════════════════╗"
log "║  MULTI-TURN PHASE 2 — Fill xl/long gaps                 ║"
log "║  Profiles: $MT_PROFILES"
log "╚══════════════════════════════════════════════════════════╝"

# Build job lists by TP
local -a tp1_jobs=() 2>/dev/null || declare -a tp1_jobs=()
local -a tp2_jobs=() 2>/dev/null || declare -a tp2_jobs=()

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r name path tp sglang_extra vllm_extra max_len gpu_mem <<< "$model_spec"
    if ! [[ -d "$path" ]]; then
        log "SKIP $name — not found at $path"
        continue
    fi
    local job="${name}|${path}|${sglang_extra}|${vllm_extra}|${max_len}|${gpu_mem}" 2>/dev/null || job="${name}|${path}|${sglang_extra}|${vllm_extra}|${max_len}|${gpu_mem}"
    case "$tp" in
        1) tp1_jobs+=("$job") ;;
        2) tp2_jobs+=("$job") ;;
    esac
done

log "Jobs: TP=1: ${#tp1_jobs[@]}, TP=2: ${#tp2_jobs[@]}"

if [[ ${#tp1_jobs[@]} -gt 0 ]]; then
    run_parallel_batch 1 "vllm" "bench_callback" "${tp1_jobs[@]}"
fi
if [[ ${#tp2_jobs[@]} -gt 0 ]]; then
    run_parallel_batch 2 "vllm" "bench_callback" "${tp2_jobs[@]}"
fi

log "╔══════════════════════════════════════════════════════════╗"
log "║  MULTI-TURN PHASE 2 COMPLETE                            ║"
log "╚══════════════════════════════════════════════════════════╝"
