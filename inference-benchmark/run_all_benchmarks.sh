#!/usr/bin/env bash
# =============================================================================
# Master Benchmark Orchestrator — All models × SGLang + vLLM
# Runs production profiles with high concurrency sweep to find TPOT saturation.
# Uses GPU parallelism: TP=1 runs 4 models at once, TP=2 runs 2, TP=4 runs 1.
#
# Usage: ./run_all_benchmarks.sh [sglang|vllm|all]
#   Default: all (SGLang first, then vLLM)
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$(which python)}"
API_KEY="test"
WARMUP=5
TIMEOUT=300
CONC_SWEEP="1 10 20 40 80 120 160 200 256 320"
CONC_SWEEP_LOW="1 10 20 40 80"  # For memory-constrained configs (tight KV budget)
PROFILES="chat-short chat-medium chat-long coding-agent prefill-heavy decode-heavy random-1k"
MAX_SERVER_WAIT=1800  # 30 min for large MoE models

# Source the GPU scheduler
source "$REPO_ROOT/scripts/gpu_scheduler.sh"

log() { echo -e "\033[0;32m[ORCH]\033[0m $1"; }
err() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }

# =============================================================================
# Model registry: name|path|tp_list|sglang_extra|vllm_extra|max_model_len|gpu_mem
# =============================================================================
declare -a MODELS=(
    # --- Small dense (TP=1,2) ---
    "Llama-3.1-8B|/workspace/models/Llama-3.1-8B-Instruct|1,2||--enable-chunked-prefill|32768|0.80"
    "Qwen3.5-9B|/workspace/models/Qwen3.5-9B|1,2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code --gdn-prefill-backend triton|32768|0.80"
    # --- Small MoE (TP=1,2) ---
    "gpt-oss-20b|/workspace/models/gpt-oss-20b|1,2||--enable-chunked-prefill|32768|0.80"
    # --- Medium dense (TP=2) ---
    "Qwen3.5-27B|/workspace/models/Qwen3.5-27B|2|--trust-remote-code --disable-overlap-schedule|--enable-chunked-prefill --trust-remote-code --gdn-prefill-backend triton|16384|0.92"
    # --- Medium MoE (TP=2 only) ---
    "gpt-oss-120b|/workspace/models/gpt-oss-120b|2||--enable-chunked-prefill|32768|0.80"
    # --- Large dense (TP=4 safe, TP=2 low-conc only) ---
    "Qwen2.5-72B|/workspace/models/Qwen2.5-72B-Instruct|4,2|--trust-remote-code --disable-piecewise-cuda-graph|--enable-chunked-prefill --trust-remote-code|4096|0.90"
    "Llama-3.1-70B|/workspace/models/Llama-3.1-70B-Instruct|4,2|--disable-piecewise-cuda-graph|--enable-chunked-prefill|4096|0.90"
    "Llama-3.3-70B|/workspace/models/Llama-3.3-70B-Instruct|4,2|--disable-piecewise-cuda-graph|--enable-chunked-prefill|4096|0.90"
)

model_exists() {
    local path="$1"
    [[ -d "$path" ]] && [[ -f "$path/config.json" ]] && ls "$path"/*.safetensors &>/dev/null
}

# =============================================================================
# Benchmark callback — called by scheduler per model on its assigned port
# Args: engine, name, path, tp, port, slot_idx
# =============================================================================
bench_callback() {
    local engine="$1" name="$2" path="$3" tp="$4" port="$5" slot_idx="$6"

    local tag="${name}_tp${tp}_${engine}"
    local run_date=$(date +%Y-%m-%d)
    local results_dir="results/${tag}/${run_date}"
    mkdir -p "$results_dir"

    local url="http://localhost:${port}/v1/chat/completions"

    # Select concurrency sweep
    local conc_list="$CONC_SWEEP"
    case "$name" in
        Llama-3.1-70B|Llama-3.3-70B|Qwen2.5-72B)
            if [[ "$tp" -le 2 ]]; then
                conc_list="$CONC_SWEEP_LOW"
                slot_log "$slot_idx" "Low concurrency sweep for $name TP=$tp (tight VRAM)"
            fi
            ;;
        Qwen3.5-27B)
            if [[ "$tp" -le 2 ]]; then
                conc_list="$CONC_SWEEP_LOW"
                slot_log "$slot_idx" "Low concurrency sweep for $name TP=$tp (gpu_mem=0.92, memory leak risk)"
            fi
            ;;
        gpt-oss-120b)
            conc_list="$CONC_SWEEP_LOW"
            slot_log "$slot_idx" "Low concurrency sweep for $name (large MoE, memory leak risk)"
            ;;
    esac

    for PROFILE in $PROFILES; do
        slot_log "$slot_idx" "━━━ $name TP=$tp: $PROFILE ━━━"
        for CONC in $conc_list; do
            # nreq must always exceed concurrency to ensure full load pressure
            NREQ=$(( CONC * 2 ))
            [[ "$NREQ" -lt 200 ]] && NREQ=200
            [[ "$CONC" -eq 1 ]] && NREQ=50

            local out="${results_dir}/${PROFILE}_conc${CONC}.json"

            # Resume support: skip valid existing results (check ALL date folders)
            local skip_run=false
            for existing_dir in results/${tag}/*/; do
                local existing_file="${existing_dir}${PROFILE}_conc${CONC}.json"
                if [[ -f "$existing_file" ]] && [[ -s "$existing_file" ]]; then
                    completed=$("$PYTHON" -c "import json; d=json.load(open('$existing_file')); s=d.get('summary',{}); print(s.get('successful_requests', d.get('num_requests_completed', d.get('completed_requests',0))))" 2>/dev/null || echo "0")
                    if [[ "$completed" -gt 0 ]]; then
                        slot_log "$slot_idx" "  SKIP conc=$CONC (exists in $(basename "$existing_dir"), $completed reqs)"
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
                --url        "$url" \
                --model      "$path" \
                --backend    "$engine" \
                --profile    "$PROFILE" \
                --concurrency "$CONC" \
                --num-requests "$NREQ" \
                --warmup     "$WARMUP" \
                --timeout    "$TIMEOUT" \
                --api-key    "$API_KEY" \
                --output     "$out" \
                2>&1 || {
                    slot_log "$slot_idx" "  FAILED: $PROFILE conc=$CONC"
                }
        done
    done

    slot_log "$slot_idx" "Complete: $tag ($(find "$results_dir" -name "*.json" 2>/dev/null | wc -l) files)"
}

# =============================================================================
# Build job lists per TP level, applying engine-specific skips
# =============================================================================
run_engine() {
    local engine="$1"
    log ""
    log "╔══════════════════════════════════════════════════════════╗"
    log "║  Engine: $(printf '%-47s' "$engine")  ║"
    log "╚══════════════════════════════════════════════════════════╝"

    local -a tp1_jobs=()
    local -a tp2_jobs=()
    local -a tp4_jobs=()

    for model_spec in "${MODELS[@]}"; do
        IFS='|' read -r name path tp_list sglang_extra vllm_extra max_len gpu_mem <<< "$model_spec"

        if ! model_exists "$path"; then
            warn "SKIP $name — not found at $path"
            continue
        fi

        # SGLang: skip MXFP4 models
        if [[ "$engine" == "sglang" ]]; then
            case "$name" in
                gpt-oss-20b|gpt-oss-120b)
                    warn "SKIP $name on SGLang — MXFP4 tinygemm kernel incompatible"
                    continue
                    ;;
            esac
        fi

        # Job format for scheduler: name|path|sglang_extra|vllm_extra|max_len|gpu_mem
        local job="${name}|${path}|${sglang_extra}|${vllm_extra}|${max_len}|${gpu_mem}"

        IFS=',' read -ra tps <<< "$tp_list"
        for tp in "${tps[@]}"; do
            # SGLang: skip TP=4 for 70B+ (CUDA crash in rc build)
            if [[ "$engine" == "sglang" && "$tp" -ge 4 ]]; then
                case "$name" in
                    Qwen2.5-72B|Llama-3.1-70B|Llama-3.3-70B)
                        warn "SKIP $name TP=$tp on SGLang — CUDA crash in 0.5.10rc0"
                        continue
                        ;;
                esac
            fi

            case "$tp" in
                1) tp1_jobs+=("$job") ;;
                2) tp2_jobs+=("$job") ;;
                4) tp4_jobs+=("$job") ;;
            esac
        done
    done

    log "Job counts: TP=1: ${#tp1_jobs[@]}, TP=2: ${#tp2_jobs[@]}, TP=4: ${#tp4_jobs[@]}"

    # Run batches: TP=1 first (most parallelism), then TP=2, then TP=4
    if [[ ${#tp1_jobs[@]} -gt 0 ]]; then
        run_parallel_batch 1 "$engine" "bench_callback" "${tp1_jobs[@]}"
    fi
    if [[ ${#tp2_jobs[@]} -gt 0 ]]; then
        run_parallel_batch 2 "$engine" "bench_callback" "${tp2_jobs[@]}"
    fi
    if [[ ${#tp4_jobs[@]} -gt 0 ]]; then
        run_parallel_batch 4 "$engine" "bench_callback" "${tp4_jobs[@]}"
    fi
}

# =============================================================================
# Main
# =============================================================================
TARGET="${1:-all}"

log "╔══════════════════════════════════════════════════════════╗"
log "║  MASTER BENCHMARK ORCHESTRATOR (PARALLEL)               ║"
log "║  4x NVIDIA H100 80GB — $(date +%B\ %Y)                     ║"
log "║  $(date)                        ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
log "Target:      $TARGET"
log "Concurrency: $CONC_SWEEP"
log "Profiles:    $PROFILES"
log ""

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

case "$TARGET" in
    sglang) run_engine "sglang" ;;
    vllm)   run_engine "vllm" ;;
    all)    run_engine "sglang"; run_engine "vllm" ;;
    *)      echo "Usage: $0 [sglang|vllm|all]"; exit 1 ;;
esac

log ""
log "╔══════════════════════════════════════════════════════════╗"
log "║  ALL BENCHMARKS COMPLETE                                ║"
log "╚══════════════════════════════════════════════════════════╝"
log ""
log "Results summary:"
for d in results/*_sglang results/*_vllm; do
    [[ -d "$d" ]] || continue
    count=$(find "$d" -name "*.json" 2>/dev/null | wc -l)
    log "  $(basename "$d"): $count files"
done
