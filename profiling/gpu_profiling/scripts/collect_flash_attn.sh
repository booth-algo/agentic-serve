#!/usr/bin/env bash
set -euo pipefail

# Collect NCU reports for the vLLM FlashAttention decode-shape sweep.
#
# Usage:
#   bash profiling/profile/scripts/collect_flash_attn.sh smoke
#   bash profiling/profile/scripts/collect_flash_attn.sh full
#
# Environment overrides:
#   GPU_ID=6
#   TMP_ROOT=/data48/kevinlau/tmp
#   PYTHON_BIN=$HOME/miniconda3/envs/vllm/bin/python
#   NCU_BIN=/usr/local/cuda-12.1/bin/ncu
#   NCU_SET=full
#   MAX_TOTAL_KV_TOKENS=500000
#   MAX_ALLOC_GB=32

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" ]]; then
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

GPU_LABEL="${GPU_LABEL:-H100}"
GPU_ID="${GPU_ID:-6}"
TMP_ROOT="${TMP_ROOT:-/data48/kevinlau/tmp}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/miniconda3/envs/vllm/bin/python}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda-12.1/bin/ncu}"
NCU_SET="${NCU_SET:-full}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
MAX_TOTAL_KV_TOKENS="${MAX_TOTAL_KV_TOKENS:-500000}"
MAX_ALLOC_GB="${MAX_ALLOC_GB:-32}"
EVENT_WARMUPS="${EVENT_WARMUPS:-10}"
EVENT_REPEATS="${EVENT_REPEATS:-30}"
EVENT_INNER_ITERS="${EVENT_INNER_ITERS:-10}"
NCU_WARMUPS="${NCU_WARMUPS:-1}"
NCU_REPEATS="${NCU_REPEATS:-1}"
NCU_INNER_ITERS="${NCU_INNER_ITERS:-1}"
RUNNER="${REPO_ROOT}/profiling/profile/scripts/run_vllm_profile.py"

DEFAULT_OUT_DIR="${REPO_ROOT}/profile_data/results"
DEFAULT_OUT_DIR="${DEFAULT_OUT_DIR}/ncu_flash_attention_${GPU_LABEL}_${MODE}_${DATE_TAG}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
NCU_DIR="${OUT_DIR}/ncu"
LOG_DIR="${OUT_DIR}/logs"
NCU_EVENT_DIR="${OUT_DIR}/cuda_events_under_ncu"
CUDA_EVENT_SWEEP="${OUT_DIR}/flash_attention_cuda_events.csv"
SUMMARY_CSV="${OUT_DIR}/flash_attention_ncu_summary.csv"
DEFAULT_NCU_SUMMARIZER="${REPO_ROOT}/profiling/process"
DEFAULT_NCU_SUMMARIZER="${DEFAULT_NCU_SUMMARIZER}/summarize_ncu_decode_kernel_reports.py"
NCU_SUMMARIZER="${NCU_SUMMARIZER:-$DEFAULT_NCU_SUMMARIZER}"
mkdir -p "$OUT_DIR" "$NCU_DIR" "$LOG_DIR" "$NCU_EVENT_DIR" "$TMP_ROOT" "$TMP_ROOT/.cache"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -x "$NCU_BIN" ]]; then
  if [[ -x /usr/local/cuda/bin/ncu ]]; then
    NCU_BIN=/usr/local/cuda/bin/ncu
  else
    echo "NCU not executable: $NCU_BIN" >&2
    exit 1
  fi
fi

if [[ "$MODE" == "smoke" ]]; then
  FLASH_BATCHES=(1)
  FLASH_CONTEXTS=(512)
else
  FLASH_BATCHES=(1 2 4 8 16 32 64 128 256)
  FLASH_CONTEXTS=(512 1024 2048 4096 8192 16384)
fi

run_env() {
  env \
    TMPDIR="$TMP_ROOT" \
    XDG_CACHE_HOME="$TMP_ROOT/.cache" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    "$@"
}

run_profile() {
  local source="$1"
  local target="$2"
  shift 2
  run_env "$PYTHON_BIN" "$RUNNER" \
    --source "$source" \
    --target "$target" \
    --python-bin "$PYTHON_BIN" \
    --gpu-id "$GPU_ID" \
    --tmp-root "$TMP_ROOT" \
    --ncu-bin "$NCU_BIN" \
    --ncu-set "$NCU_SET" \
    "$@"
}

shape_is_enabled() {
  local batch="$1"
  local context="$2"
  local kv_len=$((context + 1))
  local total_kv=$((batch * kv_len))
  [[ "$total_kv" -le "$MAX_TOTAL_KV_TOKENS" ]]
}

run_event_sweep() {
  echo "== Standalone CUDA-event flash sweep =="
  run_profile cuda-events flash-attn \
    -- \
    --gpu-label "$GPU_LABEL" \
    --output "$CUDA_EVENT_SWEEP" \
    --batch-sizes "${FLASH_BATCHES[@]}" \
    --context-lengths "${FLASH_CONTEXTS[@]}" \
    --warmups "$EVENT_WARMUPS" \
    --repeats "$EVENT_REPEATS" \
    --inner-iters "$EVENT_INNER_ITERS" \
    --max-total-kv-tokens "$MAX_TOTAL_KV_TOKENS" \
    --max-alloc-gb "$MAX_ALLOC_GB" \
    > "$LOG_DIR/flash_attention_cuda_events.log" 2>&1
}

run_ncu_case() {
  local batch="$1"
  local context="$2"
  local tag="flash_attn_B${batch}_T${context}"
  local report_base="${NCU_DIR}/${tag}"
  echo "== NCU ${tag} =="
  run_profile ncu flash-attn \
    --ncu-output "$report_base" \
    --ncu-csv "${NCU_DIR}/${tag}.csv" \
    -- \
    --gpu-label "$GPU_LABEL" \
    --output "$NCU_EVENT_DIR/${tag}.csv" \
    --batch-sizes "$batch" \
    --context-lengths "$context" \
    --warmups "$NCU_WARMUPS" \
    --repeats "$NCU_REPEATS" \
    --inner-iters "$NCU_INNER_ITERS" \
    --max-total-kv-tokens "$MAX_TOTAL_KV_TOKENS" \
    --max-alloc-gb "$MAX_ALLOC_GB" \
    > "${LOG_DIR}/${tag}.ncu.stdout" 2> "${LOG_DIR}/${tag}.ncu.stderr"
}

run_ncu_sweep() {
  local count=0
  for batch in "${FLASH_BATCHES[@]}"; do
    for context in "${FLASH_CONTEXTS[@]}"; do
      if ! shape_is_enabled "$batch" "$context"; then
        echo "SKIP flash_attn_B${batch}_T${context} total_kv>$MAX_TOTAL_KV_TOKENS"
        continue
      fi
      count=$((count + 1))
      run_ncu_case "$batch" "$context"
    done
  done
  echo "Captured $count NCU flash-attention reports."
}

summarize_ncu_csvs() {
  echo "== Summarizing flash NCU raw CSVs =="
  run_env "$PYTHON_BIN" "$NCU_SUMMARIZER" \
    --kind flash \
    --ncu-dir "$NCU_DIR" \
    --output "$SUMMARY_CSV" \
    --cuda-event-flash "$CUDA_EVENT_SWEEP"
}

echo "FlashAttention NCU collection"
echo "  mode:      $MODE"
echo "  gpu:       $GPU_LABEL CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "  python:    $PYTHON_BIN"
echo "  ncu:       $NCU_BIN --set $NCU_SET"
echo "  tmp:       $TMP_ROOT"
echo "  max kv:    $MAX_TOTAL_KV_TOKENS"
echo "  output:    $OUT_DIR"

run_event_sweep
run_ncu_sweep
summarize_ncu_csvs

echo "Done. Outputs:"
echo "  $OUT_DIR"
