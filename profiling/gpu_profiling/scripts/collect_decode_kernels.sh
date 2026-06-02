#!/usr/bin/env bash
set -euo pipefail

# Collect NCU reports for the Experiment 2 decode-kernel microbenchmarks.
#
# Usage:
#   bash profiling/profile/scripts/collect_decode_kernels.sh smoke
#   bash profiling/profile/scripts/collect_decode_kernels.sh full
#   bash profiling/profile/scripts/collect_decode_kernels.sh gemm_full
#   bash profiling/profile/scripts/collect_decode_kernels.sh fused_full
#
# Environment overrides:
#   GPU_ID=6
#   TMP_ROOT=/data48/kevinlau/tmp
#   PYTHON_BIN=$HOME/miniconda3/envs/vllm/bin/python
#   NCU_BIN=/usr/local/cuda-12.1/bin/ncu
#   NCU_SET=full
#   MAX_TOTAL_KV_TOKENS=500000

MODE="${1:-smoke}"
if [[ "$MODE" != "smoke" && "$MODE" != "full" && \
      "$MODE" != "gemm_smoke" && "$MODE" != "gemm_full" && \
      "$MODE" != "fused_smoke" && "$MODE" != "fused_full" ]]; then
  echo "Usage: $0 [smoke|full|gemm_smoke|gemm_full|fused_smoke|fused_full]" >&2
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
RUNNER="${REPO_ROOT}/profiling/profile/scripts/run_vllm_profile.py"

DEFAULT_OUT_DIR="${REPO_ROOT}/profile_data/results"
DEFAULT_OUT_DIR="${DEFAULT_OUT_DIR}/ncu_decode_kernels_${GPU_LABEL}_${MODE}_${DATE_TAG}"
OUT_DIR="${OUT_DIR:-$DEFAULT_OUT_DIR}"
NCU_DIR="${OUT_DIR}/ncu"
LOG_DIR="${OUT_DIR}/logs"
NCU_EVENT_DIR="${OUT_DIR}/cuda_events_under_ncu"
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

if [[ "$MODE" == "smoke" || "$MODE" == "gemm_smoke" || "$MODE" == "fused_smoke" ]]; then
  GEMM_BATCHES=(1 32)
  GEMM_OPS=(qkv_fused o_proj gate_up_fused down_proj)
  FUSED_BATCHES=(1 32)
  FUSED_CONTEXTS=(512 8192)
  FUSED_KERNELS=(rms_norm silu_and_mul rotary_embedding)
else
  GEMM_BATCHES=(1 2 4 8 16 32 64 128 256)
  GEMM_OPS=(qkv_fused o_proj gate_up_fused down_proj)
  FUSED_BATCHES=(1 2 4 8 16 32 64 128 256)
  FUSED_CONTEXTS=(512 1024 2048 4096 8192 16384)
  FUSED_KERNELS=(rms_norm silu_and_mul rotary_embedding kv_cache_write sampling_topk)
fi

RUN_GEMM=1
RUN_FUSED=1
if [[ "$MODE" == gemm_* ]]; then
  RUN_FUSED=0
elif [[ "$MODE" == fused_* ]]; then
  RUN_GEMM=0
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

fused_shape_is_enabled() {
  local batch="$1"
  local context="$2"
  local kv_len=$((context + 1))
  local total_kv=$((batch * kv_len))
  [[ "$total_kv" -le "$MAX_TOTAL_KV_TOKENS" ]]
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

run_event_sanity() {
  echo "== CUDA-event sanity sweeps =="
  if [[ "$RUN_GEMM" == 1 ]]; then
    run_profile cuda-events decode-gemm \
      -- \
      --gpu-label "$GPU_LABEL" \
      --output "$OUT_DIR/decode_gemm_cuda_events.csv" \
      --batch-sizes "${GEMM_BATCHES[@]}" \
      --ops "${GEMM_OPS[@]}" \
      --warmups 5 \
      --repeats 10 \
      --inner-iters 10 \
      > "$LOG_DIR/decode_gemm_cuda_events.log" 2>&1
  fi

  if [[ "$RUN_FUSED" == 1 ]]; then
    run_profile cuda-events decode-fused-kernels \
      -- \
      --gpu-label "$GPU_LABEL" \
      --output "$OUT_DIR/decode_fused_kernels_cuda_events.csv" \
      --batch-sizes "${FUSED_BATCHES[@]}" \
      --context-lengths "${FUSED_CONTEXTS[@]}" \
      --kernels "${FUSED_KERNELS[@]}" \
      --max-total-kv-tokens "$MAX_TOTAL_KV_TOKENS" \
      --warmups 5 \
      --repeats 10 \
      --inner-iters 10 \
      > "$LOG_DIR/decode_fused_kernels_cuda_events.log" 2>&1
  fi
}

summarize_ncu_csvs() {
  echo "== Summarizing NCU raw CSVs =="
  if [[ "$RUN_GEMM" == 1 ]]; then
    run_env "$PYTHON_BIN" "$NCU_SUMMARIZER" \
      --kind gemm \
      --ncu-dir "$NCU_DIR" \
      --output "$OUT_DIR/decode_gemm_ncu_summary.csv"
  fi
  if [[ "$RUN_FUSED" == 1 ]]; then
    run_env "$PYTHON_BIN" "$NCU_SUMMARIZER" \
      --kind fused \
      --ncu-dir "$NCU_DIR" \
      --output "$OUT_DIR/decode_fused_kernels_ncu_summary.csv"
  fi
}

run_ncu_case() {
  local tag="$1"
  local target="$2"
  shift 2
  local report_base="${NCU_DIR}/${tag}"
  echo "== NCU ${tag} =="
  run_profile ncu "$target" \
    --ncu-output "$report_base" \
    --ncu-csv "${NCU_DIR}/${tag}.csv" \
    -- \
    "$@" \
    > "${LOG_DIR}/${tag}.ncu.stdout" 2> "${LOG_DIR}/${tag}.ncu.stderr"
}

run_ncu_gemm() {
  for batch in "${GEMM_BATCHES[@]}"; do
    for op in "${GEMM_OPS[@]}"; do
      run_ncu_case "gemm_${op}_B${batch}" decode-gemm \
        --gpu-label "$GPU_LABEL" \
        --output "$NCU_EVENT_DIR/gemm_${op}_B${batch}.csv" \
        --batch-sizes "$batch" \
        --ops "$op" \
        --warmups 1 \
        --repeats 1 \
        --inner-iters 1
    done
  done
}

run_ncu_fused() {
  for batch in "${FUSED_BATCHES[@]}"; do
    for context in "${FUSED_CONTEXTS[@]}"; do
      if ! fused_shape_is_enabled "$batch" "$context"; then
        echo "SKIP fused_B${batch}_T${context} total_kv>$MAX_TOTAL_KV_TOKENS"
        continue
      fi
      for kernel in "${FUSED_KERNELS[@]}"; do
        run_ncu_case "fused_${kernel}_B${batch}_T${context}" decode-fused-kernels \
          --gpu-label "$GPU_LABEL" \
          --output "$NCU_EVENT_DIR/fused_${kernel}_B${batch}_T${context}.csv" \
          --batch-sizes "$batch" \
          --context-lengths "$context" \
          --kernels "$kernel" \
          --max-total-kv-tokens "$MAX_TOTAL_KV_TOKENS" \
          --warmups 1 \
          --repeats 1 \
          --inner-iters 1
      done
    done
  done
}

echo "Experiment 2 NCU decode-kernel collection"
echo "  mode:      $MODE"
echo "  gpu:       $GPU_LABEL CUDA_VISIBLE_DEVICES=$GPU_ID"
echo "  python:    $PYTHON_BIN"
echo "  ncu:       $NCU_BIN --set $NCU_SET"
echo "  tmp:       $TMP_ROOT"
echo "  max kv:    $MAX_TOTAL_KV_TOKENS"
echo "  output:    $OUT_DIR"

run_event_sanity
if [[ "$RUN_GEMM" == 1 ]]; then
  run_ncu_gemm
fi
if [[ "$RUN_FUSED" == 1 ]]; then
  run_ncu_fused
fi
if [[ "$RUN_GEMM" == 1 || "$RUN_FUSED" == 1 ]]; then
  summarize_ncu_csvs
fi

echo "Done. Outputs:"
echo "  $OUT_DIR"
