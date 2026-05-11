#!/usr/bin/env bash
# Rebuild the private dashboard from the durable local benchmark store.
#
# Defaults:
#   raw results: /mnt/100g/agent-bench/results
#   state root:  /mnt/100g/agent-bench/state
#   JSON base:   /agentic-serve
#
# This is the freshness path for the Tailscale dashboard. R2 mirroring is
# optional and happens only after local artifacts validate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_DIR="$BENCH_ROOT/dashboard"

RESULTS_DIR="${BENCHMARK_RESULTS_DIR:-/mnt/100g/agent-bench/results}"
STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
JSON_BASE="${DASHBOARD_JSON_BASE:-/agentic-serve}"
LIVE_DIST="$DASHBOARD_DIR/dist"
NEXT_DIST="${DASHBOARD_NEXT_DIST:-$DASHBOARD_DIR/dist.next}"
PREV_DIST="${DASHBOARD_PREV_DIST:-$DASHBOARD_DIR/dist.prev}"
GPU_STATE_OUT="${GPU_STATE_OUT:-$NEXT_DIST/gpu-state.json}"
GPU_STATE_REPORT="${GPU_STATE_REPORT:-/tmp/agentic-serve-gpu-state-latest.md}"
GPU_STATE_SSH_TIMEOUT="${GPU_STATE_SSH_TIMEOUT:-12}"
GPU_STATE_HOSTS="${GPU_STATE_HOSTS:-}"

ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
MIRROR_R2="${MIRROR_R2:-0}"

usage() {
    sed -n '1,13p' "$0"
    cat <<'EOF'

Options:
  --results-dir PATH   Raw benchmark results root
  --state-root PATH    Orchestrator state root
  --json-base PATH     Dashboard JSON base URL (default: /agentic-serve)
  --mirror-r2          Best-effort upload of validated JSON artifacts to R2
  --no-mirror-r2       Disable R2 JSON artifact upload
EOF
}

require_option_value() {
    if [[ $# -lt 2 ]]; then
        echo "missing value for $1" >&2
        usage >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            require_option_value "$@"
            RESULTS_DIR="$2"
            shift 2
            ;;
        --state-root)
            require_option_value "$@"
            STATE_ROOT="$2"
            shift 2
            ;;
        --json-base)
            require_option_value "$@"
            JSON_BASE="$2"
            shift 2
            ;;
        --mirror-r2)
            MIRROR_R2=1
            shift
            ;;
        --no-mirror-r2)
            MIRROR_R2=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "missing results dir: $RESULTS_DIR" >&2
    exit 1
fi

mkdir -p "$STATE_ROOT" "$DASHBOARD_DIR/public"

echo "Building sweep-state.json from $STATE_ROOT"
python3 "$SCRIPT_DIR/publish_sweep_state.py" \
    --state-dir "$STATE_ROOT" \
    --out "$DASHBOARD_DIR/public/sweep-state.json" \
    --no-upload

echo "Building data.json from $RESULTS_DIR"
(
    cd "$DASHBOARD_DIR"
    BENCHMARK_RESULTS_DIR="$RESULTS_DIR" npm run build:data
)

echo "Validating data.json"
(
    cd "$DASHBOARD_DIR"
    SWEEP_STATE_PATH="$DASHBOARD_DIR/public/sweep-state.json" npm run validate:data
)

echo "Building local dashboard bundle with JSON base $JSON_BASE into $NEXT_DIST"
rm -rf "$NEXT_DIST"
(
    cd "$DASHBOARD_DIR"
    VITE_R2_JSON_BASE="$JSON_BASE" npm run build -- --outDir "$NEXT_DIST" --emptyOutDir
)

echo "Building private gpu-state.json from $STATE_ROOT"
gpu_state_args=(
    --jobs-config "$SCRIPT_DIR/sweep.yaml"
    --scope "${BENCH_JOBS_SCOPE:-synthetic_distributional}"
    --state-dir "$STATE_ROOT"
    --ssh-timeout "$GPU_STATE_SSH_TIMEOUT"
    --out "$GPU_STATE_REPORT"
    --json-out "$GPU_STATE_OUT"
    --once
)
if [[ -n "$GPU_STATE_HOSTS" ]]; then
    IFS=', ' read -r -a gpu_state_hosts <<< "$GPU_STATE_HOSTS"
    gpu_state_args+=(--hosts "${gpu_state_hosts[@]}")
fi
python3 "$SCRIPT_DIR/sweep_progress_report.py" "${gpu_state_args[@]}"

echo "Promoting rebuilt dashboard bundle to $LIVE_DIST"
rm -rf "$PREV_DIST"
if [[ -d "$LIVE_DIST" ]]; then
    mv "$LIVE_DIST" "$PREV_DIST"
fi
mv "$NEXT_DIST" "$LIVE_DIST"
rm -rf "$PREV_DIST"

if [[ "$MIRROR_R2" == "1" ]]; then
    if command -v aws >/dev/null 2>&1; then
        echo "Mirroring validated dashboard JSON artifacts to R2"
        # gpu-state.json intentionally stays local/private: it includes host,
        # user, port, and process occupancy details for the Tailscale dashboard.
        for artifact in \
            data.json \
            data.trace_replay.json \
            data.synthetic_distributional.json \
            data.archived.json \
            sweep-state.json \
            gemm-eval.json \
            serving-predictions.json \
            profiling-state.json \
            predictor-coverage.json \
            roofline-data.json \
            roofline-quadrant.json \
            gemm-extrapolation.json
        do
            path="$DASHBOARD_DIR/public/$artifact"
            if [[ -f "$path" ]]; then
                aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 cp \
                    "$path" "s3://$BUCKET/json/current/$artifact" \
                    --only-show-errors || echo "warning: failed to mirror $artifact" >&2
            fi
        done
    else
        echo "warning: aws cli not found; skipping R2 mirror" >&2
    fi
fi

echo "Local dashboard rebuild complete"
