#!/usr/bin/env bash
# Smoke-test the systemd GPU orchestrator entrypoint without remote dispatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

JOBS_FILE="$TMP_DIR/bench_jobs.txt"
STATE_ROOT="$TMP_DIR/state"
RESULTS_ROOT="$TMP_DIR/results"
LOG="$TMP_DIR/orchestrator.log"

cat > "$JOBS_FILE" <<'EOF'
# Benchmark job matrix consumed by bench_orchestrator.sh.
# SCOPE: synthetic_distributional
# Format: HOST|MODEL_PATH|TP|SHORT|MODE|BACKEND|MAX_LEN|GPU_MEM|CONCS|PROFILES|EXTRA_ENV
gpu-4|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
3090|/models/Tiny|1|Tiny|multi|sglang|2048|0.5|1|chat-multiturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF

mkdir -p "$STATE_ROOT/synthetic_distributional"
printf 'running\n' > "$STATE_ROOT/synthetic_distributional/3090_Tiny_tp1_multi_sglang.status"
printf '8089\n' > "$STATE_ROOT/synthetic_distributional/3090_Tiny_tp1_multi_sglang.port"

BENCH_JOBS_FILE="$JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$STATE_ROOT" \
BENCH_RESULTS_ROOT="$RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "dry-run enabled" "$LOG"
grep -q "remote slot probing disabled" "$LOG"
grep -q "gpu-4_Tiny_tp1_single: dry-run would run on gpu-4" "$LOG"
grep -q "3090_Tiny_tp1_multi_sglang: dry-run would inspect remote outputs" "$LOG"
grep -q "dry-run: skipping sweep-state publish" "$LOG"

if [[ -e "$STATE_ROOT/synthetic_distributional/gpu-4_Tiny_tp1_single.status" ]]; then
    echo "dry-run unexpectedly wrote pending job state" >&2
    exit 1
fi

if [[ "$(cat "$STATE_ROOT/synthetic_distributional/3090_Tiny_tp1_multi_sglang.status")" != "running" ]]; then
    echo "dry-run unexpectedly mutated existing running job state" >&2
    exit 1
fi

echo "bench orchestrator service dry-run smoke test passed"
