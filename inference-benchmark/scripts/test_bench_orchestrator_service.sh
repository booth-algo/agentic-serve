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
a100|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
3090|/models/Tiny|1|Tiny|multi|sglang|2048|0.5|1|chat-multiturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
2080ti|/models/Busy|1|Busy|single|vllm|2048|0.5|1|chat-singleturn-synth|CUDA_VISIBLE_DEVICES=1 RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
2080ti|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|CUDA_VISIBLE_DEVICES=1 PATH=/home/kevinlau/miniconda3/envs/vllm/bin:$PATH RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF

mkdir -p "$STATE_ROOT/synthetic"
printf 'running\n' > "$STATE_ROOT/synthetic/3090_Tiny_tp1_multi_sglang.status"
printf '8089\n' > "$STATE_ROOT/synthetic/3090_Tiny_tp1_multi_sglang.port"
mkdir -p "$STATE_ROOT/synthetic_distributional"
printf 'running\n' > "$STATE_ROOT/synthetic_distributional/2080ti_Busy_tp1_single.status"
printf '8090\n' > "$STATE_ROOT/synthetic_distributional/2080ti_Busy_tp1_single.port"
printf '1\n' > "$STATE_ROOT/synthetic_distributional/2080ti_Busy_tp1_single.gpus"

BENCH_JOBS_FILE="$JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic \
BENCH_STATE_ROOT="$STATE_ROOT" \
BENCH_RESULTS_ROOT="$RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "dry-run enabled" "$LOG"
grep -q "remote slot probing disabled" "$LOG"
grep -q "a100_Tiny_tp1_single: dry-run would run on a100" "$LOG"
grep -q "3090_Tiny_tp1_multi_sglang: dry-run would inspect remote outputs" "$LOG"
grep -q "2080ti_Tiny_tp1_single: preferred CUDA_VISIBLE_DEVICES=\\[1\\] busy; flexing to \\[0\\]" "$LOG"
grep -q "2080ti_Tiny_tp1_single: dry-run would run on 2080ti: setsid bash -c 'PORT=8089 CUDA_VISIBLE_DEVICES=0" "$LOG"
grep -q "BENCH_RUN_ID=run_" "$LOG"
grep -q "BENCH_JOB_ID=2080ti_Tiny_tp1_single" "$LOG"
grep -q "PATH=/home/kevinlau/miniconda3/envs/vllm/bin:" "$LOG"
grep -q "dry-run: skipping sweep-state publish" "$LOG"

if [[ -e "$STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.status" ]]; then
    echo "dry-run unexpectedly wrote pending job state" >&2
    exit 1
fi

if [[ -e "$STATE_ROOT/synthetic_distributional/3090_Tiny_tp1_multi_sglang.status" ]]; then
    echo "dry-run unexpectedly migrated legacy synthetic state" >&2
    exit 1
fi

if [[ "$(cat "$STATE_ROOT/synthetic/3090_Tiny_tp1_multi_sglang.status")" != "running" ]]; then
    echo "dry-run unexpectedly mutated existing running job state" >&2
    exit 1
fi

NEXT_LEN_SNIPPET=$(awk '/^next_oom_max_len\(\)/,/^}/ {print}' "$SCRIPT_DIR/bench_orchestrator.sh")
NEXT_LEN_OUTPUT=$(bash -c "$NEXT_LEN_SNIPPET; next_oom_max_len 32768 11328; next_oom_max_len 4096; next_oom_max_len 4096 1024")
EXPECTED_NEXT_LEN=$'8192\n2048\n2048'
if [[ "$NEXT_LEN_OUTPUT" != "$EXPECTED_NEXT_LEN" ]]; then
    echo "unexpected OOM max_len retry plan:" >&2
    echo "$NEXT_LEN_OUTPUT" >&2
    exit 1
fi

DRAIN_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR"' EXIT
DRAIN_JOBS_FILE="$DRAIN_TMP_DIR/bench_jobs.txt"
DRAIN_STATE_ROOT="$DRAIN_TMP_DIR/state"
DRAIN_RESULTS_ROOT="$DRAIN_TMP_DIR/results"
DRAIN_LOG="$DRAIN_TMP_DIR/orchestrator.log"
mkdir -p "$DRAIN_STATE_ROOT/control"
cat > "$DRAIN_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
3090|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF
printf '3090\n' > "$DRAIN_STATE_ROOT/control/drained-hosts.txt"

BENCH_JOBS_FILE="$DRAIN_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$DRAIN_STATE_ROOT" \
BENCH_RESULTS_ROOT="$DRAIN_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$DRAIN_LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "host 3090 is drained; preserving running jobs but skipping new dispatches" "$DRAIN_LOG"
if grep -q "dry-run would run on 3090" "$DRAIN_LOG"; then
    echo "drained host unexpectedly dispatched a pending job" >&2
    exit 1
fi

CACHE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR" "$CACHE_TMP_DIR"' EXIT
CACHE_JOBS_FILE="$CACHE_TMP_DIR/bench_jobs.txt"
CACHE_STATE_ROOT="$CACHE_TMP_DIR/state"
CACHE_RESULTS_ROOT="$CACHE_TMP_DIR/results"
CACHE_LOG="$CACHE_TMP_DIR/orchestrator.log"
mkdir -p "$CACHE_RESULTS_ROOT/synthetic_distributional/a100_Tiny_tp1_vllm"
cat > "$CACHE_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
a100|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF
printf '{}\n' > "$CACHE_RESULTS_ROOT/synthetic_distributional/a100_Tiny_tp1_vllm/Tiny_tp1_vllm_chat-singleturn-synth_conc1.json"

BENCH_JOBS_FILE="$CACHE_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$CACHE_STATE_ROOT" \
BENCH_RESULTS_ROOT="$CACHE_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$CACHE_LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "a100_Tiny_tp1_single: dry-run would mark DONE from local cache (1/1 expected outputs)" "$CACHE_LOG"
if grep -q "a100_Tiny_tp1_single: dry-run would run on a100" "$CACHE_LOG"; then
    echo "locally complete pending job unexpectedly dispatched" >&2
    exit 1
fi

H100_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR" "$CACHE_TMP_DIR" "$H100_TMP_DIR"' EXIT
H100_JOBS_FILE="$H100_TMP_DIR/bench_jobs.txt"
H100_STATE_ROOT="$H100_TMP_DIR/state"
H100_RESULTS_ROOT="$H100_TMP_DIR/results"
H100_LOG="$H100_TMP_DIR/orchestrator.log"
cat > "$H100_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
h100|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF

BENCH_JOBS_FILE="$H100_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$H100_STATE_ROOT" \
BENCH_RESULTS_ROOT="$H100_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$H100_LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "h100_Tiny_tp1_single: dry-run would run on h100" "$H100_LOG"
grep -q "BENCH_REMOTE_TMP=/data48/tmp BENCH_REMOTE_ROOT=/data48/tmp/inference-benchmark" "$H100_LOG"
grep -q "bash /data48/tmp/inference-benchmark/scripts/sweep_all_profiles.sh" "$H100_LOG"
grep -q "/data48/tmp/results/synthetic_distributional/h100_Tiny_tp1_vllm" "$H100_LOG"
grep -q "> '/data48/tmp/bench_Tiny_tp1_single_vllm_p8089.log'" "$H100_LOG"

STALE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR" "$CACHE_TMP_DIR" "$H100_TMP_DIR" "$STALE_TMP_DIR"' EXIT
STALE_JOBS_FILE="$STALE_TMP_DIR/bench_jobs.txt"
STALE_STATE_ROOT="$STALE_TMP_DIR/state"
STALE_RESULTS_ROOT="$STALE_TMP_DIR/results"
STALE_LOG="$STALE_TMP_DIR/orchestrator.log"
FAKE_BIN="$STALE_TMP_DIR/bin"
mkdir -p "$FAKE_BIN" "$STALE_STATE_ROOT/synthetic_distributional"
cat > "$STALE_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
a100|/models/Tiny|1|Tiny|single|sglang|2048|0.5|1|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF
printf 'running\n' > "$STALE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single_sglang.status"
printf '8091\n' > "$STALE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single_sglang.port"
printf '5\n' > "$STALE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single_sglang.gpus"
touch -d '2 hours ago' "$STALE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single_sglang.status"
cat > "$FAKE_BIN/ssh" <<'EOF'
#!/usr/bin/env bash
cat <<'OUT'
GPUS:5
PORTCMD:8091|python -m sglang.launch_server --model-path /models/Other --port 8091
PORTS:8091
OUT
EOF
chmod +x "$FAKE_BIN/ssh"

PATH="$FAKE_BIN:$PATH" \
BENCH_JOBS_FILE="$STALE_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$STALE_STATE_ROOT" \
BENCH_RESULTS_ROOT="$STALE_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$STALE_LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "a100_Tiny_tp1_single_sglang: not reserving stale recorded slot on a100:8091" "$STALE_LOG"
grep -q "a100_Tiny_tp1_single_sglang: recorded port a100:8091 is held by a different command" "$STALE_LOG"
grep -q "a100_Tiny_tp1_single_sglang: dry-run would inspect remote outputs and update terminal state" "$STALE_LOG"

DONE_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR" "$CACHE_TMP_DIR" "$H100_TMP_DIR" "$STALE_TMP_DIR" "$DONE_TMP_DIR"' EXIT
DONE_JOBS_FILE="$DONE_TMP_DIR/bench_jobs.txt"
DONE_STATE_ROOT="$DONE_TMP_DIR/state"
DONE_RESULTS_ROOT="$DONE_TMP_DIR/results"
DONE_LOG="$DONE_TMP_DIR/orchestrator.log"
DONE_FAKE_BIN="$DONE_TMP_DIR/bin"
mkdir -p "$DONE_FAKE_BIN" "$DONE_STATE_ROOT/synthetic_distributional"
cat > "$DONE_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
a100|/models/Tiny|1|Tiny|multi|sglang|2048|0.5|1|chat-multiturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF
printf 'running\n' > "$DONE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_multi_sglang.status"
printf '8092\n' > "$DONE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_multi_sglang.port"
printf '5\n' > "$DONE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_multi_sglang.gpus"
touch -d '2 hours ago' "$DONE_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_multi_sglang.status"
cat > "$DONE_FAKE_BIN/ssh" <<'EOF'
#!/usr/bin/env bash
if [[ "$*" == *"done; results in"* ]]; then
    echo yes
    exit 0
fi
cat <<'OUT'
GPUS:5
PORTCMD:8092|python -m sglang.launch_server --model-path /models/Tiny --port 8092
PORTS:8092
OUT
EOF
chmod +x "$DONE_FAKE_BIN/ssh"

PATH="$DONE_FAKE_BIN:$PATH" \
BENCH_JOBS_FILE="$DONE_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$DONE_STATE_ROOT" \
BENCH_RESULTS_ROOT="$DONE_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$DONE_LOG" \
BENCH_ORCHESTRATOR_DRY_RUN=1 \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "a100_Tiny_tp1_multi_sglang: sweep log /tmp/bench_Tiny_tp1_multi_sglang_p8092.log is complete but a100:8092 still listens; finalizing" "$DONE_LOG"
grep -q "a100_Tiny_tp1_multi_sglang: dry-run would inspect remote outputs and update terminal state" "$DONE_LOG"

RETRY_TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR" "$DRAIN_TMP_DIR" "$CACHE_TMP_DIR" "$H100_TMP_DIR" "$STALE_TMP_DIR" "$DONE_TMP_DIR" "$RETRY_TMP_DIR"' EXIT
RETRY_JOBS_FILE="$RETRY_TMP_DIR/bench_jobs.txt"
RETRY_STATE_ROOT="$RETRY_TMP_DIR/state"
RETRY_RESULTS_ROOT="$RETRY_TMP_DIR/results"
RETRY_LOG="$RETRY_TMP_DIR/orchestrator.log"
RETRY_FAKE_BIN="$RETRY_TMP_DIR/bin"
mkdir -p "$RETRY_FAKE_BIN" "$RETRY_STATE_ROOT/synthetic_distributional/runs"
cat > "$RETRY_JOBS_FILE" <<'EOF'
# SCOPE: synthetic_distributional
a100|/models/Tiny|1|Tiny|single|vllm|2048|0.5|1 2|chat-singleturn-synth|RESULT_SCOPE=synthetic_distributional DASHBOARD_SCOPE=synthetic_distributional
EOF
printf 'running\n' > "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.status"
printf '8089\n' > "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.port"
printf '0\n' > "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.gpus"
printf '1\n' > "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.attempt"
printf 'run_retry_test\n' > "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.run_id"
touch -d '2 hours ago' "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.status"
cat > "$RETRY_STATE_ROOT/synthetic_distributional/runs/run_retry_test.json" <<'EOF'
{"run_id":"run_retry_test","job_id":"a100_Tiny_tp1_single","status":"running","port":"8089","gpus":["0"]}
EOF
cat > "$RETRY_FAKE_BIN/ssh" <<'EOF'
#!/usr/bin/env bash
args="$*"
if [[ "$args" == *"grep -q 'done; results in '"* ]]; then
    echo yes
    exit 0
fi
if [[ "$args" == *"grep -E 'ABORT:"* ]]; then
    echo "ABORT: Success rate 40.7% below minimum 75% (407/1000)"
    exit 0
fi
if [[ "$args" == *"for d in"* ]]; then
    echo "/tmp/results/synthetic_distributional/a100_Tiny_tp1_vllm"
    exit 0
fi
if [[ "$args" == *"ls '/tmp/results/synthetic_distributional/a100_Tiny_tp1_vllm'"* ]]; then
    echo 1
    exit 0
fi
cat <<'OUT'
GPUS:0
PORTCMD:8089|python -m vllm.entrypoints.openai.api_server --model /models/Tiny --port 8089
PORTS:8089
OUT
EOF
cat > "$RETRY_FAKE_BIN/rsync" <<'EOF'
#!/usr/bin/env bash
dest="${@: -1}"
mkdir -p "$dest"
printf '{}\n' > "$dest/Tiny_tp1_vllm_chat-singleturn-synth_conc1.json"
EOF
cat > "$RETRY_FAKE_BIN/aws" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$RETRY_FAKE_BIN/ssh" "$RETRY_FAKE_BIN/rsync" "$RETRY_FAKE_BIN/aws"

PATH="$RETRY_FAKE_BIN:$PATH" \
BENCH_JOBS_FILE="$RETRY_JOBS_FILE" \
BENCH_JOBS_SCOPE=synthetic_distributional \
BENCH_STATE_ROOT="$RETRY_STATE_ROOT" \
BENCH_RESULTS_ROOT="$RETRY_RESULTS_ROOT" \
BENCH_ORCHESTRATOR_LOG="$RETRY_LOG" \
BENCH_ORCHESTRATOR_MAX_INCOMPLETE_RETRIES=2 \
BENCH_ORCHESTRATOR_SKIP_PUBLISH=1 \
BENCH_COVERAGE_REPORT="$RETRY_TMP_DIR/coverage.md" \
BENCH_COVERAGE_MISSING_JOBS="$RETRY_TMP_DIR/missing_jobs.txt" \
BENCH_COVERAGE_BLOCKERS_JSON="$RETRY_TMP_DIR/coverage-blockers.json" \
BENCH_COVERAGE_SWEEP_STATE_OUT="$RETRY_TMP_DIR/sweep-state.json" \
BENCH_SYNC_GPU_CODE=0 \
    bash "$SCRIPT_DIR/run-bench-orchestrator-service.sh"

grep -q "a100_Tiny_tp1_single: SKIPPED incomplete retry limit" "$RETRY_LOG"
grep -q "BENCH_ORCHESTRATOR_SKIP_PUBLISH enabled" "$RETRY_LOG"
if [[ "$(cat "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.status")" != "skipped" ]]; then
    echo "incomplete retry limit did not mark the job skipped" >&2
    exit 1
fi
if [[ "$(cat "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.attempt")" != "2" ]]; then
    echo "incomplete retry limit did not bump attempt to 2" >&2
    exit 1
fi
grep -q "retry limit reached after 2/2 incomplete attempts" "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.reason"
grep -q "Tiny_tp1_vllm_chat-singleturn-synth_conc2.json" "$RETRY_STATE_ROOT/synthetic_distributional/a100_Tiny_tp1_single.failure.json"
grep -q '"status": "skipped"' "$RETRY_STATE_ROOT/synthetic_distributional/runs/run_retry_test.json"

echo "bench orchestrator service dry-run smoke test passed"
