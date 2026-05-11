#!/usr/bin/env bash
# Cron-driven benchmark orchestrator.
# - Reads bench_jobs.txt (job matrix)
# - Per host: if host idle, fire next pending job; if host busy, skip
# - Detects completed sweeps (rsync + s3 sync, mark done)
# - Detects OOM failures (parse vllm log), retries once with reduced max_len
# - Idempotent — safe to run on cron
#
# Cron line:
#   */30 * * * * BENCH_STATE_ROOT=/mnt/100g/agent-bench/state bash /root/agentic-serve/inference-benchmark/scripts/bench_orchestrator.sh >> /tmp/bench_orchestrator.cron.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JOBS_FILE="${BENCH_JOBS_FILE:-$REPO_ROOT/inference-benchmark/scripts/bench_jobs.txt}"
STATE_ROOT="${BENCH_STATE_ROOT:-/mnt/100g/agent-bench/state}"
LEGACY_STATE_ROOT="${BENCH_LEGACY_STATE_ROOT:-/tmp/bench_jobs/state}"
LOCAL_RESULTS_ROOT="${BENCH_RESULTS_ROOT:-${BENCHMARK_RESULTS_DIR:-/mnt/100g/agent-bench/results}}"
LOG="${BENCH_ORCHESTRATOR_LOG:-/tmp/bench_orchestrator.log}"
EP="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
# Raw benchmark outputs are namespaced so trace replay, synthetic
# distributional, and retired archived runs do not overwrite each other in R2.
# Override only for one-off maintenance, e.g. RESULT_SCOPE=archived/foo.
DEFAULT_RESULT_SCOPE="${RESULT_SCOPE:-}"
DRY_RUN="${BENCH_ORCHESTRATOR_DRY_RUN:-0}"
SKIP_REMOTE_PROBE="${BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE:-0}"
MAX_DISPATCHES="${BENCH_ORCHESTRATOR_MAX_DISPATCHES:-0}"
DISPATCHES=0
MAX_OOM_RETRIES="${BENCH_ORCHESTRATOR_MAX_OOM_RETRIES:-3}"
if ! [[ "$MAX_OOM_RETRIES" =~ ^[0-9]+$ ]]; then
    MAX_OOM_RETRIES=3
fi

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }
truthy() { [[ "${1:-}" == "1" || "${1:-}" == "true" || "${1:-}" == "yes" ]]; }
dry_run() { truthy "$DRY_RUN"; }

canonical_scope() {
    case "${1:-}" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical|fixed|fixed-grid|mse|archived) echo "archived" ;;
        *) echo "${1:-}" ;;
    esac
}

state_scope_aliases() {
    case "$(canonical_scope "$1")" in
        synthetic_distributional) echo "synthetic_distributional synthetic latest synthetic-distributional" ;;
        trace_replay) echo "trace_replay archive" ;;
        archived) echo "archived current canonical fixed fixed-grid mse" ;;
        *) echo "$1" ;;
    esac
}

LOCK_FILE="/tmp/bench_orchestrator.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "another bench_orchestrator.sh tick is already running; exiting"
    exit 0
fi

RAW_JOBS_SCOPE=$(awk -F': ' '/^# SCOPE:/ {print $2; exit}' "$JOBS_FILE" 2>/dev/null || true)
RAW_EXPECTED_JOBS_SCOPE="${BENCH_JOBS_SCOPE:-${RAW_JOBS_SCOPE:-fixed}}"
JOBS_SCOPE="$(canonical_scope "$RAW_JOBS_SCOPE")"
EXPECTED_JOBS_SCOPE="$(canonical_scope "$RAW_EXPECTED_JOBS_SCOPE")"
if [[ "$EXPECTED_JOBS_SCOPE" != "all" ]]; then
    if [[ "$JOBS_SCOPE" != "$EXPECTED_JOBS_SCOPE" ]]; then
        log "refusing to run: $JOBS_FILE has scope='${RAW_JOBS_SCOPE:-missing}' normalized='${JOBS_SCOPE:-missing}', expected '${RAW_EXPECTED_JOBS_SCOPE}' normalized='$EXPECTED_JOBS_SCOPE'"
        exit 1
    fi
fi
STATE_SCOPE="$EXPECTED_JOBS_SCOPE"
if [[ "$STATE_SCOPE" == "all" ]]; then
    STATE_SCOPE="${JOBS_SCOPE:-all}"
fi
STATE_DIR="$STATE_ROOT/$STATE_SCOPE"
LEGACY_STATE_DIR="$LEGACY_STATE_ROOT/$STATE_SCOPE"
mkdir -p "$STATE_DIR"
mkdir -p "$LOCAL_RESULTS_ROOT"
log "using jobs scope=${JOBS_SCOPE:-missing} expected_scope=$EXPECTED_JOBS_SCOPE state_dir=$STATE_DIR results_root=$LOCAL_RESULTS_ROOT"
dry_run && log "dry-run enabled: no state writes, rsync, R2 upload, sweep-state publish, or remote launch"
if [[ "$MAX_DISPATCHES" =~ ^[0-9]+$ && "$MAX_DISPATCHES" -gt 0 ]]; then
    log "max dispatches for this tick: $MAX_DISPATCHES"
fi

host_prefix() {
    case "$1" in
        gpu-4)  echo "a100"  ;;
        3090)   echo "3090"  ;;
        2080ti) echo "2080ti" ;;
        *)      echo "$1"    ;;
    esac
}

host_python() {
    # args: host [backend]
    local host="$1" backend="${2:-vllm}"
    if [[ "$backend" == "sglang" ]]; then
        case "$host" in
            gpu-4)       echo "/data/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            h100)        echo "/data/kevinlau/miniconda3/envs/sglang/bin/python" ;;
            h100-2)      echo "/home/kevinlau/miniconda3/envs/sglang/bin/python" ;;
        esac
    else
        case "$host" in
            gpu-4)       echo "/data/kevinlau/miniconda3/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
            h100|h100-2) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
        esac
    fi
}

# job_id keeps the legacy "host_model_tpN_mode" shape for vllm so existing
# state files in /tmp/bench_jobs/state/ remain valid. sglang cells get a
# "_sglang" suffix to disambiguate from the vllm run of the same cell.
job_id() {
    local jid="${1}_${2}_tp${3}_${4}"
    if [[ "${5:-vllm}" != "vllm" ]]; then
        jid="${jid}_${5}"
    fi
    echo "$jid"
}

extra_env_value() {
    local key="$1" text="${2:-}" part
    for part in $text; do
        if [[ "$part" == "$key="* ]]; then
            echo "${part#*=}"
            return
        fi
    done
}

row_result_scope() {
    local extra_env="${1:-}" scope
    scope=$(extra_env_value "RESULT_SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    scope=$(extra_env_value "DASHBOARD_SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    scope=$(extra_env_value "SCOPE" "$extra_env")
    [[ -n "$scope" ]] && { echo "$scope"; return; }
    [[ -n "$DEFAULT_RESULT_SCOPE" ]] && { echo "$DEFAULT_RESULT_SCOPE"; return; }
    [[ "$JOBS_SCOPE" != "all" && -n "$JOBS_SCOPE" ]] && { echo "$JOBS_SCOPE"; return; }
    [[ "$EXPECTED_JOBS_SCOPE" != "all" && -n "$EXPECTED_JOBS_SCOPE" ]] && { echo "$EXPECTED_JOBS_SCOPE"; return; }
    echo "current"
}

dashboard_scope_for() {
    case "$1" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical|fixed|fixed-grid|mse|archived|archived/*) echo "archived" ;;
        *) echo "$1" ;;
    esac
}

storage_scope_for() {
    case "$1" in
        synthetic|latest|synthetic-distributional|synthetic_distributional) echo "synthetic_distributional" ;;
        archive|trace_replay) echo "trace_replay" ;;
        current|canonical) echo "archived/canonical" ;;
        fixed|fixed-grid) echo "archived/fixed-grid" ;;
        mse) echo "archived/mse" ;;
        archived) echo "archived" ;;
        archived/*) echo "$1" ;;
        *) echo "$1" ;;
    esac
}

expected_output_summary() {
    local dir="$1" short="$2" tp="$3" backend="$4" mode="$5" concs="$6" profiles="$7"
    local profile conc file total=0 present=0
    local missing=()
    for profile in $profiles; do
        for conc in $concs; do
            total=$((total + 1))
            if [[ "$mode" == "multi" ]]; then
                file="$dir/${profile}_conc${conc}.json"
            else
                file="$dir/${short}_tp${tp}_${backend}_${profile}_conc${conc}.json"
            fi
            if [[ -s "$file" ]]; then
                present=$((present + 1))
            elif [[ "${#missing[@]}" -lt 4 ]]; then
                missing+=("$(basename "$file")")
            fi
        done
    done
    EXPECTED_OUTPUT_TOTAL="$total"
    EXPECTED_OUTPUT_PRESENT="$present"
    EXPECTED_OUTPUT_MISSING_SAMPLE="${missing[*]:-}"
}

oom_log_on_host() {
    local host="$1" port="$2"
    ssh "$host" "grep -l -i -E 'OutOfMemoryError|CUDA out of memory|out of memory|No available memory for the cache blocks|Available KV cache memory: -|larger than the available KV cache memory|estimated maximum model length|max seq len .*larger than' /tmp/vllm_${port}.log 2>/dev/null" < /dev/null || true
}

next_oom_max_len() {
    local max_len="$1" next
    next=$((max_len / 2))
    [[ "$next" -lt 2048 ]] && next=2048
    echo "$next"
}

can_retry_oom() {
    local oom="$1" attempt="$2" max_len="$3"
    [[ -n "$oom" && "$attempt" -lt "$MAX_OOM_RETRIES" && "$max_len" -gt 2048 ]]
}

state_read_file() {
    local jid="$1" suffix="$2" primary scope candidate
    primary="$STATE_DIR/${jid}.${suffix}"
    if [[ -f "$primary" ]]; then
        echo "$primary"
        return
    fi
    for scope in $(state_scope_aliases "$STATE_SCOPE"); do
        candidate="$STATE_ROOT/$scope/${jid}.${suffix}"
        if [[ "$candidate" != "$primary" && -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    for scope in $(state_scope_aliases "$STATE_SCOPE"); do
        candidate="$LEGACY_STATE_ROOT/$scope/${jid}.${suffix}"
        if [[ -f "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done
    candidate="$LEGACY_STATE_ROOT/${jid}.${suffix}"
    if [[ -f "$candidate" ]]; then
        echo "$candidate"
        return
    fi
    echo "$primary"
}

read_status()  { cat "$(state_read_file "$1" status)" 2>/dev/null || echo "pending"; }
write_state_value() {
    local jid="$1" suffix="$2" value="$3"
    if dry_run; then
        log "$jid: dry-run would write ${suffix}=$value"
    else
        echo "$value" > "$STATE_DIR/${jid}.${suffix}"
    fi
}
remove_state_file() {
    local jid="$1" suffix="$2"
    if dry_run; then
        log "$jid: dry-run would remove ${suffix}"
    else
        rm -f "$STATE_DIR/${jid}.${suffix}"
    fi
}
write_status() { write_state_value "$1" status "$2"; }
read_attempt() { cat "$(state_read_file "$1" attempt)" 2>/dev/null || echo "0"; }
bump_attempt() { local n=$(($(read_attempt "$1") + 1)); write_state_value "$1" attempt "$n"; }
read_signature() { cat "$(state_read_file "$1" signature)" 2>/dev/null || true; }
write_signature() { write_state_value "$1" signature "$2"; }

# Phase 1: multi-slot GPU scheduling — scan per-GPU and per-port usage.
declare -A HOST_GPU_COUNT=( [gpu-4]=8 [3090]=8 [2080ti]=8 [h100]=8 [h100-2]=4 )
PORT_RANGE=(8089 8090 8091 8092 8093 8094 8095 8096)
declare -A HOST_USED_GPUS
declare -A HOST_USED_PORTS
declare -A HOST_OBSERVED_PORTS

if truthy "$SKIP_REMOTE_PROBE"; then
    log "remote slot probing disabled by BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE"
else
    for HOST in gpu-4 3090 2080ti h100 h100-2; do
        SLOT_INFO=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" '
            echo "GPUS:$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F", " "\$2 > 100 {printf \$1\" \"}")"
            echo "PORTS:$(for p in 8089 8090 8091 8092 8093 8094 8095 8096; do ss -ltn 2>/dev/null | grep -q ":${p} " && printf "%s " $p; done)"
        ' 2>/dev/null || true)
        HOST_USED_GPUS[$HOST]=$(echo "$SLOT_INFO" | grep "^GPUS:" | sed 's/^GPUS://')
        HOST_USED_PORTS[$HOST]=$(echo "$SLOT_INFO" | grep "^PORTS:" | sed 's/^PORTS://')
        HOST_OBSERVED_PORTS[$HOST]="${HOST_USED_PORTS[$HOST]:-}"
        log "slots $HOST: used_gpus=[${HOST_USED_GPUS[$HOST]:-}] used_ports=[${HOST_USED_PORTS[$HOST]:-}]"
    done
fi

find_free_gpus() {
    local host="$1" needed="$2"
    local total=${HOST_GPU_COUNT[$host]:-0}
    local used=" ${HOST_USED_GPUS[$host]:-} "
    local free=()
    for ((i=0; i<total; i++)); do
        [[ "$used" == *" $i "* ]] && continue
        free+=("$i")
        [[ ${#free[@]} -ge $needed ]] && break
    done
    [[ ${#free[@]} -ge $needed ]] && { IFS=,; echo "${free[*]}"; } || true
}

find_free_port() {
    local host="$1"
    local used=" ${HOST_USED_PORTS[$host]:-} "
    for p in "${PORT_RANGE[@]}"; do
        [[ "$used" == *" $p "* ]] && continue
        echo "$p"
        return
    done
}

claim_slot() {
    local host="$1" gpus="$2" port="$3"
    HOST_USED_GPUS[$host]="${HOST_USED_GPUS[$host]:-} ${gpus//,/ } "
    HOST_USED_PORTS[$host]="${HOST_USED_PORTS[$host]:-} $port "
}

# Reserve recently dispatched running jobs from state before considering
# pending rows. This protects jobs that are still loading and have not opened a
# port or allocated noticeable GPU memory yet.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue
    : "${BACKEND:=vllm}"
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    [[ "$STATUS" != "running" ]] && continue

    JOB_PORT=$(cat "$(state_read_file "$JID" port)" 2>/dev/null || true)
    JOB_GPUS=$(cat "$(state_read_file "$JID" gpus)" 2>/dev/null || true)
    [[ -z "$JOB_PORT$JOB_GPUS" ]] && continue

    STATUS_FILE="$(state_read_file "$JID" status)"
    AGE=999999999
    [[ -f "$STATUS_FILE" ]] && AGE=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))

    claim_slot "$HOST" "$JOB_GPUS" "$JOB_PORT"
    log "$JID: reserving recorded running slot on $HOST port=$JOB_PORT gpus=[$JOB_GPUS] age=${AGE}s"
done < "$JOBS_FILE"

# Phase 2: scan jobs, decide actions.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue
    if [[ "$EXPECTED_JOBS_SCOPE" == "fixed" && "$CONCS" != "200 320" ]]; then
        log "skipping non-fixed-grid row for $HOST/$SHORT/tp$TP/$MODE/$BACKEND: CONCS='$CONCS'"
        continue
    fi

    : "${BACKEND:=vllm}"  # default if column missing (legacy rows)
    ROW_RESULT_SCOPE=$(row_result_scope "$EXTRA_ENV")
    ROW_DASHBOARD_SCOPE=$(dashboard_scope_for "$ROW_RESULT_SCOPE")
    ROW_STORAGE_SCOPE=$(storage_scope_for "$ROW_RESULT_SCOPE")
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    JOB_SIGNATURE="${ROW_STORAGE_SCOPE}|${ROW_DASHBOARD_SCOPE}|${MAX_LEN}|${GPU_MEM}|${CONCS}|${PROFILES}|${EXTRA_ENV}"
    OLD_SIGNATURE=$(read_signature "$JID")
    if [[ "$STATUS" =~ ^(done|skipped|failed)$ && -n "$OLD_SIGNATURE" && "$OLD_SIGNATURE" != "$JOB_SIGNATURE" ]]; then
        log "$JID: job shape changed since terminal $STATUS; retrying as pending"
        STATUS="pending"
        write_status "$JID" pending
        write_state_value "$JID" attempt "0"
        remove_state_file "$JID" max_len_override
    elif [[ "$STATUS" =~ ^(skipped|failed)$ && -z "$OLD_SIGNATURE" && "$MODE" == "multi" && "$PROFILES" != *"swebench-multiturn"* && "$PROFILES" != *"terminalbench-multiturn"* ]]; then
        log "$JID: legacy terminal $STATUS predates profile filtering; retrying reduced-profile job"
        STATUS="pending"
        write_status "$JID" pending
        write_state_value "$JID" attempt "0"
        remove_state_file "$JID" max_len_override
    fi
    PREFIX=$(host_prefix "$HOST")
    RESULT_DIR_NAME="${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"
    OUT_DIR_REMOTE="/tmp/results/${ROW_STORAGE_SCOPE}/${RESULT_DIR_NAME}"
    # Fallback for jobs launched before RESULT_SCOPE existed; completed jobs
    # still upload into the normalized R2 namespace to avoid legacy collisions.
    LEGACY_OUT_DIR_REMOTE="/tmp/results/${RESULT_DIR_NAME}"
    R2_DIR="${ROW_STORAGE_SCOPE}/${RESULT_DIR_NAME}"
    OUT_DIR_LOCAL="$LOCAL_RESULTS_ROOT/$R2_DIR"
    RUN_MAX_LEN="$MAX_LEN"
    OVERRIDE_FILE="$(state_read_file "$JID" max_len_override)"
    if [[ -f "$OVERRIDE_FILE" ]]; then
        RUN_MAX_LEN=$(cat "$OVERRIDE_FILE")
    fi

    case "$STATUS" in
        done|skipped|failed)
            continue
            ;;
        running)
            JOB_PORT=$(cat "$(state_read_file "$JID" port)" 2>/dev/null || echo "8089")
            if [[ " ${HOST_OBSERVED_PORTS[$HOST]:-} " == *" $JOB_PORT "* ]]; then
                log "$JID: still running on $HOST:$JOB_PORT"
                continue
            fi
            # Grace period: weight-load + CUDA graph compilation.
            # vllm: ~3-5 min typical, 10 min max.
            # sglang: aggressive torch compilation, 10-15 min for large/MoE models.
            WARMUP_TIMEOUT=600
            [[ "$BACKEND" == "sglang" ]] && WARMUP_TIMEOUT=900
            STATUS_FILE="$(state_read_file "$JID" status)"
            AGE=0
            if [[ -f "$STATUS_FILE" ]]; then
                AGE=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))
                if [[ "$AGE" -lt "$WARMUP_TIMEOUT" ]]; then
                    if [[ "$BACKEND" == "sglang" ]]; then
                        SCRIPT_NAME="sweep_all_profiles_sglang.sh"
                        [[ "$MODE" == "multi" ]] && SCRIPT_NAME="sweep_multiturn_profiles_sglang.sh"
                    else
                        SCRIPT_NAME="sweep_all_profiles.sh"
                        [[ "$MODE" == "multi" ]] && SCRIPT_NAME="sweep_multiturn_profiles.sh"
                    fi
                    if truthy "$SKIP_REMOTE_PROBE"; then
                        REMOTE_SCRIPT_ALIVE=""
                    else
                        REMOTE_SCRIPT_ALIVE=$(ssh "$HOST" "ps -eo args= | awk -v script='/tmp/inference-benchmark/scripts/${SCRIPT_NAME}' -v needle=' ${TP} ${SHORT} ${BACKEND} ' -v concs=' ${CONCS} ' '\$1 == \"bash\" && \$2 == script && index(\$0, needle) && index(\$0, concs) { found=1 } END { exit found ? 0 : 1 }' && echo yes" < /dev/null 2>/dev/null || true)
                    fi
                    if [[ "$REMOTE_SCRIPT_ALIVE" == "yes" ]]; then
                        log "$JID: dispatched ${AGE}s ago (<$(( WARMUP_TIMEOUT / 60 ))min), still warming up on port $JOB_PORT"
                        JOB_GPUS=$(cat "$(state_read_file "$JID" gpus)" 2>/dev/null || true)
                        [[ -n "$JOB_GPUS" ]] && HOST_USED_GPUS[$HOST]="${HOST_USED_GPUS[$HOST]:-} ${JOB_GPUS//,/ } "
                        HOST_USED_PORTS[$HOST]="${HOST_USED_PORTS[$HOST]:-} $JOB_PORT "
                        continue
                    fi
                    log "$JID: no listener and no live sweep process after ${AGE}s; finalizing early"
                fi
            fi
            log "$JID: slot idle after ${AGE}s warmup ($BACKEND) — finalizing"
            if dry_run; then
                log "$JID: dry-run would inspect remote outputs and update terminal state"
                continue
            fi
            # All ssh/rsync/aws calls inside this `while read ... done <JOBS`
            # loop must close stdin (`< /dev/null`), otherwise they consume
            # the jobs file and iteration ends early — 2080ti rows were
            # silently skipped on any tick that also dispatched a 3090 job.
            REMOTE_SYNC_DIR=$(ssh "$HOST" "if [ -d '$OUT_DIR_REMOTE' ] && [ \$(ls -1 '$OUT_DIR_REMOTE' 2>/dev/null | wc -l) -gt 0 ]; then echo '$OUT_DIR_REMOTE'; elif [ -d '$LEGACY_OUT_DIR_REMOTE' ] && [ \$(ls -1 '$LEGACY_OUT_DIR_REMOTE' 2>/dev/null | wc -l) -gt 0 ]; then echo '$LEGACY_OUT_DIR_REMOTE'; fi" < /dev/null)
            if [[ -n "$REMOTE_SYNC_DIR" ]]; then
                COUNT=$(ssh "$HOST" "ls '$REMOTE_SYNC_DIR' 2>/dev/null | wc -l" < /dev/null)
                mkdir -p "$OUT_DIR_LOCAL"
                if rsync -az "$HOST:$REMOTE_SYNC_DIR/" "$OUT_DIR_LOCAL/" < /dev/null >> "$LOG" 2>&1; then
                    expected_output_summary "$OUT_DIR_LOCAL" "$SHORT" "$TP" "$BACKEND" "$MODE" "$CONCS" "$PROFILES"
                    MIRROR_STATUS="not_mirrored"
                    if [[ "$EXPECTED_OUTPUT_PRESENT" -gt 0 ]]; then
                        if aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync \
                            "$OUT_DIR_LOCAL/" "s3://$BUCKET/results/$R2_DIR/" < /dev/null >> "$LOG" 2>&1; then
                            MIRROR_STATUS="r2_mirrored"
                        else
                            MIRROR_STATUS="r2_mirror_failed"
                        fi
                    fi
                    if [[ "$EXPECTED_OUTPUT_PRESENT" -eq "$EXPECTED_OUTPUT_TOTAL" ]]; then
                        write_status "$JID" done
                        log "$JID: DONE ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs; $COUNT files copied to $OUT_DIR_LOCAL, warmup=${AGE}s backend=$BACKEND mirror=$MIRROR_STATUS)"
                    elif [[ "$EXPECTED_OUTPUT_PRESENT" -gt 0 ]]; then
                        write_status "$JID" pending
                        log "$JID: INCOMPLETE ($EXPECTED_OUTPUT_PRESENT/$EXPECTED_OUTPUT_TOTAL expected outputs; missing=${EXPECTED_OUTPUT_MISSING_SAMPLE:-unknown}; $COUNT files copied to $OUT_DIR_LOCAL, warmup=${AGE}s backend=$BACKEND mirror=$MIRROR_STATUS); leaving pending"
                    else
                        OOM=$(oom_log_on_host "$HOST" "$JOB_PORT")
                        ATT=$(read_attempt "$JID")
                        if can_retry_oom "$OOM" "$ATT" "$RUN_MAX_LEN"; then
                            bump_attempt "$JID"
                            write_status "$JID" pending
                            NEW_MAX=$(next_oom_max_len "$RUN_MAX_LEN")
                            write_state_value "$JID" max_len_override "$NEW_MAX"
                            log "$JID: zero expected outputs after copying $COUNT files; OOM detected, retry with max_len=$NEW_MAX"
                        else
                            write_status "$JID" skipped
                            log "$JID: SKIPPED (0/$EXPECTED_OUTPUT_TOTAL expected outputs; copied only stale/non-matching files from $REMOTE_SYNC_DIR; attempt=$ATT, oom_log=$OOM)"
                        fi
                    fi
                else
                    write_status "$JID" pending
                    log "$JID: local rsync failed for $REMOTE_SYNC_DIR -> $OUT_DIR_LOCAL; leaving pending"
                fi
            else
                # Widen detection to include vLLM's KV-cache budget failures,
                # which are reported as ValueError rather than torch OOM.
                OOM=$(oom_log_on_host "$HOST" "$JOB_PORT")
                ATT=$(read_attempt "$JID")
                if can_retry_oom "$OOM" "$ATT" "$RUN_MAX_LEN"; then
                    bump_attempt "$JID"
                    write_status "$JID" pending
                    NEW_MAX=$(next_oom_max_len "$RUN_MAX_LEN")
                    write_state_value "$JID" max_len_override "$NEW_MAX"
                    log "$JID: OOM detected, retry with max_len=$NEW_MAX"
                else
                    write_status "$JID" skipped
                    log "$JID: SKIPPED (zero results, attempt=$ATT, oom_log=$OOM)"
                fi
            fi
            ;;
        pending)
            if [[ "$MAX_DISPATCHES" =~ ^[0-9]+$ && "$MAX_DISPATCHES" -gt 0 && "$DISPATCHES" -ge "$MAX_DISPATCHES" ]]; then
                continue
            fi

            # Extract explicit CUDA_VISIBLE_DEVICES from extra_env if present
            CELL_CVD=""
            if [[ "$EXTRA_ENV" == *CUDA_VISIBLE_DEVICES=* ]]; then
                CELL_CVD=$(echo "$EXTRA_ENV" | sed -n 's/.*CUDA_VISIBLE_DEVICES=\([^ ]*\).*/\1/p')
            fi

            if [[ -n "$CELL_CVD" ]]; then
                # Cell pins specific GPUs — check they're all free
                SLOT_FREE=1
                USED=" ${HOST_USED_GPUS[$HOST]:-} "
                for g in ${CELL_CVD//,/ }; do
                    [[ "$USED" == *" $g "* ]] && SLOT_FREE=0 && break
                done
                [[ "$SLOT_FREE" == "0" ]] && continue
                SLOT_GPUS="$CELL_CVD"
            else
                SLOT_GPUS=$(find_free_gpus "$HOST" "$TP")
                [[ -z "$SLOT_GPUS" ]] && continue
            fi

            SLOT_PORT=$(find_free_port "$HOST")
            [[ -z "$SLOT_PORT" ]] && continue

            PY=$(host_python "$HOST" "$BACKEND")
            if [[ "$BACKEND" == "sglang" ]]; then
                SCRIPT="sweep_all_profiles_sglang.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles_sglang.sh"
            else
                SCRIPT="sweep_all_profiles.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles.sh"
            fi

            claim_slot "$HOST" "$SLOT_GPUS" "$SLOT_PORT"
            write_state_value "$JID" port "$SLOT_PORT"
            write_state_value "$JID" gpus "$SLOT_GPUS"
            write_signature "$JID" "$JOB_SIGNATURE"

            log "$JID: dispatching on $HOST:$SLOT_PORT gpus=[$SLOT_GPUS] ($BACKEND, scope=$ROW_DASHBOARD_SCOPE, storage=$ROW_STORAGE_SCOPE, max_len=$RUN_MAX_LEN, mode=$MODE)"
            write_status "$JID" running

            # Build env: PORT + CUDA_VISIBLE_DEVICES (unless cell already pins)
            SLOT_ENV="PORT=$SLOT_PORT"
            if [[ -z "$CELL_CVD" ]]; then
                SLOT_ENV="$SLOT_ENV CUDA_VISIBLE_DEVICES=$SLOT_GPUS"
            fi

            CMD="$SLOT_ENV ${EXTRA_ENV} RESULT_SCOPE=${ROW_STORAGE_SCOPE} DASHBOARD_SCOPE=${ROW_DASHBOARD_SCOPE} bash /tmp/inference-benchmark/scripts/${SCRIPT} \
                ${MODEL_PATH} ${TP} ${SHORT} ${BACKEND} ${OUT_DIR_REMOTE} \
                ${PY} ${GPU_MEM} ${RUN_MAX_LEN} \"${CONCS}\" \"${PROFILES}\""
            REMOTE_LOG="/tmp/bench_${SHORT}_tp${TP}_${MODE}_${BACKEND}_p${SLOT_PORT}.log"
            if dry_run; then
                log "$JID: dry-run would run on $HOST: setsid bash -c '$CMD' > '$REMOTE_LOG' 2>&1 </dev/null &"
            else
                ssh "$HOST" "setsid bash -c '${CMD}' > '${REMOTE_LOG}' 2>&1 </dev/null &" < /dev/null
                log "$JID: dispatched"
            fi
            DISPATCHES=$((DISPATCHES + 1))
            ;;
    esac
done < "$JOBS_FILE"

if dry_run; then
    log "dry-run: skipping sweep-state publish"
else
    # Publish sweep-state.json to R2 so the dashboard reflects the latest cell
    # status (pending/running/done/skipped/known_oom). Non-fatal — if this
    # fails, the tick still succeeds; the next tick will republish.
    python3 "$REPO_ROOT/inference-benchmark/scripts/publish_sweep_state.py" \
        --state-dir "$STATE_ROOT" \
        --endpoint "$EP" --bucket "$BUCKET" --profile "$PROFILE" \
        >> "$LOG" 2>&1 || log "publish_sweep_state.py failed"
fi

log "tick complete"
