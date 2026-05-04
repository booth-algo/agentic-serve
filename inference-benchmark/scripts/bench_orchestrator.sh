#!/usr/bin/env bash
# Cron-driven benchmark orchestrator.
# - Reads bench_jobs.txt (job matrix)
# - Per host: if host idle, fire next pending job; if host busy, skip
# - Detects completed sweeps (rsync + s3 sync, mark done)
# - Detects OOM failures (parse vllm log), retries once with reduced max_len
# - Idempotent — safe to run on cron
#
# Cron line:
#   */30 * * * * bash /root/agentic-serve/inference-benchmark/scripts/bench_orchestrator.sh >> /tmp/bench_orchestrator.cron.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JOBS_FILE="$REPO_ROOT/inference-benchmark/scripts/bench_jobs.txt"
STATE_DIR="/tmp/bench_jobs/state"
LOG="/tmp/bench_orchestrator.log"
EP="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
# Raw benchmark outputs are namespaced so the new distributional/current runs
# cannot overwrite historical flat-prefix artifacts in R2. Override only for
# one-off maintenance, e.g. RESULT_SCOPE=archive/foo.
RESULT_SCOPE="${RESULT_SCOPE:-current}"

mkdir -p "$STATE_DIR"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

LOCK_FILE="/tmp/bench_orchestrator.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    log "another bench_orchestrator.sh tick is already running; exiting"
    exit 0
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
        esac
    else
        case "$host" in
            gpu-4)       echo "/data/kevinlau/miniconda3/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
            h100)        echo "/data/kevinlau/miniconda3/envs/vllm/bin/python" ;;
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

read_status()  { cat "$STATE_DIR/${1}.status" 2>/dev/null || echo "pending"; }
write_status() { echo "$2" > "$STATE_DIR/${1}.status"; }
read_attempt() { cat "$STATE_DIR/${1}.attempt" 2>/dev/null || echo "0"; }
bump_attempt() { local n=$(($(read_attempt "$1") + 1)); echo "$n" > "$STATE_DIR/${1}.attempt"; }
read_signature() { cat "$STATE_DIR/${1}.signature" 2>/dev/null || true; }
write_signature() { echo "$2" > "$STATE_DIR/${1}.signature"; }

# Phase 1: multi-slot GPU scheduling — scan per-GPU and per-port usage.
declare -A HOST_GPU_COUNT=( [gpu-4]=8 [3090]=8 [2080ti]=8 [h100]=8 )
PORT_RANGE=(8089 8090 8091 8092 8093 8094 8095 8096)
declare -A HOST_USED_GPUS
declare -A HOST_USED_PORTS

for HOST in gpu-4 3090 2080ti h100; do
    SLOT_INFO=$(ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" '
        echo "GPUS:$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F", " "\$2 > 100 {printf \$1\" \"}")"
        echo "PORTS:$(for p in 8089 8090 8091 8092 8093 8094 8095 8096; do ss -ltn 2>/dev/null | grep -q ":${p} " && printf "%s " $p; done)"
    ' 2>/dev/null || true)
    HOST_USED_GPUS[$HOST]=$(echo "$SLOT_INFO" | grep "^GPUS:" | sed 's/^GPUS://')
    HOST_USED_PORTS[$HOST]=$(echo "$SLOT_INFO" | grep "^PORTS:" | sed 's/^PORTS://')
    log "slots $HOST: used_gpus=[${HOST_USED_GPUS[$HOST]:-}] used_ports=[${HOST_USED_PORTS[$HOST]:-}]"
done

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

# Phase 2: scan jobs, decide actions.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue

    : "${BACKEND:=vllm}"  # default if column missing (legacy rows)
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    JOB_SIGNATURE="${MAX_LEN}|${GPU_MEM}|${CONCS}|${PROFILES}|${EXTRA_ENV}"
    OLD_SIGNATURE=$(read_signature "$JID")
    if [[ "$STATUS" =~ ^(done|skipped|failed)$ && -n "$OLD_SIGNATURE" && "$OLD_SIGNATURE" != "$JOB_SIGNATURE" ]]; then
        log "$JID: job shape changed since terminal $STATUS; retrying as pending"
        STATUS="pending"
        write_status "$JID" pending
        echo "0" > "$STATE_DIR/${JID}.attempt"
        rm -f "$STATE_DIR/${JID}.max_len_override"
    elif [[ "$STATUS" =~ ^(skipped|failed)$ && -z "$OLD_SIGNATURE" && "$MODE" == "multi" && "$PROFILES" != *"swebench-multiturn"* && "$PROFILES" != *"terminalbench-multiturn"* ]]; then
        log "$JID: legacy terminal $STATUS predates profile filtering; retrying reduced-profile job"
        STATUS="pending"
        write_status "$JID" pending
        echo "0" > "$STATE_DIR/${JID}.attempt"
        rm -f "$STATE_DIR/${JID}.max_len_override"
    fi
    PREFIX=$(host_prefix "$HOST")
    RESULT_DIR_NAME="${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"
    OUT_DIR_REMOTE="/tmp/results/${RESULT_SCOPE}/${RESULT_DIR_NAME}"
    # Fallback for jobs launched before RESULT_SCOPE existed; completed jobs
    # still upload into the current R2 namespace to avoid legacy collisions.
    LEGACY_OUT_DIR_REMOTE="/tmp/results/${RESULT_DIR_NAME}"
    OUT_DIR_LOCAL="/tmp/bench_${RESULT_SCOPE//\//_}_${RESULT_DIR_NAME}"
    R2_DIR="${RESULT_SCOPE}/${RESULT_DIR_NAME}"

    case "$STATUS" in
        done|skipped|failed)
            continue
            ;;
        running)
            JOB_PORT=$(cat "$STATE_DIR/${JID}.port" 2>/dev/null || echo "8089")
            if [[ " ${HOST_USED_PORTS[$HOST]:-} " == *" $JOB_PORT "* ]]; then
                log "$JID: still running on $HOST:$JOB_PORT"
                continue
            fi
            # Grace period: weight-load + CUDA graph compilation.
            # vllm: ~3-5 min typical, 10 min max.
            # sglang: aggressive torch compilation, 10-15 min for large/MoE models.
            WARMUP_TIMEOUT=600
            [[ "$BACKEND" == "sglang" ]] && WARMUP_TIMEOUT=900
            STATUS_FILE="$STATE_DIR/${JID}.status"
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
                    REMOTE_SCRIPT_ALIVE=$(ssh "$HOST" "ps -eo args= | awk -v script='/tmp/inference-benchmark/scripts/${SCRIPT_NAME}' -v needle=' ${TP} ${SHORT} ${BACKEND} ' -v concs=' ${CONCS} ' '\$1 == \"bash\" && \$2 == script && index(\$0, needle) && index(\$0, concs) { found=1 } END { exit found ? 0 : 1 }' && echo yes" < /dev/null 2>/dev/null || true)
                    if [[ "$REMOTE_SCRIPT_ALIVE" == "yes" ]]; then
                        log "$JID: dispatched ${AGE}s ago (<$(( WARMUP_TIMEOUT / 60 ))min), still warming up on port $JOB_PORT"
                        JOB_GPUS=$(cat "$STATE_DIR/${JID}.gpus" 2>/dev/null || true)
                        [[ -n "$JOB_GPUS" ]] && HOST_USED_GPUS[$HOST]="${HOST_USED_GPUS[$HOST]:-} ${JOB_GPUS//,/ } "
                        HOST_USED_PORTS[$HOST]="${HOST_USED_PORTS[$HOST]:-} $JOB_PORT "
                        continue
                    fi
                    log "$JID: no listener and no live sweep process after ${AGE}s; finalizing early"
                fi
            fi
            log "$JID: slot idle after ${AGE}s warmup ($BACKEND) — finalizing"
            # All ssh/rsync/aws calls inside this `while read ... done <JOBS`
            # loop must close stdin (`< /dev/null`), otherwise they consume
            # the jobs file and iteration ends early — 2080ti rows were
            # silently skipped on any tick that also dispatched a 3090 job.
            REMOTE_SYNC_DIR=$(ssh "$HOST" "if [ -d '$OUT_DIR_REMOTE' ] && [ \$(ls -1 '$OUT_DIR_REMOTE' 2>/dev/null | wc -l) -gt 0 ]; then echo '$OUT_DIR_REMOTE'; elif [ -d '$LEGACY_OUT_DIR_REMOTE' ] && [ \$(ls -1 '$LEGACY_OUT_DIR_REMOTE' 2>/dev/null | wc -l) -gt 0 ]; then echo '$LEGACY_OUT_DIR_REMOTE'; fi" < /dev/null)
            if [[ -n "$REMOTE_SYNC_DIR" ]]; then
                COUNT=$(ssh "$HOST" "ls '$REMOTE_SYNC_DIR' 2>/dev/null | wc -l" < /dev/null)
                mkdir -p "$OUT_DIR_LOCAL"
                rsync -az "$HOST:$REMOTE_SYNC_DIR/" "$OUT_DIR_LOCAL/" < /dev/null >> "$LOG" 2>&1
                aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync \
                    "$OUT_DIR_LOCAL/" "s3://$BUCKET/results/$R2_DIR/" < /dev/null >> "$LOG" 2>&1
                write_status "$JID" done
                log "$JID: DONE ($COUNT files uploaded to results/$R2_DIR, warmup=${AGE}s backend=$BACKEND)"
            else
                # Widen detection to include vLLM's KV-cache budget failure
                # ("Available KV cache memory: -...", "No available memory for
                # the cache blocks") so model+cudagraph oversubscription also
                # triggers the halved-max_len retry path.
                OOM=$(ssh "$HOST" "grep -l 'OutOfMemoryError\\|out of memory\\|No available memory for the cache blocks\\|Available KV cache memory: -' /tmp/vllm_${JOB_PORT}.log 2>/dev/null" < /dev/null || true)
                ATT=$(read_attempt "$JID")
                if [[ -n "$OOM" && "$ATT" -lt 1 ]]; then
                    bump_attempt "$JID"
                    write_status "$JID" pending
                    NEW_MAX=$((MAX_LEN / 2))
                    [[ "$NEW_MAX" -lt 2048 ]] && NEW_MAX=2048
                    echo "$NEW_MAX" > "$STATE_DIR/${JID}.max_len_override"
                    log "$JID: OOM detected, retry with max_len=$NEW_MAX"
                else
                    write_status "$JID" skipped
                    log "$JID: SKIPPED (zero results, attempt=$ATT, oom_log=$OOM)"
                fi
            fi
            ;;
        pending)
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

            OVERRIDE_FILE="$STATE_DIR/${JID}.max_len_override"
            if [[ -f "$OVERRIDE_FILE" ]]; then
                MAX_LEN=$(cat "$OVERRIDE_FILE")
            fi
            PY=$(host_python "$HOST" "$BACKEND")
            if [[ "$BACKEND" == "sglang" ]]; then
                SCRIPT="sweep_all_profiles_sglang.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles_sglang.sh"
            else
                SCRIPT="sweep_all_profiles.sh"
                [[ "$MODE" == "multi" ]] && SCRIPT="sweep_multiturn_profiles.sh"
            fi

            claim_slot "$HOST" "$SLOT_GPUS" "$SLOT_PORT"
            echo "$SLOT_PORT" > "$STATE_DIR/${JID}.port"
            echo "$SLOT_GPUS" > "$STATE_DIR/${JID}.gpus"
            write_signature "$JID" "$JOB_SIGNATURE"

            log "$JID: dispatching on $HOST:$SLOT_PORT gpus=[$SLOT_GPUS] ($BACKEND, max_len=$MAX_LEN, mode=$MODE)"
            write_status "$JID" running

            # Build env: PORT + CUDA_VISIBLE_DEVICES (unless cell already pins)
            SLOT_ENV="PORT=$SLOT_PORT"
            if [[ -z "$CELL_CVD" ]]; then
                SLOT_ENV="$SLOT_ENV CUDA_VISIBLE_DEVICES=$SLOT_GPUS"
            fi

            CMD="$SLOT_ENV ${EXTRA_ENV} bash /tmp/inference-benchmark/scripts/${SCRIPT} \
                ${MODEL_PATH} ${TP} ${SHORT} ${BACKEND} ${OUT_DIR_REMOTE} \
                ${PY} ${GPU_MEM} ${MAX_LEN} \"${CONCS}\" \"${PROFILES}\""
            REMOTE_LOG="/tmp/bench_${SHORT}_tp${TP}_${MODE}_${BACKEND}_p${SLOT_PORT}.log"
            ssh "$HOST" "setsid bash -c '${CMD}' > '${REMOTE_LOG}' 2>&1 </dev/null &" < /dev/null
            log "$JID: dispatched"
            ;;
    esac
done < "$JOBS_FILE"

# Publish sweep-state.json to R2 so the dashboard reflects the latest cell
# status (pending/running/done/skipped/known_oom). Non-fatal — if this
# fails, the tick still succeeds; the next tick will republish.
python3 "$REPO_ROOT/inference-benchmark/scripts/publish_sweep_state.py" \
    --endpoint "$EP" --bucket "$BUCKET" --profile "$PROFILE" \
    >> "$LOG" 2>&1 || log "publish_sweep_state.py failed"

log "tick complete"
