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

mkdir -p "$STATE_DIR"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

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
        esac
    else
        case "$host" in
            gpu-4)       echo "/data/kevinlau/miniconda3/bin/python" ;;
            3090|2080ti) echo "/home/kevinlau/miniconda3/envs/vllm/bin/python" ;;
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

# Phase 1: detect which hosts are busy with vllm.
declare -A HOST_BUSY
for HOST in gpu-4 3090 2080ti; do
    # Check for a listener on the vllm API port (8089). Prior versions used
    # pgrep -f "vllm.entrypoints.openai" which self-matched the ssh wrapper's
    # own cmdline and reported every host busy on every tick.
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "$HOST" 'ss -ltn 2>/dev/null | awk "\$4 ~ /:8089\$/ {found=1} END{exit !found}"' 2>/dev/null; then
        HOST_BUSY[$HOST]=1
        log "host busy: $HOST"
    else
        HOST_BUSY[$HOST]=0
    fi
done

# Phase 2: scan jobs, decide actions.
while IFS='|' read -r HOST MODEL_PATH TP SHORT MODE BACKEND MAX_LEN GPU_MEM CONCS PROFILES EXTRA_ENV || [[ -n "$HOST" ]]; do
    HOST=$(echo "$HOST" | tr -d ' ')
    [[ -z "$HOST" || "${HOST:0:1}" == "#" ]] && continue

    : "${BACKEND:=vllm}"  # default if column missing (legacy rows)
    JID=$(job_id "$HOST" "$SHORT" "$TP" "$MODE" "$BACKEND")
    STATUS=$(read_status "$JID")
    PREFIX=$(host_prefix "$HOST")
    OUT_DIR_REMOTE="/tmp/results/${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"
    OUT_DIR_LOCAL="/tmp/bench_${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"
    R2_DIR="${PREFIX}_${SHORT}_tp${TP}_${BACKEND}"

    case "$STATUS" in
        done|abandoned|failed)
            continue
            ;;
        running)
            if [[ "${HOST_BUSY[$HOST]}" == "1" ]]; then
                log "$JID: still running on $HOST"
                continue
            fi
            # Grace period: 70B/72B weight-load takes 3-5 min. If the
            # status file was written <10 min ago, the job is probably
            # still coming up (port 8089 not yet listening). Don't
            # finalize prematurely.
            STATUS_FILE="$STATE_DIR/${JID}.status"
            if [[ -f "$STATUS_FILE" ]]; then
                AGE=$(( $(date +%s) - $(stat -c %Y "$STATUS_FILE") ))
                if [[ "$AGE" -lt 600 ]]; then
                    log "$JID: dispatched ${AGE}s ago (<10min), still warming up"
                    # Treat host as busy so the next pending job on it
                    # doesn't get dispatched on top of a warming vllm.
                    HOST_BUSY[$HOST]=1
                    continue
                fi
            fi
            log "$JID: host idle — finalizing"
            # All ssh/rsync/aws calls inside this `while read ... done <JOBS`
            # loop must close stdin (`< /dev/null`), otherwise they consume
            # the jobs file and iteration ends early — 2080ti rows were
            # silently skipped on any tick that also dispatched a 3090 job.
            COUNT=$(ssh "$HOST" "ls $OUT_DIR_REMOTE 2>/dev/null | wc -l" < /dev/null)
            if [[ "$COUNT" -gt 0 ]]; then
                mkdir -p "$OUT_DIR_LOCAL"
                rsync -az "$HOST:$OUT_DIR_REMOTE/" "$OUT_DIR_LOCAL/" < /dev/null >> "$LOG" 2>&1
                aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync \
                    "$OUT_DIR_LOCAL/" "s3://$BUCKET/results/$R2_DIR/" < /dev/null >> "$LOG" 2>&1
                write_status "$JID" done
                log "$JID: DONE ($COUNT files uploaded)"
            else
                # Widen detection to include vLLM's KV-cache budget failure
                # ("Available KV cache memory: -...", "No available memory for
                # the cache blocks") so model+cudagraph oversubscription also
                # triggers the halved-max_len retry path.
                OOM=$(ssh "$HOST" "grep -l 'OutOfMemoryError\\|out of memory\\|No available memory for the cache blocks\\|Available KV cache memory: -' /tmp/vllm_8089.log 2>/dev/null" < /dev/null || true)
                ATT=$(read_attempt "$JID")
                if [[ -n "$OOM" && "$ATT" -lt 1 ]]; then
                    bump_attempt "$JID"
                    write_status "$JID" pending
                    NEW_MAX=$((MAX_LEN / 2))
                    [[ "$NEW_MAX" -lt 2048 ]] && NEW_MAX=2048
                    echo "$NEW_MAX" > "$STATE_DIR/${JID}.max_len_override"
                    log "$JID: OOM detected, retry with max_len=$NEW_MAX"
                else
                    write_status "$JID" abandoned
                    log "$JID: ABANDONED (zero results, attempt=$ATT, oom_log=$OOM)"
                fi
            fi
            ;;
        pending)
            if [[ "${HOST_BUSY[$HOST]}" == "1" ]]; then
                continue
            fi
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
            log "$JID: dispatching on $HOST ($BACKEND, max_len=$MAX_LEN, mode=$MODE)"
            write_status "$JID" running
            HOST_BUSY[$HOST]=1
            CMD="${EXTRA_ENV} bash /tmp/inference-benchmark/scripts/${SCRIPT} \
                ${MODEL_PATH} ${TP} ${SHORT} ${BACKEND} ${OUT_DIR_REMOTE} \
                ${PY} ${GPU_MEM} ${MAX_LEN} \"${CONCS}\" \"${PROFILES}\""
            # setsid + </dev/null lets the process survive ssh disconnect
            # reliably. Per-job remote log for debugging (vllm_8089.log
            # rotates per sweep and loses history).
            REMOTE_LOG="/tmp/bench_${SHORT}_tp${TP}_${MODE}_${BACKEND}.log"
            # Local `< /dev/null` on the ssh call too — the `</dev/null`
            # inside the quoted command only redirects the REMOTE shell.
            # Without this, ssh slurps the outer while-read jobs file.
            ssh "$HOST" "setsid bash -c '${CMD}' > '${REMOTE_LOG}' 2>&1 </dev/null &" < /dev/null
            log "$JID: dispatched"
            ;;
    esac
done < "$JOBS_FILE"

# Publish sweep-state.json to R2 so the dashboard reflects the latest cell
# status (pending/running/done/abandoned/known_oom). Non-fatal — if this
# fails, the tick still succeeds; the next tick will republish.
python3 "$REPO_ROOT/inference-benchmark/scripts/publish_sweep_state.py" \
    --endpoint "$EP" --bucket "$BUCKET" --profile "$PROFILE" \
    >> "$LOG" 2>&1 || log "publish_sweep_state.py failed"

log "tick complete"
