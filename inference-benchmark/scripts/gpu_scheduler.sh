#!/usr/bin/env bash
# =============================================================================
# GPU Scheduler — Parallel model execution across multiple GPUs
#
# Source this file from orchestrator scripts:
#   source "$REPO_ROOT/scripts/gpu_scheduler.sh"
#
# Provides:
#   - Slot-based GPU assignment (TP=1: 4 slots, TP=2: 2 slots, TP=4: 1 slot)
#   - Parallel server launch + benchmark execution
#   - Per-slot log isolation
#   - Targeted server kill (per-port, not blanket pkill)
# =============================================================================

# Slot definitions: CUDA_VISIBLE_DEVICES|PORT
# Ports 8000-8001 are behind nginx on RunPod — use 9000+ for direct access
SLOTS_TP1=("0|9000" "1|9001" "2|9002" "3|9003")
SLOTS_TP2=("0,1|9000" "2,3|9001")
SLOTS_TP4=("0,1,2,3|9000")

SCHED_MAX_SERVER_WAIT="${MAX_SERVER_WAIT:-1800}"
SERVER_PYTHON="${SERVER_PYTHON:-$PYTHON}"

slog() { echo -e "\033[0;36m[SCHED]\033[0m $1"; }
slot_log() { echo -e "\033[0;35m[SLOT$1]\033[0m $2"; }

# Kill server process group by stored PID file
# Each server is launched with setsid, so kill -PGID kills the whole tree
kill_server_on_slot() {
    local slot_idx="$1"
    local pidfile="/tmp/sched_server_slot${slot_idx}.pid"
    if [[ -f "$pidfile" ]]; then
        local pid
        pid=$(cat "$pidfile")
        # Kill the entire process group (setsid made it a group leader)
        kill -- -"$pid" 2>/dev/null || true
        sleep 3
        kill -9 -- -"$pid" 2>/dev/null || true
        sleep 2
        rm -f "$pidfile"
    fi
}

# Wait for server health on a specific port
wait_for_server_on_port() {
    local port="$1"
    local max_wait="${2:-$SCHED_MAX_SERVER_WAIT}"
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            return 1
        fi
    done
    return 0
}

# Start a vLLM server on a specific GPU slot
start_vllm_on_slot() {
    local model_path="$1" tp="$2" extra_flags="$3" max_len="$4" gpu_mem="$5" tag="$6" cuda_devs="$7" port="$8"

    local cmd="CUDA_VISIBLE_DEVICES=$cuda_devs $PYTHON -m vllm.entrypoints.openai.api_server \
        --model $model_path \
        --tensor-parallel-size $tp \
        --port $port \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --gpu-memory-utilization $gpu_mem \
        --max-model-len $max_len \
        --enable-prefix-caching \
        --api-key $API_KEY"

    [[ -n "$extra_flags" ]] && cmd="$cmd $extra_flags"

    setsid bash -c "$cmd" > "/tmp/server_${tag}_port${port}.log" 2>&1 &
    echo $!
}

# Start a SGLang server on a specific GPU slot
start_sglang_on_slot() {
    local model_path="$1" tp="$2" extra_flags="$3" max_len="$4" gpu_mem="$5" tag="$6" cuda_devs="$7" port="$8"

    local cmd="CUDA_VISIBLE_DEVICES=$cuda_devs $SERVER_PYTHON -m sglang.launch_server \
        --model-path $model_path \
        --tp $tp \
        --port $port \
        --dtype bfloat16 \
        --host 0.0.0.0 \
        --mem-fraction-static $gpu_mem \
        --context-length $max_len \
        --api-key $API_KEY"

    [[ -n "$extra_flags" ]] && cmd="$cmd $extra_flags"

    setsid bash -c "$cmd" > "/tmp/server_${tag}_port${port}.log" 2>&1 &
    echo $!
}

# Run a single job on an assigned slot (called in background subshell)
# Args: engine, name, path, tp, sglang_extra, vllm_extra, max_len, gpu_mem, cuda_devs, port, slot_idx, benchmark_callback
run_slot_job() {
    local engine="$1" name="$2" path="$3" tp="$4"
    local sglang_extra="$5" vllm_extra="$6" max_len="$7" gpu_mem="$8"
    local cuda_devs="$9" port="${10}" slot_idx="${11}" benchmark_callback="${12}"

    local tag="${name}_tp${tp}_${engine}"
    local status_file="/tmp/sched_status_${tag}_slot${slot_idx}"

    slot_log "$slot_idx" "START $name TP=$tp on GPU=$cuda_devs port=$port"

    # Clean slot first
    kill_server_on_slot "$slot_idx"

    # Start server
    local server_pid
    if [[ "$engine" == "sglang" ]]; then
        server_pid=$(start_sglang_on_slot "$path" "$tp" "$sglang_extra" "$max_len" "$gpu_mem" "$tag" "$cuda_devs" "$port")
    else
        server_pid=$(start_vllm_on_slot "$path" "$tp" "$vllm_extra" "$max_len" "$gpu_mem" "$tag" "$cuda_devs" "$port")
    fi

    # Save PID for process group cleanup
    echo "$server_pid" > "/tmp/sched_server_slot${slot_idx}.pid"

    slot_log "$slot_idx" "Server PID=$server_pid, waiting for health..."

    if ! wait_for_server_on_port "$port"; then
        slot_log "$slot_idx" "FAILED $name TP=$tp — server didn't start. Log: /tmp/server_${tag}_port${port}.log"
        kill_server_on_slot "$slot_idx"
        echo "FAIL" > "$status_file"
        return 1
    fi

    slot_log "$slot_idx" "Server ready, running benchmarks..."

    # Call the benchmark function (provided by orchestrator)
    $benchmark_callback "$engine" "$name" "$path" "$tp" "$port" "$slot_idx"

    # Cleanup
    kill_server_on_slot "$slot_idx"
    slot_log "$slot_idx" "DONE $name TP=$tp"
    echo "OK" > "$status_file"
    return 0
}

# Run a batch of jobs at a given TP level in parallel
# Args: tp, engine, benchmark_callback, job1, job2, ...
#   Each job is: name|path|sglang_extra|vllm_extra|max_len|gpu_mem
run_parallel_batch() {
    local tp="$1" engine="$2" benchmark_callback="$3"
    shift 3
    local jobs=("$@")

    if [[ ${#jobs[@]} -eq 0 ]]; then
        return 0
    fi

    # Select slots for this TP level
    local -a slots
    case "$tp" in
        1) slots=("${SLOTS_TP1[@]}") ;;
        2) slots=("${SLOTS_TP2[@]}") ;;
        4) slots=("${SLOTS_TP4[@]}") ;;
        *) slog "Unknown TP=$tp"; return 1 ;;
    esac

    local num_slots=${#slots[@]}
    slog "TP=$tp batch: ${#jobs[@]} jobs, $num_slots parallel slots"

    # Track active slot PIDs
    declare -A active_pids    # slot_idx -> PID
    local job_idx=0
    local total_ok=0
    local total_fail=0

    while [[ $job_idx -lt ${#jobs[@]} ]] || [[ ${#active_pids[@]} -gt 0 ]]; do
        # Fill empty slots with queued jobs
        for i in $(seq 0 $((num_slots - 1))); do
            if [[ -z "${active_pids[$i]:-}" ]] && [[ $job_idx -lt ${#jobs[@]} ]]; then
                local job="${jobs[$job_idx]}"
                job_idx=$((job_idx + 1))

                IFS='|' read -r name path sglang_extra vllm_extra max_len gpu_mem <<< "$job"
                IFS='|' read -r cuda_devs port <<< "${slots[$i]}"

                (
                    set +e
                    run_slot_job "$engine" "$name" "$path" "$tp" \
                        "$sglang_extra" "$vllm_extra" "$max_len" "$gpu_mem" \
                        "$cuda_devs" "$port" "$i" "$benchmark_callback"
                ) &
                active_pids[$i]=$!
            fi
        done

        # Wait for any slot to finish
        if [[ ${#active_pids[@]} -gt 0 ]]; then
            local finished_pid
            wait -n -p finished_pid "${active_pids[@]}" 2>/dev/null || true

            # Find which slot finished
            for i in "${!active_pids[@]}"; do
                if ! kill -0 "${active_pids[$i]}" 2>/dev/null; then
                    local tag_file="/tmp/sched_status_*_slot${i}"
                    local status
                    status=$(cat $tag_file 2>/dev/null | tail -1)
                    if [[ "$status" == "OK" ]]; then
                        total_ok=$((total_ok + 1))
                    else
                        total_fail=$((total_fail + 1))
                    fi
                    rm -f $tag_file
                    unset active_pids[$i]
                fi
            done
        fi
    done

    slog "TP=$tp batch complete: $total_ok ok, $total_fail failed"
}

# Kill all servers on all scheduler ports
kill_all_scheduled_servers() {
    for slot in 0 1 2 3; do
        kill_server_on_slot "$slot"
    done
}
