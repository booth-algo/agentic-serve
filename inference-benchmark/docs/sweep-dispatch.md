# Sweep Dispatch Guide

How to run the benchmark sweep pipeline end-to-end.

## Files

| File | Role |
|------|------|
| `scripts/sweep.yaml` | Single source of truth. Defines hosts, models, presets, cells (host x model x tp x mode). |
| `scripts/compile_sweep.py` | Reads sweep.yaml, emits `scripts/bench_jobs.txt`. Handles feasibility, OOM, profile infeasibility. |
| `scripts/bench_jobs.txt` | Generated job list consumed by the orchestrator. **Never edit directly.** |
| `scripts/bench_orchestrator.sh` | Multi-slot GPU scheduler. Reads bench_jobs.txt, dispatches via SSH, uploads results to R2. |
| `scripts/publish_sweep_state.py` | Publishes sweep-state.json to R2 (runs at end of each orchestrator tick). |
| `scripts/sweep_monitor.py` | Read-only diagnostic. Compares local state, R2, and sweep.yaml. |
| `scripts/reconcile_sweep_coverage.py` | Coverage-aware repair tool. Maps missing profile/concurrency rows back to coarse bench jobs and can reset stale terminal states. |
| `scripts/sweep_all_profiles.sh` | Single-turn benchmark runner (launched on GPU host by orchestrator). |
| `scripts/sweep_multiturn_profiles.sh` | Multi-turn benchmark runner (launched on GPU host by orchestrator). |

## Dispatch flow

```
sweep.yaml  ──>  compile_sweep.py  ──>  bench_jobs.txt  ──>  bench_orchestrator.sh
                                                                  │
                                                    SSH to GPU hosts
                                                    ├── sweep_all_profiles.sh (single-turn)
                                                    └── sweep_multiturn_profiles.sh (multi-turn)
                                                                  │
                                                    Results → R2 (s3://agent-bench/results/current/)
                                                    State  → R2 (sweep-state.json)
```

## Step-by-step

```bash
cd /root/agentic-serve/inference-benchmark

# 1. Edit sweep.yaml (add/remove cells, change presets)
vim scripts/sweep.yaml

# 2. Compile to bench_jobs.txt (DO NOT manually filter the output)
python3 scripts/compile_sweep.py

# 3. Sync code to all GPU hosts
for host in gpu-4 3090 2080ti; do
  rsync -az --delete --exclude='dashboard/node_modules' --exclude='dashboard/dist' --exclude='results/' \
    . "$host:/tmp/inference-benchmark/" < /dev/null
done
rsync -az --delete --exclude='dashboard/' --exclude='node_modules/' --exclude='results/' \
  . "h100:/tmp/inference-benchmark/" < /dev/null

# 4. Launch orchestrator (single tick, then exits)
bash scripts/bench_orchestrator.sh

# 5. For continuous operation, use a monitor loop (ticks every 120s)
# The orchestrator runs one tick per invocation. Wrap in a loop:
while true; do
  bash scripts/bench_orchestrator.sh
  sleep 120
done
```

## Orchestrator behavior

- **Multi-slot**: Dispatches multiple jobs per host using free GPUs + ports.
- **Signature detection**: Compares `max_len|gpu_mem|concs|profiles|extra_env` against stored signature. Re-runs "done" jobs if config changed.
- **Skip logic**: The sweep scripts (`sweep_all_profiles.sh`) skip result files that already exist. Re-dispatching a "done" job is cheap if results are present.
- **Warmup timeout**: 10 min. If a server doesn't start serving within 10 min, the job is considered failed.
- **OOM retry**: First OOM halves max_len and retries once.
- **R2 upload**: `aws s3 sync` on job completion. Results go to `s3://agent-bench/results/current/<hw>_<model>_<tp>_<backend>/`.

## State files

Stored in `/tmp/bench_jobs/state/` (local, not on GPU hosts):

| File | Content |
|------|---------|
| `<job_id>.status` | `pending`, `running`, `done`, `skipped`, `failed`, `known_oom` |
| `<job_id>.signature` | `max_len\|gpu_mem\|concs\|profiles\|extra_env` |
| `<job_id>.port` | Port number (while running) |
| `<job_id>.gpus` | GPU IDs (while running) |

Job ID format: `<host>_<model_short>_tp<N>_<mode>[_sglang]`

## Presets

| Preset | max_len | Concurrencies | Profiles |
|--------|---------|--------------|----------|
| `single_small` | 32768 | 1,10,20,40,80,160,256,320 | chat-singleturn, coding-singleturn |
| `single_medium` | 16384 | 1,10,20,40,80,160,256,320 | chat-singleturn |
| `single_large` | 8192 | 1,10,20,40,80,160,256,320 | chat-singleturn |
| `single_tight` | 4096 | 1,10,20,40,80,160,256,320 | chat-singleturn |
| `multi_small` | 32768 | 5,20,40,80,160 | chat/swebench/terminalbench/osworld-multiturn |
| `multi_medium` | 8192 | 5,20,40,80,160 | chat/swebench/terminalbench/osworld-multiturn |
| `fixed_single` | 32768 | 5,40,80,200,320 | chat-singleturn, coding-singleturn |
| `fixed_multi` | 8192 | 5,40,80,200,320 | chat-multiturn, osworld-multiturn |

## Known constraints

- **sglang**: Availability is host-specific and changes over time. Confirm with
  the current host env before assuming support; the orchestrator will mark
  zero-result launches as skipped after warmup.
- **Qwen3.5-27B on H100**: Model not downloaded. Jobs will fail.
- **gpt-oss-20b on 2080ti**: MXFP4 needs sm80+. 2080ti is sm75. Known OOM in sweep.yaml.
- **Multi-turn context overflow**: `fixed_multi` uses max_len=8192 to avoid swebench/terminalbench sessions overflowing 32K+ context at high concurrency.
- **coding-singleturn**: Requires max_len >= 32768 (17K ISL). Only feasible on `single_small` and `fixed_single` presets.

## Monitoring

```bash
# Quick status against fixed-scope data published to the website
python3 scripts/sweep_monitor.py --scope fixed --limit 5

# Coverage-aware job reconciliation.
# Dry-run: reports missing fixed-scope points and which done/skipped/failed jobs hide them.
python3 scripts/reconcile_sweep_coverage.py --scope fixed --limit 30

# Repair stale completed jobs after reviewing the dry-run.
# Default reset target is only `done`, so old partial jobs rerun without
# automatically relaunching skipped/failed jobs that may need OOM classification.
python3 scripts/reconcile_sweep_coverage.py --scope fixed --reset-stale --write-sweep-state

# Broader repair if the dry-run shows local blocking state that should not be
# terminal for a runnable sweep.yaml row.
python3 scripts/reconcile_sweep_coverage.py --scope fixed --reset-stale --reset-statuses done,known_oom --write-sweep-state

# If bench_jobs.txt drifted from sweep.yaml, rewrite it from the YAML source of truth.
python3 scripts/reconcile_sweep_coverage.py --scope fixed --write-bench-jobs

# Write a bench_jobs-format subset for the jobs that still have missing coverage.
python3 scripts/reconcile_sweep_coverage.py --scope fixed --write-missing-jobs

# Continuous fixed-scope dispatcher loop with a duplicate-run lock
setsid -f bash -lc 'cd /root/agentic-serve/inference-benchmark; exec flock -n /tmp/fixed-scope-sweep-loop.lock bash -lc '\''while true; do date -Is; bash scripts/bench_orchestrator.sh; sleep 120; done'\''' >> /tmp/fixed-scope-sweep-loop.log 2>&1

# Continuous read-only progress reporter.
# Writes the latest per-host/per-GPU markdown report and appends a history log.
setsid -f bash -lc 'cd /root/agentic-serve/inference-benchmark; exec flock -n /tmp/sweep-progress-reporter.lock python3 scripts/sweep_progress_report.py --interval-seconds 300' >> /tmp/sweep-progress-reporter.log 2>&1

# Read the latest progress snapshot
cat /tmp/sweep-progress-latest.md

# Watch reporter activity
tail -f /tmp/sweep-progress-reporter.log

# Stop the reporter
pkill -f sweep-progress-reporter.lock

# Dashboard rebuild
gh workflow run "Rebuild Dashboard Data" --repo booth-algo/agentic-serve
```

## Recovery after crash/restart

```bash
cd /root/agentic-serve/inference-benchmark

# 1. Recompile (always start here)
python3 scripts/compile_sweep.py

# 2. Reset stale running states
for f in /tmp/bench_jobs/state/*.status; do
  [ "$(cat "$f")" = "running" ] && echo "pending" > "$f"
done

# 3. Sync + launch
# (same as steps 3-5 above)
```
