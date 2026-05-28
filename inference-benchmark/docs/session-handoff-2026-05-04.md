# Session Handoff — 2026-05-04 12:45 UTC

## Current State

### Coverage Numbers
- **Expected cells (sweep.yaml)**: 2414 profile-concurrency points
- **On R2 data.json**: 843 current + 99 fixed + 4814 archive = 5756 total rows
- **Missing from R2**: ~1625 cells
- **Dashboard scope fix**: commit 829367a makes fixed scope visible on coverage page

### The Problem
We're only producing 99 fixed-scope data rows despite running ~70 "done" fixed-scope jobs. Each job should produce 5 conc × 2-4 profiles = 10-20 cells. 70 jobs × ~10 = ~700 cells expected, but only 99 on R2. **The data pipeline is broken — results exist on hosts but aren't making it to R2 data.json properly.**

### Possible causes:
1. **build-data.ts filtering**: The strict underloaded filter (`successful_requests < concurrency`) may be dropping fixed-scope results
2. **R2 upload gaps**: Many jobs "completed" by finding old result files (skip logic), uploading OLD current-scope files instead of running new fixed-scope benchmarks
3. **Scope tagging**: Old result files in the same directory have `dashboard_scope: "current"`, new ones have `"fixed"`. The directory mixes both — build-data.ts counts each file's scope individually

### Active Pipeline
- **Monitor**: `b8fn5rjiv` — auto-ticking orchestrator every 120s
- **Orchestrator**: Running fixed-scope only (132 jobs in bench_jobs.txt, all with concs `[5,40,80,200,320]`)
- **State**: 69 done, 5 running, 28 pending (all on 3090 + H100)
- **a100**: DONE (28 done, 12 skipped=sglang)
- **2080ti**: DONE (12 done, 6 skipped=sglang)
- **3090**: Running (14 done, running multi-slot)
- **H100**: Running (10 done, 24 pending, zifengding intermittent)

### Dashboard Crons
- `f4d908eb` — Dashboard rebuild every 30 min
- `8bef8d07` — Sweep progress check every 15 min

## Key Files

| File | Purpose |
|------|---------|
| `scripts/sweep.yaml` | Source of truth — hosts, models, presets, cells |
| `scripts/compile_sweep.py` | Generates bench_jobs.txt from sweep.yaml |
| `scripts/bench_jobs.txt` | Currently FIXED-SCOPE ONLY (filtered to `5 40 80 200 320` concs) |
| `scripts/bench_orchestrator.sh` | Multi-slot GPU scheduler, SSH dispatch, R2 upload |
| `scripts/publish_sweep_state.py` | Publishes sweep-state.json to R2 each tick |
| `scripts/sweep_monitor.py` | Read-only diagnostic (Codex-built) |
| `dashboard/scripts/build-data.ts` | Builds data.json from R2 results — THIS IS WHERE CELLS GET LOST |
| `dashboard/scripts/validate-data.ts` | Validates data.json — fixed to accept "fixed" scope (PR #55) |
| `dashboard/src/components/CoveragePage.tsx` | Coverage grid — commit 829367a adds fixed scope |

## Presets

```yaml
fixed_single:
  max_len: 32768
  gpu_mem: 0.85
  concurrencies: [5, 40, 80, 200, 320]
  profiles: [chat-singleturn, coding-singleturn]

fixed_multi:
  max_len: 8192      # Changed from 32768 to avoid context overflow
  gpu_mem: 0.85
  concurrencies: [5, 40, 80, 200, 320]
  profiles: [chat-multiturn, osworld-multiturn]
  # swebench-multiturn and terminalbench-multiturn REMOVED — overflow at high conc
```

## Fixes Made This Session

1. **num_sessions bug** — `fd2aabe`: Distributional profiles 10→100 sessions (50 osworld)
2. **Orchestrator signature detection** — Added `done` to signature mismatch check, auto-resets stale jobs
3. **sglang warmup timeout** — 15 min (was 10 min for all backends)
4. **Warmup time logging** — `warmup=Xs backend=vllm/sglang` in DONE messages
5. **validate-data.ts** — PR #55: Accept "fixed" scope
6. **Dashboard coverage** — 829367a: Fixed scope follows canonical coverage surface
7. **fixed_multi max_len** — 32768→8192 to avoid swebench/terminalbench context overflow

## Known Issues

### Why only 99 fixed rows on R2?
The sweep scripts SKIP existing result files (`if [ -f "$OUT_FILE" ]; then skip`). When a job dispatches, it finds OLD current-scope files (conc=40, 80, etc.) and skips them. It only runs NEW concurrency levels (conc=5, 200, 320). But the OLD files have `dashboard_scope: "current"` — they don't contribute to "fixed" scope count.

**The fix**: Either:
1. Delete old result files and re-run everything from scratch
2. Or modify the sweep scripts to force-overwrite when the scope changes
3. Or have build-data.ts count cells by concurrency level regardless of scope

### sglang on a100/h100
No sglang conda env installed. Jobs timeout after 10-15 min. ~20+ cells permanently skipped.

### Qwen3.5-27B on H100
Model not downloaded. All H100 Qwen3.5-27B jobs skipped.

### Multi-turn context overflow
swebench-multiturn (94 avg turns) and terminalbench-multiturn (75 avg turns) overflow max_len at conc=200/320. Removed from fixed_multi preset. Only chat-multiturn and osworld-multiturn run in fixed scope.

### H100 setsid dispatch
Intermittent — some models fail to start via `setsid bash -c '...'` on H100. Works when run manually. Possibly quoting or env issue.

## Host Inventory

| Host | GPUs | vllm | sglang | Notes |
|------|------|------|--------|-------|
| a100 | 8x A100-40GB | YES | NO | bobgu uses GPUs 0-1 intermittently |
| 3090 | 8x RTX 3090 | YES | YES | Qwen3.5-9B tp1 OOM on 24GB |
| 2080ti | 8x RTX 2080Ti | YES | YES (sm75 limits) | gpt-oss-20b needs sm80+ |
| H100 | 8x H100-80GB | YES | NO | zifengding reclaims GPUs, Qwen3.5-27B missing |

## Recovery / Resume

```bash
cd /root/agentic-serve/inference-benchmark

# 1. Compile from sweep.yaml (DO NOT manually filter)
python3 scripts/compile_sweep.py

# 2. If fixed-scope only: filter to keep only fixed entries
# grep '5 40 80 200 320' scripts/bench_jobs.txt + comments > filtered
# BUT this causes the duplicate entry problem — better to run full and let
# signature detection handle it

# 3. Sync to hosts
for host in a100 3090 2080ti; do
  rsync -az --delete --exclude='dashboard/node_modules' --exclude='dashboard/dist' --exclude='results/' \
    . "$host:/tmp/inference-benchmark/" < /dev/null
done
rsync -az --delete --exclude='dashboard/' --exclude='node_modules/' --exclude='results/' \
  . "h100:/tmp/inference-benchmark/" < /dev/null

# 4. Reset stale states
for f in /tmp/bench_jobs/state/*.status; do
  [ "$(cat "$f")" = "running" ] && echo "pending" > "$f"
done

# 5. Launch orchestrator loop (or use Monitor for auto-ticking)
while true; do bash scripts/bench_orchestrator.sh; sleep 120; done

# 6. Dashboard rebuild
gh workflow run "Rebuild Dashboard Data" --repo booth-algo/agentic-serve
```

## State Files Location
- `/tmp/bench_jobs/state/` — job status, signatures, ports, gpus
- `/tmp/results_staging/` — local copies of host results before R2 upload
- `/tmp/bench_jobs/orchestrator.log` — orchestrator output

## Critical TODO
**Investigate why 70 "done" fixed jobs = only 99 fixed rows on R2 data.json.** The bottleneck is somewhere between result file generation → R2 upload → build-data.ts processing. Start with checking a specific result file's `dashboard_scope` field and tracing through build-data.ts.

## Codex Takeover — 2026-05-04 12:58 UTC

### What was found
- Claude's scheduled task/monitor was not a root crontab (`crontab -l` was empty).
- The last Claude-driven orchestrator tick in `/tmp/bench_orchestrator.log` was `2026-05-04T12:46:34+00:00`; no dispatcher process was alive after that.
- `sweep_monitor.py` could not check fixed scope at all: its CLI only accepted `current/archive/all`.
- `publish_sweep_state.py` emitted duplicate current/fixed sweep cells without a scope field, so monitor/UI state could not cleanly distinguish current vs fixed cells.
- The root data-generation bug was in the sweep scripts: they skipped any existing JSON by filename without checking `config.dashboard_scope`. Old `current` files at C=40/80/320 therefore blocked fixed reruns.

### Fixes made
- `scripts/publish_sweep_state.py`
  - Adds `data_scope: "fixed"` for cells whose preset starts with `fixed_`; otherwise `current`.
  - Adds matching `data_scope` to `profile_infeasible` records.
- `scripts/sweep_monitor.py`
  - Adds `--scope fixed`.
  - Filters expected points by `data_scope`.
  - Includes scope in cell/profile keys so current and fixed duplicates no longer collapse.
- `dashboard/src/types-sweep.ts`
  - Adds optional `data_scope` to sweep-state types.
- `dashboard/src/components/CoveragePage.tsx`
  - Filters sweep-state statuses and profile infeasibility by selected scope.
- `scripts/sweep_all_profiles*.sh` and `scripts/sweep_multiturn_profiles*.sh`
  - Set `DASHBOARD_SCOPE="${DASHBOARD_SCOPE:-fixed}"`.
  - Pass `--scope "$DASHBOARD_SCOPE"` to the benchmark runner.
  - Skip an existing result only when its JSON has matching `config.dashboard_scope`.
  - Rerun/overwrite stale or missing-scope JSONs, which fixes the old-current-file blocker.

### Validation
- `bash -n` passed for all four sweep launcher scripts plus `bench_orchestrator.sh`.
- `python3 -m py_compile` passed for `publish_sweep_state.py`, `sweep_monitor.py`, and `compile_sweep.py`.
- `npm run build` passed in `inference-benchmark/dashboard`.
- One foreground `bash scripts/bench_orchestrator.sh` tick succeeded:
  - Finalized stale H100/3090 jobs.
  - Uploaded result dirs to `results/current/...`.
  - Dispatched new H100 and 3090 jobs.
  - Published `json/current/sweep-state.json`.
- `python3 scripts/sweep_monitor.py --scope fixed --limit 8` after the tick:
  - Local generated vs published sweep-state drift: `0`.
  - Published rows: `4814 archive + 843 current + 100 fixed`.
  - Fixed coverage: `100 / 1310` expected fixed profile-concurrency points.
  - Missing fixed points: `1210`.

### Active loop
- Persistent dispatcher loop is running under `setsid` + `flock`.
- Lock: `/tmp/fixed-scope-sweep-loop.lock`
- Loop log: `/tmp/fixed-scope-sweep-loop.log`
- Orchestrator log: `/tmp/bench_orchestrator.log`
- Process shape to check:
  ```bash
  ps -eo pid,ppid,sid,stat,etime,cmd | grep fixed-scope-sweep-loop
  ps -eo pid,ppid,sid,stat,etime,cmd | grep bench_orchestrator
  ```
- Start command used:
  ```bash
  setsid -f bash -lc 'cd /root/agentic-serve/inference-benchmark; exec flock -n /tmp/fixed-scope-sweep-loop.lock bash -lc '\''while true; do date -Is; bash scripts/bench_orchestrator.sh; sleep 120; done'\''' >> /tmp/fixed-scope-sweep-loop.log 2>&1
  ```
- Stop command:
  ```bash
  pkill -f fixed-scope-sweep-loop.lock
  ```

### Current interpretation
The expected denominator for fixed scope is now `1310` with the current `fixed_multi` restriction to `chat-multiturn` and `osworld-multiturn`. The older `2414` number came from treating current+fixed YAML cells as one unscoped coverage surface. The important live gap is still large: published fixed data is only `100/1310`, but the harness now reruns stale current-scope JSONs instead of skipping them.

## Codex Addendum — Sweep Progress Reporter

Added and launched a read-only progress reporter:

- Script: `scripts/sweep_progress_report.py`
- Loop lock: `/tmp/sweep-progress-reporter.lock`
- Loop log: `/tmp/sweep-progress-reporter.log`
- Latest report: `/tmp/sweep-progress-latest.md`
- History: `/tmp/sweep-progress-history.md`
- Interval: 300 seconds

It reads `scripts/bench_jobs.txt` plus `/tmp/bench_jobs/state`, then polls `a100`, `3090`, `2080ti`, and `h100` over SSH with `nvidia-smi`/`ps`. The report includes per-host job counts, per-GPU memory/utilization, active sweep assignments, GPU processes, and non-sweep GPU owners.

Launcher:

```bash
setsid -f bash -lc 'cd /root/agentic-serve/inference-benchmark; exec flock -n /tmp/sweep-progress-reporter.lock python3 scripts/sweep_progress_report.py --interval-seconds 300' >> /tmp/sweep-progress-reporter.log 2>&1
```

Stop:

```bash
pkill -f 'scripts/sweep_progress_report.py --interval-seconds 300'
pkill -f sweep-progress-reporter.lock
```

Latest validated snapshot at 2026-05-04T13:10:57Z:

- 132 jobs: done=74, running=5, pending=20, skipped=31, known_oom=2.
- `a100`: no active sweep jobs; GPU0 has same-user non-sweep VLLM.
- `3090`: two active tp4 sweep jobs occupying GPUs 0-7.
- `2080ti`: idle; sweep rows are terminal done/skipped.
- `h100`: three local running states; GPUs 4, 6, and 7 are occupied by `zifengd+` VLLM processes.
