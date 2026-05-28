# Session Summary - 2026-05-14

Generated: 2026-05-14 00:03:07 UTC

## Goal

Restore synthetic coverage throughput by fixing stale GPU/server reclamation and checking all benchmark hosts, not just `3090`.

The immediate symptom was that drained or completed SGLang sweep servers still occupied GPUs and scheduler ports even when the dashboard showed no live sweep assignment or the backing job was done. This blocked the orchestrator from launching new synthetic runs and made coverage appear stalled.

## High-Level Outcome

- Added automatic reclaim policy for stale sweep-shaped GPU processes with no live assignment.
- Added automatic reclaim policy for stale managed scheduler listeners with no live assignment.
- Changed the installed GPU cleanup systemd service from audit-only to `--execute`.
- Enabled SIGKILL fallback after TERM for gated, managed cleanup candidates.
- Reclaimed stale `a100` and `3090` SGLang servers and ports.
- Verified the orchestrator launched new `a100` work after reclaim.
- Refreshed dashboard GPU state.

## Live State After Fixes

Final dashboard GPU state refresh:

- Generated at: `2026-05-13T23:59:41+00:00`
- Jobs: `done=61`, `running=13`, `pending=122`, `skipped=16`
- GPU summary: `gpus_sweep=26`, `gpus_free=6`, `hosts_ok=4`, `hosts_error=1`
- `a100`: all 8 GPUs in sweep use; two new jobs launched after cleanup.
- `3090`: 3 live sweep jobs, 5 free GPUs, still drained so no new dispatch.
- `2080ti`: 8 GPUs in live sweep use.
- `h100`: 7 live sweep GPUs, 1 free GPU.
- `h100-2`: SSH timeout, still unreachable.

Final cleanup dry-run:

- `candidates=0`
- `eligible=0`
- `events={}`

## Root Causes

### 1. Stale sweep servers were not cleanup candidates

`clean_orphan_gpus.py` only emitted candidates for:

- `same-user-orphan`
- `same-user-nonsweep`

Old SGLang processes were classified as `sweep` because their command lines still contained sweep markers like `/tmp/inference-benchmark` and `sweep_multiturn_profiles_sglang.sh`. Since they had no live assignment, they were stale, but the cleaner ignored them.

### 2. Listener-only remnants still blocked scheduling

After the stale GPU child processes were gone, parent SGLang listener processes still held scheduler ports such as `a100:8090`, `a100:8091`, `a100:8093`, `3090:8089`, and `3090:8095`.

The orchestrator marks scheduler ports as busy via `ss -ltnp`, so listener-only remnants still prevented dispatch even after GPU memory was freed.

### 3. The installed cleanup service was audit-only

The deployed service was installed without `--execute`, so the timer could observe candidates but would not signal them. The repo service file is now updated and the installed unit was reloaded.

### 4. 100 MiB idle memory threshold was too low

`h100` GPU 6 showed about `130 MiB` memory with no process. The reporter and orchestrator treated any GPU over `100 MiB` as busy, which can falsely remove a usable idle GPU from scheduling. Threshold is now `512 MiB`, configurable through `BENCH_GPU_BUSY_MEM_MIB`.

## Files Changed

| File | Purpose |
|---|---|
| `inference-benchmark/scripts/clean_orphan_gpus.py` | Added `stale-sweep-server` and `stale-sweep-listener` policies, shared observation gates, managed-port/user/run-lease checks |
| `inference-benchmark/scripts/gpu_cleanup.json` | Added `reclaim_stale_sweep_servers`; enabled `allow_sigkill`; managed ports remain `8089-8096` |
| `inference-benchmark/scripts/sweep_progress_report.py` | Captures listener PID/user/command/env metadata; exposes richer `ports[]` JSON; raised idle memory threshold to `512 MiB` |
| `inference-benchmark/scripts/bench_orchestrator.sh` | Uses `BENCH_GPU_BUSY_MEM_MIB` default `512` for remote GPU busy detection |
| `deploy/systemd/agentic-serve-gpu-orphan-cleaner.service` | Runs cleaner with `--execute` |
| `deploy/systemd/agentic-serve-gpu-orphan-cleaner.timer` | Description updated from audit to reclaim |
| `inference-benchmark/tests/test_orphan_gpu_cleaner.py` | Added stale sweep process/listener tests |
| `inference-benchmark/tests/test_sweep_progress_report.py` | Added listener metadata parser coverage |

## Reclaimed During Session

Stale sweep GPU processes first reclaimed:

- `a100` GPUs `1,2,3,4,5,7`
- `3090` GPUs `0,4`

Stale listener ports then reclaimed:

- `a100:8090`, pid `924038`
- `a100:8091`, pid `924569`
- `a100:8093`, pid `928182`
- `3090:8089`, pid `2151291`
- `3090:8095`, pid `2203992`

Audit confirmation for listener reclaim:

- Last five signal events reported `ok=True`
- `remaining_pids=[]`

## Commands / Validation

Tests and static checks:

```bash
python3 -m unittest inference-benchmark/tests/test_sweep_progress_report.py inference-benchmark/tests/test_orphan_gpu_cleaner.py
bash -n inference-benchmark/scripts/bench_orchestrator.sh
python3 -m py_compile inference-benchmark/scripts/sweep_progress_report.py inference-benchmark/scripts/clean_orphan_gpus.py
```

Results:

- `20` unit tests passed.
- Bash syntax check passed.
- Python compile check passed.

Live verification:

```bash
python3 inference-benchmark/scripts/clean_orphan_gpus.py \
  --config inference-benchmark/scripts/gpu_cleanup.json \
  --jobs-config inference-benchmark/scripts/sweep.yaml \
  --scope synthetic_distributional \
  --state-dir /mnt/100g/agent-bench/state \
  --dry-run \
  --observation-store /tmp/final-cleanup-check-observations.json \
  --audit-log /tmp/final-cleanup-check-events.jsonl \
  --json-out /tmp/final-cleanup-check-summary.json
```

Result:

```json
{
  "candidates": 0,
  "eligible": 0,
  "events": {}
}
```

Systemd status observed:

- `agentic-serve-gpu-orphan-cleaner.timer`: active, next fire around every 5 minutes.
- `agentic-serve-gpu-state-refresh.timer`: active, refreshes dashboard GPU state.
- `agentic-serve-bench-orchestrator.timer`: active, next orchestration tick around every 10 minutes.

## Important Runtime Notes

- `3090` is intentionally drained through `/mnt/100g/agent-bench/state/control/drained-hosts.txt`; the orchestrator preserves its running jobs but skips new dispatches there.
- `h100-2` still times out over SSH and remains `hosts_error=1`.
- `a100` launched new jobs immediately after stale ports were cleared, proving stale listeners were blocking coverage throughput.
- Dashboard `gpu-state.json` was refreshed after cleanup and threshold changes.

## Dirty Worktree Warning

The repo already had a large dirty worktree before this session. Do not revert unrelated changes. The files listed above are the relevant changes from this reclaim/coverage work; many other modified or untracked files predate this session.

