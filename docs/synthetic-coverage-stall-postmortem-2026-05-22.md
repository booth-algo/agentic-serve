# Synthetic Coverage Stall Post-Mortem - 2026-05-22

Generated: 2026-05-22 13:26:33 UTC

Repository: `/root/agentic-serve`

Branch at write time: `mse-prefix-aware-replay`

HEAD at write time: `294ba8d`

## Short Version

Synthetic coverage looked stuck even though the dashboard still showed only about
half the matrix complete. The immediate cause was stale terminal scheduler state:
49 coarse benchmark jobs were marked `skipped` while their fine-grained
profile/concurrency coverage points were still missing.

The orchestrator treats `done`, `skipped`, and `failed` as terminal states and
does not dispatch them again. That means a job can block coverage forever if its
coarse status is terminal but the dashboard-level coverage matrix is incomplete.

The one-off recovery was applied:

- reset 49 incomplete `skipped` jobs back to `pending`,
- manually started one orchestrator tick,
- confirmed fresh dispatches on A100, 2080Ti, and H100,
- refreshed private `gpu-state.json`,
- patched GPU state parsing so 3090 NVML failures are shown as host errors
  instead of a silent empty GPU list.

The systemic fix is still not fully applied: the scheduler needs coverage-aware
state, or at minimum an automatic reconciliation/sentry path, so a terminal
coarse job cannot hide missing coverage cells again.

## Impact

User-visible impact:

- Synthetic coverage appeared stuck around `3095/5954` in the dashboard.
- The local reconcile pass against current `sweep.yaml` saw `3095/5789` expected
  points before reset. The numerator matched the observed stall; the denominator
  mismatch is a separate expected-matrix consistency issue.
- Several GPUs were free or appeared free, but the scheduler had no eligible
  pending work for some hosts because the missing jobs were terminal `skipped`.
- The dashboard previously made 3090 look like a blank/buggy GPU config instead
  of reporting the actual NVML failure.

Operational impact:

- A100 had free capacity but no A100 pending work before the reset because many
  missing A100 cells were marked `skipped`.
- 2080Ti had free capacity but many missing 2080Ti cells were also `skipped`.
- 3090 is intentionally drained and should not receive new jobs.
- H100 has GPU 6 intentionally blocked, so at most seven H100 GPUs are available
  to the sweep.
- `h100-2` is not contributing because SSH times out.

## Evidence

### Before the recovery reset

Report: `/tmp/sweep-coverage-reconcile-synthetic-reset.md`

```text
generated_at: 2026-05-22T13:18:04+00:00
scope: synthetic_distributional
expected coverage points: 5789
present expected points: 3095 / 5789
missing expected points: 2694
expected jobs: 212
jobs with missing coverage: 120
stale terminal/blocking jobs with missing coverage: 49
reset candidates for statuses ['skipped']: 49
reset performed: 49 jobs
job status counts: {'done': 52, 'skipped': 49, 'running': 58, 'pending': 53}
missing jobs by status: {'skipped': 49, 'running': 18, 'pending': 53}
```

This is the strongest evidence for the stall: missing coverage existed, but 49
of the missing jobs were terminal/blocking.

### Orchestrator behavior

`inference-benchmark/scripts/bench_orchestrator.sh` treats terminal states as
non-dispatchable:

```text
done|skipped|failed)
    continue
```

Relevant code: `inference-benchmark/scripts/bench_orchestrator.sh:862`

The same script dispatches only `pending` jobs after checking host drain state,
free GPUs, and free ports:

```text
pending)
    if host_drained "$HOST"; then
        ... skipping new dispatches ...
        continue
    fi
```

Relevant code: `inference-benchmark/scripts/bench_orchestrator.sh:1035`

It also records the selected run id, GPUs, port, and benchmark metadata when it
does dispatch:

```text
BENCH_RUN_ID=$RUN_ID BENCH_JOB_ID=$JID BENCH_SCOPE=$ROW_STORAGE_SCOPE
BENCH_PORT=$SLOT_PORT BENCH_GPUS=$SLOT_GPUS
```

Relevant code: `inference-benchmark/scripts/bench_orchestrator.sh:1106`

### Why coarse job status is not enough

The reconciliation script documents the central mismatch:

```text
The orchestrator tracks coarse jobs: host + model + tp + mode + backend.
The dashboard coverage is finer-grained: profile + concurrency rows inside
each coarse job.
```

Relevant code: `inference-benchmark/scripts/reconcile_sweep_coverage.py:2`

That means a single coarse job can have some outputs present and some missing.
If the coarse job becomes `skipped`, all missing profile/concurrency cells under
that job become hidden from dispatch unless a reconciliation pass resets it.

### After reset and one orchestrator tick

Report: `/tmp/sweep-coverage-reconcile-synthetic-after.md`

```text
generated_at: 2026-05-22T13:23:43+00:00
expected coverage points: 5789
present expected points: 3137 / 5789
missing expected points: 2652
expected jobs: 212
jobs with missing coverage: 120
stale terminal/blocking jobs with missing coverage: 1
reset performed: 0 jobs
job status counts: {'done': 52, 'pending': 96, 'running': 63, 'skipped': 1}
missing jobs by status: {'pending': 96, 'running': 23, 'skipped': 1}
```

The system was no longer scheduler-stopped after the reset. Most missing work was
now either `pending` or `running`.

### Live dashboard/GPU state after recovery

File: `inference-benchmark/dashboard/dist/gpu-state.json`

Generated at: `2026-05-22T13:25:32+00:00`

```text
orchestrator: timer-active
job_counts: {'done': 92, 'pending': 103, 'running': 8, 'skipped': 9}
summary: {'gpus_blocked': 1, 'gpus_free': 1, 'gpus_sweep': 23, 'gpus_total': 24,
          'hosts_error': 2, 'hosts_ok': 3, 'hosts_total': 5}
```

Host details:

```text
a100:
  ok: true
  gpu_status_counts: {'sweep': 8}
  running:
    - a100_Llama-3.1-70B_tp4_single_sglang
    - a100_gpt-oss-120b_tp4_multi_sglang

2080ti:
  ok: true
  gpu_status_counts: {'sweep': 8}
  running:
    - 2080ti_Qwen3.5-9B_tp4_multi
    - 2080ti_Qwen3.5-9B_tp2_single
    - 2080ti_Qwen3.5-9B_tp2_multi

h100:
  ok: true
  blocked_gpus: ['6']
  gpu_status_counts: {'free': 1, 'sweep': 7}
  running:
    - h100_Llama-3.1-70B_tp4_multi_sglang
    - h100_Qwen3.5-27B_tp1_single_sglang
    - h100_gpt-oss-20b_tp2_single_sglang

3090:
  ok: false
  drained: true
  error: Failed to initialize NVML: Driver/library version mismatch;
         NVML library version: 580.159

h100-2:
  ok: false
  error: ssh: connect to host 10.250.30.47 port 22: Connection timed out
```

Tailscale check:

```text
https://agenticserve.tail2bcc6a.ts.net/agentic-serve/gpu-state.json
HTTP/2 200
content-type: application/json; charset=utf-8
```

## Timeline

All times UTC.

| Time | Event |
|---|---|
| 13:18:04 | Coverage reconcile found `3095 / 5789` present, `2694` missing, and `49` stale terminal `skipped` jobs with missing coverage. |
| 13:18:04 | One-off reset changed those 49 incomplete `skipped` jobs to `pending`. |
| 13:18:17 | `agentic-serve-bench-orchestrator.service` started. |
| 13:19:13 | A100 `a100_Llama-3.1-70B_tp4_single_sglang` dispatched on GPUs `4,5,6,7`. |
| 13:19:17 | 2080Ti `2080ti_Qwen3.5-9B_tp4_multi` dispatched on GPUs `1,2,3,4`. |
| 13:19:19 | 2080Ti `2080ti_Qwen3.5-9B_tp2_single` dispatched on GPUs `5,6`. |
| 13:19:22 | 2080Ti `2080ti_Qwen3.5-9B_tp2_multi` flexed to GPUs `0,7`. |
| 13:19:30 | 3090 skipped for new dispatch because host is drained. |
| 13:19:43 | H100 `h100_Qwen3.5-27B_tp1_single_sglang` dispatched on GPU `7`. |
| 13:20:01 | H100 stale `h100_Qwen3.5-27B_tp2_multi_sglang` finalized as skipped, released GPUs `4,5`. |
| 13:20:01 | H100 `h100_gpt-oss-20b_tp2_single_sglang` dispatched on GPUs `4,5`. |
| 13:20:06 | Orchestrator tick completed successfully. |
| 13:23:43 | Follow-up reconcile showed stale terminal jobs reduced from 49 to 1; missing work was mostly pending/running. |
| 13:25:32 | GPU state refresh showed 23 sweep GPUs, 1 intentionally blocked GPU, 2 host errors. |

## Root Cause

Primary root cause:

The scheduler's terminal state is too coarse for the coverage objective. A job
status represents `(host, model, tp, mode, backend)`, but coverage completion is
defined over `(hardware, model, backend, mode, profile, concurrency)`. When a
coarse job is marked `skipped`, the orchestrator stops considering it even if
some profile/concurrency outputs are missing.

Contributing causes:

- `skipped` was treated as a terminal state equivalent to no more work, even
  when coverage was incomplete.
- Reconciliation existed as a manual repair tool, but it was not automatic in
  the orchestrator or dashboard refresh path.
- The dashboard exposed the coverage stall, but did not clearly flag "missing
  coverage is blocked by terminal scheduler state."
- 3090 host health was being reported poorly: `nvidia-smi` failure was
  suppressed, so the host looked like it had no GPUs rather than a real NVML
  fault. This is now fixed in the reporter.
- The visible coverage denominator differed between dashboard and reconcile
  (`5954` vs `5789`), which made it harder to tell which expected matrix was
  authoritative.

## What Was Already Fixed In This Session

1. Requeued stale incomplete `skipped` jobs.

   Command effect:

   ```text
   reset candidates for statuses ['skipped']: 49
   reset performed: 49 jobs
   ```

   This moved missing work back into the scheduler's pending/running path.

2. Started an orchestrator tick.

   Result:

   - A100 is fully occupied by sweep jobs.
   - 2080Ti is fully occupied by sweep jobs.
   - H100 is using seven GPUs, with GPU 6 intentionally blocked.
   - 3090 remains drained.

3. Refreshed live GPU state.

   Result:

   - `gpu-state.json` is present in `dashboard/dist`.
   - Tailscale-served `gpu-state.json` returns HTTP 200.

4. Fixed GPU state reporting for `nvidia-smi` failures.

   Change:

   - `sweep_progress_report.py` now captures `nvidia-smi` stderr/stdout failure
     lines and marks the host `ok: false` if no GPUs parse and NVML/error text
     was seen.

   Test:

   ```text
   python3 -m unittest inference-benchmark/tests/test_sweep_progress_report.py
   Ran 9 tests in 0.003s
   OK
   ```

   Relevant code:

   - `inference-benchmark/scripts/sweep_progress_report.py:596`
   - `inference-benchmark/scripts/sweep_progress_report.py:757`
   - `inference-benchmark/tests/test_sweep_progress_report.py:144`

5. Added automatic coverage reconciliation before scheduler dispatch.

   Change:

   - `bench_orchestrator.sh` now runs `reconcile_sweep_coverage.py` before GPU
     reclaim and dispatch.
   - The preflight resets terminal jobs with missing coverage back to `pending`
     under a bounded policy.
   - The default policy is one automatic coverage requeue per job:

     ```text
     BENCH_RECONCILE_COVERAGE_BEFORE_DISPATCH=1
     BENCH_COVERAGE_RESET_STATUSES=done,skipped,failed,known_oom
     BENCH_COVERAGE_MAX_REQUEUES=1
     ```

   - Requeue metadata is written next to scheduler state:

     ```text
     <job>.coverage_requeue_count
     <job>.coverage_blocker.json
     ```

   - A dashboard-readable blocker summary is written to:

     ```text
     inference-benchmark/dashboard/dist/coverage-blockers.synthetic_distributional.json
     ```

   Tests:

   ```text
   python3 -m unittest inference-benchmark/tests/test_reconcile_sweep_coverage.py
   Ran 2 tests in 0.003s
   OK

   bash inference-benchmark/scripts/test_bench_orchestrator_service.sh
   bench orchestrator service dry-run smoke test passed
   ```

   Live verification:

   ```text
   2026-05-22T18:41:28+00:00 running coverage reconcile preflight ...
   reset candidates for statuses ['done', 'failed', 'known_oom', 'skipped']: 17
   reset performed: 17 jobs
   reset exhausted by coverage requeue limit 1: 0 jobs
   ```

   The same tick dispatched fresh work:

   ```text
   a100_Llama-3.1-8B_tp2_multi_sglang on a100 GPUs [0,1]
   h100_Llama-3.1-70B_tp2_multi on h100 GPUs [4,5]
   h100_gpt-oss-120b_tp1_multi on h100 GPU [7]
   ```

   H100 GPU 6 remained blocked during dispatch.

6. Refreshed dashboard GPU state after the automatic preflight.

   Evidence:

   ```text
   generated_at: 2026-05-22T18:46:50+00:00
   job_counts: {'done': 93, 'pending': 109, 'running': 6, 'skipped': 4}
   summary: {'gpus_blocked': 1, 'gpus_free': 5, 'gpus_sweep': 19,
             'gpus_total': 24, 'hosts_error': 2, 'hosts_ok': 3}
   ```

   Tailscale check:

   ```text
   https://agenticserve.tail2bcc6a.ts.net/agentic-serve/gpu-state.json
   HTTP/2 200
   ```

## What Could Have Prevented This

The following would have prevented the stall or made it self-healing:

1. Coverage-aware terminal states.

   A job should not be considered terminal-complete unless all expected
   profile/concurrency outputs are present in the local dashboard data for that
   scope. If it has failures, its terminal state should be visibly
   `coverage_blocked` or `retry_exhausted`, not silently equivalent to "do not
   schedule and do not alert."

2. Automatic reconciliation before dispatch.

   The orchestrator should run a coverage reconciliation guard before deciding
   there is no work for a host. If terminal jobs have missing coverage, it should
   either requeue them under policy or mark them as explicit coverage blockers.

3. A no-progress sentry.

   If synthetic coverage does not increase for N orchestrator/dashboard cycles
   while there are missing expected points, the system should raise a dashboard
   fault and write a short diagnosis:

   - missing points,
   - terminal blockers,
   - pending jobs by host,
   - free GPUs by host,
   - host health/drain/block controls.

4. Unified expected-matrix calculation.

   The dashboard, `reconcile_sweep_coverage.py`, and scheduler should share the
   same expected coverage point calculation. The observed `5954` dashboard
   denominator vs `5789` reconcile denominator is a warning that there are still
   multiple definitions of "expected coverage."

5. Host health gating.

   A host with NVML failure or SSH timeout should be marked unhealthy at the
   scheduler layer, not just in the GPU dashboard. This would make capacity
   math more honest and avoid pending jobs appearing launchable on a broken host.

## Fixes Still Worth Applying

### P0 - Render coverage blockers in the dashboard

The scheduler now writes a machine-readable blocker file, but the dashboard does
not yet render it as a first-class fault panel.

Minimum behavior:

- load `coverage-blockers.synthetic_distributional.json`,
- show reset-performed and reset-exhausted counts,
- highlight jobs that reached the coverage requeue cap,
- show missing profile/concurrency groups and last failure reason.

The infinite-loop guard is now in place through `coverage_requeue_count`, but
operators still need a clear UI when that guard stops retrying a cell.

### P0 - Split coarse job status from coverage point status

The durable state should track both:

- coarse run/job lifecycle: pending, dispatching, running, done, skipped, failed,
- per expected coverage point lifecycle: missing, produced, failed, waived.

This would let the scheduler rerun only missing profile/concurrency cells or
explicitly waive impossible cells without pretending the whole job is complete.

This is the durable fix for the root cause.

### P0 - Add explicit retry-exhausted metadata to coverage UI

Existing failure metadata is coarse. The dashboard should show cells/jobs that
hit retry limits as "retry exhausted" with:

- attempts used,
- max attempts,
- missing profile/concurrency list,
- last remote log path,
- last failure summary,
- whether the cell is still counted in coverage.

This makes it clear why coverage is not advancing.

### P1 - Unify the coverage denominator

Investigate and remove the discrepancy between:

- dashboard/user-visible denominator: `5954`,
- current reconcile denominator from `sweep.yaml`: `5789`.

The dashboard and reconcile tooling should use the same generated manifest or
same library function for expected coverage. If infeasible/waived cells are
included in one denominator and excluded in another, label that explicitly.

### P1 - Make host health part of scheduler capacity

The GPU dashboard now reports:

```text
3090: NVML driver/library version mismatch
h100-2: SSH timeout
```

The scheduler should also produce a host health summary and avoid implying that
pending jobs on those hosts are dispatchable. This should feed into the coverage
sentry so the reason for no progress is precise:

- host drained,
- host unhealthy,
- GPU blocked,
- all GPUs busy,
- no pending jobs,
- terminal coverage blockers.

### P1 - Add progress-rate monitoring

Persist a small synthetic progress history, for example:

```json
{
  "scope": "synthetic_distributional",
  "generated_at": "...",
  "present": 3137,
  "expected": 5789,
  "pending_jobs": 96,
  "running_jobs": 23,
  "terminal_blockers": 1
}
```

Alert or mark dashboard health degraded if `present` is unchanged across a
configurable number of timer ticks while runnable capacity exists.

### P1 - Make R2 sync status visible but not authoritative

The current intended source of truth is local `/mnt/100g/agent-bench`. R2 should
remain a mirror. The dashboard should show last successful R2 mirror time, but
coverage scheduling should not depend on R2 being current.

### P2 - Add dashboard controls for blocker handling

Useful controls:

- requeue selected retry-exhausted jobs,
- waive selected impossible coverage points with a reason,
- undrain/drain host,
- block/unblock GPU,
- show "why no dispatch on this host" in one line.

This is lower priority than the state-model fix, but it would make the system
much easier to operate without SSH/manual scripts.

## Open Questions

1. Should `skipped` mean "do not retry but still counts as missing coverage" or
   should it be renamed/split into `retry_exhausted` and `waived`?

2. Should impossible cells remain in the coverage denominator until explicitly
   waived, or should the scheduler remove them automatically after a deterministic
   infeasibility classification?

3. Should 3090 pending jobs remain in the synthetic denominator while the host is
   drained for another user, or should dashboard coverage separately show
   "temporarily unavailable capacity"?

4. Should `h100-2` jobs be excluded from active launch planning while SSH is
   unreachable, or kept pending with a host-health blocker?

5. What is the authoritative expected denominator for synthetic today: the
   dashboard's `5954` or reconcile's `5789` from current `sweep.yaml`?

## Recommended Next Patch Set

1. Add `scripts/synthetic_coverage_sentry.py` or extend
   `reconcile_sweep_coverage.py` with a machine-readable JSON output.

2. Run that sentry from `run-bench-orchestrator-service.sh` before dispatch and
   from dashboard refresh after rebuilding data.

3. Add `coverage-blockers.json` to `dashboard/dist` and render it in the GPU or
   coverage page.

4. Add controlled requeue metadata:

   ```text
   <job>.coverage_requeue_count
   <job>.coverage_blocker.json
   ```

5. Split terminal states in the dashboard:

   - `done`: all expected points present,
   - `retry_exhausted`: benchmark attempts exhausted but coverage missing,
   - `waived`: missing coverage intentionally excluded,
   - `failed`: infrastructure failure, not a benchmark result.

6. Unify expected coverage generation so dashboard, reconcile, and orchestrator
   read the same manifest/point set.

## Current Stop Condition

The sweep should be considered operational when all are true:

- `agentic-serve-bench-orchestrator.timer` is active,
- `gpu-state.json` is served with HTTP 200,
- no terminal jobs have missing coverage unless explicitly waived,
- synthetic coverage numerator increases over time while pending work remains,
- host health blockers are visible for 3090 and h100-2,
- H100 GPU 6 remains blocked until the user unblocks it.

As of the evidence in this document, the sweep is operational again but not
self-protecting against the same class of stall.
