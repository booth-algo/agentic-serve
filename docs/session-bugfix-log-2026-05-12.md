# Session Bugfix Log

Created: 2026-05-12T15:34:30+00:00

Scope: Agentic Serve dashboard, benchmark orchestration, synthetic coverage,
GPU state reporting, and orphan GPU cleanup work from this session.

## Summary

This session moved the production sweep/dashboard path back onto
`agentic-serve`, tightened the durable local state model under
`/mnt/100g/agent-bench`, made the dashboard expose GPU and orchestrator state,
and fixed stale GPU assignment/orphan-cleanup bugs that were confusing the live
dashboard.

## Bugs And Fixes

| Bug | Impact | Fix | Main artifacts |
| --- | --- | --- | --- |
| Dashboard work had drifted between `agenticservenew` and `agentic-serve`. | Tailscale/dashboard pieces could be edited in the wrong repo. | Consolidated ongoing production work in `agentic-serve`; stopped treating `agenticservenew` as the active repo. | `agentic-serve` workspace |
| R2 prefixes and result scopes were messy (`archive`, flat JSON files, mixed synthetic/current naming). | Dashboard and sweep tooling could read/write inconsistent locations. | Standardized scope naming around `trace_replay`, `synthetic_distributional`, and `archived`; documented the cleaned R2 layout and result hierarchy. | `inference-benchmark/docs/r2-data-layout.md`, `docs/data-scopes-and-profiles.md` |
| Flat result output made future runs hard to reason about. | Rebuilds and audits had to infer which result belonged to which scope/run. | Normalized production storage toward `results/<scope>/<run-dir>/`, with archived subscopes for retired fixed/canonical/MSE outputs. | `scripts/bench_orchestrator.sh`, `scripts/rebuild-local-dashboard.sh` |
| Dashboard rebuilds pulled too much data and were slow. | Loading benchmark data felt expensive and stale. | Added local dashboard rebuild flow from `/mnt/100g/agent-bench/results`, pre-scope JSON artifacts, and optional R2 mirroring after local validation. | `scripts/rebuild-local-dashboard.sh`, dashboard public JSON artifacts |
| `gpu-state.json` returned 404 on the dashboard. | GPU page could not load private GPU state. | Added a local/private GPU state refresh path that writes `dashboard/dist/gpu-state.json`; R2 mirroring intentionally excludes it because it contains host/user/process details. | `scripts/refresh-gpu-state.sh`, `deploy/systemd/agentic-serve-gpu-state-refresh.*` |
| Dashboard navigation did not make the GPU page discoverable. | Users could get stuck away from home. | Moved navigation to the top bar with Home/GPU-style tabs. | `dashboard/src/components/GpuStatePage.tsx`, app navigation code |
| GPU dashboard only showed results, not live GPU health. | It was hard to tell whether GPUs were free, occupied by sweep jobs, occupied by other users, faulted, or orphaned. | Added per-host/per-GPU memory/utilization, process owner details, sweep assignments, process classifications, and orchestrator health. | `scripts/sweep_progress_report.py`, `dashboard/src/components/GpuStatePage.tsx`, `dashboard/src/types-gpu-state.ts` |
| Host naming still used `gpu-4` in places. | A100 scheduling/status was confusing and inconsistent with actual host naming. | Replaced production references with `a100` and updated dashboard/state naming paths. | sweep config, orchestrator, dashboard types/docs |
| Synthetic concurrency coverage was hardcoded. | New trace-replay-style concurrency grids did not appear reliably in synthetic sweep state. | Moved launch shape into `sweep.yaml`/`compile_sweep.py`; synthetic_distributional now expands using the trace-replay concurrency grid. | `scripts/sweep.yaml`, `scripts/compile_sweep.py`, `docs/synthetic-scope-sweep-handoff-2026-05-05.md` |
| `bench_jobs.txt` was a mutable singleton. | The orchestrator could run stale jobs after `sweep.yaml` changed. | Made the orchestrator compile a scoped job manifest from `sweep.yaml` on each tick unless `BENCH_JOBS_FILE` is explicitly set. | `scripts/bench_orchestrator.sh`, `docs/bench-job-manifest.md` |
| Pinned GPUs were too rigid. | Free 2080Ti capacity could be skipped when a preferred slot was occupied. | Added flexible pinned-GPU behavior so jobs can move to another free GPU when safe. | `scripts/bench_orchestrator.sh`, `scripts/test_bench_orchestrator_service.sh` |
| Orchestrator was not clearly running. | Dashboard coverage could stall while users assumed a sweep loop was active. | Added systemd service/timer support and surfaced orchestrator service/timer health in `gpu-state.json`. | `deploy/systemd/agentic-serve-bench-orchestrator.*`, `scripts/sweep_progress_report.py` |
| A100 GPU 5 showed two sweep assignments. | Dashboard claimed one GPU belonged to both `a100_Llama-3.1-8B_tp1_single_sglang` and `a100_Llama-3.1-70B_tp4_multi_sglang`. | Fixed the orchestrator to capture listener commands and only trust a recorded running port when the listener command matches the job model path/basename. Stale recorded slots are no longer reserved and are finalized. | `scripts/bench_orchestrator.sh`, `scripts/test_bench_orchestrator_service.sh` |
| Dashboard displayed stale sweep assignments after state drift. | Old `running` state could keep showing even when the live GPU process belonged to another model. | Filtered visible assignments through live process matching, including parent process command matching for SGLang scheduler workers. | `scripts/sweep_progress_report.py`, `tests/test_sweep_progress_report.py` |
| Same-user non-sweep GPU holders could be stale orphan processes. | GPUs looked occupied forever even after their owning sweep died. | Added `same-user-orphan` classification for direct-init processes and vLLM workers whose `VLLM::EngineCore` parent is orphaned under init. | `scripts/sweep_progress_report.py` |
| Orphan cleanup existed only as a manual judgment call. | We could identify stuck GPUs but had no repeatable cleanup gate. | Added a dry-run-first cleaner with repeated-observation gating, age gating, exact PID targeting, audit logs, and SIGTERM-only default behavior. | `scripts/clean_orphan_gpus.py`, `scripts/gpu_cleanup.json`, `docs/gpu-orphan-cleanup.md` |
| 2080Ti GPUs 1 and 2 stayed orphaned. | Two vLLM worker processes held memory and showed as `same-user-orphan`. | Ran one targeted execute cleanup on `2080ti`; it signaled parent `3911581` and workers `3911703`, `3911704`, then refreshed GPU state. Post-check found zero remaining cleanup candidates. | `/mnt/100g/agent-bench/state/gpu-cleanup-events.jsonl`, `scripts/refresh-gpu-state.sh` |

## Live Cleanup Result

On 2026-05-12 the targeted 2080Ti cleanup produced:

- candidates: `2`
- eligible: `2`
- action: `signal`
- signaled PIDs: `3911581`, `3911703`, `3911704`
- remaining PIDs after SIGTERM: `[]`
- post-cleanup dry-run candidates: `0`

After refreshing `gpu-state.json`, 2080Ti GPU 1 and GPU 2 showed `free`.

## Important Caveats

- The recurring orphan-cleaner systemd unit remains audit-only by default:
  `gpu_cleanup.json` has `dry_run: true`, and the installed timer should not be
  converted into automatic cleanup without an explicit operational decision.
- A100 still had a live old SGLang 70B process after the orchestrator finalized
  its stale running state. The dashboard no longer shows a false sweep
  assignment for it; it appears as `same-user-nonsweep` until that process is
  intentionally stopped or reused.
- `gpu-state.json` is private local dashboard state. It should remain outside
  public R2 mirroring because it includes host names, users, ports, and process
  commands.

## Validation Evidence

Commands run during the final bugfix pass:

```bash
bash -n \
  inference-benchmark/scripts/bench_orchestrator.sh \
  inference-benchmark/scripts/test_bench_orchestrator_service.sh \
  inference-benchmark/scripts/refresh-gpu-state.sh

python3 -m py_compile \
  inference-benchmark/scripts/sweep_progress_report.py \
  inference-benchmark/scripts/clean_orphan_gpus.py

bash inference-benchmark/scripts/test_bench_orchestrator_service.sh

cd inference-benchmark
python3 -m unittest discover tests
```

Latest test result:

- `python3 -m unittest discover tests`: 50 tests passed.
- Orchestrator dry-run smoke test passed, including stale port-owner detection.
- 2080Ti cleanup post-check returned `candidates: 0`, `eligible: 0`.
