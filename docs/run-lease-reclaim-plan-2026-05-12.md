# Run Lease and GPU Reclaim Plan

Date: 2026-05-12 16:34:33 UTC

## Goal

Give the benchmark control plane a durable ownership identity for every launched run, then use that identity to make GPU reclaim safe, auditable, and ready for future preemption/reordering.

## Current Problem

The orchestrator has a stable logical `job_id`, but that ID represents a benchmark cell, not a live process lease. It is reused across retries and records status, port, GPUs, attempt, and signature. It does not strongly identify one remote launch attempt or its process group.

That means `same-user-nonsweep` cleanup currently has to infer ownership from process shape, GPU memory, ports, and age. That is acceptable for legacy stale benchmark servers, but it is not a strong enough control plane for instant reclaim or scheduling optimization.

## Design

Keep two identities:

- Logical job ID: stable cell identity such as `2080ti_Qwen3.5-9B_tp1_single`.
- Run ID / lease ID: unique launch attempt identity such as `run_20260512T162230Z_2080ti_Qwen3.5-9B_tp1_single_a83f2c`.

The run lease owns:

- host
- GPU set
- scheduler port
- logical job ID
- benchmark backend
- result/dashboard scope
- launch timestamp
- remote launcher PID when available
- remote log path

## Reclaim Policy

A `same-user-nonsweep` process is reclaimable only if all are true:

- user matches the SSH/benchmark user
- process is VLLM/SGLang server-shaped
- listener port is in scheduler range `8089-8096`
- no live sweep state maps to that port/GPU
- process age exceeds the configured threshold, initially `>= 1h`
- observed in at least two consecutive cleanup scans
- not `other-user`, not `unknown-busy`, not protected by config
- no active `BENCH_RUN_ID` lease is attached to the process

Legacy unmanaged processes can be reclaimed only through the guarded heuristic path. Processes with an active run lease are protected unless a future explicit preemption flow marks the lease reclaimable.

## Implementation Steps

1. Fix orchestrator port probing so `used_ports` reflects live listeners.
2. Generate a unique `run_id` at dispatch time.
3. Persist `<job_id>.run_id` and `runs/<run_id>.json` under the scoped benchmark state directory.
4. Pass `BENCH_RUN_ID`, `BENCH_JOB_ID`, `BENCH_SCOPE`, `BENCH_PORT`, and `BENCH_GPUS` into launched benchmark processes.
5. Extend GPU-state collection to read benchmark lease metadata from process environments where available.
6. Extend the existing orphan GPU cleaner with a separate guarded `same-user-nonsweep` reclaim policy.
7. Keep cleanup dry-run by default; execution requires explicit config/CLI enablement.
8. Surface run IDs and lease metadata on the GPU dashboard.
9. Verify with cleaner unit tests, progress-report parsing tests, and orchestrator dry-run smoke tests.

## Future Work

Once leases are consistently present, add explicit preemption states:

- `running`
- `preempting`
- `reclaimed`
- `completed`
- `failed`

Then the scheduler can safely implement instant reclaim and priority-based run reordering by marking a lease for preemption, killing its process group, releasing the slot, and dispatching a higher-value pending run.
