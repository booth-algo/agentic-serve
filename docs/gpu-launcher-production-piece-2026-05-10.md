# GPU Launcher Production Piece - 2026-05-10

Created: 2026-05-10 14:40:00 UTC

## Purpose

Drive benchmark execution continuously from the server instead of relying on a
manual shell or ad-hoc cron.

The dashboard refresh timer only rebuilds visible artifacts from local data.
This GPU launcher timer is the piece that creates new local data by dispatching
pending benchmark jobs to GPU hosts.

## Flow

```text
systemd timer
  -> run-bench-orchestrator-service.sh
  -> sync-gpu-code.sh
  -> bench_orchestrator.sh
  -> SSH launch on GPU host
  -> remote /tmp/results/<scope>/<run-dir>/
  -> local /mnt/100g/agent-bench/results/<scope>/<run-dir>/
  -> best-effort R2 raw-results mirror
  -> sweep-state.json publish
```

Scope names are normalized before launch: synthetic jobs use
`synthetic_distributional`, trace replay uses `trace_replay`, and retired
canonical/fixed-grid/MSE jobs publish dashboard rows as `archived` while storing
raw files under `archived/canonical`, `archived/fixed-grid`, or `archived/mse`.

## Units

- `deploy/systemd/agentic-serve-bench-orchestrator.service`
- `deploy/systemd/agentic-serve-bench-orchestrator.timer`

The timer is configured for one orchestration tick every 10 minutes.

## Safety

- `bench_orchestrator.sh` now supports
  `BENCH_ORCHESTRATOR_DRY_RUN=1`.
- Dry-run mode avoids state writes, rsync, R2 upload, sweep-state publish, and
  remote job launch.
- `BENCH_ORCHESTRATOR_SKIP_REMOTE_PROBE=1` disables SSH slot probing for local
  smoke tests.
- `BENCH_SYNC_GPU_CODE=0` skips remote code sync for tests.

## Test

Run:

```bash
bash inference-benchmark/scripts/test_bench_orchestrator_service.sh
systemd-analyze verify \
  deploy/systemd/agentic-serve-bench-orchestrator.service \
  deploy/systemd/agentic-serve-bench-orchestrator.timer
```

## Enable

Do not enable this timer unless real GPU dispatch is intended.

```bash
cp deploy/systemd/agentic-serve-bench-orchestrator.* /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now agentic-serve-bench-orchestrator.timer
```
