# Dashboard Data Ground Truth Plan - 2026-05-09

Created: 2026-05-09 19:52:55 UTC
Implemented: 2026-05-09 20:21:38 UTC

## Decision

Make `/mnt/100g/agent-bench` the durable local ground truth for new benchmark
runs. R2 remains the public distribution layer and backup mirror, but it should
not be the freshness-critical source for the private Tailscale dashboard.

The current GitHub Actions rebuild path is useful as a verifier and public
artifact publisher, but it is too slow and unreliable as the primary path for
new run visibility.

## Current Flow

```text
GPU host /tmp/results/<scope>/<run-dir>/
  -> server /tmp/bench_<scope>_<run-dir>/
  -> R2 s3://agent-bench/results/<scope>/<run-dir>/
  -> GitHub Action downloads R2 results
  -> GitHub Action rebuilds dashboard/public/data.json
  -> R2 s3://agent-bench/json/current/data.json
  -> dashboard browser fetches R2 json/current/*.json
```

Problems:

- New raw results are copied to transient `/tmp/bench_*` directories on the
  server instead of the durable `/mnt/100g` tree.
- Runtime state defaults to `/tmp/bench_jobs/state`, so state can disappear or
  drift independently of the durable results store.
- `sweep-state.json` updates quickly, but `data.json` only updates after a
  full rebuild path completes.
- GitHub Actions must download the result tree from R2 before rebuilding, which
  adds latency and another failure point.
- The private dashboard is hosted locally over Tailscale, but the built
  frontend still defaults to fetching generated JSON from public R2.

## Target Flow

```text
GPU host /tmp/results/<scope>/<run-dir>/
  -> server /mnt/100g/agent-bench/results/<scope>/<run-dir>/
  -> local dashboard JSON rebuild from /mnt/100g/agent-bench/results
  -> server dashboard/dist/*.json
  -> private Tailscale dashboard fetches /agentic-serve/*.json
  -> async mirror to R2 s3://agent-bench/json/current/*.json
```

R2 still receives:

- Raw result mirrors under `s3://agent-bench/results/...`, using
  `trace_replay/`, `synthetic_distributional/`, and `archived/...` scope
  names.
- Generated dashboard artifacts under `s3://agent-bench/json/current/...`.
- Public dashboard artifacts for GitHub Pages or external sharing.

But local `/mnt/100g/agent-bench` is the source of truth for new run freshness.

## Cadences

### Immediate State Path

State should be cheap and frequent.

- Use `/mnt/100g/agent-bench/state` as the durable state root.
- Keep `/tmp/bench_jobs/state` only as a legacy or transient fallback during
  migration.
- Every orchestrator tick should publish `sweep-state.json` locally.
- R2 upload of `sweep-state.json` can remain best-effort and non-fatal.

Success condition: the dashboard can show pending, running, done, skipped, and
known-OOM status without waiting for a full `data.json` rebuild.

### Derived Artifact Path

`data.json` and related dashboard JSON files are derived artifacts.

- Rebuild from `/mnt/100g/agent-bench/results`.
- Run on a timer, after new completed jobs, or both.
- Validate before replacing live dashboard artifacts.
- Mirror to R2 only after local validation passes.

Success condition: new completed runs appear in the private dashboard without a
GitHub Actions round trip.

## Implementation Steps

1. Update orchestrator completion handling.
   - Rsync completed remote output into
     `/mnt/100g/agent-bench/results/<scope>/<run-dir>/`.
   - Keep the R2 raw upload as an async mirror.
   - Avoid relying on `/tmp/bench_*` as the only server-side copy.

2. Move durable state.
   - Run future orchestrator ticks with
     `BENCH_STATE_ROOT=/mnt/100g/agent-bench/state`.
   - Preserve compatibility with existing `/tmp/bench_jobs/state` while the
     active sweep is being migrated.
   - Ensure `publish_sweep_state.py --state-dir` points at the durable state
     root for local and R2 state publication.

3. Add local dashboard artifact rebuild.
   - Reuse `inference-benchmark/scripts/refresh-dashboard-data.sh` with
     `--skip-sync` and
     `BENCHMARK_RESULTS_DIR=/mnt/100g/agent-bench/results`.
   - Write validated artifacts into the dashboard's served JSON location.
   - Rebuild or configure the private dashboard so it fetches local
     `/agentic-serve/*.json` URLs rather than public R2 URLs.

4. Mirror after validation.
   - Upload validated local artifacts to
     `s3://agent-bench/json/current/*.json`.
   - Upload raw result directories to `s3://agent-bench/results/...`.
   - Treat R2 failures as mirror failures, not local dashboard failures.

5. Keep GitHub Actions as a fallback.
   - Retain the existing GitHub workflow for public rebuilds, verification, and
     GitHub Pages deploys.
   - Do not depend on GitHub Actions for private dashboard freshness.

## Validation Plan

- Run or simulate one small completed result directory and confirm it lands
  under `/mnt/100g/agent-bench/results/<scope>/<run-dir>/`.
- Build `data.json` locally from `/mnt/100g/agent-bench/results`.
- Run `npm run validate:data` against the locally rebuilt `data.json`.
- Confirm the private Tailscale dashboard fetches local JSON through
  `/agentic-serve/data.json` and `/agentic-serve/sweep-state.json`.
- Confirm `sweep-state.json` updates independently of `data.json`.
- Confirm R2 receives updated artifacts only after local validation succeeds.

## Open Migration Notes

- `/mnt/100g/agent-bench/results` already exists and contains historical data.
- The orchestrator now defaults `BENCH_STATE_ROOT` to
  `/mnt/100g/agent-bench/state`, with read fallback to the legacy
  `/tmp/bench_jobs/state` root during migration.
- The private dashboard rebuild path uses `VITE_R2_JSON_BASE=/agentic-serve`,
  so its generated bundle fetches local JSON artifacts.
- Any local dashboard build intended for Tailscale should override the generated
  JSON URLs to local `/agentic-serve/*.json` paths.
- `inference-benchmark/scripts/rebuild-local-dashboard.sh` owns the local
  derived artifact rebuild and optional R2 mirror.
- Scope names are now normalized at the dashboard boundary:
  `archive -> trace_replay`, `synthetic/latest -> synthetic_distributional`,
  and `current/fixed-grid/mse -> archived`.
