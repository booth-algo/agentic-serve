# Local R2 Mirror

Date: 2026-05-03

Purpose: keep a local bucket-shaped copy of the `agent-bench` R2 bucket on the 100G mount so local dashboard iteration does not depend on stale repo-local result files or refill `/root`.

## Location

- Mount: `/mnt/100g`
- Mirror root: `/mnt/100g/agent-bench`
- Layout mirrors R2 bucket keys directly, for example:
  - `/mnt/100g/agent-bench/results/...`
  - `/mnt/100g/agent-bench/json/current/data.json`
  - `/mnt/100g/agent-bench/json/current/sweep-state.json`

## Commands

Full pull-only mirror sync:

```bash
./inference-benchmark/scripts/sync-r2-mirror.sh --all
```

Hydrate local dashboard public JSON from the synced mirror:

```bash
./inference-benchmark/scripts/sync-r2-mirror.sh --json-only --hydrate-public
```

Rebuild dashboard data from the mirror without syncing R2 again:

```bash
bash inference-benchmark/scripts/refresh-dashboard-data.sh --skip-sync
```

`refresh-dashboard-data.sh` now defaults `BENCHMARK_RESULTS_DIR` to `/mnt/100g/agent-bench/results` when `/mnt/100g` exists. Override with `BENCHMARK_RESULTS_DIR=...` or `--results-dir ...` when needed.

## Validation Snapshot

- R2 bucket summary via `s3api list-objects-v2`: `14268` objects, `15483039994` bytes.
- Local mirror summary: `14268` files, `15483039994` bytes.
- Mirror disk usage: `15G`.
- Dashboard public JSON was hydrated from `/mnt/100g/agent-bench/json/current`.
- Mirror-backed `refresh-dashboard-data.sh --skip-sync --output /tmp/agentic-serve-data-from-mirror.json` read `/mnt/100g/agent-bench/results`, found `13804` result JSON files, included `5604` rows, skipped `6920`, and wrote the output.

Known raw data issue: one mirrored R2 result file is malformed and is skipped by the existing builder:

```text
/mnt/100g/agent-bench/results/current/h100_Llama-3.1-8B_tp4_vllm/Llama-3.1-8B_tp4_vllm_chat-singleturn_conc1.json
```

The malformed file exists in the bucket copy; this is not a local mirror drift.
