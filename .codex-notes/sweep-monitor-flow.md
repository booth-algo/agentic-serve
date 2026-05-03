# Sweep Monitor Flow

Date: 2026-05-03

## Purpose

Deterministically replace the old Claude sentry loop for benchmark sweeping.
The monitor is read-only: it does not dispatch jobs, publish artifacts, edit
`sweep.yaml`, or mutate `/tmp/bench_jobs/state`.

## Script

```bash
cd /root/agentic-serve
python3 inference-benchmark/scripts/sweep_monitor.py
```

The script compares:

1. Local generated sweep state from
   `inference-benchmark/scripts/sweep.yaml` plus `/tmp/bench_jobs/state`.
2. Local dashboard artifacts:
   `inference-benchmark/dashboard/public/sweep-state.json` and `data.json`.
3. Published website/R2 artifacts:
   `https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current/sweep-state.json`
   and `.../data.json`.

## Checks

- Cell-state drift:
  local generated `sweep-state` vs published R2 `sweep-state`.
- Local dashboard staleness:
  local generated `sweep-state` vs local dashboard `sweep-state.json`.
- Profile-level infeasibility drift:
  local `profile_infeasible[]` vs published `profile_infeasible[]`.
- Coverage against local YAML:
  expected profile x concurrency points from local `sweep.yaml`, excluding
  `known_oom` cells and profile-level infeasible rows.
- Published missing rows:
  expected YAML points absent from published R2 `data.json`.
- Local/R2 data drift:
  current data points present locally but absent from R2, and current data
  points present in R2 but absent locally.

## Useful Commands

Offline structural check, no network:

```bash
python3 inference-benchmark/scripts/sweep_monitor.py \
  --published-state inference-benchmark/dashboard/public/sweep-state.json \
  --published-data inference-benchmark/dashboard/public/data.json
```

CI-style drift gate:

```bash
python3 inference-benchmark/scripts/sweep_monitor.py --fail-on-drift
```

CI-style full coverage gate:

```bash
python3 inference-benchmark/scripts/sweep_monitor.py --fail-on-drift --fail-on-missing
```

## Current Reading

Run on 2026-05-03:

- Local generated sweep state and published R2 `sweep-state.json` matched:
  138 cells, 76 `profile_infeasible` records, zero cell drift.
- Local generated sweep state and local dashboard `sweep-state.json` also
  matched.
- Published R2 `data.json` had 786 current rows.
- Local dashboard `data.json` had 690 current rows.
- Published R2 data was ahead of local data by 96 current profile-concurrency
  points.
- Against local `sweep.yaml`, published R2 data had 732 / 1842 expected
  current profile-concurrency points, leaving 1110 expected points missing.

Interpretation: the website sweep state is in sync with local YAML/state, but
local `dashboard/public/data.json` is stale relative to R2. The remaining
missing count is benchmark coverage, not a state-publication mismatch.
