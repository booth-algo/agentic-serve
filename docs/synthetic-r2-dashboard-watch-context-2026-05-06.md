# Synthetic R2/Dashboard Watch Context — 2026-05-06

This note records the current synthetic sweep visibility monitor and the coverage pipeline risks it is watching.

## Active Watchdog

A tmux monitor is running on Hetzner:

```bash
tmux attach -t synthetic-r2-watch
```

It runs every 300 seconds:

```bash
/root/agentic-serve/inference-benchmark/scripts/watch_synthetic_r2_dashboard.sh
```

Outputs:

- Latest watch report: `/tmp/synthetic-r2-dashboard-watch-latest.md`
- Watch history: `/tmp/synthetic-r2-dashboard-watch-history.md`
- Reconcile report: `/tmp/synthetic-coverage-watch-latest.md`
- Missing jobs subset: `/tmp/bench_jobs/missing_synthetic_bench_jobs.txt`

Quick status:

```bash
sed -n '1,140p' /tmp/synthetic-r2-dashboard-watch-latest.md
```

## Current Finding

At the last check:

- Raw R2 synthetic base result JSONs: `353`
- Raw synthetic latest object: `2026-05-06 11:48 UTC`
- Published dashboard `data.json`: `2026-05-06 09:43 UTC`
- Published `sweep-state.json`: `2026-05-06 11:53 UTC`
- Dashboard synthetic rows: `269`
- Expected synthetic cells: `660`
- Present expected cells: `106 / 660`
- Missing expected cells: `554`
- Stale terminal/blocking jobs: `37`

Interpretation: sweep status is fresh, but dashboard coverage data is stale. If the website only reads `json/current/data.json`, it will not reflect raw synthetic results uploaded after `09:43 UTC` until the dashboard rebuild republishes `data.json`.

## Pipeline

```text
scripts/sweep.yaml
  -> scripts/compile_sweep.py --scope synthetic_distributional
  -> scripts/bench_jobs.txt
  -> scripts/bench_orchestrator.sh
  -> remote /tmp/results/synthetic_distributional/...
  -> R2 s3://agent-bench/results/synthetic_distributional/...
  -> GitHub Action "Rebuild Dashboard Data"
  -> dashboard/scripts/build-data.ts
  -> R2 s3://agent-bench/json/current/data.json
  -> website coverage page
```

## Important Files

- `/root/agentic-serve/inference-benchmark/scripts/watch_synthetic_r2_dashboard.sh`
- `/root/agentic-serve/inference-benchmark/scripts/reconcile_sweep_coverage.py`
- `/root/agentic-serve/inference-benchmark/scripts/compile_sweep.py`
- `/root/agentic-serve/inference-benchmark/scripts/bench_orchestrator.sh`
- `/root/agentic-serve/inference-benchmark/dashboard/scripts/build-data.ts`
- `/root/agentic-serve/inference-benchmark/dashboard/src/components/CoveragePage.tsx`
- `/root/agentic-serve/inference-benchmark/docs/synthetic-scope-sweep-handoff-2026-05-05.md`

## Robustness Issues

1. `bench_jobs.txt` is a mutable singleton. Current observed file had `134` rows while compiled synthetic dry-run had `132`.
2. Runtime state lives under `/tmp/bench_jobs/state/<scope>`, so it is fragile and can drift.
3. `sweep-state.json` can be fresh while `data.json` is stale.
4. Orchestrator status is coarse job-level, but coverage is profile by concurrency. Reconcile showed `done`/`skipped` jobs still missing coverage.
5. Dashboard build filters rows, so raw R2 count and dashboard coverage can diverge.

## Commands

Check raw R2 synthetic result count:

```bash
aws --profile r2 \
  --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com \
  s3 ls s3://agent-bench/results/synthetic_distributional/ --recursive | wc -l
```

Check published dashboard artifact timestamps:

```bash
aws --profile r2 --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com \
  s3 ls s3://agent-bench/json/current/data.json

aws --profile r2 --endpoint-url https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com \
  s3 ls s3://agent-bench/json/current/sweep-state.json
```

Run one reconcile manually:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/reconcile_sweep_coverage.py \
  --scope synthetic_distributional \
  --report /tmp/synthetic-coverage-watch-latest.md \
  --write-missing-jobs /tmp/bench_jobs/missing_synthetic_bench_jobs.txt \
  --limit 30
```

Compile expected synthetic grid dry-run:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/compile_sweep.py --scope synthetic_distributional --dry-run >/tmp/bench_jobs.synthetic.generated.txt
```

Expected compile sanity:

- `132` emitted rows
- no `coding-singleturn`
- all rows include:
  - `DISTRIBUTIONAL_PREFIX_AWARE=1`
  - `DISTRIBUTIONAL_SHARED_PREFIX_TOKENS=1024`
  - `RESULT_SCOPE=synthetic_distributional`
  - `DASHBOARD_SCOPE=synthetic_distributional`

## What To Watch

Healthy:

- Raw R2 synthetic base result count increases after jobs finish.
- After GitHub rebuild, `json/current/data.json` timestamp becomes newer.
- Dashboard synthetic present cells increases.
- Stale terminal/blocking jobs decreases.

Bad:

- Raw R2 grows but `data.json` stays old after rebuild.
- `data.json` rebuilds but synthetic present cells does not increase.
- Stale terminal jobs remain nonzero.
- `bench_jobs.txt` drifts further from compiled synthetic matrix.

Do not claim coverage is complete based only on orchestrator `done`. Use `reconcile_sweep_coverage.py` as the coverage source of truth.
