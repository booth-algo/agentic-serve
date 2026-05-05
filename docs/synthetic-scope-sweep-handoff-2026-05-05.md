# Synthetic Scope Sweep Handoff — 2026-05-05

## Target

Replace the old dashboard-facing `latest` sweep surface with a real `synthetic` scope and launch the APC-aware synthetic replay grid.

This is the sweep surface to run now:

- `chat-singleturn-synth`
- `chat-multiturn-synth`
- `swebench-multiturn-synth`
- `terminalbench-multiturn-synth`
- `osworld-multiturn-synth`

`coding-singleturn` is intentionally excluded.

## Setup Created

Code now treats `synthetic` as its own scope instead of copying fixed coverage under the label `latest`.

- `scripts/compile_sweep.py`
  - `--scope synthetic` derives from the fixed C=200/320 grid.
  - Fixed profiles are remapped to `*-synth` names.
  - Every emitted row includes:
    - `DISTRIBUTIONAL_SYNTHETIC_STYLE=code`
    - `DISTRIBUTIONAL_TARGET_CHARS_PER_TOKEN=3.8`
    - `DISTRIBUTIONAL_PREFIX_AWARE=1`
    - `DISTRIBUTIONAL_SHARED_PREFIX_TOKENS=1024`
    - `DASHBOARD_SCOPE=synthetic`
    - `RESULT_SCOPE=synthetic`
- `scripts/publish_sweep_state.py`
  - materializes synthetic cells in `sweep-state.json`, so dashboard coverage tracks synthetic status directly.
- `src/workloads/profiles.py`
  - adds the five `*-synth` profiles.
- `src/benchmark/runner.py`
  - accepts `--scope synthetic`.
  - defaults `*-synth` profiles to `dashboard_scope=synthetic`.
  - keeps the existing guardrail: multi-turn sessions are floored at concurrency, so `num_sessions=1` profiles do not cap C=200/320 runs.
- Dashboard
  - UI scope is now `Synthetic replay`, not `Latest runs`.
  - legacy `scope=latest` URLs/localStorage normalize to `synthetic`.
  - serving predictions are not shown for synthetic until synthetic predictor artifacts exist.

## Current Generated Matrix

Generated locally with:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/compile_sweep.py --scope synthetic --out scripts/bench_jobs.txt
python3 scripts/publish_sweep_state.py --no-upload
```

Observed:

- `scripts/bench_jobs.txt`: 132 runnable job rows.
- `dashboard/public/sweep-state.json`: 415 total state cells, including synthetic cells.
- Synthetic state:
  - 137 synthetic cells total.
  - 134 synthetic runnable cells after known-OOM.
  - 670 expected profile-concurrency points.
  - 0 profile-level context infeasible blocks. Synthetic sweep intentionally does not inherit fixed-scope long-trace context filters, because the synthetic runner can enforce context budgets and the 75% success-rate gate should decide whether a cell is usable.
- Validation checks:
  - `coding-singleturn` does not appear in `scripts/bench_jobs.txt`.
  - all five `*-synth` profiles appear.
  - all 132 rows include the APC-aware synthetic generator env.
  - Python runner/profile tests pass.
  - Dashboard production build passes.

Note: the user remembered "1114 expected cells" for old `latest`. The currently generated first-class synthetic state is 670 profile-concurrency points after removing `coding-singleturn`, known-OOM cells, and fixed-scope profile infeasible inheritance. If the live dashboard still shows 1114, treat that as stale `latest` state or coverage UI/backend expansion drift, not the authoritative synthetic matrix.

## Launch Instructions for Claude

Do the launch from the Claude Code sweep session, not from this Codex editing session.

```bash
cd /root/agentic-serve/inference-benchmark

# Recreate the matrix after pulling/syncing these edits.
python3 scripts/compile_sweep.py --scope synthetic --out scripts/bench_jobs.txt

# Sanity checks before launch.
grep -q '# SCOPE: synthetic' scripts/bench_jobs.txt
! grep -q 'coding-singleturn' scripts/bench_jobs.txt
grep -q 'DISTRIBUTIONAL_PREFIX_AWARE=1' scripts/bench_jobs.txt
grep -q 'swebench-multiturn-synth' scripts/bench_jobs.txt

# Launch using the existing orchestrator flow.
BENCH_JOBS_SCOPE=synthetic JOBS_SCOPE=synthetic bash scripts/bench_orchestrator.sh
```

Recommended monitor pane:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/sweep_progress_report.py \
  --jobs-file scripts/bench_jobs.txt \
  --out /tmp/sweep-progress-synthetic.md \
  --history /tmp/sweep-progress-synthetic-history.md \
  --interval-seconds 300
```

Coverage reconciliation:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/reconcile_sweep_coverage.py \
  --scope synthetic \
  --write-missing-jobs /tmp/bench_jobs/missing_synthetic_bench_jobs.txt \
  --report /tmp/synthetic-coverage-report.md
```

Dashboard status dry-run:

```bash
cd /root/agentic-serve/inference-benchmark
python3 scripts/publish_sweep_state.py --no-upload
```

Only upload state/data after confirming the synthetic jobs are genuinely running or complete.

## Must-Check During Runs

- Every result JSON should have `config.dashboard_scope == "synthetic"`.
- Every result JSON should have one of the five `*-synth` profiles.
- Multi-turn result JSONs should show `profile_metadata.num_sessions >= config.concurrency`.
- `profile_metadata.num_sessions_source` should usually be `concurrency_floor` for distributional synthetic profiles.
- `summary.successful_requests / summary.num_requests >= 0.75`; runner should fail before saving bad data when below threshold.
- `workload_schema_version` / request metadata should show the APC-aware synthetic generator settings.
- `swebench` and `terminalbench` synthetic profiles are intentionally allowed on 4K/8K model-length cells; if they fail, trust the success-rate gate and inspect that specific cell instead of pre-filtering it out as fixed-scope infeasible.

## If the Sweep Setup Still Feels Bad

The current setup is improved but still has structural issues:

- Scope derivation is hardcoded in Python. Better: put derived scopes in `sweep.yaml`, e.g. `derived_scopes.synthetic.source_scope=fixed`, `profile_map`, and `extra_env`.
- `bench_jobs.txt` is a mutable singleton. Better: emit `bench_jobs.synthetic.txt`, `bench_jobs.fixed.txt`, etc., then symlink/copy the active one for the orchestrator.
- Runtime state under `/tmp/bench_jobs/state/<scope>` is fragile. Move active sweep state to `/mnt/100g/bench_state/<scope>` or another durable mount.
- `data.json` validation currently depends on fixed rows existing locally; local data snapshots can be stale. Validation should distinguish "scope exists in sweep-state" from "scope has launched rows".
- Profile infeasible rules are still too global. Synthetic currently bypasses fixed-scope profile infeasible rules in code; this policy should live declaratively in `sweep.yaml`.

The launch can proceed with the current setup, but the first three cleanup items should be done before the next large sweep.
