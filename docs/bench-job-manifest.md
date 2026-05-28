# Benchmark Job Manifest

Created: 2026-05-11T13:36:03+00:00

`inference-benchmark/scripts/sweep.yaml` is the editable source of truth for
benchmark launch configuration. Generated `bench_jobs` artifacts are derived
from it and should not be edited directly.

## Runtime Contract

- `BENCH_JOBS_SCOPE` selects the dispatch scope, for example
  `synthetic_distributional`.
- `BENCH_JOBS_CONFIG` can point the orchestrator at an alternate sweep YAML.
- `BENCH_JOBS_MANIFEST` can choose where the generated JSON manifest is written.
- `BENCH_JOBS_CACHE_DIR` chooses the cache directory for generated row and JSON
  artifacts.
- `BENCH_JOBS_FILE` remains a legacy override for an explicit pipe-delimited
  row file. When it is unset, the orchestrator compiles fresh jobs from
  `sweep.yaml` on every tick.

## Formats

The structured generated format is JSON:

```bash
python3 inference-benchmark/scripts/compile_sweep.py \
  --scope synthetic_distributional \
  --format json \
  --out /tmp/bench_jobs/bench_jobs.synthetic_distributional.json
```

The pipe-delimited format still exists as an internal compatibility row stream
for shell code:

```bash
python3 inference-benchmark/scripts/compile_sweep.py \
  --scope synthetic_distributional \
  --format text \
  --out /tmp/bench_jobs/bench_jobs.synthetic_distributional.txt
```

## Stop Condition

The checked-in `bench_jobs.txt` must not be treated as production launch state.
Production should compile jobs from `sweep.yaml` using the requested scope so
concurrency/profile changes are picked up by code without manually rewriting a
singleton file.
