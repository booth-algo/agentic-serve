# R2 Data Layout

Bucket: `agent-bench`

Public base URL used by the dashboard:
`https://pub-38e30ed030784867856634f1625c7130.r2.dev/`

Private S3 endpoint:
`https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com`

## Current Top-Level Prefixes

As of 2026-05-10 the bucket root is intentionally small:

| Prefix | Role | Keep active? |
| --- | --- | --- |
| `archive/` | Dated snapshots of retired generated artifacts and retired auxiliary prefixes. | Yes |
| `data/` | Source trace datasets such as SWE-bench, TerminalBench, OSWorld, and coding single-turn prompts. | Yes |
| `json/` | Generated dashboard/state bundles, with `json/current/` as the public dashboard feed. | Yes |
| `results/` | Raw benchmark result JSONs from benchmark sweeps. | Archive old runs; keep new canonical runs active. |
| bucket root `*.json` | Legacy generated dashboard/state bundles from the pre-`json/current/` layout. | No |

The old top-level `bench_mse/`, `perkernel/`, `predictor/`, and
`profiling-data/` prefixes were moved to
`archive/2026-05-10-prefix-cleanup/`. The active dashboard publication path is
`json/current/`; the local `/mnt/100g/agent-bench` tree is the durable source of
truth for rebuilds and private dashboard freshness.

The generated dashboard JSONs are not under `results/` because `results/` is the
raw measurement layer. Files like `data.json`, `sweep-state.json`, and
`predictor-coverage.json` are derived publication artifacts. They should be
grouped under a generated-artifact prefix, not mixed into raw benchmark output.

## Current Results Layout

```text
s3://agent-bench/
  archive/
    2026-05-10-prefix-cleanup/
      bench_mse/
      perkernel/
      predictor/
      profiling-data/
  data/                         # source workload/trace corpora
  results/
    trace_replay/               # real trace replay rows; matches Hugging Face split naming
    synthetic_distributional/   # APC-aware synthetic distributional rows
    archived/
      canonical/                # retired former results/current/ outputs
      fixed-grid/               # retired former fixed-grid outputs
      mse/                      # retired MSE validation outputs
      <stamp>/                  # dated cleanup archives
  json/
    current/                    # generated dashboard/state bundles
    archive/<stamp>/            # snapshots of old generated bundles
```

Recommended active generated JSON keys:

```text
json/current/data.json
json/current/sweep-state.json
json/current/profiling-state.json
json/current/predictor-coverage.json
json/current/roofline-quadrant.json
json/current/gemm-extrapolation.json
json/current/gemm-eval.json
json/current/serving-predictions.json
```

Do not publish generated JSONs at the bucket root. Current publishers should
write only to `json/current/...`; old root JSONs should be retained only in a
dated archive snapshot.

## Archive Policy

Use a date-stamped archive name that describes the boundary:

```text
2026-04-30-pre-distributional
```

The safe sequence for future cleanup is:

1. Inventory the current bucket.
2. Copy retired generated JSONs to `json/archive/<stamp>/`.
3. Copy retired auxiliary prefixes to `archive/<stamp>/`.
4. Copy old raw result prefixes to `results/archived/<stamp>/`.
5. Verify object counts and byte totals.
6. Delete old source paths only after an explicit cleanup approval.

The `data/`, `results/trace_replay/`,
`results/synthetic_distributional/`, and `json/current/` prefixes are the
active R2 surfaces. `results/archived/` is retained for old canonical,
fixed-grid, MSE, and dated cleanup runs. Predictor/profiling JSONs are retained
as archived dashboard artifacts; their former local source tree was removed from
the repository, so regenerate them only from an explicit external archive.

## Helper

Dry-run the generated JSON archive/copy plan:

```bash
python inference-benchmark/scripts/r2_archive.py --archive-name 2026-04-30-pre-distributional
```

Execute non-destructive copies:

```bash
python inference-benchmark/scripts/r2_archive.py \
  --archive-name 2026-04-30-pre-distributional \
  --execute
```

This copies generated JSON artifacts only. It does not delete anything.
