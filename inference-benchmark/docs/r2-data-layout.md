# R2 Data Layout

Bucket: `agent-bench`

Public base URL used by the dashboard:
`https://pub-38e30ed030784867856634f1625c7130.r2.dev/`

Private S3 endpoint:
`https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com`

## Current Top-Level Prefixes

As of 2026-04-30 the bucket is split into a few different kinds of data:

| Prefix | Role | Keep active? |
| --- | --- | --- |
| `data/` | Source trace datasets such as SWE-bench, TerminalBench, OSWorld, and coding single-turn prompts. | Yes |
| `profiling-data/` | Kernel profiling source data and raw profiling assets. | Yes |
| `predictor/` | Predictor artifacts: GEMM tables, elementwise calibration, trained models, serving shapes. | Yes |
| `results/` | Raw benchmark result JSONs from benchmark sweeps. | Archive old runs; keep new canonical runs active. |
| bucket root `*.json` | Generated dashboard/state bundles consumed by the website. | Move toward `json/current/`. |

The generated dashboard JSONs are not under `results/` because `results/` is the
raw measurement layer. Files like `data.json`, `sweep-state.json`, and
`predictor-coverage.json` are derived publication artifacts. They should be
grouped under a generated-artifact prefix, not mixed into raw benchmark output.

## Proposed Layout

```text
s3://agent-bench/
  data/                         # source workload/trace corpora
  profiling-data/               # raw/profiled kernel data
  predictor/                    # trained predictor inputs/artifacts
  results/
    current/                    # canonical benchmark outputs for the current paper surface
    archive/<stamp>/            # old raw benchmark outputs
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

Keep root copies temporarily while the live dashboard and publish scripts are
transitioned. Once the dashboard fetches `json/current/...` and all publishers
dual-publish or publish only there, the root JSONs can be removed.

## Archive Policy

Use a date-stamped archive name that describes the boundary:

```text
2026-04-30-pre-distributional
```

The safe sequence is:

1. Inventory the current bucket.
2. Copy generated root JSONs to `json/archive/<stamp>/`.
3. Copy the same generated JSONs to `json/current/`.
4. Leave root JSONs in place until the website has switched to `json/current/`.
5. Copy old raw result prefixes to `results/archive/<stamp>/`.
6. Verify object counts and byte totals.
7. Delete old source paths only after an explicit cleanup approval.

The `data/`, `profiling-data/`, and `predictor/` prefixes are reusable inputs
and should not be archived as old benchmark results.

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
