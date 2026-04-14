# Results Data Pipeline

## End-to-End Flow

```
Benchmark run → results/*.json → GCS upload → build-data.ts → data.json → Dashboard
```

## 1. Benchmark Output

Each benchmark run produces a JSON file in `results/<model>_tp<N>_<engine>/`:
```
results/
├── Llama-3.1-8B_tp1_sglang/
│   ├── Llama-3.1-8B_tp1_sglang_chatbot-short_conc1.json
│   ├── Llama-3.1-8B_tp1_sglang_chatbot-short_conc10.json
│   └── ...
├── Qwen2.5-72B_tp2_sglang/
│   └── ...
└── smoke_test.json
```

Each JSON contains:
- `config`: model, backend, profile, concurrency, mode, flags
- `summary`: aggregated metrics (TTFT, TPOT, E2EL percentiles, throughput)
- `requests`: per-request timing data
- `per_turn` (if multi-turn): per-turn aggregated stats
- `scatter_data` (if enabled): input_tokens vs ttft scatter points

## 2. GCS Upload

Upload to `gs://sequrity-experiments/inference-benchmark/results/`:
```bash
/sync-gcs
```
See `gcs-storage.md` for details.

## 3. Dashboard Data Build

`dashboard/scripts/build-data.ts` reads all result JSONs and produces `dashboard/public/data.json`:
```bash
cd dashboard && npx tsx scripts/build-data.ts
```

The build script:
- Walks `results/` recursively for `*.json` files
- Skips `smoke_test`, `test_`, `rng_` prefixed files
- Enriches each result with: `hardware`, `quant`, `modelShort`, `seriesKey`, `filename`
- Extracts `perTurn` and `scatterData` if present
- Writes the full array to `public/data.json`

## 4. Python Loader (for analysis)

`src/reporting/loader.py` provides `load_all()` which returns a pandas DataFrame:
- Loads from local `results/` by default
- Can load from GCS with env vars: `GCS_BUCKET`, `GCS_PREFIX`
- Auto-detects hardware from filename patterns (H100, A6000, etc.)

## Current Inventory

Old results (2xH100, March 2026, old profile names) archived to `gs://sequrity-experiments/inference-benchmark/results/archive/2xH100-mar2026/`.

New round (4xH100, April 2026) starting fresh with current profiles from `src/workloads/profiles.py`.
See `.claude/docs/gcs_structure.md` for the full GCS layout and proposed v2 structure.
