# GCS Storage — Uploading Benchmark Results

## Overview

Benchmark result JSON files are stored in Google Cloud Storage for persistence and sharing.
The dashboard can load data from either local `results/` or GCS.

## Bucket & Paths

- **Bucket:** `sequrity-experiments`
- **Prefix:** `inference-benchmark/results/`
- **Full path:** `gs://sequrity-experiments/inference-benchmark/results/`
- **Structure mirrors local:** `results/<model>_tp<N>_<engine>/<filename>.json`

## Authentication

- **Service account key:** `~/gcp-key.json`
- **Principal:** `seqlen-logs@sequrity-experiments-log.iam.gserviceaccount.com`
- **Role:** Storage Object Admin on `sequrity-experiments` bucket
- **Python package:** `google-cloud-storage` (install via `pip install google-cloud-storage`)

## Upload All Results

```bash
GOOGLE_APPLICATION_CREDENTIALS=~/gcp-key.json python3 -c "
from google.cloud import storage
from pathlib import Path
c = storage.Client()
b = c.bucket('sequrity-experiments')
count = 0
for f in sorted(Path('results').rglob('*.json')):
    blob = b.blob(f'inference-benchmark/results/{f.relative_to(\"results\")}')
    blob.upload_from_filename(str(f))
    count += 1
print(f'Uploaded {count} files')
"
```

Or use the slash command: `/sync-gcs`

## Upload a Single File

```bash
GOOGLE_APPLICATION_CREDENTIALS=~/gcp-key.json python3 -c "
from src.reporting.loader import upload_to_gcs
from pathlib import Path
upload_to_gcs(Path('results/YOUR_FILE.json'))
"
```

## Python Loader

`src/reporting/loader.py` can load results from GCS or local:
- Defaults: `GCS_BUCKET=sequrity-experiments`, `GCS_PREFIX=inference-benchmark/results/`
- Override via env vars or function args
- Lazy-inits GCS client from `~/gcp-key.json`

## After Uploading

Rebuild dashboard data so the website reflects new results:
```bash
cd dashboard && npx tsx scripts/build-data.ts
```
