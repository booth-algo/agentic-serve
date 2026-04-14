# Sync benchmark results to GCS

Upload benchmark result JSON files to `gs://sequrity-experiments/inference-benchmark/results/`.

## Prerequisites
- Service account key: `~/gcp-key.json`
- Principal: `seqlen-logs@sequrity-experiments-log.iam.gserviceaccount.com`
- Role: Storage Object Admin on `sequrity-experiments` bucket

## Usage

### Upload all results
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

### Upload a single result
```bash
GOOGLE_APPLICATION_CREDENTIALS=~/gcp-key.json python3 -c "
from src.reporting.loader import upload_to_gcs
from pathlib import Path
upload_to_gcs(Path('results/YOUR_FILE.json'))
"
```

### Update dashboard data
After uploading new results, rebuild the dashboard data:
```bash
cd dashboard && npx tsx scripts/build-data.ts
git add public/data.json && git commit -m 'Update benchmark data' && git push
```

## GCS paths
- Bucket: `sequrity-experiments`
- Prefix: `inference-benchmark/results/`
- Full: `gs://sequrity-experiments/inference-benchmark/results/`
