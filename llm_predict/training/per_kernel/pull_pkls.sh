#!/usr/bin/env bash
# Pull trained per-kernel pkls + labeled CSV from Cloudflare R2.
# Inverse of upload_to_r2.sh.
set -euo pipefail

EP="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PKL_ROOT="$REPO_ROOT/llm_predict/profiling/data"
LABELED_CSV="$SCRIPT_DIR/data/kernels_labeled.csv"

sync_down() {
  local remote="$1" local_path="$2"
  mkdir -p "$local_path"
  echo "[*] sync s3://$BUCKET/$remote/ -> $local_path/"
  aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync "s3://$BUCKET/$remote/" "$local_path/" \
      --exclude "*" --include "*.pkl"
}

for GPU in A100 RTX3090 RTX2080Ti; do
  sync_down "profiling-data/$GPU/trained/per_kernel" \
            "$PKL_ROOT/$GPU/trained/per_kernel"
done

mkdir -p "$(dirname "$LABELED_CSV")"
echo "[*] cp s3://$BUCKET/profiling-data/kernels_labeled.csv -> $LABELED_CSV"
aws --profile "$PROFILE" --endpoint-url "$EP" s3 cp \
    "s3://$BUCKET/profiling-data/kernels_labeled.csv" "$LABELED_CSV" || true

echo "[+] done"
