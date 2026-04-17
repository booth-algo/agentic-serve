#!/usr/bin/env bash
# Upload trained per-kernel pkls + labeled CSV to Cloudflare R2.
#
# Expects `[r2]` profile in ~/.aws/credentials and aws CLI v2 installed.
# Bucket: s3://agent-bench/
set -euo pipefail

EP="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PKL_ROOT="$REPO_ROOT/llm_predict/profiling/data"
LABELED_CSV="$SCRIPT_DIR/data/kernels_labeled.csv"

upload() {
  local src="$1" dst="$2"
  if [ ! -e "$src" ]; then
    echo "[skip] missing $src"; return
  fi
  echo "[*] $src -> s3://$BUCKET/$dst"
  aws --profile "$PROFILE" --endpoint-url "$EP" s3 cp "$src" "s3://$BUCKET/$dst"
}

sync_dir() {
  local src="$1" dst="$2"
  if [ ! -d "$src" ]; then
    echo "[skip] missing $src"; return
  fi
  echo "[*] sync $src/ -> s3://$BUCKET/$dst/"
  aws --profile "$PROFILE" --endpoint-url "$EP" s3 sync "$src/" "s3://$BUCKET/$dst/" \
      --exclude "*" --include "*.pkl"
}

for GPU in A100 RTX3090 RTX2080Ti; do
  sync_dir "$PKL_ROOT/$GPU/trained/per_kernel" \
           "profiling-data/$GPU/trained/per_kernel"
done

upload "$LABELED_CSV" "profiling-data/kernels_labeled.csv"

echo "[+] done"
