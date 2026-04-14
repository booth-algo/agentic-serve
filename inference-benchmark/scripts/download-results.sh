#!/bin/bash
# Download benchmark results from Cloudflare R2
# Usage: ./scripts/download-results.sh [--filter PATTERN]
#
# Examples:
#   ./scripts/download-results.sh                    # download all
#   ./scripts/download-results.sh --filter a100_     # only A100 results
#   ./scripts/download-results.sh --filter Llama      # only Llama models

set -euo pipefail

ENDPOINT="https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com"
BUCKET="s3://agent-bench/results/"
DEST="$(dirname "$0")/../results/"
PROFILE="r2"

# Parse args
FILTER=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --filter) FILTER="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        *) echo "Usage: $0 [--filter PATTERN] [--profile AWS_PROFILE]"; exit 1 ;;
    esac
done

# Check aws cli
if ! command -v aws &>/dev/null; then
    echo "Error: aws cli not found. Install with: brew install awscli / pip install awscli"
    exit 1
fi

# Check credentials
if ! aws s3 ls "$BUCKET" --endpoint-url "$ENDPOINT" --profile "$PROFILE" &>/dev/null; then
    echo "Error: Cannot access R2. Configure credentials:"
    echo "  aws configure --profile r2"
    echo "  # Access Key ID: <from Cloudflare R2 API tokens>"
    echo "  # Secret Access Key: <from Cloudflare R2 API tokens>"
    echo "  # Region: auto"
    exit 1
fi

mkdir -p "$DEST"

if [[ -n "$FILTER" ]]; then
    echo "Downloading results matching '$FILTER'..."
    aws s3 sync "$BUCKET" "$DEST" \
        --endpoint-url "$ENDPOINT" \
        --profile "$PROFILE" \
        --exclude "*" --include "*${FILTER}*"
else
    echo "Downloading all results..."
    aws s3 sync "$BUCKET" "$DEST" \
        --endpoint-url "$ENDPOINT" \
        --profile "$PROFILE"
fi

echo "Done. Results in: $DEST"
ls -d "$DEST"*/ 2>/dev/null | wc -l | xargs -I{} echo "{} result directories"
