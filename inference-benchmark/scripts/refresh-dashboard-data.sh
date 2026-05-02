#!/usr/bin/env bash
# Sync benchmark results from R2, then rebuild dashboard/public/data.json.
#
# Examples:
#   bash scripts/refresh-dashboard-data.sh
#   bash scripts/refresh-dashboard-data.sh --filter current/
#   bash scripts/refresh-dashboard-data.sh --skip-sync --output /tmp/data.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_DIR="$ROOT_DIR/dashboard"
RESULTS_DIR="$ROOT_DIR/results"

ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
FILTER=""
SKIP_SYNC=0
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --endpoint)
            ENDPOINT="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --skip-sync)
            SKIP_SYNC=1
            shift
            ;;
        -h|--help)
            sed -n '1,12p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "$RESULTS_DIR"

if [[ "$SKIP_SYNC" -eq 0 ]]; then
    if ! command -v aws >/dev/null 2>&1; then
        echo "aws cli not found; install/configure aws or rerun with --skip-sync" >&2
        exit 1
    fi

    echo "Syncing R2 benchmark results into $RESULTS_DIR"
    if [[ -n "$FILTER" ]]; then
        aws s3 sync "s3://$BUCKET/results/" "$RESULTS_DIR/" \
            --endpoint-url "$ENDPOINT" \
            --profile "$PROFILE" \
            --exclude "*" \
            --include "*${FILTER}*"
    else
        aws s3 sync "s3://$BUCKET/results/" "$RESULTS_DIR/" \
            --endpoint-url "$ENDPOINT" \
            --profile "$PROFILE"
    fi
fi

echo "Rebuilding dashboard benchmark data"
if [[ -n "$OUTPUT" ]]; then
    (cd "$DASHBOARD_DIR" && DASHBOARD_DATA_OUTPUT="$OUTPUT" npx tsx scripts/build-data.ts)
else
    (cd "$DASHBOARD_DIR" && npx tsx scripts/build-data.ts)
fi
