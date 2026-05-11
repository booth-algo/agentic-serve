#!/usr/bin/env bash
# Sync the Cloudflare R2 benchmark bucket into a local bucket-shaped mirror.
#
# Default mirror root:
#   /mnt/100g/agent-bench
#
# Local layout mirrors the bucket root:
#   /mnt/100g/agent-bench/<bucket key>
#
# This script is pull-only. It never uploads to R2 and never deletes local files
# unless --delete is explicitly provided.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_PUBLIC="$ROOT_DIR/dashboard/public"

ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
MIRROR_ROOT="${LOCAL_R2_MIRROR:-/mnt/100g/agent-bench}"
RESULTS_PREFIX="results/archived/canonical/"
SYNC_MODE="all"
HYDRATE_PUBLIC=0
DELETE_FLAG=()
SYNC_FLAGS=(--only-show-errors)

usage() {
    sed -n '1,18p' "$0"
    cat <<'EOF'

Options:
  --mirror-root PATH       Local mirror root (default: /mnt/100g/agent-bench)
  --all                   Sync the whole bucket (default)
  --current-only           Sync only json/current/ and results/archived/canonical/
  --results-prefix PREFIX  R2 results prefix for current-only/results-only
  --all-results            Sync all results/ instead of results/archived/canonical/
  --json-only              Sync only json/current/
  --results-only           Sync only results
  --hydrate-public         Copy json/current/*.json into dashboard/public/
  --delete                 Pass --delete to aws s3 sync
  --verbose                Show aws s3 sync transfer output
  --profile NAME           AWS profile (default: r2)
  --endpoint URL           R2 endpoint override
  --bucket NAME            R2 bucket name (default: agent-bench)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mirror-root)
            MIRROR_ROOT="$2"
            shift 2
            ;;
        --all)
            SYNC_MODE="all"
            shift
            ;;
        --current-only)
            SYNC_MODE="current"
            RESULTS_PREFIX="results/archived/canonical/"
            shift
            ;;
        --results-prefix)
            SYNC_MODE="results"
            RESULTS_PREFIX="$2"
            shift 2
            ;;
        --all-results)
            SYNC_MODE="results"
            RESULTS_PREFIX="results/"
            shift
            ;;
        --json-only)
            SYNC_MODE="json"
            shift
            ;;
        --results-only)
            SYNC_MODE="results"
            shift
            ;;
        --hydrate-public)
            HYDRATE_PUBLIC=1
            shift
            ;;
        --delete)
            DELETE_FLAG=(--delete)
            shift
            ;;
        --verbose)
            SYNC_FLAGS=()
            shift
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
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! command -v aws >/dev/null 2>&1; then
    echo "aws cli not found; install/configure aws or rerun on a host with aws" >&2
    exit 1
fi

mkdir -p "$MIRROR_ROOT"

if [[ "$SYNC_MODE" == "all" ]]; then
    echo "Syncing s3://$BUCKET/ -> $MIRROR_ROOT/"
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 sync \
        "s3://$BUCKET/" "$MIRROR_ROOT/" \
        "${SYNC_FLAGS[@]}" \
        "${DELETE_FLAG[@]}"
fi

if [[ "$SYNC_MODE" == "current" || "$SYNC_MODE" == "json" ]]; then
    mkdir -p "$MIRROR_ROOT/json/current"
    echo "Syncing s3://$BUCKET/json/current/ -> $MIRROR_ROOT/json/current/"
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 sync \
        "s3://$BUCKET/json/current/" "$MIRROR_ROOT/json/current/" \
        "${SYNC_FLAGS[@]}" \
        "${DELETE_FLAG[@]}"
fi

if [[ "$SYNC_MODE" == "current" || "$SYNC_MODE" == "results" ]]; then
    mkdir -p "$MIRROR_ROOT/results"
    echo "Syncing s3://$BUCKET/$RESULTS_PREFIX -> $MIRROR_ROOT/$RESULTS_PREFIX"
    mkdir -p "$MIRROR_ROOT/$RESULTS_PREFIX"
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 sync \
        "s3://$BUCKET/$RESULTS_PREFIX" "$MIRROR_ROOT/$RESULTS_PREFIX" \
        "${SYNC_FLAGS[@]}" \
        "${DELETE_FLAG[@]}"
fi

if [[ "$HYDRATE_PUBLIC" -eq 1 ]]; then
    echo "Hydrating dashboard/public from $MIRROR_ROOT/json/current/"
    mkdir -p "$DASHBOARD_PUBLIC"
    cp "$MIRROR_ROOT"/json/current/*.json "$DASHBOARD_PUBLIC"/
fi

echo "Mirror root: $MIRROR_ROOT"
du -sh "$MIRROR_ROOT" 2>/dev/null || true
