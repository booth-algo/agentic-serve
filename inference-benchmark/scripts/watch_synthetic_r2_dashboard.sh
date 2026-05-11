#!/usr/bin/env bash
# Periodically compare raw R2 synthetic result files with the published
# dashboard coverage view. This is intentionally read-only.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
ENDPOINT="${R2_ENDPOINT:-https://b33fe7347f25479b27ec9680eff19b78.r2.cloudflarestorage.com}"
BUCKET="${R2_BUCKET:-agent-bench}"
PROFILE="${AWS_PROFILE:-r2}"
REPORT="${REPORT:-/tmp/synthetic-r2-dashboard-watch-latest.md}"
HISTORY="${HISTORY:-/tmp/synthetic-r2-dashboard-watch-history.md}"
RECONCILE_REPORT="${RECONCILE_REPORT:-/tmp/synthetic-coverage-watch-latest.md}"
MISSING_JOBS="${MISSING_JOBS:-/tmp/bench_jobs/missing_synthetic_distributional_bench_jobs.txt}"
AWS_LIST_CACHE="${AWS_LIST_CACHE:-/tmp/synthetic-r2-objects.txt}"
DATA_JSON_CACHE="${DATA_JSON_CACHE:-/tmp/synthetic-dashboard-json-objects.txt}"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$HISTORY" >&2
}

one_tick() {
  local generated_at
  generated_at="$(date -Is)"

  mkdir -p "$(dirname "$REPORT")" "$(dirname "$HISTORY")" "$(dirname "$MISSING_JOBS")"

  local aws_ok=0
  aws --profile "$PROFILE" \
    --endpoint-url "$ENDPOINT" \
    s3 ls "s3://$BUCKET/results/synthetic_distributional/" --recursive \
    > "$AWS_LIST_CACHE.tmp" 2> "$AWS_LIST_CACHE.err" && aws_ok=1
  if [[ "$aws_ok" -eq 1 ]]; then
    mv "$AWS_LIST_CACHE.tmp" "$AWS_LIST_CACHE"
  else
    rm -f "$AWS_LIST_CACHE.tmp"
  fi

  {
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 ls "s3://$BUCKET/json/current/data.json" || true
    aws --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3 ls "s3://$BUCKET/json/current/sweep-state.json" || true
  } > "$DATA_JSON_CACHE.tmp" 2> "$DATA_JSON_CACHE.err"
  mv "$DATA_JSON_CACHE.tmp" "$DATA_JSON_CACHE"

  local reconcile_ok=0
  python3 "$ROOT_DIR/scripts/reconcile_sweep_coverage.py" \
    --scope synthetic_distributional \
    --report "$RECONCILE_REPORT" \
    --write-missing-jobs "$MISSING_JOBS" \
    --limit 20 \
    > "$RECONCILE_REPORT.stdout" 2> "$RECONCILE_REPORT.stderr" && reconcile_ok=1

  {
    echo "# Synthetic R2/Dashboard Watch"
    echo
    echo "- generated_at: $generated_at"
    echo "- interval_seconds: $INTERVAL_SECONDS"
    echo "- raw_r2_prefix: s3://$BUCKET/results/synthetic_distributional/"
    echo "- dashboard_data: https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current/data.json"
    echo "- reconcile_report: $RECONCILE_REPORT"
    echo "- missing_jobs: $MISSING_JOBS"
    echo
    echo "## Raw R2 Synthetic Objects"
    if [[ "$aws_ok" -eq 1 ]]; then
      python3 - "$AWS_LIST_CACHE" <<'PY'
import re
import sys
from collections import Counter
from datetime import datetime, timezone

path = sys.argv[1]
profile_counts = Counter()
base_files = 0
json_files = 0
latest = None
latest_key = ""
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split(maxsplit=3)
        if len(parts) != 4:
            continue
        date_s, time_s, _size, key = parts
        try:
            ts = datetime.fromisoformat(f"{date_s}T{time_s}+00:00")
        except ValueError:
            ts = None
        if ts and (latest is None or ts > latest):
            latest = ts
            latest_key = key
        if not key.endswith(".json"):
            continue
        json_files += 1
        if key.endswith("_per_turn.json"):
            continue
        base_files += 1
        name = key.rsplit("/", 1)[-1]
        match = re.search(r"([A-Za-z0-9-]+(?:-multiturn|-singleturn)-synth)_conc\d+\.json$", name)
        if match:
            profile_counts[match.group(1)] += 1

print(f"- json_files: {json_files}")
print(f"- base_result_jsons: {base_files}")
if latest:
    age = datetime.now(timezone.utc) - latest
    print(f"- latest_object_utc: {latest.isoformat()} ({int(age.total_seconds() // 60)} min old)")
    print(f"- latest_object_key: {latest_key}")
print("- base_result_jsons_by_profile:")
for profile, count in sorted(profile_counts.items()):
    print(f"  - {profile}: {count}")
PY
    else
      echo "- status: ERROR listing R2"
      sed 's/^/  /' "$AWS_LIST_CACHE.err" || true
    fi
    echo
    echo "## Published Dashboard Coverage"
    echo "Published JSON object timestamps:"
    sed 's/^/- /' "$DATA_JSON_CACHE" || true
    echo
    if [[ "$reconcile_ok" -eq 1 ]]; then
      sed -n '1,32p' "$RECONCILE_REPORT"
    else
      echo "- status: ERROR reconciling dashboard coverage"
      sed 's/^/  /' "$RECONCILE_REPORT.stderr" || true
    fi
    echo
    echo "## Interpretation"
    echo "- Healthy: raw R2 base_result_jsons should increase after jobs finish, and published dashboard present points should increase after the rebuild action completes."
    echo "- Bad: raw R2 increases while dashboard synthetic rows/present points stay flat after a rebuild; that means the rebuild/data publish path is stale or filtering rows out."
    echo "- Bad: terminal done/skipped jobs with missing coverage stay nonzero; the orchestrator can believe work is complete while coverage is incomplete."
  } > "$REPORT.tmp"

  mv "$REPORT.tmp" "$REPORT"
  cat "$REPORT" >> "$HISTORY"
  log "wrote $REPORT"
}

while true; do
  one_tick
  sleep "$INTERVAL_SECONDS"
done
