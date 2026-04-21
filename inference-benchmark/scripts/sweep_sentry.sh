#!/usr/bin/env bash
# sweep_sentry — Claude-as-sentry harness for the benchmark sweep.
#
# Fires every N hours via cron. Collects fresh state from R2 + orchestrator
# log + local job state, then hands the raw context to `claude -p` with a
# scoped diagnosis prompt. Claude appends a dated entry to the journal
# summarizing findings and proposing (but NOT applying) fixes.
#
# Designed to be cheap, read-heavy, write-light. This first version is
# diagnosis-only — it does not edit bench_jobs.txt, sweep.yaml, or the
# state dir. Promote those actions manually from the journal once trust
# in the recommendations is established.
#
# Cron:
#   0 */2 * * * bash /root/agentic-serve/inference-benchmark/scripts/sweep_sentry.sh >> /tmp/sweep_sentry.cron.log 2>&1
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
JOURNAL_DIR="$REPO_ROOT/.claude/sweep_sentry"
JOURNAL="$JOURNAL_DIR/journal.md"
LOG="/tmp/sweep_sentry.log"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

mkdir -p "$JOURNAL_DIR"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

log "sentry tick start"

# --- Gather context ---

# Remote sweep-state.json (the dashboard's live source of truth).
curl -fsSL --max-time 15 "https://pub-38e30ed030784867856634f1625c7130.r2.dev/sweep-state.json" \
    -o "$TMP_DIR/sweep-state.json" 2>>"$LOG" \
    || { log "WARN: R2 fetch failed; falling back to local"; cp "$REPO_ROOT/inference-benchmark/dashboard/public/sweep-state.json" "$TMP_DIR/sweep-state.json" 2>/dev/null || true; }

# Orchestrator log tail (last 80 lines ~= last 40 hours of ticks).
tail -80 /tmp/bench_orchestrator.log 2>/dev/null > "$TMP_DIR/orchestrator.tail.log" || true

# Job state summary.
{
    echo "# Non-done job states:"
    for f in /tmp/bench_jobs/state/*.status; do
        [[ -f "$f" ]] || continue
        s=$(cat "$f")
        if [[ "$s" != "done" ]]; then
            name=$(basename "$f" .status)
            att=$(cat "${f%.status}.attempt" 2>/dev/null || echo "-")
            ovr=$(cat "${f%.status}.max_len_override" 2>/dev/null || echo "-")
            printf "  %-45s %s  attempt=%s  max_len_override=%s\n" "$name" "$s" "$att" "$ovr"
        fi
    done
    echo ""
    echo "# Counts:"
    echo "  done      : $(grep -l done /tmp/bench_jobs/state/*.status 2>/dev/null | wc -l)"
    echo "  abandoned : $(grep -l abandoned /tmp/bench_jobs/state/*.status 2>/dev/null | wc -l)"
    echo "  running   : $(grep -l running /tmp/bench_jobs/state/*.status 2>/dev/null | wc -l)"
    echo "  pending   : $(grep -l pending /tmp/bench_jobs/state/*.status 2>/dev/null | wc -l)"
} > "$TMP_DIR/job-state.txt"

# Host liveness snapshot (best effort; if ssh fails, note and continue).
{
    for H in gpu-4 3090 2080ti; do
        echo "## $H"
        ssh -o ConnectTimeout=5 -o BatchMode=yes "$H" '
            port=$(ss -ltn 2>/dev/null | awk "\$4 ~ /:8089\$/ {print \$4; exit}")
            echo "  vllm port 8089: ${port:-idle}"
            ps -ef | grep -E "sweep_all_profiles|vllm.entrypoints.openai" | grep -v grep | head -1 | cut -c1-200 | sed "s/^/  proc: /"
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | awk -F", " "\$2+0>5000 {printf \"  gpu%s: %s | %s util\\n\", \$1, \$2, \$3}" | head -8
        ' 2>&1 | head -20
        echo ""
    done
} > "$TMP_DIR/hosts.txt"

JOURNAL_TAIL=""
if [[ -f "$JOURNAL" ]]; then
    JOURNAL_TAIL=$(tail -100 "$JOURNAL")
fi

# --- Assemble prompt ---

PROMPT=$(cat <<EOF
You are the sentry for a benchmark sweep running across three GPU hosts
(gpu-4 = 8x A100-40GB, 3090 = 8x RTX 3090, 2080ti = 8x RTX 2080Ti).

A Linux cron runs \`bench_orchestrator.sh\` every 30 minutes to dispatch
pending jobs, finalize completed ones, and publish sweep-state.json to
R2. You fire separately every 2 hours to diagnose the sweep health and
suggest corrections.

## Current context

### sweep-state.json (dashboard truth)
$(cat "$TMP_DIR/sweep-state.json" 2>/dev/null | head -400 || echo "(unavailable)")

### orchestrator log tail
$(cat "$TMP_DIR/orchestrator.tail.log")

### local job state
$(cat "$TMP_DIR/job-state.txt")

### live host snapshot
$(cat "$TMP_DIR/hosts.txt")

### previous journal tail (for continuity)
$JOURNAL_TAIL

## Task

PRINT (to stdout only — do NOT use Write/Edit tools; the harness
handles file IO) a concise markdown diagnosis entry.

Use this exact format, keeping the entire entry under 30 lines:

\`\`\`
## $(date -Is)

### Health
- one-line overall status: healthy / degraded / stuck / idle / error

### Observations
- bullet list of notable observations since last tick
  (new completions, new abandonments, hosts idle, etc.)

### Anomalies
- bullet list of concerning patterns (cells stuck > 1h in running,
  repeated abandon-retry cycles, hosts unreachable, VRAM leaks).
  Cite specific job IDs and attempt counts. Skip this section if
  nothing anomalous.

### Suggested actions
- concrete, bounded suggestions like "reset state for
  gpu-4_Qwen2.5-72B_tp4_single (crashed @ max_len=4096; try
  max_len=2048)". Each suggestion MUST name the exact job ID and
  the exact file/line or command to change. Skip if none.
- If you would normally push a code change, say so but DO NOT apply
  it — propose only, humans promote.

### Next tick watch
- one-line hint on what to look for in the next tick.
\`\`\`

Do NOT use Write/Edit tools at all. Do NOT edit bench_jobs.txt,
sweep.yaml, state files, or commit/push anything. Print the
markdown entry on stdout; the harness captures it and appends to
the journal. Read-only exploration of the repo via Read/Grep/Glob
is fine if you need more detail to diagnose.
EOF
)

# --- Invoke Claude (non-interactive, single turn) ---

log "invoking claude (prompt ~$(echo "$PROMPT" | wc -c) chars)"

echo "$PROMPT" | claude -p --output-format text >> "$JOURNAL" 2>>"$LOG" || {
    log "ERROR: claude invocation failed"
    exit 1
}

# Ensure a trailing newline and a horizontal rule between entries.
printf '\n---\n' >> "$JOURNAL"

log "sentry tick done; latest entry appended to $JOURNAL"
