import type { CoverageBlockersState } from '../types-coverage-blockers';

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || 'unknown';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function pct(present: number, expected: number): string {
  if (!expected) return '0.0%';
  return `${((present / expected) * 100).toFixed(1)}%`;
}

function countText(counts: Record<string, number>): string {
  const order = ['done', 'running', 'pending', 'skipped', 'failed', 'known_oom'];
  const parts = order.filter((key) => counts[key]).map((key) => `${key} ${counts[key]}`);
  for (const key of Object.keys(counts).sort()) {
    if (!order.includes(key)) parts.push(`${key} ${counts[key]}`);
  }
  return parts.join(' · ') || 'none';
}

function failureText(counts?: Record<string, number>): string {
  if (!counts) return 'none';
  const labels: Record<string, string> = {
    oom_or_kv_cache: 'OOM/KV',
    driver_failure: 'driver',
    success_rate_below_min: 'success-rate',
    benchmark_failed: 'bench failed',
    zero_results: 'zero results',
    incomplete_outputs: 'incomplete',
    unknown: 'unknown',
  };
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .map(([key, count]) => `${labels[key] ?? key} ${count}`)
    .join(' · ') || 'none';
}

function attemptText(attempt?: number | null, maxAttempts?: number | null): string | null {
  if (attempt == null) return null;
  if (maxAttempts != null) return `${attempt}/${maxAttempts} attempts`;
  return `${attempt} attempts`;
}

function statusTone(status: string, disposition?: string | null): { bg: string; border: string; text: string } {
  if (disposition === 'na') {
    return { bg: 'rgba(100,181,246,0.12)', border: 'rgba(100,181,246,0.38)', text: '#64b5f6' };
  }
  if (disposition === 'failed') {
    return { bg: 'rgba(248,81,73,0.14)', border: 'rgba(248,81,73,0.42)', text: '#f85149' };
  }
  if (status === 'running') {
    return { bg: 'rgba(88,166,255,0.12)', border: 'rgba(88,166,255,0.35)', text: '#58a6ff' };
  }
  if (status === 'pending') {
    return { bg: 'rgba(210,153,34,0.14)', border: 'rgba(210,153,34,0.4)', text: '#d29922' };
  }
  return { bg: 'rgba(248,81,73,0.14)', border: 'rgba(248,81,73,0.42)', text: '#f85149' };
}

export function CoverageBlockersPanel({
  blockersState,
  loading,
  error,
  compact = false,
}: {
  blockersState: CoverageBlockersState | null;
  loading: boolean;
  error: string | null;
  compact?: boolean;
}) {
  if (loading) {
    return (
      <section className="rounded-lg border border-[#21262d] bg-[#161b22] p-4 text-sm text-[#8b949e]">
        Loading coverage blockers...
      </section>
    );
  }

  if (error) {
    return (
      <section className="rounded-lg border border-[#d29922]/35 bg-[#d29922]/10 p-4">
        <div className="text-sm font-semibold text-[#d29922]">Coverage blocker state unavailable</div>
        <div className="mt-1 text-xs text-[#8b949e]">Failed to load coverage-blockers.synthetic_distributional.json: {error}</div>
      </section>
    );
  }

  if (!blockersState) return null;

  const observedPresent = blockersState.observed_present_points ?? blockersState.present_points;
  const optionalPresent = blockersState.optional_present_points_count ?? 0;
  const requiredPoints = blockersState.coverage_required_points ?? blockersState.expected_points;
  const missingRequired = blockersState.coverage_missing_required_points ?? blockersState.missing_points;
  const naPoints = blockersState.coverage_na_points ?? 0;
  const failedPoints = blockersState.coverage_failed_points ?? 0;
  const coverage = pct(observedPresent, requiredPoints);
  const exhausted = blockersState.reset_exhausted.length;
  const reset = blockersState.reset_performed.length;
  const visibleBlockers = blockersState.blockers
    .slice()
    .sort((a, b) => b.missing_count - a.missing_count || a.job_id.localeCompare(b.job_id))
    .slice(0, compact ? 4 : 8);
  const borderColor = failedPoints > 0
    ? 'border-[#f85149]/45'
    : naPoints > 0
      ? 'border-[#64b5f6]/40'
      : blockersState.stale_terminal_jobs > 0
      ? 'border-[#d29922]/45'
      : 'border-[#30363d]';
  const bgColor = failedPoints > 0
    ? 'bg-[#f85149]/10'
    : naPoints > 0
      ? 'bg-[#64b5f6]/10'
      : blockersState.stale_terminal_jobs > 0
      ? 'bg-[#d29922]/10'
      : 'bg-[#161b22]';

  return (
    <section className={`rounded-lg border ${borderColor} ${bgColor} p-4`}>
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold uppercase tracking-wide text-[#e6edf3]">Synthetic Coverage Blockers</h3>
            <span className="rounded border border-[#30363d] bg-[#0d1117] px-2 py-0.5 font-mono text-[11px] text-[#8b949e]">
              {coverage}
            </span>
            {naPoints > 0 && (
              <span className="rounded border border-[#64b5f6]/40 bg-[#64b5f6]/10 px-2 py-0.5 text-[11px] font-medium text-[#64b5f6]">
                {naPoints.toLocaleString()} N/A attempted
              </span>
            )}
            {failedPoints > 0 && (
              <span className="rounded border border-[#f85149]/40 bg-[#f85149]/12 px-2 py-0.5 text-[11px] font-medium text-[#f85149]">
                {failedPoints.toLocaleString()} failed cells
              </span>
            )}
            {exhausted > 0 && (
              <span className="rounded border border-[#30363d] bg-[#0d1117] px-2 py-0.5 text-[11px] font-medium text-[#8b949e]">
                {exhausted} requeue capped
              </span>
            )}
            {reset > 0 && (
              <span className="rounded border border-[#58a6ff]/35 bg-[#58a6ff]/10 px-2 py-0.5 text-[11px] font-medium text-[#58a6ff]">
                {reset} requeued this scan
              </span>
            )}
          </div>
          <div className="mt-1 text-xs text-[#8b949e]">
            Generated {formatTime(blockersState.generated_at)} · {observedPresent.toLocaleString()}/{requiredPoints.toLocaleString()} fillable observed{optionalPresent > 0 ? ` · ${optionalPresent.toLocaleString()} optional` : ''} · {missingRequired.toLocaleString()} missing fillable · {blockersState.expected_points.toLocaleString()} grid points
          </div>
        </div>
        <div className="grid gap-2 text-xs text-[#8b949e] sm:grid-cols-3 lg:min-w-[34rem]">
          <div className="rounded border border-[#30363d] bg-[#0d1117] px-3 py-2">
            <div className="text-[11px] uppercase tracking-wide text-[#6e7681]">Terminal Blockers</div>
            <div className="mt-1 font-mono text-lg font-semibold text-[#f85149]">{blockersState.stale_terminal_jobs}</div>
          </div>
          <div className="rounded border border-[#30363d] bg-[#0d1117] px-3 py-2">
            <div className="text-[11px] uppercase tracking-wide text-[#6e7681]">Missing Jobs</div>
            <div className="mt-1 font-mono text-lg font-semibold text-[#d29922]">{blockersState.jobs_with_missing_coverage}</div>
          </div>
          <div className="rounded border border-[#30363d] bg-[#0d1117] px-3 py-2">
            <div className="text-[11px] uppercase tracking-wide text-[#6e7681]">Requeue Cap</div>
            <div className="mt-1 font-mono text-lg font-semibold text-[#c9d1d9]">{blockersState.max_requeues}</div>
          </div>
        </div>
      </div>

      <div className="mt-3 rounded border border-[#30363d] bg-[#0d1117] px-3 py-2 text-xs text-[#8b949e]">
        <span className="font-medium text-[#c9d1d9]">Job states:</span> {countText(blockersState.job_status_counts)}
        <span className="mx-2 text-[#30363d]">|</span>
        <span className="font-medium text-[#c9d1d9]">Missing by state:</span> {countText(blockersState.missing_jobs_by_status)}
        <span className="mx-2 text-[#30363d]">|</span>
        <span className="font-medium text-[#c9d1d9]">Failure classes:</span> {failureText(blockersState.failure_category_counts)}
      </div>

      {visibleBlockers.length > 0 ? (
        <div className="mt-3 grid gap-2">
          {visibleBlockers.map((blocker) => {
            const tone = statusTone(blocker.status, blocker.coverage_disposition);
            const attempts = attemptText(blocker.failure?.attempt ?? blocker.attempt, blocker.failure?.max_attempts);
            // Prefer the artifact's structured label (e.g. "N/A — OOM at max
            // gpu_mem", "failed — model not staged", "TODO — raise gpu_mem");
            // fall back to the coarse disposition for older artifacts.
            const dispositionLabel = blocker.coverage_label
              ?? (blocker.coverage_disposition === 'na'
                ? 'N/A attempted'
                : blocker.coverage_disposition === 'failed'
                  ? 'failed coverage'
                  : blocker.coverage_disposition === 'todo'
                    ? 'TODO'
                    : null);
            return (
              <div key={blocker.job_id} className="rounded-md border border-[#30363d] bg-[#0d1117] p-3">
                <div className="flex flex-col gap-2 md:flex-row md:items-start md:justify-between">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="truncate font-mono text-xs font-semibold text-[#e6edf3]" title={blocker.job_id}>
                        {blocker.job_id}
                      </span>
                      <span
                        className="rounded border px-2 py-0.5 text-[11px] font-medium"
                        style={{ backgroundColor: tone.bg, borderColor: tone.border, color: tone.text }}
                      >
                        {blocker.status}
                      </span>
                      {dispositionLabel && (
                        <span
                          className="rounded border px-2 py-0.5 text-[11px] font-medium"
                          style={{ backgroundColor: tone.bg, borderColor: tone.border, color: tone.text }}
                        >
                          {dispositionLabel}
                        </span>
                      )}
                      {blocker.failure && (
                        <span className="rounded border border-[#64b5f6]/35 bg-[#64b5f6]/10 px-2 py-0.5 text-[11px] font-medium text-[#64b5f6]">
                          {blocker.failure.label}{attempts ? ` · ${attempts}` : ''}
                        </span>
                      )}
                    </div>
                    <div className="mt-1 text-xs text-[#8b949e]">
                      {blocker.host} · {blocker.hardware} · {blocker.model} · tp{blocker.tp} · {blocker.backend}/{blocker.mode}
                    </div>
                  </div>
                  <div className="font-mono text-xs text-[#d29922]">
                    {blocker.present}/{blocker.expected} · missing {blocker.missing_count}
                  </div>
                </div>
                <div className="mt-2 break-words text-xs text-[#c9d1d9]">{blocker.missing}</div>
                {(blocker.coverage_explanation || blocker.failure?.reason || blocker.reason) && (
                  <div className="mt-1 line-clamp-2 break-words text-[11px] leading-4 text-[#6e7681]" title={blocker.coverage_explanation ?? blocker.failure?.reason ?? blocker.reason ?? undefined}>
                    {blocker.coverage_explanation ?? blocker.failure?.reason ?? blocker.reason}
                  </div>
                )}
              </div>
            );
          })}
          {blockersState.blockers.length > visibleBlockers.length && (
            <div className="text-xs text-[#8b949e]">
              Showing {visibleBlockers.length} of {blockersState.blockers.length} blockers, sorted by missing count.
            </div>
          )}
        </div>
      ) : (
        <div className="mt-3 rounded border border-[#238636]/35 bg-[#238636]/10 px-3 py-2 text-sm text-[#7ee787]">
          No terminal coverage blockers in the latest scan.
        </div>
      )}
    </section>
  );
}
