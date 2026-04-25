import type { ProfilingState, ProfilingStatus, PredictorResults } from '../types-profiling';

interface ProfilingPageProps {
  profilingState: ProfilingState | null;
  loading: boolean;
}

type ColKey = 'per_kernel_prefill' | 'per_kernel_roofline' | 'per_op_cuda_events' | 'per_op_trained_pkl';

const COLUMNS: Array<{ key: ColKey; label: string; sub: string }> = [
  { key: 'per_kernel_prefill',  label: 'Per-kernel',   sub: 'prefill ncu' },
  { key: 'per_kernel_roofline', label: 'Per-kernel',   sub: 'roofline ncu' },
  { key: 'per_op_cuda_events',  label: 'Per-op',       sub: 'cuda events' },
  { key: 'per_op_trained_pkl',  label: 'Per-op',       sub: 'trained pkl' },
];

const STATUS_COLOR: Record<ProfilingStatus['status'], string> = {
  done:                'bg-[#3fb950]/15 text-[#3fb950] border-[#3fb950]/40',
  missing:             'bg-[#ff9800]/10 text-[#ff9800] border-[#ff9800]/40',
  infeasible:          'bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40',
  partial:             'bg-[#8b5cf6]/15 text-[#8b5cf6] border-[#8b5cf6]/40',
  pending_infra_access:'bg-[#8b949e]/15 text-[#8b949e] border-[#8b949e]/40',
};

const STATUS_LABEL: Record<ProfilingStatus['status'], string> = {
  done:                'DONE',
  missing:             'TODO',
  infeasible:          'N/A',
  partial:             'PART',
  pending_infra_access:'PEND',
};

function StatusBadge({ s }: { s: ProfilingStatus }) {
  const cls = STATUS_COLOR[s.status];
  const label = STATUS_LABEL[s.status];
  const tooltip = [
    s.reason && `reason: ${s.reason}`,
    s.rows !== undefined && `rows: ${s.rows}`,
    s.version && `version: ${s.version}`,
  ].filter(Boolean).join(' · ');
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}
      title={tooltip || undefined}
    >
      {label}
    </span>
  );
}

function SummaryCell({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-3">
      <div className="text-xs uppercase tracking-wide text-[#8b949e]">{label}</div>
      <div className="mt-1 font-mono text-xl font-semibold" style={{ color }}>{value}</div>
      <div className="text-[11px] text-[#8b949e]">{sub}</div>
    </div>
  );
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results || (!results.per_kernel && !results.per_op && !results.wallclock)) return null;
  const pkGpus = Object.keys(results.per_kernel ?? {}).sort();
  const poGpus = Object.keys(results.per_op ?? {}).sort();
  const wcGpus = Object.keys(results.wallclock ?? {}).sort();
  const allFamilies: string[] = [];
  for (const g of pkGpus) {
    const fams = Object.keys(results.per_kernel![g]?.heldout_mape_per_family ?? {});
    for (const f of fams) if (!allFamilies.includes(f)) allFamilies.push(f);
  }
  const heldModels: string[] = [];
  for (const g of pkGpus) {
    const ms = Object.keys(results.per_kernel![g]?.aggregate_err_per_model ?? {});
    for (const m of ms) if (!heldModels.includes(m)) heldModels.push(m);
  }
  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(2) + '%');
  return (
    <div className="mt-6 space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Predictor MAPE</div>

      {pkGpus.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-kernel held-out MAPE (per family)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                {allFamilies.map(f => <th key={f} className="px-3 py-2 text-right font-medium">{f}</th>)}
              </tr></thead>
              <tbody>
                {pkGpus.map(g => (
                  <tr key={g} className="border-b border-[#21262d]/50">
                    <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{g}</td>
                    {allFamilies.map(f => (
                      <td key={f} className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">
                        {fmt(results.per_kernel![g]?.heldout_mape_per_family?.[f])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {pkGpus.length > 0 && heldModels.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-kernel TTFT error on prefill_seq128_bs1 (sum of kernel preds)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                {heldModels.map(m => <th key={m} className="px-3 py-2 text-right font-medium">{m}</th>)}
              </tr></thead>
              <tbody>
                {pkGpus.map(g => (
                  <tr key={g} className="border-b border-[#21262d]/50">
                    <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{g}</td>
                    {heldModels.map(m => (
                      <td key={m} className="px-3 py-1.5 text-right font-mono text-[#3fb950]">
                        {fmt(results.per_kernel![g]?.aggregate_err_per_model?.[m])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {poGpus.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-op held-out MAPE (strict: all 70B-class held out, no arch-anchor)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                <th className="px-3 py-2 text-right font-medium">MAPE</th>
                <th className="px-3 py-2 text-left font-medium">Pool</th>
                <th className="px-3 py-2 text-left font-medium">Held-out</th>
              </tr></thead>
              <tbody>
                {poGpus.map(g => {
                  const r = results.per_op![g];
                  return (
                    <tr key={g} className="border-b border-[#21262d]/50">
                      <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{g}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-[#ff9800]">{fmt(r.heldout_mape)}</td>
                      <td className="px-3 py-1.5 text-[#8b949e]">{(r.pool_models ?? []).join(', ')}</td>
                      <td className="px-3 py-1.5 text-[#8b949e]">{(r.heldout_models ?? []).join(', ') || '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {wcGpus.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">
            Per-kernel TTFT wall-clock (vs real vLLM 0.19 TTFT, fixed-seq128, bs=1)
          </div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                <th className="px-3 py-2 text-left font-medium">Model</th>
                <th className="px-3 py-2 text-left font-medium">arch</th>
                <th className="px-3 py-2 text-right font-medium">pred (ms)</th>
                <th className="px-3 py-2 text-right font-medium">measured (ms)</th>
                <th className="px-3 py-2 text-right font-medium">MAPE</th>
                <th className="px-3 py-2 text-right font-medium">ncu Σ (ms)</th>
                <th className="px-3 py-2 text-right font-medium">overhead %</th>
              </tr></thead>
              <tbody>
                {wcGpus.flatMap(g => {
                  const wc = results.wallclock![g];
                  return (wc.rows ?? []).map((r, i) => {
                    const archColor = r.arch === 'supported' ? 'text-[#3fb950]' : 'text-[#8b949e]';
                    const errColor = r.arch === 'supported' ? 'text-[#3fb950]' : 'text-[#8b949e]';
                    return (
                      <tr key={`${g}-${r.model}-${i}`} className="border-b border-[#21262d]/50">
                        <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{g}</td>
                        <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{r.model}</td>
                        <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.arch}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.predicted_ms.toFixed(2)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.measured_ms.toFixed(2)}</td>
                        <td className={`px-3 py-1.5 text-right font-mono ${errColor}`}>{r.abs_err_pct.toFixed(2)}%</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#8b949e]">{r.ncu_sum_ms !== undefined ? r.ncu_sum_ms.toFixed(2) : '—'}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#8b949e]">{r.overhead_pct !== undefined ? `${r.overhead_pct.toFixed(1)}%` : '—'}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-[#8b949e]">
            Supported MAPE (dense, full-attn, non-MoE): {wcGpus.map(g => {
              const m = results.wallclock![g]?.supported_mape;
              return `${g}: ${m !== undefined ? m.toFixed(2) + '%' : '—'}`;
            }).join(' · ')}
          </div>
        </div>
      )}

      <div className="text-[11px] text-[#8b949e]">
        See <code className="rounded bg-[#21262d] px-1">.claude/paper/per_op_vs_per_kernel_tradeoff.md</code> for cross-family and cross-scale analysis.
      </div>
    </div>
  );
}

export function ProfilingPage({ profilingState, loading }: ProfilingPageProps) {
  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-[#8b949e]">Loading profiling data...</div>
      </div>
    );
  }

  if (!profilingState) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#21262d] bg-[#161b22]">
        <div className="text-center text-sm text-[#8b949e]">
          <div className="mb-2 text-base font-semibold text-[#c9d1d9]">No profiling data available</div>
          <div>Run <code className="rounded bg-[#21262d] px-1">python scripts/publish_profiling_state.py --no-upload</code> to generate profiling-state.json</div>
        </div>
      </div>
    );
  }

  const allStatuses = profilingState.cells.flatMap((c) =>
    COLUMNS.map((col) => c[col.key].status),
  );
  const countBy = (s: ProfilingStatus['status']) => allStatuses.filter((x) => x === s).length;

  const total = allStatuses.length;
  const doneCount = countBy('done');
  const pct = total > 0 ? ((doneCount / total) * 100).toFixed(1) : '0.0';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${doneCount}/${total} cells`} color="#00bcd4" />
        <SummaryCell label="Done" value={`${countBy('done')}`} sub="traces collected" color="#3fb950" />
        <SummaryCell label="Missing" value={`${countBy('missing')}`} sub="not yet run" color="#ff9800" />
        <SummaryCell label="Partial" value={`${countBy('partial')}`} sub="incomplete traces" color="#8b5cf6" />
        <SummaryCell label="Infeasible" value={`${countBy('infeasible')}`} sub="hw/model constraint" color="#64b5f6" />
        <SummaryCell label="Pending Infra" value={`${countBy('pending_infra_access')}`} sub="awaiting access" color="#8b949e" />
      </div>

      <div className="flex flex-wrap items-center gap-4 rounded-md border border-[#21262d] bg-[#161b22] px-4 py-2 text-xs text-[#8b949e]">
        {(Object.keys(STATUS_COLOR) as Array<ProfilingStatus['status']>).map((s) => (
          <span key={s} className="flex items-center gap-1.5">
            <span className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${STATUS_COLOR[s]}`}>
              {STATUS_LABEL[s]}
            </span>
            {s.replace(/_/g, ' ')}
          </span>
        ))}
        <span className="ml-auto font-mono">
          {profilingState.cells.length} rows · {profilingState.gpus.length} GPUs · {profilingState.models.length} models
          <span className="ml-2">· generated {new Date(profilingState.generated_at).toLocaleString()}</span>
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
        <table className="min-w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-3 py-2 text-left font-medium">GPU</th>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              {COLUMNS.map((col) => (
                <th key={col.key} className="px-3 py-2 text-center font-medium">
                  <div>{col.label}</div>
                  <div className="font-normal text-[10px] text-[#8b949e]">{col.sub}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profilingState.cells.map((cell, i) => {
              const prev = profilingState.cells[i - 1];
              const gpuChange = !prev || cell.gpu !== prev.gpu;
              return (
                <tr
                  key={`${cell.gpu}|${cell.model}`}
                  className={`border-b border-[#21262d]/50 ${gpuChange ? 'border-t-2 border-t-[#30363d]' : ''}`}
                >
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[#c9d1d9]">
                    {gpuChange ? cell.gpu : ''}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">{cell.model}</td>
                  {COLUMNS.map((col) => (
                    <td key={col.key} className="px-3 py-1.5 text-center">
                      <StatusBadge s={cell[col.key]} />
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <PredictorResultsSection results={profilingState.results} />

      <div className="space-y-1 text-xs text-[#8b949e]">
        <p>
          Profiling coverage sourced from{' '}
          <code className="rounded bg-[#21262d] px-1">llm_predict/training/per_kernel/profiling_manifest.yaml</code>{' '}
          via <code className="rounded bg-[#21262d] px-1">scripts/publish_profiling_state.py</code>.
        </p>
        <p>
          <span className="text-[#3fb950]">Done</span> = ncu/cuda-event traces collected and ingested.{' '}
          <span className="text-[#8b5cf6]">Partial</span> = some rows present but trace incomplete.{' '}
          <span className="text-[#64b5f6]">Infeasible</span> = GPU/model combination cannot be profiled (e.g. VRAM).{' '}
          <span className="text-[#8b949e]">Pending Infra</span> = waiting for ncu/infra access.
        </p>
      </div>
    </div>
  );
}
