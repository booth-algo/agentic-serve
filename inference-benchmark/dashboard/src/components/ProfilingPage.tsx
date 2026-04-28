import { useState } from 'react';
import type { PredictorResults, ServingE2EConcResult, GemmExtrapResult } from '../types-profiling';

interface ProfilingPageProps {
  profilingState: { results?: PredictorResults } | null;
  loading: boolean;
}

type Metric = 'e2el' | 'tpot' | 'ttft';

function mapeColor(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return 'text-[#484f58]';
  if (v < 15) return 'text-[#3fb950]';
  if (v < 30) return 'text-[#ff9800]';
  return 'text-[#f85149]';
}

function mapeBg(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return '';
  if (v < 15) return 'bg-[#3fb950]/10';
  if (v < 30) return 'bg-[#ff9800]/10';
  return 'bg-[#f85149]/10';
}

function fmt(v: number | null | undefined): string {
  if (v === null || v === undefined || isNaN(v)) return '—';
  return v.toFixed(1) + '%';
}

function ConcHeatmap({ concData }: { concData: Record<string, Record<string, ServingE2EConcResult>> }) {
  const [metric, setMetric] = useState<Metric>('e2el');

  const allConcs = new Set<number>();
  const gpus = Object.keys(concData).sort();
  for (const g of gpus) {
    for (const p of Object.keys(concData[g])) {
      const pr = concData[g][p] as ServingE2EConcResult;
      for (const row of pr.per_conc) allConcs.add(row.conc);
    }
  }
  const concs = Array.from(allConcs).sort((a, b) => a - b);

  const ALL_PROFILES = [
    'chat-short', 'chat-medium', 'chat-long',
    'chat-multiturn-short', 'chat-multiturn-medium', 'chat-multiturn-long',
    'coding-agent', 'prefill-heavy', 'decode-heavy',
    'terminalbench-multiturn-short', 'terminalbench-multiturn-medium',
    'swebench-multiturn-short', 'swebench-multiturn-medium',
    'osworld-multiturn-short', 'osworld-multiturn-medium',
  ];

  const rows: { gpu: string; profile: string; pr: ServingE2EConcResult | null; isFirstInGpu: boolean }[] = [];
  for (const g of gpus) {
    ALL_PROFILES.forEach((p, i) => {
      const pr = concData[g]?.[p] as ServingE2EConcResult | undefined;
      rows.push({ gpu: g, profile: p, pr: pr ?? null, isFirstInGpu: i === 0 });
    });
  }

  const getValue = (pr: ServingE2EConcResult | null, conc: number): number | null => {
    if (!pr) return null;
    const row = pr.per_conc.find(r => r.conc === conc);
    if (!row) return null;
    if (metric === 'e2el') return row.e2el_mape;
    if (metric === 'tpot') return row.tpot_mape;
    return row.ttft_mape;
  };

  const getAvg = (pr: ServingE2EConcResult | null): number | null => {
    if (!pr) return null;
    const vals = pr.per_conc.map(r => metric === 'e2el' ? r.e2el_mape : metric === 'tpot' ? r.tpot_mape : r.ttft_mape).filter(v => v !== null && !isNaN(v));
    if (vals.length === 0) return null;
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs text-[#8b949e]">Metric:</span>
        {(['e2el', 'tpot', 'ttft'] as Metric[]).map(m => (
          <button
            key={m}
            onClick={() => setMetric(m)}
            className={`rounded px-2 py-0.5 text-xs font-mono transition-colors ${
              metric === m
                ? 'bg-[#58a6ff]/20 text-[#58a6ff] border border-[#58a6ff]/40'
                : 'text-[#8b949e] border border-[#30363d] hover:text-[#c9d1d9]'
            }`}
          >
            {m.toUpperCase()}
          </button>
        ))}
        <span className="ml-4 text-[11px] text-[#484f58]">
          <span className="text-[#3fb950]">&lt;15%</span>{' · '}
          <span className="text-[#ff9800]">15–30%</span>{' · '}
          <span className="text-[#f85149]">&gt;30%</span>
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
        <table className="min-w-full border-collapse text-xs">
          <thead>
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-2 py-2 text-left font-medium sticky left-0 bg-[#161b22] z-10">GPU</th>
              <th className="px-2 py-2 text-left font-medium sticky left-[60px] bg-[#161b22] z-10">Profile</th>
              <th className="px-2 py-2 text-right font-medium">Avg</th>
              {concs.map(c => (
                <th key={c} className="px-2 py-2 text-right font-medium min-w-[52px]">C={c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ gpu, profile, pr, isFirstInGpu }) => {
              const avg = getAvg(pr);
              return (
                <tr key={`${gpu}-${profile}`} className={`border-b border-[#21262d]/50 ${isFirstInGpu ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                  <td className="px-2 py-1 font-mono text-[#c9d1d9] sticky left-0 bg-[#161b22] z-10 whitespace-nowrap">
                    {isFirstInGpu ? gpu : ''}
                  </td>
                  <td className="px-2 py-1 text-[#c9d1d9] sticky left-[60px] bg-[#161b22] z-10 whitespace-nowrap">{profile}</td>
                  <td className={`px-2 py-1 text-right font-mono font-semibold ${mapeColor(avg)} ${mapeBg(avg)}`}>
                    {fmt(avg)}
                  </td>
                  {concs.map(c => {
                    const v = getValue(pr, c);
                    return (
                      <td key={c} className={`px-2 py-1 text-right font-mono ${mapeColor(v)} ${mapeBg(v)}`}>
                        {fmt(v)}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] text-[#484f58]">
        Steady-state batch size model, validated across C=1–500. Supported architectures only (excludes MoE, hybrid attention).
      </div>
    </div>
  );
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results) return null;
  const concData = results.serving_e2e_conc ?? {};
  const hasConcData = Object.keys(concData).length > 0;

  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Serving E2E Predictor — Concurrency Sweep</div>

      {hasConcData && (
        <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
          <div className="flex items-baseline gap-3">
            <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">Per-kernel (ncu) × Concurrency</div>
          </div>
          <ConcHeatmap concData={concData} />
        </div>
      )}

      {results.gemm_extrapolation && Object.keys(results.gemm_extrapolation).length > 0 && (
        <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
          <div className="flex items-baseline gap-3">
            <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">GEMM Shape Extrapolation</div>
            <div className="text-[11px] text-[#8b949e]">50 random unseen (M,N,K) shapes — proves XGBoost generalizes, not memorizes</div>
          </div>
          {Object.keys(results.gemm_extrapolation).sort().map(gpu => {
            const ex = results.gemm_extrapolation![gpu] as GemmExtrapResult;
            return (
              <div key={gpu} className="space-y-2">
                <div className="flex gap-6 text-xs text-[#8b949e]">
                  <span>{gpu}: {ex.n_shapes} shapes</span>
                  <span>MAPE: <span className="font-mono text-[#3fb950]">{ex.mape}%</span></span>
                  <span>Median: <span className="font-mono text-[#3fb950]">{ex.median_err}%</span></span>
                  <span>Within 20%: <span className="font-mono text-[#3fb950]">{ex.within_20pct}%</span></span>
                </div>
                <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22] max-h-64 overflow-y-auto">
                  <table className="min-w-full border-collapse text-xs">
                    <thead className="sticky top-0 bg-[#161b22]"><tr className="border-b border-[#21262d] text-[#8b949e]">
                      <th className="px-3 py-2 text-right font-medium">M</th>
                      <th className="px-3 py-2 text-right font-medium">N</th>
                      <th className="px-3 py-2 text-right font-medium">K</th>
                      <th className="px-3 py-2 text-right font-medium">Predicted (ms)</th>
                      <th className="px-3 py-2 text-right font-medium">Measured (ms)</th>
                      <th className="px-3 py-2 text-right font-medium">Error</th>
                    </tr></thead>
                    <tbody>
                      {ex.rows.map((r, i) => {
                        const errColor = r.err_pct < 15 ? 'text-[#3fb950]' : r.err_pct < 30 ? 'text-[#ff9800]' : 'text-[#f85149]';
                        return (
                          <tr key={i} className="border-b border-[#21262d]/50">
                            <td className="px-3 py-1 text-right font-mono text-[#c9d1d9]">{r.M}</td>
                            <td className="px-3 py-1 text-right font-mono text-[#c9d1d9]">{r.N}</td>
                            <td className="px-3 py-1 text-right font-mono text-[#c9d1d9]">{r.K}</td>
                            <td className="px-3 py-1 text-right font-mono text-[#c9d1d9]">{r.pred_ms.toFixed(4)}</td>
                            <td className="px-3 py-1 text-right font-mono text-[#c9d1d9]">{r.meas_ms.toFixed(4)}</td>
                            <td className={`px-3 py-1 text-right font-mono ${errColor}`}>{r.err_pct.toFixed(1)}%</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      )}
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

  if (!profilingState?.results) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#21262d] bg-[#161b22]">
        <div className="text-center text-sm text-[#8b949e]">
          <div className="mb-2 text-base font-semibold text-[#c9d1d9]">No predictor data available</div>
          <div>Run <code className="rounded bg-[#21262d] px-1">python scripts/publish_profiling_state.py --no-upload</code> to generate profiling-state.json</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <PredictorResultsSection results={profilingState.results} />
    </div>
  );
}
