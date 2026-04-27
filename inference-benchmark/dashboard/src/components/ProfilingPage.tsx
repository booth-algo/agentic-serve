import type { PredictorResults, ServingE2ERow, ServingE2EProfileResult } from '../types-profiling';

interface ProfilingPageProps {
  profilingState: { results?: PredictorResults } | null;
  loading: boolean;
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results?.serving_e2e) return null;
  const pkData = results.serving_e2e;
  const allGpus = Object.keys(pkData).sort();
  if (allGpus.length === 0) return null;

  const allProfiles = new Set<string>();
  for (const g of allGpus) {
    for (const p of Object.keys(pkData[g] ?? {})) allProfiles.add(p);
  }
  const profiles = [...allProfiles].sort();

  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '\u2014' : v.toFixed(2) + '%');
  const fmtMs = (v: number | undefined | null) => (v === undefined || v === null ? '\u2014' : v.toFixed(2));
  const fmtPct = (v: number | undefined | null) => (v === undefined || v === null ? '\u2014' : v.toFixed(1) + '%');

  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Serving E2E Predictor MAPE</div>

      <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
        <div className="flex items-baseline gap-3">
          <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">Per-kernel (ncu) Predictions</div>
          <div className="text-[11px] text-[#8b949e]">ISL/OSL-aware TTFT + TPOT + E2EL vs real vLLM benchmarks, bs=1</div>
        </div>

        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">MAPE by GPU \u00d7 profile (supported architectures only)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                <th className="px-3 py-2 text-left font-medium">Profile</th>
                <th className="px-3 py-2 text-right font-medium">TTFT MAPE</th>
                <th className="px-3 py-2 text-right font-medium">TPOT MAPE</th>
                <th className="px-3 py-2 text-right font-medium">E2EL MAPE</th>
                <th className="px-3 py-2 text-right font-medium">n rows</th>
              </tr></thead>
              <tbody>
                {allGpus.flatMap(g => {
                  const gpuProfiles = pkData[g];
                  const profileNames = Object.keys(gpuProfiles).sort();
                  return profileNames.map((p, i) => {
                    const pr = gpuProfiles[p];
                    const n = pr.rows.filter((r: ServingE2ERow) => r.arch === 'supported').length;
                    return (
                      <tr key={`${g}-${p}`} className={`border-b border-[#21262d]/50 ${i === 0 ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                        <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{i === 0 ? g : ''}</td>
                        <td className="px-3 py-1.5 text-[#c9d1d9]">{p}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.ttft)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.tpot)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.e2el)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#8b949e]">{n}</td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-row detail (all architectures)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                <th className="px-3 py-2 text-left font-medium">Model</th>
                <th className="px-3 py-2 text-left font-medium">arch</th>
                <th className="px-3 py-2 text-left font-medium">Profile</th>
                <th className="px-3 py-2 text-right font-medium">ISL</th>
                <th className="px-3 py-2 text-right font-medium">OSL</th>
                <th className="px-3 py-2 text-right font-medium">pred TTFT</th>
                <th className="px-3 py-2 text-right font-medium">meas TTFT</th>
                <th className="px-3 py-2 text-right font-medium">TTFT err</th>
                <th className="px-3 py-2 text-right font-medium">pred TPOT</th>
                <th className="px-3 py-2 text-right font-medium">meas TPOT</th>
                <th className="px-3 py-2 text-right font-medium">TPOT err</th>
                <th className="px-3 py-2 text-right font-medium">pred E2EL</th>
                <th className="px-3 py-2 text-right font-medium">meas E2EL</th>
                <th className="px-3 py-2 text-right font-medium">E2EL err</th>
              </tr></thead>
              <tbody>
                {allGpus.flatMap(g =>
                  Object.keys(pkData[g]).sort().flatMap(profile =>
                    pkData[g][profile].rows.map((r: ServingE2ERow, i: number) => {
                      const archColor = r.arch === 'supported' ? 'text-[#3fb950]' : 'text-[#8b949e]';
                      return (
                        <tr key={`${g}-${profile}-${r.model}-${i}`} className="border-b border-[#21262d]/50">
                          <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{g}</td>
                          <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{r.model}</td>
                          <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.arch}</td>
                          <td className="px-3 py-1.5 text-[#c9d1d9]">{profile}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.isl}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.osl}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.pred_ttft_ms.toFixed(2)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.meas_ttft_ms.toFixed(2)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${archColor}`}>{r.ttft_err_pct.toFixed(1)}%</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{fmtMs(r.pred_tpot_ms)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{fmtMs(r.meas_tpot_ms)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${archColor}`}>{fmtPct(r.tpot_err_pct)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.pred_e2el_ms.toFixed(2)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{fmtMs(r.meas_e2el_ms)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${archColor}`}>{fmtPct(r.e2el_err_pct)}</td>
                        </tr>
                      );
                    }),
                  ),
                )}
              </tbody>
            </table>
          </div>
        </div>
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
