import type { PredictorResults, ServingE2ERow, ServingE2EConcResult } from '../types-profiling';

interface ProfilingPageProps {
  profilingState: { results?: PredictorResults } | null;
  loading: boolean;
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results) return null;
  const concData = results.serving_e2e_conc ?? {};
  const pkData = results.serving_e2e ?? {};
  const hasConcData = Object.keys(concData).length > 0;
  const hasPkData = Object.keys(pkData).length > 0;
  if (!hasConcData && !hasPkData) return null;

  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(1) + '%');
  const fmtMs = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(2));
  const fmtPct = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(1) + '%');

  const concGpus = Object.keys(concData).sort();

  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Serving E2E Predictor — Concurrency Sweep</div>

      {hasConcData && (
        <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
          <div className="flex items-baseline gap-3">
            <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">Per-kernel (ncu) × Concurrency</div>
            <div className="text-[11px] text-[#8b949e]">Little&apos;s Law bs_eff model, validated across C=1–500</div>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-[#8b949e]">Overall MAPE by GPU × profile (supported architectures, all concurrencies)</div>
            <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
              <table className="min-w-full border-collapse text-xs">
                <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                  <th className="px-3 py-2 text-left font-medium">GPU</th>
                  <th className="px-3 py-2 text-left font-medium">Profile</th>
                  <th className="px-3 py-2 text-right font-medium">TPOT MAPE</th>
                  <th className="px-3 py-2 text-right font-medium">E2EL MAPE</th>
                  <th className="px-3 py-2 text-right font-medium">Conc levels</th>
                </tr></thead>
                <tbody>
                  {concGpus.flatMap(g => {
                    const gpuProfiles = concData[g];
                    const profileNames = Object.keys(gpuProfiles).sort();
                    return profileNames.map((p, i) => {
                      const pr = gpuProfiles[p] as ServingE2EConcResult;
                      const tpotColor = (pr.overall.tpot ?? 100) < 15 ? 'text-[#3fb950]' : (pr.overall.tpot ?? 100) < 30 ? 'text-[#ff9800]' : 'text-[#f85149]';
                      const e2elColor = (pr.overall.e2el ?? 100) < 15 ? 'text-[#3fb950]' : (pr.overall.e2el ?? 100) < 30 ? 'text-[#ff9800]' : 'text-[#f85149]';
                      return (
                        <tr key={`${g}-${p}`} className={`border-b border-[#21262d]/50 ${i === 0 ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                          <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{i === 0 ? g : ''}</td>
                          <td className="px-3 py-1.5 text-[#c9d1d9]">{p}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${tpotColor}`}>{fmt(pr.overall.tpot)}</td>
                          <td className={`px-3 py-1.5 text-right font-mono ${e2elColor}`}>{fmt(pr.overall.e2el)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#8b949e]">{pr.per_conc.length}</td>
                        </tr>
                      );
                    });
                  })}
                </tbody>
              </table>
            </div>
            <div className="text-[11px] text-[#8b949e]">
              <span className="text-[#3fb950]">&lt;15%</span> · <span className="text-[#ff9800]">15–30%</span> · <span className="text-[#f85149]">&gt;30%</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs text-[#8b949e]">Per-concurrency breakdown (sample: first GPU)</div>
            {concGpus.slice(0, 1).map(g => {
              const firstProfile = Object.keys(concData[g]).sort()[0];
              if (!firstProfile) return null;
              const pr = concData[g][firstProfile] as ServingE2EConcResult;
              return (
                <div key={g} className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
                  <div className="px-3 py-1.5 text-[11px] text-[#8b949e] border-b border-[#21262d]">{g} — {firstProfile}</div>
                  <table className="min-w-full border-collapse text-xs">
                    <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                      <th className="px-3 py-2 text-right font-medium">Concurrency</th>
                      <th className="px-3 py-2 text-right font-medium">bs_eff</th>
                      <th className="px-3 py-2 text-right font-medium">TTFT MAPE</th>
                      <th className="px-3 py-2 text-right font-medium">TPOT MAPE</th>
                      <th className="px-3 py-2 text-right font-medium">E2EL MAPE</th>
                    </tr></thead>
                    <tbody>
                      {pr.per_conc.map(row => (
                        <tr key={row.conc} className="border-b border-[#21262d]/50">
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{row.conc}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#8b949e]">{row.bs_eff.toFixed(1)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{fmtPct(row.ttft_mape)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(row.tpot_mape)}</td>
                          <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(row.e2el_mape)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {hasPkData && !hasConcData && (
        <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
          <div className="flex items-baseline gap-3">
            <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">Per-kernel (ncu) Predictions</div>
            <div className="text-[11px] text-[#8b949e]">C=1 only (no concurrency sweep data)</div>
          </div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead><tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">GPU</th>
                <th className="px-3 py-2 text-left font-medium">Profile</th>
                <th className="px-3 py-2 text-right font-medium">TTFT MAPE</th>
                <th className="px-3 py-2 text-right font-medium">TPOT MAPE</th>
                <th className="px-3 py-2 text-right font-medium">E2EL MAPE</th>
              </tr></thead>
              <tbody>
                {Object.keys(pkData).sort().flatMap(g =>
                  Object.keys(pkData[g]).sort().map((p, i) => {
                    const pr = pkData[g][p];
                    return (
                      <tr key={`${g}-${p}`} className={`border-b border-[#21262d]/50 ${i === 0 ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                        <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{i === 0 ? g : ''}</td>
                        <td className="px-3 py-1.5 text-[#c9d1d9]">{p}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.ttft)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.tpot)}</td>
                        <td className="px-3 py-1.5 text-right font-mono text-[#3fb950]">{fmt(pr.mape.e2el)}</td>
                      </tr>
                    );
                  }),
                )}
              </tbody>
            </table>
          </div>
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
