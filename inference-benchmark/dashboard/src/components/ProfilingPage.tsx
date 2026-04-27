import type { PredictorResults, ServingE2ERow, ServingE2EProfileResult } from '../types-profiling';

interface ProfilingPageProps {
  profilingState: { results?: PredictorResults } | null;
  loading: boolean;
}

function SubPanelHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div className="flex items-baseline gap-3">
      <div className="text-xs font-semibold text-[#58a6ff] uppercase tracking-wider">{title}</div>
      <div className="text-[11px] text-[#8b949e]">{subtitle}</div>
    </div>
  );
}

function AblationSection({ results }: { results: PredictorResults }) {
  const pkData = results.serving_e2e ?? {};
  const poData = results.serving_e2e_perop ?? {};
  const allGpus = [...new Set([...Object.keys(pkData), ...Object.keys(poData)])].sort();
  if (allGpus.length === 0) return null;

  const allProfiles = new Set<string>();
  for (const g of allGpus) {
    for (const p of Object.keys(pkData[g] ?? {})) allProfiles.add(p);
    for (const p of Object.keys(poData[g] ?? {})) allProfiles.add(p);
  }
  const profiles = [...allProfiles].sort();

  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(2) + '%');
  const fmtMs = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(2));
  const fmtPct = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(1) + '%');

  type AblationRow = {
    gpu: string; model: string; arch: string; profile: string;
    isl: number; osl: number;
    meas_ttft: number; meas_tpot?: number; meas_e2el?: number;
    pk_ttft_err?: number; pk_tpot_err?: number; pk_e2el_err?: number;
    po_ttft_err?: number; po_tpot_err?: number; po_e2el_err?: number;
  };

  const ablationRows: AblationRow[] = [];
  for (const g of allGpus) {
    for (const p of profiles) {
      const pkProfile = pkData[g]?.[p];
      const poProfile = poData[g]?.[p];
      const pkRows = pkProfile?.rows ?? [];
      const poRows = poProfile?.rows ?? [];
      const poByModel = new Map(poRows.map((r: ServingE2ERow) => [r.model, r]));

      for (const pkr of pkRows) {
        const por = poByModel.get(pkr.model);
        ablationRows.push({
          gpu: g, model: pkr.model, arch: pkr.arch, profile: p,
          isl: pkr.isl, osl: pkr.osl,
          meas_ttft: pkr.meas_ttft_ms,
          meas_tpot: pkr.meas_tpot_ms,
          meas_e2el: pkr.meas_e2el_ms,
          pk_ttft_err: pkr.ttft_err_pct,
          pk_tpot_err: pkr.tpot_err_pct,
          pk_e2el_err: pkr.e2el_err_pct,
          po_ttft_err: por?.ttft_err_pct,
          po_tpot_err: por?.tpot_err_pct,
          po_e2el_err: por?.e2el_err_pct,
        });
        poByModel.delete(pkr.model);
      }
      for (const [, por] of poByModel) {
        ablationRows.push({
          gpu: g, model: por.model, arch: por.arch, profile: p,
          isl: por.isl, osl: por.osl,
          meas_ttft: por.meas_ttft_ms,
          meas_tpot: por.meas_tpot_ms,
          meas_e2el: por.meas_e2el_ms,
          po_ttft_err: por.ttft_err_pct,
          po_tpot_err: por.tpot_err_pct,
          po_e2el_err: por.e2el_err_pct,
        });
      }
    }
  }

  return (
    <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
      <SubPanelHeader
        title="Serving E2E — Ablation: per-kernel (ncu) vs per-op (torch.profiler)"
        subtitle="same model, same benchmark, same integration — different measurement source"
      />

      <div className="space-y-2">
        <div className="text-xs text-[#8b949e]">MAPE by GPU × profile (supported architectures only)</div>
        <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
          <table className="min-w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium" rowSpan={2}>GPU</th>
                <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Profile</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>TTFT MAPE</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>TPOT MAPE</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>E2EL MAPE</th>
                <th className="px-3 py-2 text-right font-medium border-l border-[#21262d]" rowSpan={2}>n</th>
              </tr>
              <tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
              </tr>
            </thead>
            <tbody>
              {allGpus.flatMap(g => {
                return profiles.map((p, i) => {
                  const pkP = pkData[g]?.[p] as ServingE2EProfileResult | undefined;
                  const poP = poData[g]?.[p] as ServingE2EProfileResult | undefined;
                  const n = (pkP?.rows ?? poP?.rows ?? []).filter((r: ServingE2ERow) => r.arch === 'supported').length;
                  if (!pkP && !poP) return null;
                  return (
                    <tr key={`${g}-${p}`} className={`border-b border-[#21262d]/50 ${i === 0 ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                      <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{i === 0 ? g : ''}</td>
                      <td className="px-3 py-1.5 text-[#c9d1d9]">{p}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#8b949e] border-l border-[#21262d]">{n}</td>
                    </tr>
                  );
                }).filter(Boolean);
              })}
            </tbody>
          </table>
        </div>
        <div className="flex gap-4 text-[11px] text-[#8b949e]">
          <span><span className="font-mono text-[#3fb950]">green</span> = per-kernel (ncu)</span>
          <span><span className="font-mono text-[#f85149]">red</span> = per-op (torch.profiler)</span>
        </div>
      </div>

      {ablationRows.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-row ablation detail (all architectures)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#21262d] text-[#8b949e]">
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>GPU</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Model</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>arch</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Profile</th>
                  <th className="px-3 py-2 text-right font-medium" rowSpan={2}>ISL</th>
                  <th className="px-3 py-2 text-right font-medium" rowSpan={2}>OSL</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>TTFT (ms)</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>TPOT (ms)</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>E2EL (ms)</th>
                </tr>
                <tr className="border-b border-[#21262d] text-[#8b949e]">
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                </tr>
              </thead>
              <tbody>
                {ablationRows.map((r, i) => {
                  const archColor = r.arch === 'supported' ? 'text-[#c9d1d9]' : 'text-[#8b949e]';
                  return (
                    <tr key={`${r.gpu}-${r.profile}-${r.model}-${i}`} className="border-b border-[#21262d]/50">
                      <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{r.gpu}</td>
                      <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.model}</td>
                      <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.arch}</td>
                      <td className="px-3 py-1.5 text-[#c9d1d9]">{r.profile}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.isl}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.osl}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_ttft_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_ttft_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_tpot_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_tpot_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_e2el_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_e2el_err)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-[#8b949e]">
            <span className="text-[#3fb950]">PK</span> = per-kernel (ncu-based XGBoost) · <span className="text-[#f85149]">PO</span> = per-op (torch.profiler XGBoost) · same 8-point trapezoidal decode integration
          </div>
        </div>
      )}
    </div>
  );
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results || (!results.serving_e2e && !results.serving_e2e_perop)) return null;
  const hasServing = Object.keys(results.serving_e2e ?? {}).length > 0 || Object.keys(results.serving_e2e_perop ?? {}).length > 0;
  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(2) + '%');
  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Predictor MAPE</div>

      {hasServing && <AblationSection results={results} />}

      <div className="text-[11px] text-[#8b949e]">{subtitle}</div>
    </div>
  );
}

function AblationSection({ results }: { results: PredictorResults }) {
  const pkData = results.serving_e2e ?? {};
  const poData = results.serving_e2e_perop ?? {};
  const allGpus = [...new Set([...Object.keys(pkData), ...Object.keys(poData)])].sort();
  if (allGpus.length === 0) return null;

  const allProfiles = new Set<string>();
  for (const g of allGpus) {
    for (const p of Object.keys(pkData[g] ?? {})) allProfiles.add(p);
    for (const p of Object.keys(poData[g] ?? {})) allProfiles.add(p);
  }
  const profiles = [...allProfiles].sort();

  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(2) + '%');
  const fmtMs = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(2));
  const fmtPct = (v: number | undefined | null) => (v === undefined || v === null ? '—' : v.toFixed(1) + '%');

  type AblationRow = {
    gpu: string; model: string; arch: string; profile: string;
    isl: number; osl: number;
    meas_ttft: number; meas_tpot?: number; meas_e2el?: number;
    pk_ttft_err?: number; pk_tpot_err?: number; pk_e2el_err?: number;
    po_ttft_err?: number; po_tpot_err?: number; po_e2el_err?: number;
  };

  const ablationRows: AblationRow[] = [];
  for (const g of allGpus) {
    for (const p of profiles) {
      const pkProfile = pkData[g]?.[p];
      const poProfile = poData[g]?.[p];
      const pkRows = pkProfile?.rows ?? [];
      const poRows = poProfile?.rows ?? [];
      const poByModel = new Map(poRows.map((r: ServingE2ERow) => [r.model, r]));

      for (const pkr of pkRows) {
        const por = poByModel.get(pkr.model);
        ablationRows.push({
          gpu: g, model: pkr.model, arch: pkr.arch, profile: p,
          isl: pkr.isl, osl: pkr.osl,
          meas_ttft: pkr.meas_ttft_ms,
          meas_tpot: pkr.meas_tpot_ms,
          meas_e2el: pkr.meas_e2el_ms,
          pk_ttft_err: pkr.ttft_err_pct,
          pk_tpot_err: pkr.tpot_err_pct,
          pk_e2el_err: pkr.e2el_err_pct,
          po_ttft_err: por?.ttft_err_pct,
          po_tpot_err: por?.tpot_err_pct,
          po_e2el_err: por?.e2el_err_pct,
        });
        poByModel.delete(pkr.model);
      }
      for (const [, por] of poByModel) {
        ablationRows.push({
          gpu: g, model: por.model, arch: por.arch, profile: p,
          isl: por.isl, osl: por.osl,
          meas_ttft: por.meas_ttft_ms,
          meas_tpot: por.meas_tpot_ms,
          meas_e2el: por.meas_e2el_ms,
          po_ttft_err: por.ttft_err_pct,
          po_tpot_err: por.tpot_err_pct,
          po_e2el_err: por.e2el_err_pct,
        });
      }
    }
  }

  return (
    <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
      <SubPanelHeader
        title="Serving E2E — Ablation: per-kernel (ncu) vs per-op (torch.profiler)"
        subtitle="same model, same benchmark, same integration — different measurement source"
      />

      <div className="space-y-2">
        <div className="text-xs text-[#8b949e]">MAPE by GPU × profile (supported architectures only)</div>
        <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
          <table className="min-w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium" rowSpan={2}>GPU</th>
                <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Profile</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>TTFT MAPE</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>TPOT MAPE</th>
                <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={2}>E2EL MAPE</th>
                <th className="px-3 py-2 text-right font-medium border-l border-[#21262d]" rowSpan={2}>n</th>
              </tr>
              <tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">per-kernel</th>
                <th className="px-2 py-1 text-right text-[10px] font-normal">per-op</th>
              </tr>
            </thead>
            <tbody>
              {allGpus.flatMap(g => {
                return profiles.map((p, i) => {
                  const pkP = pkData[g]?.[p] as ServingE2EProfileResult | undefined;
                  const poP = poData[g]?.[p] as ServingE2EProfileResult | undefined;
                  const n = (pkP?.rows ?? poP?.rows ?? []).filter((r: ServingE2ERow) => r.arch === 'supported').length;
                  if (!pkP && !poP) return null;
                  return (
                    <tr key={`${g}-${p}`} className={`border-b border-[#21262d]/50 ${i === 0 ? 'border-t-2 border-t-[#30363d]' : ''}`}>
                      <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{i === 0 ? g : ''}</td>
                      <td className="px-3 py-1.5 text-[#c9d1d9]">{p}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950] border-l border-[#21262d]">{fmt(pkP?.mape.e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmt(poP?.mape.e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#8b949e] border-l border-[#21262d]">{n}</td>
                    </tr>
                  );
                }).filter(Boolean);
              })}
            </tbody>
          </table>
        </div>
        <div className="flex gap-4 text-[11px] text-[#8b949e]">
          <span><span className="font-mono text-[#3fb950]">green</span> = per-kernel (ncu)</span>
          <span><span className="font-mono text-[#f85149]">red</span> = per-op (torch.profiler)</span>
        </div>
      </div>

      {ablationRows.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-[#8b949e]">Per-row ablation detail (all architectures)</div>
          <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
            <table className="min-w-full border-collapse text-xs">
              <thead>
                <tr className="border-b border-[#21262d] text-[#8b949e]">
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>GPU</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Model</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>arch</th>
                  <th className="px-3 py-2 text-left font-medium" rowSpan={2}>Profile</th>
                  <th className="px-3 py-2 text-right font-medium" rowSpan={2}>ISL</th>
                  <th className="px-3 py-2 text-right font-medium" rowSpan={2}>OSL</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>TTFT (ms)</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>TPOT (ms)</th>
                  <th className="px-3 py-2 text-center font-medium border-l border-[#21262d]" colSpan={3}>E2EL (ms)</th>
                </tr>
                <tr className="border-b border-[#21262d] text-[#8b949e]">
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal border-l border-[#21262d]">measured</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#3fb950]">PK err</th>
                  <th className="px-2 py-1 text-right text-[10px] font-normal text-[#f85149]">PO err</th>
                </tr>
              </thead>
              <tbody>
                {ablationRows.map((r, i) => {
                  const archColor = r.arch === 'supported' ? 'text-[#c9d1d9]' : 'text-[#8b949e]';
                  return (
                    <tr key={`${r.gpu}-${r.profile}-${r.model}-${i}`} className="border-b border-[#21262d]/50">
                      <td className="px-3 py-1.5 font-mono text-[#c9d1d9]">{r.gpu}</td>
                      <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.model}</td>
                      <td className={`px-3 py-1.5 font-mono ${archColor}`}>{r.arch}</td>
                      <td className="px-3 py-1.5 text-[#c9d1d9]">{r.profile}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.isl}</td>
                      <td className="px-3 py-1.5 text-right font-mono text-[#c9d1d9]">{r.osl}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_ttft)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_ttft_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_ttft_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_tpot)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_tpot_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_tpot_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#c9d1d9] border-l border-[#21262d]">{fmtMs(r.meas_e2el)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#3fb950]">{fmtPct(r.pk_e2el_err)}</td>
                      <td className="px-2 py-1.5 text-right font-mono text-[#f85149]">{fmtPct(r.po_e2el_err)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-[#8b949e]">
            <span className="text-[#3fb950]">PK</span> = per-kernel (ncu-based XGBoost) · <span className="text-[#f85149]">PO</span> = per-op (torch.profiler XGBoost) · same 8-point trapezoidal decode integration
          </div>
        </div>
      )}
    </div>
  );
}

function PredictorResultsSection({ results }: { results?: PredictorResults }) {
  if (!results || (!results.serving_e2e && !results.serving_e2e_perop)) return null;
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
  const hasServing = Object.keys(results.serving_e2e ?? {}).length > 0 || Object.keys(results.serving_e2e_perop ?? {}).length > 0;
  const fmt = (v: number | undefined | null) => (v === undefined || v === null || isNaN(v) ? '—' : v.toFixed(2) + '%');
  return (
    <div className="space-y-6">
      <div className="text-sm font-semibold text-[#c9d1d9]">Predictor MAPE</div>

      {hasServing && <AblationSection results={results} />}

      {hasMicrobench && (
        <div className="space-y-4 rounded-lg border border-[#30363d] p-4">
          <SubPanelHeader
            title="Microbench TTFT"
            subtitle="per-kernel ncu + per-op held-out + wall-clock fixed-seq128 bs=1"
          />

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
              <div className="text-xs text-[#8b949e]">Per-kernel e2e error on prefill_seq128_bs1 (sum of kernel preds)</div>
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
                Per-kernel e2e wall-clock (vs real vLLM 0.19 TTFT, fixed-seq128, bs=1)
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
