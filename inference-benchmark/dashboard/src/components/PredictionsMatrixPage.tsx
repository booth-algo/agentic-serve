import { useEffect, useMemo, useState } from 'react';
import { servingPredictionsJsonUrl, forwardPredictionsJsonUrl } from '../dataUrls';
import type { DataScope } from '../profileMeta';
import { DATA_SCOPE_META } from '../profileMeta';
import {
  buildFwdLookup,
  fwdKey,
  type ForwardRow,
  type FwdLookup,
} from '../forwardPredictions';
import {
  buildServingIndex,
  type ServingIndex,
  type ServingRow,
} from './ServingPredictionsPage';

// The Predictions tab: one large hardware × model matrix. Rows = hardware configs (gpu_key),
// columns = models. A metric toggle (E2EL / TTFT / TPOT) selects the metric; a SOURCE toggle picks
// the predictor: Backtest (build_simulator_rows, reads the realized trajectory_pool + scores) vs
// Forward (simulator.forward, fed the same workload as a client would — no-GT path) vs Δ (forward
// MAPE − backtester MAPE, green = forward closer to measured). Both score against the SAME measured
// GT, so the delta is the cost of going GT-free. Bands match the Simulator tab.

interface MetricAgg {
  pred: number | null;
  meas: number | null;
}

interface CellAgg {
  ttft: MetricAgg;
  tpot: MetricAgg;
  e2el: MetricAgg;
  ttftMape: number | null;
  tpotMape: number | null;
  e2elMape: number | null;
  // Forward predictor: cohort-mean pred + MAPE vs the same measured GT (null if no forward row).
  fwdTtftPred: number | null;
  fwdTpotPred: number | null;
  fwdE2elPred: number | null;
  fwdTtftMape: number | null;
  fwdTpotMape: number | null;
  fwdE2elMape: number | null;
  // Backtester MAPE over ONLY the forward-matched cells — the fair Δ baseline (forward may cover
  // fewer concurrencies than the backtester, so the full btMape isn't apples-to-apples).
  btMatchedTtftMape: number | null;
  btMatchedTpotMape: number | null;
  btMatchedE2elMape: number | null;
  n: number;
}

function average(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function aggregateCell(rows: ServingRow[], gpuKey: string, fwd: FwdLookup): CellAgg {
  const collect = (key: keyof ServingRow): number[] => {
    const out: number[] = [];
    for (const row of rows) {
      const v = row[key];
      if (typeof v === 'number' && Number.isFinite(v)) out.push(v);
    }
    return out;
  };
  const metric = (k: 'ttft' | 'tpot' | 'e2el'): MetricAgg => ({
    pred: average(collect(`${k}_pred` as keyof ServingRow)),
    meas: average(collect(`${k}_meas` as keyof ServingRow)),
  });
  // Join the forward rows by (gpu_key, model, profile, concurrency).
  const fwdPred: Record<'ttft' | 'tpot' | 'e2el', number[]> = { ttft: [], tpot: [], e2el: [] };
  const fwdErr: Record<'ttft' | 'tpot' | 'e2el', number[]> = { ttft: [], tpot: [], e2el: [] };
  const btMatchedErr: Record<'ttft' | 'tpot' | 'e2el', number[]> = { ttft: [], tpot: [], e2el: [] };
  for (const row of rows) {
    const r = row as ServingRow & { profile?: string; concurrency?: number };
    if (r.model == null || r.profile == null || r.concurrency == null) continue;
    const f = fwd.get(fwdKey(gpuKey, r.model, r.profile, r.concurrency));
    if (!f) continue;
    for (const k of ['ttft', 'tpot', 'e2el'] as const) {
      const p = f[`fwd_${k}_pred` as keyof ForwardRow];
      const e = f[`fwd_${k}_err` as keyof ForwardRow];
      if (typeof p === 'number' && Number.isFinite(p)) fwdPred[k].push(p);
      if (typeof e === 'number' && Number.isFinite(e)) fwdErr[k].push(e);
      const be = row[`${k}_err` as keyof ServingRow];
      if (typeof be === 'number' && Number.isFinite(be)) btMatchedErr[k].push(be);
    }
  }
  return {
    ttft: metric('ttft'),
    tpot: metric('tpot'),
    e2el: metric('e2el'),
    ttftMape: average(collect('ttft_err')),
    tpotMape: average(collect('tpot_err')),
    e2elMape: average(collect('e2el_err')),
    fwdTtftPred: average(fwdPred.ttft),
    fwdTpotPred: average(fwdPred.tpot),
    fwdE2elPred: average(fwdPred.e2el),
    fwdTtftMape: average(fwdErr.ttft),
    fwdTpotMape: average(fwdErr.tpot),
    fwdE2elMape: average(fwdErr.e2el),
    btMatchedTtftMape: average(btMatchedErr.ttft),
    btMatchedTpotMape: average(btMatchedErr.tpot),
    btMatchedE2elMape: average(btMatchedErr.e2el),
    n: rows.length,
  };
}

const BT_MAPE = { ttft: 'ttftMape', tpot: 'tpotMape', e2el: 'e2elMape' } as const;
const BT_MATCHED = { ttft: 'btMatchedTtftMape', tpot: 'btMatchedTpotMape', e2el: 'btMatchedE2elMape' } as const;
const FWD_MAPE = { ttft: 'fwdTtftMape', tpot: 'fwdTpotMape', e2el: 'fwdE2elMape' } as const;
const FWD_PRED = { ttft: 'fwdTtftPred', tpot: 'fwdTpotPred', e2el: 'fwdE2elPred' } as const;

function btMape(cell: CellAgg, metric: MetricKey): number | null {
  return cell[BT_MAPE[metric]];
}
function btMatchedMape(cell: CellAgg, metric: MetricKey): number | null {
  return cell[BT_MATCHED[metric]];
}
function fwdMape(cell: CellAgg, metric: MetricKey): number | null {
  return cell[FWD_MAPE[metric]];
}

// The pred MetricAgg the cell shows for the selected source (measured is the same GT either way).
function predFor(cell: CellAgg, metric: MetricKey, source: SourceKey): MetricAgg {
  if (source === 'forward') return { pred: cell[FWD_PRED[metric]], meas: cell[metric].meas };
  return cell[metric];
}

function formatMs(value: number | null): string {
  if (value == null) return '—';
  if (value >= 10000) return `${(value / 1000).toFixed(1)} s`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  if (value >= 100) return `${value.toFixed(0)} ms`;
  return `${value.toFixed(1)} ms`;
}

function mapeTone(mape: number | null): { cell: string; badge: string } {
  if (mape == null) return { cell: 'bg-transparent', badge: 'text-[#6e7681]' };
  const v = Math.abs(mape);
  if (v < 10) return { cell: 'bg-[#3fb950]/10', badge: 'text-[#3fb950]' };
  if (v < 25) return { cell: 'bg-[#58a6ff]/10', badge: 'text-[#58a6ff]' };
  if (v < 50) return { cell: 'bg-[#f0883e]/10', badge: 'text-[#f0883e]' };
  return { cell: 'bg-[#f85149]/10', badge: 'text-[#f85149]' };
}

// Signed tone for the Δ view: negative = forward closer to measured (green), positive = worse (red).
function deltaTone(d: number | null): { cell: string; badge: string } {
  if (d == null) return { cell: 'bg-transparent', badge: 'text-[#6e7681]' };
  if (d <= -3) return { cell: 'bg-[#3fb950]/10', badge: 'text-[#3fb950]' };
  if (d >= 3) return { cell: 'bg-[#f85149]/10', badge: 'text-[#f85149]' };
  return { cell: 'bg-[#21262d]/40', badge: 'text-[#8b949e]' };
}

function hardwareParts(gpuKey: string, rows: ServingRow[]): { gpu: string; tp: number; backend: string } {
  const backend = /\(sglang\)/i.test(gpuKey) ? 'sglang' : 'vllm';
  const base = gpuKey.replace(/\s*\(sglang\)\s*/i, '');
  const tpMatch = base.match(/x(\d+)$/);
  const rowTp = rows
    .map(r => (r as { tensor_parallel_size?: number }).tensor_parallel_size)
    .find(v => typeof v === 'number');
  return {
    gpu: tpMatch ? base.slice(0, -tpMatch[0].length) : base,
    tp: rowTp ?? (tpMatch ? Number(tpMatch[1]) : 1),
    backend,
  };
}

const METRICS = [
  { key: 'e2el', label: 'E2EL' },
  { key: 'ttft', label: 'TTFT' },
  { key: 'tpot', label: 'TPOT' },
] as const;
type MetricKey = (typeof METRICS)[number]['key'];

const BACKENDS = [
  { key: 'all', label: 'All' },
  { key: 'vllm', label: 'vLLM' },
  { key: 'sglang', label: 'SGLang' },
] as const;
type BackendKey = (typeof BACKENDS)[number]['key'];

const SOURCES = [
  { key: 'backtester', label: 'Backtest' },
  { key: 'forward', label: 'Forward' },
  { key: 'delta', label: 'Δ' },
] as const;
type SourceKey = (typeof SOURCES)[number]['key'];

function cellTooltip(gpuKey: string, model: string, cell: CellAgg): string {
  const line = (label: string, m: MetricAgg, mape: number | null, fwdM: number | null) =>
    `${label} bt ${formatMs(m.pred)}/${formatMs(m.meas)}` +
    (mape != null ? ` (${mape.toFixed(1)}%)` : '') +
    (fwdM != null ? ` · fwd ${fwdM.toFixed(1)}%` : '');
  const head = `${gpuKey} × ${model} — avg over ${cell.n} cells (backtest vs forward MAPE)`;
  return `${head}\n${line('TTFT', cell.ttft, cell.ttftMape, cell.fwdTtftMape)}\n${line('TPOT', cell.tpot, cell.tpotMape, cell.fwdTpotMape)}\n${line('E2EL', cell.e2el, cell.e2elMape, cell.fwdE2elMape)}`;
}

export function PredictionsMatrixPage({
  dataScope,
  predictionsUrl = servingPredictionsJsonUrl,
}: {
  dataScope: DataScope;
  predictionsUrl?: string;
}) {
  const [servingIndex, setServingIndex] = useState<ServingIndex | null>(null);
  const [fwd, setFwd] = useState<FwdLookup>(new Map());
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [metric, setMetric] = useState<MetricKey>('e2el');
  const [backend, setBackend] = useState<BackendKey>('vllm');
  const [source, setSource] = useState<SourceKey>('backtester');

  useEffect(() => {
    setLoading(true);
    setFailed(false);
    fetch(predictionsUrl)
      .then(r => r.json())
      .then((json: Record<string, ServingRow[]>) => {
        setServingIndex(buildServingIndex(json));
        setLoading(false);
      })
      .catch(() => {
        setFailed(true);
        setLoading(false);
      });
  }, [predictionsUrl]);

  // Forward predictions are optional — absent (404) until build_forward_rows has run.
  useEffect(() => {
    fetch(forwardPredictionsJsonUrl)
      .then(r => (r.ok ? r.json() : null))
      .then((json: Record<string, ForwardRow[]> | null) => setFwd(buildFwdLookup(json)))
      .catch(() => setFwd(new Map()));
  }, []);

  const scopeIndex = servingIndex?.[dataScope];

  const matrix = useMemo(() => {
    if (!scopeIndex) return null;
    const models = new Set<string>();
    for (const rows of Object.values(scopeIndex.rowsByGpu)) {
      for (const row of rows) if (row.model) models.add(row.model);
    }
    const modelList = Array.from(models).sort();
    const hardware = scopeIndex.gpuOptions.map(gpuKey => {
      const rows = scopeIndex.rowsByGpu[gpuKey] ?? [];
      const byModel: Record<string, CellAgg> = {};
      for (const model of modelList) {
        const modelRows = rows.filter(r => r.model === model);
        if (modelRows.length) byModel[model] = aggregateCell(modelRows, gpuKey, fwd);
      }
      return { gpuKey, parts: hardwareParts(gpuKey, rows), byModel };
    }).filter(h => Object.keys(h.byModel).length > 0);
    return { modelList, hardware };
  }, [scopeIndex, fwd]);

  if (loading) {
    return <div className="flex h-64 items-center justify-center text-[#8b949e]">Loading predictions…</div>;
  }
  if (failed || !matrix) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#f97583]/30 bg-[#f97583]/10 text-[#f97583]">
        Failed to load predictions data.
      </div>
    );
  }
  if (!matrix.hardware.length) {
    return (
      <div className="flex h-64 items-center justify-center text-[#8b949e]">
        No prediction rows in the {DATA_SCOPE_META[dataScope].label.toLowerCase()} scope.
      </div>
    );
  }

  const availableBackends = Array.from(new Set(matrix.hardware.map(h => h.parts.backend)));
  const effBackend: BackendKey =
    backend !== 'all' && !availableBackends.includes(backend) ? 'all' : backend;
  const shownHardware = matrix.hardware.filter(h => effBackend === 'all' || h.parts.backend === effBackend);
  const shownModelList = matrix.modelList.filter(model => shownHardware.some(h => h.byModel[model]));
  const hasForward = fwd.size > 0;
  const metricLabel = METRICS.find(mm => mm.key === metric)!.label;
  const sourceLabel = source === 'forward' ? 'forward' : source === 'delta' ? 'Δ (forward − backtest)' : 'backtest';

  const toggle = (
    items: readonly { key: string; label: string }[],
    active: string,
    on: (k: string) => void,
    disabledKeys: string[] = [],
  ) => (
    <div className="inline-flex overflow-hidden rounded-md border border-[#30363d] text-xs">
      {items.map(({ key, label }) => {
        const disabled = disabledKeys.includes(key);
        return (
          <button
            key={key}
            type="button"
            disabled={disabled}
            onClick={() => on(key)}
            className={`px-3 py-1 font-medium transition-colors ${
              active === key ? 'bg-[#1f6feb] text-white'
                : disabled ? 'bg-[#161b22] text-[#484f58] cursor-not-allowed'
                : 'bg-[#161b22] text-[#8b949e] hover:bg-[#21262d]'
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[#e6edf3]">Predictions matrix</h2>
          <p className="text-sm text-[#8b949e]">
            Per hardware config × model, averaged over all profiles and concurrencies
            ({DATA_SCOPE_META[dataScope].label.toLowerCase()}). Source = <span className="text-[#e6edf3]">{sourceLabel}</span>;
            cells show {metricLabel} <span className="text-[#e6edf3]">predicted</span> /{' '}
            <span className="text-[#8b949e]">measured</span>; background = MAPE. Hover for backtest vs forward.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="flex items-center gap-2">
            {toggle(SOURCES, source, k => setSource(k as SourceKey), hasForward ? [] : ['forward', 'delta'])}
            {availableBackends.length > 1 &&
              toggle(BACKENDS.filter(b => b.key === 'all' || availableBackends.includes(b.key)),
                effBackend, k => setBackend(k as BackendKey))}
          </div>
          {toggle(METRICS, metric, k => setMetric(k as MetricKey))}
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="font-medium text-[#8b949e]">{source === 'delta' ? 'Δ MAPE (− = fwd better)' : `${metricLabel} MAPE`}:</span>
            {source === 'delta' ? (
              <>
                <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">fwd ≥3pt better</span>
                <span className="rounded border border-[#30363d] bg-[#21262d] px-2 py-0.5 text-[#8b949e]">~equal</span>
                <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">fwd ≥3pt worse</span>
              </>
            ) : (
              <>
                <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">&lt;10%</span>
                <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 text-[#58a6ff]">10–25%</span>
                <span className="rounded border border-[#f0883e]/30 bg-[#f0883e]/10 px-2 py-0.5 text-[#f0883e]">25–50%</span>
                <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">≥50%</span>
              </>
            )}
            <span className="rounded border border-[#30363d] bg-[#21262d] px-2 py-0.5 text-[#6e7681]">no GT</span>
          </div>
        </div>
      </div>

      <div className="overflow-auto rounded-lg border border-[#30363d]" style={{ maxHeight: 'calc(100vh - 180px)' }}>
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-30 border-b border-r border-[#30363d] bg-[#161b22] px-2.5 py-1 text-left font-medium text-[#8b949e]">
                Hardware config
              </th>
              {shownModelList.map(model => (
                <th key={model} className="sticky top-0 z-20 whitespace-nowrap border-b border-[#30363d] bg-[#161b22] px-2.5 py-1 text-left font-medium text-[#e6edf3]">
                  {model}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shownHardware.map(({ gpuKey, parts, byModel }) => (
              <tr key={gpuKey} className="odd:bg-[#0d1117] even:bg-[#10151c]">
                <td className="sticky left-0 z-10 whitespace-nowrap border-r border-t border-[#30363d] bg-[#161b22] px-2.5 py-0.5 align-middle">
                  <div className="flex items-baseline gap-1.5 leading-none">
                    <span className="font-medium text-[#e6edf3]">{parts.gpu}</span>
                    <span className="text-xs text-[#8b949e]">tp{parts.tp} · {parts.backend}</span>
                  </div>
                </td>
                {shownModelList.map(model => {
                  const cell = byModel[model];
                  if (!cell) {
                    return <td key={model} className="border-t border-[#30363d] px-2.5 py-1 text-center align-middle text-[#484f58]">—</td>;
                  }
                  if (source === 'delta') {
                    const bt = btMatchedMape(cell, metric);  // matched cells only — fair vs forward
                    const fw = fwdMape(cell, metric);
                    const d = bt != null && fw != null ? fw - bt : null;
                    const tone = deltaTone(d);
                    return (
                      <td key={model} className={`whitespace-nowrap border-t border-[#30363d] px-2.5 py-0.5 align-middle leading-none ${d != null ? tone.cell : 'bg-[#21262d]/40'}`} title={cellTooltip(gpuKey, model, cell)}>
                        <div className="flex items-baseline justify-between gap-2 font-mono text-xs">
                          <span className="tabular-nums text-[#6e7681]">{bt != null ? `${bt.toFixed(0)}` : '—'}<span className="text-[#484f58]">→</span>{fw != null ? `${fw.toFixed(0)}` : '—'}</span>
                          {d != null && <span className={`tabular-nums text-[10px] ${tone.badge}`}>{d > 0 ? '+' : ''}{d.toFixed(0)}</span>}
                        </div>
                      </td>
                    );
                  }
                  const m = predFor(cell, metric, source);
                  const mape = source === 'forward' ? fwdMape(cell, metric) : btMape(cell, metric);
                  const tone = mapeTone(mape);
                  const hasGt = mape != null;
                  return (
                    <td key={model} className={`whitespace-nowrap border-t border-[#30363d] px-2.5 py-0.5 align-middle leading-none ${hasGt ? tone.cell : 'bg-[#21262d]/40'}`} title={cellTooltip(gpuKey, model, cell)}>
                      <div className="flex items-baseline justify-between gap-2 font-mono text-xs">
                        <span className="tabular-nums text-[#e6edf3]">{formatMs(m.pred)}</span>
                        <span className="tabular-nums text-[#6e7681]">/ {m.meas != null ? formatMs(m.meas) : '—'}</span>
                        {hasGt && <span className={`tabular-nums text-[10px] ${tone.badge}`}>{mape!.toFixed(0)}%</span>}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
