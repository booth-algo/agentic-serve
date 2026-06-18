import { useEffect, useMemo, useState } from 'react';
import { servingPredictionsJsonUrl } from '../dataUrls';
import type { DataScope } from '../profileMeta';
import { DATA_SCOPE_META } from '../profileMeta';
import {
  buildServingIndex,
  type ServingIndex,
  type ServingRow,
} from './ServingPredictionsPage';

// The Predictions tab: one large hardware × model matrix. Rows = hardware configs
// (gpu + tensor-parallel + serving backend, i.e. the payload's gpu_key), columns =
// models. A metric toggle (E2EL / TTFT / TPOT) selects which metric the cells show:
// each cell = average predicted / measured of the selected metric over every
// (profile × concurrency) row for that pair, and the cell background = that metric's
// MAPE — i.e. the toggle switches between a self-consistent E2EL, TTFT, or TPOT matrix.
// Bands match the Simulator tab (servingErrorTone: <10% green, 10–25% blue, 25–50%
// orange, ≥50% red). Hover a cell for all three metrics.

interface MetricAgg {
  pred: number | null;
  meas: number | null;
}

interface CellAgg {
  ttft: MetricAgg;
  tpot: MetricAgg;
  e2el: MetricAgg;
  // Per-metric MAPE (mean of the row *_err %). The SELECTED metric's MAPE drives the cell
  // background + badge, so each toggle is a self-consistent TTFT / TPOT / E2EL matrix.
  ttftMape: number | null;
  tpotMape: number | null;
  e2elMape: number | null;
  n: number;
}

function average(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function aggregateCell(rows: ServingRow[]): CellAgg {
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
  return {
    ttft: metric('ttft'),
    tpot: metric('tpot'),
    e2el: metric('e2el'),
    ttftMape: average(collect('ttft_err')),
    tpotMape: average(collect('tpot_err')),
    e2elMape: average(collect('e2el_err')),
    n: rows.length,
  };
}

// The MAPE that drives the cell color/badge for the currently-selected metric.
function mapeFor(cell: CellAgg, metric: MetricKey): number | null {
  return metric === 'ttft' ? cell.ttftMape : metric === 'tpot' ? cell.tpotMape : cell.e2elMape;
}

function formatMs(value: number | null): string {
  if (value == null) return '—';
  if (value >= 10000) return `${(value / 1000).toFixed(1)} s`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  if (value >= 100) return `${value.toFixed(0)} ms`;
  return `${value.toFixed(1)} ms`;
}

// Same bands/colors as the Simulator tab's servingErrorTone, applied to the whole cell.
function mapeTone(mape: number | null): { cell: string; badge: string } {
  if (mape == null) {
    return { cell: 'bg-transparent', badge: 'text-[#6e7681]' };
  }
  const v = Math.abs(mape);
  if (v < 10) return { cell: 'bg-[#3fb950]/10', badge: 'text-[#3fb950]' };
  if (v < 25) return { cell: 'bg-[#58a6ff]/10', badge: 'text-[#58a6ff]' };
  if (v < 50) return { cell: 'bg-[#f0883e]/10', badge: 'text-[#f0883e]' };
  return { cell: 'bg-[#f85149]/10', badge: 'text-[#f85149]' };
}

// gpu_key encodes the hardware config: "H100x2" = H100, tp2, vLLM;
// "RTX3090x4 (sglang)" = RTX3090, tp4, sglang. Rows carry tensor_parallel_size
// when present, which wins over the key parse.
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

// All three metrics (pred / meas + MAPE) for the cell hover tooltip.
function cellTooltip(gpuKey: string, model: string, cell: CellAgg): string {
  const line = (label: string, m: MetricAgg, mape: number | null) =>
    `${label} ${formatMs(m.pred)} / ${formatMs(m.meas)}` + (mape != null ? ` (${mape.toFixed(1)}% MAPE)` : '');
  const head = `${gpuKey} × ${model} — avg over ${cell.n} profile×concurrency cells (predicted / measured)`;
  return `${head}\n${line('TTFT', cell.ttft, cell.ttftMape)}\n${line('TPOT', cell.tpot, cell.tpotMape)}\n${line('E2EL', cell.e2el, cell.e2elMape)}`;
}

export function PredictionsMatrixPage({
  dataScope,
  predictionsUrl = servingPredictionsJsonUrl,
}: {
  dataScope: DataScope;
  predictionsUrl?: string;
}) {
  const [servingIndex, setServingIndex] = useState<ServingIndex | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [metric, setMetric] = useState<MetricKey>('e2el');

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
        if (modelRows.length) byModel[model] = aggregateCell(modelRows);
      }
      return { gpuKey, parts: hardwareParts(gpuKey, rows), byModel };
    }).filter(h => Object.keys(h.byModel).length > 0);
    return { modelList, hardware };
  }, [scopeIndex]);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center text-[#8b949e]">
        Loading predictions…
      </div>
    );
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

  const metricLabel = METRICS.find(mm => mm.key === metric)!.label;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[#e6edf3]">Predictions matrix</h2>
          <p className="text-sm text-[#8b949e]">
            Per hardware config × model, averaged over all profiles and concurrencies
            ({DATA_SCOPE_META[dataScope].label.toLowerCase()}). Each cell:{' '}
            <span className="text-[#e6edf3]">predicted</span> /{' '}
            <span className="text-[#8b949e]">measured</span>; cells show the selected metric, background = its MAPE. Hover for all metrics.
          </p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <div className="inline-flex overflow-hidden rounded-md border border-[#30363d] text-xs">
            {METRICS.map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => setMetric(key)}
                className={`px-3 py-1 font-medium transition-colors ${
                  metric === key
                    ? 'bg-[#1f6feb] text-white'
                    : 'bg-[#161b22] text-[#8b949e] hover:bg-[#21262d]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-medium text-[#8b949e]">{metricLabel} MAPE:</span>
          <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">&lt;10%</span>
          <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 text-[#58a6ff]">10–25%</span>
          <span className="rounded border border-[#f0883e]/30 bg-[#f0883e]/10 px-2 py-0.5 text-[#f0883e]">25–50%</span>
          <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">≥50%</span>
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
              {matrix.modelList.map(model => (
                <th
                  key={model}
                  className="sticky top-0 z-20 whitespace-nowrap border-b border-[#30363d] bg-[#161b22] px-2.5 py-1 text-left font-medium text-[#e6edf3]"
                >
                  {model}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.hardware.map(({ gpuKey, parts, byModel }) => (
              <tr key={gpuKey} className="odd:bg-[#0d1117] even:bg-[#10151c]">
                <td className="sticky left-0 z-10 whitespace-nowrap border-r border-t border-[#30363d] bg-[#161b22] px-2.5 py-0.5 align-middle">
                  <div className="flex items-baseline gap-1.5 leading-none">
                    <span className="font-medium text-[#e6edf3]">{parts.gpu}</span>
                    <span className="text-xs text-[#8b949e]">tp{parts.tp} · {parts.backend}</span>
                  </div>
                </td>
                {matrix.modelList.map(model => {
                  const cell = byModel[model];
                  if (!cell) {
                    return (
                      <td key={model} className="border-t border-[#30363d] px-2.5 py-1 text-center align-middle text-[#484f58]">
                        —
                      </td>
                    );
                  }
                  const m = cell[metric];
                  const mape = mapeFor(cell, metric);
                  const tone = mapeTone(mape);
                  const hasGt = mape != null;
                  return (
                    <td
                      key={model}
                      className={`whitespace-nowrap border-t border-[#30363d] px-2.5 py-0.5 align-middle leading-none ${hasGt ? tone.cell : 'bg-[#21262d]/40'}`}
                      title={cellTooltip(gpuKey, model, cell)}
                    >
                      <div className="flex items-baseline justify-between gap-2 font-mono text-xs">
                        <span className="tabular-nums text-[#e6edf3]">{formatMs(m.pred)}</span>
                        <span className="tabular-nums text-[#6e7681]">/ {m.meas != null ? formatMs(m.meas) : '—'}</span>
                        {hasGt && (
                          <span className={`tabular-nums text-[10px] ${tone.badge}`}>{mape!.toFixed(0)}%</span>
                        )}
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
