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
// models, cells = average predicted AND measured TTFT / TPOT / E2EL over every
// (profile × concurrency) prediction row for that pair. The cell background is the
// cell's E2EL MAPE, using the same error bands as the Simulator tab
// (servingErrorTone: <10% green, 10–25% blue, 25–50% orange, ≥50% red).

interface MetricAgg {
  pred: number | null;
  meas: number | null;
}

interface CellAgg {
  ttft: MetricAgg;
  tpot: MetricAgg;
  e2el: MetricAgg;
  e2elMape: number | null; // mean of row e2el_err (%) — drives the background band
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
    e2elMape: average(collect('e2el_err')),
    n: rows.length,
  };
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

const METRIC_ROWS = [
  { key: 'ttft', label: 'TTFT' },
  { key: 'tpot', label: 'TPOT' },
  { key: 'e2el', label: 'E2EL' },
] as const;

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

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-[#e6edf3]">Predictions matrix</h2>
          <p className="text-sm text-[#8b949e]">
            Per hardware config × model, averaged over all profiles and concurrencies
            ({DATA_SCOPE_META[dataScope].label.toLowerCase()}). Each cell:{' '}
            <span className="text-[#e6edf3]">predicted</span> /{' '}
            <span className="text-[#8b949e]">measured</span>; background = E2EL MAPE.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="font-medium text-[#8b949e]">E2EL MAPE:</span>
          <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">&lt;10%</span>
          <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 text-[#58a6ff]">10–25%</span>
          <span className="rounded border border-[#f0883e]/30 bg-[#f0883e]/10 px-2 py-0.5 text-[#f0883e]">25–50%</span>
          <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">≥50%</span>
          <span className="rounded border border-[#30363d] bg-[#21262d] px-2 py-0.5 text-[#6e7681]">no GT</span>
        </div>
      </div>

      <div className="overflow-auto rounded-lg border border-[#30363d]" style={{ maxHeight: 'calc(100vh - 220px)' }}>
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              <th className="sticky left-0 top-0 z-30 border-b border-r border-[#30363d] bg-[#161b22] px-3 py-2 text-left font-medium text-[#8b949e]">
                Hardware config
              </th>
              {matrix.modelList.map(model => (
                <th
                  key={model}
                  className="sticky top-0 z-20 whitespace-nowrap border-b border-[#30363d] bg-[#161b22] px-3 py-2 text-left font-medium text-[#e6edf3]"
                >
                  {model}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.hardware.map(({ gpuKey, parts, byModel }) => (
              <tr key={gpuKey} className="odd:bg-[#0d1117] even:bg-[#10151c]">
                <td className="sticky left-0 z-10 whitespace-nowrap border-r border-t border-[#30363d] bg-[#161b22] px-3 py-2 align-top">
                  <div className="font-medium text-[#e6edf3]">{parts.gpu}</div>
                  <div className="text-xs text-[#8b949e]">
                    tp{parts.tp} · {parts.backend}
                  </div>
                </td>
                {matrix.modelList.map(model => {
                  const cell = byModel[model];
                  if (!cell) {
                    return (
                      <td key={model} className="border-t border-[#30363d] px-3 py-2 align-top text-[#484f58]">
                        —
                      </td>
                    );
                  }
                  const tone = mapeTone(cell.e2elMape);
                  const hasMeas = cell.ttft.meas != null || cell.tpot.meas != null || cell.e2el.meas != null;
                  return (
                    <td
                      key={model}
                      className={`whitespace-nowrap border-t border-[#30363d] px-3 py-2 align-top ${hasMeas ? tone.cell : 'bg-[#21262d]/40'}`}
                      title={`${gpuKey} × ${model}: averaged over ${cell.n} profile×concurrency cells${cell.e2elMape != null ? ` · E2EL MAPE ${cell.e2elMape.toFixed(1)}%` : ' · no ground truth'}`}
                    >
                      <div className="grid grid-cols-[auto_1fr_1fr] gap-x-2 text-xs leading-5">
                        {METRIC_ROWS.map(({ key, label }) => {
                          const m = cell[key];
                          return (
                            <MetricRow key={key} label={label} pred={m.pred} meas={m.meas} />
                          );
                        })}
                      </div>
                      {cell.e2elMape != null && (
                        <div className={`mt-1 text-right font-mono text-[10px] ${tone.badge}`}>
                          {cell.e2elMape.toFixed(1)}% e2el
                        </div>
                      )}
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

function MetricRow({ label, pred, meas }: { label: string; pred: number | null; meas: number | null }) {
  return (
    <>
      <span className="text-[#8b949e]">{label}</span>
      <span className="text-right tabular-nums text-[#e6edf3]">{formatMs(pred)}</span>
      <span className="text-right tabular-nums text-[#8b949e]">{formatMs(meas)}</span>
    </>
  );
}
