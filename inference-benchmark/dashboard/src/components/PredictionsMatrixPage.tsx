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
// models, cells = the average TTFT / TPOT / E2EL over every (profile × concurrency)
// prediction row for that pair. A toggle switches between predicted values and the
// measured benchmark values where ground truth exists.

type MetricMode = 'pred' | 'meas';

interface CellAgg {
  ttft: number | null;
  tpot: number | null;
  e2el: number | null;
  n: number; // rows (profile × concurrency cells) behind the averages
}

function average(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function aggregateCell(rows: ServingRow[], mode: MetricMode): CellAgg {
  const collect = (key: 'ttft' | 'tpot' | 'e2el'): number[] => {
    const out: number[] = [];
    for (const row of rows) {
      const v = mode === 'pred' ? row[`${key}_pred`] : row[`${key}_meas`];
      if (typeof v === 'number' && Number.isFinite(v)) out.push(v);
    }
    return out;
  };
  return {
    ttft: average(collect('ttft')),
    tpot: average(collect('tpot')),
    e2el: average(collect('e2el')),
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

// gpu_key encodes the hardware config: "H100x2" = H100, tp2, vLLM;
// "RTX3090x4 (sglang)" = RTX3090, tp4, sglang. Derive the (gpu, tp, backend)
// sub-label from the key (rows carry tensor_parallel_size when present, which wins).
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

export function PredictionsMatrixPage({
  dataScope,
  predictionsUrl = servingPredictionsJsonUrl,
}: {
  dataScope: DataScope;
  predictionsUrl?: string;
}) {
  const [servingIndex, setServingIndex] = useState<ServingIndex | null>(null);
  const [mode, setMode] = useState<MetricMode>('pred');
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
        if (modelRows.length) byModel[model] = aggregateCell(modelRows, mode);
      }
      return { gpuKey, parts: hardwareParts(gpuKey, rows), byModel };
    }).filter(h => Object.keys(h.byModel).length > 0);
    return { modelList, hardware };
  }, [scopeIndex, mode]);

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
            Average {mode === 'pred' ? 'predicted' : 'measured'} TTFT / TPOT / E2EL per
            hardware config × model, across all profiles and concurrencies
            ({DATA_SCOPE_META[dataScope].label.toLowerCase()}).
          </p>
        </div>
        <div className="flex overflow-hidden rounded-md border border-[#30363d] text-sm">
          {(['pred', 'meas'] as const).map(m => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={
                m === mode
                  ? 'bg-[#1f6feb] px-3 py-1.5 font-medium text-white'
                  : 'bg-[#161b22] px-3 py-1.5 text-[#8b949e] hover:text-[#e6edf3]'
              }
            >
              {m === 'pred' ? 'Predicted' : 'Measured'}
            </button>
          ))}
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
                  if (!cell || (cell.ttft == null && cell.tpot == null && cell.e2el == null)) {
                    return (
                      <td key={model} className="border-t border-[#30363d] px-3 py-2 align-top text-[#484f58]">
                        —
                      </td>
                    );
                  }
                  return (
                    <td
                      key={model}
                      className="whitespace-nowrap border-t border-[#30363d] px-3 py-2 align-top"
                      title={`${gpuKey} × ${model}: averaged over ${cell.n} profile×concurrency cells (${mode === 'pred' ? 'predicted' : 'measured'})`}
                    >
                      <div className="grid grid-cols-[auto_1fr] gap-x-2 text-xs leading-5">
                        <span className="text-[#8b949e]">TTFT</span>
                        <span className="text-right tabular-nums text-[#e6edf3]">{formatMs(cell.ttft)}</span>
                        <span className="text-[#8b949e]">TPOT</span>
                        <span className="text-right tabular-nums text-[#e6edf3]">{formatMs(cell.tpot)}</span>
                        <span className="text-[#8b949e]">E2EL</span>
                        <span className="text-right tabular-nums text-[#e6edf3]">{formatMs(cell.e2el)}</span>
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
