import { Fragment, useState, useEffect } from 'react';
import {
  type DataScope,
  isProfileInScope,
  normalizeProfileName,
  profileDisplayName,
} from '../profileMeta';
import { gemmEvalJsonUrl, servingPredictionsJsonUrl } from '../dataUrls';

interface GemmPoint {
  M: number; N: number; K: number;
  measured: number; predicted: number; roofline: number;
}
interface ModelResult {
  model: string; status: string;
  n_shapes?: number; mape?: number; roofline_mape?: number;
  points?: { M: number; N: number; K: number; measured_us: number; predicted_us: number; roofline_us: number }[];
}
interface GpuData {
  n_shapes: number; roofline_mape: number; xgb_mape: number;
  model_results: ModelResult[];
  scatter: GemmPoint[];
}
interface GemmEval {
  generated_at: string;
  gpus: Record<string, GpuData>;
}

function mapeColor(v: number): string {
  if (v < 5) return '#3fb950';
  if (v < 10) return '#58a6ff';
  if (v < 20) return '#ff9800';
  return '#f85149';
}

const SERVING_GPU_ORDER = [
  'H100',
  'H100x2',
  'H100x4',
  'A100',
  'A100x2',
  'A100x4',
  'A100x8',
  'RTX3090',
  'RTX3090x2',
  'RTX3090x4',
  'RTX3090x8',
  'RTX2080Ti',
  'RTX2080Tix2',
  'RTX2080Tix4',
];

function compareServingGpus(a: string, b: string): number {
  const aRank = SERVING_GPU_ORDER.indexOf(a);
  const bRank = SERVING_GPU_ORDER.indexOf(b);
  const normalizedARank = aRank === -1 ? SERVING_GPU_ORDER.length : aRank;
  const normalizedBRank = bRank === -1 ? SERVING_GPU_ORDER.length : bRank;
  if (normalizedARank !== normalizedBRank) return normalizedARank - normalizedBRank;
  return a.localeCompare(b);
}

function availableServingGpus(data: Record<string, ServingRow[]>, dataScope: DataScope): string[] {
  return Object.keys(data)
    .filter(g => scopedServingRows(data, g, dataScope).length > 0)
    .sort(compareServingGpus);
}

export function GemmPage({ dataScope }: { dataScope: DataScope }) {
  const [data, setData] = useState<GemmEval | null>(null);
  const [gpu, setGpu] = useState('H100');
  const [servingGpu, setServingGpu] = useState('H100');
  const [servingGpuOptions, setServingGpuOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(gemmEvalJsonUrl)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    fetch(servingPredictionsJsonUrl)
      .then(r => r.json())
      .then((d: Record<string, ServingRow[]>) => {
        const options = availableServingGpus(d, dataScope);
        setServingGpuOptions(options);
        setServingGpu(current => options.includes(current) ? current : (options[0] ?? current));
      })
      .catch(() => setServingGpuOptions([]));
  }, [dataScope]);

  if (loading) return <div className="text-[#8b949e] p-8">Loading GEMM evaluation data...</div>;
  if (!data) return <div className="text-[#f85149] p-8">Failed to load gemm-eval.json</div>;

  const gpus = Object.keys(data.gpus);
  const servingTabs = servingGpuOptions.length ? servingGpuOptions : gpus;
  const gpuData = data.gpus[gpu];
  if (!gpuData) return null;

  const testedModels = gpuData.model_results.filter(m => m.status === 'tested' && m.mape !== undefined);
  const sharedModels = gpuData.model_results.filter(m => m.status === 'shared');

  const scatterMax = Math.max(...gpuData.scatter.map(s => Math.max(s.measured, s.predicted)));

  return (
    <div className="space-y-6">
      {/* Explainer */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] px-5 py-4 text-sm text-[#c9d1d9]">
        <h3 className="text-base font-semibold text-[#e6edf3] mb-2">Per-Kernel GEMM Predictor</h3>
        <p className="mb-2">
          In transformer inference, <strong className="text-[#e6edf3]">GEMM (matrix multiply) kernels account for
          92–97% of decode compute time</strong> and 60–80% of prefill time. Predicting GEMM latency accurately
          is the single most important driver for end-to-end serving latency prediction.
        </p>
        <p className="mb-2">
          We profile each GPU's cuBLAS/cuBLASLt GEMM dispatch by sweeping <code className="text-[#79c0ff] bg-[#21262d] px-1 rounded">nn.Linear</code> across
          the exact (M, N, K) shapes that LLM serving emits — QKV projections, FFN gate/up/down, and LM head.
          An XGBoost model learns the residual between a roofline baseline and measured ncu kernel time:
        </p>
        <div className="bg-[#0d1117] rounded px-3 py-2 font-mono text-xs text-[#8b949e] mb-2">
          prediction = roofline(M, N, K) × exp(XGBoost_residual(log₂M, log₂N, log₂K, OI, log(roofline)))
        </div>
        <p className="text-[#8b949e] text-xs">
          Evaluation: leave-one-model-out — train on all models except one, predict the held-out model's unique GEMM shapes.
          Models sharing identical shapes with training models are marked "shared" (no extrapolation test needed).
        </p>
      </div>

      {/* GPU selector */}
      <div className="flex gap-2">
        {gpus.map(g => (
          <button
            key={g}
            onClick={() => setGpu(g)}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              g === gpu
                ? 'bg-[#1f6feb] text-white'
                : 'bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3]'
            }`}
          >
            {g}
          </button>
        ))}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <SummaryCard label="Shapes Profiled" value={gpuData.n_shapes.toString()} />
        <SummaryCard label="Roofline-Only MAPE" value={`${gpuData.roofline_mape}%`} color="#f85149" />
        <SummaryCard label="XGBoost In-Sample MAPE" value={`${gpuData.xgb_mape}%`} color="#3fb950" />
        <SummaryCard
          label="Leave-One-Model-Out Median MAPE"
          value={`${median(testedModels.map(m => m.mape!)).toFixed(1)}%`}
          color="#58a6ff"
        />
      </div>

      {/* Two-column: roofline vs xgb scatter + model bar chart */}
      <div className="grid grid-cols-2 gap-4">
        {/* Roofline vs XGBoost comparison scatter */}
        <div className="rounded-lg border border-[#21262d] bg-[#0d1117] p-4">
          <h4 className="text-sm font-semibold text-[#e6edf3] mb-1">Roofline vs XGBoost+Roofline — Predicted vs Measured</h4>
          <p className="text-xs text-[#484f58] mb-3">Each dot is one GEMM shape. Closer to the diagonal = better prediction.</p>
          <svg viewBox="0 0 400 380" className="w-full">
            {/* Perfect diagonal */}
            <line x1="50" y1="340" x2="380" y2="10" stroke="#30363d" strokeWidth="1" strokeDasharray="4" />
            {/* Axes */}
            <line x1="50" y1="340" x2="380" y2="340" stroke="#484f58" strokeWidth="1" />
            <line x1="50" y1="340" x2="50" y2="10" stroke="#484f58" strokeWidth="1" />
            {/* Roofline points (red, behind) */}
            {gpuData.scatter.map((s, i) => {
              const logMax = Math.log10(Math.max(scatterMax, 10));
              const x = 50 + (Math.log10(Math.max(s.measured, 0.1)) / logMax) * 330;
              const y = 340 - (Math.log10(Math.max(s.roofline, 0.1)) / logMax) * 330;
              return <circle key={`r${i}`} cx={x} cy={y} r="2.5" fill="#f85149" opacity="0.35" />;
            })}
            {/* XGBoost points (blue, on top) */}
            {gpuData.scatter.map((s, i) => {
              const logMax = Math.log10(Math.max(scatterMax, 10));
              const x = 50 + (Math.log10(Math.max(s.measured, 0.1)) / logMax) * 330;
              const y = 340 - (Math.log10(Math.max(s.predicted, 0.1)) / logMax) * 330;
              return <circle key={`x${i}`} cx={x} cy={y} r="2.5" fill="#58a6ff" opacity="0.6" />;
            })}
            {/* Legend */}
            <circle cx="70" cy="20" r="4" fill="#f85149" opacity="0.5" />
            <text x="80" y="24" fill="#f85149" fontSize="10">Roofline only ({gpuData.roofline_mape}% MAPE)</text>
            <circle cx="230" cy="20" r="4" fill="#58a6ff" opacity="0.7" />
            <text x="240" y="24" fill="#58a6ff" fontSize="10">XGBoost+Roofline ({gpuData.xgb_mape}% MAPE)</text>
            {/* Axis labels */}
            <text x="215" y="370" fill="#8b949e" fontSize="11" textAnchor="middle">Measured (μs, log scale)</text>
            <text x="15" y="175" fill="#8b949e" fontSize="11" textAnchor="middle" transform="rotate(-90 15 175)">Predicted (μs, log scale)</text>
          </svg>
        </div>

        {/* Leave-one-model-out bar chart */}
        <div className="rounded-lg border border-[#21262d] bg-[#0d1117] p-4 min-h-[380px]">
          <h4 className="text-sm font-semibold text-[#e6edf3] mb-3">Leave-One-Model-Out MAPE</h4>
          <div className="space-y-2">
            {testedModels.sort((a, b) => (b.mape ?? 0) - (a.mape ?? 0)).map(m => (
              <div key={m.model} className="flex items-center gap-2">
                <span className="text-xs text-[#8b949e] w-28 truncate text-right">{m.model}</span>
                <div className="flex-1 bg-[#21262d] rounded-full h-5 relative">
                  <div
                    className="h-5 rounded-full flex items-center justify-end pr-2"
                    style={{
                      width: `${Math.min((m.mape ?? 0) / 50 * 100, 100)}%`,
                      backgroundColor: mapeColor(m.mape ?? 0),
                      opacity: 0.8,
                      minWidth: '40px',
                    }}
                  >
                    <span className="text-[10px] font-bold text-white">{m.mape?.toFixed(1)}%</span>
                  </div>
                </div>
                <span className="text-[10px] text-[#484f58] w-16">{m.n_shapes} shapes</span>
              </div>
            ))}
            {sharedModels.map(m => (
              <div key={m.model} className="flex items-center gap-2">
                <span className="text-xs text-[#8b949e] w-28 truncate text-right">{m.model}</span>
                <div className="flex-1">
                  <span className="text-xs text-[#484f58] italic ml-2">shared shapes — no extrapolation needed</span>
                </div>
              </div>
            ))}
          </div>
          {/* Roofline comparison line */}
          <div className="mt-3 pt-2 border-t border-[#21262d] flex items-center gap-2">
            <span className="text-xs text-[#8b949e] w-28 text-right">Roofline only</span>
            <div className="flex-1 bg-[#21262d] rounded-full h-5 relative">
              <div
                className="h-5 rounded-full flex items-center justify-end pr-2"
                style={{
                  width: `${Math.min(gpuData.roofline_mape / 50 * 100, 100)}%`,
                  backgroundColor: '#f85149',
                  opacity: 0.5,
                  minWidth: '60px',
                }}
              >
                <span className="text-[10px] font-bold text-white">{gpuData.roofline_mape}%</span>
              </div>
            </div>
            <span className="text-[10px] text-[#484f58] w-16">baseline</span>
          </div>
        </div>
      </div>

      {/* Serving Predictions Section */}
      <div className="mt-4 border-t border-[#30363d] pt-6">
        <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h3 className="text-base font-semibold text-[#e6edf3]">Serving Latency Predictions</h3>
            <p className="text-xs text-[#484f58] mt-1">
              High-concurrency predictions vs measured benchmark results. {dataScope === 'current' ? 'Showing the canonical paper profiles.' : 'Showing archived legacy and stress profiles.'}
              Multi-turn TTFT reflects cache-aware serving behavior, not cumulative full-prefill latency.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5 lg:justify-end">
            {servingTabs.map(g => (
              <button
                key={`s-${g}`}
                onClick={() => setServingGpu(g)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                  g === servingGpu
                    ? 'bg-[#1f6feb] text-white'
                    : 'bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3]'
                }`}
              >
                {g}
              </button>
            ))}
          </div>
        </div>
        <ServingTable gpu={servingGpu} dataScope={dataScope} />
      </div>
    </div>
  );
}

function SummaryCard({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="rounded-lg border border-[#21262d] bg-[#161b22] px-4 py-3">
      <div className="text-xs text-[#8b949e]">{label}</div>
      <div className="text-2xl font-bold mt-1" style={{ color: color ?? '#e6edf3' }}>{value}</div>
    </div>
  );
}

interface ServingTurnPrediction {
  turn_index: number;
  successful: number;
  total_context_tokens: number;
  new_prefill_tokens: number;
  cached_context_tokens: number;
  cache_hit_rate: number;
  output_tokens: number;
  ttft_pred?: number; ttft_meas?: number; ttft_err?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  e2el_pred?: number; e2el_meas?: number; e2el_err?: number;
}

interface ServingRow {
  model: string; backend?: string; profile: string; concurrency?: number; isl: number; osl: number;
  data_scope?: DataScope;
  dataScope?: DataScope;
  calibration_status?: string;
  ttft_validation_scope?: string;
  total_context_tokens?: number;
  new_prefill_tokens?: number;
  cached_context_tokens?: number;
  cache_hit_rate?: number;
  cache_aware_applied?: boolean;
  cache_feature_source?: string;
  cache_prediction_regime?: string;
  ttft_prediction_supported?: boolean;
  unsupported_reason?: string;
  multiturn_prediction_mode?: string;
  predicted_turn_count?: number;
  total_successful_turn_requests?: number;
  mean_predicted_turn_ttft_ms?: number;
  mean_predicted_turn_tpot_ms?: number;
  multiturn_turn_predictions?: ServingTurnPrediction[];
  ttft_pred?: number; ttft_meas?: number; ttft_err?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  e2el_pred?: number; e2el_meas?: number; e2el_err?: number;
}

type ServingPerTurnRow = ServingRow & { multiturn_turn_predictions: ServingTurnPrediction[] };

interface ServingMatrixRow {
  key: string;
  model: string;
  backend?: string;
  profile: string;
  cells: Record<number, ServingRow>;
}

interface ServingProfileGroup {
  key: string;
  profile: string;
  backendRows: ServingMatrixRow[];
}

function scopedServingRows(
  data: Record<string, ServingRow[]>,
  gpu: string,
  dataScope: DataScope,
): ServingRow[] {
  return (data[gpu] ?? [])
    .map(row => ({ ...row, profile: normalizeProfileName(row.profile) }))
    .filter(row => (row.data_scope ?? row.dataScope ?? 'archive') === dataScope && isProfileInScope(row.profile, dataScope));
}

type ServingMetricKey =
  | 'ttft_pred' | 'ttft_meas' | 'ttft_err'
  | 'tpot_pred' | 'tpot_meas' | 'tpot_err'
  | 'e2el_pred' | 'e2el_meas' | 'e2el_err';

interface ServingMetric {
  label: string;
  description: string;
  color: string;
  predKey: ServingMetricKey;
  measKey: ServingMetricKey;
  errKey: ServingMetricKey;
  isTotal?: boolean;
}

const SERVING_METRICS: ServingMetric[] = [
  {
    label: 'TTFT',
    description: 'first token',
    color: '#f0883e',
    predKey: 'ttft_pred',
    measKey: 'ttft_meas',
    errKey: 'ttft_err',
  },
  {
    label: 'TPOT',
    description: 'per output token',
    color: '#58a6ff',
    predKey: 'tpot_pred',
    measKey: 'tpot_meas',
    errKey: 'tpot_err',
  },
  {
    label: 'E2EL',
    description: 'end-to-end',
    color: '#a855f7',
    predKey: 'e2el_pred',
    measKey: 'e2el_meas',
    errKey: 'e2el_err',
    isTotal: true,
  },
];

const SERVING_PROFILE_ORDER = [
  'chat-singleturn',
  'coding-singleturn',
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
  'chat-short',
  'chat-medium',
  'fixed-seq128',
  'prefill-heavy',
  'decode-heavy',
  'random-1k',
  'chat-multiturn-short',
  'chat-multiturn-medium',
  'chat-multiturn-long',
  'swebench-multiturn-short',
  'swebench-multiturn-medium',
  'swebench-multiturn-long',
  'terminalbench-multiturn-short',
  'terminalbench-multiturn-medium',
  'terminalbench-multiturn-long',
  'osworld-multiturn-short',
  'osworld-multiturn-medium',
  'osworld-multiturn-long',
];

function servingProfileRank(profile: string): number {
  const index = SERVING_PROFILE_ORDER.indexOf(normalizeProfileName(profile));
  if (index >= 0) return index;
  if (profile.includes('multiturn')) return 1000;
  return 500;
}

function servingRowKey(row: ServingRow): string {
  return `${row.model}|${row.backend ?? ''}|${row.profile}|${row.concurrency ?? 1}`;
}

function compareServingRows(a: ServingRow, b: ServingRow): number {
  const modelOrder = a.model.localeCompare(b.model);
  if (modelOrder !== 0) return modelOrder;
  const profileOrder = servingProfileRank(a.profile) - servingProfileRank(b.profile);
  if (profileOrder !== 0) return profileOrder;
  const profileNameOrder = a.profile.localeCompare(b.profile);
  if (profileNameOrder !== 0) return profileNameOrder;
  const backendOrder = (a.backend ?? '').localeCompare(b.backend ?? '');
  if (backendOrder !== 0) return backendOrder;
  return (a.concurrency ?? 1) - (b.concurrency ?? 1);
}

function hasTurnPredictions(row: ServingRow): row is ServingPerTurnRow {
  return Array.isArray(row.multiturn_turn_predictions) && row.multiturn_turn_predictions.length > 0;
}

function ServingTable({
  gpu,
  dataScope,
}: {
  gpu: string;
  dataScope: DataScope;
}) {
  const [rows, setRows] = useState<ServingRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPerTurnKey, setSelectedPerTurnKey] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch(servingPredictionsJsonUrl)
      .then(r => r.json())
      .then((d: Record<string, ServingRow[]>) => {
        setRows(scopedServingRows(d, gpu, dataScope));
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [dataScope, gpu]);

  if (loading) return <div className="text-[#484f58] text-sm py-4 text-center">Loading...</div>;

  if (rows.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="text-[#484f58] text-sm mb-2">No {dataScope} serving predictions available yet</div>
        <div className="text-[#30363d] text-xs">
          Run <code className="bg-[#21262d] px-1 rounded">python3 -m llm_predict.validate</code> to generate predictions
        </div>
      </div>
    );
  }

  const concurrencies = Array.from(new Set(rows.map(r => r.concurrency ?? 1))).sort((a, b) => a - b);
  const matrixByKey: Record<string, ServingMatrixRow> = {};
  for (const row of rows) {
    const key = `${row.model}|${row.backend ?? ''}|${row.profile}`;
    if (!matrixByKey[key]) {
      matrixByKey[key] = {
        key,
        model: row.model,
        backend: row.backend,
        profile: row.profile,
        cells: {},
      };
    }
    matrixByKey[key].cells[row.concurrency ?? 1] = row;
  }

  const matrixRows = Object.values(matrixByKey).sort((a, b) => {
    const modelOrder = a.model.localeCompare(b.model);
    if (modelOrder !== 0) return modelOrder;
    const profileOrder = servingProfileRank(a.profile) - servingProfileRank(b.profile);
    if (profileOrder !== 0) return profileOrder;
    const profileNameOrder = a.profile.localeCompare(b.profile);
    if (profileNameOrder !== 0) return profileNameOrder;
    return (a.backend ?? '').localeCompare(b.backend ?? '');
  });
  const perTurnRows = rows.filter(hasTurnPredictions).sort(compareServingRows);
  const selectedPerTurnRow = perTurnRows.find(row => servingRowKey(row) === selectedPerTurnKey) ?? perTurnRows[0];
  const selectedPerTurnRowKey = selectedPerTurnRow ? servingRowKey(selectedPerTurnRow) : null;

  const profileGroupsByModel: Record<string, Record<string, ServingProfileGroup>> = {};
  for (const row of matrixRows) {
    if (!profileGroupsByModel[row.model]) profileGroupsByModel[row.model] = {};
    const profileGroups = profileGroupsByModel[row.model];
    if (!profileGroups[row.profile]) {
      profileGroups[row.profile] = {
        key: `${row.model}|${row.profile}`,
        profile: row.profile,
        backendRows: [],
      };
    }
    profileGroups[row.profile].backendRows.push(row);
  }

  const groupedByModel: Record<string, ServingProfileGroup[]> = {};
  for (const [model, groups] of Object.entries(profileGroupsByModel)) {
    groupedByModel[model] = Object.values(groups)
      .map(group => ({
        ...group,
        backendRows: [...group.backendRows].sort((a, b) => (a.backend ?? '').localeCompare(b.backend ?? '')),
      }))
      .sort((a, b) => {
        const rankOrder = servingProfileRank(a.profile) - servingProfileRank(b.profile);
        if (rankOrder !== 0) return rankOrder;
        return a.profile.localeCompare(b.profile);
      });
  }

  return (
    <div className="space-y-3">
      <div className="grid overflow-hidden rounded-md border border-[#21262d] bg-[#161b22] md:grid-cols-3 md:divide-x md:divide-[#21262d]">
        {SERVING_METRICS.map(metric => (
          <ServingMetricSummary key={metric.label} metric={metric} rows={rows} />
        ))}
      </div>

      <div className="overflow-x-auto rounded-md border border-[#21262d] bg-[#161b22]">
        <table
          className="w-full table-fixed border-collapse text-xs"
          style={{ minWidth: `${310 + concurrencies.length * 82}px` }}
        >
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="w-[210px] px-3 py-2 text-left font-medium">Profile</th>
              <th className="w-[72px] px-2 py-2 text-left font-medium">Backend</th>
              {concurrencies.map(concurrency => (
                <th key={concurrency} className="px-1.5 py-2 text-center font-mono font-normal">
                  {concurrency}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(groupedByModel).map(([model, profileGroups]) => (
              <Fragment key={model}>
                <tr className="border-b-2 border-t-2 border-[#30363d] bg-[#0d1117]">
                  <td colSpan={2 + concurrencies.length} className="px-3 py-1.5">
                    <span className="font-mono text-sm font-semibold text-[#c9d1d9]">{model}</span>
                    <span className="ml-2 text-[10px] text-[#6e7681]">{profileGroups.length} profiles</span>
                  </td>
                </tr>
                {profileGroups.map(group => (
                  <Fragment key={group.key}>
                    {group.backendRows.map((row, backendIndex) => (
                      <tr key={row.key} className="border-b border-[#21262d]/50 transition-colors hover:bg-[#0d1117]">
                        {backendIndex === 0 && (
                          <td rowSpan={group.backendRows.length} className="border-r border-[#21262d]/50 px-3 py-1.5 align-middle">
                            <div className="flex min-w-[190px] items-center gap-1.5">
                              <span className="truncate text-[11px] text-[#c9d1d9]" title={profileDisplayName(group.profile)}>
                                {profileDisplayName(group.profile)}
                              </span>
                            </div>
                          </td>
                        )}
                        <td className="px-2 py-1.5 align-middle">
                          {row.backend && <span className="text-[9px] uppercase text-[#6e7681]">{row.backend}</span>}
                        </td>
                        {concurrencies.map(concurrency => (
                          <ServingMatrixCell
                            key={concurrency}
                            row={row.cells[concurrency]}
                            selectedKey={selectedPerTurnRowKey}
                            onSelectPerTurn={setSelectedPerTurnKey}
                          />
                        ))}
                      </tr>
                    ))}
                  </Fragment>
                ))}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>

      <ServingPerTurnBreakdown row={selectedPerTurnRow} />

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#6e7681]">
        <span>Cells show % error left-to-right: <span className="text-[#f0883e]">TTFT</span> / <span className="text-[#58a6ff]">TPOT</span> / <span className="text-[#a855f7]">E2EL</span>.</span>
        <span className="font-medium text-[#8b949e]">Error bands:</span>
        <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">&lt;10%</span>
        <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 text-[#58a6ff]">10-25%</span>
        <span className="rounded border border-[#f0883e]/30 bg-[#f0883e]/10 px-2 py-0.5 text-[#f0883e]">25-50%</span>
        <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">&gt;=50%</span>
      </div>
    </div>
  );
}

function ServingPerTurnBreakdown({ row }: { row?: ServingPerTurnRow }) {
  if (!row) return null;

  const turns = [...row.multiturn_turn_predictions].sort((a, b) => a.turn_index - b.turn_index);
  const meanHit = turns.length
    ? turns.reduce((total, turn) => total + turn.cache_hit_rate, 0) / turns.length
    : 0;

  return (
    <div className="rounded-md border border-[#21262d] bg-[#161b22]">
      <div className="flex flex-col gap-3 border-b border-[#21262d] px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm font-semibold text-[#e6edf3]">Per-Turn Multi-Turn Prediction</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-[#6e7681]">
            <span className="font-mono text-[#8b949e]">{row.model}</span>
            <span>{row.backend ?? 'backend'}</span>
            <span>{profileDisplayName(row.profile)}</span>
            <span>c{row.concurrency ?? 1}</span>
            <span>{turns.length} turns</span>
            <span>{row.total_successful_turn_requests ?? 0} successful turn requests</span>
            <span>mean cache hit {(meanHit * 100).toFixed(0)}%</span>
            <span>mean TTFT {formatLatency(row.mean_predicted_turn_ttft_ms)}</span>
            <span>mean TPOT {formatLatency(row.mean_predicted_turn_tpot_ms)}</span>
          </div>
        </div>
        <div className="rounded border border-[#30363d] bg-[#0d1117] px-2 py-1 font-mono text-[10px] text-[#79c0ff]">
          selected from serving table
        </div>
      </div>

      <div className="overflow-x-auto border-b border-[#21262d]">
        <div className="flex min-w-max gap-2 p-3">
          {turns.map(turn => (
            <div key={turn.turn_index} className="w-[122px] shrink-0 rounded border border-[#21262d] bg-[#0d1117] p-2">
              <div className="mb-2 flex items-center justify-between">
                <span className="font-mono text-[11px] font-semibold text-[#c9d1d9]">Turn {displayTurn(turn)}</span>
                <span className="text-[10px] text-[#6e7681]">{turn.successful} req</span>
              </div>
              <ServingTurnCacheBar turn={turn} compact />
              <div className="mt-2 space-y-1">
                {SERVING_METRICS.map(metric => (
                  <ServingTurnErrorBadge key={metric.label} turn={turn} metric={metric} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full min-w-[1120px] border-collapse text-xs">
          <thead>
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-3 py-2 text-left font-medium">Turn</th>
              <th className="px-2 py-2 text-right font-medium">Req</th>
              <th className="px-2 py-2 text-right font-medium">Ctx</th>
              <th className="px-2 py-2 text-right font-medium">New</th>
              <th className="px-2 py-2 text-right font-medium">Cached</th>
              <th className="w-[150px] px-2 py-2 text-left font-medium">Hit</th>
              <th className="px-2 py-2 text-right font-medium">Out</th>
              {SERVING_METRICS.map(metric => (
                <th key={metric.label} className="w-[170px] px-2 py-2 text-left font-medium" style={{ color: metric.color }}>
                  {metric.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {turns.map(turn => (
              <tr key={turn.turn_index} className="border-b border-[#21262d]/50 hover:bg-[#0d1117]">
                <td className="px-3 py-2 font-mono text-[#c9d1d9]">{displayTurn(turn)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.successful)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.total_context_tokens)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.new_prefill_tokens)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.cached_context_tokens)}</td>
                <td className="px-2 py-2"><ServingTurnCacheBar turn={turn} /></td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.output_tokens)}</td>
                {SERVING_METRICS.map(metric => (
                  <ServingTurnMetricCell key={metric.label} turn={turn} metric={metric} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ServingMetricSummary({ metric, rows }: { metric: ServingMetric; rows: ServingRow[] }) {
  const values = rows
    .map(row => numericMetric(row, metric.errKey))
    .filter((value): value is number => value !== undefined);
  const med = values.length ? median(values) : undefined;
  const best = values.length ? Math.min(...values) : undefined;
  const worst = values.length ? Math.max(...values) : undefined;

  return (
    <div className="border-b border-[#21262d] px-3 py-2.5 last:border-b-0 md:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: metric.color }}>{metric.label}</div>
          <div className="mt-0.5 text-[11px] text-[#6e7681]">{metric.description}</div>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold text-[#e6edf3]">{formatPercent(med)}</div>
          <div className="text-[10px] text-[#6e7681]">median error</div>
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-[#21262d] pt-2 text-[10px] text-[#6e7681]">
        <span>{values.length} rows</span>
        <span>best {formatPercent(best)} / worst {formatPercent(worst)}</span>
      </div>
    </div>
  );
}

function ServingTurnCacheBar({ turn, compact = false }: { turn: ServingTurnPrediction; compact?: boolean }) {
  const pct = Math.max(0, Math.min(100, turn.cache_hit_rate * 100));
  return (
    <div className={compact ? 'space-y-1' : 'flex items-center gap-2'}>
      <span className={compact ? 'block text-[9px] uppercase text-[#6e7681]' : 'w-8 text-[10px] uppercase text-[#6e7681]'}>Hit</span>
      <div className="relative h-4 flex-1 overflow-hidden rounded bg-[#21262d]">
        <div className="h-full rounded bg-[#58a6ff]/70" style={{ width: `${pct}%` }} />
        <span className="absolute inset-0 flex items-center justify-center font-mono text-[10px] text-[#e6edf3]">
          {pct.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

function ServingTurnErrorBadge({ turn, metric }: { turn: ServingTurnPrediction; metric: ServingMetric }) {
  const err = numericTurnMetric(turn, metric.errKey);
  const tone = servingErrorTone(err);
  return (
    <div className="grid grid-cols-[34px_1fr] items-center gap-1">
      <span className="text-[9px] font-semibold uppercase" style={{ color: metric.color }}>{metric.label}</span>
      <span className={`rounded px-1.5 py-0.5 text-right font-mono text-[10px] leading-none ${tone.className}`}>
        {formatCompactPercent(err)}
      </span>
    </div>
  );
}

function ServingTurnMetricCell({ turn, metric }: { turn: ServingTurnPrediction; metric: ServingMetric }) {
  const pred = numericTurnMetric(turn, metric.predKey);
  const meas = numericTurnMetric(turn, metric.measKey);
  const err = numericTurnMetric(turn, metric.errKey);
  const tone = servingErrorTone(err);
  return (
    <td className="px-2 py-2 align-top">
      <div className="space-y-1">
        <MetricLine label="Pred" value={formatLatency(pred, metric.isTotal)} />
        <MetricLine label="Actual" value={formatLatency(meas, metric.isTotal)} />
        <div className="flex items-center justify-between gap-2">
          <span className="text-[10px] text-[#6e7681]">Err</span>
          <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${tone.className}`}>
            {formatPercent(err)}
          </span>
        </div>
      </div>
    </td>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-[10px] text-[#6e7681]">{label}</span>
      <span className="font-mono text-[10px] text-[#c9d1d9]">{value}</span>
    </div>
  );
}

function ServingMatrixCell({
  row,
  selectedKey,
  onSelectPerTurn,
}: {
  row?: ServingRow;
  selectedKey: string | null;
  onSelectPerTurn: (key: string) => void;
}) {
  if (!row) {
    return (
      <td className="border-l border-[#21262d]/50 px-1.5 py-1 text-center">
        <span className="text-[#30363d]">·</span>
      </td>
    );
  }
  const canSelect = hasTurnPredictions(row);
  const rowKey = servingRowKey(row);
  const selected = canSelect && rowKey === selectedKey;
  return (
    <td
      onClick={canSelect ? () => onSelectPerTurn(rowKey) : undefined}
      className={`border-l border-[#21262d]/50 px-1 py-0.5 align-middle transition-colors ${
        canSelect ? 'cursor-pointer hover:bg-[#1f6feb]/10' : ''
      } ${selected ? 'bg-[#1f6feb]/10 shadow-[inset_0_0_0_1px_#58a6ff]' : ''}`}
      title={canSelect ? `Show ${row.multiturn_turn_predictions.length} per-turn predictions` : undefined}
    >
      <div className="min-w-0 space-y-0.5" title={`ISL→OSL ${row.isl}→${row.osl}`}>
        <div className="grid min-w-0 grid-cols-3 gap-0.5">
          {SERVING_METRICS.map(metric => (
            <ServingMiniMetric key={metric.label} row={row} metric={metric} />
          ))}
        </div>
      </div>
    </td>
  );
}

function ServingMiniMetric({ row, metric }: { row: ServingRow; metric: ServingMetric }) {
  const pred = numericMetric(row, metric.predKey);
  const meas = numericMetric(row, metric.measKey);
  const err = numericMetric(row, metric.errKey);
  const tone = servingErrorTone(err);
  const title = [
    `${metric.label}: ${formatPercent(err)} error`,
    `pred ${formatLatency(pred, metric.isTotal)}`,
    `meas ${formatLatency(meas, metric.isTotal)}`,
    `ISL→OSL ${row.isl}→${row.osl}`,
    cacheTooltip(row),
  ].join(' · ');

  return (
    <span
      title={title}
      className={`block rounded px-1 py-0.5 text-center font-mono text-[9px] leading-none ${tone.className}`}
    >
      {formatCompactPercent(err)}
    </span>
  );
}

function cacheTooltip(row: ServingRow): string {
  if (row.cache_prediction_regime === 'unknown_prefix_cache') {
    return `prefix cache features missing${row.unsupported_reason ? `; ${row.unsupported_reason}` : ''}`;
  }
  if (!row.cache_aware_applied) return 'full prefill';
  const hit = row.cache_hit_rate === undefined ? 'n/a' : `${(row.cache_hit_rate * 100).toFixed(0)}%`;
  const total = row.total_context_tokens ?? row.isl;
  const fresh = row.new_prefill_tokens ?? total;
  const cached = row.cached_context_tokens ?? Math.max(0, total - fresh);
  const source = row.cache_feature_source ? `; source ${row.cache_feature_source}` : '';
  const multiturn = row.multiturn_prediction_mode
    ? `; ${row.multiturn_prediction_mode} ${row.predicted_turn_count ?? 0} turns`
    : '';
  return `cache hit ${hit}; new/full ${fresh}/${total}; cached ${cached}${source}${multiturn}`;
}

function numericMetric(row: ServingRow, key: ServingMetricKey): number | undefined {
  const value = row[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function numericTurnMetric(turn: ServingTurnPrediction, key: ServingMetricKey): number | undefined {
  const value = turn[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function servingErrorTone(err: number | undefined): { className: string } {
  if (err === undefined) return { className: 'border border-[#30363d] bg-[#21262d] text-[#6e7681]' };
  const value = Math.abs(err);
  if (value < 10) return { className: 'border border-[#3fb950]/30 bg-[#3fb950]/10 text-[#3fb950]' };
  if (value < 25) return { className: 'border border-[#58a6ff]/30 bg-[#58a6ff]/10 text-[#58a6ff]' };
  if (value < 50) return { className: 'border border-[#f0883e]/30 bg-[#f0883e]/10 text-[#f0883e]' };
  return { className: 'border border-[#f85149]/30 bg-[#f85149]/10 text-[#f85149]' };
}

function formatLatency(value: number | undefined, isTotal?: boolean): string {
  if (value === undefined) return 'n/a';
  return `${isTotal ? value.toFixed(0) : value.toFixed(1)} ms`;
}

function formatPercent(value: number | undefined): string {
  if (value === undefined) return 'n/a';
  return `${value.toFixed(1)}%`;
}

function formatCompactPercent(value: number | undefined): string {
  if (value === undefined) return '-';
  return `${value.toFixed(0)}%`;
}

function formatTokenCount(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '-';
  return Math.round(value).toLocaleString();
}

function displayTurn(turn: ServingTurnPrediction): number {
  return turn.turn_index + 1;
}

function median(arr: number[]): number {
  if (!arr.length) return 0;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}
