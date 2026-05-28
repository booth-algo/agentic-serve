import { Fragment, useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  DATA_SCOPE_META,
  type DataScope,
  isProfileInScope,
  normalizeDataScope,
  normalizeProfileName,
  profileDisplayName,
} from '../profileMeta';
import { llama31H100TpotFitJsonUrl, servingPredictionsJsonUrl } from '../dataUrls';

interface ServingTurnPrediction {
  turn_index: number;
  successful: number;
  total_context_tokens: number;
  new_prefill_tokens: number;
  cached_context_tokens: number;
  cache_hit_rate: number;
  output_tokens: number;
  backend_trace_summary?: BackendTraceSummary;
  backend_cache_work?: BackendCacheWork;
  backend_step_trace?: BackendStepTrace[];
  ttft_pred?: number; ttft_meas?: number; ttft_err?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  tpot_pred_llm_d?: number;
  tpot_pred_two_roofline?: number;
  tpot_signed_err_ms?: number; tpot_abs_err_ms?: number;
  base_tpot_signed_err_ms?: number; base_tpot_abs_err_ms?: number;
  e2el_pred?: number; e2el_meas?: number; e2el_err?: number;
  scheduled_requests?: number;
  base_tpot_pred?: number;
  decode_waves?: number;
  decode_wave_token_pressure?: number;
  max_wave_batch?: number;
  batch_utilization?: number;
  scheduled_utilization?: number;
  continuous_batching_mode?: string;
  scheduling_regime?: string;
  turn_position_bin?: string;
  context_cache_regime?: string;
  decode_load_regime?: string;
  workload_regime?: string;
  turn_batching_regime?: string;
  startup_prefill_token_budget_scale?: number;
  steady_state_ttft_ms?: number;
  steady_state_request_count?: number;
}

interface BackendTraceSummary {
  total_steps?: number;
  max_decode_batch?: number;
  mean_decode_batch?: number;
  max_active_requests?: number;
  max_waiting_requests?: number;
  total_prefill_tokens?: number;
  total_decode_tokens?: number;
  scheduler_overhead_ms?: number;
  effective_prefill_tokens?: number;
  realized_cached_tokens?: number;
  replayed_cached_tokens?: number;
  evicted_cached_tokens?: number;
  logical_cached_tokens?: number;
  cache_pressure?: number;
}

interface BackendCacheWork {
  effective_prefill_tokens?: number;
  realized_cached_tokens?: number;
  replayed_cached_tokens?: number;
  evicted_cached_tokens?: number;
  logical_cached_tokens?: number;
  cache_pressure?: number;
}

interface BackendStepTrace {
  step_index: number;
  wall_start_ms: number;
  step_ms: number;
  scheduler_overhead_ms: number;
  prefill_tokens: number;
  decode_batch: number;
  active_requests: number;
  waiting_requests: number;
  max_context_tokens: number;
  kv_resident_tokens: number;
}

interface BackendSpecSummary {
  name?: string;
  max_num_batched_tokens?: number;
  max_num_seqs?: number;
  prefill_policy?: string;
  decode_policy?: string;
  cache_mode?: string;
  cache_realization_rate?: number;
  kv_block_tokens?: number;
  kv_budget_tokens?: number;
}

interface ServingRow {
  model: string; backend?: string; profile: string; concurrency?: number; isl: number; osl: number;
  data_scope?: string;
  dataScope?: string;
  mode?: string;
  total_context_tokens?: number;
  new_prefill_tokens?: number;
  cached_context_tokens?: number;
  cache_hit_rate?: number;
  cache_aware_applied?: boolean;
  cache_feature_source?: string;
  cache_prediction_regime?: string;
  unsupported_reason?: string;
  measurement_semantics_warning?: string;
  multiturn_prediction_mode?: string;
  predicted_turn_count?: number;
  total_successful_turn_requests?: number;
  scheduled_request_count?: number;
  mean_predicted_turn_ttft_ms?: number;
  mean_predicted_turn_tpot_ms?: number;
  continuous_batching_mode?: string;
  backend_emulator_status?: string;
  backend_spec?: BackendSpecSummary;
  backend_trace_summary?: BackendTraceSummary;
  kernel_source_summary?: Record<string, number>;
  multiturn_turn_predictions?: ServingTurnPrediction[];
  ttft_pred?: number; ttft_meas?: number; ttft_err?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  tpot_signed_err_ms?: number; tpot_abs_err_ms?: number; tpot_max_abs_err_ms?: number;
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

interface GpuConfigSummary {
  gpu: string;
  rows: number;
  models: number;
  profiles: number;
  backends: number;
  concurrencies: number;
  meanTtftMape?: number;
  meanTpotMape?: number;
  meanE2elMape?: number;
}

interface ServingScopeIndex {
  rowsByGpu: Record<string, ServingRow[]>;
  gpuOptions: string[];
  summaries: GpuConfigSummary[];
  summariesByGpu: Record<string, GpuConfigSummary>;
}

interface ServingIndex {
  trace_replay: ServingScopeIndex;
  synthetic_distributional: ServingScopeIndex;
  archived: ServingScopeIndex;
}

interface ServingFocus {
  gpu: string;
  model: string;
  title: string;
  description: string;
  profiles?: string[];
}

type OptionalMetric = number | null | undefined;

interface FixedTpotFitData {
  experiment: {
    name: string;
    model: string;
    gpu: string;
    backend: string;
    target: string;
    scope_note: string;
    dashboard_scope: DataScope;
  };
  fit_summary: {
    rows: number;
    physics_loo_mape?: number;
    physics_loo_median_ape?: number;
    physics_loo_max_ape?: number;
    interp_loo_mape?: number;
    interp_loo_median_ape?: number;
    interp_loo_max_ape?: number;
    physics_in_sample_mape?: number;
    kernel_composed_mape?: number;
    kernel_composed_median_ape?: number;
    kernel_composed_max_ape?: number;
    trace_cross_check_mape?: number;
    trace_cross_check_median_ape?: number;
    trace_cross_check_max_ape?: number;
    small_kernel_exact_rows?: number;
    small_kernel_missing_rows?: number;
    small_kernel_component_count?: number;
    attention_scale?: number;
    dense_by_batch_ms?: Record<string, number>;
  };
  dashboard_comparison: FixedTpotDashboardComparison[];
  page_comparisons?: Partial<Record<PredictionPageKind, FixedTpotDashboardComparison[]>>;
  sources: Record<string, string>;
  worst_rows?: {
    physics_loo?: FixedTpotWorstRow[];
    interpolation_loo?: FixedTpotWorstRow[];
  };
}

type PredictionPageKind = 'serving' | 'simulator';

interface FixedTpotDashboardComparison {
  backend: string;
  label?: string;
  rows: number;
  ttft_mape?: OptionalMetric;
  ttft_median_ape?: OptionalMetric;
  ttft_max_ape?: OptionalMetric;
  tpot_mape?: OptionalMetric;
  tpot_median_ape?: OptionalMetric;
  tpot_max_ape?: OptionalMetric;
  e2el_mape?: OptionalMetric;
  e2el_median_ape?: OptionalMetric;
  e2el_max_ape?: OptionalMetric;
}

interface FixedTpotWorstRow {
  batch_size: number;
  context_len: number;
  actual_ms: number;
  physics_loo_pred_ms: number;
  physics_loo_pct_error: number;
  interp_loo_pred_ms: number;
  interp_loo_pct_error: number;
}

const EMPTY_GPU_OPTIONS: string[] = [];

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

const SERVING_PROFILE_ORDER = [
  'chat-singleturn',
  'coding-singleturn',
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
  'chat-singleturn-synth',
  'chat-multiturn-synth',
  'swebench-multiturn-synth',
  'terminalbench-multiturn-synth',
  'osworld-multiturn-synth',
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
const SERVING_TPOT_METRIC = SERVING_METRICS[1];
const SERVING_MAPE_COLUMN_WIDTH = 74;
const SERVING_MAPE_RAIL_WIDTH = SERVING_METRICS.length * SERVING_MAPE_COLUMN_WIDTH;

export function ServingPredictionsPage({
  dataScope,
  focus,
  predictionsUrl = servingPredictionsJsonUrl,
  pageKind = 'serving',
}: {
  dataScope: DataScope;
  focus?: ServingFocus;
  predictionsUrl?: string;
  pageKind?: PredictionPageKind;
}) {
  const [servingIndex, setServingIndex] = useState<ServingIndex | null>(null);
  const [fixedTpotFit, setFixedTpotFit] = useState<FixedTpotFitData | null>(null);
  const [gpu, setGpu] = useState('H100');
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

  useEffect(() => {
    fetch(llama31H100TpotFitJsonUrl)
      .then(response => response.ok ? response.json() : null)
      .then((json: FixedTpotFitData | null) => setFixedTpotFit(json))
      .catch(() => setFixedTpotFit(null));
  }, []);

  const scopeIndex = servingIndex?.[dataScope];
  const gpuOptions = scopeIndex?.gpuOptions ?? EMPTY_GPU_OPTIONS;
  const selectedGpu = focus?.gpu ?? gpu;

  useEffect(() => {
    if (focus?.gpu) return;
    if (!gpuOptions.length) return;
    setGpu(current => gpuOptions.includes(current) ? current : gpuOptions[0]);
  }, [focus?.gpu, gpuOptions]);

  const rows = useMemo(
    () => applyServingFocus(scopeIndex?.rowsByGpu[selectedGpu] ?? [], focus),
    [scopeIndex, selectedGpu, focus],
  );
  const showFixedTpotFit = fixedTpotFit
    && fixedTpotFit.experiment.gpu === selectedGpu
    && fixedTpotFit.experiment.dashboard_scope === dataScope
    && (!focus || focus.model === fixedTpotFit.experiment.model);
  const fixedTpotOnly = Boolean(showFixedTpotFit && pageKind === 'simulator');
  const tableSourceRows = useMemo(
    () => fixedTpotOnly ? rows.filter(row => !isSingleTurnServingRow(row)) : rows,
    [fixedTpotOnly, rows],
  );
  const fixedTpotRows = fixedTpotOnly && fixedTpotFit
    ? fixedTpotServingRows(fixedTpotFit, pageKind)
    : undefined;
  const useFixedTpotRows = Boolean(fixedTpotRows && tableSourceRows.length === 0);
  const tableRows = useFixedTpotRows && fixedTpotRows ? fixedTpotRows : tableSourceRows;
  const tableSummaryRows = fixedTpotRows ?? tableRows;
  const tableSummaryRowCount = fixedTpotRows ? fixedTpotFit?.fit_summary.rows : undefined;

  if (loading) return <div className="p-8 text-[#8b949e]">Loading predictions...</div>;
  if (failed || !scopeIndex) return <div className="p-8 text-[#f85149]">Failed to load predictions JSON</div>;

  return (
    <div className="space-y-4">
      <div className="border-b border-[#21262d] pb-4">
        <div>
          <h2 className="text-lg font-semibold text-[#e6edf3]">{focus?.title ?? 'Predictions'}</h2>
          <p className="mt-1 max-w-3xl text-xs text-[#8b949e]">
            {focus?.description ?? `High-concurrency predictions vs measured benchmark results from ${DATA_SCOPE_META[dataScope].label.toLowerCase()}.`}
            Multi-turn TTFT reflects cache-aware serving behavior, not cumulative full-prefill latency.
          </p>
        </div>
      </div>

      {focus ? (
        <ServingFocusSummary
          rows={rows}
          focus={focus}
          dataScope={dataScope}
          fixedTpotFit={fixedTpotOnly && fixedTpotFit ? fixedTpotFit : undefined}
          pageKind={pageKind}
        />
      ) : (
        <GpuConfigSelector
          scopeIndex={scopeIndex}
          selectedGpu={gpu}
          onSelect={setGpu}
        />
      )}

      {showFixedTpotFit && <FixedTpotFitPanel data={fixedTpotFit} pageKind={pageKind} />}

      <ServingTable
        rows={tableRows}
        summaryRows={tableSummaryRows}
        summaryRowCount={tableSummaryRowCount}
        dataScope={dataScope}
        focus={focus}
        tpotOnly={fixedTpotOnly}
        validationRows={useFixedTpotRows}
      />
    </div>
  );
}

function FixedTpotFitPanel({
  data,
  pageKind,
}: {
  data: FixedTpotFitData;
  pageKind: PredictionPageKind;
}) {
  const fit = data.fit_summary;
  const comparisonRows = data.page_comparisons?.[pageKind] ?? data.dashboard_comparison;
  const primaryServing = comparisonRows.find(row => row.backend === data.experiment.backend)
    ?? comparisonRows[0];
  const secondaryComparisonRows = comparisonRows.filter(row => row.backend !== data.experiment.backend);
  const worstPhysics = data.worst_rows?.physics_loo ?? [];
  const comparisonLabel = pageKind === 'simulator' ? 'simulator' : 'serving';
  const kernelMape = fit.kernel_composed_mape ?? fit.physics_loo_mape;
  const kernelMedianApe = fit.kernel_composed_median_ape ?? fit.physics_loo_median_ape;
  const kernelMaxApe = fit.kernel_composed_max_ape ?? fit.physics_loo_max_ape;
  const traceMape = fit.trace_cross_check_mape ?? fit.interp_loo_mape;
  const primaryLabel = primaryServing?.label ?? primaryServing?.backend ?? 'kernel-composed';
  const smallKernelRows = fit.small_kernel_exact_rows !== undefined && fit.small_kernel_missing_rows !== undefined
    ? `${fit.small_kernel_exact_rows}/${fit.rows}`
    : fit.small_kernel_component_count !== undefined
      ? `${fit.small_kernel_component_count} components`
    : 'partial';
  const smallKernelSubvalue = fit.small_kernel_component_count !== undefined
    ? 'source-of-truth component models'
    : 'exact rows in current profile';

  return (
    <section className="rounded-md border border-[#21262d] bg-[#161b22]">
      <div className="flex flex-col gap-3 border-b border-[#21262d] px-4 py-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">Fixed decode-step experiment</div>
          <div className="mt-1 text-sm font-semibold text-[#e6edf3]">{data.experiment.name}</div>
          <p className="mt-1 max-w-4xl text-xs text-[#8b949e]">{data.experiment.scope_note}</p>
        </div>
        <div className="grid gap-2 sm:grid-cols-2 lg:min-w-[360px]">
          <MetricBadge label="Kernel TPOT" value={kernelMape} />
          <MetricBadge label={fit.trace_cross_check_mape === undefined ? 'Kernel Baseline' : 'Trace Check'} value={traceMape} />
        </div>
      </div>

      <div className="grid gap-3 p-4 lg:grid-cols-[1fr_1.2fr]">
        <div className="grid gap-2 sm:grid-cols-2">
          <FixedTpotStat
            label="Validation Rows"
            value={fit.rows.toLocaleString()}
            subvalue={data.experiment.target}
          />
          <FixedTpotStat
            label="Kernel Median"
            value={formatPercent(kernelMedianApe)}
            subvalue="median APE"
          />
          <FixedTpotStat
            label="TPOT MAPE"
            value={formatPercent(primaryServing?.tpot_mape)}
            subvalue={primaryLabel}
          />
          <FixedTpotStat
            label="Small Kernels"
            value={smallKernelRows}
            subvalue={smallKernelSubvalue}
          />
        </div>

        <div className="overflow-hidden rounded border border-[#21262d] bg-[#0d1117]">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="border-b border-[#21262d] text-[#8b949e]">
                <th className="px-3 py-2 text-left font-medium">Source</th>
                <th className="px-2 py-2 text-right font-medium">Rows</th>
                <th className="px-2 py-2 text-right font-medium">TTFT MAPE</th>
                <th className="px-2 py-2 text-right font-medium">TPOT MAPE</th>
                <th className="px-2 py-2 text-right font-medium">TPOT Median</th>
                <th className="px-2 py-2 text-right font-medium">TPOT Worst</th>
                <th className="px-2 py-2 text-right font-medium">E2EL MAPE</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-[#21262d]/60">
                <td className="px-3 py-2 text-[#c9d1d9]">{primaryLabel}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{fit.rows}</td>
                <td className="px-2 py-2 text-right font-mono text-[#6e7681]">N/A</td>
                <td className="px-2 py-2 text-right font-mono text-[#3fb950]">{formatPercent(kernelMape)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(kernelMedianApe)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(kernelMaxApe)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#6e7681]">N/A</td>
              </tr>
              {secondaryComparisonRows.map(row => (
                <tr key={row.backend} className="border-b border-[#21262d]/60 last:border-b-0">
                  <td className="px-3 py-2 text-[#c9d1d9]">{row.label ?? `webpage ${comparisonLabel} ${row.backend}`}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{row.rows}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(row.ttft_mape)}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(row.tpot_mape)}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(row.tpot_median_ape)}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(row.tpot_max_ape)}</td>
                  <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatPercent(row.e2el_mape)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {worstPhysics.length > 0 && (
        <div className="border-t border-[#21262d] px-4 py-3">
          <div className="mb-2 text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">Worst kernel-composed rows</div>
          <div className="flex flex-wrap gap-2">
            {worstPhysics.slice(0, 5).map(row => (
              <span
                key={`${row.batch_size}-${row.context_len}`}
                className={`rounded border px-2 py-1 font-mono text-[10px] ${servingErrorTone(row.physics_loo_pct_error).className}`}
                title={`actual ${formatLatency(row.actual_ms)}; predicted ${formatLatency(row.physics_loo_pred_ms)}`}
              >
                B{row.batch_size} T{row.context_len}: {formatPercent(row.physics_loo_pct_error)}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function FixedTpotStat({
  label,
  value,
  subvalue,
}: {
  label: string;
  value: string;
  subvalue: string;
}) {
  return (
    <div className="rounded border border-[#21262d] bg-[#0d1117] px-3 py-2">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">{label}</div>
      <div className="mt-1 font-mono text-lg font-semibold text-[#e6edf3]">{value}</div>
      <div className="mt-0.5 text-[10px] text-[#8b949e]">{subvalue}</div>
    </div>
  );
}

function applyServingFocus(rows: ServingRow[], focus?: ServingFocus): ServingRow[] {
  if (!focus) return rows;
  const profileSet = focus.profiles
    ? new Set(focus.profiles.map(profile => normalizeProfileName(profile)))
    : null;
  return rows.filter(row => {
    if (row.model !== focus.model) return false;
    if (profileSet && !profileSet.has(normalizeProfileName(row.profile))) return false;
    return true;
  });
}

function isSingleTurnServingRow(row: ServingRow): boolean {
  const profile = normalizeProfileName(row.profile).replace('_', '-').toLowerCase();
  return profile.includes('singleturn') || profile.includes('single-turn') || row.mode === 'single-turn';
}

function fixedTpotServingRows(data: FixedTpotFitData, pageKind: PredictionPageKind): ServingRow[] {
  const comparisons = data.page_comparisons?.[pageKind] ?? data.dashboard_comparison;
  const fit = data.fit_summary;
  if (comparisons.length === 0) {
    return [{
      model: data.experiment.model,
      backend: data.experiment.backend,
      profile: 'kernel-composed TPOT',
      concurrency: fit.rows,
      isl: 0,
      osl: 1,
      data_scope: data.experiment.dashboard_scope,
      tpot_err: finiteMetric(fit.kernel_composed_mape) ?? finiteMetric(fit.physics_loo_mape),
    }];
  }

  return comparisons.map(comparison => ({
    model: data.experiment.model,
    backend: comparison.backend,
    profile: comparison.label ?? comparison.backend,
    concurrency: comparison.rows,
    isl: 0,
    osl: 1,
    data_scope: data.experiment.dashboard_scope,
    tpot_err: finiteMetric(comparison.tpot_mape),
  }));
}

function finiteMetric(value: OptionalMetric): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function ServingFocusSummary({
  rows,
  focus,
  dataScope,
  fixedTpotFit,
  pageKind,
}: {
  rows: ServingRow[];
  focus: ServingFocus;
  dataScope: DataScope;
  fixedTpotFit?: FixedTpotFitData;
  pageKind?: PredictionPageKind;
}) {
  const fixedComparison = fixedTpotFit
    ? (fixedTpotFit.page_comparisons?.[pageKind ?? 'serving'] ?? fixedTpotFit.dashboard_comparison)[0]
    : undefined;
  const summary = fixedComparison
    ? {
      gpu: focus.gpu,
      rows: fixedComparison.rows,
      models: 1,
      profiles: 1,
      backends: 1,
      concurrencies: 0,
      meanTtftMape: undefined,
      meanTpotMape: fixedComparison.tpot_mape ?? undefined,
      meanE2elMape: undefined,
    }
    : summarizeGpuConfig(focus.gpu, rows);
  const profiles = fixedComparison ? 1 : new Set(rows.map(row => row.profile)).size;
  const backends = fixedComparison
    ? [fixedComparison.label ?? fixedComparison.backend]
    : Array.from(new Set(rows.map(row => row.backend).filter(Boolean))).sort();
  const concurrencies = fixedComparison
    ? []
    : Array.from(new Set(rows.map(row => row.concurrency ?? 1))).sort((a, b) => a - b);
  const emulatorRows = fixedComparison ? [] : rows.filter(row => row.backend_emulator_status === 'event_loop_enabled');
  const steadyRows = fixedComparison ? [] : rows.filter(row => isSteadyStateRow(row));
  const replayedTokens = fixedComparison
    ? undefined
    : emulatorRows.reduce((total, row) => total + (row.backend_trace_summary?.replayed_cached_tokens ?? 0), 0);

  return (
    <section className="rounded-md border border-[#21262d] bg-[#161b22] p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">Focused target</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 font-mono text-[#79c0ff]">
              {focus.gpu}
            </span>
            <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 font-mono text-[#3fb950]">
              {focus.model}
            </span>
            <span className="text-[#6e7681]">{DATA_SCOPE_META[dataScope].label}</span>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <MetricBadge label="TTFT MAPE" value={summary.meanTtftMape} />
          <MetricBadge label="TPOT MAPE" value={summary.meanTpotMape} />
          <MetricBadge label="E2EL MAPE" value={summary.meanE2elMape} />
        </div>
      </div>

      <div className="mt-3 grid gap-2 border-t border-[#21262d] pt-3 text-xs text-[#8b949e] sm:grid-cols-7">
        <FocusStat label="Rows" value={(fixedComparison?.rows ?? rows.length).toLocaleString()} />
        <FocusStat label="Profiles" value={profiles.toLocaleString()} />
        <FocusStat label="Backends" value={backends.length ? backends.join(', ') : '-'} />
        <FocusStat label="Concurrency" value={fixedComparison ? 'B/T grid' : formatConcurrencyRange(concurrencies)} />
        <FocusStat label="Emulator" value={fixedComparison ? 'N/A' : `${emulatorRows.length}/${rows.length}`} />
        <FocusStat label="Steady" value={fixedComparison ? 'N/A' : `${steadyRows.length}/${rows.length}`} />
        <FocusStat label="Replay" value={fixedComparison ? 'N/A' : formatTokenCount(replayedTokens)} />
      </div>
    </section>
  );
}

function FocusStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">{label}</div>
      <div className="mt-0.5 font-mono text-[#c9d1d9]">{value}</div>
    </div>
  );
}

function GpuConfigSelector({
  scopeIndex,
  selectedGpu,
  onSelect,
}: {
  scopeIndex: ServingScopeIndex;
  selectedGpu: string;
  onSelect: (gpu: string) => void;
}) {
  const selectedSummary = scopeIndex.summariesByGpu[selectedGpu];
  const groups = useMemo(
    () => groupGpuSummaries(scopeIndex.summaries),
    [scopeIndex.summaries],
  );

  return (
    <section className="space-y-3">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">GPU config</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
            <span className="font-mono text-sm font-semibold text-[#e6edf3]">{selectedGpu}</span>
            {selectedSummary && (
              <>
                <span className="text-[#6e7681]">{selectedSummary.rows} rows</span>
                <span className="text-[#6e7681]">{selectedSummary.models} models</span>
                <span className="text-[#6e7681]">{selectedSummary.profiles} profiles</span>
                <MetricBadge label="TTFT MAPE" value={selectedSummary.meanTtftMape} />
                <MetricBadge label="TPOT MAPE" value={selectedSummary.meanTpotMape} />
                <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${servingErrorTone(selectedSummary.meanE2elMape).className}`}>
                  E2EL MAPE {formatCompactPercent(selectedSummary.meanE2elMape)}
                </span>
              </>
            )}
          </div>
        </div>
        <div className="text-xs text-[#6e7681]">
          {scopeIndex.summaries.length} configs in {Object.keys(groups).length} families
        </div>
      </div>

      <div className="space-y-2 rounded-md border border-[#21262d] bg-[#161b22] p-2">
        {Object.entries(groups).map(([family, familySummaries]) => (
          <div key={family} className="flex flex-col gap-1.5 sm:flex-row sm:items-center">
            <div className="w-20 shrink-0 text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">
              {family}
            </div>
            <div className="flex flex-1 flex-wrap gap-1.5">
              {familySummaries.map(summary => (
                <GpuConfigButton
                  key={summary.gpu}
                  summary={summary}
                  selected={summary.gpu === selectedGpu}
                  onClick={() => onSelect(summary.gpu)}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function GpuConfigButton({
  summary,
  selected,
  onClick,
}: {
  summary: GpuConfigSummary;
  selected: boolean;
  onClick: () => void;
}) {
  const acceleratorCount = gpuAcceleratorCount(summary.gpu);

  return (
    <button
      onClick={onClick}
      className={`min-h-[72px] min-w-[146px] rounded-md border px-2.5 py-1.5 text-left transition-colors ${
        selected
          ? 'border-[#58a6ff] bg-[#1f6feb]/12 shadow-[inset_0_0_0_1px_rgba(88,166,255,0.35)]'
          : 'border-[#21262d] bg-[#161b22] hover:border-[#30363d] hover:bg-[#1c2129]'
      }`}
      title={`${summary.gpu}: TTFT MAPE ${formatPercent(summary.meanTtftMape)}, TPOT MAPE ${formatPercent(summary.meanTpotMape)}, E2EL MAPE ${formatPercent(summary.meanE2elMape)}`}
    >
      <div className="min-w-0">
        <div className="flex items-center justify-between gap-2">
          <div className="font-mono text-xs font-semibold text-[#e6edf3]">{summary.gpu}</div>
          <span className={`shrink-0 rounded px-1.5 py-0.5 font-mono text-[10px] ${servingErrorTone(summary.meanE2elMape).className}`}>
            E2EL {formatCompactPercent(summary.meanE2elMape)}
          </span>
        </div>
        <div className="mt-0.5 text-[9px] uppercase tracking-wide text-[#6e7681]">
          {acceleratorCount === 1 ? '1 GPU' : `${acceleratorCount} GPUs`} · {summary.models} models
        </div>
        <div className="mt-1.5 grid grid-cols-2 gap-1">
          <MetricBadge label="TTFT MAPE" value={summary.meanTtftMape} compact />
          <MetricBadge label="TPOT MAPE" value={summary.meanTpotMape} compact />
        </div>
      </div>
    </button>
  );
}

function summarizeGpuConfig(gpu: string, rows: ServingRow[]): GpuConfigSummary {
  return {
    gpu,
    rows: rows.length,
    models: new Set(rows.map(row => row.model)).size,
    profiles: new Set(rows.map(row => row.profile)).size,
    backends: new Set(rows.map(row => row.backend ?? '')).size,
    concurrencies: new Set(rows.map(row => row.concurrency ?? 1)).size,
    meanTtftMape: meanMetricError(rows, 'ttft_err'),
    meanTpotMape: meanMetricError(rows, 'tpot_err'),
    meanE2elMape: meanMetricError(rows, 'e2el_err'),
  };
}

function MetricBadge({
  label,
  value,
  compact = false,
}: {
  label: string;
  value: OptionalMetric;
  compact?: boolean;
}) {
  return (
    <span className={`inline-flex items-center justify-between gap-1 rounded px-1.5 py-0.5 font-mono ${compact ? 'text-[9px]' : 'text-[10px]'} ${servingErrorTone(value).className}`}>
      <span className="font-sans font-semibold uppercase tracking-wide">{label}</span>
      <span>{formatCompactPercent(value)}</span>
    </span>
  );
}

function meanMetricError(rows: ServingRow[], errKey: ServingMetricKey): number | undefined {
  const errors = rows
    .map(row => numericMetric(row, errKey))
    .filter((value): value is number => value !== undefined)
    .map(value => Math.abs(value));
  return errors.length ? mean(errors) : undefined;
}

function groupGpuSummaries(summaries: GpuConfigSummary[]): Record<string, GpuConfigSummary[]> {
  const groups: Record<string, GpuConfigSummary[]> = {};
  for (const summary of summaries) {
    const family = gpuFamily(summary.gpu);
    if (!groups[family]) groups[family] = [];
    groups[family].push(summary);
  }
  return groups;
}

function gpuFamily(gpu: string): string {
  if (gpu.startsWith('H100')) return 'H100';
  if (gpu.startsWith('A100')) return 'A100';
  if (gpu.startsWith('RTX3090')) return 'RTX 3090';
  if (gpu.startsWith('RTX2080Ti')) return 'RTX 2080 Ti';
  return 'Other';
}

function gpuAcceleratorCount(gpu: string): number {
  const match = gpu.match(/x(\d+)$/);
  return match ? Number(match[1]) : 1;
}

function compareServingGpus(a: string, b: string): number {
  const aRank = SERVING_GPU_ORDER.indexOf(a);
  const bRank = SERVING_GPU_ORDER.indexOf(b);
  const normalizedARank = aRank === -1 ? SERVING_GPU_ORDER.length : aRank;
  const normalizedBRank = bRank === -1 ? SERVING_GPU_ORDER.length : bRank;
  if (normalizedARank !== normalizedBRank) return normalizedARank - normalizedBRank;
  return a.localeCompare(b);
}

function createServingScopeIndex(): ServingScopeIndex {
  return {
    rowsByGpu: {},
    gpuOptions: [],
    summaries: [],
    summariesByGpu: {},
  };
}

function buildServingIndex(data: Record<string, ServingRow[]>): ServingIndex {
  const index: ServingIndex = {
    trace_replay: createServingScopeIndex(),
    synthetic_distributional: createServingScopeIndex(),
    archived: createServingScopeIndex(),
  };

  for (const [gpu, rows] of Object.entries(data)) {
    for (const row of rows) {
      const dataScope = normalizeDataScope(row.data_scope ?? row.dataScope ?? null) ?? 'archived';

      const profile = normalizeProfileName(row.profile);
      if (!isProfileInScope(profile, dataScope)) continue;

      const normalizedRow = profile === row.profile ? row : { ...row, profile };
      const rowsByGpu = index[dataScope].rowsByGpu;
      if (!rowsByGpu[gpu]) rowsByGpu[gpu] = [];
      rowsByGpu[gpu].push(normalizedRow);
    }
  }

  for (const scope of ['trace_replay', 'synthetic_distributional', 'archived'] as const) {
    const scopeIndex = index[scope];
    scopeIndex.gpuOptions = Object.keys(scopeIndex.rowsByGpu)
      .filter(gpu => scopeIndex.rowsByGpu[gpu].length > 0)
      .sort(compareServingGpus);
    scopeIndex.summaries = scopeIndex.gpuOptions.map(gpu => (
      summarizeGpuConfig(gpu, scopeIndex.rowsByGpu[gpu] ?? [])
    ));
    scopeIndex.summariesByGpu = Object.fromEntries(
      scopeIndex.summaries.map(summary => [summary.gpu, summary]),
    );
  }

  return index;
}

function ServingTable({
  rows,
  summaryRows,
  summaryRowCount,
  dataScope,
  focus,
  tpotOnly = false,
  validationRows = false,
}: {
  rows: ServingRow[];
  summaryRows?: ServingRow[];
  summaryRowCount?: number;
  dataScope: DataScope;
  focus?: ServingFocus;
  tpotOnly?: boolean;
  validationRows?: boolean;
}) {
  const [selectedPerTurnKey, setSelectedPerTurnKey] = useState<string | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<ServingMetric>(SERVING_TPOT_METRIC);
  const tableData = useMemo(() => {
    const concurrencies = Array.from(new Set(rows.map(r => r.concurrency ?? 1))).sort((a, b) => a - b);
    const matrixRows = buildServingMatrixRows(rows);
    const perTurnRows = rows.filter(hasTurnPredictions).sort(compareServingRows);
    const groupedByModel = groupServingRowsByModel(matrixRows);
    return { concurrencies, perTurnRows, groupedByModel };
  }, [rows]);
  const selectedPerTurnRow = useMemo(
    () => tableData.perTurnRows.find(row => servingRowKey(row) === selectedPerTurnKey) ?? tableData.perTurnRows[0],
    [selectedPerTurnKey, tableData.perTurnRows],
  );
  const selectedPerTurnRowKey = selectedPerTurnRow ? servingRowKey(selectedPerTurnRow) : null;

  if (rows.length === 0) {
    return (
      <div className="py-8 text-center">
        <div className="mb-2 text-sm text-[#484f58]">No {dataScope} predictions available yet</div>
        <div className="text-xs text-[#30363d]">
          {focus
            ? `Expected ${focus.gpu} / ${focus.model} rows in predictions JSON`
            : (
              <>Run <code className="rounded bg-[#21262d] px-1">python3 -m llm_predict.validate</code> to generate predictions</>
            )}
        </div>
      </div>
    );
  }

  const { concurrencies, groupedByModel } = tableData;
  const metricSummaryRows = summaryRows ?? rows;

  return (
    <div className="space-y-3">
      <div className="grid overflow-hidden rounded-md border border-[#21262d] bg-[#161b22] md:grid-cols-3 md:divide-x md:divide-[#21262d]">
        {SERVING_METRICS.map(metric => (
          <ServingMetricSummary
            key={metric.label}
            metric={metric}
            rows={metricSummaryRows}
            rowCount={summaryRowCount}
          />
        ))}
      </div>

      <div className="overflow-x-auto rounded-md border border-[#21262d] bg-[#161b22]">
        <table
          className="w-full table-fixed border-collapse text-xs"
          style={{ minWidth: `${310 + concurrencies.length * 82 + SERVING_METRICS.length * 74}px` }}
        >
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th rowSpan={2} className="w-[210px] px-3 py-2 text-left font-medium">Profile</th>
              <th rowSpan={2} className="w-[72px] px-2 py-2 text-left font-medium">Backend</th>
              <th colSpan={concurrencies.length} className="px-1.5 py-1.5 text-left text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">
                {validationRows ? 'Validation Rows' : 'Concurrency'}
              </th>
              <th
                colSpan={SERVING_METRICS.length}
                className="serving-mape-rail serving-mape-rail-start sticky z-30 px-2 py-1.5 text-left"
                style={{ right: 0, width: `${SERVING_MAPE_RAIL_WIDTH}px` }}
              >
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[#c9d1d9]">Row MAPE</span>
                  <span className="text-[9px] font-normal text-[#6e7681]">mean abs error</span>
                </div>
              </th>
            </tr>
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              {concurrencies.map(concurrency => (
                <th key={concurrency} className="px-1.5 py-2 text-center font-mono font-normal">
                  {concurrency}
                </th>
              ))}
              {SERVING_METRICS.map((metric, metricIndex) => (
                <th
                  key={`mean-${metric.label}`}
                  className={`serving-mape-rail sticky z-20 w-[74px] px-1.5 py-2 text-center font-mono text-[10px] font-semibold ${
                    metricIndex === 0 ? 'serving-mape-rail-start' : 'border-l border-[#1f2937]'
                  }`}
                  style={{ right: `${(SERVING_METRICS.length - metricIndex - 1) * SERVING_MAPE_COLUMN_WIDTH}px` }}
                  title={`Mean absolute ${metric.label} error across displayed concurrencies`}
                >
                  {metric.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(groupedByModel).map(([model, profileGroups]) => (
              <Fragment key={model}>
                <tr className="border-b-2 border-t-2 border-[#30363d] bg-[#0d1117]">
                  <td colSpan={2 + concurrencies.length + SERVING_METRICS.length} className="px-3 py-1.5">
                    <span className="font-mono text-sm font-semibold text-[#c9d1d9]">{model}</span>
                    <span className="ml-2 text-[10px] text-[#6e7681]">{profileGroups.length} profiles</span>
                  </td>
                </tr>
                {profileGroups.map(group => (
                  <Fragment key={group.key}>
                    {group.backendRows.map((row, backendIndex) => (
                      <tr key={row.key} className="group border-b border-[#21262d]/50 transition-colors hover:bg-[#0d1117]">
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
                          {row.backend && (
                            <div className="flex flex-col gap-1">
                              <span className="text-[9px] uppercase text-[#6e7681]">{row.backend}</span>
                              {matrixRowUsesBackendEmulator(row) && (
                                <span
                                  className="w-fit rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-1 py-0.5 font-mono text-[8px] uppercase leading-none text-[#3fb950]"
                                  title={backendTooltipForMatrixRow(row)}
                                >
                                  emu
                                </span>
                              )}
                              {matrixRowUsesSteadyState(row) && (
                                <span
                                  className="w-fit rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-1 py-0.5 font-mono text-[8px] uppercase leading-none text-[#79c0ff]"
                                  title={backendTooltipForMatrixRow(row)}
                                >
                                  steady
                                </span>
                              )}
                            </div>
                          )}
                        </td>
                        {concurrencies.map(concurrency => (
                          <ServingMatrixCell
                            key={concurrency}
                            row={row.cells[concurrency]}
                            selectedKey={selectedPerTurnRowKey}
                            onSelectPerTurn={setSelectedPerTurnKey}
                          />
                        ))}
                        {SERVING_METRICS.map((metric, metricIndex) => (
                          <ServingRowMeanCell
                            key={metric.label}
                            matrixRow={row}
                            metric={metric}
                            metricIndex={metricIndex}
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

      <ServingPerTurnBreakdown
        row={selectedPerTurnRow}
        selectedMetric={selectedMetric}
        onSelectMetric={setSelectedMetric}
      />

      <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#6e7681]">
        <span>
          {tpotOnly ? 'Cells show TPOT-only kernel-composed error; TTFT and E2EL are N/A.' : (
            <>Cells show % error left-to-right: <span className="text-[#f0883e]">TTFT</span> / <span className="text-[#58a6ff]">TPOT</span> / <span className="text-[#a855f7]">E2EL</span>.</>
          )}
        </span>
        <span className="font-medium text-[#8b949e]">Error bands:</span>
        <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">&lt;10%</span>
        <span className="rounded border border-[#58a6ff]/30 bg-[#58a6ff]/10 px-2 py-0.5 text-[#58a6ff]">10-25%</span>
        <span className="rounded border border-[#f0883e]/30 bg-[#f0883e]/10 px-2 py-0.5 text-[#f0883e]">25-50%</span>
        <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">&gt;=50%</span>
        <span>Rightmost MAPE columns are mean absolute row errors across concurrency cells.</span>
      </div>
    </div>
  );
}

function buildServingMatrixRows(rows: ServingRow[]): ServingMatrixRow[] {
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

  return Object.values(matrixByKey).sort((a, b) => {
    const modelOrder = a.model.localeCompare(b.model);
    if (modelOrder !== 0) return modelOrder;
    const profileOrder = servingProfileRank(a.profile) - servingProfileRank(b.profile);
    if (profileOrder !== 0) return profileOrder;
    const profileNameOrder = a.profile.localeCompare(b.profile);
    if (profileNameOrder !== 0) return profileNameOrder;
    return (a.backend ?? '').localeCompare(b.backend ?? '');
  });
}

function groupServingRowsByModel(matrixRows: ServingMatrixRow[]): Record<string, ServingProfileGroup[]> {
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
  return groupedByModel;
}

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

function ServingPerTurnChart({
  turns,
  metric,
  onSelectMetric,
}: {
  turns: ServingTurnPrediction[];
  metric: ServingMetric;
  onSelectMetric: (m: ServingMetric) => void;
}) {
  // Build (turn_index, meas, pred) rows the chart can plot.  Nulls for
  // missing entries so Recharts breaks the line at gaps rather than
  // interpolating across them.
  const chartData = useMemo(
    () =>
      turns.map(turn => {
        const meas = turn[metric.measKey];
        const pred = turn[metric.predKey];
        // llm-d-inference-sim only models TPOT (no per-token TTFT or E2EL).
        // Constant across turns within a (profile, c) cell by design — the
        // line will appear flat, illustrating that the model is turn-blind.
        const llmd = metric.label === 'TPOT' ? turn.tpot_pred_llm_d : undefined;
        // Two-roofline: physical TPOT (no fits). See
        // simulator/two_roofline_tpot.py and
        // profiling/docs/two-roofline-tpot-2026-05-28.md.
        const twoRoofline =
          metric.label === 'TPOT' ? turn.tpot_pred_two_roofline : undefined;
        return {
          turn: displayTurn(turn),
          meas: typeof meas === 'number' && Number.isFinite(meas) ? meas : null,
          pred: typeof pred === 'number' && Number.isFinite(pred) ? pred : null,
          llmd: typeof llmd === 'number' && Number.isFinite(llmd) ? llmd : null,
          twoRoofline:
            typeof twoRoofline === 'number' && Number.isFinite(twoRoofline)
              ? twoRoofline
              : null,
        };
      }),
    [turns, metric.measKey, metric.predKey, metric.label],
  );
  const showLlmd = metric.label === 'TPOT' && chartData.some(d => d.llmd !== null);
  const showTwoRoofline =
    metric.label === 'TPOT' && chartData.some(d => d.twoRoofline !== null);
  if (chartData.length === 0) return null;
  return (
    <div className="border-b border-[#21262d] px-4 py-3">
      <div className="mb-2 flex items-center gap-2">
        <div className="text-[11px] uppercase tracking-wide text-[#8b949e]">Per-Turn</div>
        <div className="flex gap-1">
          {SERVING_METRICS.map(m => {
            const selected = m.label === metric.label;
            return (
              <button
                key={m.label}
                type="button"
                onClick={() => onSelectMetric(m)}
                className={`rounded border px-2 py-0.5 text-[10px] font-mono uppercase transition-colors ${
                  selected
                    ? 'border-[#58a6ff] bg-[#1f6feb]/20 text-[#e6edf3]'
                    : 'border-[#30363d] bg-[#0d1117] text-[#8b949e] hover:border-[#58a6ff]/60 hover:text-[#e6edf3]'
                }`}
                style={selected ? { borderColor: m.color, color: m.color } : undefined}
              >
                {m.label}
              </button>
            );
          })}
        </div>
        <span className="text-[10px] text-[#6e7681]">{metric.description} · actual vs predicted (ms)</span>
      </div>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 24, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="#21262d" strokeDasharray="3 3" />
            <XAxis
              dataKey="turn"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              stroke="#30363d"
              label={{ value: 'turn', position: 'insideBottomRight', offset: -2, fill: '#6e7681', fontSize: 10 }}
            />
            <YAxis
              tick={{ fill: '#8b949e', fontSize: 11 }}
              stroke="#30363d"
              width={48}
              label={{ value: 'ms', angle: -90, position: 'insideLeft', offset: 12, fill: '#6e7681', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#0d1117',
                border: '1px solid #30363d',
                fontSize: 11,
              }}
              labelStyle={{ color: '#c9d1d9' }}
              formatter={(value) =>
                typeof value === 'number' ? `${value.toFixed(2)} ms` : '—'
              }
              labelFormatter={(turn) => `Turn ${turn}`}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: '#c9d1d9' }}
              content={() => (
                <div className="mt-1 flex flex-wrap justify-center gap-4">
                  <span className="flex items-center gap-2">
                    <svg width="26" height="8" aria-hidden>
                      <line x1="0" y1="4" x2="26" y2="4" stroke={metric.color} strokeWidth="2" />
                    </svg>
                    <span className="text-[11px] text-[#c9d1d9]">{metric.label} actual</span>
                  </span>
                  <span className="flex items-center gap-2">
                    <svg width="26" height="8" aria-hidden>
                      <line x1="0" y1="4" x2="26" y2="4" stroke={metric.color} strokeWidth="2" strokeDasharray="5 4" />
                    </svg>
                    <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (roofline)</span>
                  </span>
                  {showLlmd && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#f59e0b" strokeWidth="2" strokeDasharray="2 3" />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (llm-d)</span>
                    </span>
                  )}
                  {showTwoRoofline && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#22c55e" strokeWidth="2" />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (two-roofline)</span>
                    </span>
                  )}
                </div>
              )}
            />
            <Line
              type="monotone"
              dataKey="meas"
              name={`${metric.label} actual`}
              stroke={metric.color}
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name={`${metric.label} predicted (roofline)`}
              stroke={metric.color}
              strokeDasharray="5 4"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            {showLlmd && (
              <Line
                type="monotone"
                dataKey="llmd"
                name={`${metric.label} predicted (llm-d)`}
                stroke="#f59e0b"
                strokeDasharray="2 3"
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            )}
            {showTwoRoofline && (
              <Line
                type="monotone"
                dataKey="twoRoofline"
                name={`${metric.label} predicted (two-roofline)`}
                stroke="#22c55e"
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ServingPerTurnBreakdown({
  row,
  selectedMetric,
  onSelectMetric,
}: {
  row?: ServingPerTurnRow;
  selectedMetric: ServingMetric;
  onSelectMetric: (m: ServingMetric) => void;
}) {
  const turns = useMemo(
    () => row ? [...row.multiturn_turn_predictions].sort((a, b) => a.turn_index - b.turn_index) : [],
    [row],
  );
  const meanHit = turns.length
    ? turns.reduce((total, turn) => total + turn.cache_hit_rate, 0) / turns.length
    : 0;
  if (!row) return null;

  const meanSignedTpotErr = row.tpot_signed_err_ms ?? (
    turns.length
      ? turns.reduce((total, turn) => total + (turnSignedErrorMs(turn, SERVING_TPOT_METRIC) ?? 0), 0) / turns.length
      : undefined
  );
  const meanAbsTpotErr = row.tpot_abs_err_ms ?? (
    turns.length
      ? turns.reduce((total, turn) => total + Math.abs(turnSignedErrorMs(turn, SERVING_TPOT_METRIC) ?? 0), 0) / turns.length
      : undefined
  );

  return (
    <div className="rounded-md border border-[#21262d] bg-[#161b22]">
      <div className="flex flex-col gap-3 border-b border-[#21262d] px-4 py-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm font-semibold text-[#e6edf3]">Per-Turn Multi-Turn Prediction</div>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-[#6e7681]">
            <span className="font-mono text-[#8b949e]">{row.model}</span>
            <span>{row.backend ?? 'backend'}</span>
            {row.backend_emulator_status === 'event_loop_enabled' && (
              <>
                <span>backend steps {formatTokenCount(row.backend_trace_summary?.total_steps)}</span>
                <span>max decode {formatTokenCount(row.backend_trace_summary?.max_decode_batch)}</span>
                <span>replay {formatTokenCount(row.backend_trace_summary?.replayed_cached_tokens)}</span>
              </>
            )}
            <span>{profileDisplayName(row.profile)}</span>
            <span>c{row.concurrency ?? 1}</span>
            <span>{turns.length} turns</span>
            <span>{row.total_successful_turn_requests ?? 0} successful turn requests</span>
            <span>mean cache hit {(meanHit * 100).toFixed(0)}%</span>
            <span>mean TTFT {formatLatency(row.mean_predicted_turn_ttft_ms)}</span>
            <span>mean TPOT {formatLatency(row.mean_predicted_turn_tpot_ms)}</span>
            <span>TPOT signed err {formatSignedLatency(meanSignedTpotErr)}</span>
            <span>TPOT MAE {formatLatency(meanAbsTpotErr)}</span>
          </div>
        </div>
        <div className="rounded border border-[#30363d] bg-[#0d1117] px-2 py-1 font-mono text-[10px] text-[#79c0ff]">
          selected from predictions table
        </div>
      </div>

      <ServingPerTurnChart
        turns={turns}
        metric={selectedMetric}
        onSelectMetric={onSelectMetric}
      />

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
        <table className="w-full min-w-[1240px] border-collapse text-xs">
          <thead>
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-3 py-2 text-left font-medium">Turn</th>
              <th className="px-2 py-2 text-left font-medium">Regime</th>
              <th className="px-2 py-2 text-right font-medium">Req</th>
              <th className="px-2 py-2 text-right font-medium">Ctx</th>
              <th className="px-2 py-2 text-right font-medium">New</th>
              <th className="px-2 py-2 text-right font-medium">Cached</th>
              <th className="w-[150px] px-2 py-2 text-left font-medium">Hit</th>
              <th className="px-2 py-2 text-right font-medium">Out</th>
              <th className="px-2 py-2 text-right font-medium">Steps/Waves</th>
              <th className="px-2 py-2 text-right font-medium">Replay</th>
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
                <td className="px-2 py-2 text-[10px] text-[#8b949e]" title={turn.scheduling_regime ?? turn.workload_regime ?? turn.turn_batching_regime}>
                  <div>{compactRegime(turn.turn_position_bin)}</div>
                  <div className="font-mono text-[#6e7681]">{compactRegime(turn.scheduling_regime ?? turn.decode_load_regime)}</div>
                </td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.successful)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.total_context_tokens)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.new_prefill_tokens)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.cached_context_tokens)}</td>
                <td className="px-2 py-2"><ServingTurnCacheBar turn={turn} /></td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.output_tokens)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.decode_waves ?? turn.backend_trace_summary?.total_steps)}</td>
                <td className="px-2 py-2 text-right font-mono text-[#8b949e]">{formatTokenCount(turn.backend_cache_work?.replayed_cached_tokens)}</td>
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

function ServingMetricSummary({
  metric,
  rows,
  rowCount,
}: {
  metric: ServingMetric;
  rows: ServingRow[];
  rowCount?: number;
}) {
  const absoluteErrors = rows
    .map(row => numericMetric(row, metric.errKey))
    .filter((value): value is number => value !== undefined)
    .map(value => Math.abs(value));
  const mape = absoluteErrors.length ? mean(absoluteErrors) : undefined;
  const best = absoluteErrors.length ? Math.min(...absoluteErrors) : undefined;
  const worst = absoluteErrors.length ? Math.max(...absoluteErrors) : undefined;
  const displayedRowCount = mape !== undefined && rowCount !== undefined
    ? rowCount
    : absoluteErrors.length;

  return (
    <div className="border-b border-[#21262d] px-3 py-2.5 last:border-b-0 md:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: metric.color }}>{metric.label}</div>
          <div className="mt-0.5 text-[11px] text-[#6e7681]">{metric.description}</div>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold text-[#e6edf3]">{formatPercent(mape)}</div>
          <div className="text-[10px] text-[#6e7681]">MAPE</div>
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-[#21262d] pt-2 text-[10px] text-[#6e7681]">
        <span>{displayedRowCount} rows</span>
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
  const signedMs = turnSignedErrorMs(turn, metric);
  const tone = servingErrorTone(err);
  return (
    <div className="grid grid-cols-[34px_1fr] items-center gap-1">
      <span className="text-[9px] font-semibold uppercase" style={{ color: metric.color }}>{metric.label}</span>
      <span className={`rounded px-1.5 py-0.5 text-right font-mono text-[10px] leading-none ${tone.className}`}>
        {formatCompactPercent(err)}
        {signedMs !== undefined && (
          <span className="ml-1 text-[#8b949e]">{formatSignedLatency(signedMs)}</span>
        )}
      </span>
    </div>
  );
}

function ServingTurnMetricCell({ turn, metric }: { turn: ServingTurnPrediction; metric: ServingMetric }) {
  const pred = numericTurnMetric(turn, metric.predKey);
  const meas = numericTurnMetric(turn, metric.measKey);
  const err = numericTurnMetric(turn, metric.errKey);
  const signedMs = turnSignedErrorMs(turn, metric);
  const tone = servingErrorTone(err);
  return (
    <td className="px-2 py-2 align-top">
      <div className="space-y-1">
        <MetricLine label="Pred" value={formatLatency(pred, metric.isTotal)} />
        <MetricLine label="Actual" value={formatLatency(meas, metric.isTotal)} />
        <MetricLine label="Signed" value={formatSignedLatency(signedMs)} />
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
        <span className="text-[#30363d]">.</span>
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
      title={canSelect ? `Show ${row.multiturn_turn_predictions.length} per-turn predictions — pick the metric with the toggle above the chart` : undefined}
    >
      <div className="min-w-0 space-y-0.5" title={`ISL->OSL ${row.isl}->${row.osl}`}>
        <div className="grid min-w-0 grid-cols-3 gap-0.5">
          {SERVING_METRICS.map(metric => (
            <ServingMiniMetric key={metric.label} row={row} metric={metric} />
          ))}
        </div>
      </div>
    </td>
  );
}

function ServingRowMeanCell({
  matrixRow,
  metric,
  metricIndex,
}: {
  matrixRow: ServingMatrixRow;
  metric: ServingMetric;
  metricIndex: number;
}) {
  const value = meanMatrixRowMetricError(matrixRow, metric.errKey);
  const tone = servingErrorTone(value);
  const rows = Object.values(matrixRow.cells).length;

  return (
    <td
      className={`serving-mape-rail sticky z-10 px-1 py-0.5 align-middle ${
        metricIndex === 0 ? 'serving-mape-rail-start' : 'border-l border-[#1f2937]'
      }`}
      style={{ right: `${(SERVING_METRICS.length - metricIndex - 1) * SERVING_MAPE_COLUMN_WIDTH}px` }}
      title={`${matrixRow.profile} ${matrixRow.backend ?? ''}: mean absolute ${metric.label} error across ${rows} concurrency cells`}
    >
      <span className={`block rounded px-1 py-0.5 text-center font-mono text-[10px] leading-none ${tone.className}`}>
        {formatCompactPercent(value)}
      </span>
    </td>
  );
}

function meanMatrixRowMetricError(matrixRow: ServingMatrixRow, errKey: ServingMetricKey): number | undefined {
  const values = Object.values(matrixRow.cells)
    .map(row => numericMetric(row, errKey))
    .filter((value): value is number => value !== undefined)
    .map(value => Math.abs(value));
  return values.length ? mean(values) : undefined;
}

function representativeMatrixRowCell(matrixRow: ServingMatrixRow): ServingRow | undefined {
  return Object.values(matrixRow.cells)[0];
}

function matrixRowUsesBackendEmulator(matrixRow: ServingMatrixRow): boolean {
  return Object.values(matrixRow.cells).some(row => row.backend_emulator_status === 'event_loop_enabled');
}

function matrixRowUsesSteadyState(matrixRow: ServingMatrixRow): boolean {
  return Object.values(matrixRow.cells).some(row => isSteadyStateRow(row));
}

function isSteadyStateRow(row: ServingRow): boolean {
  return row.continuous_batching_mode?.includes('steady_state') ?? false;
}

function backendTooltipForMatrixRow(matrixRow: ServingMatrixRow): string {
  const row = representativeMatrixRowCell(matrixRow);
  return row ? backendTooltip(row) : 'legacy scheduler';
}

function ServingMiniMetric({ row, metric }: { row: ServingRow; metric: ServingMetric }) {
  const pred = numericMetric(row, metric.predKey);
  const meas = numericMetric(row, metric.measKey);
  const err = numericMetric(row, metric.errKey);
  const signedMs = rowSignedErrorMs(row, metric);
  const tone = servingErrorTone(err);
  const title = [
    `${metric.label}: ${formatPercent(err)} error`,
    `signed ${formatSignedLatency(signedMs)}`,
    `pred ${formatLatency(pred, metric.isTotal)}`,
    `meas ${formatLatency(meas, metric.isTotal)}`,
    `ISL->OSL ${row.isl}->${row.osl}`,
    cacheTooltip(row),
    backendTooltip(row),
    measurementTooltip(row),
  ].join(' | ');

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

function backendTooltip(row: ServingRow): string {
  if (row.backend_emulator_status !== 'event_loop_enabled') return 'legacy scheduler';
  const summary = row.backend_trace_summary;
  const spec = row.backend_spec;
  const batching = row.continuous_batching_mode
    ? `; batching ${row.continuous_batching_mode}`
    : '';
  const scheduled = row.scheduled_request_count
    ? `; scheduled ${formatTokenCount(row.scheduled_request_count)}`
    : '';
  const sourceSummary = row.kernel_source_summary
    ? `; kernels ${formatKernelSourceSummary(row.kernel_source_summary)}`
    : '';
  return [
    `backend emulator ${spec?.name ?? row.backend ?? 'selected'}`,
    `policy ${spec?.prefill_policy ?? 'n/a'}`,
    `cache ${spec?.cache_mode ?? 'n/a'}`,
    `steps ${formatTokenCount(summary?.total_steps)}`,
    `max decode ${formatTokenCount(summary?.max_decode_batch)}`,
    `cache replay ${formatTokenCount(summary?.replayed_cached_tokens)}`,
  ].join('; ') + batching + scheduled + sourceSummary;
}

function formatKernelSourceSummary(summary: Record<string, number>): string {
  return Object.entries(summary)
    .filter(([, count]) => count > 0)
    .map(([key, count]) => `${key}=${count}`)
    .join(', ');
}

function measurementTooltip(row: ServingRow): string {
  if (row.measurement_semantics_warning === 'measured_e2el_lt_ttft') {
    return 'measurement warning: measured E2EL is below measured TTFT';
  }
  return 'measurement semantics ok';
}

function numericMetric(row: ServingRow, key: ServingMetricKey): number | undefined {
  const value = row[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function numericTurnMetric(turn: ServingTurnPrediction, key: ServingMetricKey): number | undefined {
  const value = turn[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function rowSignedErrorMs(row: ServingRow, metric: ServingMetric): number | undefined {
  if (metric.label === 'TPOT' && typeof row.tpot_signed_err_ms === 'number' && Number.isFinite(row.tpot_signed_err_ms)) {
    return row.tpot_signed_err_ms;
  }
  const pred = numericMetric(row, metric.predKey);
  const meas = numericMetric(row, metric.measKey);
  return pred !== undefined && meas !== undefined ? pred - meas : undefined;
}

function turnSignedErrorMs(turn: ServingTurnPrediction, metric: ServingMetric): number | undefined {
  if (metric.label === 'TPOT' && typeof turn.tpot_signed_err_ms === 'number' && Number.isFinite(turn.tpot_signed_err_ms)) {
    return turn.tpot_signed_err_ms;
  }
  const pred = numericTurnMetric(turn, metric.predKey);
  const meas = numericTurnMetric(turn, metric.measKey);
  return pred !== undefined && meas !== undefined ? pred - meas : undefined;
}

function servingErrorTone(err: OptionalMetric): { className: string } {
  if (err === undefined || err === null) return { className: 'border border-[#30363d] bg-[#21262d] text-[#6e7681]' };
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

function formatSignedLatency(value: number | undefined): string {
  if (value === undefined) return 'n/a';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)} ms`;
}

function formatPercent(value: OptionalMetric): string {
  if (value === undefined || value === null) return 'N/A';
  return `${value.toFixed(1)}%`;
}

function formatCompactPercent(value: OptionalMetric): string {
  if (value === undefined || value === null) return 'N/A';
  return `${value.toFixed(0)}%`;
}

function formatTokenCount(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '-';
  return Math.round(value).toLocaleString();
}

function compactRegime(value: string | undefined): string {
  if (!value) return '-';
  return value
    .replace('startup_0_4', 'startup')
    .replace('ramp_5_9', 'ramp')
    .replace('steady_10_19', 'steady')
    .replace('tail_20_plus', 'tail')
    .replace('queued_saturated_decode', 'queued sat')
    .replace('saturated_decode', 'sat decode')
    .replace('high_decode', 'high decode')
    .replace('medium_decode', 'med decode')
    .replace('low_decode', 'low decode')
    .split('_').join(' ');
}

function formatConcurrencyRange(values: number[]): string {
  if (!values.length) return '-';
  if (values.length === 1) return `c${values[0]}`;
  return `c${values[0]}-c${values[values.length - 1]} (${values.length})`;
}

function displayTurn(turn: ServingTurnPrediction): number {
  return turn.turn_index + 1;
}

function mean(arr: number[]): number {
  if (!arr.length) return 0;
  return arr.reduce((total, value) => total + value, 0) / arr.length;
}
