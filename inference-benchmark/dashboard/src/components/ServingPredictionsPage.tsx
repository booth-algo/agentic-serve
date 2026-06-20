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
import { forwardPredictionsJsonUrl, llama31H100TpotFitJsonUrl, servingPredictionsJsonUrl } from '../dataUrls';
import { buildFwdLookup, fwdKey, type FwdLookup } from '../forwardPredictions';

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
  ttft_signed_err_ms?: number; ttft_abs_err_ms?: number;
  ttft_pred_static?: number; ttft_err_static?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  tpot_pred_kernel?: number;
  tpot_pred_kernel_hint?: number;
  tpot_pred_ramp?: number;
  tpot_regime?: string;
  tpot_signed_err_ms?: number; tpot_abs_err_ms?: number;
  base_tpot_signed_err_ms?: number; base_tpot_abs_err_ms?: number;
  e2el_pred?: number; e2el_meas?: number; e2el_err?: number;
  e2el_signed_err_ms?: number; e2el_abs_err_ms?: number;
  e2el_pred_static?: number; e2el_err_static?: number;
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

export interface ServingRow {
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
  ttft_signed_err_ms?: number; ttft_abs_err_ms?: number;
  ttft_pred_static?: number; ttft_err_static?: number;
  tpot_pred?: number; tpot_meas?: number; tpot_err?: number;
  tpot_signed_err_ms?: number; tpot_abs_err_ms?: number; tpot_max_abs_err_ms?: number;
  e2el_pred?: number; e2el_meas?: number; e2el_err?: number;
  e2el_signed_err_ms?: number; e2el_abs_err_ms?: number;
  e2el_pred_static?: number; e2el_err_static?: number;
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

export interface ServingIndex {
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
  | 'ttft_signed_err_ms' | 'ttft_abs_err_ms'
  | 'tpot_pred' | 'tpot_meas' | 'tpot_err'
  | 'tpot_signed_err_ms' | 'tpot_abs_err_ms'
  | 'e2el_pred' | 'e2el_meas' | 'e2el_err'
  | 'e2el_signed_err_ms' | 'e2el_abs_err_ms';

interface ServingMetric {
  label: string;
  description: string;
  color: string;
  predKey: ServingMetricKey;
  measKey: ServingMetricKey;
  errKey: ServingMetricKey;
  // Explicit per-turn / cell signed + abs latency-error fields written by the
  // augmenter (ttft_signed_err_ms, tpot_abs_err_ms, …). When present they are
  // read directly; otherwise the (pred - meas) fallback applies.
  signedKey: ServingMetricKey;
  absKey: ServingMetricKey;
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
    signedKey: 'ttft_signed_err_ms',
    absKey: 'ttft_abs_err_ms',
  },
  {
    label: 'TPOT',
    description: 'per output token',
    color: '#58a6ff',
    predKey: 'tpot_pred',
    measKey: 'tpot_meas',
    errKey: 'tpot_err',
    signedKey: 'tpot_signed_err_ms',
    absKey: 'tpot_abs_err_ms',
  },
  {
    label: 'E2EL',
    description: 'end-to-end',
    color: '#a855f7',
    predKey: 'e2el_pred',
    measKey: 'e2el_meas',
    errKey: 'e2el_err',
    signedKey: 'e2el_signed_err_ms',
    absKey: 'e2el_abs_err_ms',
    isTotal: true,
  },
];
const SERVING_TPOT_METRIC = SERVING_METRICS[1];
const SERVING_MAPE_COLUMN_WIDTH = 74;
const SERVING_MAPE_RAIL_WIDTH = SERVING_METRICS.length * SERVING_MAPE_COLUMN_WIDTH;

// Prediction source the Simulator page renders, mirroring the Predictions matrix' source toggle.
type ValueMode = 'mape' | 'delta';
type PredictionSource = 'backtester' | 'forward' | 'delta';

const PREDICTION_SOURCES: { key: PredictionSource; label: string }[] = [
  { key: 'backtester', label: 'Backtest' },
  { key: 'forward', label: 'Forward' },
  { key: 'delta', label: 'Δ' },
];

function finiteOrUndef(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

// Re-key the backtester serving rows onto the forward predictor's per-cell metrics, joined by
// (gpu_key, model, profile, concurrency) — same join the Predictions matrix uses.
//  - 'backtester' is identity (build_simulator_rows, scored vs measured GT).
//  - 'forward' swaps pred/err to the no-GT forward values; measured GT is unchanged. Forward has no
//    per-turn / emulator detail, so those are cleared (the per-turn panel + emu/steady tags hide).
//  - 'delta' replaces each *_err with the signed MAPE gap |fwd| - |bt| (negative = forward closer to
//    measured); pred/meas are cleared because the cell now shows a gap, not a latency.
function applyPredictionSource(
  rows: ServingRow[],
  gpuKey: string,
  source: PredictionSource,
  fwd: FwdLookup,
): ServingRow[] {
  if (source === 'backtester') return rows;
  return rows.map(row => {
    const match = row.model != null && row.profile != null && row.concurrency != null
      ? fwd.get(fwdKey(gpuKey, row.model, row.profile, row.concurrency))
      : undefined;
    const base: ServingRow = {
      ...row,
      multiturn_turn_predictions: undefined,
      backend_emulator_status: undefined,
      continuous_batching_mode: undefined,
      ttft_signed_err_ms: undefined,
      tpot_signed_err_ms: undefined,
      e2el_signed_err_ms: undefined,
      ttft_abs_err_ms: undefined,
      tpot_abs_err_ms: undefined,
      e2el_abs_err_ms: undefined,
    };
    if (source === 'forward') {
      return {
        ...base,
        ttft_pred: finiteOrUndef(match?.fwd_ttft_pred),
        tpot_pred: finiteOrUndef(match?.fwd_tpot_pred),
        e2el_pred: finiteOrUndef(match?.fwd_e2el_pred),
        ttft_err: finiteOrUndef(match?.fwd_ttft_err),
        tpot_err: finiteOrUndef(match?.fwd_tpot_err),
        e2el_err: finiteOrUndef(match?.fwd_e2el_err),
      };
    }
    const gap = (fwdErr: number | null | undefined, btErr: number | undefined): number | undefined => {
      const fe = finiteOrUndef(fwdErr);
      return fe !== undefined && btErr !== undefined ? Math.abs(fe) - Math.abs(btErr) : undefined;
    };
    return {
      ...base,
      ttft_pred: undefined,
      tpot_pred: undefined,
      e2el_pred: undefined,
      ttft_meas: undefined,
      tpot_meas: undefined,
      e2el_meas: undefined,
      ttft_err: gap(match?.fwd_ttft_err, finiteOrUndef(row.ttft_err)),
      tpot_err: gap(match?.fwd_tpot_err, finiteOrUndef(row.tpot_err)),
      e2el_err: gap(match?.fwd_e2el_err, finiteOrUndef(row.e2el_err)),
    };
  });
}

// Δ tone mirrors the Predictions matrix: green = forward >=3pt closer to measured, red = >=3pt
// worse, neutral between.
function servingDeltaTone(value: OptionalMetric): { className: string } {
  if (value === undefined || value === null) return { className: 'border border-[#30363d] bg-[#21262d] text-[#6e7681]' };
  if (value <= -3) return { className: 'border border-[#3fb950]/30 bg-[#3fb950]/10 text-[#3fb950]' };
  if (value >= 3) return { className: 'border border-[#f85149]/30 bg-[#f85149]/10 text-[#f85149]' };
  return { className: 'border border-[#30363d] bg-[#21262d] text-[#8b949e]' };
}

function formatSignedDeltaPct(value: OptionalMetric): string {
  if (value === undefined || value === null) return 'N/A';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(0)}`;
}

// Signed Δ with one decimal — for the headline / best / worst summary figures.
function formatSignedDeltaValue(value: OptionalMetric): string {
  if (value === undefined || value === null) return 'N/A';
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}`;
}

// Source-aware tone + compact formatting shared by the cells, the row-MAPE rail and the badges.
function toneFor(value: OptionalMetric, valueMode: ValueMode): { className: string } {
  return valueMode === 'delta' ? servingDeltaTone(value) : servingErrorTone(value);
}

function compactValueFor(value: OptionalMetric, valueMode: ValueMode): string {
  return valueMode === 'delta' ? formatSignedDeltaPct(value) : formatCompactPercent(value);
}

function meanSignedMetric(rows: ServingRow[], errKey: ServingMetricKey): number | undefined {
  const values = rows
    .map(row => numericMetric(row, errKey))
    .filter((value): value is number => value !== undefined);
  return values.length ? mean(values) : undefined;
}

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
  const [model, setModel] = useState('');
  const [backend, setBackend] = useState<'all' | 'vllm' | 'sglang'>('vllm');
  const [source, setSource] = useState<PredictionSource>('backtester');
  const [fwd, setFwd] = useState<FwdLookup>(new Map());
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

  // Forward predictions are optional — absent (404) until build_forward_rows has run.
  useEffect(() => {
    fetch(forwardPredictionsJsonUrl)
      .then(response => (response.ok ? response.json() : null))
      .then(json => setFwd(buildFwdLookup(json)))
      .catch(() => setFwd(new Map()));
  }, []);

  const scopeIndex = servingIndex?.[dataScope];
  // The simulator page self-selects model + GPU via dropdowns (no fixed focus prop).
  const selectorMode = pageKind === 'simulator' && !focus;

  // Model options = distinct models across every GPU config in the scope.
  const modelOptions = useMemo(() => {
    if (!scopeIndex) return EMPTY_GPU_OPTIONS;
    const models = new Set<string>();
    for (const gpuRows of Object.values(scopeIndex.rowsByGpu)) {
      for (const row of gpuRows) if (row.model) models.add(row.model);
    }
    return Array.from(models).sort();
  }, [scopeIndex]);
  const selectedModel = focus?.model ?? (modelOptions.includes(model) ? model : (modelOptions[0] ?? ''));

  // GPU options — restricted to those that actually have rows for the selected model
  // (the "GPU given model" dropdown); otherwise every GPU config in the scope.
  // Does this scope have any SGLang configs? (gpu_key carries a "(sglang)" suffix.) Drives the
  // backend dropdown so SGLang and vLLM configs don't conflate in the GPU selector.
  const hasSglang = useMemo(
    () => (scopeIndex ? scopeIndex.gpuOptions.some(g => /\(sglang\)/i.test(g)) : false),
    [scopeIndex]);
  const gpuOptions = useMemo(() => {
    if (!scopeIndex) return EMPTY_GPU_OPTIONS;
    if (!selectorMode) return scopeIndex.gpuOptions;
    const matchesBackend = (g: string) =>
      backend === 'all' ? true : backend === 'sglang' ? /\(sglang\)/i.test(g) : !/\(sglang\)/i.test(g);
    return scopeIndex.gpuOptions.filter(g =>
      (scopeIndex.rowsByGpu[g] ?? []).some(row => row.model === selectedModel) && matchesBackend(g));
  }, [scopeIndex, selectorMode, selectedModel, backend]);
  const selectedGpu = focus?.gpu ?? (gpuOptions.includes(gpu) ? gpu : (gpuOptions[0] ?? gpu));

  useEffect(() => {
    if (focus?.gpu) return;
    if (!gpuOptions.length) return;
    setGpu(current => gpuOptions.includes(current) ? current : gpuOptions[0]);
  }, [focus?.gpu, gpuOptions]);

  const rows = useMemo(() => {
    const base = scopeIndex?.rowsByGpu[selectedGpu] ?? [];
    if (focus) return applyServingFocus(base, focus);
    if (selectorMode) return base.filter(row => row.model === selectedModel);
    return base;
  }, [scopeIndex, selectedGpu, focus, selectorMode, selectedModel]);
  // Source toggle (Backtest / Forward / Δ). Forward/Δ collapse to Backtest until forward rows load.
  const hasForward = fwd.size > 0;
  const effectiveSource: PredictionSource = source !== 'backtester' && !hasForward ? 'backtester' : source;
  const valueMode: ValueMode = effectiveSource === 'delta' ? 'delta' : 'mape';
  const sourcedRows = useMemo(
    () => applyPredictionSource(rows, selectedGpu, effectiveSource, fwd),
    [rows, selectedGpu, effectiveSource, fwd],
  );
  // In selector mode, keep the table's focused single-config view via a synthesized focus.
  const tableFocus: ServingFocus | undefined = selectorMode && selectedModel
    ? { gpu: selectedGpu, model: selectedModel, title: 'Simulator', description: '' }
    : focus;
  const showFixedTpotFit = fixedTpotFit
    && fixedTpotFit.experiment.gpu === selectedGpu
    && fixedTpotFit.experiment.dashboard_scope === dataScope
    && (focus
      ? focus.model === fixedTpotFit.experiment.model
      : (!selectorMode || selectedModel === fixedTpotFit.experiment.model));
  // The fixed-TPOT validation overlay is a backtester-only artifact; Forward/Δ show the live rows.
  const fixedTpotOnly = Boolean(showFixedTpotFit && pageKind === 'simulator' && effectiveSource === 'backtester');
  const tableSourceRows = useMemo(
    () => fixedTpotOnly ? sourcedRows.filter(row => !isSingleTurnServingRow(row)) : sourcedRows,
    [fixedTpotOnly, sourcedRows],
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
      {selectorMode ? (
        <SimulatorTargetBar
          modelOptions={modelOptions}
          selectedModel={selectedModel}
          onModel={setModel}
          gpuOptions={gpuOptions}
          selectedGpu={selectedGpu}
          onGpu={setGpu}
          backend={backend}
          onBackend={setBackend}
          showBackend={hasSglang}
          source={effectiveSource}
          onSource={setSource}
          hasForward={hasForward}
          valueMode={valueMode}
          ttftMape={valueMode === 'delta' ? meanSignedMetric(sourcedRows, 'ttft_err') : meanMetricError(sourcedRows, 'ttft_err')}
          tpotMape={valueMode === 'delta' ? meanSignedMetric(sourcedRows, 'tpot_err') : meanMetricError(sourcedRows, 'tpot_err')}
          e2elMape={valueMode === 'delta' ? meanSignedMetric(sourcedRows, 'e2el_err') : meanMetricError(sourcedRows, 'e2el_err')}
        />
      ) : (
        <>
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
        </>
      )}

      <ServingTable
        rows={tableRows}
        summaryRows={tableSummaryRows}
        summaryRowCount={tableSummaryRowCount}
        dataScope={dataScope}
        focus={tableFocus}
        tpotOnly={fixedTpotOnly}
        validationRows={useFixedTpotRows}
        valueMode={valueMode}
      />
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

function LabeledSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">{label}</span>
      <select
        value={value}
        onChange={event => onChange(event.target.value)}
        disabled={options.length <= 1}
        className="min-w-[150px] rounded border border-[#30363d] bg-[#0d1117] px-2 py-1 font-mono text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff] disabled:cursor-not-allowed disabled:opacity-70"
      >
        {options.length === 0 && <option value="">—</option>}
        {options.map(option => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function SimulatorTargetBar({
  modelOptions,
  selectedModel,
  onModel,
  gpuOptions,
  selectedGpu,
  onGpu,
  backend,
  onBackend,
  showBackend,
  source,
  onSource,
  hasForward,
  valueMode,
  ttftMape,
  tpotMape,
  e2elMape,
}: {
  modelOptions: string[];
  selectedModel: string;
  onModel: (model: string) => void;
  gpuOptions: string[];
  selectedGpu: string;
  onGpu: (gpu: string) => void;
  backend: 'all' | 'vllm' | 'sglang';
  onBackend: (backend: 'all' | 'vllm' | 'sglang') => void;
  showBackend: boolean;
  source: PredictionSource;
  onSource: (source: PredictionSource) => void;
  hasForward: boolean;
  valueMode: ValueMode;
  ttftMape: OptionalMetric;
  tpotMape: OptionalMetric;
  e2elMape: OptionalMetric;
}) {
  const badgeSuffix = valueMode === 'delta' ? 'Δ' : 'MAPE';
  return (
    <section className="rounded-md border border-[#21262d] bg-[#161b22] px-4 py-3">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="flex flex-wrap items-end gap-3">
          <LabeledSelect label="Model" value={selectedModel} options={modelOptions} onChange={onModel} />
          <LabeledSelect label="GPU" value={selectedGpu} options={gpuOptions} onChange={onGpu} />
          {showBackend && (
            <label className="flex flex-col gap-1">
              <span className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">Backend</span>
              <select
                value={backend}
                onChange={event => onBackend(event.target.value as 'all' | 'vllm' | 'sglang')}
                className="min-w-[150px] rounded border border-[#30363d] bg-[#0d1117] px-2 py-1 font-mono text-sm text-[#e6edf3] outline-none focus:border-[#58a6ff]"
              >
                <option value="vllm">vLLM</option>
                <option value="sglang">SGLang</option>
                <option value="all">All</option>
              </select>
            </label>
          )}
          <label className="flex flex-col gap-1">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-[#6e7681]">Source</span>
            <div className="inline-flex overflow-hidden rounded-md border border-[#30363d] text-xs">
              {PREDICTION_SOURCES.map(({ key, label }) => {
                const disabled = key !== 'backtester' && !hasForward;
                const active = source === key;
                return (
                  <button
                    key={key}
                    type="button"
                    disabled={disabled}
                    onClick={() => onSource(key)}
                    title={disabled ? 'forward-predictions.json not loaded yet' : undefined}
                    className={`px-3 py-[5px] font-medium transition-colors ${
                      active ? 'bg-[#1f6feb] text-white'
                        : disabled ? 'bg-[#161b22] text-[#484f58] cursor-not-allowed'
                        : 'bg-[#161b22] text-[#8b949e] hover:bg-[#21262d]'
                    }`}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </label>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <MetricBadge label={`TTFT ${badgeSuffix}`} value={ttftMape} mode={valueMode} />
          <MetricBadge label={`TPOT ${badgeSuffix}`} value={tpotMape} mode={valueMode} />
          <MetricBadge label={`E2EL ${badgeSuffix}`} value={e2elMape} mode={valueMode} />
        </div>
      </div>
    </section>
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
  mode = 'mape',
}: {
  label: string;
  value: OptionalMetric;
  compact?: boolean;
  mode?: ValueMode;
}) {
  return (
    <span className={`inline-flex items-center justify-between gap-1 rounded px-1.5 py-0.5 font-mono ${compact ? 'text-[9px]' : 'text-[10px]'} ${toneFor(value, mode).className}`}>
      <span className="font-sans font-semibold uppercase tracking-wide">{label}</span>
      <span>{compactValueFor(value, mode)}</span>
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

export function buildServingIndex(data: Record<string, ServingRow[]>): ServingIndex {
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
  valueMode = 'mape',
}: {
  rows: ServingRow[];
  summaryRows?: ServingRow[];
  summaryRowCount?: number;
  dataScope: DataScope;
  focus?: ServingFocus;
  tpotOnly?: boolean;
  validationRows?: boolean;
  valueMode?: ValueMode;
}) {
  const isDelta = valueMode === 'delta';
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
            fallbackRows={rows}
            valueMode={valueMode}
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
                  <span className="text-[10px] font-semibold uppercase tracking-wide text-[#c9d1d9]">{isDelta ? 'Row Δ' : 'Row MAPE'}</span>
                  <span className="text-[9px] font-normal text-[#6e7681]">{isDelta ? 'fwd − bt (pt)' : 'mean abs error'}</span>
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
                            valueMode={valueMode}
                          />
                        ))}
                        {SERVING_METRICS.map((metric, metricIndex) => (
                          <ServingRowMeanCell
                            key={metric.label}
                            matrixRow={row}
                            metric={metric}
                            metricIndex={metricIndex}
                            valueMode={valueMode}
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

      {isDelta ? (
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-[#6e7681]">
          <span>
            Cells show <span className="text-[#c9d1d9]">Δ MAPE</span> (forward − backtest, pp) left-to-right:{' '}
            <span className="text-[#f0883e]">TTFT</span> / <span className="text-[#58a6ff]">TPOT</span> / <span className="text-[#a855f7]">E2EL</span>. Negative = forward closer to measured.
          </span>
          <span className="font-medium text-[#8b949e]">Δ bands:</span>
          <span className="rounded border border-[#3fb950]/30 bg-[#3fb950]/10 px-2 py-0.5 text-[#3fb950]">fwd ≥3pt better</span>
          <span className="rounded border border-[#30363d] bg-[#21262d] px-2 py-0.5 text-[#8b949e]">~equal</span>
          <span className="rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-0.5 text-[#f85149]">fwd ≥3pt worse</span>
          <span>Rightmost columns are the mean Δ across concurrency cells.</span>
        </div>
      ) : (
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
      )}
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
        // Kernel-composition: measured decode-kernel step + pressure-driven
        // amplifier to an output-amortized saturation ceiling. Workload-only,
        // no engine telemetry. See simulator/kernel_tpot.py.
        const kernel =
          metric.label === 'TPOT' ? turn.tpot_pred_kernel : undefined;
        // Kernel + classifier stepping-ramp soft hint. The standalone regime
        // classifier predicts WHEN saturation steps (KV-eviction crossing) and over
        // how many turns it ramps; this pulls the kernel up over that window (one-
        // sided, never lowers). Beats the headline kernel on every profile but is
        // shown as a comparison line. See simulator/_legacy/kernel_tpot_hint.py (retired 2026-06-10, L7).
        const kernelHint =
          metric.label === 'TPOT' ? turn.tpot_pred_kernel_hint : undefined;
        // Forward 3D-roofline eviction-deficit ramp predictor (comparison line).
        // Forecasts the cohort drain from the profile survival curve and ramps the
        // saturation from the eviction-watermark crossing. See simulator/ramp_tpot.py.
        const rampPred =
          metric.label === 'TPOT' ? turn.tpot_pred_ramp : undefined;
        // Static-M0 prediction (comparison line for TTFT/E2EL). The headline pred line is
        // now the queue sim (metric.predKey = ttft_pred/e2el_pred); this static comparison
        // (prefill baseline × Little's-law amplifier) is the line the queue sim beat. TPOT
        // has no static variant. See simulator/ttft_predict.py.
        const staticPred =
          metric.label === 'TTFT'
            ? turn.ttft_pred_static
            : metric.label === 'E2EL'
              ? turn.e2el_pred_static
              : undefined;
        return {
          turn: displayTurn(turn),
          meas: typeof meas === 'number' && Number.isFinite(meas) ? meas : null,
          pred: typeof pred === 'number' && Number.isFinite(pred) ? pred : null,
          kernel:
            typeof kernel === 'number' && Number.isFinite(kernel)
              ? kernel
              : null,
          kernelHint:
            typeof kernelHint === 'number' && Number.isFinite(kernelHint)
              ? kernelHint
              : null,
          rampPred:
            typeof rampPred === 'number' && Number.isFinite(rampPred)
              ? rampPred
              : null,
          staticPred:
            typeof staticPred === 'number' && Number.isFinite(staticPred)
              ? staticPred
              : null,
        };
      }),
    [turns, metric.measKey, metric.predKey, metric.label],
  );
  const showKernel =
    metric.label === 'TPOT' && chartData.some(d => d.kernel !== null);
  const showKernelHint =
    metric.label === 'TPOT' && chartData.some(d => d.kernelHint !== null);
  const showRamp =
    metric.label === 'TPOT' && chartData.some(d => d.rampPred !== null);
  const showStatic =
    metric.label !== 'TPOT' && chartData.some(d => d.staticPred !== null);
  // Which plotted line the prediction TABLE + MAPE badge actually use. The table reads <metric>_err,
  // and for TPOT that error is repointed to the kernel composition: it is the 'kernel' line on H100
  // (where tpot_pred is the roofline) but plain tpot_pred on tp2 (no separate kernel line) — i.e. the
  // kernel-composed line either way. TTFT/E2EL err is always against the headline 'pred' (queue sim).
  const tableKey = metric.label === 'TPOT' ? (showKernel ? 'kernel' : 'pred') : 'pred';
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
                <div className="mt-1">
                  <div className="flex flex-wrap justify-center gap-4">
                  <span className="flex items-center gap-2">
                    <svg width="26" height="8" aria-hidden>
                      <line x1="0" y1="4" x2="26" y2="4" stroke="#c9d1d9" strokeWidth="2" strokeDasharray="6 3" />
                    </svg>
                    <span className="text-[11px] text-[#c9d1d9]">{metric.label} actual</span>
                  </span>
                  <span className="flex items-center gap-2">
                    <svg width="26" height="8" aria-hidden>
                      <line x1="0" y1="4" x2="26" y2="4" stroke={metric.color} strokeWidth={tableKey === 'pred' ? 3.5 : 2} />
                    </svg>
                    <span className="text-[11px] text-[#c9d1d9]">
                      {(metric.label === 'TPOT'
                        ? `${metric.label} predicted`
                        : `${metric.label} predicted (queue sim)`)
                        + (tableKey === 'pred' ? ' ★ table' : '')}
                    </span>
                  </span>
                  {showKernel && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#facc15" strokeWidth={tableKey === 'kernel' ? 3.5 : 2} />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{`${metric.label} predicted (kernel)` + (tableKey === 'kernel' ? ' ★ table' : '')}</span>
                    </span>
                  )}
                  {showKernelHint && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#fb7185" strokeWidth="2" />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (kernel+hint)</span>
                    </span>
                  )}
                  {showRamp && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#2dd4bf" strokeWidth="2" />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (fwd-ramp)</span>
                    </span>
                  )}
                  {showStatic && (
                    <span className="flex items-center gap-2">
                      <svg width="26" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="26" y2="4" stroke="#3fb950" strokeWidth="2" />
                      </svg>
                      <span className="text-[11px] text-[#c9d1d9]">{metric.label} predicted (static M0)</span>
                    </span>
                  )}
                  </div>
                  <div className="mt-1 text-center text-[10px] text-[#6e7681]">
                    ★ = the line the prediction table &amp; MAPE badge use · other predicted lines are comparison-only
                  </div>
                </div>
              )}
            />
            <Line
              type="monotone"
              dataKey="meas"
              name={`${metric.label} actual`}
              stroke="#c9d1d9"
              strokeDasharray="6 3"
              strokeWidth={2}
              dot={{ r: 2 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="pred"
              name={
                (metric.label === 'TPOT'
                  ? `${metric.label} predicted (roofline)`
                  : `${metric.label} predicted (queue sim)`)
                + (tableKey === 'pred' ? ' ★ table' : '')
              }
              stroke={metric.color}
              strokeWidth={tableKey === 'pred' ? 3.5 : 2}
              dot={{ r: 2 }}
              activeDot={tableKey === 'pred' ? { r: 5 } : undefined}
              connectNulls={false}
              isAnimationActive={false}
            />
            {showKernel && (
              <Line
                type="monotone"
                dataKey="kernel"
                name={`${metric.label} predicted (kernel)` + (tableKey === 'kernel' ? ' ★ table' : '')}
                stroke="#facc15"
                strokeWidth={tableKey === 'kernel' ? 3.5 : 2}
                dot={{ r: 2 }}
                activeDot={tableKey === 'kernel' ? { r: 5 } : undefined}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
            {showKernelHint && (
              <Line
                type="monotone"
                dataKey="kernelHint"
                name={`${metric.label} predicted (kernel+hint)`}
                stroke="#fb7185"
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
            {showRamp && (
              <Line
                type="monotone"
                dataKey="rampPred"
                name={`${metric.label} predicted (fwd-ramp)`}
                stroke="#2dd4bf"
                strokeWidth={2}
                dot={{ r: 2 }}
                connectNulls={false}
                isAnimationActive={false}
              />
            )}
            {showStatic && (
              <Line
                type="monotone"
                dataKey="staticPred"
                name={`${metric.label} predicted (static M0)`}
                stroke="#3fb950"
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

  // Per-metric signed-err / MAE headers. Prefer the explicit cell-level field
  // (tpot/ttft/e2el_signed_err_ms, *_abs_err_ms) the augmenter now writes; fall
  // back to the per-turn average (turnSignedErrorMs is exact now too).
  const metricErrSummaries = SERVING_METRICS.map(metric => {
    const cellSigned = numericMetric(row, metric.signedKey);
    const cellAbs = numericMetric(row, metric.absKey);
    const signedErr = cellSigned ?? (
      turns.length
        ? turns.reduce((total, turn) => total + (turnSignedErrorMs(turn, metric) ?? 0), 0) / turns.length
        : undefined
    );
    const absErr = cellAbs ?? (
      turns.length
        ? turns.reduce((total, turn) => total + Math.abs(turnSignedErrorMs(turn, metric) ?? 0), 0) / turns.length
        : undefined
    );
    return { metric, signedErr, absErr };
  });

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
            {metricErrSummaries.map(({ metric, signedErr, absErr }) => (
              <Fragment key={metric.label}>
                <span>{metric.label} signed err {formatSignedLatency(signedErr)}</span>
                <span>{metric.label} MAE {formatLatency(absErr)}</span>
              </Fragment>
            ))}
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
  fallbackRows,
  valueMode = 'mape',
}: {
  metric: ServingMetric;
  rows: ServingRow[];
  rowCount?: number;
  fallbackRows?: ServingRow[];
  valueMode?: ValueMode;
}) {
  const isDelta = valueMode === 'delta';
  // MAPE view aggregates absolute errors; Δ view keeps the sign so a net-better metric reads negative.
  const valuesFrom = (rs: ServingRow[]) => rs
    .map(row => numericMetric(row, metric.errKey))
    .filter((value): value is number => value !== undefined)
    .map(value => (isDelta ? value : Math.abs(value)));
  // Primary rows (e.g. the fixed TPOT fit set) only carry tpot_err; for TTFT/E2EL
  // fall back to the real per-cell rows, which carry ttft_err / e2el_err.
  let values = valuesFrom(rows);
  let usedFallback = false;
  if (!values.length && fallbackRows && fallbackRows !== rows) {
    values = valuesFrom(fallbackRows);
    usedFallback = values.length > 0;
  }
  const headline = values.length ? mean(values) : undefined;
  const best = values.length ? Math.min(...values) : undefined;
  const worst = values.length ? Math.max(...values) : undefined;
  const displayedRowCount = headline !== undefined && rowCount !== undefined && !usedFallback
    ? rowCount
    : values.length;
  const fmt = (value: OptionalMetric) => (isDelta ? formatSignedDeltaValue(value) : formatPercent(value));

  return (
    <div className="border-b border-[#21262d] px-3 py-2.5 last:border-b-0 md:border-b-0">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: metric.color }}>{metric.label}</div>
          <div className="mt-0.5 text-[11px] text-[#6e7681]">{metric.description}</div>
        </div>
        <div className="text-right">
          <div className="text-lg font-semibold text-[#e6edf3]">{fmt(headline)}</div>
          <div className="text-[10px] text-[#6e7681]">{isDelta ? 'mean Δ (pp)' : 'MAPE'}</div>
        </div>
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-[#21262d] pt-2 text-[10px] text-[#6e7681]">
        <span>{displayedRowCount} rows</span>
        <span>best {fmt(best)} / worst {fmt(worst)}</span>
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
  valueMode = 'mape',
}: {
  row?: ServingRow;
  selectedKey: string | null;
  onSelectPerTurn: (key: string) => void;
  valueMode?: ValueMode;
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
            <ServingMiniMetric key={metric.label} row={row} metric={metric} valueMode={valueMode} />
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
  valueMode = 'mape',
}: {
  matrixRow: ServingMatrixRow;
  metric: ServingMetric;
  metricIndex: number;
  valueMode?: ValueMode;
}) {
  const value = valueMode === 'delta'
    ? meanMatrixRowMetricSigned(matrixRow, metric.errKey)
    : meanMatrixRowMetricError(matrixRow, metric.errKey);
  const tone = toneFor(value, valueMode);
  const rows = Object.values(matrixRow.cells).length;

  return (
    <td
      className={`serving-mape-rail sticky z-10 px-1 py-0.5 align-middle ${
        metricIndex === 0 ? 'serving-mape-rail-start' : 'border-l border-[#1f2937]'
      }`}
      style={{ right: `${(SERVING_METRICS.length - metricIndex - 1) * SERVING_MAPE_COLUMN_WIDTH}px` }}
      title={valueMode === 'delta'
        ? `${matrixRow.profile} ${matrixRow.backend ?? ''}: mean ${metric.label} Δ MAPE (forward − backtest) across ${rows} concurrency cells`
        : `${matrixRow.profile} ${matrixRow.backend ?? ''}: mean absolute ${metric.label} error across ${rows} concurrency cells`}
    >
      <span className={`block rounded px-1 py-0.5 text-center font-mono text-[10px] leading-none ${tone.className}`}>
        {compactValueFor(value, valueMode)}
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

// Signed mean (no abs) for the Δ view — preserves the sign so a row of forward-better cells reads
// negative.
function meanMatrixRowMetricSigned(matrixRow: ServingMatrixRow, errKey: ServingMetricKey): number | undefined {
  const values = Object.values(matrixRow.cells)
    .map(row => numericMetric(row, errKey))
    .filter((value): value is number => value !== undefined);
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

function ServingMiniMetric({ row, metric, valueMode = 'mape' }: { row: ServingRow; metric: ServingMetric; valueMode?: ValueMode }) {
  const pred = numericMetric(row, metric.predKey);
  const meas = numericMetric(row, metric.measKey);
  const err = numericMetric(row, metric.errKey);
  const signedMs = rowSignedErrorMs(row, metric);
  const tone = toneFor(err, valueMode);
  const title = valueMode === 'delta'
    ? [
        `${metric.label} Δ MAPE ${formatSignedDeltaPct(err)} pp (forward − backtest)`,
        'negative = forward closer to measured',
        `ISL->OSL ${row.isl}->${row.osl}`,
      ].join(' | ')
    : [
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
      {compactValueFor(err, valueMode)}
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
  // Prefer the explicit metric-keyed signed field (tpot/ttft/e2el_signed_err_ms)
  // now that the augmenter writes it for every metric; fall back to (pred - meas).
  const explicit = numericMetric(row, metric.signedKey);
  if (explicit !== undefined) return explicit;
  const pred = numericMetric(row, metric.predKey);
  const meas = numericMetric(row, metric.measKey);
  return pred !== undefined && meas !== undefined ? pred - meas : undefined;
}

function turnSignedErrorMs(turn: ServingTurnPrediction, metric: ServingMetric): number | undefined {
  // Prefer the explicit per-turn signed field; fall back to (pred - meas).
  const explicit = numericTurnMetric(turn, metric.signedKey);
  if (explicit !== undefined) return explicit;
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
