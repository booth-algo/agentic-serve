import type { DataScope } from './profileMeta';

declare const __BUILD_HASH__: string;

const DEFAULT_R2_JSON_BASE = 'https://pub-38e30ed030784867856634f1625c7130.r2.dev/json/current';

function joinUrl(base: string, path: string): string {
  return `${base.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
}

function withBuildHash(url: string): string {
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}v=${__BUILD_HASH__}`;
}

const jsonBase = import.meta.env.VITE_R2_JSON_BASE || DEFAULT_R2_JSON_BASE;
const dashboardApiBase = import.meta.env.VITE_DASHBOARD_API_BASE || (jsonBase.startsWith('http') ? '' : jsonBase);
const publicAssetBase = import.meta.env.BASE_URL || '/';

export const dataJsonUrl = withBuildHash(
  import.meta.env.VITE_DATA_JSON_URL || joinUrl(jsonBase, 'data.json'),
);

const scopedDataUrlOverrides: Partial<Record<DataScope, string | undefined>> = {
  trace_replay: import.meta.env.VITE_TRACE_REPLAY_DATA_JSON_URL,
  synthetic_distributional: import.meta.env.VITE_SYNTHETIC_DISTRIBUTIONAL_DATA_JSON_URL,
  archived: import.meta.env.VITE_ARCHIVED_DATA_JSON_URL,
};

export function dataJsonUrlForScope(scope: DataScope): string {
  return withBuildHash(scopedDataUrlOverrides[scope] || joinUrl(jsonBase, `data.${scope}.json`));
}

export const sweepStateUrl = withBuildHash(
  import.meta.env.VITE_SWEEP_STATE_URL || joinUrl(jsonBase, 'sweep-state.json'),
);

export const gemmEvalJsonUrl = withBuildHash(
  import.meta.env.VITE_GEMM_EVAL_JSON_URL || joinUrl(jsonBase, 'gemm-eval.json'),
);

export const servingPredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_SERVING_PREDICTIONS_JSON_URL || joinUrl(jsonBase, 'serving-predictions.json'),
);

export const llama31H100TpotFitJsonUrl = withBuildHash(
  import.meta.env.VITE_LLAMA31_H100_TPOT_FIT_JSON_URL || joinUrl(publicAssetBase, 'llama31-8b-h100-tpot-fit.json'),
);

export const simulatorPredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_SIMULATOR_PREDICTIONS_JSON_URL || joinUrl(publicAssetBase, 'simulator-predictions.json'),
);

// Forward predictor (no-GT path, run over the same cells) — MAPE counterpart to the backtester's
// simulator-predictions.json. Joined per (gpu_key, model, profile, concurrency) in the matrix.
export const forwardPredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_FORWARD_PREDICTIONS_JSON_URL || joinUrl(publicAssetBase, 'forward-predictions.json'),
);

export const simulatorV2PredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_SIMULATOR_V2_PREDICTIONS_JSON_URL || joinUrl(publicAssetBase, 'simulator-v2-predictions.json'),
);

export const simulatorV3PredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_SIMULATOR_V3_PREDICTIONS_JSON_URL || joinUrl(publicAssetBase, 'simulator-v3-predictions.json'),
);

export const profilingStateJsonUrl = withBuildHash(
  import.meta.env.VITE_PROFILING_STATE_JSON_URL || joinUrl(jsonBase, 'profiling-state.json'),
);

export const predictorCoverageJsonUrl = withBuildHash(
  import.meta.env.VITE_PREDICTOR_COVERAGE_JSON_URL || joinUrl(jsonBase, 'predictor-coverage.json'),
);

export const gpuStateJsonUrl = withBuildHash(
  import.meta.env.VITE_GPU_STATE_JSON_URL || joinUrl(jsonBase, 'gpu-state.json'),
);

export const coverageBlockersJsonUrl = withBuildHash(
  import.meta.env.VITE_COVERAGE_BLOCKERS_JSON_URL || joinUrl(publicAssetBase, 'coverage-blockers.synthetic_distributional.json'),
);

export const hostDrainApiUrl = import.meta.env.VITE_HOST_DRAIN_API_URL || (
  dashboardApiBase ? joinUrl(dashboardApiBase, 'api/host-drain') : ''
);

export const gpuBlockApiUrl = import.meta.env.VITE_GPU_BLOCK_API_URL || (
  dashboardApiBase ? joinUrl(dashboardApiBase, 'api/gpu-block') : ''
);
