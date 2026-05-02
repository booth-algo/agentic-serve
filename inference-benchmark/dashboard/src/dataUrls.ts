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

export const dataJsonUrl = withBuildHash(
  import.meta.env.VITE_DATA_JSON_URL || joinUrl(jsonBase, 'data.json'),
);

export const sweepStateUrl = withBuildHash(
  import.meta.env.VITE_SWEEP_STATE_URL || joinUrl(jsonBase, 'sweep-state.json'),
);

export const gemmEvalJsonUrl = withBuildHash(
  import.meta.env.VITE_GEMM_EVAL_JSON_URL || joinUrl(jsonBase, 'gemm-eval.json'),
);

export const servingPredictionsJsonUrl = withBuildHash(
  import.meta.env.VITE_SERVING_PREDICTIONS_JSON_URL || joinUrl(jsonBase, 'serving-predictions.json'),
);

export const profilingStateJsonUrl = withBuildHash(
  import.meta.env.VITE_PROFILING_STATE_JSON_URL || joinUrl(jsonBase, 'profiling-state.json'),
);

export const predictorCoverageJsonUrl = withBuildHash(
  import.meta.env.VITE_PREDICTOR_COVERAGE_JSON_URL || joinUrl(jsonBase, 'predictor-coverage.json'),
);
