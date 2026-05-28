export interface BenchmarkConfig {
  url: string;
  model: string;
  backend: string;
  profile: string;
  concurrency: number;
  num_requests: number;
  api_key?: string;
  arrival: string;
  target_rate?: number;
  warmup?: number;
  seed?: number;
  timeout?: number;
  output?: string;
  ignore_eos?: boolean;
  mode?: string | null;
  benchmark_schema_version?: number;
  workload_schema_version?: string;
  dashboard_scope?: 'trace_replay' | 'synthetic_distributional' | 'archived' | 'synthetic' | 'synthetic-distributional' | 'latest' | 'current' | 'archive' | 'fixed' | 'mse';
  profile_metadata?: Record<string, unknown>;
  prediction_metadata?: Record<string, unknown>;
}

export interface BenchmarkSummary {
  model: string;
  profile: string;
  concurrency: number;
  num_requests: number;
  duration_s: number;
  successful_requests: number;
  failed_requests: number;
  request_throughput: number;
  input_token_throughput: number;
  output_token_throughput: number;
  total_token_throughput: number;
  total_input_tokens: number;
  total_output_tokens: number;
  mean_ttft_ms: number;
  median_ttft_ms: number;
  p90_ttft_ms: number;
  p99_ttft_ms: number;
  mean_tpot_ms: number;
  median_tpot_ms: number;
  p90_tpot_ms: number;
  p99_tpot_ms: number;
  mean_itl_ms: number;
  median_itl_ms: number;
  p90_itl_ms: number;
  p99_itl_ms: number;
  mean_e2el_ms: number;
  median_e2el_ms: number;
  p90_e2el_ms: number;
  p99_e2el_ms: number;
  errors: unknown[];
}

export interface PerTurnEntry {
  turn_index: number;
  num_requests: number;
  successful: number;
  mean_ttft_ms: number;
  median_ttft_ms: number;
  p90_ttft_ms: number;
  p99_ttft_ms: number;
  mean_tpot_ms: number;
  median_tpot_ms: number;
  mean_e2el_ms: number;
  median_e2el_ms: number;
  avg_input_tokens: number;
  avg_output_tokens: number;
  median_input_tokens?: number;
  median_output_tokens?: number;
  avg_new_prefill_tokens?: number;
  median_new_prefill_tokens?: number;
  avg_cached_context_tokens?: number;
  median_cached_context_tokens?: number;
  avg_cache_hit_rate?: number;
  median_cache_hit_rate?: number;
  avg_block_aligned_new_prefill_tokens?: number;
  median_block_aligned_new_prefill_tokens?: number;
  avg_block_aligned_cached_context_tokens?: number;
  median_block_aligned_cached_context_tokens?: number;
  avg_block_aligned_cache_hit_rate?: number;
  median_block_aligned_cache_hit_rate?: number;
  avg_uncached_prefix_tail_tokens?: number;
  median_uncached_prefix_tail_tokens?: number;
  median_client_queue_wait_ms?: number;
}

export interface ScatterPoint {
  input_tokens: number;
  ttft_ms: number;
  turn_index: number;
}

export interface BenchmarkResult {
  config: BenchmarkConfig;
  summary: BenchmarkSummary;
  // Enriched by build script
  hardware: string;
  quant: string;
  modelShort: string;
  seriesKey: string;
  filename: string;
  engineVersion?: string;  // "0.19.0" — from _engine_version.txt sidecar or fallback
  dataScope?: 'trace_replay' | 'synthetic_distributional' | 'archived' | 'synthetic' | 'synthetic-distributional' | 'latest' | 'current' | 'archive' | 'fixed' | 'mse';
  perTurn?: PerTurnEntry[];
  scatterData?: ScatterPoint[];
}

export interface FilterState {
  hardware: string[];
  model: string[];
  backend: string[];
  agentType: string[];
  turnStyle: string[];
  profile: string[];
}

export type TabId = 'latency' | 'throughput' | 'comparison' | 'multi-turn' | 'raw';

export interface FilterOptions {
  hardware: string[];
  model: string[];
  backend: string[];
  agentType: string[];
  turnStyle: string[];
  profile: string[];
}
