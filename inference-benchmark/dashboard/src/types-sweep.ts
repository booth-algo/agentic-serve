// Runtime sweep state published by scripts/publish_sweep_state.py to R2.
// Schema mirrors sweep.yaml + /tmp/bench_jobs/state/<jid>.status.

export type CellStatus = 'pending' | 'running' | 'done' | 'skipped' | 'failed' | 'known_oom';

export interface SweepCell {
  data_scope?: 'trace_replay' | 'synthetic_distributional' | 'archived' | 'synthetic' | 'synthetic-distributional' | 'latest' | 'current' | 'fixed' | 'mse' | 'archive';
  source_scope?: string;
  host: string;
  hw_label: string;  // e.g. "A100-40GBx4"
  model: string;
  tp: number;
  mode: 'single' | 'multi';
  backend: string;   // "vllm" | "sglang"
  status: CellStatus;
  attempt: number;
  max_len: number | null;
  gpu_mem: number | null;
  profiles: string[];
  concurrencies: number[];
  max_len_override: number | null;
  reason: string | null;
  updated_at: string | null;  // ISO-8601 UTC
  run_id?: string | null;
  failure_metadata?: {
    kind?: string;
    status?: string;
    reason?: string;
    attempt?: number | null;
    max_attempts?: number | null;
    expected_outputs_present?: number | null;
    expected_outputs_total?: number | null;
    missing_outputs?: string[];
    remote_log?: string;
    mirror_status?: string;
    updated_at?: string;
  } | null;
}

export interface SweepHost {
  hardware_label: string;
  vram_gb_per_gpu: number;
  total_gpus: number;
}

export interface SweepModel {
  weights_gb: number;
}

export interface SweepProfileInfeasible {
  data_scope?: 'trace_replay' | 'synthetic_distributional' | 'archived' | 'synthetic' | 'synthetic-distributional' | 'latest' | 'current' | 'fixed' | 'mse' | 'archive';
  source_scope?: string;
  host: string;
  hw_label: string;
  model: string;
  tp: number;
  mode: 'single' | 'multi';
  backend: string;
  profile: string;
  max_len: number;
  reason: string;
}

export interface SweepState {
  generated_at: string;
  feasibility_ratio: number;
  hosts: Record<string, SweepHost>;
  models: Record<string, SweepModel>;
  cells: SweepCell[];
  profile_infeasible?: SweepProfileInfeasible[];
}
