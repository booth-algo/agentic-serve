// Runtime sweep state published by scripts/publish_sweep_state.py to R2.
// Schema mirrors sweep.yaml + /tmp/bench_jobs/state/<jid>.status.

export type CellStatus = 'pending' | 'running' | 'done' | 'skipped' | 'known_oom';

export interface SweepCell {
  host: string;
  hw_label: string;  // e.g. "A100-40GBx4"
  model: string;
  tp: number;
  mode: 'single' | 'multi';
  backend: string;   // "vllm" | "sglang"
  status: CellStatus;
  attempt: number;
  max_len_override: number | null;
  reason: string | null;
  updated_at: string | null;  // ISO-8601 UTC
}

export interface SweepHost {
  hardware_label: string;
  vram_gb_per_gpu: number;
  total_gpus: number;
}

export interface SweepModel {
  weights_gb: number;
}

export interface SweepState {
  generated_at: string;
  feasibility_ratio: number;
  hosts: Record<string, SweepHost>;
  models: Record<string, SweepModel>;
  cells: SweepCell[];
}
