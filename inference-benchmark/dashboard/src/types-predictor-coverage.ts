// Predictor profiler coverage published by scripts/publish_predictor_coverage.py.
// Mirrors types-sweep.ts shape: per-(gpu, model) row-count breakdowns with
// expected denominators so the dashboard can render heatmap cells the same
// way the bench coverage page does.

export type PredCoverageStatus = 'present' | 'partial' | 'missing' | 'pending';

// Per-kernel row, per (gpu, model). prefill_rows comes from the
// `<model>_prefill` source in kernels_labeled.csv; flash_rows is the
// per-model dense flash sweep when present.
export interface KernelModelCell {
  gpu: string;
  model: string;
  prefill_rows: number;
  flash_rows: number;
  total_rows: number;
  held_out: boolean;
  status: PredCoverageStatus;
  expected_prefill: number;
  expected_flash: number;
}

// Per-kernel rows that are model-agnostic. roofline + misc are profiled
// once per GPU, then composed with per-model prefill/flash via the
// composer. sweep_version tracks whether the GEMM sweep used the
// post-fix nn.Linear path or the legacy a@b matmul.
export interface KernelSharedRow {
  gpu: string;
  roofline_rows: number;
  misc_rows: number;
  sweep_version: 'pre-fix' | 'post-fix' | 'unknown';
  expected_roofline: number;
  expected_misc: number;
}

// Per-op cell. rows_per_op breaks down the per-(bs, seq) grid by op family
// (attn, ffn, norm_pre, norm_post). grid_cells is the unique (bs, seq)
// count present; density is 'dense' when >=256 cells, 'thin' otherwise.
export interface PerOpCell {
  gpu: string;
  model: string;
  rows_per_op: Record<string, number>;
  ops_present: string[];
  ops_missing: string[];
  total_rows: number;
  grid_cells: number;
  density: 'dense' | 'thin' | 'partial';
  held_out: boolean;
  status: PredCoverageStatus;
  expected_grid: number;
}

export interface PredictorCoverage {
  generated_at: string;
  gpus: string[];
  models: string[];
  expected_ops: string[];
  per_kernel: {
    shared: KernelSharedRow[];
    cells: KernelModelCell[];
  };
  per_op: PerOpCell[];
}
