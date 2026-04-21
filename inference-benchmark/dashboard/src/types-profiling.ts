export interface ProfilingStatus {
  status: 'done' | 'missing' | 'infeasible' | 'partial' | 'pending_infra_access';
  reason?: string;
  rows?: number;
  version?: string;
}
export interface ProfilingCell {
  gpu: string;
  model: string;
  per_kernel_prefill: ProfilingStatus;
  per_kernel_roofline: ProfilingStatus;
  per_op_cuda_events: ProfilingStatus;
  per_op_trained_pkl: ProfilingStatus;
}

export interface PerKernelResults {
  heldout_mape_per_family?: Record<string, number>;
  aggregate_err_per_model?: Record<string, number>;
}
export interface PerOpResults {
  heldout_mape?: number | null;
  pool_models?: string[];
  heldout_models?: string[];
}
export interface PredictorResults {
  per_kernel?: Record<string, PerKernelResults>;
  per_op?: Record<string, PerOpResults>;
}

export interface ProfilingState {
  generated_at: string;
  cells: ProfilingCell[];
  gpus: string[];
  models: string[];
  results?: PredictorResults;
}
