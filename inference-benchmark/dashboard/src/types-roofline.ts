export interface RooflineKernel {
  name: string;
  category: string;
  cuda_time_us: number;
  calls: number;
  flops: number;
  arithmetic_intensity: number;
  achieved_tflops: number;
  model: string;
  phase: string;
  batch_size: number;
}

export interface RooflineCategorySummary {
  category: string;
  total_cuda_time_us: number;
  duration_pct: number;
  avg_arithmetic_intensity: number;
  avg_achieved_tflops: number;
  kernel_count: number;
}

export interface RooflineEntry {
  model: string;
  phase: string;
  batch_size: number;
  seq_len: number;
  total_cuda_time_us: number;
  kernels: RooflineKernel[];
  category_summary: RooflineCategorySummary[];
}

export interface RooflineData {
  entries: RooflineEntry[];
  models: string[];
  phases: string[];
  batch_sizes: number[];
}
