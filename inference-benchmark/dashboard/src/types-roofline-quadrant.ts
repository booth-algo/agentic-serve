export interface QuadrantHardware {
  peak_tflops: number;
  memory_bw_tbs: number;
  hbm_gb: number;
}

export interface QuadrantPoint {
  model: string;
  profile: string;
  concurrency: number;
  engine: string;
  hardware: string;
  oi: number;
  cf_gb: number;
  output_tput: number;
  tpot_ms: number;
  ttft_ms: number;
}

export interface RooflineQuadrantData {
  hardware: QuadrantHardware;
  points: QuadrantPoint[];
}
