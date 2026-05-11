export type GpuStatus =
  | 'free'
  | 'sweep'
  | 'other-user'
  | 'same-user-nonsweep'
  | 'mixed-other-user'
  | 'mixed-same-user'
  | 'unknown-busy';

export interface GpuJobState {
  id: string;
  host: string;
  model_path: string;
  model_short: string;
  tp: number;
  mode: string;
  backend: string;
  scope: string;
  status: string;
  gpus: string[];
  port: string;
  attempt: string;
  age_seconds: number | null;
  age: string;
  max_len_override: string;
}

export interface GpuProcessState {
  gpu_index: string;
  gpu_uuid: string;
  pid: string;
  process_name: string;
  used_memory_mib: number | null;
  user: string;
  ppid: string;
  age_seconds: number | null;
  age: string;
  command: string;
  kind: 'sweep' | 'sweep-slot' | 'other-user' | 'same-user-nonsweep' | 'unknown';
}

export interface GpuPortState {
  port: string;
  detail: string;
}

export interface GpuDeviceState {
  index: string;
  uuid: string;
  name: string;
  memory_used_mib: number | null;
  memory_total_mib: number | null;
  util_pct: number | null;
  status: GpuStatus;
  assignments: GpuJobState[];
  processes: GpuProcessState[];
}

export interface GpuHostState {
  host: string;
  ok: boolean;
  remote_user: string;
  error: string;
  job_counts: Record<string, number>;
  jobs_total: number;
  running_jobs: GpuJobState[];
  ports: GpuPortState[];
  gpus: GpuDeviceState[];
  unmapped_processes: GpuProcessState[];
  gpu_status_counts?: Record<string, number>;
}

export interface GpuState {
  generated_at: string;
  jobs_file: string;
  state_dir: string;
  total_jobs: number;
  job_counts: Record<string, number>;
  summary: Record<string, number>;
  hosts: GpuHostState[];
  health?: string;
  error?: string;
}
