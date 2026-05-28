export type GpuStatus =
  | 'free'
  | 'sweep'
  | 'other-user'
  | 'same-user-nonsweep'
  | 'same-user-orphan'
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
  run_id?: string;
}

export interface GpuProcessState {
  gpu_index: string;
  gpu_uuid: string;
  pid: string;
  process_name: string;
  used_memory_mib: number | null;
  user: string;
  ppid: string;
  pgid?: string;
  sid?: string;
  stat?: string;
  age_seconds: number | null;
  age: string;
  command: string;
  parent_user?: string;
  parent_ppid?: string;
  parent_pgid?: string;
  parent_sid?: string;
  parent_stat?: string;
  parent_age_seconds?: number | null;
  parent_age?: string;
  parent_command?: string;
  grandparent_user?: string;
  grandparent_ppid?: string;
  grandparent_pgid?: string;
  grandparent_sid?: string;
  grandparent_stat?: string;
  grandparent_age_seconds?: number | null;
  grandparent_age?: string;
  grandparent_command?: string;
  bench_run_id?: string;
  bench_job_id?: string;
  bench_scope?: string;
  bench_port?: string;
  bench_gpus?: string;
  orphan_reason?: string;
  kind: 'sweep' | 'sweep-slot' | 'other-user' | 'same-user-nonsweep' | 'same-user-orphan' | 'unknown';
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
  blocked?: boolean;
  assignments: GpuJobState[];
  processes: GpuProcessState[];
}

export interface GpuHostState {
  host: string;
  ok: boolean;
  remote_user: string;
  error: string;
  drained?: boolean;
  blocked_gpus?: string[];
  job_counts: Record<string, number>;
  jobs_total: number;
  running_jobs: GpuJobState[];
  ports: GpuPortState[];
  gpus: GpuDeviceState[];
  unmapped_processes: GpuProcessState[];
  gpu_status_counts?: Record<string, number>;
}

export type OrchestratorHealth = 'running' | 'timer-active' | 'faulted' | 'not-installed' | 'inactive' | 'unknown';

export interface OrchestratorUnitState {
  id: string;
  ok: boolean;
  error?: string;
  load_state: string;
  active_state: string;
  sub_state: string;
  result: string;
  unit_file_state?: string;
  exec_main_code?: string;
  exec_main_status?: string;
  n_restarts?: string;
  active_enter_timestamp?: string;
  inactive_enter_timestamp?: string;
  state_change_timestamp?: string;
  next_elapse_realtime?: string;
  last_trigger?: string;
  stderr?: string;
}

export interface OrchestratorState {
  health: OrchestratorHealth;
  message: string;
  service: OrchestratorUnitState;
  timer: OrchestratorUnitState;
}

export interface GpuState {
  generated_at: string;
  orchestrator?: OrchestratorState;
  jobs_file: string;
  state_dir: string;
  control?: {
    drained_hosts?: string[];
    drained_hosts_file?: string;
    blocked_gpus?: Array<{ host: string; gpu: string }>;
    blocked_gpus_file?: string;
  };
  total_jobs: number;
  job_counts: Record<string, number>;
  summary: Record<string, number>;
  hosts: GpuHostState[];
  health?: string;
  error?: string;
}
