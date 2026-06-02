export interface CoveragePoint {
  hardware: string;
  model: string;
  backend: string;
  mode: string;
  profile: string;
  concurrency: number;
  // Per-cell annotation (RFC §4.5): a job's missing cells can carry their own
  // disposition (e.g. low-success at high concurrency) distinct from the job.
  disposition?: 'failed' | 'na' | 'todo' | null;
  failure_class?: string | null;
  label?: string | null;
}

// Structured failure classes (see docs/coverage-classification-rfc.md). Captured
// at the launcher; the dashboard renders, it never re-classifies.
export type FailureClass =
  | 'none' | 'model_missing' | 'hw_infeasible' | 'oom_kv_cache' | 'engine_crash'
  | 'requests_aborted' | 'low_success_rate' | 'timeout' | 'incomplete_partial'
  | 'not_attempted' | 'driver_fault' | 'unknown';

export interface CoverageEvidence {
  gpu_mem_util?: number | null;
  outputs_present?: number | null;
  outputs_expected?: number | null;
  oom_log_excerpt?: string | null;
  success_rate?: number | null;
}

export interface CoverageFailure {
  category: string;            // alias of failure_class (legacy readers)
  failure_class?: FailureClass;
  evidence?: CoverageEvidence | null;
  label: string;
  kind?: string | null;
  status?: string | null;
  reason?: string | null;
  attempt?: number | null;
  max_attempts?: number | null;
  expected_outputs_present?: number | null;
  expected_outputs_total?: number | null;
  missing_outputs?: string[];
  remote_log?: string | null;
  mirror_status?: string | null;
  updated_at?: string | null;
}

export interface CoverageBlocker {
  attempt?: number | null;
  backend: string;
  coverage_disposition?: 'failed' | 'na' | 'todo' | null;
  coverage_failure_class?: FailureClass | null;
  coverage_label?: string | null;
  coverage_evidence?: CoverageEvidence | null;
  coverage_explanation?: string | null;
  expected: number;
  expected_points?: CoveragePoint[];
  failure?: CoverageFailure | null;
  hardware: string;
  host: string;
  job_id: string;
  missing: string;
  missing_count: number;
  missing_points?: CoveragePoint[];
  mode: string;
  model: string;
  present: number;
  present_points?: CoveragePoint[];
  reason?: string | null;
  scope: string;
  status: string;
  tp: number;
}

export interface CoverageBlockersState {
  blockers: CoverageBlocker[];
  coverage_failed_points?: number;
  coverage_missing_required_points?: number;
  coverage_na_points?: number;
  coverage_todo_points?: number;
  coverage_required_points?: number;
  data_rows: number;
  data_scopes: Record<string, number>;
  expected_points: number;
  failure_category_counts?: Record<string, number>;
  failure_disposition_counts?: Record<string, number>;
  failure_disposition_point_counts?: Record<string, number>;
  generated_at: string;
  job_status_counts: Record<string, number>;
  jobs?: CoverageBlocker[];
  jobs_total: number;
  jobs_with_missing_coverage: number;
  max_requeues: number;
  missing_jobs_by_status: Record<string, number>;
  missing_points: number;
  observed_present_points?: number;
  optional_present_points?: CoveragePoint[];
  optional_present_points_count?: number;
  present_points: number;
  reset_exhausted: string[];
  reset_performed: string[];
  reset_statuses: string[];
  scope: string;
  stale_terminal_jobs: number;
}
