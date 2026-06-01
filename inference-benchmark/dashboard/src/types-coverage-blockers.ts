export interface CoveragePoint {
  hardware: string;
  model: string;
  backend: string;
  mode: string;
  profile: string;
  concurrency: number;
}

export interface CoverageFailure {
  category: string;
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
  coverage_disposition?: 'failed' | 'na' | null;
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
