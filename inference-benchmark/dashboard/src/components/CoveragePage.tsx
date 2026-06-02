import { useMemo, useState } from 'react';
import { useCoverageBlockers } from '../hooks/useCoverageBlockers';
import type { BenchmarkResult } from '../types';
import type { CoverageBlocker, CoverageFailure, CoveragePoint } from '../types-coverage-blockers';
import type { SweepCell, SweepState } from '../types-sweep';
import { DATA_SCOPE_META, normalizeDataScope, profileDisplayName, type DataScope } from '../profileMeta';


interface CoveragePageProps {
  allData: BenchmarkResult[];
  sweepState: SweepState | null;
  loading: boolean;
  dataScope: DataScope;
}

const CURRENT_SINGLE_CONCS = [1, 10, 20, 40, 80, 160, 256, 320];
const CURRENT_MULTI_CONCS = [5, 20, 40, 80, 160];
const FIXED_SINGLE_CONCS = [200, 320];
const FIXED_MULTI_CONCS = [200, 320];
const MSE_SINGLE_CONCS: number[] = [];
const MSE_MULTI_CONCS = [40, 80];
const ARCHIVE_SINGLE_CONCS = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500];
const ARCHIVE_MULTI_CONCS = [5, 10, 20, 40, 80, 120, 160, 200, 256, 320];

const CURRENT_SINGLE_PROFILES = [
  'chat-singleturn',
  'coding-singleturn',
];
const CURRENT_MULTI_PROFILES = [
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
];
const FIXED_SINGLE_PROFILES = [
  'chat-singleturn',
];
const FIXED_MULTI_PROFILES = [
  'chat-multiturn',
  'swebench-multiturn',
  'terminalbench-multiturn',
  'osworld-multiturn',
];
const MSE_SINGLE_PROFILES: string[] = [];
const MSE_MULTI_PROFILES = [
  'swebench-multiturn-mse',
  'swebench-multiturn-short',
  'terminalbench-multiturn-mse',
  'terminalbench-multiturn-short',
  'osworld-multiturn-mse',
  'osworld-multiturn-short',
];
const ARCHIVE_SINGLE_PROFILES = [
  'chat-short', 'chat-medium', 'chat-singleturn',
  'coding-singleturn', 'prefill-heavy', 'decode-heavy', 'random-1k', 'fixed-seq128',
];
const ARCHIVE_MULTI_PROFILES = [
  'chat-multiturn-short', 'chat-multiturn-medium', 'chat-multiturn-long',
  'swebench-multiturn-short', 'swebench-multiturn-medium', 'swebench-multiturn-long',
  'terminalbench-multiturn-short', 'terminalbench-multiturn-medium', 'terminalbench-multiturn-long',
  'osworld-multiturn-short', 'osworld-multiturn-medium', 'osworld-multiturn-long',
];

const TP_OPTIONS = [1, 2, 4];

// Backends we always want a coverage row for. sglang is active now that
// the orchestrator routes by backend and all three hosts have sglang 0.5.9
// environments. Each (hw, model) gets a row for every active backend, plus
// any historical backend with data in data.json.
const ACTIVE_BACKENDS = ['vllm', 'sglang'];
const KNOWN_BACKENDS = ['vllm', 'sglang'];

type ModelFamily = 'Llama' | 'Qwen' | 'GPT-OSS' | 'Mixtral' | 'Gemma' | 'Granite' | 'Other';

const FAMILY_ORDER: ModelFamily[] = ['Llama', 'Qwen', 'GPT-OSS', 'Mixtral', 'Gemma', 'Granite', 'Other'];

function modelFamily(model: string): ModelFamily {
  const normalized = model.toLowerCase();
  if (normalized.startsWith('llama')) return 'Llama';
  if (normalized.startsWith('qwen')) return 'Qwen';
  if (normalized.startsWith('gpt-oss')) return 'GPT-OSS';
  if (normalized.startsWith('mixtral')) return 'Mixtral';
  if (normalized.startsWith('gemma')) return 'Gemma';
  if (normalized.startsWith('granite')) return 'Granite';
  return 'Other';
}

function compareModels(a: string, b: string): number {
  const familyDelta = FAMILY_ORDER.indexOf(modelFamily(a)) - FAMILY_ORDER.indexOf(modelFamily(b));
  if (familyDelta !== 0) return familyDelta;
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

interface ProfileRow {
  profile: string;
  isMultiTurn: boolean;
  expected: number[];
  present: Set<number>;
  blocked?: Map<number, string>;
  failed?: Map<number, string>;
  infeasibleReason?: string;
  todoReason?: string;
}

interface DataModel {
  kind: 'data';
  hardware: string;
  model: string;
  backend: string;
  engineVersion?: string;
  profiles: ProfileRow[];
  // Aggregate coverage across all profiles.
  totalHave: number;
  totalNeed: number;
}

interface StatusModel {
  kind: 'status';
  hardware: string;
  model: string;
  backend: string;
  status: 'oom' | 'untested' | 'infeasible' | 'running' | 'skipped' | 'pending' | 'failed';
  reason?: string;
  attempt?: number;
  maxAttempts?: number | null;
  failure?: CoverageFailure | null;
  updatedAt?: string | null;
  totalNeed: number;
  profiles?: ProfileRow[];
}

type ModelEntry = DataModel | StatusModel;

interface HwGroup {
  hardware: string;
  models: ModelEntry[];
  // Aggregate counts for the header summary.
  summary: {
    complete: number;  // model has data + all expected concs
    partial: number;   // model has data but incomplete
    running: number;
    pending: number;
    failed: number;
    skipped: number;
    oom: number;
    infeasible: number;
    untested: number;
    totalHave: number;
    totalNeed: number;
    failedCells: number;
  };
}

function hwLabel(base: string, tp: number): string {
  return tp === 1 ? base : `${base}x${tp}`;
}

function infeasibilityReason(
  vramGb: number | undefined,
  weightsGb: number | undefined,
  tp: number,
  ratio: number,
): string | null {
  if (!vramGb || !weightsGb) return null;
  const budget = vramGb * tp * ratio;
  if (weightsGb > budget) {
    const minGb = Math.ceil(weightsGb / ratio);
    return `needs ≥${minGb} GB VRAM (weights ${weightsGb} GB); this config has ${vramGb * tp} GB`;
  }
  return null;
}

const STATUS_PRIORITY: Record<SweepCell['status'], number> = {
  known_oom: 5, skipped: 4, failed: 4, running: 3, pending: 2, done: 1,
};

function aggregateCells(cells: SweepCell[]): Map<string, SweepCell> {
  const out = new Map<string, SweepCell>();
  for (const c of cells) {
    const key = `${c.hw_label}|${c.model}|${c.backend}`;
    const prev = out.get(key);
    if (!prev || STATUS_PRIORITY[c.status] > STATUS_PRIORITY[prev.status]) {
      out.set(key, c);
    }
  }
  return out;
}

function stateCellScope(cell: SweepCell): DataScope {
  return normalizeDataScope(cell.data_scope ?? null) ?? 'archived';
}

function isMultiTurnProfile(profile: string): boolean {
  return profile.includes('multiturn') || profile.includes('multi-turn');
}

function usesCanonicalCoverage(scope: DataScope): boolean {
  return scope === 'synthetic_distributional';
}

function coverageGridScope(scope: DataScope): DataScope {
  return scope;
}

function uniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values));
}

function uniqueNumbers(values: number[]): number[] {
  return Array.from(new Set(values)).sort((a, b) => a - b);
}

function sweepProfilesForMode(cells: SweepCell[], mode: SweepCell['mode']): string[] {
  return uniqueStrings(cells.flatMap((cell) => cell.mode === mode ? cell.profiles ?? [] : []));
}

function sweepConcurrenciesForMode(cells: SweepCell[], mode: SweepCell['mode']): number[] {
  return uniqueNumbers(cells.flatMap((cell) => cell.mode === mode ? cell.concurrencies ?? [] : []));
}

function jobIdForCell(cell: Pick<SweepCell, 'host' | 'model' | 'tp' | 'mode' | 'backend'>): string {
  const base = `${cell.host}_${cell.model}_tp${cell.tp}_${cell.mode}`;
  return cell.backend && cell.backend !== 'vllm' ? `${base}_${cell.backend}` : base;
}

function pointKey(hw: string, model: string, backend: string, profile: string, concurrency: number): string {
  return `${hw}|${model}|${backend}|${profile}|${concurrency}`;
}

function pointKeyFromSummary(point: CoveragePoint): string {
  return pointKey(point.hardware, point.model, point.backend, point.profile, point.concurrency);
}

function profileInfeasibleKeys(hw: string, model: string, backend: string, profile: string): string[] {
  const profiles = new Set([profile]);
  if (profile.endsWith('-synth')) {
    profiles.add(profile.replace(/-synth$/, ''));
  } else {
    profiles.add(`${profile}-synth`);
  }
  return Array.from(profiles, (p) => `${hw}|${model}|${backend}|${p}`);
}

function summarizeReasons(reasons: Array<string | undefined>, fallback?: string): string | undefined {
  const unique = Array.from(new Set(reasons.filter((reason): reason is string => Boolean(reason))));
  if (unique.length === 0) return fallback;
  if (unique.length === 1) return unique[0];
  return unique.join(' | ');
}

function failureLabel(failure?: CoverageFailure | null): string | undefined {
  if (!failure) return undefined;
  const attempt = failure.attempt;
  const maxAttempts = failure.max_attempts;
  const attempts = attempt == null
    ? null
    : maxAttempts == null ? `${attempt} attempts` : `${attempt}/${maxAttempts} attempts`;
  return `${failure.label}${attempts ? ` after ${attempts}` : ''}`;
}

function failureReason(failure?: CoverageFailure | null, fallback?: string | null): string | undefined {
  const label = failureLabel(failure);
  const reason = failure?.reason ?? fallback ?? undefined;
  if (label && reason) return `${label}: ${reason}`;
  return label ?? reason;
}

// LEGACY — non-compact scopes only (no coverage artifact, e.g. some
// trace_replay/archived cells). The synthetic_distributional path is driven
// entirely by the artifact's `coverage_disposition` (see disposition() below);
// this regex is NOT the source of truth and must not be extended for policy.
// Coverage classification policy lives in reconcile_sweep_coverage.py.
// (RFC: docs/coverage-classification-rfc.md §4.3 — dumb renderer.)
function failureCategoryFromReason(reason?: string | null): string | undefined {
  const lower = (reason ?? '').toLowerCase();
  if (!lower) return undefined;
  if (/(xid|nvml|driver|gpu has fallen off|cuda error|uncorrectable|nvidia-smi|cuda initialization)/.test(lower)) return 'driver_failure';
  if (/(out of memory|cuda out of memory|kv-cache|kv cache|cache blocks)/.test(lower)) return 'oom_or_kv_cache';
  if (lower.includes('success rate') && lower.includes('below minimum')) return 'success_rate_below_min';
  if (lower.includes('[warn]') && lower.includes('failed')) return 'benchmark_failed';
  if (lower.includes('zero results') || lower.includes('zero expected outputs') || lower.includes('no usable temporary directory')) return 'zero_results';
  if (lower.includes('incomplete') || lower.includes('expected outputs missing')) return 'incomplete_outputs';
  return 'unknown';
}

function failureCategory(failure?: CoverageFailure | null, fallback?: string | null): string | undefined {
  return failure?.category ?? failureCategoryFromReason(failure?.reason ?? fallback);
}

// Legacy N/A categories for the non-compact fallback only (matches
// classifySweepFailure's vocabulary). The compact artifact never reaches here.
const N_A_FAILURE_CATEGORIES = new Set([
  'oom_or_kv_cache',
  'success_rate_below_min',
  'zero_results',
  'incomplete_outputs',
]);

function failureCoverageDisposition(
  failure?: CoverageFailure | null,
  fallback?: string | null,
  explicit?: CoverageBlocker['coverage_disposition'],
): 'failed' | 'na' | undefined {
  // The coverage artifact is authoritative (dumb renderer): when an explicit
  // disposition is present we NEVER re-derive from the reason string.
  if (explicit === 'failed' || explicit === 'na') return explicit;
  if (explicit === 'todo') return undefined;  // fillable work, not a blocked cell
  // Legacy fallback: only scopes without a coverage artifact get here.
  const category = failureCategory(failure, fallback);
  if (!category) return undefined;
  return N_A_FAILURE_CATEGORIES.has(category) ? 'na' : 'failed';
}

function shouldCountFailureAsMissing(
  failure?: CoverageFailure | null,
  fallback?: string | null,
  explicit?: CoverageBlocker['coverage_disposition'],
): boolean {
  return failureCoverageDisposition(failure, fallback, explicit) === 'failed';
}

const TERMINAL_BLOCKER_STATUSES = new Set(['skipped', 'failed', 'known_oom']);

// Coverage artifact older than this (minutes) -> warn that it may be stale
// (the orchestrator regenerates it every tick; staleness means it stopped).
// RFC: docs/coverage-classification-rfc.md §4.7 (provenance).
const COVERAGE_STALE_MINUTES = 20;

function shouldRenderBlockedPoints(blocker: CoverageBlocker, exhaustedJobIds: Set<string>): boolean {
  if ((blocker.missing_points ?? []).length === 0) return false;
  // Demoted to TODO (attempted but no captured OOM): render as outstanding work,
  // not as a blocked N/A cell. Must precede the exhausted check because these
  // jobs are also in reset_exhausted.
  if (blocker.coverage_disposition === 'todo') return false;
  if (exhaustedJobIds.has(blocker.job_id)) return true;
  if (blocker.status === 'known_oom') return true;
  return Boolean(blocker.failure) && TERMINAL_BLOCKER_STATUSES.has(blocker.status);
}

function classifySweepFailure(failure?: SweepCell['failure_metadata'] | null): CoverageFailure | null {
  if (!failure?.reason) return null;
  const lower = failure.reason.toLowerCase();
  let category = 'unknown';
  let label = 'unknown failure';
  if (/(xid|nvml|driver|gpu has fallen off|cuda error|uncorrectable|nvidia-smi|cuda initialization)/.test(lower)) {
    category = 'driver_failure';
    label = 'driver failure';
  } else if (/(out of memory|cuda out of memory|kv-cache|kv cache|cache blocks)/.test(lower)) {
    category = 'oom_or_kv_cache';
    label = 'OOM / KV-cache limit';
  } else if (lower.includes('success rate') && lower.includes('below minimum')) {
    category = 'success_rate_below_min';
    label = 'success rate below threshold';
  } else if (lower.includes('[warn]') && lower.includes('failed')) {
    category = 'benchmark_failed';
    label = 'benchmark command failed';
  } else if (lower.includes('zero results') || lower.includes('zero expected outputs')) {
    category = 'zero_results';
    label = 'zero results';
  } else if (lower.includes('incomplete') || lower.includes('expected outputs missing')) {
    category = 'incomplete_outputs';
    label = 'incomplete outputs';
  }
  return { ...failure, category, label };
}

export function CoveragePage({
  allData,
  sweepState,
  loading,
  dataScope,
}: CoveragePageProps) {
  const canonicalCoverage = usesCanonicalCoverage(dataScope);
  const gridScope = coverageGridScope(dataScope);
  const {
    blockersState,
    loading: blockersLoading,
  } = useCoverageBlockers(dataScope === 'synthetic_distributional');
  const compactCoverageJobs = dataScope === 'synthetic_distributional' ? blockersState?.jobs ?? [] : [];
  const usingCompactCoverage = dataScope === 'synthetic_distributional' && compactCoverageJobs.length > 0;

  const coveragePlan = useMemo(() => {
    const scopedSweepCells = sweepState?.cells.filter((cell) => stateCellScope(cell) === gridScope) ?? [];
    const singleProfiles = gridScope === 'trace_replay'
      ? ARCHIVE_SINGLE_PROFILES
      : gridScope === 'synthetic_distributional'
        ? sweepProfilesForMode(scopedSweepCells, 'single')
        : uniqueStrings([...CURRENT_SINGLE_PROFILES, ...FIXED_SINGLE_PROFILES, ...MSE_SINGLE_PROFILES]);
    const multiProfiles = gridScope === 'trace_replay'
      ? ARCHIVE_MULTI_PROFILES
      : gridScope === 'synthetic_distributional'
        ? sweepProfilesForMode(scopedSweepCells, 'multi')
        : uniqueStrings([...CURRENT_MULTI_PROFILES, ...FIXED_MULTI_PROFILES, ...MSE_MULTI_PROFILES]);
    const singleConcs = gridScope === 'trace_replay'
      ? ARCHIVE_SINGLE_CONCS
      : gridScope === 'synthetic_distributional'
        ? sweepConcurrenciesForMode(scopedSweepCells, 'single')
        : uniqueNumbers([...CURRENT_SINGLE_CONCS, ...FIXED_SINGLE_CONCS, ...MSE_SINGLE_CONCS]);
    const multiConcs = gridScope === 'trace_replay'
      ? ARCHIVE_MULTI_CONCS
      : gridScope === 'synthetic_distributional'
        ? sweepConcurrenciesForMode(scopedSweepCells, 'multi')
        : uniqueNumbers([...CURRENT_MULTI_CONCS, ...FIXED_MULTI_CONCS, ...MSE_MULTI_CONCS]);
    return {
      singleProfiles,
      multiProfiles,
      singleConcs,
      multiConcs,
      expectedCellsPerModel: singleProfiles.length * singleConcs.length + multiProfiles.length * multiConcs.length,
    };
  }, [gridScope, sweepState]);

  const { groups, hardwareList } = useMemo(() => {
    const { singleProfiles, multiProfiles, singleConcs, multiConcs, expectedCellsPerModel } = coveragePlan;
    const scopedSweepCells = sweepState?.cells.filter((cell) => stateCellScope(cell) === gridScope) ?? [];
    const baseHwLabels = sweepState
      ? Object.values(sweepState.hosts).map((h) => h.hardware_label)
      : ['A100-40GB', '3090', '2080Ti', 'H100'];
    const dataHw = new Set(
      usingCompactCoverage
        ? compactCoverageJobs.map((job) => job.hardware)
        : allData.map((r) => r.hardware),
    );
    const expectedHw: string[] = [];
    if (!canonicalCoverage) {
      expectedHw.push(...Array.from(dataHw).sort());
    } else if (dataScope === 'synthetic_distributional') {
      const scopedHw = new Set<string>([...scopedSweepCells.map((cell) => cell.hw_label), ...dataHw]);
      expectedHw.push(...Array.from(scopedHw).sort());
    } else {
      for (const base of baseHwLabels) {
        for (const tp of TP_OPTIONS) expectedHw.push(hwLabel(base, tp));
      }
      for (const hw of dataHw) {
        if (!hw.endsWith('x8') && !expectedHw.includes(hw)) expectedHw.push(hw);
      }
    }

    const expectedModels = new Set<string>();
    if (canonicalCoverage && sweepState) {
      if (dataScope === 'synthetic_distributional') {
        for (const cell of scopedSweepCells) expectedModels.add(cell.model);
      } else {
        for (const m of Object.keys(sweepState.models)) expectedModels.add(m);
      }
    }
    for (const r of allData) expectedModels.add(r.modelShort);
    for (const job of compactCoverageJobs) expectedModels.add(job.model);
    const modelList = Array.from(expectedModels).sort(compareModels);

    const vramByBase = new Map<string, number>();
    if (sweepState) {
      for (const h of Object.values(sweepState.hosts)) vramByBase.set(h.hardware_label, h.vram_gb_per_gpu);
    }
    const vramFor = (hw: string): number | undefined => {
      const m = hw.match(/^(.+?)(?:x(\d+))?$/);
      return m ? vramByBase.get(m[1]) : undefined;
    };
    const tpOf = (hw: string): number => {
      const m = hw.match(/x(\d+)$/);
      return m ? parseInt(m[1], 10) : 1;
    };
    const weightsFor = (model: string): number | undefined =>
      sweepState?.models[model]?.weights_gb;
    const ratio = sweepState?.feasibility_ratio ?? 0.85;
    const profileInfeasible = new Map<string, string>();
    if (canonicalCoverage) {
      for (const item of sweepState?.profile_infeasible ?? []) {
        if ((normalizeDataScope(item.data_scope ?? null) ?? 'archived') !== gridScope) continue;
        for (const key of profileInfeasibleKeys(item.hw_label, item.model, item.backend, item.profile)) {
          profileInfeasible.set(key, item.reason);
        }
      }
    }
    const profileInfeasibleReasonFor = (
      hw: string,
      model: string,
      backend: string,
      profile: string,
    ): string | undefined => {
      for (const key of profileInfeasibleKeys(hw, model, backend, profile)) {
        const reason = profileInfeasible.get(key);
        if (reason) return reason;
      }
      return undefined;
    };

    const sweepCellsByMb = new Map<string, SweepCell[]>();
    if (dataScope === 'synthetic_distributional') {
      for (const cell of scopedSweepCells) {
        const key = `${cell.hw_label}|${cell.model}|${cell.backend}`;
        const cells = sweepCellsByMb.get(key) ?? [];
        cells.push(cell);
        sweepCellsByMb.set(key, cells);
      }
    }
    const sweepCellsFor = (hw: string, model: string, backend: string): SweepCell[] =>
      sweepCellsByMb.get(`${hw}|${model}|${backend}`) ?? [];
    const sweepConcsForProfile = (
      hw: string,
      model: string,
      backend: string,
      profile: string,
      doneOnly: boolean,
    ): number[] => {
      const concs = new Set<number>();
      for (const cell of sweepCellsFor(hw, model, backend)) {
        if (doneOnly && cell.status !== 'done') continue;
        if (!(cell.profiles ?? []).includes(profile)) continue;
        for (const concurrency of cell.concurrencies ?? []) concs.add(concurrency);
      }
      return uniqueNumbers(Array.from(concs));
    };
    const hasDoneSweepCells = (hw: string, model: string, backend: string): boolean =>
      sweepCellsFor(hw, model, backend).some((cell) => cell.status === 'done');
    const buildStatusProfiles = (
      hw: string,
      model: string,
      backend: string,
      status: StatusModel['status'],
      reason?: string,
    ): ProfileRow[] => {
      const specs = new Map<string, { profile: string; isMultiTurn: boolean; expected: Set<number> }>();
      const addSpec = (profile: string, isMultiTurn: boolean, concurrencies: number[]) => {
        const spec = specs.get(profile) ?? { profile, isMultiTurn, expected: new Set<number>() };
        spec.isMultiTurn = spec.isMultiTurn || isMultiTurn;
        for (const concurrency of concurrencies) spec.expected.add(concurrency);
        specs.set(profile, spec);
      };

      if (dataScope === 'synthetic_distributional') {
        for (const cell of sweepCellsFor(hw, model, backend)) {
          const isMultiTurn = cell.mode === 'multi';
          for (const profile of cell.profiles ?? []) {
            addSpec(profile, isMultiTurn, cell.concurrencies ?? []);
          }
        }
      } else {
        for (const profile of singleProfiles) addSpec(profile, false, singleConcs);
        for (const profile of multiProfiles) addSpec(profile, true, multiConcs);
      }

      const rows: ProfileRow[] = [];
      const modelLevelNa = status === 'oom' || status === 'skipped' || status === 'infeasible';
      const modelLevelFailed = status === 'failed';
      const todoReason = reason ?? 'expected by sweep grid; no completed run exists yet';
      for (const spec of specs.values()) {
        const expected = uniqueNumbers(Array.from(spec.expected));
        const profileReason = profileInfeasibleReasonFor(hw, model, backend, spec.profile);
        const infeasibleReason = profileReason ?? (status === 'infeasible' ? reason : undefined);
        const blocked = new Map<number, string>();
        const failed = new Map<number, string>();
        if (modelLevelNa && !infeasibleReason) {
          const blockedReason = reason ?? labelForStatus(status);
          for (const concurrency of expected) blocked.set(concurrency, blockedReason);
        }
        if (modelLevelFailed && !infeasibleReason) {
          const failedReason = reason ?? labelForStatus(status);
          for (const concurrency of expected) {
            failed.set(
              concurrency,
              failedPointReasons.get(pointKey(hw, model, backend, spec.profile, concurrency)) ?? failedReason,
            );
          }
        }
        rows.push({
          profile: spec.profile,
          isMultiTurn: spec.isMultiTurn,
          expected,
          present: new Set<number>(),
          blocked,
          failed,
          infeasibleReason,
          todoReason: status === 'untested' || status === 'pending' ? todoReason : undefined,
        });
      }
      return rows;
    };

    const expectedCountFor = (hw: string, model: string, backend: string): number => {
      // For scoped sweeps, derive expected count from the actual sweep-state cells,
      // not the full profile×concurrency cross-product.
      if (dataScope === 'synthetic_distributional') {
        let total = 0;
        for (const cell of sweepCellsFor(hw, model, backend)) {
          for (const profile of cell.profiles ?? []) {
            if (!profileInfeasibleReasonFor(hw, model, backend, profile)) {
              total += (cell.concurrencies ?? []).length;
            }
          }
        }
        return total;
      }
      let total = 0;
      for (const profile of singleProfiles) {
        if (!profileInfeasibleReasonFor(hw, model, backend, profile)) total += singleConcs.length;
      }
      for (const profile of multiProfiles) {
        if (!profileInfeasibleReasonFor(hw, model, backend, profile)) total += multiConcs.length;
      }
      return total;
    };

    const exhaustedJobIds = new Set(blockersState?.reset_exhausted ?? []);
    const blockerByJobId = new Map<string, CoverageBlocker>();
    const blockedPointReasons = new Map<string, string>();
    const failedPointReasons = new Map<string, string>();
    for (const blocker of blockersState?.blockers ?? []) {
      blockerByJobId.set(blocker.job_id, blocker);
      if (!shouldRenderBlockedPoints(blocker, exhaustedJobIds)) continue;
      const reason = blocker.coverage_explanation
        ?? blocker.reason
        ?? failureReason(blocker.failure, blocker.reason)
        ?? `coverage requeue exhausted after ${blocker.attempt ?? 'unknown'} attempts`;
      // Per-cell granularity (RFC §4.5): each missing point can carry its own
      // disposition, so a job that serves at low concurrency but is quality-
      // rejected at high concurrency splits across tones instead of inheriting
      // one job-level disposition. Falls back to the job disposition.
      const jobDisposition = shouldCountFailureAsMissing(blocker.failure, blocker.reason, blocker.coverage_disposition)
        ? 'failed'
        : 'na';
      for (const point of blocker.missing_points ?? []) {
        const key = pointKeyFromSummary(point);
        const cellDisposition = point.disposition ?? jobDisposition;
        const cellReason = point.label ?? reason;
        if (cellDisposition === 'failed') {
          failedPointReasons.set(key, cellReason);
        } else if (cellDisposition !== 'todo') {
          blockedPointReasons.set(key, cellReason);  // na
        }
        // cellDisposition === 'todo' -> leave unblocked; renders as fillable work
      }
    }

    const compactJobById = new Map<string, CoverageBlocker>();
    for (const job of compactCoverageJobs) compactJobById.set(job.job_id, job);

    const bucket = new Map<string, Set<number>>();
    const mbHasData = new Map<string, Set<string>>();  // hw -> Set<"model|backend">
    const engineVersionByMb = new Map<string, string>();  // "hw|model|backend" -> version
    const profilesByMb = new Map<string, Set<string>>();  // "hw|model|backend" -> profiles with data
    if (usingCompactCoverage) {
      for (const job of compactCoverageJobs) {
        const mbKey = `${job.hardware}|${job.model}|${job.backend}`;
        if (job.present > 0) {
          if (!mbHasData.has(job.hardware)) mbHasData.set(job.hardware, new Set());
          mbHasData.get(job.hardware)!.add(`${job.model}|${job.backend}`);
        }
        for (const point of job.present_points ?? []) {
          const k = `${point.hardware}|${point.model}|${point.backend}|${point.profile}`;
          if (!bucket.has(k)) bucket.set(k, new Set());
          bucket.get(k)!.add(point.concurrency);
          if (!profilesByMb.has(mbKey)) profilesByMb.set(mbKey, new Set());
          profilesByMb.get(mbKey)!.add(point.profile);
        }
      }
    } else {
      for (const r of allData) {
        const backend = r.config.backend;
        const k = `${r.hardware}|${r.modelShort}|${backend}|${r.config.profile}`;
        if (!bucket.has(k)) bucket.set(k, new Set());
        bucket.get(k)!.add(r.config.concurrency);
        if (!mbHasData.has(r.hardware)) mbHasData.set(r.hardware, new Set());
        mbHasData.get(r.hardware)!.add(`${r.modelShort}|${backend}`);
        const mbKey = `${r.hardware}|${r.modelShort}|${backend}`;
        if (!profilesByMb.has(mbKey)) profilesByMb.set(mbKey, new Set());
        profilesByMb.get(mbKey)!.add(r.config.profile);
        if (r.engineVersion && !engineVersionByMb.has(mbKey)) {
          engineVersionByMb.set(mbKey, r.engineVersion);
        }
      }
    }

    const aggStatus = sweepState
      ? aggregateCells(scopedSweepCells)
      : new Map<string, SweepCell>();

    const hwGroups: HwGroup[] = [];
    for (const hw of expectedHw) {
      const models: ModelEntry[] = [];
      const summary = {
        complete: 0, partial: 0,
        running: 0, pending: 0, failed: 0, skipped: 0,
        oom: 0, infeasible: 0, untested: 0,
        totalHave: 0, totalNeed: 0, failedCells: 0,
      };
      for (const model of modelList) {
        // Always include ACTIVE_BACKENDS (current sweep target) plus any
        // other known backend that actually has data for this (hw, model).
        const backendSet = new Set<string>();
        if (canonicalCoverage) {
          if (dataScope === 'synthetic_distributional') {
            for (const cell of scopedSweepCells) {
              if (cell.hw_label === hw && cell.model === model) backendSet.add(cell.backend);
            }
            for (const item of mbHasData.get(hw) ?? []) {
              const [dataModel, dataBackend] = item.split('|');
              if (dataModel === model && dataBackend) backendSet.add(dataBackend);
            }
          } else {
            for (const b of ACTIVE_BACKENDS) backendSet.add(b);
            for (const b of KNOWN_BACKENDS) {
              if (mbHasData.get(hw)?.has(`${model}|${b}`)) backendSet.add(b);
            }
          }
        } else {
          for (const item of mbHasData.get(hw) ?? []) {
            const [dataModel, dataBackend] = item.split('|');
            if (dataModel === model && dataBackend) backendSet.add(dataBackend);
          }
        }
        const backendsForCell = Array.from(backendSet).sort();
        for (const backend of backendsForCell) {
          const mbKey = `${hw}|${model}|${backend}`;
          const hasData = (mbHasData.get(hw)?.has(`${model}|${backend}`) ?? false)
            || (!usingCompactCoverage && dataScope === 'synthetic_distributional' && hasDoneSweepCells(hw, model, backend));
          const cell = aggStatus.get(mbKey);
            const cellJobId = cell ? jobIdForCell(cell) : null;
            const compactJob = cellJobId ? compactJobById.get(cellJobId) : undefined;
            const blocker = cellJobId ? blockerByJobId.get(cellJobId) : undefined;
            const stateReason = blocker?.coverage_explanation ?? compactJob?.coverage_explanation ?? blocker?.reason ?? compactJob?.reason ?? cell?.reason;
            const expectedForModel = canonicalCoverage
              ? expectedCountFor(hw, model, backend)
              : expectedCellsPerModel;

          if (hasData) {
            const profiles: ProfileRow[] = [];
            let totalHave = 0;
            let totalNeed = 0;
            if (!canonicalCoverage) {
              const mbKey = `${hw}|${model}|${backend}`;
              const observedProfiles = Array.from(profilesByMb.get(mbKey) ?? []).sort();
              for (const profile of observedProfiles) {
                const present = bucket.get(`${mbKey}|${profile}`) ?? new Set<number>();
                const observedConcs = Array.from(present).sort((a, b) => a - b);
                totalHave += observedConcs.length;
                totalNeed += observedConcs.length;
                profiles.push({ profile, isMultiTurn: isMultiTurnProfile(profile), expected: observedConcs, present });
              }
            } else {
              const isScoped = dataScope === 'synthetic_distributional';
              const addProfile = (profile: string, isMultiTurn: boolean) => {
                const observed = bucket.get(`${mbKey}|${profile}`);
                const present = new Set<number>(observed ? Array.from(observed) : []);
                const expected = isScoped
                  ? sweepConcsForProfile(hw, model, backend, profile, false)
                  : isMultiTurn ? multiConcs : singleConcs;
                const blocked = new Map<number, string>();
                const failed = new Map<number, string>();
                for (const concurrency of expected) {
                  const reason = blockedPointReasons.get(pointKey(hw, model, backend, profile, concurrency));
                  if (reason) blocked.set(concurrency, reason);
                  const failedReason = failedPointReasons.get(pointKey(hw, model, backend, profile, concurrency));
                  if (failedReason) failed.set(concurrency, failedReason);
                }
                if (isScoped) {
                  for (const concurrency of sweepConcsForProfile(hw, model, backend, profile, true)) {
                    present.add(concurrency);
                  }
                }
                const infeasibleReason = profileInfeasibleReasonFor(hw, model, backend, profile);
                if (isScoped && expected.length === 0 && present.size === 0 && !infeasibleReason) return;
                if (!infeasibleReason) {
                  const have = [...present].filter((c) => expected.includes(c) && !blocked.has(c)).length;
                  const failedMissing = expected.filter((c) => failed.has(c) && !blocked.has(c) && !present.has(c)).length;
                  totalHave += have;
                  totalNeed += expected.filter((c) => !blocked.has(c)).length;
                  summary.failedCells += failedMissing;
                }
                profiles.push({ profile, isMultiTurn, expected, present, blocked, failed, infeasibleReason });
              };
              for (const profile of singleProfiles) addProfile(profile, false);
              for (const profile of multiProfiles) addProfile(profile, true);
            }
            const engineVersion = engineVersionByMb.get(mbKey);
            models.push({ kind: 'data', hardware: hw, model, backend, engineVersion, profiles, totalHave, totalNeed });
            summary.totalHave += totalHave;
            summary.totalNeed += totalNeed;
            if (totalNeed === 0) summary.skipped += 1;
            else if (totalHave === totalNeed) summary.complete += 1;
            else if (profiles.some((profile) => (profile.failed?.size ?? 0) > 0)) summary.failed += 1;
            else summary.partial += 1;
            continue;
          }

          if (cell) {
            const failure = blocker?.failure ?? compactJob?.failure ?? classifySweepFailure(cell.failure_metadata);
            const statusReason = stateReason ? String(stateReason) : failureReason(failure);
            const attempt = failure?.attempt ?? cell.attempt;
            const maxAttempts = failure?.max_attempts ?? null;
            if (cell.status === 'known_oom') {
              const profiles = buildStatusProfiles(hw, model, backend, 'oom', statusReason);
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'oom', reason: statusReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: 0, profiles });
              summary.oom += 1;
              continue;
            }
            const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
            if (infReason) {
              const profiles = buildStatusProfiles(hw, model, backend, 'infeasible', infReason);
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'infeasible', reason: infReason, totalNeed: 0, profiles });
              summary.infeasible += 1;
              continue;
            }
            const baseProfiles = buildStatusProfiles(hw, model, backend, 'untested', statusReason);
            const profileBlockedReason = summarizeReasons(baseProfiles.map((profile) => profile.infeasibleReason));
            if (baseProfiles.length > 0 && baseProfiles.every((profile) => Boolean(profile.infeasibleReason))) {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'infeasible', reason: profileBlockedReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: 0, profiles: baseProfiles });
              summary.infeasible += 1;
              continue;
            }
            if (cell.status === 'running') {
              const profiles = buildStatusProfiles(hw, model, backend, 'running', statusReason);
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'running', reason: statusReason, attempt: cell.attempt, updatedAt: cell.updated_at, totalNeed: expectedForModel, profiles });
              summary.running += 1;
              summary.totalNeed += expectedForModel;
              continue;
            }
            if (cell.status === 'skipped') {
              const disposition = blocker?.coverage_disposition ?? compactJob?.coverage_disposition;
              if (disposition === 'todo') {
                // Attempted but produced nothing and no OOM was captured: not
                // proven infeasible, so surface as outstanding TODO work.
                const profiles = buildStatusProfiles(hw, model, backend, 'untested', statusReason);
                models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested', reason: statusReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: expectedForModel, profiles });
                summary.untested += 1;
                summary.totalNeed += expectedForModel;
                continue;
              }
              if (shouldCountFailureAsMissing(failure, statusReason, disposition)) {
                const profiles = buildStatusProfiles(hw, model, backend, 'failed', statusReason);
                const failureReasons = Array.from(new Set(
                  profiles.flatMap((profile) => Array.from(profile.failed?.values() ?? [])),
                ));
                const reasonSummary = summarizeReasons(failureReasons, statusReason);
                const failedReason = reasonSummary && expectedForModel > 0
                  ? `failed ${expectedForModel}/${expectedForModel} expected cells${failureReasons.length > 1 ? ` across ${failureReasons.length} failure groups` : ''}; ${reasonSummary}`
                  : reasonSummary;
                models.push({ kind: 'status', hardware: hw, model, backend, status: 'failed', reason: failedReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: expectedForModel, profiles });
                summary.failed += 1;
                summary.totalNeed += expectedForModel;
                summary.failedCells += expectedForModel;
                continue;
              }
              const profiles = buildStatusProfiles(hw, model, backend, 'skipped', statusReason);
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'skipped', reason: statusReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: 0, profiles });
              summary.skipped += 1;
              continue;
            }
            if (cell.status === 'pending' || cell.status === 'done') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested', reason: statusReason, attempt: attempt ?? undefined, maxAttempts, failure, totalNeed: expectedForModel, profiles: baseProfiles });
              summary.untested += 1;
              summary.totalNeed += expectedForModel;
              continue;
            }
          }

          const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
          if (infReason) {
            const profiles = buildStatusProfiles(hw, model, backend, 'infeasible', infReason);
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'infeasible', reason: infReason, totalNeed: 0, profiles });
            summary.infeasible += 1;
          } else {
            const profiles = buildStatusProfiles(hw, model, backend, 'untested');
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested', totalNeed: expectedForModel, profiles });
            summary.untested += 1;
            summary.totalNeed += expectedForModel;
          }
        }
      }
      hwGroups.push({ hardware: hw, models, summary });
    }

    return { groups: hwGroups, hardwareList: expectedHw };
  }, [allData, blockersState, canonicalCoverage, compactCoverageJobs, coveragePlan, dataScope, gridScope, sweepState, usingCompactCoverage]);

  const [expandedHw, setExpandedHw] = useState<Set<string>>(new Set());
  const [expandedModel, setExpandedModel] = useState<Set<string>>(new Set());

  const toggleHw = (hw: string) => {
    setExpandedHw((prev) => {
      const next = new Set(prev);
      if (next.has(hw)) next.delete(hw); else next.add(hw);
      return next;
    });
  };
  const toggleModel = (key: string) => {
    setExpandedModel((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };
    const expandAll = () => {
      setExpandedHw(new Set(groups.map((g) => g.hardware)));
      const keys = new Set<string>();
      for (const g of groups) {
        for (const m of g.models) {
          if (m.kind === 'data' || (m.profiles?.length ?? 0) > 0) {
            keys.add(`${g.hardware}|${m.model}|${m.backend}`);
          }
        }
      }
      setExpandedModel(keys);
    };
  const collapseAll = () => {
    setExpandedHw(new Set());
    setExpandedModel(new Set());
  };

  const allConcs = useMemo(
    () => {
      if (!canonicalCoverage) {
        const observed = new Set<number>();
        for (const r of allData) observed.add(r.config.concurrency);
        return Array.from(observed).sort((a, b) => a - b);
      }
      return Array.from(new Set([...coveragePlan.singleConcs, ...coveragePlan.multiConcs])).sort((a, b) => a - b);
    },
    [allData, canonicalCoverage, coveragePlan],
  );

  if (loading || (dataScope === 'synthetic_distributional' && blockersLoading && !blockersState)) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-[#8b949e]">
          {dataScope === 'synthetic_distributional' ? 'Loading compact coverage summary...' : 'Loading coverage data...'}
        </div>
      </div>
    );
  }

  const grand = groups.reduce(
    (acc, g) => {
      acc.complete += g.summary.complete;
      acc.partial += g.summary.partial;
      acc.running += g.summary.running;
      acc.pending += g.summary.pending;
      acc.failed += g.summary.failed;
      acc.skipped += g.summary.skipped;
      acc.oom += g.summary.oom;
      acc.infeasible += g.summary.infeasible;
      acc.untested += g.summary.untested;
      acc.totalHave += g.summary.totalHave;
      acc.totalNeed += g.summary.totalNeed;
      acc.failedCells += g.summary.failedCells;
      return acc;
    },
    { complete: 0, partial: 0, running: 0, pending: 0, failed: 0, skipped: 0, oom: 0, infeasible: 0, untested: 0, totalHave: 0, totalNeed: 0, failedCells: 0 },
  );
  const optionalPresentCount = dataScope === 'synthetic_distributional'
    ? blockersState?.optional_present_points_count ?? 0
    : 0;
  const displayedHave = dataScope === 'synthetic_distributional'
    ? blockersState?.observed_present_points ?? grand.totalHave
    : grand.totalHave;
  const requiredTotal = dataScope === 'synthetic_distributional'
    ? blockersState?.coverage_required_points ?? grand.totalNeed
    : grand.totalNeed;
  const failedCells = dataScope === 'synthetic_distributional'
    ? blockersState?.coverage_failed_points ?? grand.failedCells
    : grand.failedCells;
  const naAttemptedCells = dataScope === 'synthetic_distributional'
    ? blockersState?.coverage_na_points ?? 0
    : 0;
  const pct = requiredTotal > 0
    ? ((displayedHave / requiredTotal) * 100).toFixed(1)
    : '0.0';
  const cellSummary = !canonicalCoverage
    ? `${grand.totalHave} cells filled`
    : `${displayedHave}/${requiredTotal} fillable filled${optionalPresentCount > 0 ? ` · ${optionalPresentCount} optional` : ''}${naAttemptedCells > 0 ? ` · ${naAttemptedCells} N/A attempted` : ''}${failedCells > 0 ? ` · ${failedCells} failed` : ''}`;
  const primarySummary = !canonicalCoverage ? `${grand.totalHave} cells filled` : `${pct}%`;
  const scopeSummary = dataScope === 'synthetic_distributional'
    ? 'APC-aware synthetic profiles on the active sweep-state grid'
    : dataScope === 'archived'
      ? 'retired canonical, fixed-grid, and MSE runs kept as inventory'
      : 'real trace replay profiles containing full single-turn, short/medium/long multi-turn, and stress workloads';
  const coverageLabel = `${DATA_SCOPE_META[dataScope].shortLabel} coverage`;
  // Provenance (RFC §4.7): surface the artifact's generated_at + a staleness
  // warning so a stale coverage blob can never be silently rendered as truth.
  const coverageGeneratedAt = dataScope === 'synthetic_distributional'
    ? blockersState?.generated_at ?? null
    : null;
  const coverageGenMs = coverageGeneratedAt ? Date.parse(coverageGeneratedAt) : NaN;
  const coverageAgeMin = Number.isFinite(coverageGenMs)
    ? Math.max(0, Math.floor((Date.now() - coverageGenMs) / 60000))
    : null;
  const coverageStale = coverageAgeMin !== null && coverageAgeMin >= COVERAGE_STALE_MINUTES;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-[#8b949e]">
              <span className="font-medium uppercase tracking-wide text-[#00bcd4]">
                {coverageLabel}
              </span>
              <span>{hardwareList.length} hardware targets</span>
              {coverageGeneratedAt && (
                <span
                  className={coverageStale
                    ? 'rounded border border-[#f0883e] bg-[#3a2a12] px-1.5 py-0.5 font-medium text-[#f0883e]'
                    : 'text-[#6e7681]'}
                  title={`coverage artifact generated_at ${coverageGeneratedAt}`}
                >
                  {coverageStale
                    ? `⚠ coverage data ${coverageAgeMin}m stale — orchestrator may be stopped`
                    : `coverage fresh · ${coverageAgeMin}m old`}
                </span>
              )}
            </div>
            <div className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="font-mono text-3xl font-semibold text-[#e6edf3]">{primarySummary}</span>
              {canonicalCoverage && (
                <span className="font-mono text-sm text-[#8b949e]">{cellSummary}</span>
              )}
              <span className="text-xs text-[#8b949e]">{scopeSummary}</span>
            </div>
            <CoverageProgress value={Number(pct)} />
          </div>

          <div className="flex items-center justify-end gap-2">
            <button onClick={expandAll} className="rounded-md border border-[#30363d] bg-[#21262d] px-3 py-1.5 text-[11px] font-medium text-[#c9d1d9] transition-colors hover:border-[#58a6ff] hover:text-[#58a6ff]">Expand</button>
            <button onClick={collapseAll} className="rounded-md border border-[#30363d] bg-[#21262d] px-3 py-1.5 text-[11px] font-medium text-[#c9d1d9] transition-colors hover:border-[#f97583] hover:text-[#f97583]">Collapse</button>
          </div>
        </div>
      </div>

      <CoverageLegend dataScope={dataScope} />

      <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
        <table className="min-w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-3 py-1.5 text-left font-medium" colSpan={3}></th>
              <th className="px-1.5 py-1.5 text-center text-[10px] font-semibold uppercase tracking-wide text-[#8b949e]" colSpan={allConcs.length}>
                Concurrency
              </th>
              <th className="px-3 py-1.5 text-right font-medium"></th>
            </tr>
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="w-[116px] px-3 py-2 text-left font-medium">Family</th>
              <th className="w-[220px] px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-left font-medium">Profile / status</th>
              {allConcs.map((c) => (
                <th key={c} className="px-1.5 py-2 text-center font-mono font-normal">{c}</th>
              ))}
              <th className="px-3 py-2 text-right font-medium">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => {
              const hwOpen = expandedHw.has(g.hardware);
              return (
                <GroupRows
                  key={g.hardware}
                  group={g}
                  hwOpen={hwOpen}
                  expandedModel={expandedModel}
                  onToggleHw={() => toggleHw(g.hardware)}
                  onToggleModel={toggleModel}
                  allConcs={allConcs}
                  expectedCellsPerModel={coveragePlan.expectedCellsPerModel}
                />
              );
            })}
          </tbody>
        </table>
      </div>

    </div>
  );
}

// --- Row renderers ---

interface GroupRowsProps {
  group: HwGroup;
  hwOpen: boolean;
  expandedModel: Set<string>;
  onToggleHw: () => void;
  onToggleModel: (key: string) => void;
  allConcs: number[];
  expectedCellsPerModel: number;
}

function GroupRows({ group, hwOpen, expandedModel, onToggleHw, onToggleModel, allConcs, expectedCellsPerModel }: GroupRowsProps) {
  const g = group;
  const pct = g.summary.totalNeed > 0
    ? Math.round((g.summary.totalHave / g.summary.totalNeed) * 100)
    : 0;
  const blocked = g.summary.skipped + g.summary.oom + g.summary.infeasible;
  const chips = ([
    { count: g.summary.complete, label: 'complete', tone: 'good' },
    { count: g.summary.failed, label: 'failed', tone: 'danger', title: `${g.summary.failedCells} failed expected cells` },
    { count: blocked, label: 'N/A', tone: 'na', title: `${g.summary.skipped} skipped, ${g.summary.oom} OOM, ${g.summary.infeasible} infeasible` },
    { count: g.summary.untested, label: 'TODO', tone: 'todo' },
  ] satisfies Array<{ count: number; label: string; tone: StatusTone; title?: string }>).filter(({ count }) => count > 0);

  return (
    <>
      <tr
        className="cursor-pointer border-b-2 border-t-2 border-[#30363d] bg-[#0d1117] hover:bg-[#161b22]"
        onClick={onToggleHw}
      >
        <td colSpan={3} className="px-3 py-2">
          <span className="mr-2 inline-block w-4 text-[#8b949e]">{hwOpen ? '▼' : '▶'}</span>
          <span className="font-mono text-sm font-semibold text-[#c9d1d9]">{g.hardware}</span>
          <span className="ml-3 inline-flex flex-wrap items-center gap-1.5 text-[#8b949e]">
            {chips.map(({ count, label, tone, title }) => (
              <GroupChip key={label} count={count} label={label} tone={tone} title={title} />
            ))}
          </span>
        </td>
        <td colSpan={allConcs.length} className="px-3 py-2 text-right text-[#8b949e]">
          {g.summary.totalNeed > 0 && (
            <span className={pct === 100 ? 'text-[#3fb950]' : pct === 0 ? 'text-[#8b949e]' : 'text-[#ff9800]'}>
              {g.summary.totalHave}/{g.summary.totalNeed} filled
            </span>
          )}
          {g.summary.failedCells > 0 && (
            <span className="ml-2 text-[#f85149]">
              {g.summary.failedCells} failed
            </span>
          )}
        </td>
        <td className="px-3 py-2 text-right font-mono">
          <span className={pct === 100 ? 'text-[#3fb950]' : pct === 0 ? 'text-[#8b949e]' : 'text-[#ff9800]'}>
            {pct}%
          </span>
        </td>
      </tr>
      {hwOpen && g.models.map((m, index) => {
        const mKey = `${g.hardware}|${m.model}|${m.backend}`;
        const showFamily = index === 0 || modelFamily(g.models[index - 1].model) !== modelFamily(m.model);
        return (
          <ModelRows
            key={mKey}
            hwName={g.hardware}
            model={m}
            showFamily={showFamily}
            open={expandedModel.has(mKey)}
            onToggle={() => onToggleModel(mKey)}
            allConcs={allConcs}
            expectedCellsPerModel={expectedCellsPerModel}
          />
        );
      })}
    </>
  );
}

interface ModelRowsProps {
  hwName: string;
  model: ModelEntry;
  showFamily: boolean;
  open: boolean;
  onToggle: () => void;
  allConcs: number[];
  expectedCellsPerModel: number;
}

function ModelRows({ hwName, model, showFamily, open, onToggle, allConcs, expectedCellsPerModel }: ModelRowsProps) {
  const family = modelFamily(model.model);

  if (model.kind === 'status') {
    const bg = bgForStatus(model.status);
    const txt = colorForStatus(model.status);
    const label = labelForStatus(model.status);
    const totalNeed = model.totalNeed ?? expectedCellsPerModel;
    const profiles = model.profiles ?? [];
    const canExpand = profiles.length > 0;
    const requeueMatch = model.reason?.match(/coverage requeue limit reached ([^;]+)/);
    const requeueLabel = model.status === 'failed' && requeueMatch ? `coverage requeue ${requeueMatch[1]}` : null;
    const attemptLabel = model.attempt !== undefined && model.attempt > 0
      ? model.maxAttempts != null
        ? `${model.status === 'failed' ? 'run attempt ' : ''}${model.attempt}/${model.maxAttempts}`
        : `${model.status === 'failed' ? 'run attempt ' : ''}${model.attempt}`
      : null;
    const concStats = new Map<number, { present: number; expected: number; failed: number; na: number; reason?: string; failedReason?: string }>();
    for (const p of profiles) {
      for (const c of p.expected) {
        const s = concStats.get(c) ?? { present: 0, expected: 0, failed: 0, na: 0 };
        const blockedReason = p.blocked?.get(c);
        const naReason = p.infeasibleReason ?? blockedReason;
        const failedReason = p.failed?.get(c);
        if (naReason) {
          s.na += 1;
          s.reason = s.reason ?? naReason;
        } else {
          s.expected += 1;
          if (p.present.has(c)) s.present += 1;
          if (failedReason && !p.present.has(c)) {
            s.failed += 1;
            s.failedReason = s.failedReason ?? failedReason;
          }
        }
        concStats.set(c, s);
      }
    }
    return (
      <>
        <tr
          className={`border-b border-[#21262d]/50 ${bg} ${canExpand ? 'cursor-pointer hover:bg-[#1b222a]' : ''}`}
          onClick={canExpand ? onToggle : undefined}
        >
          <td className="whitespace-nowrap px-3 py-1.5">
            <FamilyGroupCell family={family} showLabel={showFamily} />
          </td>
          <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">
            <span className={`mr-2 inline-block w-3 ${canExpand ? 'text-[#8b949e]' : 'text-[#30363d]'}`}>
              {canExpand ? (open ? '▼' : '▶') : '·'}
            </span>
            {model.model}
            <BackendBadge backend={model.backend} />
          </td>
          <td className="px-3 py-1.5">
            <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
              <StatusBadge kind={model.status} />
              <span className={`font-medium ${txt}`}>{label}</span>
              {profiles.length > 0 && <span className="text-[10px] uppercase tracking-wide text-[#8b949e]">· {profiles.length} profiles</span>}
              {requeueLabel && <span className="text-[#f85149]">· {requeueLabel}</span>}
              {attemptLabel && <span className="text-[#8b949e]">· {attemptLabel}</span>}
              {model.reason && <span className="min-w-[220px] flex-1 whitespace-normal break-words text-[#8b949e]" title={model.reason}>— {model.reason}</span>}
            </div>
          </td>
          {allConcs.map((c) => {
            const s = concStats.get(c);
            if (!s) return <td key={c} className="px-1 py-1.5 text-center"><Cell state="na" title={model.reason} /></td>;
            if (s.expected === 0) return <td key={c} className="px-1 py-1.5 text-center"><Cell state="excluded" title={s.reason ?? model.reason} /></td>;
            return <td key={c} className="px-1 py-1.5 text-center"><PartialCell present={s.present} expected={s.expected} failed={s.failed} title={s.failedReason} /></td>;
          })}
          <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
            {model.status === 'skipped' || model.status === 'oom' || model.status === 'infeasible' ? (
              <span className="text-[#64b5f6]" title={model.reason}>N/A</span>
            ) : model.status === 'untested' || model.status === 'pending' ||
             model.status === 'running' || model.status === 'failed' ? (
              <span className={model.status === 'failed' ? 'text-[#f85149]' : 'text-[#8b949e]'}>
                0/{totalNeed}{model.status === 'failed' ? ' failed' : ''}
              </span>
            ) : (
              <span className="text-[#8b949e]">—</span>
            )}
          </td>
        </tr>
        {canExpand && open && (
          <ProfileDetailRows
            hwName={hwName}
            modelName={model.model}
            backend={model.backend}
            profiles={profiles}
            allConcs={allConcs}
          />
        )}
      </>
    );
  }

  // model.kind === 'data'
  const rowPct = model.totalNeed > 0 ? Math.round((model.totalHave / model.totalNeed) * 100) : 0;
  // Per-concurrency fill fraction across all profiles. A conc is "full" only
  // when every profile that expects it actually has a run at that conc.
  const concStats = new Map<number, { present: number; expected: number; failed: number; na: number; reason?: string; failedReason?: string }>();
  for (const p of model.profiles) {
    if (p.infeasibleReason) continue;
    for (const c of p.expected) {
      const s = concStats.get(c) ?? { present: 0, expected: 0, failed: 0, na: 0 };
      const blockedReason = p.blocked?.get(c);
      if (blockedReason) {
        s.na += 1;
        s.reason = s.reason ?? blockedReason;
      } else {
        s.expected += 1;
        if (p.present.has(c)) s.present += 1;
        const failedReason = p.failed?.get(c);
        if (failedReason && !p.present.has(c)) {
          s.failed += 1;
          s.failedReason = s.failedReason ?? failedReason;
        }
      }
      concStats.set(c, s);
    }
  }

  return (
    <>
      <tr
        className="cursor-pointer border-b border-[#21262d]/50 hover:bg-[#1b222a]"
        onClick={onToggle}
      >
        <td className="whitespace-nowrap px-3 py-1.5">
          <FamilyGroupCell family={family} showLabel={showFamily} />
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">
          <span className="mr-2 inline-block w-3 text-[#8b949e]">{open ? '▼' : '▶'}</span>
          {model.model}
          <BackendBadge backend={model.backend} version={model.engineVersion} />
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-[#8b949e]">
          <span className="text-[10px] uppercase tracking-wide">{model.profiles.length} profiles</span>
        </td>
        {allConcs.map((c) => {
          const s = concStats.get(c);
          if (!s) return <td key={c} className="px-1 py-1.5 text-center"><Cell state="na" /></td>;
          if (s.expected === 0 && s.na > 0) return <td key={c} className="px-1 py-1.5 text-center"><Cell state="excluded" title={s.reason} /></td>;
          return <td key={c} className="px-1 py-1.5 text-center"><PartialCell present={s.present} expected={s.expected} failed={s.failed} title={s.failedReason} /></td>;
        })}
        <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
          {model.totalNeed === 0 ? (
            <span className="text-[#64b5f6]">N/A</span>
          ) : (
            <span
              className={
                rowPct === 100 ? 'text-[#3fb950]' :
                rowPct === 0 ? 'text-[#8b949e]' :
                'text-[#ff9800]'
              }
            >
              {model.totalHave}/{model.totalNeed}
            </span>
          )}
        </td>
      </tr>
      {open && (
        <ProfileDetailRows
          hwName={hwName}
          modelName={model.model}
          backend={model.backend}
          profiles={model.profiles}
          allConcs={allConcs}
        />
      )}
      </>
    );
  }

function ProfileDetailRows({
  hwName,
  modelName,
  backend,
  profiles,
  allConcs,
}: {
  hwName: string;
  modelName: string;
  backend: string;
  profiles: ProfileRow[];
  allConcs: number[];
}) {
  return (
    <>
      {profiles.map((p) => {
        const blockedCount = p.blocked?.size ?? 0;
        const failedCount = p.failed?.size ?? 0;
        const firstBlockedReason = p.blocked ? Array.from(p.blocked.values())[0] : undefined;
        const firstFailedReason = p.failed ? Array.from(p.failed.values())[0] : undefined;
        const have = [...p.present].filter((c) => p.expected.includes(c) && !p.blocked?.has(c)).length;
        const need = p.infeasibleReason ? 0 : p.expected.filter((c) => !p.blocked?.has(c)).length;
        const profPct = need > 0 ? Math.round((have / need) * 100) : 0;
        const profUntested = !p.infeasibleReason && blockedCount === 0 && failedCount === 0 && have === 0;
        const todoReason = p.todoReason ?? 'expected by sweep grid; no completed run exists yet';
        const displayName = profileDisplayName(p.profile);
        return (
          <tr key={`${hwName}|${modelName}|${backend}|${p.profile}`} className="border-b border-[#21262d]/50 bg-[#0d1117]/50">
            <td className="px-3 py-1.5">
              <span className="inline-block min-w-[82px]" aria-hidden="true" />
            </td>
            <td className="whitespace-nowrap px-3 py-1.5 pl-8 text-[#8b949e]">
              {/* empty — profile rows sit under the model row, matching the predictor table grouping */}
            </td>
            <td className="px-3 py-1.5 text-[#8b949e]">
              <div className="flex min-w-0 flex-wrap items-center gap-x-1 gap-y-1">
                <span className="text-[#c9d1d9]" title={p.profile}>{displayName}</span>
                {displayName !== p.profile && <span className="text-[10px] text-[#6e7681]">{p.profile}</span>}
                {p.isMultiTurn && <span className="rounded bg-[#8b5cf6]/20 px-1 text-[10px] text-[#8b5cf6]">mt</span>}
                {p.infeasibleReason && <span className="rounded border border-[#64b5f6]/40 bg-[#64b5f6]/10 px-1 text-[10px] text-[#64b5f6] uppercase" title={p.infeasibleReason}>N/A</span>}
                {blockedCount > 0 && <span className="rounded border border-[#64b5f6]/40 bg-[#64b5f6]/10 px-1 text-[10px] text-[#64b5f6] uppercase" title={firstBlockedReason}>{blockedCount} N/A</span>}
                {failedCount > 0 && <span className="rounded border border-[#f85149]/40 bg-[#f85149]/10 px-1 text-[10px] text-[#f85149] uppercase" title={firstFailedReason}>{failedCount} failed</span>}
                {profUntested && <span className="rounded border border-[#ff9800]/40 bg-[#ff9800]/10 px-1 text-[10px] text-[#ff9800] uppercase" title={todoReason}>TODO</span>}
                {p.infeasibleReason && <span className="min-w-[220px] flex-1 whitespace-normal break-words text-[10px] text-[#8b949e]" title={p.infeasibleReason}>— {p.infeasibleReason}</span>}
                {!p.infeasibleReason && firstFailedReason && <span className="min-w-[220px] flex-1 whitespace-normal break-words text-[10px] text-[#8b949e]" title={firstFailedReason}>— {firstFailedReason}</span>}
                {!p.infeasibleReason && firstBlockedReason && <span className="min-w-[220px] flex-1 whitespace-normal break-words text-[10px] text-[#8b949e]" title={firstBlockedReason}>— {firstBlockedReason}</span>}
                {profUntested && !p.infeasibleReason && !firstBlockedReason && <span className="min-w-[220px] flex-1 whitespace-normal break-words text-[10px] text-[#8b949e]" title={todoReason}>— {todoReason}</span>}
              </div>
            </td>
            {allConcs.map((c) => {
              const expected = p.expected.includes(c);
              const present = p.present.has(c);
              const blockedReason = p.blocked?.get(c);
              const failedReason = p.failed?.get(c);
              const state: 'present' | 'missing' | 'na' | 'excluded' =
                p.infeasibleReason || blockedReason ? 'excluded' : !expected ? 'na' : present ? 'present' : 'missing';
              const finalState = failedReason && !present && expected && !p.infeasibleReason && !blockedReason ? 'fail' : state;
              return <td key={c} className="px-1 py-1.5 text-center"><Cell state={finalState} title={failedReason ?? blockedReason ?? p.infeasibleReason} /></td>;
            })}
            <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
              {p.infeasibleReason || (need === 0 && blockedCount > 0) ? (
                <span className="text-[#64b5f6]" title={p.infeasibleReason ?? firstBlockedReason}>N/A</span>
              ) : (
                <span
                  className={
                    profPct === 100 ? 'text-[#3fb950]' :
                    failedCount > 0 ? 'text-[#f85149]' :
                    profPct === 0 ? 'text-[#8b949e]' :
                    'text-[#ff9800]'
                  }
                >
                  {have}/{need}{failedCount > 0 ? ' failed' : ''}
                </span>
              )}
            </td>
          </tr>
        );
      })}
    </>
  );
}

// --- UI helpers ---

type StatusTone = 'good' | 'warn' | 'danger' | 'active' | 'na' | 'todo' | 'muted';

const TONE_CLASS: Record<StatusTone, string> = {
  good: 'border-[#3fb950]/35 bg-[#3fb950]/10 text-[#3fb950]',
  warn: 'border-[#ff9800]/35 bg-[#ff9800]/10 text-[#ffb74d]',
  danger: 'border-[#f85149]/35 bg-[#f85149]/10 text-[#f85149]',
  active: 'border-[#58a6ff]/35 bg-[#58a6ff]/10 text-[#58a6ff]',
  na: 'border-[#64b5f6]/35 bg-[#64b5f6]/10 text-[#64b5f6]',
  todo: 'border-[#ff9800]/35 bg-[#ff9800]/10 text-[#ff9800]',
  muted: 'border-[#30363d] bg-[#21262d]/60 text-[#8b949e]',
};

function CoverageProgress({ value }: { value: number }) {
  return (
    <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#0d1117]">
      <div
        className="h-full rounded-full bg-[#00bcd4]"
        style={{ width: `${Math.max(0, Math.min(100, value))}%` }}
      />
    </div>
  );
}

function GroupChip({
  count,
  label,
  tone,
  title,
}: {
  count: number;
  label: string;
  tone: StatusTone;
  title?: string;
}) {
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${TONE_CLASS[tone]}`}
      title={title}
    >
      {count} {label}
    </span>
  );
}

function CoverageLegend({ dataScope }: { dataScope: DataScope }) {
  const canonicalCoverage = usesCanonicalCoverage(dataScope);
  const scopeNote = dataScope === 'synthetic_distributional'
    ? 'Synthetic coverage tracks APC-aware synthetic-suffixed profiles on the active sweep-state grid. coding-singleturn is intentionally excluded.'
    : dataScope === 'archived'
      ? 'Archived coverage is inventory-style: it shows retired canonical, fixed-grid, and MSE runs that exist and does not count missing cells.'
      : 'Trace replay coverage is inventory-style: it shows real replay runs that exist and does not count missing legacy cells.';

  return (
    <div className="rounded-md border border-[#21262d] bg-[#161b22] px-4 py-3 text-xs text-[#8b949e]">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-medium text-[#c9d1d9]">Coverage legend</span>
        <span>cell states and scope</span>
      </div>
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(280px,1fr)]">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-1.5"><Cell state="present" />present</span>
          {canonicalCoverage && (
            <span className="flex items-center gap-1.5"><Cell state="missing" />expected and missing</span>
          )}
          {canonicalCoverage && (
            <span className="flex items-center gap-1.5"><Cell state="fail" />failed after retry</span>
          )}
          {canonicalCoverage && (
            <span className="flex items-center gap-1.5"><Cell state="excluded" />N/A after retry / blocked</span>
          )}
            <span className="flex items-center gap-1.5">
              <Cell state="na" />
              {canonicalCoverage ? 'not expected in this row' : 'not observed'}
            </span>
          </div>
        <div className="space-y-1 leading-relaxed">
          <p>{scopeNote}{canonicalCoverage ? ' Exhausted OOM, success-rate, zero-output, and incomplete-output limits are excluded as N/A; benchmark, driver, and unknown failures stay red.' : ''}</p>
        </div>
      </div>
    </div>
  );
}

function Cell({ state, title }: { state: 'present' | 'missing' | 'fail' | 'na' | 'excluded'; title?: string }) {
  const cls =
    state === 'present' ? 'bg-[#3fb950] border-[#3fb950]' :
    state === 'fail' ? 'bg-[#f85149]/30 border-[#f85149]' :
    state === 'excluded' ? 'bg-[#64b5f6]/35 border-[#64b5f6]/70' :
    state === 'missing' ? 'bg-transparent border-[#30363d]' :
    'bg-[#21262d]/50 border-transparent';
  return <span className={`inline-block h-3 w-3 rounded-sm border ${cls}`} title={title} />;
}

function BackendBadge({ backend, version }: { backend: string; version?: string }) {
  const cls =
    backend === 'vllm'   ? 'bg-[#3fb950]/15 text-[#3fb950] border-[#3fb950]/40' :
    backend === 'sglang' ? 'bg-[#ffb74d]/15 text-[#ffb74d] border-[#ffb74d]/40' :
                           'bg-[#21262d] text-[#8b949e] border-[#30363d]';
  return (
    <span className={`ml-2 rounded border px-1.5 py-0.5 text-[10px] font-medium lowercase tracking-wide ${cls}`}>
      {backend}{version ? ` ${version}` : ''}
    </span>
  );
}

function familyStyle(family: ModelFamily): { chip: string; mark: string } {
  const styles: Record<ModelFamily, { chip: string; mark: string }> = {
    Llama: {
      chip: 'text-[#9ecbff]',
      mark: 'bg-[#58a6ff]',
    },
    Qwen: {
      chip: 'text-[#7ee787]',
      mark: 'bg-[#3fb950]',
    },
    'GPT-OSS': {
      chip: 'text-[#d2a8ff]',
      mark: 'bg-[#d2a8ff]',
    },
    Mixtral: {
      chip: 'text-[#ffb74d]',
      mark: 'bg-[#ffb74d]',
    },
    Gemma: {
      chip: 'text-[#00bcd4]',
      mark: 'bg-[#00bcd4]',
    },
    Granite: {
      chip: 'text-[#f97583]',
      mark: 'bg-[#f97583]',
    },
    Other: {
      chip: 'text-[#8b949e]',
      mark: 'bg-[#8b949e]',
    },
  };
  return styles[family];
}

function FamilyGroupCell({ family, showLabel }: { family: ModelFamily; showLabel: boolean }) {
  const style = familyStyle(family);
  if (!showLabel) {
    return <span className="inline-block min-w-[82px]" aria-hidden="true" />;
  }
  return (
    <span className={`inline-flex min-w-[82px] items-center gap-1.5 px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${style.chip}`}>
      <span className={`h-2 w-2 rounded-sm ${style.mark}`} />
      {family}
    </span>
  );
}

// Aggregate cell for model-row summaries. Solid green only when every
// profile that expects this concurrency has a run at it; partial fill from
// bottom proportional to fraction otherwise. Empty outline = 0 / N.
function PartialCell({
  present,
  expected,
  failed = 0,
  title,
}: {
  present: number;
  expected: number;
  failed?: number;
  title?: string;
}) {
  if (expected === 0) return <span className="inline-block h-3 w-3 rounded-sm border border-transparent bg-[#21262d]/50" />;
  if (present === 0 && failed > 0) return <span className="inline-block h-3 w-3 rounded-sm border border-[#f85149] bg-[#f85149]/30" title={title ?? `${failed}/${expected} failed`} />;
  if (present === 0) return <span className="inline-block h-3 w-3 rounded-sm border border-[#30363d] bg-transparent" />;
  if (present >= expected) return <span className="inline-block h-3 w-3 rounded-sm border border-[#3fb950] bg-[#3fb950]" title={`${present}/${expected}`} />;
  const fillPct = Math.round((present / expected) * 100);
  return (
    <span
      className="relative inline-block h-3 w-3 overflow-hidden rounded-sm border border-[#3fb950]/60 bg-transparent"
      title={`${present}/${expected}`}
    >
      <span
        className="absolute inset-x-0 bottom-0 bg-[#3fb950]"
        style={{ height: `${fillPct}%` }}
      />
    </span>
  );
}

type BadgeKind = StatusModel['status'];

function StatusBadge({ kind }: { kind: BadgeKind }) {
    const map: Record<BadgeKind, [string, string]> = {
      oom:        ['bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40', 'N/A'],
      infeasible: ['bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40', 'N/A'],
      running:    ['bg-[#58a6ff]/15 text-[#58a6ff] border-[#58a6ff]/40', 'RUN'],
      failed:     ['bg-[#f85149]/15 text-[#f85149] border-[#f85149]/40', 'FAIL'],
      pending:    ['bg-[#ff9800]/10 text-[#ff9800] border-[#ff9800]/40', 'TODO'],
      skipped:  ['bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40', 'N/A'],
      untested:   ['bg-[#ff9800]/10 text-[#ff9800] border-[#ff9800]/40', 'TODO'],
    };
  const [cls, label] = map[kind];
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}>
      {label}
    </span>
  );
}

function bgForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'bg-[#64b5f6]/5';
    case 'infeasible': return 'bg-[#64b5f6]/5';
    case 'running':    return 'bg-[#58a6ff]/5';
    case 'failed':     return 'bg-[#f85149]/5';
    case 'skipped':  return 'bg-[#64b5f6]/5';
    case 'pending':    return 'bg-[#ff9800]/5';
    default:           return '';
  }
}

function colorForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'text-[#64b5f6]';
    case 'infeasible': return 'text-[#64b5f6]';
    case 'running':    return 'text-[#58a6ff]';
    case 'failed':     return 'text-[#f85149]';
    case 'skipped':  return 'text-[#64b5f6]';
    case 'pending':    return 'text-[#ff9800]';
    default:           return 'text-[#8b949e]';
  }
}

function labelForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'blocked by OOM / capacity';
    case 'infeasible': return 'not runnable on this target';
    case 'running':    return 'running now';
    case 'failed':     return 'failed after retry';
    case 'skipped':  return 'not applicable / skipped';
    case 'pending':    return 'expected, not run yet';
    default:           return 'expected, not run yet';
  }
}
