import { useMemo, useState } from 'react';
import type { BenchmarkResult } from '../types';
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
const SYNTHETIC_SINGLE_CONCS = [200, 320];
const SYNTHETIC_MULTI_CONCS = [200, 320];
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
const SYNTHETIC_SINGLE_PROFILES = [
  'chat-singleturn-synth',
];
const SYNTHETIC_MULTI_PROFILES = [
  'chat-multiturn-synth',
  'swebench-multiturn-synth',
  'terminalbench-multiturn-synth',
  'osworld-multiturn-synth',
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
  infeasibleReason?: string;
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
  status: 'oom' | 'untested' | 'infeasible' | 'running' | 'skipped' | 'pending';
  reason?: string;
  attempt?: number;
  updatedAt?: string | null;
  totalNeed: number;
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
    skipped: number;
    oom: number;
    infeasible: number;
    untested: number;
    totalHave: number;
    totalNeed: number;
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
  return normalizeDataScope(cell.data_scope ?? null) ?? 'current';
}

function isMultiTurnProfile(profile: string): boolean {
  return profile.includes('multiturn') || profile.includes('multi-turn');
}

function usesCanonicalCoverage(scope: DataScope): boolean {
  return scope !== 'archive';
}

function coverageGridScope(scope: DataScope): DataScope {
  return scope;
}

export function CoveragePage({
  allData,
  sweepState,
  loading,
  dataScope,
}: CoveragePageProps) {
  const canonicalCoverage = usesCanonicalCoverage(dataScope);
  const gridScope = coverageGridScope(dataScope);

  const coveragePlan = useMemo(() => {
    const singleProfiles = gridScope === 'archive'
      ? ARCHIVE_SINGLE_PROFILES
      : gridScope === 'synthetic'
        ? SYNTHETIC_SINGLE_PROFILES
      : gridScope === 'fixed'
        ? FIXED_SINGLE_PROFILES
        : gridScope === 'mse'
          ? MSE_SINGLE_PROFILES
        : CURRENT_SINGLE_PROFILES;
    const multiProfiles = gridScope === 'archive'
      ? ARCHIVE_MULTI_PROFILES
      : gridScope === 'synthetic'
        ? SYNTHETIC_MULTI_PROFILES
      : gridScope === 'fixed'
        ? FIXED_MULTI_PROFILES
        : gridScope === 'mse'
          ? MSE_MULTI_PROFILES
        : CURRENT_MULTI_PROFILES;
    const singleConcs = gridScope === 'current'
      ? CURRENT_SINGLE_CONCS
      : gridScope === 'synthetic'
        ? SYNTHETIC_SINGLE_CONCS
        : gridScope === 'fixed'
          ? FIXED_SINGLE_CONCS
          : gridScope === 'mse'
            ? MSE_SINGLE_CONCS
            : ARCHIVE_SINGLE_CONCS;
    const multiConcs = gridScope === 'current'
      ? CURRENT_MULTI_CONCS
      : gridScope === 'synthetic'
        ? SYNTHETIC_MULTI_CONCS
        : gridScope === 'fixed'
          ? FIXED_MULTI_CONCS
          : gridScope === 'mse'
            ? MSE_MULTI_CONCS
            : ARCHIVE_MULTI_CONCS;
    return {
      singleProfiles,
      multiProfiles,
      singleConcs,
      multiConcs,
      expectedCellsPerModel: singleProfiles.length * singleConcs.length + multiProfiles.length * multiConcs.length,
    };
  }, [gridScope]);

  const { groups, hardwareList } = useMemo(() => {
    const { singleProfiles, multiProfiles, singleConcs, multiConcs, expectedCellsPerModel } = coveragePlan;
    const scopedSweepCells = sweepState?.cells.filter((cell) => stateCellScope(cell) === gridScope) ?? [];
    const baseHwLabels = sweepState
      ? Object.values(sweepState.hosts).map((h) => h.hardware_label)
      : ['A100-40GB', '3090', '2080Ti', 'H100'];
    const dataHw = new Set(allData.map((r) => r.hardware));
    const expectedHw: string[] = [];
    if (dataScope === 'archive') {
      expectedHw.push(...Array.from(dataHw).sort());
    } else if (dataScope === 'mse') {
      const mseHw = new Set<string>([...scopedSweepCells.map((cell) => cell.hw_label), ...dataHw]);
      expectedHw.push(...Array.from(mseHw).sort());
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
      if (dataScope === 'mse') {
        for (const cell of scopedSweepCells) expectedModels.add(cell.model);
      } else {
        for (const m of Object.keys(sweepState.models)) expectedModels.add(m);
      }
    }
    for (const r of allData) expectedModels.add(r.modelShort);
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
        if ((item.data_scope ?? 'current') !== gridScope) continue;
        profileInfeasible.set(
          `${item.hw_label}|${item.model}|${item.backend}|${item.profile}`,
          item.reason,
        );
      }
    }
    const profileInfeasibleReasonFor = (
      hw: string,
      model: string,
      backend: string,
      profile: string,
    ): string | undefined =>
      profileInfeasible.get(`${hw}|${model}|${backend}|${profile}`);
    const expectedCountFor = (hw: string, model: string, backend: string): number => {
      let total = 0;
      for (const profile of singleProfiles) {
        if (!profileInfeasibleReasonFor(hw, model, backend, profile)) total += singleConcs.length;
      }
      for (const profile of multiProfiles) {
        if (!profileInfeasibleReasonFor(hw, model, backend, profile)) total += multiConcs.length;
      }
      return total;
    };

    const bucket = new Map<string, Set<number>>();
    const mbHasData = new Map<string, Set<string>>();  // hw -> Set<"model|backend">
    const engineVersionByMb = new Map<string, string>();  // "hw|model|backend" -> version
    const profilesByMb = new Map<string, Set<string>>();  // "hw|model|backend" -> profiles with data
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

    const aggStatus = sweepState
      ? aggregateCells(scopedSweepCells)
      : new Map<string, SweepCell>();

    const hwGroups: HwGroup[] = [];
    for (const hw of expectedHw) {
      const models: ModelEntry[] = [];
      const summary = {
        complete: 0, partial: 0,
        running: 0, pending: 0, skipped: 0,
        oom: 0, infeasible: 0, untested: 0,
        totalHave: 0, totalNeed: 0,
      };
      for (const model of modelList) {
        // Always include ACTIVE_BACKENDS (current sweep target) plus any
        // other known backend that actually has data for this (hw, model).
        const backendSet = new Set<string>();
        if (canonicalCoverage) {
          if (dataScope === 'mse') {
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
          const hasData = mbHasData.get(hw)?.has(`${model}|${backend}`) ?? false;
          // sweep-state status only applies to the vllm backend.
          const cell = aggStatus.get(`${hw}|${model}|${backend}`);
          const expectedForModel = canonicalCoverage
            ? expectedCountFor(hw, model, backend)
            : expectedCellsPerModel;

          if (hasData) {
            const profiles: ProfileRow[] = [];
            let totalHave = 0;
            let totalNeed = 0;
            if (dataScope === 'archive') {
              const mbKey = `${hw}|${model}|${backend}`;
              const observedProfiles = Array.from(profilesByMb.get(mbKey) ?? []).sort();
              for (const profile of observedProfiles) {
                const present = bucket.get(`${hw}|${model}|${backend}|${profile}`) ?? new Set<number>();
                const observedConcs = Array.from(present).sort((a, b) => a - b);
                totalHave += observedConcs.length;
                totalNeed += observedConcs.length;
                profiles.push({ profile, isMultiTurn: isMultiTurnProfile(profile), expected: observedConcs, present });
              }
            } else {
              for (const profile of singleProfiles) {
                const present = bucket.get(`${hw}|${model}|${backend}|${profile}`) ?? new Set<number>();
                const infeasibleReason = profileInfeasibleReasonFor(hw, model, backend, profile);
                if (!infeasibleReason) {
                  const have = [...present].filter((c) => singleConcs.includes(c)).length;
                  totalHave += have;
                  totalNeed += singleConcs.length;
                }
                profiles.push({ profile, isMultiTurn: false, expected: singleConcs, present, infeasibleReason });
              }
              for (const profile of multiProfiles) {
                const present = bucket.get(`${hw}|${model}|${backend}|${profile}`) ?? new Set<number>();
                const infeasibleReason = profileInfeasibleReasonFor(hw, model, backend, profile);
                if (!infeasibleReason) {
                  const have = [...present].filter((c) => multiConcs.includes(c)).length;
                  totalHave += have;
                  totalNeed += multiConcs.length;
                }
                profiles.push({ profile, isMultiTurn: true, expected: multiConcs, present, infeasibleReason });
              }
            }
            const engineVersion = engineVersionByMb.get(`${hw}|${model}|${backend}`);
            models.push({ kind: 'data', hardware: hw, model, backend, engineVersion, profiles, totalHave, totalNeed });
            summary.totalHave += totalHave;
            summary.totalNeed += totalNeed;
            if (totalHave === totalNeed) summary.complete += 1;
            else summary.partial += 1;
            continue;
          }

          if (cell) {
            if (cell.status === 'known_oom') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'oom', reason: cell.reason ?? undefined, totalNeed: 0 });
              summary.oom += 1;
              continue;
            }
            if (cell.status === 'running') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'running', attempt: cell.attempt, updatedAt: cell.updated_at, totalNeed: expectedForModel });
              summary.running += 1;
              summary.totalNeed += expectedForModel;
              continue;
            }
            if (cell.status === 'skipped') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'skipped', reason: cell.reason ?? undefined, attempt: cell.attempt, totalNeed: 0 });
              summary.skipped += 1;
              continue;
            }
            if (cell.status === 'pending' || cell.status === 'done') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested', totalNeed: expectedForModel });
              summary.untested += 1;
              summary.totalNeed += expectedForModel;
              continue;
            }
          }

          const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
          if (infReason) {
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'infeasible', reason: infReason, totalNeed: 0 });
            summary.infeasible += 1;
          } else {
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested', totalNeed: expectedForModel });
            summary.untested += 1;
            summary.totalNeed += expectedForModel;
          }
        }
      }
      hwGroups.push({ hardware: hw, models, summary });
    }

    return { groups: hwGroups, hardwareList: expectedHw };
  }, [allData, canonicalCoverage, coveragePlan, dataScope, gridScope, sweepState]);

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
    for (const g of groups) for (const m of g.models) if (m.kind === 'data') keys.add(`${g.hardware}|${m.model}|${m.backend}`);
    setExpandedModel(keys);
  };
  const collapseAll = () => {
    setExpandedHw(new Set());
    setExpandedModel(new Set());
  };

  const allConcs = useMemo(
    () => {
      if (dataScope === 'archive') {
        const observed = new Set<number>();
        for (const r of allData) observed.add(r.config.concurrency);
        return Array.from(observed).sort((a, b) => a - b);
      }
      return Array.from(new Set([...coveragePlan.singleConcs, ...coveragePlan.multiConcs])).sort((a, b) => a - b);
    },
    [allData, coveragePlan, dataScope],
  );

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-[#8b949e]">Loading benchmark data...</div>
      </div>
    );
  }

  const grand = groups.reduce(
    (acc, g) => {
      acc.complete += g.summary.complete;
      acc.partial += g.summary.partial;
      acc.running += g.summary.running;
      acc.pending += g.summary.pending;
      acc.skipped += g.summary.skipped;
      acc.oom += g.summary.oom;
      acc.infeasible += g.summary.infeasible;
      acc.untested += g.summary.untested;
      acc.totalHave += g.summary.totalHave;
      acc.totalNeed += g.summary.totalNeed;
      return acc;
    },
    { complete: 0, partial: 0, running: 0, pending: 0, skipped: 0, oom: 0, infeasible: 0, untested: 0, totalHave: 0, totalNeed: 0 },
  );
  const pct = grand.totalNeed > 0
    ? ((grand.totalHave / grand.totalNeed) * 100).toFixed(1)
    : '0.0';
  const cellSummary = dataScope === 'archive'
    ? `${grand.totalHave} cells filled`
    : `${grand.totalHave}/${grand.totalNeed} expected cells`;
  const primarySummary = dataScope === 'archive' ? `${grand.totalHave} cells filled` : `${pct}%`;
  const scopeSummary = dataScope === 'synthetic'
    ? '5 APC-aware synthetic profiles on the C=200/320 grid'
    : dataScope === 'current'
      ? '6 paper profiles'
      : dataScope === 'fixed'
      ? '5 fixed-scope profiles on the fixed concurrency grid'
      : dataScope === 'mse'
        ? 'synthetic-vs-real validation pairs at C=40/80'
      : 'legacy profiles containing full runs of single-turn, short/medium/long multi-turn, and stress workloads';
  const coverageLabel = `${DATA_SCOPE_META[dataScope].shortLabel} coverage`;

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
              {g.summary.totalHave}/{g.summary.totalNeed} cells
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
    return (
      <tr className={`border-b border-[#21262d]/50 ${bg}`}>
        <td className="whitespace-nowrap px-3 py-1.5">
          <FamilyGroupCell family={family} showLabel={showFamily} />
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">
          <span className="mr-2 inline-block w-3 text-[#30363d]">·</span>
          {model.model}
          <BackendBadge backend={model.backend} />
        </td>
        <td colSpan={allConcs.length + 1} className="px-3 py-1.5">
          <div className="flex min-w-0 items-center gap-2">
            <StatusBadge kind={model.status} />
            <span className={`min-w-0 ${txt}`}>
              {label}
              {model.attempt !== undefined && model.attempt > 0 && <span className="ml-1 text-[#8b949e]">· attempt {model.attempt}</span>}
              {model.reason && <span className="ml-1 inline-block max-w-[720px] truncate align-bottom text-[#8b949e]" title={model.reason}>— {model.reason}</span>}
            </span>
          </div>
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
          {model.status === 'untested' || model.status === 'pending' ||
           model.status === 'running'  || model.status === 'skipped' ? (
            <span className="text-[#8b949e]">0/{totalNeed}</span>
          ) : (
            <span className="text-[#8b949e]">—</span>
          )}
        </td>
      </tr>
    );
  }

  // model.kind === 'data'
  const rowPct = model.totalNeed > 0 ? Math.round((model.totalHave / model.totalNeed) * 100) : 0;
  // Per-concurrency fill fraction across all profiles. A conc is "full" only
  // when every profile that expects it actually has a run at that conc.
  const concStats = new Map<number, { present: number; expected: number }>();
  for (const p of model.profiles) {
    if (p.infeasibleReason) continue;
    for (const c of p.expected) {
      const s = concStats.get(c) ?? { present: 0, expected: 0 };
      s.expected += 1;
      if (p.present.has(c)) s.present += 1;
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
          return <td key={c} className="px-1 py-1.5 text-center"><PartialCell present={s.present} expected={s.expected} /></td>;
        })}
        <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
          <span
            className={
              rowPct === 100 ? 'text-[#3fb950]' :
              rowPct === 0 ? 'text-[#8b949e]' :
              'text-[#ff9800]'
            }
          >
            {model.totalHave}/{model.totalNeed}
          </span>
        </td>
      </tr>
      {open && model.profiles.map((p) => {
        const have = [...p.present].filter((c) => p.expected.includes(c)).length;
        const need = p.infeasibleReason ? 0 : p.expected.length;
        const profPct = need > 0 ? Math.round((have / need) * 100) : 0;
        const profUntested = !p.infeasibleReason && have === 0;
        const displayName = profileDisplayName(p.profile);
        return (
          <tr key={`${hwName}|${model.model}|${p.profile}`} className="border-b border-[#21262d]/50 bg-[#0d1117]/50">
            <td className="px-3 py-1.5">
              <span className="inline-block min-w-[82px]" aria-hidden="true" />
            </td>
            <td className="whitespace-nowrap px-3 py-1.5 pl-8 text-[#8b949e]">
              {/* empty — profile rows sit under the model row, matching the predictor table grouping */}
            </td>
            <td className="whitespace-nowrap px-3 py-1.5 text-[#8b949e]">
              <span className="text-[#c9d1d9]" title={p.profile}>{displayName}</span>
              {displayName !== p.profile && <span className="ml-1 text-[10px] text-[#6e7681]">{p.profile}</span>}
              {p.isMultiTurn && <span className="ml-1 rounded bg-[#8b5cf6]/20 px-1 text-[10px] text-[#8b5cf6]">mt</span>}
              {p.infeasibleReason && <span className="ml-1 rounded border border-[#64b5f6]/40 bg-[#64b5f6]/10 px-1 text-[10px] text-[#64b5f6] uppercase" title={p.infeasibleReason}>N/A</span>}
              {p.infeasibleReason && <span className="ml-1 inline-block max-w-[360px] truncate align-bottom text-[10px] text-[#8b949e]" title={p.infeasibleReason}>— {p.infeasibleReason}</span>}
              {profUntested && <span className="ml-1 rounded border border-[#ff9800]/40 bg-[#ff9800]/10 px-1 text-[10px] text-[#ff9800] uppercase">todo</span>}
            </td>
            {allConcs.map((c) => {
              const expected = p.expected.includes(c);
              const present = p.present.has(c);
              const state: 'present' | 'missing' | 'na' =
                p.infeasibleReason || !expected ? 'na' : present ? 'present' : 'missing';
              return <td key={c} className="px-1 py-1.5 text-center"><Cell state={state} title={p.infeasibleReason} /></td>;
            })}
            <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
              {p.infeasibleReason ? (
                <span className="text-[#64b5f6]" title={p.infeasibleReason}>N/A</span>
              ) : (
                <span
                  className={
                    profPct === 100 ? 'text-[#3fb950]' :
                    profPct === 0 ? 'text-[#8b949e]' :
                    'text-[#ff9800]'
                  }
                >
                  {have}/{need}
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

type StatusTone = 'good' | 'warn' | 'active' | 'na' | 'todo' | 'muted';

const TONE_CLASS: Record<StatusTone, string> = {
  good: 'border-[#3fb950]/35 bg-[#3fb950]/10 text-[#3fb950]',
  warn: 'border-[#ff9800]/35 bg-[#ff9800]/10 text-[#ffb74d]',
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
  const scopeNote = dataScope === 'synthetic'
    ? 'Synthetic coverage tracks APC-aware synthetic-suffixed profiles on the reduced C=200/320 grid. coding-singleturn is intentionally excluded.'
    : dataScope === 'current'
      ? 'Current coverage tracks the canonical paper profile surface and expected concurrency grid.'
      : dataScope === 'fixed'
      ? 'Fixed coverage tracks chat-singleturn plus all canonical multi-turn profiles on the corrected concurrency grid; context-overflow cells show as infeasible.'
      : dataScope === 'mse'
        ? 'MSE coverage tracks validation pairs only: one synthetic distributional run and one matched real short-trajectory run per dataset and concurrency.'
      : 'Archive coverage is inventory-style: it shows historical runs that exist and does not count missing legacy cells.';

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
          <span className="flex items-center gap-1.5">
            <Cell state="na" />
            {canonicalCoverage ? 'not expected / infeasible' : 'not observed'}
          </span>
        </div>
        <div className="space-y-1 leading-relaxed">
          <p>{scopeNote}</p>
        </div>
      </div>
    </div>
  );
}

function Cell({ state, title }: { state: 'present' | 'missing' | 'na'; title?: string }) {
  const cls =
    state === 'present' ? 'bg-[#3fb950] border-[#3fb950]' :
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
function PartialCell({ present, expected }: { present: number; expected: number }) {
  if (expected === 0) return <span className="inline-block h-3 w-3 rounded-sm border border-transparent bg-[#21262d]/50" />;
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
    case 'skipped':  return 'text-[#64b5f6]';
    case 'pending':    return 'text-[#ff9800]';
    default:           return 'text-[#8b949e]';
  }
}

function labelForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'not applicable';
    case 'infeasible': return 'not applicable';
    case 'running':    return 'being run';
    case 'skipped':  return 'not applicable';
    case 'pending':    return 'TODO';
    default:           return 'TODO';
  }
}
