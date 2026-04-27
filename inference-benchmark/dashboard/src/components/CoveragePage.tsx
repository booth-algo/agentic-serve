import { Fragment, useMemo, useState } from 'react';
import type { BenchmarkResult } from '../types';
import type { SweepCell, SweepState } from '../types-sweep';
import type {


interface CoveragePageProps {
  allData: BenchmarkResult[];
  sweepState: SweepState | null;
  loading: boolean;
}

const SINGLE_CONCS = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500];
const MULTI_CONCS = [1, 5, 10, 20, 40, 80, 120, 160, 200, 256, 320];

const ALL_SINGLE_PROFILES = [
  'chat-short', 'chat-medium', 'chat-long',
  'coding-agent', 'prefill-heavy', 'decode-heavy', 'random-1k',
];
const ALL_MULTI_PROFILES = [
  'chat-multiturn-short', 'chat-multiturn-medium', 'chat-multiturn-long',
  'terminalbench-multiturn-short', 'terminalbench-multiturn-medium',
  'swebench-multiturn-short', 'swebench-multiturn-medium',
  'osworld-multiturn-short', 'osworld-multiturn-medium', 'osworld-multiturn-long',
];

const TP_OPTIONS = [1, 2, 4, 8];

// Backends we always want a coverage row for. sglang is active now that
// the orchestrator routes by backend and all three hosts have sglang 0.5.9
// environments. Each (hw, model) gets a row for every active backend, plus
// any historical backend with data in data.json.
const ACTIVE_BACKENDS = ['vllm', 'sglang'];
const KNOWN_BACKENDS = ['vllm', 'sglang'];

// Max feasible cells per (hw, model, backend) — every single+multi profile
// at every expected concurrency. Used as the denominator for cells that
// haven't yet produced data (running/pending/skipped/untested) so the
// "N/M" coverage readout reflects the full target, not just attempted.
const EXPECTED_CELLS_PER_MODEL =
  ALL_SINGLE_PROFILES.length * SINGLE_CONCS.length +
  ALL_MULTI_PROFILES.length * MULTI_CONCS.length;

interface ProfileRow {
  profile: string;
  isMultiTurn: boolean;
  expected: number[];
  present: Set<number>;
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
  known_oom: 5, skipped: 4, running: 3, pending: 2, done: 1,
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

export function CoveragePage({
  allData,
  sweepState,
  loading,
}: CoveragePageProps) {
  const [view, setView] = useState<CoverageView>('bench');
  const { groups, hardwareList, sweepMtime } = useMemo(() => {
    const baseHwLabels = sweepState
      ? Object.values(sweepState.hosts).map((h) => h.hardware_label)
      : ['A100-40GB', '3090', '2080Ti', 'H100'];
    const expectedHw: string[] = [];
    for (const base of baseHwLabels) {
      for (const tp of TP_OPTIONS) expectedHw.push(hwLabel(base, tp));
    }
    const dataHw = new Set(allData.map((r) => r.hardware));
    for (const hw of dataHw) if (!expectedHw.includes(hw)) expectedHw.push(hw);

    const expectedModels = new Set<string>();
    if (sweepState) for (const m of Object.keys(sweepState.models)) expectedModels.add(m);
    for (const r of allData) expectedModels.add(r.modelShort);
    const modelList = Array.from(expectedModels).sort();

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

    const bucket = new Map<string, Set<number>>();
    const mbHasData = new Map<string, Set<string>>();  // hw -> Set<"model|backend">
    const engineVersionByMb = new Map<string, string>();  // "hw|model|backend" -> version
    for (const r of allData) {
      const backend = r.config.backend;
      const k = `${r.hardware}|${r.modelShort}|${backend}|${r.config.profile}`;
      if (!bucket.has(k)) bucket.set(k, new Set());
      bucket.get(k)!.add(r.config.concurrency);
      if (!mbHasData.has(r.hardware)) mbHasData.set(r.hardware, new Set());
      mbHasData.get(r.hardware)!.add(`${r.modelShort}|${backend}`);
      const mbKey = `${r.hardware}|${r.modelShort}|${backend}`;
      if (r.engineVersion && !engineVersionByMb.has(mbKey)) {
        engineVersionByMb.set(mbKey, r.engineVersion);
      }
    }

    const aggStatus = sweepState
      ? aggregateCells(sweepState.cells)
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
        const backendSet = new Set<string>(ACTIVE_BACKENDS);
        for (const b of KNOWN_BACKENDS) {
          if (mbHasData.get(hw)?.has(`${model}|${b}`)) backendSet.add(b);
        }
        const backendsForCell = Array.from(backendSet).sort();
        for (const backend of backendsForCell) {
          const hasData = mbHasData.get(hw)?.has(`${model}|${backend}`) ?? false;
          // sweep-state status only applies to the vllm backend.
          const cell = aggStatus.get(`${hw}|${model}|${backend}`);

          if (hasData) {
            const profiles: ProfileRow[] = [];
            let totalHave = 0;
            let totalNeed = 0;
            for (const profile of ALL_SINGLE_PROFILES) {
              const present = bucket.get(`${hw}|${model}|${backend}|${profile}`) ?? new Set<number>();
              const have = [...present].filter((c) => SINGLE_CONCS.includes(c)).length;
              totalHave += have;
              totalNeed += SINGLE_CONCS.length;
              profiles.push({ profile, isMultiTurn: false, expected: SINGLE_CONCS, present });
            }
            for (const profile of ALL_MULTI_PROFILES) {
              const present = bucket.get(`${hw}|${model}|${backend}|${profile}`) ?? new Set<number>();
              const have = [...present].filter((c) => MULTI_CONCS.includes(c)).length;
              totalHave += have;
              totalNeed += MULTI_CONCS.length;
              profiles.push({ profile, isMultiTurn: true, expected: MULTI_CONCS, present });
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
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'oom', reason: cell.reason ?? undefined });
              summary.oom += 1;
              continue;
            }
            if (cell.status === 'running') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'running', attempt: cell.attempt, updatedAt: cell.updated_at });
              summary.running += 1;
              summary.totalNeed += EXPECTED_CELLS_PER_MODEL;
              continue;
            }
            if (cell.status === 'skipped') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'skipped', reason: cell.reason ?? undefined, attempt: cell.attempt });
              summary.skipped += 1;
              summary.totalNeed += EXPECTED_CELLS_PER_MODEL;
              continue;
            }
            if (cell.status === 'pending' || cell.status === 'done') {
              models.push({ kind: 'status', hardware: hw, model, backend, status: 'pending' });
              summary.pending += 1;
              summary.totalNeed += EXPECTED_CELLS_PER_MODEL;
              continue;
            }
          }

          const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
          if (infReason) {
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'infeasible', reason: infReason });
            summary.infeasible += 1;
          } else {
            models.push({ kind: 'status', hardware: hw, model, backend, status: 'untested' });
            summary.untested += 1;
            summary.totalNeed += EXPECTED_CELLS_PER_MODEL;
          }
        }
      }
      hwGroups.push({ hardware: hw, models, summary });
    }

    return { groups: hwGroups, hardwareList: expectedHw, sweepMtime: sweepState?.generated_at ?? null };
  }, [allData, sweepState]);

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
    for (const g of groups) for (const m of g.models) if (m.kind === 'data') keys.add(`${g.hardware}|${m.model}`);
    setExpandedModel(keys);
  };
  const collapseAll = () => {
    setExpandedHw(new Set());
    setExpandedModel(new Set());
  };

  const allConcs = useMemo(
    () => Array.from(new Set([...SINGLE_CONCS, ...MULTI_CONCS])).sort((a, b) => a - b),
    [],
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

  return (
    <div className="space-y-4">

      {view === 'per-kernel' && (
        <PerKernelCoverage data={predictorCoverage ?? null} loading={predictorCoverageLoading ?? false} />
      )}
      {view === 'per-op' && (
        <PerOpCoverage data={predictorCoverage ?? null} loading={predictorCoverageLoading ?? false} />
      )}

      {view === 'bench' && (<>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${grand.totalHave}/${grand.totalNeed} cells`} color="#00bcd4" />
        <SummaryCell label="Complete" value={`${grand.complete}`} sub="all concs present" color="#3fb950" />
        <SummaryCell label="Partial" value={`${grand.partial}`} sub="some missing" color="#ff9800" />
        <SummaryCell label="Running" value={`${grand.running}`} sub="in progress" color="#58a6ff" />
        <SummaryCell label="Pending" value={`${grand.pending}`} sub="queued" color="#a5b4fc" />
        <SummaryCell label="Skipped" value={`${grand.skipped}`} sub="not attempted / blocked" color="#f97583" />
        <SummaryCell label="OOM" value={`${grand.oom}`} sub="structurally blocked" color="#e040fb" />
        <SummaryCell label="Infeasible" value={`${grand.infeasible}`} sub="VRAM too small" color="#64b5f6" />
      </div>

      <div className="flex flex-wrap items-center gap-4 rounded-md border border-[#21262d] bg-[#161b22] px-4 py-2 text-xs text-[#8b949e]">
        <span className="flex items-center gap-1.5"><Cell state="present" />present</span>
        <span className="flex items-center gap-1.5"><Cell state="missing" />expected &amp; missing</span>
        <span className="flex items-center gap-1.5"><Cell state="na" />not expected</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="running" />running</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="pending" />pending</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="skipped" />skipped</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="oom" />OOM</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="infeasible" />infeasible</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="untested" />untested</span>
        <span className="ml-auto flex items-center gap-3 font-mono">
          <button onClick={expandAll} className="rounded-md border border-[#30363d] bg-[#21262d] px-3 py-1 text-[11px] font-medium text-[#c9d1d9] shadow-sm transition-all hover:border-[#58a6ff] hover:bg-[#58a6ff]/10 hover:text-[#58a6ff] active:scale-95">▼ Expand All</button>
          <button onClick={collapseAll} className="rounded-md border border-[#30363d] bg-[#21262d] px-3 py-1 text-[11px] font-medium text-[#c9d1d9] shadow-sm transition-all hover:border-[#f97583] hover:bg-[#f97583]/10 hover:text-[#f97583] active:scale-95">▲ Collapse All</button>
          <span>hardware: {hardwareList.length}</span>
          {sweepMtime && <span>· sweep-state: {new Date(sweepMtime).toLocaleTimeString()}</span>}
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
        <table className="min-w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="w-[160px] px-3 py-2 text-left font-medium">Hardware / Model</th>
              <th className="px-3 py-2 text-left font-medium">Profile</th>
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
                />
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="space-y-1 text-xs text-[#8b949e]">
        <p>
          Sweep matrix sourced from <code className="rounded bg-[#21262d] px-1">scripts/sweep.yaml</code> via
          <code className="ml-1 rounded bg-[#21262d] px-1">sweep-state.json</code>; runtime status updates every
          orchestrator cron tick (every 30 min).
        </p>
        <p>
          <span className="text-[#64b5f6]">Infeasible</span> is auto-computed from sweep-state feasibility_ratio.
          <span className="ml-1 text-[#e040fb]">OOM</span> is reserved for structural failures declared in
          <code className="ml-1 rounded bg-[#21262d] px-1">known_oom</code>.
        </p>
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
}

function GroupRows({ group, hwOpen, expandedModel, onToggleHw, onToggleModel, allConcs }: GroupRowsProps) {
  const g = group;
  const pct = g.summary.totalNeed > 0
    ? Math.round((g.summary.totalHave / g.summary.totalNeed) * 100)
    : 0;
  const chips: [number, string, string][] = [
    [g.summary.complete,   'complete',   'text-[#3fb950]'],
    [g.summary.partial,    'partial',    'text-[#ff9800]'],
    [g.summary.running,    'running',    'text-[#58a6ff]'],
    [g.summary.pending,    'pending',    'text-[#a5b4fc]'],
    [g.summary.skipped,  'skipped',  'text-[#f97583]'],
    [g.summary.oom,        'OOM',        'text-[#e040fb]'],
    [g.summary.infeasible, 'infeasible', 'text-[#64b5f6]'],
    [g.summary.untested,   'untested',   'text-[#ff9800]'],
  ].filter(([n]) => (n as number) > 0) as [number, string, string][];

  return (
    <>
      <tr
        className="cursor-pointer border-b-2 border-t-2 border-[#30363d] bg-[#0d1117] hover:bg-[#161b22]"
        onClick={onToggleHw}
      >
        <td colSpan={2} className="px-3 py-2">
          <span className="mr-2 inline-block w-4 text-[#8b949e]">{hwOpen ? '▼' : '▶'}</span>
          <span className="font-mono text-sm font-semibold text-[#c9d1d9]">{g.hardware}</span>
          <span className="ml-3 text-[#8b949e]">
            {chips.map(([n, label, cls], i) => (
              <span key={label} className={cls}>
                {i > 0 && <span className="mx-1 text-[#30363d]">·</span>}
                {n} {label}
              </span>
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
      {hwOpen && g.models.map((m) => {
        const mKey = `${g.hardware}|${m.model}|${m.backend}`;
        return (
          <ModelRows
            key={mKey}
            hwName={g.hardware}
            model={m}
            open={expandedModel.has(mKey)}
            onToggle={() => onToggleModel(mKey)}
            allConcs={allConcs}
          />
        );
      })}
    </>
  );
}

interface ModelRowsProps {
  hwName: string;
  model: ModelEntry;
  open: boolean;
  onToggle: () => void;
  allConcs: number[];
}

function ModelRows({ hwName, model, open, onToggle, allConcs }: ModelRowsProps) {
  if (model.kind === 'status') {
    const bg = bgForStatus(model.status);
    const txt = colorForStatus(model.status);
    const label = labelForStatus(model.status);
    return (
      <tr className={`border-b border-[#21262d]/50 ${bg}`}>
        <td className="whitespace-nowrap px-3 py-1.5 pl-10 text-[#c9d1d9]">
          <span className="mr-2 inline-block w-3 text-[#30363d]">·</span>
          {model.model}
          <BackendBadge backend={model.backend} />
        </td>
        <td colSpan={allConcs.length + 1} className="whitespace-nowrap px-3 py-1.5">
          <div className="flex items-center gap-2">
            <StatusBadge kind={model.status} />
            <span className={txt}>
              {label}
              {model.attempt !== undefined && model.attempt > 0 && <span className="ml-1 text-[#8b949e]">· attempt {model.attempt}</span>}
              {model.reason && <span className="ml-1 text-[#8b949e]">— {model.reason}</span>}
              {model.updatedAt && <span className="ml-1 text-[#8b949e]">· since {new Date(model.updatedAt).toLocaleTimeString()}</span>}
            </span>
          </div>
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
          {model.status === 'untested' || model.status === 'pending' ||
           model.status === 'running'  || model.status === 'skipped' ? (
            <span className="text-[#8b949e]">0/{EXPECTED_CELLS_PER_MODEL}</span>
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
        <td className="whitespace-nowrap px-3 py-1.5 pl-10 text-[#c9d1d9]">
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
        const need = p.expected.length;
        const profPct = need > 0 ? Math.round((have / need) * 100) : 0;
        const profUntested = have === 0;
        return (
          <tr key={`${hwName}|${model.model}|${p.profile}`} className="border-b border-[#21262d]/50 bg-[#0d1117]/50">
            <td className="whitespace-nowrap px-3 py-1.5 pl-16 text-[#8b949e]">
              {/* empty — hw/model context established by parent rows */}
            </td>
            <td className="whitespace-nowrap px-3 py-1.5 text-[#8b949e]">
              {p.profile}
              {p.isMultiTurn && <span className="ml-1 rounded bg-[#8b5cf6]/20 px-1 text-[10px] text-[#8b5cf6]">mt</span>}
              {profUntested && <span className="ml-1 rounded border border-[#ff9800]/40 bg-[#ff9800]/10 px-1 text-[10px] text-[#ff9800] uppercase">todo</span>}
            </td>
            {allConcs.map((c) => {
              const expected = p.expected.includes(c);
              const present = p.present.has(c);
              const state: 'present' | 'missing' | 'na' =
                !expected ? 'na' : present ? 'present' : 'missing';
              return <td key={c} className="px-1 py-1.5 text-center"><Cell state={state} /></td>;
            })}
            <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
              <span
                className={
                  profPct === 100 ? 'text-[#3fb950]' :
                  profPct === 0 ? 'text-[#8b949e]' :
                  'text-[#ff9800]'
                }
              >
                {have}/{need}
              </span>
            </td>
          </tr>
        );
      })}
    </>
  );
}

// --- UI helpers ---

function SummaryCell({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-3">
      <div className="text-xs uppercase tracking-wide text-[#8b949e]">{label}</div>
      <div className="mt-1 font-mono text-xl font-semibold" style={{ color }}>{value}</div>
      <div className="text-[11px] text-[#8b949e]">{sub}</div>
    </div>
  );
}

function Cell({ state }: { state: 'present' | 'missing' | 'na' }) {
  const cls =
    state === 'present' ? 'bg-[#3fb950] border-[#3fb950]' :
    state === 'missing' ? 'bg-transparent border-[#30363d]' :
    'bg-[#21262d]/50 border-transparent';
  return <span className={`inline-block h-3 w-3 rounded-sm border ${cls}`} />;
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
    oom:        ['bg-[#e040fb]/15 text-[#e040fb] border-[#e040fb]/40', 'OOM'],
    infeasible: ['bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40', 'N/A'],
    running:    ['bg-[#58a6ff]/15 text-[#58a6ff] border-[#58a6ff]/40', 'RUN'],
    pending:    ['bg-[#a5b4fc]/15 text-[#a5b4fc] border-[#a5b4fc]/40', 'PND'],
    skipped:  ['bg-[#f97583]/15 text-[#f97583] border-[#f97583]/40', 'SKIP'],
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
    case 'oom':        return 'bg-[#e040fb]/5';
    case 'infeasible': return 'bg-[#64b5f6]/5';
    case 'running':    return 'bg-[#58a6ff]/5';
    case 'skipped':  return 'bg-[#f97583]/5';
    case 'pending':    return 'bg-[#a5b4fc]/5';
    default:           return '';
  }
}

function colorForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'text-[#e040fb]';
    case 'infeasible': return 'text-[#64b5f6]';
    case 'running':    return 'text-[#58a6ff]';
    case 'skipped':  return 'text-[#f97583]';
    case 'pending':    return 'text-[#a5b4fc]';
    default:           return 'text-[#8b949e]';
  }
}

function labelForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'OOM';
    case 'infeasible': return 'infeasible';
    case 'running':    return 'running';
    case 'skipped':  return 'skipped';
    case 'pending':    return 'pending';
    default:           return 'untested';
  }
}

// --- Predictor coverage sub-views ---

interface SubTabsProps {
  view: CoverageView;
  onChange: (v: CoverageView) => void;
  predictorCoverage: PredictorCoverage | null;
}

