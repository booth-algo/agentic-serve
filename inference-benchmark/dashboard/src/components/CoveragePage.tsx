import { useMemo, useState } from 'react';
import type { BenchmarkResult } from '../types';
import type { SweepCell, SweepState } from '../types-sweep';

interface CoveragePageProps {
  allData: BenchmarkResult[];
  sweepState: SweepState | null;
  loading: boolean;
}

const SINGLE_CONCS = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500];
const MULTI_CONCS = [5, 10, 20, 40, 80, 120, 160];

const ALL_SINGLE_PROFILES = [
  'chat-short', 'chat-medium', 'chat-long',
  'coding-agent', 'prefill-heavy', 'decode-heavy', 'random-1k',
];
const ALL_MULTI_PROFILES = [
  'chat-multiturn-short', 'chat-multiturn-medium', 'chat-multiturn-long',
  'terminalbench-multiturn-short',
];

const TP_OPTIONS = [1, 2, 4, 8];

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
  profiles: ProfileRow[];
  // Aggregate coverage across all profiles.
  totalHave: number;
  totalNeed: number;
}

interface StatusModel {
  kind: 'status';
  hardware: string;
  model: string;
  status: 'oom' | 'untested' | 'infeasible' | 'running' | 'abandoned' | 'pending';
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
    abandoned: number;
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
  known_oom: 5, abandoned: 4, running: 3, pending: 2, done: 1,
};

function aggregateCells(cells: SweepCell[]): Map<string, SweepCell> {
  const out = new Map<string, SweepCell>();
  for (const c of cells) {
    const key = `${c.hw_label}|${c.model}`;
    const prev = out.get(key);
    if (!prev || STATUS_PRIORITY[c.status] > STATUS_PRIORITY[prev.status]) {
      out.set(key, c);
    }
  }
  return out;
}

export function CoveragePage({ allData, sweepState, loading }: CoveragePageProps) {
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
    const hwHasData = new Map<string, Set<string>>();
    for (const r of allData) {
      const k = `${r.hardware}|${r.modelShort}|${r.config.profile}`;
      if (!bucket.has(k)) bucket.set(k, new Set());
      bucket.get(k)!.add(r.config.concurrency);
      if (!hwHasData.has(r.hardware)) hwHasData.set(r.hardware, new Set());
      hwHasData.get(r.hardware)!.add(r.modelShort);
    }

    const aggStatus = sweepState
      ? aggregateCells(sweepState.cells)
      : new Map<string, SweepCell>();

    const hwGroups: HwGroup[] = [];
    for (const hw of expectedHw) {
      const models: ModelEntry[] = [];
      const summary = {
        complete: 0, partial: 0,
        running: 0, pending: 0, abandoned: 0,
        oom: 0, infeasible: 0, untested: 0,
        totalHave: 0, totalNeed: 0,
      };
      for (const model of modelList) {
        const key = `${hw}|${model}`;
        const hasData = hwHasData.get(hw)?.has(model) ?? false;
        const cell = aggStatus.get(key);

        if (hasData) {
          const profiles: ProfileRow[] = [];
          let totalHave = 0;
          let totalNeed = 0;
          for (const profile of ALL_SINGLE_PROFILES) {
            const present = bucket.get(`${hw}|${model}|${profile}`) ?? new Set<number>();
            const have = [...present].filter((c) => SINGLE_CONCS.includes(c)).length;
            totalHave += have;
            totalNeed += SINGLE_CONCS.length;
            profiles.push({ profile, isMultiTurn: false, expected: SINGLE_CONCS, present });
          }
          for (const profile of ALL_MULTI_PROFILES) {
            const present = bucket.get(`${hw}|${model}|${profile}`) ?? new Set<number>();
            const have = [...present].filter((c) => MULTI_CONCS.includes(c)).length;
            totalHave += have;
            totalNeed += MULTI_CONCS.length;
            profiles.push({ profile, isMultiTurn: true, expected: MULTI_CONCS, present });
          }
          models.push({ kind: 'data', hardware: hw, model, profiles, totalHave, totalNeed });
          summary.totalHave += totalHave;
          summary.totalNeed += totalNeed;
          if (totalHave === totalNeed) summary.complete += 1;
          else summary.partial += 1;
          continue;
        }

        if (cell) {
          if (cell.status === 'known_oom') {
            models.push({ kind: 'status', hardware: hw, model, status: 'oom', reason: cell.reason ?? undefined });
            summary.oom += 1;
            continue;
          }
          if (cell.status === 'running') {
            models.push({ kind: 'status', hardware: hw, model, status: 'running', attempt: cell.attempt, updatedAt: cell.updated_at });
            summary.running += 1;
            continue;
          }
          if (cell.status === 'abandoned') {
            models.push({ kind: 'status', hardware: hw, model, status: 'abandoned', reason: cell.reason ?? undefined, attempt: cell.attempt });
            summary.abandoned += 1;
            continue;
          }
          if (cell.status === 'pending' || cell.status === 'done') {
            models.push({ kind: 'status', hardware: hw, model, status: 'pending' });
            summary.pending += 1;
            continue;
          }
        }

        const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
        if (infReason) {
          models.push({ kind: 'status', hardware: hw, model, status: 'infeasible', reason: infReason });
          summary.infeasible += 1;
        } else {
          models.push({ kind: 'status', hardware: hw, model, status: 'untested' });
          summary.untested += 1;
        }
      }
      hwGroups.push({ hardware: hw, models, summary });
    }

    return { groups: hwGroups, hardwareList: expectedHw, sweepMtime: sweepState?.generated_at ?? null };
  }, [allData, sweepState]);

  const [expandedHw, setExpandedHw] = useState<Set<string>>(() => {
    // Auto-expand HW groups that have any active status (running/abandoned/partial).
    const set = new Set<string>();
    for (const g of groups) {
      if (g.summary.running > 0 || g.summary.abandoned > 0 || g.summary.partial > 0) {
        set.add(g.hardware);
      }
    }
    return set;
  });
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
      acc.abandoned += g.summary.abandoned;
      acc.oom += g.summary.oom;
      acc.infeasible += g.summary.infeasible;
      acc.untested += g.summary.untested;
      acc.totalHave += g.summary.totalHave;
      acc.totalNeed += g.summary.totalNeed;
      return acc;
    },
    { complete: 0, partial: 0, running: 0, pending: 0, abandoned: 0, oom: 0, infeasible: 0, untested: 0, totalHave: 0, totalNeed: 0 },
  );
  const pct = grand.totalNeed > 0
    ? ((grand.totalHave / grand.totalNeed) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${grand.totalHave}/${grand.totalNeed} cells`} color="#00bcd4" />
        <SummaryCell label="Complete" value={`${grand.complete}`} sub="all concs present" color="#3fb950" />
        <SummaryCell label="Partial" value={`${grand.partial}`} sub="some missing" color="#ff9800" />
        <SummaryCell label="Running" value={`${grand.running}`} sub="in progress" color="#58a6ff" />
        <SummaryCell label="Pending" value={`${grand.pending}`} sub="queued" color="#a5b4fc" />
        <SummaryCell label="Abandoned" value={`${grand.abandoned}`} sub="failed after retry" color="#f97583" />
        <SummaryCell label="OOM" value={`${grand.oom}`} sub="structurally blocked" color="#e040fb" />
        <SummaryCell label="Infeasible" value={`${grand.infeasible}`} sub="VRAM too small" color="#64b5f6" />
      </div>

      <div className="flex flex-wrap items-center gap-4 rounded-md border border-[#21262d] bg-[#161b22] px-4 py-2 text-xs text-[#8b949e]">
        <span className="flex items-center gap-1.5"><Cell state="present" />present</span>
        <span className="flex items-center gap-1.5"><Cell state="missing" />expected &amp; missing</span>
        <span className="flex items-center gap-1.5"><Cell state="na" />not expected</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="running" />running</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="pending" />pending</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="abandoned" />abandoned</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="oom" />OOM</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="infeasible" />infeasible</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="untested" />untested</span>
        <span className="ml-auto flex items-center gap-3 font-mono">
          <button onClick={expandAll} className="rounded border border-[#30363d] px-2 py-0.5 text-[11px] text-[#c9d1d9] hover:border-[#58a6ff] hover:text-[#58a6ff]">expand all</button>
          <button onClick={collapseAll} className="rounded border border-[#30363d] px-2 py-0.5 text-[11px] text-[#c9d1d9] hover:border-[#58a6ff] hover:text-[#58a6ff]">collapse all</button>
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
    [g.summary.abandoned,  'abandoned',  'text-[#f97583]'],
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
      {hwOpen && g.models.map((m) => (
        <ModelRows
          key={`${g.hardware}|${m.model}`}
          hwName={g.hardware}
          model={m}
          open={expandedModel.has(`${g.hardware}|${m.model}`)}
          onToggle={() => onToggleModel(`${g.hardware}|${m.model}`)}
          allConcs={allConcs}
        />
      ))}
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
        <td className="px-3 py-1.5 text-right text-[#8b949e]">—</td>
      </tr>
    );
  }

  // model.kind === 'data'
  const rowPct = model.totalNeed > 0 ? Math.round((model.totalHave / model.totalNeed) * 100) : 0;
  // Aggregate presence across all profiles, per concurrency, for a glance view
  // on the collapsed model row.
  const aggPresent = new Set<number>();
  const aggExpected = new Set<number>();
  for (const p of model.profiles) {
    for (const c of p.present) aggPresent.add(c);
    for (const c of p.expected) aggExpected.add(c);
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
        </td>
        <td className="whitespace-nowrap px-3 py-1.5 text-[#8b949e]">
          <span className="text-[10px] uppercase tracking-wide">{model.profiles.length} profiles</span>
        </td>
        {allConcs.map((c) => {
          const expected = aggExpected.has(c);
          const present = aggPresent.has(c);
          const state: 'present' | 'missing' | 'na' =
            !expected ? 'na' : present ? 'present' : 'missing';
          return <td key={c} className="px-1 py-1.5 text-center"><Cell state={state} /></td>;
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

type BadgeKind = StatusModel['status'];

function StatusBadge({ kind }: { kind: BadgeKind }) {
  const map: Record<BadgeKind, [string, string]> = {
    oom:        ['bg-[#e040fb]/15 text-[#e040fb] border-[#e040fb]/40', 'OOM'],
    infeasible: ['bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40', 'N/A'],
    running:    ['bg-[#58a6ff]/15 text-[#58a6ff] border-[#58a6ff]/40', 'RUN'],
    pending:    ['bg-[#a5b4fc]/15 text-[#a5b4fc] border-[#a5b4fc]/40', 'PND'],
    abandoned:  ['bg-[#f97583]/15 text-[#f97583] border-[#f97583]/40', 'ABD'],
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
    case 'abandoned':  return 'bg-[#f97583]/5';
    case 'pending':    return 'bg-[#a5b4fc]/5';
    default:           return '';
  }
}

function colorForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'text-[#e040fb]';
    case 'infeasible': return 'text-[#64b5f6]';
    case 'running':    return 'text-[#58a6ff]';
    case 'abandoned':  return 'text-[#f97583]';
    case 'pending':    return 'text-[#a5b4fc]';
    default:           return 'text-[#8b949e]';
  }
}

function labelForStatus(s: StatusModel['status']): string {
  switch (s) {
    case 'oom':        return 'OOM';
    case 'infeasible': return 'infeasible';
    case 'running':    return 'running';
    case 'abandoned':  return 'abandoned';
    case 'pending':    return 'pending';
    default:           return 'untested';
  }
}
