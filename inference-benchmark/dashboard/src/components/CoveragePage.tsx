import { useMemo } from 'react';
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

interface DataRow {
  kind: 'data';
  hardware: string;
  model: string;
  profile: string;
  isMultiTurn: boolean;
  expected: number[];
  present: Set<number>;
}

interface StatusRow {
  kind: 'status';
  hardware: string;
  model: string;
  status: 'oom' | 'untested' | 'infeasible' | 'running' | 'abandoned' | 'pending';
  reason?: string;
  attempt?: number;
  updatedAt?: string | null;
}

type AnyRow = DataRow | StatusRow;

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
  known_oom: 5,
  abandoned: 4,
  running: 3,
  pending: 2,
  done: 1,
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
  const { rows, hardwareList, sweepMtime } = useMemo(() => {
    const baseHwLabels = sweepState
      ? Object.values(sweepState.hosts).map((h) => h.hardware_label)
      : ['A100-40GB', '3090', '2080Ti', 'H100'];
    const expectedHw: string[] = [];
    for (const base of baseHwLabels) {
      for (const tp of TP_OPTIONS) {
        expectedHw.push(hwLabel(base, tp));
      }
    }
    const dataHw = new Set(allData.map((r) => r.hardware));
    for (const hw of dataHw) {
      if (!expectedHw.includes(hw)) expectedHw.push(hw);
    }

    const expectedModels = new Set<string>();
    if (sweepState) {
      for (const m of Object.keys(sweepState.models)) expectedModels.add(m);
    }
    for (const r of allData) expectedModels.add(r.modelShort);
    const modelList = Array.from(expectedModels).sort();

    const vramByBase = new Map<string, number>();
    if (sweepState) {
      for (const h of Object.values(sweepState.hosts)) {
        vramByBase.set(h.hardware_label, h.vram_gb_per_gpu);
      }
    }
    function vramFor(hw: string): number | undefined {
      const m = hw.match(/^(.+?)(?:x(\d+))?$/);
      if (!m) return undefined;
      return vramByBase.get(m[1]);
    }
    function tpOf(hw: string): number {
      const m = hw.match(/x(\d+)$/);
      return m ? parseInt(m[1], 10) : 1;
    }
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

    const out: AnyRow[] = [];
    for (const hw of expectedHw) {
      for (const model of modelList) {
        const key = `${hw}|${model}`;
        const hasData = hwHasData.get(hw)?.has(model) ?? false;
        const cell = aggStatus.get(key);

        if (hasData) {
          for (const profile of ALL_SINGLE_PROFILES) {
            out.push({
              kind: 'data', hardware: hw, model, profile, isMultiTurn: false,
              expected: SINGLE_CONCS,
              present: bucket.get(`${hw}|${model}|${profile}`) ?? new Set(),
            });
          }
          for (const profile of ALL_MULTI_PROFILES) {
            out.push({
              kind: 'data', hardware: hw, model, profile, isMultiTurn: true,
              expected: MULTI_CONCS,
              present: bucket.get(`${hw}|${model}|${profile}`) ?? new Set(),
            });
          }
          continue;
        }

        if (cell) {
          if (cell.status === 'known_oom') {
            out.push({ kind: 'status', hardware: hw, model, status: 'oom', reason: cell.reason ?? undefined });
            continue;
          }
          if (cell.status === 'running') {
            out.push({ kind: 'status', hardware: hw, model, status: 'running', attempt: cell.attempt, updatedAt: cell.updated_at });
            continue;
          }
          if (cell.status === 'abandoned') {
            out.push({ kind: 'status', hardware: hw, model, status: 'abandoned', reason: cell.reason ?? undefined, attempt: cell.attempt });
            continue;
          }
          if (cell.status === 'pending' || cell.status === 'done') {
            out.push({ kind: 'status', hardware: hw, model, status: 'pending' });
            continue;
          }
        }

        const infReason = infeasibilityReason(vramFor(hw), weightsFor(model), tpOf(hw), ratio);
        if (infReason) {
          out.push({ kind: 'status', hardware: hw, model, status: 'infeasible', reason: infReason });
        } else {
          out.push({ kind: 'status', hardware: hw, model, status: 'untested' });
        }
      }
    }

    return { rows: out, hardwareList: expectedHw, sweepMtime: sweepState?.generated_at ?? null };
  }, [allData, sweepState]);

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

  const summary = rows.reduce(
    (acc, r) => {
      if (r.kind === 'status') {
        if (r.status === 'oom') acc.oom += 1;
        else if (r.status === 'infeasible') acc.infeasible += 1;
        else if (r.status === 'running') acc.running += 1;
        else if (r.status === 'abandoned') acc.abandoned += 1;
        else if (r.status === 'pending') acc.pending += 1;
        else acc.untested += 1;
        return acc;
      }
      const have = [...r.present].filter((c) => r.expected.includes(c)).length;
      const need = r.expected.length;
      if (have === 0) acc.empty += 1;
      else if (have < need) acc.partial += 1;
      else acc.complete += 1;
      acc.totalHave += have;
      acc.totalNeed += need;
      return acc;
    },
    {
      complete: 0, partial: 0, empty: 0,
      oom: 0, infeasible: 0, running: 0, abandoned: 0, pending: 0, untested: 0,
      totalHave: 0, totalNeed: 0,
    },
  );

  const pct = summary.totalNeed > 0
    ? ((summary.totalHave / summary.totalNeed) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-8">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${summary.totalHave}/${summary.totalNeed} cells`} color="#00bcd4" />
        <SummaryCell label="Complete" value={`${summary.complete}`} sub="all concs present" color="#3fb950" />
        <SummaryCell label="Partial" value={`${summary.partial}`} sub="some missing" color="#ff9800" />
        <SummaryCell label="Running" value={`${summary.running}`} sub="in progress" color="#58a6ff" />
        <SummaryCell label="Pending" value={`${summary.pending}`} sub="queued" color="#a5b4fc" />
        <SummaryCell label="Abandoned" value={`${summary.abandoned}`} sub="failed after retry" color="#f97583" />
        <SummaryCell label="OOM" value={`${summary.oom}`} sub="structurally blocked" color="#e040fb" />
        <SummaryCell label="Infeasible" value={`${summary.infeasible}`} sub="VRAM too small" color="#64b5f6" />
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
        <span className="ml-auto font-mono">
          rows: {rows.length} · hardware: {hardwareList.length}
          {sweepMtime && <span className="ml-2">· sweep-state: {new Date(sweepMtime).toLocaleTimeString()}</span>}
        </span>
      </div>

      <div className="overflow-x-auto rounded-lg border border-[#21262d] bg-[#161b22]">
        <table className="min-w-full border-collapse text-xs">
          <thead className="sticky top-0 z-10 bg-[#161b22]">
            <tr className="border-b border-[#21262d] text-[#8b949e]">
              <th className="px-3 py-2 text-left font-medium">Hardware</th>
              <th className="px-3 py-2 text-left font-medium">Model</th>
              <th className="px-3 py-2 text-left font-medium">Profile</th>
              {allConcs.map((c) => (
                <th key={c} className="px-1.5 py-2 text-center font-mono font-normal">{c}</th>
              ))}
              <th className="px-3 py-2 text-right font-medium">Coverage</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const prevHw = i > 0 ? rows[i - 1].hardware : null;
              const prevModel = i > 0 ? rows[i - 1].model : null;
              const hwChange = r.hardware !== prevHw;
              const modelChange = r.model !== prevModel || hwChange;

              if (r.kind === 'status') {
                const bg = bgForStatus(r.status);
                const txt = colorForStatus(r.status);
                const label = labelForStatus(r.status);
                return (
                  <tr
                    key={`${r.hardware}|${r.model}|__${r.status}`}
                    className={`border-b border-[#21262d]/50 ${bg} ${hwChange ? 'border-t-2 border-t-[#30363d]' : ''}`}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[#c9d1d9]">
                      {hwChange ? r.hardware : ''}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">
                      {modelChange ? r.model : ''}
                    </td>
                    <td colSpan={allConcs.length + 1} className="whitespace-nowrap px-3 py-1.5">
                      <div className="flex items-center gap-2">
                        <StatusBadge kind={r.status} />
                        <span className={txt}>
                          {label}
                          {r.attempt !== undefined && r.attempt > 0 && <span className="ml-1 text-[#8b949e]">· attempt {r.attempt}</span>}
                          {r.reason && <span className="ml-1 text-[#8b949e]">— {r.reason}</span>}
                          {r.updatedAt && <span className="ml-1 text-[#8b949e]">· since {new Date(r.updatedAt).toLocaleTimeString()}</span>}
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              }

              const have = [...r.present].filter((c) => r.expected.includes(c)).length;
              const need = r.expected.length;
              const rowPct = need > 0 ? Math.round((have / need) * 100) : 0;
              const profileUntested = have === 0;
              return (
                <tr
                  key={`${r.hardware}|${r.model}|${r.profile}`}
                  className={`border-b border-[#21262d]/50 ${hwChange ? 'border-t-2 border-t-[#30363d]' : ''}`}
                >
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[#c9d1d9]">
                    {hwChange ? r.hardware : ''}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-[#c9d1d9]">
                    {modelChange ? r.model : ''}
                  </td>
                  <td className="whitespace-nowrap px-3 py-1.5 text-[#8b949e]">
                    {r.profile}
                    {r.isMultiTurn && <span className="ml-1 rounded bg-[#8b5cf6]/20 px-1 text-[10px] text-[#8b5cf6]">mt</span>}
                    {profileUntested && <span className="ml-1 rounded border border-[#ff9800]/40 bg-[#ff9800]/10 px-1 text-[10px] text-[#ff9800] uppercase">todo</span>}
                  </td>
                  {allConcs.map((c) => {
                    const expected = r.expected.includes(c);
                    const present = r.present.has(c);
                    const state: 'present' | 'missing' | 'na' =
                      !expected ? 'na' : present ? 'present' : 'missing';
                    return (
                      <td key={c} className="px-1 py-1.5 text-center"><Cell state={state} /></td>
                    );
                  })}
                  <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
                    <span
                      className={
                        rowPct === 100 ? 'text-[#3fb950]' :
                        rowPct === 0 ? 'text-[#8b949e]' :
                        'text-[#ff9800]'
                      }
                    >
                      {have}/{need}
                    </span>
                  </td>
                </tr>
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

type BadgeKind = StatusRow['status'];

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

function bgForStatus(s: StatusRow['status']): string {
  switch (s) {
    case 'oom':        return 'bg-[#e040fb]/5';
    case 'infeasible': return 'bg-[#64b5f6]/5';
    case 'running':    return 'bg-[#58a6ff]/5';
    case 'abandoned':  return 'bg-[#f97583]/5';
    case 'pending':    return 'bg-[#a5b4fc]/5';
    default:           return '';
  }
}

function colorForStatus(s: StatusRow['status']): string {
  switch (s) {
    case 'oom':        return 'text-[#e040fb]';
    case 'infeasible': return 'text-[#64b5f6]';
    case 'running':    return 'text-[#58a6ff]';
    case 'abandoned':  return 'text-[#f97583]';
    case 'pending':    return 'text-[#a5b4fc]';
    default:           return 'text-[#8b949e]';
  }
}

function labelForStatus(s: StatusRow['status']): string {
  switch (s) {
    case 'oom':        return 'OOM';
    case 'infeasible': return 'infeasible';
    case 'running':    return 'running';
    case 'abandoned':  return 'abandoned';
    case 'pending':    return 'pending';
    default:           return 'untested';
  }
}
