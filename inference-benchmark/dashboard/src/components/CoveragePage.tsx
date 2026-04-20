import { useMemo } from 'react';
import type { BenchmarkResult } from '../types';

interface CoveragePageProps {
  allData: BenchmarkResult[];
  loading: boolean;
}

// Expected sweep matrix — mirrors inference-benchmark/scripts/bench_jobs.txt
// and the sweep_{all,multiturn}_profiles.sh default CONC lists.
const SINGLE_CONCS = [1, 10, 20, 40, 80, 120, 160, 200, 256, 320, 500];
const MULTI_CONCS = [5, 10, 20, 40, 80, 120, 160];

const SINGLE_PROFILES = ['chat-short', 'chat-medium', 'chat-long'];
const MULTI_PROFILES = [
  'chat-multiturn-short',
  'chat-multiturn-medium',
  'chat-multiturn-long',
  'terminalbench-multiturn-short',
];

const HARDWARE_ORDER = [
  'H100', 'H100x2', 'H100x4', 'H100x8',
  'A100', 'A100x2', 'A100x4', 'A100x8',
  '3090', '3090x2', '3090x4', '3090x8',
  '2080Ti', '2080Tix2', '2080Tix4',
  'A6000',
];

function sortHardware(a: string, b: string): number {
  const ai = HARDWARE_ORDER.indexOf(a);
  const bi = HARDWARE_ORDER.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b);
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}

interface Row {
  hardware: string;
  model: string;
  profile: string;
  isMultiTurn: boolean;
  expected: number[];
  present: Set<number>;
}

export function CoveragePage({ allData, loading }: CoveragePageProps) {
  const { rows, hardwareSet } = useMemo(() => {
    const bucket = new Map<string, Set<number>>();
    const hwSet = new Set<string>();
    const modelByHw = new Map<string, Set<string>>();

    for (const r of allData) {
      const key = `${r.hardware}|${r.modelShort}|${r.config.profile}`;
      if (!bucket.has(key)) bucket.set(key, new Set());
      bucket.get(key)!.add(r.config.concurrency);
      hwSet.add(r.hardware);
      if (!modelByHw.has(r.hardware)) modelByHw.set(r.hardware, new Set());
      modelByHw.get(r.hardware)!.add(r.modelShort);
    }

    const out: Row[] = [];
    const sortedHw = Array.from(hwSet).sort(sortHardware);
    for (const hw of sortedHw) {
      const models = Array.from(modelByHw.get(hw) ?? []).sort();
      for (const model of models) {
        for (const profile of SINGLE_PROFILES) {
          const key = `${hw}|${model}|${profile}`;
          out.push({
            hardware: hw,
            model,
            profile,
            isMultiTurn: false,
            expected: SINGLE_CONCS,
            present: bucket.get(key) ?? new Set(),
          });
        }
        for (const profile of MULTI_PROFILES) {
          const key = `${hw}|${model}|${profile}`;
          const present = bucket.get(key) ?? new Set();
          if (present.size === 0) continue;
          out.push({
            hardware: hw,
            model,
            profile,
            isMultiTurn: true,
            expected: MULTI_CONCS,
            present,
          });
        }
      }
    }
    return { rows: out, hardwareSet: sortedHw };
  }, [allData]);

  const allConcs = useMemo(() => {
    return Array.from(new Set([...SINGLE_CONCS, ...MULTI_CONCS])).sort((a, b) => a - b);
  }, []);

  if (loading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-[#8b949e]">Loading benchmark data...</div>
      </div>
    );
  }

  const summary = rows.reduce(
    (acc, r) => {
      const have = [...r.present].filter((c) => r.expected.includes(c)).length;
      const need = r.expected.length;
      if (have === 0) acc.empty += 1;
      else if (have < need) acc.partial += 1;
      else acc.complete += 1;
      acc.totalHave += have;
      acc.totalNeed += need;
      return acc;
    },
    { complete: 0, partial: 0, empty: 0, totalHave: 0, totalNeed: 0 }
  );

  const pct = summary.totalNeed > 0
    ? ((summary.totalHave / summary.totalNeed) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${summary.totalHave}/${summary.totalNeed} cells`} color="#00bcd4" />
        <SummaryCell label="Complete rows" value={`${summary.complete}`} sub="all concs present" color="#3fb950" />
        <SummaryCell label="Partial rows" value={`${summary.partial}`} sub="some missing" color="#ff9800" />
        <SummaryCell label="Empty rows" value={`${summary.empty}`} sub="profile untested" color="#f97583" />
      </div>

      <div className="flex flex-wrap items-center gap-4 rounded-md border border-[#21262d] bg-[#161b22] px-4 py-2 text-xs text-[#8b949e]">
        <span className="flex items-center gap-1.5"><Cell state="present" />present</span>
        <span className="flex items-center gap-1.5"><Cell state="missing" />expected &amp; missing</span>
        <span className="flex items-center gap-1.5"><Cell state="na" />not expected</span>
        <span className="ml-auto font-mono">
          rows: {rows.length} · hardware: {hardwareSet.length}
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
              const have = [...r.present].filter((c) => r.expected.includes(c)).length;
              const need = r.expected.length;
              const rowPct = need > 0 ? Math.round((have / need) * 100) : 0;
              const prevHw = i > 0 ? rows[i - 1].hardware : null;
              const prevModel = i > 0 ? rows[i - 1].model : null;
              const hwChange = r.hardware !== prevHw;
              const modelChange = r.model !== prevModel || hwChange;
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
                  </td>
                  {allConcs.map((c) => {
                    const expected = r.expected.includes(c);
                    const present = r.present.has(c);
                    const state = !expected ? 'na' : present ? 'present' : 'missing';
                    return (
                      <td key={c} className="px-1 py-1.5 text-center">
                        <Cell state={state} />
                      </td>
                    );
                  })}
                  <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono">
                    <span
                      className={
                        rowPct === 100 ? 'text-[#3fb950]' :
                        rowPct === 0 ? 'text-[#f97583]' :
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

      <div className="text-xs text-[#8b949e]">
        <p>
          Expected matrix is the sweep defaults in{' '}
          <code className="rounded bg-[#21262d] px-1">inference-benchmark/scripts/bench_jobs.txt</code>
          {' '}(single-turn concurrencies 1–500, multi-turn 5–160). Multi-turn rows only appear when
          at least one run exists for that (hardware, model, profile); unseen multi-turn profiles
          are not plotted as gaps since many jobs in the matrix skip them.
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
