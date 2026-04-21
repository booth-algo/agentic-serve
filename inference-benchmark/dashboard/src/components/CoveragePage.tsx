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

// All single-turn profiles benchmark runner knows about (see
// inference-benchmark/src/workloads/profiles.py). Shown in the
// table whenever the (hardware, model) has at least one result.
const ALL_SINGLE_PROFILES = [
  'chat-short',
  'chat-medium',
  'chat-long',
  'coding-agent',
  'prefill-heavy',
  'decode-heavy',
  'random-1k',
];

const ALL_MULTI_PROFILES = [
  'chat-multiturn-short',
  'chat-multiturn-medium',
  'chat-multiturn-long',
  'terminalbench-multiturn-short',
];

// Canonical hardware list — show every TP variant we physically have
// so 3090x4 / 3090x8 / 2080Tix4 untested combos are visible too.
// "A100-40GB" labels match build-data.ts so data from the a100_*
// result dirs (all SXM4-40GB in our cluster) lines up.
const EXPECTED_HARDWARE = [
  'H100', 'H100x2', 'H100x4', 'H100x8',
  'A100-40GB', 'A100-40GBx2', 'A100-40GBx4', 'A100-40GBx8',
  '3090', '3090x2', '3090x4', '3090x8',
  '2080Ti', '2080Tix2', '2080Tix4', '2080Tix8',
];

const HARDWARE_ORDER = EXPECTED_HARDWARE.concat(['A6000']);

// Total VRAM per hardware config (GB, rounded).
const HARDWARE_VRAM_GB: Record<string, number> = {
  'H100': 80, 'H100x2': 160, 'H100x4': 320, 'H100x8': 640,
  'A100-40GB': 40, 'A100-40GBx2': 80, 'A100-40GBx4': 160, 'A100-40GBx8': 320,
  '3090': 24, '3090x2': 48, '3090x4': 96, '3090x8': 192,
  '2080Ti': 22, '2080Tix2': 44, '2080Tix4': 88, '2080Tix8': 176,
  'A6000': 48,
};

// Model weight sizes in GB at bf16 (2 bytes per param). For MoE
// models this is the all-experts-resident figure; vLLM needs all
// experts in memory regardless of activation sparsity.
const MODEL_WEIGHTS_GB: Record<string, number> = {
  'Llama-3.1-8B': 16,
  'Llama-3.1-70B': 140,
  'Llama-3.3-70B': 140,
  'Mixtral-8x7B': 94,
  'Qwen2.5-72B': 144,
  'Qwen3.5-9B': 18,
  'Qwen3.5-27B': 54,
  'gpt-oss-20b': 40,
  'gpt-oss-120b': 240,
};

// Every model we care about benchmarking.
const ALL_MODELS = [
  'Llama-3.1-8B',
  'Llama-3.1-70B',
  'Llama-3.3-70B',
  'Mixtral-8x7B',
  'Qwen2.5-72B',
  'Qwen3.5-9B',
  'Qwen3.5-27B',
  'gpt-oss-20b',
  'gpt-oss-120b',
];

// Known-structural failures: (hardware|model) → short reason.
// Reserved for things that fail after a real attempt, not for things
// that are obviously too large (those are caught by checkFeasibility).
// OOM takes priority over the static feasibility rule, so entries here
// override "needs ≥X GB" messaging with the specific failure we saw.
const KNOWN_OOM: Record<string, string> = {
  // A100-40GBx4 (gpu-4, TP=4): 70B/72B weights + vLLM v0.19 cudagraph
  // accounting overrun even at gpu_mem=0.95, max_len=2048. Would fit on
  // A100-40GBx8 or on v0.18.
  'A100-40GBx4|Llama-3.1-70B': 'vLLM v0.19 cudagraph OOM — gpu_mem=0.95 + max_len=2048 still exceeds 160 GB',
  'A100-40GBx4|Llama-3.3-70B': 'vLLM v0.19 cudagraph OOM — gpu_mem=0.95 + max_len=2048 still exceeds 160 GB',
  'A100-40GBx4|Qwen2.5-72B':   '144 GB weights + cudagraph overhead exceeds 160 GB on vLLM v0.19',
  // Qwen3.5 hybrid attention: triton chunk_gated_delta_rule kernel OOMs
  // on prefill regardless of TP size; not weight-fit related.
  'A100-40GBx2|Qwen3.5-27B': 'hybrid-attn triton chunk_gated_delta_rule OOMs on prefill',
  'A100-40GBx4|Qwen3.5-27B': 'hybrid-attn triton chunk_gated_delta_rule OOMs on prefill',
  // 3090x4: Mixtral all-experts-resident (94 GB) doesn't fit with
  // cudagraph overhead in 96 GB. Would fit on 3090x8.
  '3090x4|Mixtral-8x7B': '94 GB MoE weights exceed 96 GB after cudagraph allocation',
};

// Infeasible if weights don't leave at least 15% headroom for KV cache
// + activations + cuda graphs. Returns a human-readable reason when the
// combo can't realistically fit, null otherwise.
function checkFeasibility(hw: string, model: string): string | null {
  const vram = HARDWARE_VRAM_GB[hw];
  const weights = MODEL_WEIGHTS_GB[model];
  if (!vram || !weights) return null;
  if (weights > vram * 0.85) {
    const minGb = Math.ceil(weights / 0.85);
    return `needs ≥${minGb} GB VRAM (weights ${weights} GB); this config has ${vram} GB`;
  }
  return null;
}

function sortHardware(a: string, b: string): number {
  const ai = HARDWARE_ORDER.indexOf(a);
  const bi = HARDWARE_ORDER.indexOf(b);
  if (ai === -1 && bi === -1) return a.localeCompare(b);
  if (ai === -1) return 1;
  if (bi === -1) return -1;
  return ai - bi;
}

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
  status: 'oom' | 'untested' | 'infeasible';
  reason?: string;
}

type AnyRow = DataRow | StatusRow;

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

    const out: AnyRow[] = [];
    const fullHwSet = new Set<string>([...EXPECTED_HARDWARE, ...hwSet]);
    const sortedHw = Array.from(fullHwSet).sort(sortHardware);

    for (const hw of sortedHw) {
      const modelsWithData = modelByHw.get(hw) ?? new Set();
      const extras = Array.from(modelsWithData).filter((m) => !ALL_MODELS.includes(m));
      const modelList = [...ALL_MODELS, ...extras.sort()];

      for (const model of modelList) {
        const hasData = modelsWithData.has(model);
        const oomReason = KNOWN_OOM[`${hw}|${model}`];
        const infeasReason = checkFeasibility(hw, model);

        // OOM takes priority (confirmed via real run). Then infeasibility
        // (static rule). Then untested if no data. If data exists, emit
        // a full profile grid regardless.
        if (!hasData) {
          let status: StatusRow['status'];
          let reason: string | undefined;
          if (oomReason) {
            status = 'oom';
            reason = oomReason;
          } else if (infeasReason) {
            status = 'infeasible';
            reason = infeasReason;
          } else {
            status = 'untested';
          }
          out.push({ kind: 'status', hardware: hw, model, status, reason });
          continue;
        }

        // Has data → list ALL profiles (single + multi), even profiles
        // with zero runs, so gaps are explicit.
        for (const profile of ALL_SINGLE_PROFILES) {
          const key = `${hw}|${model}|${profile}`;
          out.push({
            kind: 'data',
            hardware: hw,
            model,
            profile,
            isMultiTurn: false,
            expected: SINGLE_CONCS,
            present: bucket.get(key) ?? new Set(),
          });
        }
        for (const profile of ALL_MULTI_PROFILES) {
          const key = `${hw}|${model}|${profile}`;
          out.push({
            kind: 'data',
            hardware: hw,
            model,
            profile,
            isMultiTurn: true,
            expected: MULTI_CONCS,
            present: bucket.get(key) ?? new Set(),
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
      if (r.kind === 'status') {
        if (r.status === 'oom') acc.oom += 1;
        else if (r.status === 'infeasible') acc.infeasible += 1;
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
    { complete: 0, partial: 0, empty: 0, oom: 0, infeasible: 0, untested: 0, totalHave: 0, totalNeed: 0 }
  );

  const pct = summary.totalNeed > 0
    ? ((summary.totalHave / summary.totalNeed) * 100).toFixed(1)
    : '0.0';

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        <SummaryCell label="Overall" value={`${pct}%`} sub={`${summary.totalHave}/${summary.totalNeed} cells`} color="#00bcd4" />
        <SummaryCell label="Complete" value={`${summary.complete}`} sub="all concs present" color="#3fb950" />
        <SummaryCell label="Partial" value={`${summary.partial}`} sub="some missing" color="#ff9800" />
        <SummaryCell label="Empty" value={`${summary.empty}`} sub="profile tried, 0 results" color="#f97583" />
        <SummaryCell label="OOM" value={`${summary.oom}`} sub="structurally blocked" color="#e040fb" />
        <SummaryCell label="Infeasible" value={`${summary.infeasible}`} sub="VRAM too small" color="#64b5f6" />
        <SummaryCell label="Untested" value={`${summary.untested}`} sub="not yet attempted" color="#ff9800" />
      </div>

      <div className="flex flex-wrap items-center gap-4 rounded-md border border-[#21262d] bg-[#161b22] px-4 py-2 text-xs text-[#8b949e]">
        <span className="flex items-center gap-1.5"><Cell state="present" />present</span>
        <span className="flex items-center gap-1.5"><Cell state="missing" />expected &amp; missing</span>
        <span className="flex items-center gap-1.5"><Cell state="na" />not expected</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="oom" />OOM / structural</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="infeasible" />infeasible (VRAM)</span>
        <span className="flex items-center gap-1.5"><StatusBadge kind="untested" />untested</span>
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
              const prevHw = i > 0 ? rows[i - 1].hardware : null;
              const prevModel = i > 0 ? rows[i - 1].model : null;
              const hwChange = r.hardware !== prevHw;
              const modelChange = r.model !== prevModel || hwChange;

              if (r.kind === 'status') {
                const bg =
                  r.status === 'oom' ? 'bg-[#e040fb]/5' :
                  r.status === 'infeasible' ? 'bg-[#64b5f6]/5' :
                  '';
                const txtColor =
                  r.status === 'oom' ? 'text-[#e040fb]' :
                  r.status === 'infeasible' ? 'text-[#64b5f6]' :
                  'text-[#8b949e]';
                const label =
                  r.status === 'oom' ? 'OOM' :
                  r.status === 'infeasible' ? 'infeasible' :
                  'untested';
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
                    <td
                      colSpan={allConcs.length + 1}
                      className="whitespace-nowrap px-3 py-1.5"
                    >
                      <div className="flex items-center gap-2">
                        <StatusBadge kind={r.status} />
                        <span className={txtColor}>
                          {label}
                          {r.reason ? <span className="ml-1 text-[#8b949e]">— {r.reason}</span> : null}
                        </span>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-right font-mono text-[#8b949e]">—</td>
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
                    // Concurrencies outside the expected set render muted.
                    // Everything in the expected set that we don't yet have
                    // — including fully-untested profiles — renders as
                    // "expected & missing" (outlined), NOT muted, since
                    // these are runs we still want to do.
                    const state: 'present' | 'missing' | 'na' =
                      !expected ? 'na' : present ? 'present' : 'missing';
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
          Expected sweep matrix: single-turn concurrencies {SINGLE_CONCS.join(', ')}; multi-turn {MULTI_CONCS.join(', ')}.
          Single-turn profiles: {ALL_SINGLE_PROFILES.length}. Multi-turn profiles: {ALL_MULTI_PROFILES.length}.
        </p>
        <p>
          <span className="text-[#64b5f6]">Infeasible</span> is auto-computed from hardware VRAM vs. model weights (bf16),
          using a 0.85 fit ratio. <span className="text-[#e040fb]">OOM</span> is reserved for confirmed run-time failures that
          need a specific stack-level fix.
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

function StatusBadge({ kind }: { kind: 'oom' | 'untested' | 'infeasible' }) {
  let cls: string;
  let label: string;
  if (kind === 'oom') {
    cls = 'bg-[#e040fb]/15 text-[#e040fb] border-[#e040fb]/40';
    label = 'OOM';
  } else if (kind === 'infeasible') {
    cls = 'bg-[#64b5f6]/15 text-[#64b5f6] border-[#64b5f6]/40';
    label = 'N/A';
  } else {
    // Untested = "expected, not yet run" — use the same amber as
    // partial coverage so it reads as a gap we intend to fill, not
    // as "not expected".
    cls = 'bg-[#ff9800]/10 text-[#ff9800] border-[#ff9800]/40';
    label = 'TODO';
  }
  return (
    <span className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${cls}`}>
      {label}
    </span>
  );
}
