import { useState, useEffect, useMemo, useCallback, Component, type ReactNode } from 'react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  ReferenceArea,
  ReferenceLine,
  ZAxis,
} from 'recharts';
import type { RooflineQuadrantData, QuadrantPoint } from '../types-roofline-quadrant';

declare const __BUILD_HASH__: string;

// ── Quadrant colors ─────────────────────────────────────────────────────────
const QUADRANT = {
  computeBound:   { bg: 'rgba(0,114,178,0.18)',  label: 'Compute Bound',         color: '#0072B2' },
  capacityBound:  { bg: 'rgba(255,183,77,0.18)', label: 'Memory Capacity Bound', color: '#ffb74d' },
  bwBound:        { bg: 'rgba(63,185,80,0.18)',   label: 'Memory BW Bound',       color: '#3fb950' },
  bothBound:      { bg: 'rgba(139,148,158,0.12)', label: 'Capacity + BW Bound',   color: '#8b949e' },
};

// ── Model colors (colorblind-safe) ──────────────────────────────────────────
const MODEL_COLORS: Record<string, string> = {
  'Llama-3.1-8B':  '#0072B2',
  'Qwen3.5-9B':    '#E69F00',
  'gpt-oss-20b':   '#009E73',
  'Qwen3.5-27B':   '#CC79A7',
  'Qwen3-32B':     '#D55E00',
  'Qwen2.5-72B':   '#56B4E9',
  'Llama-3.1-70B': '#F0E442',
  'Llama-3.3-70B': '#f97583',
  'gpt-oss-120b':  '#a855f7',
};

function modelColor(model: string): string {
  return MODEL_COLORS[model] ?? '#8b949e';
}

// ── Profile shapes ──────────────────────────────────────────────────────────
type ProfileType = 'chat' | 'coding' | 'synthetic' | 'multiturn';

function profileType(profile: string): ProfileType {
  if (profile.startsWith('chat-multiturn') || profile.includes('multiturn')) return 'multiturn';
  if (profile.startsWith('chat-')) return 'chat';
  if (profile.startsWith('coding') || profile.startsWith('swebench') || profile.startsWith('terminalbench')) return 'coding';
  return 'synthetic';
}

const PROFILE_TYPE_LABELS: Record<ProfileType, string> = {
  chat: 'Chat',
  coding: 'Coding/Agent',
  synthetic: 'Synthetic',
  multiturn: 'Multi-turn',
};

// ── Error boundary ──────────────────────────────────────────────────────────
interface EBProps { children: ReactNode; fallback?: ReactNode }
interface EBState { hasError: boolean; error?: Error }
class ChartErrorBoundary extends Component<EBProps, EBState> {
  state: EBState = { hasError: false };
  static getDerivedStateFromError(error: Error) { return { hasError: true, error }; }
  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? (
        <div className="flex h-64 items-center justify-center rounded-lg border border-red-800 bg-red-900/20 text-red-400">
          <div className="text-center">
            <div className="text-sm font-semibold">Chart rendering error</div>
            <div className="mt-1 text-xs opacity-70">{this.state.error?.message}</div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// ── Custom scatter shape ────────────────────────────────────────────────────
interface DotProps {
  cx?: number;
  cy?: number;
  payload?: QuadrantPoint & { _size: number };
}

function QuadrantDot({ cx = 0, cy = 0, payload }: DotProps) {
  if (!payload || !Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  const color = modelColor(payload.model);
  const r = Math.max(4, Math.min(18, (payload._size ?? 5)));
  const pType = profileType(payload.profile);

  if (pType === 'coding') {
    // Square
    const s = r * 1.6;
    return (
      <rect
        x={cx - s / 2} y={cy - s / 2}
        width={s} height={s}
        fill={color} fillOpacity={0.55}
        stroke={color} strokeWidth={1} strokeOpacity={0.3}
        rx={2}
      />
    );
  }
  if (pType === 'synthetic') {
    // Triangle
    const h = r * 1.8;
    const points = `${cx},${cy - h / 2} ${cx - h / 2},${cy + h / 2} ${cx + h / 2},${cy + h / 2}`;
    return (
      <polygon
        points={points}
        fill={color} fillOpacity={0.55}
        stroke={color} strokeWidth={1} strokeOpacity={0.3}
      />
    );
  }
  if (pType === 'multiturn') {
    // Diamond
    const s = r * 1.4;
    const points = `${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`;
    return (
      <polygon
        points={points}
        fill={color} fillOpacity={0.55}
        stroke={color} strokeWidth={1} strokeOpacity={0.3}
      />
    );
  }
  // Circle (chat)
  return (
    <circle
      cx={cx} cy={cy} r={r}
      fill={color} fillOpacity={0.55}
      stroke={color} strokeWidth={1} strokeOpacity={0.3}
    />
  );
}

// Tooltip removed — side panel provides point details on hover

// ── Log tick formatter ──────────────────────────────────────────────────────
function logTick(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(0)}k`;
  if (value >= 1) return `${value}`;
  return `${value}`;
}

// ── Multi-select filter component ───────────────────────────────────────────
interface MultiSelectProps {
  label: string;
  options: string[];
  selected: Set<string>;
  onToggle: (v: string) => void;
  colorFn?: (v: string) => string;
}

function MultiSelect({ label, options, selected, onToggle, colorFn }: MultiSelectProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#8b949e]">
        {label}
      </span>
      <div className="flex flex-wrap gap-1">
        {options.map((opt) => {
          const isActive = selected.has(opt);
          const accent = colorFn?.(opt) ?? '#00bcd4';
          return (
            <button
              key={opt}
              onClick={() => onToggle(opt)}
              className="rounded border px-2 py-1 text-xs font-medium transition-colors"
              style={isActive ? {
                borderColor: `${accent}80`,
                backgroundColor: `${accent}18`,
                color: accent,
              } : {
                borderColor: '#30363d',
                backgroundColor: 'transparent',
                color: '#8b949e',
              }}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Range slider ────────────────────────────────────────────────────────────
interface RangeSliderProps {
  label: string;
  min: number;
  max: number;
  value: [number, number];
  onChange: (range: [number, number]) => void;
}

function RangeSlider({ label, min, max, value, onChange }: RangeSliderProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] font-semibold uppercase tracking-widest text-[#8b949e]">
        {label}
      </span>
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs text-[#8b949e]">{value[0]}</span>
        <input
          type="range"
          min={min}
          max={max}
          value={value[0]}
          onChange={(e) => onChange([Number(e.target.value), value[1]])}
          className="h-1.5 w-16 cursor-pointer appearance-none rounded bg-[#30363d] accent-[#00bcd4]"
        />
        <input
          type="range"
          min={min}
          max={max}
          value={value[1]}
          onChange={(e) => onChange([value[0], Number(e.target.value)])}
          className="h-1.5 w-16 cursor-pointer appearance-none rounded bg-[#30363d] accent-[#00bcd4]"
        />
        <span className="font-mono text-xs text-[#8b949e]">{value[1]}</span>
      </div>
    </div>
  );
}

// ── Detail panel ────────────────────────────────────────────────────────────
function DetailPanel({ point }: { point: QuadrantPoint | null }) {
  if (!point) {
    return (
      <div className="flex h-full items-center justify-center text-xs text-[#484f58]">
        Hover a point to see details
      </div>
    );
  }
  const rows: [string, string][] = [
    ['Model', point.model],
    ['Profile', point.profile],
    ['Concurrency', String(point.concurrency)],
    ['Engine', point.engine],
    ['Hardware', point.hardware],
    ['', ''],
    ['OI (ops/byte)', point.oi.toFixed(2)],
    ['CF (GB)', point.cf_gb.toFixed(1)],
    ['', ''],
    ['Output tok/s', point.output_tput.toFixed(1)],
    ['TPOT (ms)', point.tpot_ms.toFixed(2)],
    ['TTFT (ms)', point.ttft_ms.toFixed(1)],
  ];
  return (
    <div className="space-y-1">
      <div className="mb-3 text-sm font-semibold" style={{ color: modelColor(point.model) }}>
        {point.model}
      </div>
      {rows.map(([label, value], i) =>
        label === '' ? (
          <div key={i} className="h-1" />
        ) : (
          <div key={i} className="flex justify-between text-xs">
            <span className="text-[#8b949e]">{label}</span>
            <span className="font-mono text-[#e6edf3]">{value}</span>
          </div>
        ),
      )}
    </div>
  );
}

// ── Main component ──────────────────────────────────────────────────────────
export function RooflinePage() {
  const [data, setData] = useState<RooflineQuadrantData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedProfiles, setSelectedProfiles] = useState<Set<string>>(new Set());
  const [selectedEngines, setSelectedEngines] = useState<Set<string>>(new Set());
  const [selectedHardware, setSelectedHardware] = useState<Set<string>>(new Set());
  const [concRange, setConcRange] = useState<[number, number]>([1, 200]);
  const [hoveredPoint, setHoveredPoint] = useState<QuadrantPoint | null>(null);

  // Load data
  useEffect(() => {
    fetch(`https://pub-38e30ed030784867856634f1625c7130.r2.dev/roofline-quadrant.json?v=${__BUILD_HASH__}`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json() as Promise<RooflineQuadrantData>;
      })
      .then((d) => {
        setData(d);
        setSelectedModels(new Set(d.points.map((p) => p.model)));
        setSelectedProfiles(new Set(d.points.map((p) => p.profile)));
        setSelectedEngines(new Set(d.points.map((p) => p.engine)));
        setSelectedHardware(new Set(d.points.map((p) => p.hardware)));
        const concs = d.points.map((p) => p.concurrency);
        setConcRange([Math.min(...concs), Math.max(...concs)]);
      })
      .catch((e: Error) => setLoadError(e.message));
  }, []);

  // Extract filter options
  const filterOptions = useMemo(() => {
    if (!data) return { models: [], profiles: [], engines: [], hardware: [], concMin: 1, concMax: 200 };
    const models = [...new Set(data.points.map((p) => p.model))].sort();
    const profiles = [...new Set(data.points.map((p) => p.profile))].sort();
    const engines = [...new Set(data.points.map((p) => p.engine))].sort();
    const hardware = [...new Set(data.points.map((p) => p.hardware))].sort();
    const concs = data.points.map((p) => p.concurrency);
    return {
      models,
      profiles,
      engines,
      hardware,
      concMin: Math.min(...concs),
      concMax: Math.max(...concs),
    };
  }, [data]);

  // Filtered points
  const filteredPoints = useMemo(() => {
    if (!data) return [];
    return data.points.filter(
      (p) =>
        selectedModels.has(p.model) &&
        selectedProfiles.has(p.profile) &&
        selectedEngines.has(p.engine) &&
        selectedHardware.has(p.hardware) &&
        p.concurrency >= concRange[0] &&
        p.concurrency <= concRange[1],
    );
  }, [data, selectedModels, selectedProfiles, selectedEngines, selectedHardware, concRange]);

  // Map to scatter data with size based on throughput
  const scatterData = useMemo(() => {
    if (filteredPoints.length === 0) return [];
    const maxTput = Math.max(...filteredPoints.map((p) => p.output_tput));
    return filteredPoints.map((p) => ({
      ...p,
      x: p.oi,
      y: p.cf_gb,
      _size: 4 + (p.output_tput / (maxTput || 1)) * 14,
    }));
  }, [filteredPoints]);

  // Ridge point and HBM threshold
  const ridgeOI = data ? data.hardware.peak_tflops / data.hardware.memory_bw_tbs : 295;
  const hbmGB = data?.hardware.hbm_gb ?? 80;

  const toggleSet = useCallback((setter: React.Dispatch<React.SetStateAction<Set<string>>>, value: string) => {
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(value)) {
        next.delete(value);
      } else {
        next.add(value);
      }
      return next;
    });
  }, []);

  // ── Render ──────────────────────────────────────────────────────────────
  if (loadError) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#f97583]/30 bg-[#f97583]/10 text-[#f97583]">
        <div className="text-center">
          <div className="mb-2 text-lg font-semibold">Failed to load roofline quadrant data</div>
          <div className="text-sm">{loadError}</div>
          <div className="mt-2 text-xs text-[#8b949e]">
            Run{' '}
            <code className="rounded bg-[#21262d] px-1">
              npx tsx scripts/build-roofline-quadrant.ts
            </code>{' '}
            to generate roofline-quadrant.json
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex h-64 items-center justify-center text-[#8b949e]">
        Loading roofline quadrant data...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ── Filters ──────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-start gap-3 rounded-lg border border-[#21262d] bg-[#161b22] px-4 py-3">
        <MultiSelect
          label="Model"
          options={filterOptions.models}
          selected={selectedModels}
          onToggle={(v) => toggleSet(setSelectedModels, v)}
          colorFn={modelColor}
        />
        <div className="h-5 w-px self-center bg-[#30363d]" />
        <MultiSelect
          label="Profile"
          options={filterOptions.profiles}
          selected={selectedProfiles}
          onToggle={(v) => toggleSet(setSelectedProfiles, v)}
        />
        <div className="h-5 w-px self-center bg-[#30363d]" />
        <MultiSelect
          label="Engine"
          options={filterOptions.engines}
          selected={selectedEngines}
          onToggle={(v) => toggleSet(setSelectedEngines, v)}
        />
        <div className="h-5 w-px self-center bg-[#30363d]" />
        <MultiSelect
          label="Hardware"
          options={filterOptions.hardware}
          selected={selectedHardware}
          onToggle={(v) => toggleSet(setSelectedHardware, v)}
        />
        <div className="h-5 w-px self-center bg-[#30363d]" />
        <RangeSlider
          label="Conc"
          min={filterOptions.concMin}
          max={filterOptions.concMax}
          value={concRange}
          onChange={setConcRange}
        />
        <div className="ml-auto self-center text-xs text-[#8b949e]">
          {filteredPoints.length} points
        </div>
      </div>

      {/* ── Chart + detail panel ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_240px]">
        {/* Main chart */}
        <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-[#e6edf3]">
                System-Level OI vs CF Quadrant — H100 SXM5 BF16
              </h3>
              <p className="mt-0.5 text-xs text-[#8b949e]">
                Ridge OI: {ridgeOI.toFixed(0)} ops/byte (peak {data.hardware.peak_tflops} TFLOPS / {data.hardware.memory_bw_tbs} TB/s)
                {' '}| HBM: {hbmGB} GB
              </p>
            </div>
            {/* Legend */}
            <div className="flex flex-wrap gap-x-4 gap-y-1.5">
              {filterOptions.models
                .filter((m) => selectedModels.has(m))
                .map((m) => (
                  <div key={m} className="flex items-center gap-1.5 text-xs text-[#8b949e]">
                    <span
                      style={{
                        display: 'inline-block',
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        backgroundColor: modelColor(m),
                      }}
                    />
                    {m}
                  </div>
                ))}
              <div className="ml-2 flex items-center gap-3 text-[10px] text-[#484f58]">
                {(Object.entries(PROFILE_TYPE_LABELS) as [ProfileType, string][]).map(([type, label]) => (
                  <span key={type} className="flex items-center gap-1">
                    {type === 'chat' && <span style={{ display: 'inline-block', width: 7, height: 7, borderRadius: '50%', border: '1.5px solid #484f58' }} />}
                    {type === 'coding' && <span style={{ display: 'inline-block', width: 7, height: 7, border: '1.5px solid #484f58', borderRadius: 1 }} />}
                    {type === 'synthetic' && (
                      <svg width="9" height="9" viewBox="0 0 9 9">
                        <polygon points="4.5,0 9,9 0,9" fill="none" stroke="#484f58" strokeWidth="1.5" />
                      </svg>
                    )}
                    {type === 'multiturn' && (
                      <svg width="9" height="9" viewBox="0 0 9 9">
                        <polygon points="4.5,0 9,4.5 4.5,9 0,4.5" fill="none" stroke="#484f58" strokeWidth="1.5" />
                      </svg>
                    )}
                    {label}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {filteredPoints.length === 0 ? (
            <div className="flex h-[480px] items-center justify-center text-[#8b949e]">
              No data for selected filters
            </div>
          ) : (
            <ChartErrorBoundary>
              <ResponsiveContainer width="100%" height={520}>
                <ScatterChart margin={{ top: 20, right: 30, bottom: 50, left: 30 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />

                  {/* Quadrant background areas */}
                  {/* Top-left: high OI, high CF = Compute Bound */}
                  <ReferenceArea
                    x1={ridgeOI} x2={10000}
                    y1={hbmGB} y2={5000}
                    fill={QUADRANT.computeBound.bg}
                    fillOpacity={1}
                    ifOverflow="hidden"
                  />
                  {/* Top-right: low OI, high CF = Capacity Bound */}
                  <ReferenceArea
                    x1={0.01} x2={ridgeOI}
                    y1={hbmGB} y2={5000}
                    fill={QUADRANT.capacityBound.bg}
                    fillOpacity={1}
                    ifOverflow="hidden"
                  />
                  {/* Bottom-left: high OI, low CF = BW Bound */}
                  <ReferenceArea
                    x1={ridgeOI} x2={10000}
                    y1={0.01} y2={hbmGB}
                    fill={QUADRANT.bwBound.bg}
                    fillOpacity={1}
                    ifOverflow="hidden"
                  />
                  {/* Bottom-right: low OI, low CF = Both Bound */}
                  <ReferenceArea
                    x1={0.01} x2={ridgeOI}
                    y1={0.01} y2={hbmGB}
                    fill={QUADRANT.bothBound.bg}
                    fillOpacity={1}
                    ifOverflow="hidden"
                  />

                  {/* Threshold lines */}
                  <ReferenceLine
                    x={ridgeOI}
                    stroke="#f97583"
                    strokeDasharray="8 4"
                    strokeOpacity={0.8}
                    strokeWidth={2}
                    label={{
                      value: `Ridge OI = ${ridgeOI.toFixed(0)}`,
                      fill: '#f97583',
                      fontSize: 12,
                      fontWeight: 600,
                      position: 'top',
                    }}
                  />
                  <ReferenceLine
                    y={hbmGB}
                    stroke="#ffb74d"
                    strokeDasharray="8 4"
                    strokeOpacity={0.8}
                    strokeWidth={2}
                    label={{
                      value: `HBM = ${hbmGB} GB`,
                      fill: '#ffb74d',
                      fontSize: 12,
                      fontWeight: 600,
                      position: 'right',
                    }}
                  />

                  <XAxis
                    dataKey="x"
                    type="number"
                    scale="log"
                    domain={[10, 20000]}
                    allowDataOverflow
                    tick={{ fill: '#8b949e', fontSize: 11 }}
                    axisLine={{ stroke: '#30363d' }}
                    tickLine={{ stroke: '#30363d' }}
                    tickFormatter={logTick}
                    ticks={[10, 50, 100, 295, 500, 1000, 5000, 10000]}
                    label={{
                      value: 'Operational Intensity — OI (ops/byte)',
                      position: 'insideBottom',
                      offset: -32,
                      fill: '#8b949e',
                      fontSize: 12,
                    }}
                  />
                  <YAxis
                    dataKey="y"
                    type="number"
                    scale="log"
                    domain={[5, 5000]}
                    allowDataOverflow
                    tick={{ fill: '#8b949e', fontSize: 11 }}
                    axisLine={{ stroke: '#30363d' }}
                    tickLine={{ stroke: '#30363d' }}
                    tickFormatter={logTick}
                    ticks={[5, 10, 20, 50, 80, 200, 500, 1000, 2000, 5000]}
                    label={{
                      value: 'Capacity Footprint — CF (GB)',
                      angle: -90,
                      position: 'insideLeft',
                      offset: 10,
                      fill: '#8b949e',
                      fontSize: 12,
                    }}
                  />
                  <ZAxis dataKey="_size" range={[15, 200]} />
                  <RechartsTooltip
                    content={() => null}
                    cursor={{ strokeDasharray: '4 4', stroke: '#e6edf3', strokeOpacity: 0.4 }}
                  />

                  {/* One scatter per model for coloring */}
                  {filterOptions.models
                    .filter((m) => selectedModels.has(m))
                    .map((m) => (
                      <Scatter
                        key={m}
                        name={m}
                        data={scatterData.filter((p) => p.model === m)}
                        shape={<QuadrantDot />}
                        legendType="none"
                        fill={modelColor(m)}
                        onMouseEnter={(data: { payload?: QuadrantPoint }) => {
                          if (data?.payload) setHoveredPoint(data.payload);
                        }}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                    ))}
                  {/* Highlight ring for hovered point */}
                  {hoveredPoint && (
                    <Scatter
                      data={[{ x: hoveredPoint.oi, y: hoveredPoint.cf_gb, _size: 1 }]}
                      shape={(props: { cx?: number; cy?: number }) => {
                        const { cx = 0, cy = 0 } = props;
                        if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
                        return (
                          <g>
                            <circle cx={cx} cy={cy} r={14} fill="none" stroke="#ffffff" strokeWidth={2.5} strokeOpacity={0.9} />
                            <circle cx={cx} cy={cy} r={14} fill="none" stroke={modelColor(hoveredPoint.model)} strokeWidth={1.5} strokeOpacity={0.6} />
                          </g>
                        );
                      }}
                      legendType="none"
                      isAnimationActive={false}
                    />
                  )}
                </ScatterChart>
              </ResponsiveContainer>
            </ChartErrorBoundary>
          )}

          {/* Quadrant labels */}
          <div className="mt-3 flex flex-wrap justify-center gap-4 text-[10px]">
            {Object.values(QUADRANT).map((q) => (
              <span key={q.label} className="flex items-center gap-1.5">
                <span
                  style={{
                    display: 'inline-block',
                    width: 10,
                    height: 10,
                    borderRadius: 2,
                    backgroundColor: q.color,
                    opacity: 0.4,
                  }}
                />
                <span style={{ color: q.color }}>{q.label}</span>
              </span>
            ))}
          </div>
        </div>

        {/* Side detail panel */}
        <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
          <div className="mb-3 text-[11px] font-semibold uppercase tracking-widest text-[#8b949e]">
            Point Details
          </div>
          <DetailPanel point={hoveredPoint} />

          {/* Summary stats */}
          <div className="mt-6 space-y-3 border-t border-[#21262d] pt-4">
            <div className="text-[11px] font-semibold uppercase tracking-widest text-[#8b949e]">
              Summary
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#8b949e]">Total points</span>
              <span className="font-mono text-[#e6edf3]">{filteredPoints.length}</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#8b949e]">Models</span>
              <span className="font-mono text-[#e6edf3]">
                {new Set(filteredPoints.map((p) => p.model)).size}
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#8b949e]">OI range</span>
              <span className="font-mono text-[#e6edf3]">
                {filteredPoints.length > 0
                  ? `${Math.min(...filteredPoints.map((p) => p.oi)).toFixed(1)} - ${Math.max(...filteredPoints.map((p) => p.oi)).toFixed(1)}`
                  : '-'}
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-[#8b949e]">CF range</span>
              <span className="font-mono text-[#e6edf3]">
                {filteredPoints.length > 0
                  ? `${Math.min(...filteredPoints.map((p) => p.cf_gb)).toFixed(0)} - ${Math.max(...filteredPoints.map((p) => p.cf_gb)).toFixed(0)} GB`
                  : '-'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
