import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Bar,
  BarChart,
  ScatterChart as RechartsScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';
import type { BenchmarkResult } from '../../types';
import { profileDisplayName } from '../../profileMeta';

interface PerTurnChartProps {
  data: BenchmarkResult[];
}

const COLORS = [
  '#00bcd4', '#ff9800', '#a855f7', '#3fb950', '#f97583',
  '#79c0ff', '#d2a8ff', '#ffa657', '#7ee787', '#ff7b72',
];

function shortenKey(key: string): string {
  const parts = key.split(' / ');
  const profile = parts[parts.length - 1] || key;
  return profileDisplayName(profile);
}

// Unique key per result: include concurrency to distinguish runs
function perTurnKey(r: BenchmarkResult): string {
  return `${shortenKey(r.seriesKey)} conc=${r.config.concurrency}`;
}

export function PerTurnChart({ data }: PerTurnChartProps) {
  // Filter to only results with perTurn data
  const multiTurnResults = data.filter((r) => r.perTurn && r.perTurn.length > 0);

  if (multiTurnResults.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#21262d] bg-[#161b22] text-[#8b949e]">
        No multi-turn data available. Run a multi-turn benchmark to see per-turn metrics.
      </div>
    );
  }

  // Build TTFT chart data: x = turn number, one line per unique run
  const maxTurns = Math.max(...multiTurnResults.map((r) => r.perTurn!.length));
  const ttftData = Array.from({ length: maxTurns }, (_, i) => {
    const point: Record<string, number> = { turn: i + 1 };
    for (const r of multiTurnResults) {
      const key = perTurnKey(r);
      const entry = r.perTurn![i];
      if (entry) {
        point[`${key}_ttft`] = Math.round(entry.median_ttft_ms * 100) / 100;
        point[`${key}_isl`] = Math.round(entry.avg_input_tokens);
      }
    }
    return point;
  });

  // Build ISL growth chart data
  const islData = Array.from({ length: maxTurns }, (_, i) => {
    const point: Record<string, number> = { turn: i + 1 };
    for (const r of multiTurnResults) {
      const entry = r.perTurn![i];
      if (entry) {
        point[perTurnKey(r)] = Math.round(entry.avg_input_tokens);
      }
    }
    return point;
  });

  // Build TPOT chart data
  const tpotData = Array.from({ length: maxTurns }, (_, i) => {
    const point: Record<string, number> = { turn: i + 1 };
    for (const r of multiTurnResults) {
      const entry = r.perTurn![i];
      if (entry) {
        point[perTurnKey(r)] = Math.round(entry.median_tpot_ms * 100) / 100;
      }
    }
    return point;
  });

  // Build requests per turn bar data
  const reqData = Array.from({ length: maxTurns }, (_, i) => {
    const point: Record<string, number> = { turn: i + 1 };
    for (const r of multiTurnResults) {
      const entry = r.perTurn![i];
      if (entry) {
        point[perTurnKey(r)] = entry.successful;
      }
    }
    return point;
  });

  const seriesKeys = multiTurnResults.map((r) => perTurnKey(r));

  const chartStyle = {
    grid: '#21262d',
    tick: '#8b949e',
  };

  const tooltipStyle = {
    backgroundColor: '#161b22',
    border: '1px solid #21262d',
    borderRadius: '8px',
    fontSize: '12px',
    color: '#e6edf3',
  };

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      {/* TTFT per turn — the key chart */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4 lg:col-span-2">
        <h3 className="mb-1 text-sm font-medium text-[#e6edf3]">
          TTFT per Turn
          <span className="ml-2 text-xs text-[#8b949e]">median, ms — prefix cache effect visible in slope</span>
        </h3>
        <p className="mb-3 text-xs text-[#8b949e]">
          Sub-linear TTFT growth = prefix cache is reusing KV entries from earlier turns
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={ttftData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyle.grid} />
            <XAxis
              dataKey="turn"
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Turn', position: 'insideBottom', offset: -2, fill: chartStyle.tick, fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'TTFT (ms)', angle: -90, position: 'insideLeft', fill: chartStyle.tick, fontSize: 11 }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(v) => `Turn ${v}`}
              formatter={(value: unknown, name: unknown) => {
                const v = value as number;
                const key = (name as string).replace(/_ttft$/, '');
                const turnIdx = ttftData.findIndex((d) => d[name as string] === v);
                const islKey = `${key}_isl`;
                const isl = turnIdx >= 0 ? ttftData[turnIdx][islKey] : undefined;
                return [
                  `${v.toFixed(1)} ms${isl ? ` (ISL: ~${isl})` : ''}`,
                  key,
                ];
              }}
            />
            {seriesKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={`${key}_ttft`}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2.5}
                dot={{ r: 4, fill: COLORS[i % COLORS.length] }}
                activeDot={{ r: 6 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Avg ISL growth */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
        <h3 className="mb-3 text-sm font-medium text-[#e6edf3]">
          Context Length Growth
          <span className="ml-2 text-xs text-[#8b949e]">avg input tokens per turn</span>
        </h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={islData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyle.grid} />
            <XAxis
              dataKey="turn"
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Turn', position: 'insideBottom', offset: -2, fill: chartStyle.tick, fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Tokens', angle: -90, position: 'insideLeft', fill: chartStyle.tick, fontSize: 11 }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(v) => `Turn ${v}`}
              formatter={(value: unknown, name: unknown) => [`${value as number} tokens`, name as string]}
            />
            {seriesKeys.map((key, i) => (
              <Bar
                key={key}
                dataKey={key}
                fill={COLORS[i % COLORS.length]}
                opacity={0.8}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* TPOT stability */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
        <h3 className="mb-3 text-sm font-medium text-[#e6edf3]">
          TPOT per Turn
          <span className="ml-2 text-xs text-[#8b949e]">median, ms — should stay flat</span>
        </h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={tpotData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyle.grid} />
            <XAxis
              dataKey="turn"
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Turn', position: 'insideBottom', offset: -2, fill: chartStyle.tick, fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'TPOT (ms)', angle: -90, position: 'insideLeft', fill: chartStyle.tick, fontSize: 11 }}
              domain={['dataMin - 2', 'dataMax + 2']}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(v) => `Turn ${v}`}
              formatter={(value: unknown, name: unknown) => [`${(value as number).toFixed(1)} ms`, name as string]}
            />
            {seriesKeys.map((key, i) => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[i % COLORS.length]}
                strokeWidth={2}
                dot={{ r: 3, fill: COLORS[i % COLORS.length] }}
                activeDot={{ r: 5 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* TTFT vs ISL scatter — per-request level */}
      {(() => {
        const allScatter = multiTurnResults.flatMap((r) =>
          (r.scatterData || []).map((p) => ({ ...p, series: shortenKey(r.seriesKey) }))
        );
        if (allScatter.length === 0) return null;

        // Group by turn index for coloring
        const maxTurnIdx = Math.max(...allScatter.map((p) => p.turn_index));
        const turnGroups = Array.from({ length: maxTurnIdx + 1 }, (_, i) =>
          allScatter.filter((p) => p.turn_index === i)
        );

        // Color scale: gradient from teal to coral across turns
        const turnColors = Array.from({ length: maxTurnIdx + 1 }, (_, i) => {
          const t = maxTurnIdx > 0 ? i / maxTurnIdx : 0;
          // Interpolate from teal (#00bcd4) to coral (#ef5350)
          const r = Math.round(0x00 + t * (0xef - 0x00));
          const g = Math.round(0xbc + t * (0x53 - 0xbc));
          const b = Math.round(0xd4 + t * (0x50 - 0xd4));
          return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        });

        return (
          <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4 lg:col-span-2">
            <h3 className="mb-1 text-sm font-medium text-[#e6edf3]">
              TTFT vs Input Length
              <span className="ml-2 text-xs text-[#8b949e]">per-request, colored by turn number</span>
            </h3>
            <p className="mb-3 text-xs text-[#8b949e]">
              Each dot is one request. Later turns (warmer colors) have longer context but may benefit from prefix cache.
            </p>
            <ResponsiveContainer width="100%" height={350}>
              <RechartsScatterChart margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={chartStyle.grid} />
                <XAxis
                  dataKey="input_tokens"
                  type="number"
                  name="Input Tokens"
                  tick={{ fill: chartStyle.tick, fontSize: 11 }}
                  axisLine={{ stroke: chartStyle.grid }}
                  tickLine={{ stroke: chartStyle.grid }}
                  label={{ value: 'Input Tokens', position: 'insideBottom', offset: -2, fill: chartStyle.tick, fontSize: 11 }}
                />
                <YAxis
                  dataKey="ttft_ms"
                  type="number"
                  name="TTFT"
                  tick={{ fill: chartStyle.tick, fontSize: 11 }}
                  axisLine={{ stroke: chartStyle.grid }}
                  tickLine={{ stroke: chartStyle.grid }}
                  label={{ value: 'TTFT (ms)', angle: -90, position: 'insideLeft', fill: chartStyle.tick, fontSize: 11 }}
                />
                <ZAxis range={[20, 20]} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value: unknown, name: unknown) => {
                    const v = value as number;
                    const n = name as string;
                    if (n === 'Input Tokens') return [`${v}`, n];
                    if (n === 'TTFT') return [`${v.toFixed(1)} ms`, n];
                    return [`${v}`, n];
                  }}
                  labelFormatter={() => ''}
                />
                {turnGroups.map((points, i) =>
                  points.length > 0 ? (
                    <Scatter
                      key={`turn-${i}`}
                      name={`Turn ${i + 1}`}
                      data={points}
                      fill={turnColors[i]}
                      opacity={0.7}
                    />
                  ) : null
                )}
              </RechartsScatterChart>
            </ResponsiveContainer>
          </div>
        );
      })()}

      {/* Requests per turn */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4 lg:col-span-2">
        <h3 className="mb-3 text-sm font-medium text-[#e6edf3]">
          Sessions per Turn
          <span className="ml-2 text-xs text-[#8b949e]">sessions drop off as shorter conversations end</span>
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={reqData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartStyle.grid} />
            <XAxis
              dataKey="turn"
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Turn', position: 'insideBottom', offset: -2, fill: chartStyle.tick, fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: chartStyle.tick, fontSize: 11 }}
              axisLine={{ stroke: chartStyle.grid }}
              tickLine={{ stroke: chartStyle.grid }}
              label={{ value: 'Sessions', angle: -90, position: 'insideLeft', fill: chartStyle.tick, fontSize: 11 }}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              labelFormatter={(v) => `Turn ${v}`}
              formatter={(value: unknown, name: unknown) => [`${value as number} sessions`, name as string]}
            />
            {seriesKeys.map((key, i) => (
              <Bar
                key={key}
                dataKey={key}
                fill={COLORS[i % COLORS.length]}
                opacity={0.7}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
