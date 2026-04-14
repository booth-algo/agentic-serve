import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { BenchmarkResult } from '../../types';

interface ComparisonChartProps {
  seriesData: Map<string, BenchmarkResult[]>;
}

const METRIC_OPTIONS = [
  { value: 'median_ttft_ms', label: 'TTFT (median, ms)' },
  { value: 'p90_ttft_ms', label: 'TTFT (p90, ms)' },
  { value: 'median_tpot_ms', label: 'TPOT (median, ms)' },
  { value: 'p90_tpot_ms', label: 'TPOT (p90, ms)' },
  { value: 'median_itl_ms', label: 'ITL (median, ms)' },
  { value: 'p90_itl_ms', label: 'ITL (p90, ms)' },
  { value: 'median_e2el_ms', label: 'E2EL (median, ms)' },
  { value: 'output_token_throughput', label: 'Output Tok/s' },
  { value: 'total_token_throughput', label: 'Total Tok/s' },
  { value: 'request_throughput', label: 'Request Throughput' },
];

function shortenSeriesKey(key: string): string {
  const parts = key.split(' / ');
  if (parts.length < 4) return key;
  return `${parts[0]} ${parts[1].replace('Llama-3.1-', '')} ${parts[2]} ${parts[3]}`;
}

export function ComparisonChart({ seriesData }: ComparisonChartProps) {
  const seriesNames = Array.from(seriesData.keys());
  const [seriesA, setSeriesA] = useState(seriesNames[0] || '');
  const [seriesB, setSeriesB] = useState(seriesNames[1] || seriesNames[0] || '');
  const [metric, setMetric] = useState('median_tpot_ms');

  if (seriesData.size < 1) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#21262d] bg-[#161b22] text-[#8b949e]">
        Need at least one series to compare. Adjust filters.
      </div>
    );
  }

  // Build chart data
  const concSet = new Set<number>();
  const aResults = seriesData.get(seriesA) || [];
  const bResults = seriesData.get(seriesB) || [];
  for (const r of [...aResults, ...bResults]) concSet.add(r.config.concurrency);
  const concLevels = Array.from(concSet).sort((a, b) => a - b);

  const aMap = new Map<number, number>();
  for (const r of aResults) {
    aMap.set(r.config.concurrency, r.summary[metric as keyof typeof r.summary] as number);
  }
  const bMap = new Map<number, number>();
  for (const r of bResults) {
    bMap.set(r.config.concurrency, r.summary[metric as keyof typeof r.summary] as number);
  }

  const chartData = concLevels.map((conc) => {
    const point: Record<string, number | undefined> = { concurrency: conc };
    const aVal = aMap.get(conc);
    const bVal = bMap.get(conc);
    if (aVal !== undefined) point['Series A'] = Math.round(aVal * 100) / 100;
    if (bVal !== undefined) point['Series B'] = Math.round(bVal * 100) / 100;
    return point;
  });

  const metricLabel = METRIC_OPTIONS.find((m) => m.value === metric)?.label || metric;

  return (
    <div className="space-y-4">
      {/* Selectors */}
      <div className="grid grid-cols-1 gap-3 rounded-lg border border-[#21262d] bg-[#161b22] p-4 sm:grid-cols-3">
        <div>
          <label className="mb-1 block text-xs text-[#8b949e]">Series A</label>
          <select
            value={seriesA}
            onChange={(e) => setSeriesA(e.target.value)}
            className="w-full rounded-md border border-[#21262d] bg-[#0d1117] px-3 py-1.5 text-xs text-[#e6edf3] outline-none focus:border-[#00bcd4]"
          >
            {seriesNames.map((name) => (
              <option key={name} value={name}>
                {shortenSeriesKey(name)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[#8b949e]">Series B</label>
          <select
            value={seriesB}
            onChange={(e) => setSeriesB(e.target.value)}
            className="w-full rounded-md border border-[#21262d] bg-[#0d1117] px-3 py-1.5 text-xs text-[#e6edf3] outline-none focus:border-[#00bcd4]"
          >
            {seriesNames.map((name) => (
              <option key={name} value={name}>
                {shortenSeriesKey(name)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs text-[#8b949e]">Metric</label>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
            className="w-full rounded-md border border-[#21262d] bg-[#0d1117] px-3 py-1.5 text-xs text-[#e6edf3] outline-none focus:border-[#00bcd4]"
          >
            {METRIC_OPTIONS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Chart */}
      <div className="rounded-lg border border-[#21262d] bg-[#161b22] p-4">
        <div className="mb-3 flex items-center gap-2">
          <h3 className="text-sm font-semibold text-[#e6edf3]">{metricLabel} Comparison</h3>
          <span className="rounded bg-[#21262d] px-1.5 py-0.5 text-[10px] font-medium text-[#8b949e]">vs concurrency</span>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis
              dataKey="concurrency"
              scale="log"
              domain={['dataMin', 'dataMax']}
              type="number"
              tick={{ fill: '#8b949e', fontSize: 11 }}
              axisLine={{ stroke: '#21262d' }}
              tickLine={{ stroke: '#21262d' }}
              label={{ value: 'Concurrency', position: 'insideBottom', offset: -2, fill: '#8b949e', fontSize: 11 }}
            />
            <YAxis
              tick={{ fill: '#8b949e', fontSize: 11 }}
              axisLine={{ stroke: '#21262d' }}
              tickLine={{ stroke: '#21262d' }}
              label={{ value: metricLabel, angle: -90, position: 'insideLeft', fill: '#8b949e', fontSize: 11 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#161b22',
                border: '1px solid #30363d',
                borderRadius: '8px',
                fontSize: '12px',
                color: '#e6edf3',
                boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
              }}
              labelFormatter={(v) => `Concurrency: ${v}`}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              itemSorter={(item: any) => -(Number(item.value) || 0)}
            />
            <Line
              type="monotone"
              dataKey="Series A"
              name={shortenSeriesKey(seriesA)}
              stroke="#00bcd4"
              strokeWidth={2}
              dot={{ r: 4, fill: '#00bcd4' }}
              activeDot={{ r: 6 }}
              connectNulls
            />
            <Line
              type="monotone"
              dataKey="Series B"
              name={shortenSeriesKey(seriesB)}
              stroke="#ff9800"
              strokeWidth={2}
              dot={{ r: 4, fill: '#ff9800' }}
              activeDot={{ r: 6 }}
              connectNulls
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
