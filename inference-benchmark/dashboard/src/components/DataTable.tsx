import { useState, useMemo } from 'react';
import type { BenchmarkResult } from '../types';
import { PROFILE_META, AGENT_TYPE_COLORS, DATA_SOURCE_COLORS } from '../profileMeta';

interface DataTableProps {
  data: BenchmarkResult[];
}

type SortField =
  | 'hardware'
  | 'modelShort'
  | 'backend'
  | 'profile'
  | 'type'
  | 'source'
  | 'concurrency'
  | 'successful_requests'
  | 'failed_requests'
  | 'output_token_throughput'
  | 'median_tpot_ms'
  | 'median_itl_ms'
  | 'median_ttft_ms'
  | 'median_e2el_ms';

interface ColumnDef {
  key: SortField;
  label: string;
  align: 'left' | 'right';
  format?: (r: BenchmarkResult) => string;
}

const COLUMNS: ColumnDef[] = [
  { key: 'hardware', label: 'Hardware', align: 'left' },
  { key: 'modelShort', label: 'Model', align: 'left' },
  { key: 'backend', label: 'Backend', align: 'left' },
  { key: 'profile', label: 'Profile', align: 'left' },
  { key: 'type', label: 'Type', align: 'left' },
  { key: 'source', label: 'Source', align: 'left' },
  { key: 'concurrency', label: 'Conc', align: 'right' },
  { key: 'successful_requests', label: 'OK', align: 'right' },
  { key: 'failed_requests', label: 'Fail', align: 'right' },
  {
    key: 'output_token_throughput',
    label: 'Out Tok/s',
    align: 'right',
    format: (r) => r.summary.output_token_throughput.toFixed(1),
  },
  {
    key: 'median_tpot_ms',
    label: 'TPOT p50',
    align: 'right',
    format: (r) => r.summary.median_tpot_ms.toFixed(2),
  },
  {
    key: 'median_itl_ms',
    label: 'ITL p50',
    align: 'right',
    format: (r) => r.summary.median_itl_ms?.toFixed(2) ?? '—',
  },
  {
    key: 'median_ttft_ms',
    label: 'TTFT p50',
    align: 'right',
    format: (r) => r.summary.median_ttft_ms.toFixed(1),
  },
  {
    key: 'median_e2el_ms',
    label: 'E2EL p50',
    align: 'right',
    format: (r) => r.summary.median_e2el_ms.toFixed(0),
  },
];

function getValue(r: BenchmarkResult, field: SortField): string | number {
  switch (field) {
    case 'hardware':
      return r.hardware;
    case 'modelShort':
      return r.modelShort;
    case 'backend':
      return r.config.backend;
    case 'profile':
      return r.config.profile;
    case 'type':
      return PROFILE_META[r.config.profile]?.agentType ?? '';
    case 'source':
      return PROFILE_META[r.config.profile]?.dataSource ?? '';
    case 'concurrency':
      return r.config.concurrency;
    case 'successful_requests':
      return r.summary.successful_requests;
    case 'failed_requests':
      return r.summary.failed_requests;
    case 'output_token_throughput':
      return r.summary.output_token_throughput;
    case 'median_tpot_ms':
      return r.summary.median_tpot_ms;
    case 'median_itl_ms':
      return r.summary.median_itl_ms ?? 0;
    case 'median_ttft_ms':
      return r.summary.median_ttft_ms;
    case 'median_e2el_ms':
      return r.summary.median_e2el_ms;
  }
}

function getDisplay(r: BenchmarkResult, col: ColumnDef): string {
  if (col.format) return col.format(r);
  const val = getValue(r, col.key);
  return String(val);
}

export function DataTable({ data }: DataTableProps) {
  const [sortField, setSortField] = useState<SortField>('hardware');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');

  const sorted = useMemo(() => {
    const arr = [...data];
    arr.sort((a, b) => {
      const va = getValue(a, sortField);
      const vb = getValue(b, sortField);
      let cmp: number;
      if (typeof va === 'number' && typeof vb === 'number') {
        cmp = va - vb;
      } else {
        cmp = String(va).localeCompare(String(vb));
      }
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return arr;
  }, [data, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDir('asc');
    }
  };

  if (data.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center rounded-lg border border-[#21262d] bg-[#161b22] text-[#8b949e]">
        No data matches current filters
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-[#21262d]">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b-2 border-[#30363d] bg-[#161b22]">
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                className={`cursor-pointer whitespace-nowrap px-3 py-3 text-[11px] font-semibold uppercase tracking-wider text-[#8b949e] transition-colors hover:text-[#e6edf3] ${
                  col.align === 'right' ? 'text-right' : 'text-left'
                }`}
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortField === col.key && (
                  <span className="ml-1">{sortDir === 'asc' ? '\u2191' : '\u2193'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => {
            const meta = PROFILE_META[r.config.profile];
            return (
              <tr
                key={`${r.filename}-${r.config.concurrency}`}
                className={`border-b border-[#21262d] transition-colors hover:bg-[#1c2128] ${
                  i % 2 === 0 ? 'bg-[#0d1117]' : 'bg-[#161b22]'
                }`}
              >
                {COLUMNS.map((col) => {
                  if (col.key === 'type' && meta) {
                    const colors = AGENT_TYPE_COLORS[meta.agentType] ?? { bg: 'rgba(139,148,158,0.12)', text: '#8b949e', border: 'rgba(139,148,158,0.3)' };
                    return (
                      <td key={col.key} className="whitespace-nowrap px-3 py-2 text-left">
                        <span
                          className="inline-block rounded-full border px-2 py-0.5 text-[10px] font-medium"
                          style={{ backgroundColor: colors.bg, color: colors.text, borderColor: colors.border }}
                        >
                          {meta.agentType}
                        </span>
                      </td>
                    );
                  }
                  if (col.key === 'source' && meta) {
                    const colors = DATA_SOURCE_COLORS[meta.dataSource] ?? { bg: 'rgba(139,148,158,0.12)', text: '#8b949e', border: 'rgba(139,148,158,0.3)' };
                    return (
                      <td key={col.key} className="whitespace-nowrap px-3 py-2 text-left">
                        <span
                          className="inline-block rounded-full border px-2 py-0.5 text-[10px] font-medium"
                          style={{ backgroundColor: colors.bg, color: colors.text, borderColor: colors.border }}
                        >
                          {meta.dataSource}
                        </span>
                      </td>
                    );
                  }
                  if ((col.key === 'type' || col.key === 'source') && !meta) {
                    return <td key={col.key} className="whitespace-nowrap px-3 py-2 text-left text-[#8b949e]">—</td>;
                  }
                  return (
                    <td
                      key={col.key}
                      className={`whitespace-nowrap px-3 py-2 ${
                        col.align === 'right' ? 'text-right font-mono' : 'text-left'
                      } ${
                        col.key === 'failed_requests' && r.summary.failed_requests > 0
                          ? 'text-[#f97583]'
                          : 'text-[#e6edf3]'
                      }`}
                    >
                      {getDisplay(r, col)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
