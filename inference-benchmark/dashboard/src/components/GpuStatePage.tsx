import { useMemo } from 'react';
import { useGpuState } from '../hooks/useGpuState';
import type {
  GpuDeviceState,
  GpuHostState,
  GpuJobState,
  GpuProcessState,
  GpuState,
  GpuStatus,
} from '../types-gpu-state';

const STATUS_META: Record<GpuStatus, { label: string; bg: string; border: string; text: string }> = {
  free: {
    label: 'Free',
    bg: 'rgba(63,185,80,0.12)',
    border: 'rgba(63,185,80,0.35)',
    text: '#3fb950',
  },
  sweep: {
    label: 'Sweep',
    bg: 'rgba(88,166,255,0.12)',
    border: 'rgba(88,166,255,0.35)',
    text: '#58a6ff',
  },
  'other-user': {
    label: 'Other user',
    bg: 'rgba(255,152,0,0.14)',
    border: 'rgba(255,152,0,0.38)',
    text: '#ffb454',
  },
  'same-user-nonsweep': {
    label: 'Same user',
    bg: 'rgba(188,140,255,0.13)',
    border: 'rgba(188,140,255,0.34)',
    text: '#bc8cff',
  },
  'mixed-other-user': {
    label: 'Sweep + other',
    bg: 'rgba(248,81,73,0.14)',
    border: 'rgba(248,81,73,0.42)',
    text: '#f85149',
  },
  'mixed-same-user': {
    label: 'Sweep + local',
    bg: 'rgba(210,153,34,0.14)',
    border: 'rgba(210,153,34,0.4)',
    text: '#d29922',
  },
  'unknown-busy': {
    label: 'Busy',
    bg: 'rgba(139,148,158,0.16)',
    border: 'rgba(139,148,158,0.35)',
    text: '#c9d1d9',
  },
};

const PROCESS_META: Record<string, { label: string; text: string }> = {
  sweep: { label: 'sweep', text: '#58a6ff' },
  'sweep-slot': { label: 'slot', text: '#58a6ff' },
  'other-user': { label: 'other', text: '#ffb454' },
  'same-user-nonsweep': { label: 'local', text: '#bc8cff' },
  unknown: { label: 'unknown', text: '#8b949e' },
};

function summaryCount(data: GpuState, key: string): number {
  return data.summary?.[key] ?? 0;
}

function formatTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value || 'unknown';
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function formatMemory(used: number | null, total: number | null): string {
  if (used == null) return '-';
  if (total == null || total <= 0) return `${used} MiB`;
  return `${used.toLocaleString()} / ${total.toLocaleString()} MiB`;
}

function memoryPercent(gpu: GpuDeviceState): number {
  if (gpu.memory_used_mib == null || !gpu.memory_total_mib) return 0;
  return Math.max(0, Math.min(100, (gpu.memory_used_mib / gpu.memory_total_mib) * 100));
}

function jobCountsText(counts: Record<string, number>): string {
  const order = ['done', 'running', 'pending', 'skipped', 'failed', 'known_oom'];
  const parts = order.filter((key) => counts[key]).map((key) => `${key} ${counts[key]}`);
  for (const key of Object.keys(counts).sort()) {
    if (!order.includes(key)) parts.push(`${key} ${counts[key]}`);
  }
  return parts.join(' · ') || 'no jobs';
}

export function GpuStatePage() {
  const { gpuState, loading, error } = useGpuState();

  const stats = useMemo(() => {
    if (!gpuState) return [];
    const otherUser = summaryCount(gpuState, 'gpus_other_user') + summaryCount(gpuState, 'gpus_mixed_other_user');
    const sweep =
      summaryCount(gpuState, 'gpus_sweep') +
      summaryCount(gpuState, 'gpus_mixed_other_user') +
      summaryCount(gpuState, 'gpus_mixed_same_user');
    return [
      {
        label: 'Hosts OK',
        value: `${summaryCount(gpuState, 'hosts_ok')}/${summaryCount(gpuState, 'hosts_total')}`,
        color: '#3fb950',
      },
      { label: 'GPUs Free', value: summaryCount(gpuState, 'gpus_free').toString(), color: '#3fb950' },
      { label: 'Sweep GPUs', value: sweep.toString(), color: '#58a6ff' },
      { label: 'Used by Others', value: otherUser.toString(), color: '#ffb454' },
      {
        label: 'Local Non-Sweep',
        value: summaryCount(gpuState, 'gpus_same_user_nonsweep').toString(),
        color: '#bc8cff',
      },
      { label: 'Busy Unknown', value: summaryCount(gpuState, 'gpus_unknown_busy').toString(), color: '#c9d1d9' },
    ];
  }, [gpuState]);

  if (loading) {
    return <div className="p-8 text-[#8b949e]">Loading GPU state...</div>;
  }

  if (error || !gpuState) {
    return (
      <div className="rounded-lg border border-[#f97583]/30 bg-[#f97583]/10 p-5 text-[#f97583]">
        Failed to load gpu-state.json{error ? `: ${error}` : ''}
      </div>
    );
  }

  if (gpuState.health === 'reporter-error') {
    return (
      <div className="rounded-lg border border-[#f97583]/30 bg-[#f97583]/10 p-5 text-[#f97583]">
        GPU reporter error: {gpuState.error}
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <section className="rounded-lg border border-[#21262d] bg-[#161b22] p-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#e6edf3]">GPU Fleet State</h2>
            <div className="mt-1 text-xs text-[#8b949e]">
              Generated {formatTime(gpuState.generated_at)} from {gpuState.state_dir}
            </div>
          </div>
          <div className="flex flex-col gap-2 md:items-end">
            <div className="text-xs text-[#8b949e] md:text-right">
              <div>{gpuState.total_jobs.toLocaleString()} jobs tracked</div>
              <div>{jobCountsText(gpuState.job_counts)}</div>
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {stats.map((stat) => (
            <div key={stat.label} className="rounded-md border border-[#30363d] bg-[#0d1117] px-3 py-2">
              <div className="text-[11px] uppercase tracking-wide text-[#8b949e]">{stat.label}</div>
              <div className="mt-1 font-mono text-2xl font-semibold" style={{ color: stat.color }}>
                {stat.value}
              </div>
            </div>
          ))}
        </div>
      </section>

      {gpuState.hosts.map((host) => (
        <HostPanel key={host.host} host={host} />
      ))}
    </div>
  );
}

function HostPanel({ host }: { host: GpuHostState }) {
  const runningPorts = host.running_jobs.map((job) => job.port).filter(Boolean);
  const statusColor = host.ok ? '#3fb950' : '#f85149';

  return (
    <section className="rounded-lg border border-[#21262d] bg-[#161b22]">
      <div className="border-b border-[#21262d] px-5 py-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="font-mono text-base font-semibold text-[#e6edf3]">{host.host}</h3>
            <span
              className="rounded border px-2 py-0.5 text-xs font-medium"
              style={{
                borderColor: host.ok ? 'rgba(63,185,80,0.4)' : 'rgba(248,81,73,0.45)',
                backgroundColor: host.ok ? 'rgba(63,185,80,0.1)' : 'rgba(248,81,73,0.12)',
                color: statusColor,
              }}
            >
              {host.ok ? 'reachable' : 'ssh error'}
            </span>
            {host.remote_user && <span className="text-xs text-[#8b949e]">ssh user {host.remote_user}</span>}
            {runningPorts.length > 0 && (
              <span className="text-xs text-[#8b949e]">running ports {runningPorts.join(', ')}</span>
            )}
          </div>
          <div className="text-xs text-[#8b949e] lg:text-right">
            <div>{host.gpus.length} GPUs · {host.running_jobs.length} running jobs</div>
            <div>{jobCountsText(host.job_counts)}</div>
          </div>
        </div>
      </div>

      {!host.ok ? (
        <div className="px-5 py-4 text-sm text-[#f97583]">{host.error || 'Host probe failed.'}</div>
      ) : (
        <div className="space-y-4 p-5">
          {host.running_jobs.length > 0 && (
            <div className="rounded-md border border-[#30363d] bg-[#0d1117] p-3">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-[#8b949e]">Running Sweep Jobs</div>
              <div className="grid gap-2 md:grid-cols-2">
                {host.running_jobs.map((job) => (
                  <JobPill key={job.id} job={job} />
                ))}
              </div>
            </div>
          )}

          {host.ports.length > 0 && (
            <div className="rounded-md border border-[#30363d] bg-[#0d1117] p-3">
              <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-[#8b949e]">Listening Benchmark Ports</div>
              <div className="grid gap-1 text-xs text-[#8b949e]">
                {host.ports.map((port) => (
                  <div key={`${host.host}-${port.port}-${port.detail}`} className="min-w-0 font-mono">
                    <span className="text-[#e6edf3]">{port.port || '?'}</span>
                    <span className="ml-2 break-all text-[#6e7681]">{port.detail}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="grid gap-3 lg:grid-cols-2">
            {host.gpus.map((gpu) => (
              <GpuTile key={`${host.host}-${gpu.index}`} gpu={gpu} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function GpuTile({ gpu }: { gpu: GpuDeviceState }) {
  const meta = STATUS_META[gpu.status];
  const memPct = memoryPercent(gpu);

  return (
    <div className="rounded-md border border-[#30363d] bg-[#0d1117] p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-mono text-sm font-semibold text-[#e6edf3]">GPU {gpu.index}</span>
            <span
              className="rounded border px-2 py-0.5 text-[11px] font-medium"
              style={{ backgroundColor: meta.bg, borderColor: meta.border, color: meta.text }}
            >
              {meta.label}
            </span>
          </div>
          <div className="mt-1 truncate text-xs text-[#8b949e]" title={gpu.name}>{gpu.name}</div>
        </div>
        <div className="shrink-0 text-right font-mono text-xs text-[#8b949e]">
          <div>{gpu.util_pct ?? 0}% util</div>
          <div>{formatMemory(gpu.memory_used_mib, gpu.memory_total_mib)}</div>
        </div>
      </div>

      <div className="mt-3 h-2 overflow-hidden rounded bg-[#21262d]" aria-label="GPU memory usage">
        <div className="h-full rounded" style={{ width: `${memPct}%`, backgroundColor: meta.text }} />
      </div>

      <div className="mt-3 space-y-2">
        <div>
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[#6e7681]">Sweep Assignment</div>
          {gpu.assignments.length > 0 ? (
            <div className="space-y-1">
              {gpu.assignments.map((job) => (
                <JobPill key={job.id} job={job} compact />
              ))}
            </div>
          ) : (
            <div className="text-xs text-[#484f58]">No local sweep reservation</div>
          )}
        </div>

        <div>
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[#6e7681]">GPU Processes</div>
          {gpu.processes.length > 0 ? (
            <div className="space-y-1.5">
              {gpu.processes.map((proc) => (
                <ProcessRow key={`${proc.pid}-${proc.gpu_uuid}-${proc.used_memory_mib}`} proc={proc} />
              ))}
            </div>
          ) : (
            <div className="text-xs text-[#484f58]">No compute processes reported</div>
          )}
        </div>
      </div>
    </div>
  );
}

function JobPill({ job, compact = false }: { job: GpuJobState; compact?: boolean }) {
  return (
    <div className="min-w-0 rounded border border-[#30363d] bg-[#161b22] px-2 py-1">
      <div className="truncate font-mono text-xs text-[#e6edf3]" title={job.id}>
        {job.id}
      </div>
      {!compact && (
        <div className="mt-0.5 text-[11px] text-[#8b949e]">
          port {job.port || '-'} · GPUs {job.gpus.join(', ') || '-'} · {job.age}
        </div>
      )}
    </div>
  );
}

function ProcessRow({ proc }: { proc: GpuProcessState }) {
  const meta = PROCESS_META[proc.kind] ?? PROCESS_META.unknown;
  return (
    <div className="min-w-0 rounded border border-[#21262d] bg-[#161b22] px-2 py-1.5">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
        <span className="font-medium" style={{ color: meta.text }}>{meta.label}</span>
        <span className="font-mono text-[#c9d1d9]">pid {proc.pid}</span>
        <span className="text-[#8b949e]">user {proc.user}</span>
        <span className="text-[#8b949e]">mem {proc.used_memory_mib ?? '-'} MiB</span>
        <span className="text-[#8b949e]">age {proc.age}</span>
      </div>
      <div className="mt-1 break-all font-mono text-[11px] leading-4 text-[#6e7681]" title={proc.command}>
        {proc.command}
      </div>
    </div>
  );
}
