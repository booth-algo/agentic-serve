import { useEffect, useMemo, useState } from 'react';
import { useGpuState } from '../hooks/useGpuState';
import { gpuBlockApiUrl, hostDrainApiUrl } from '../dataUrls';
import type {
  GpuDeviceState,
  GpuHostState,
  GpuJobState,
  GpuProcessState,
  GpuState,
  GpuStatus,
  OrchestratorState,
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
  'same-user-orphan': {
    label: 'Local orphan',
    bg: 'rgba(248,81,73,0.14)',
    border: 'rgba(248,81,73,0.42)',
    text: '#f85149',
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
  'same-user-orphan': { label: 'orphan', text: '#f85149' },
  unknown: { label: 'unknown', text: '#8b949e' },
};

const ORCHESTRATOR_META: Record<string, { label: string; bg: string; border: string; text: string }> = {
  running: {
    label: 'Running now',
    bg: 'rgba(88,166,255,0.12)',
    border: 'rgba(88,166,255,0.35)',
    text: '#58a6ff',
  },
  'timer-active': {
    label: 'Timer active',
    bg: 'rgba(63,185,80,0.12)',
    border: 'rgba(63,185,80,0.35)',
    text: '#3fb950',
  },
  faulted: {
    label: 'Faulted',
    bg: 'rgba(248,81,73,0.14)',
    border: 'rgba(248,81,73,0.42)',
    text: '#f85149',
  },
  'not-installed': {
    label: 'Not installed',
    bg: 'rgba(255,152,0,0.14)',
    border: 'rgba(255,152,0,0.38)',
    text: '#ffb454',
  },
  inactive: {
    label: 'Inactive',
    bg: 'rgba(210,153,34,0.14)',
    border: 'rgba(210,153,34,0.4)',
    text: '#d29922',
  },
  unknown: {
    label: 'Unknown',
    bg: 'rgba(139,148,158,0.16)',
    border: 'rgba(139,148,158,0.35)',
    text: '#c9d1d9',
  },
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

function systemdTime(value?: string): string {
  if (!value || value === 'n/a' || value.startsWith('0')) return '-';
  return value;
}

interface HostDrainApiState {
  drained_hosts?: string[];
}

interface GpuBlockApiState {
  blocked_gpus?: Array<{ host: string; gpu: string }>;
}

function drainedHostSet(hosts?: string[]): Set<string> {
  return new Set((hosts ?? []).filter(Boolean));
}

function gpuBlockKey(host: string, gpu: string): string {
  return `${host}:${gpu}`;
}

function blockedGpuSet(entries?: Array<{ host: string; gpu: string }>): Set<string> {
  return new Set((entries ?? []).filter((entry) => entry.host && entry.gpu).map((entry) => gpuBlockKey(entry.host, entry.gpu)));
}

function useHostDrainControls(gpuState: GpuState | null) {
  const [drainedHosts, setDrainedHosts] = useState<Set<string>>(new Set());
  const [blockedGpus, setBlockedGpus] = useState<Set<string>>(new Set());
  const [controlError, setControlError] = useState<string | null>(null);
  const [pendingHost, setPendingHost] = useState<string | null>(null);
  const [pendingGpu, setPendingGpu] = useState<string | null>(null);
  const hostControlsAvailable = Boolean(hostDrainApiUrl);
  const gpuControlsAvailable = Boolean(gpuBlockApiUrl);

  useEffect(() => {
    setDrainedHosts(drainedHostSet(gpuState?.control?.drained_hosts));
    setBlockedGpus(blockedGpuSet(gpuState?.control?.blocked_gpus));
  }, [gpuState]);

  useEffect(() => {
    if (!hostControlsAvailable) return;
    let active = true;
    fetch(hostDrainApiUrl, { cache: 'no-store' })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: HostDrainApiState) => {
        if (!active) return;
        setDrainedHosts(drainedHostSet(json.drained_hosts));
        setControlError(null);
      })
      .catch((err) => {
        if (!active) return;
        setControlError(err.message);
      });
    return () => {
      active = false;
    };
  }, [hostControlsAvailable]);

  useEffect(() => {
    if (!gpuControlsAvailable) return;
    let active = true;
    fetch(gpuBlockApiUrl, { cache: 'no-store' })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((json: GpuBlockApiState) => {
        if (!active) return;
        setBlockedGpus(blockedGpuSet(json.blocked_gpus));
        setControlError(null);
      })
      .catch((err) => {
        if (!active) return;
        setControlError(err.message);
      });
    return () => {
      active = false;
    };
  }, [gpuControlsAvailable]);

  const setHostDrained = async (host: string, drained: boolean) => {
    if (!hostControlsAvailable) return;
    setPendingHost(host);
    try {
      const res = await fetch(hostDrainApiUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ host, drained }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      setDrainedHosts(drainedHostSet(json.drained_hosts));
      setControlError(null);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : String(err));
    } finally {
      setPendingHost(null);
    }
  };

  const setGpuBlocked = async (host: string, gpu: string, blocked: boolean) => {
    if (!gpuControlsAvailable) return;
    const key = gpuBlockKey(host, gpu);
    setPendingGpu(key);
    try {
      const res = await fetch(gpuBlockApiUrl, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ host, gpu, blocked }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `HTTP ${res.status}`);
      setBlockedGpus(blockedGpuSet(json.blocked_gpus));
      setControlError(null);
    } catch (err) {
      setControlError(err instanceof Error ? err.message : String(err));
    } finally {
      setPendingGpu(null);
    }
  };

  return {
    drainedHosts,
    blockedGpus,
    hostControlsAvailable,
    gpuControlsAvailable,
    controlError,
    pendingHost,
    pendingGpu,
    setHostDrained,
    setGpuBlocked,
  };
}

export function GpuStatePage() {
  const { gpuState, loading, error } = useGpuState();
  const {
    drainedHosts,
    blockedGpus,
    hostControlsAvailable,
    gpuControlsAvailable,
    controlError,
    pendingHost,
    pendingGpu,
    setHostDrained,
    setGpuBlocked,
  } = useHostDrainControls(gpuState);
  const [expandedHosts, setExpandedHosts] = useState<Set<string>>(new Set());

  const stats = useMemo(() => {
    if (!gpuState) return [];
    const otherUser = summaryCount(gpuState, 'gpus_other_user') + summaryCount(gpuState, 'gpus_mixed_other_user');
    const localOrphans = summaryCount(gpuState, 'gpus_same_user_orphan');
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
      { label: 'Local Orphans', value: localOrphans.toString(), color: '#f85149' },
      { label: 'Busy Unknown', value: summaryCount(gpuState, 'gpus_unknown_busy').toString(), color: '#c9d1d9' },
      { label: 'Sweep Blocked', value: summaryCount(gpuState, 'gpus_blocked').toString(), color: '#d29922' },
    ];
  }, [gpuState]);

  const hostNames = useMemo(() => gpuState?.hosts.map((host) => host.host) ?? [], [gpuState]);
  const allHostsExpanded = hostNames.length > 0 && hostNames.every((host) => expandedHosts.has(host));
  const toggleHost = (host: string) => {
    setExpandedHosts((current) => {
      const next = new Set(current);
      if (next.has(host)) {
        next.delete(host);
      } else {
        next.add(host);
      }
      return next;
    });
  };
  const setAllHostsExpanded = (expanded: boolean) => {
    setExpandedHosts(expanded ? new Set(hostNames) : new Set());
  };

  if (loading) {
    return <div className="p-8 text-[#8b949e]">Loading GPU state...</div>;
  }

  if (!gpuState) {
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
    <div className="space-y-4">
      <section className="rounded-lg border border-[#21262d] bg-[#161b22] p-5">
        <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-[#e6edf3]">GPU Fleet State</h2>
            <div className="mt-1 text-xs text-[#8b949e]">
              Generated {formatTime(gpuState.generated_at)} from {gpuState.state_dir}
            </div>
            {error && <div className="mt-1 text-xs text-[#f85149]">Latest GPU state refresh failed: {error}</div>}
            {controlError && <div className="mt-1 text-xs text-[#f85149]">Host control failed: {controlError}</div>}
          </div>
          <div className="flex flex-col gap-2 md:items-end">
            <div className="text-xs text-[#8b949e] md:text-right">
              <div>{gpuState.total_jobs.toLocaleString()} jobs tracked</div>
              <div>{jobCountsText(gpuState.job_counts)}</div>
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setAllHostsExpanded(!allHostsExpanded)}
                className="rounded-md border border-[#30363d] bg-[#0d1117] px-3 py-1.5 text-xs font-medium text-[#c9d1d9] transition-colors hover:border-[#58a6ff]/55 hover:text-[#e6edf3]"
              >
                {allHostsExpanded ? 'Collapse all' : 'Expand all'}
              </button>
            </div>
          </div>
        </div>

        <OrchestratorPanel orchestrator={gpuState.orchestrator} />

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

      <div className="grid gap-4 xl:grid-cols-2">
        {gpuState.hosts.map((host) => (
          <HostPanel
            key={host.host}
            host={host}
            drained={host.drained || drainedHosts.has(host.host)}
            expanded={expandedHosts.has(host.host)}
            hostControlsAvailable={hostControlsAvailable}
            gpuControlsAvailable={gpuControlsAvailable}
            hostControlPending={pendingHost === host.host}
            pendingGpu={pendingGpu}
            blockedGpus={blockedGpus}
            onToggleExpanded={() => toggleHost(host.host)}
            onSetDrained={(drained) => setHostDrained(host.host, drained)}
            onSetGpuBlocked={(gpu, blocked) => setGpuBlocked(host.host, gpu, blocked)}
          />
        ))}
      </div>
    </div>
  );
}

function OrchestratorPanel({ orchestrator }: { orchestrator?: OrchestratorState }) {
  const health = orchestrator?.health ?? 'unknown';
  const meta = ORCHESTRATOR_META[health] ?? ORCHESTRATOR_META.unknown;
  const service = orchestrator?.service;
  const timer = orchestrator?.timer;

  return (
    <div className="mt-4 rounded-md border border-[#30363d] bg-[#0d1117] px-3 py-3">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-[#8b949e]">Orchestrator</span>
            <span
              className="rounded border px-2 py-0.5 text-[11px] font-medium"
              style={{ backgroundColor: meta.bg, borderColor: meta.border, color: meta.text }}
            >
              {meta.label}
            </span>
          </div>
          <div className="mt-1 text-sm text-[#c9d1d9]">
            {orchestrator?.message ?? 'No orchestrator status in gpu-state.json'}
          </div>
        </div>
        <div className="grid gap-2 text-xs text-[#8b949e] sm:grid-cols-2 lg:min-w-[34rem]">
          <UnitStatus label="Service" unit={service} />
          <UnitStatus label="Timer" unit={timer} />
        </div>
      </div>
    </div>
  );
}

function UnitStatus({ label, unit }: { label: string; unit?: OrchestratorState['service'] }) {
  if (!unit) {
    return (
      <div className="rounded border border-[#21262d] bg-[#161b22] px-2 py-1.5">
        <div className="font-medium text-[#e6edf3]">{label}</div>
        <div>missing from report</div>
      </div>
    );
  }

  return (
    <div className="rounded border border-[#21262d] bg-[#161b22] px-2 py-1.5">
      <div className="font-medium text-[#e6edf3]">{label}</div>
      <div>
        {unit.load_state} · {unit.active_state}/{unit.sub_state} · result {unit.result || '-'}
      </div>
      {label === 'Timer' ? (
        <div>next {systemdTime(unit.next_elapse_realtime)}</div>
      ) : (
        <div>changed {systemdTime(unit.state_change_timestamp)}</div>
      )}
      {unit.exec_main_status && unit.exec_main_status !== '0' && (
        <div className="text-[#f85149]">exit {unit.exec_main_status}</div>
      )}
    </div>
  );
}

function HostPanel({
  host,
  drained,
  expanded,
  hostControlsAvailable,
  gpuControlsAvailable,
  hostControlPending,
  pendingGpu,
  blockedGpus,
  onToggleExpanded,
  onSetDrained,
  onSetGpuBlocked,
}: {
  host: GpuHostState;
  drained: boolean;
  expanded: boolean;
  hostControlsAvailable: boolean;
  gpuControlsAvailable: boolean;
  hostControlPending: boolean;
  pendingGpu: string | null;
  blockedGpus: Set<string>;
  onToggleExpanded: () => void;
  onSetDrained: (drained: boolean) => void;
  onSetGpuBlocked: (gpu: string, blocked: boolean) => void;
}) {
  const runningPorts = host.running_jobs.map((job) => job.port).filter(Boolean);
  const statusColor = host.ok ? '#3fb950' : '#f85149';
  const busyGpuCount = host.gpus.filter((gpu) => gpu.status !== 'free').length;
  const blockedGpuCount = host.gpus.filter((gpu) => gpu.blocked || blockedGpus.has(gpuBlockKey(host.host, gpu.index))).length;

  return (
    <section className={`rounded-lg border border-[#21262d] bg-[#161b22] ${expanded ? 'xl:col-span-2' : ''}`}>
      <div className="px-4 py-3">
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
            {drained && (
              <span className="rounded border border-[#d29922]/40 bg-[#d29922]/12 px-2 py-0.5 text-xs font-medium text-[#d29922]">
                draining
              </span>
            )}
            {blockedGpuCount > 0 && (
              <span className="rounded border border-[#d29922]/40 bg-[#d29922]/12 px-2 py-0.5 text-xs font-medium text-[#d29922]">
                {blockedGpuCount} GPU{blockedGpuCount === 1 ? '' : 's'} blocked
              </span>
            )}
          </div>
          <div className="flex flex-col gap-2 text-xs text-[#8b949e] lg:items-end lg:text-right">
            <div>
              <div>
                {host.gpus.length} GPUs · {busyGpuCount} busy · {blockedGpuCount} blocked · {host.running_jobs.length} running jobs
              </div>
              <div>{jobCountsText(host.job_counts)}</div>
            </div>
            <div className="flex flex-wrap gap-2 lg:justify-end">
              {hostControlsAvailable && (
                <button
                  type="button"
                  disabled={hostControlPending}
                  onClick={() => onSetDrained(!drained)}
                  className={`w-fit rounded-md border px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-wait disabled:opacity-60 ${
                    drained
                      ? 'border-[#3fb950]/45 bg-[#3fb950]/10 text-[#3fb950] hover:bg-[#3fb950]/16'
                      : 'border-[#d29922]/45 bg-[#d29922]/10 text-[#d29922] hover:bg-[#d29922]/16'
                    }`}
                >
                  {hostControlPending ? 'Updating...' : drained ? 'Resume dispatch' : 'Drain after current'}
                </button>
              )}
              <button
                type="button"
                aria-expanded={expanded}
                onClick={onToggleExpanded}
                className="w-fit rounded-md border border-[#30363d] bg-[#0d1117] px-3 py-1.5 text-xs font-medium text-[#c9d1d9] transition-colors hover:border-[#58a6ff]/55 hover:text-[#e6edf3]"
              >
                {expanded ? 'Hide details' : 'Show details'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="border-t border-[#21262d] px-4 py-3">
        {host.ok ? (
          <div className={`grid gap-2 ${expanded ? 'sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-8' : 'grid-cols-2 sm:grid-cols-4'}`}>
            {host.gpus.map((gpu) => {
              const blocked = gpu.blocked || blockedGpus.has(gpuBlockKey(host.host, gpu.index));
              return <GpuSummaryCard key={`${host.host}-${gpu.index}-summary`} gpu={gpu} blocked={blocked} />;
            })}
          </div>
        ) : (
          <div className="text-sm text-[#f97583]">{host.error || 'Host probe failed.'}</div>
        )}
      </div>

      {!expanded ? null : (
        <div className="border-t border-[#21262d]">
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
            {host.gpus.map((gpu) => {
              const blocked = gpu.blocked || blockedGpus.has(gpuBlockKey(host.host, gpu.index));
              const pending = pendingGpu === gpuBlockKey(host.host, gpu.index);
              return (
                <GpuTile
                  key={`${host.host}-${gpu.index}`}
                  gpu={gpu}
                  blocked={blocked}
                  controlsAvailable={gpuControlsAvailable}
                  controlPending={pending}
                  onSetBlocked={(nextBlocked) => onSetGpuBlocked(gpu.index, nextBlocked)}
                />
              );
            })}
          </div>
        </div>
      )}
        </div>
      )}
    </section>
  );
}

function GpuSummaryCard({ gpu, blocked }: { gpu: GpuDeviceState; blocked: boolean }) {
  const meta = STATUS_META[gpu.status];
  const memPct = memoryPercent(gpu);
  const processCount = gpu.processes.length;
  const assignmentCount = gpu.assignments.length;
  const memLabel = gpu.memory_total_mib ? `${Math.round(memPct)}% mem` : 'mem -';

  return (
    <div
      className="min-w-0 rounded-md border bg-[#0d1117] p-2"
      style={{ borderColor: meta.border, boxShadow: `inset 0 3px 0 ${meta.text}` }}
      title={`${gpu.name} · ${meta.label} · ${formatMemory(gpu.memory_used_mib, gpu.memory_total_mib)}`}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-sm font-semibold text-[#e6edf3]">GPU {gpu.index}</div>
        <div className="flex min-w-0 items-center gap-1 text-[11px] font-medium" style={{ color: meta.text }}>
          <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: meta.text }} />
          <span className="truncate">{meta.label}</span>
        </div>
      </div>
      <div className="mt-1.5 h-1.5 overflow-hidden rounded bg-[#21262d]" aria-label="GPU memory usage">
        <div className="h-full rounded" style={{ width: `${memPct}%`, backgroundColor: meta.text }} />
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-[#8b949e]">
        <span className="font-mono">{gpu.util_pct ?? 0}% util</span>
        <span className="font-mono">{memLabel}</span>
        {blocked && <span className="text-[#d29922]">blocked</span>}
        {assignmentCount > 0 && <span>{assignmentCount} jobs</span>}
        {processCount > 0 && <span>{processCount} procs</span>}
      </div>
    </div>
  );
}

function GpuTile({
  gpu,
  blocked,
  controlsAvailable,
  controlPending,
  onSetBlocked,
}: {
  gpu: GpuDeviceState;
  blocked: boolean;
  controlsAvailable: boolean;
  controlPending: boolean;
  onSetBlocked: (blocked: boolean) => void;
}) {
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
            {blocked && (
              <span className="rounded border border-[#d29922]/40 bg-[#d29922]/12 px-2 py-0.5 text-[11px] font-medium text-[#d29922]">
                sweep blocked
              </span>
            )}
          </div>
          <div className="mt-1 truncate text-xs text-[#8b949e]" title={gpu.name}>{gpu.name}</div>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-2">
          <div className="text-right font-mono text-xs text-[#8b949e]">
            <div>{gpu.util_pct ?? 0}% util</div>
            <div>{formatMemory(gpu.memory_used_mib, gpu.memory_total_mib)}</div>
          </div>
          {controlsAvailable && (
            <button
              type="button"
              disabled={controlPending}
              onClick={() => onSetBlocked(!blocked)}
              className={`w-fit rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors disabled:cursor-wait disabled:opacity-60 ${
                blocked
                  ? 'border-[#3fb950]/45 bg-[#3fb950]/10 text-[#3fb950] hover:bg-[#3fb950]/16'
                  : 'border-[#d29922]/45 bg-[#d29922]/10 text-[#d29922] hover:bg-[#d29922]/16'
              }`}
            >
              {controlPending ? 'Updating...' : blocked ? 'Allow sweep' : 'Block sweep'}
            </button>
          )}
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
      {!compact && job.run_id && (
        <div className="mt-0.5 truncate font-mono text-[10px] text-[#6e7681]" title={job.run_id}>
          run {job.run_id}
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
        {proc.ppid && <span className="font-mono text-[#8b949e]">ppid {proc.ppid}</span>}
        <span className="text-[#8b949e]">user {proc.user}</span>
        <span className="text-[#8b949e]">mem {proc.used_memory_mib ?? '-'} MiB</span>
        <span className="text-[#8b949e]">age {proc.age}</span>
        {proc.bench_port && <span className="font-mono text-[#8b949e]">port {proc.bench_port}</span>}
      </div>
      {(proc.bench_run_id || proc.bench_job_id) && (
        <div className="mt-1 rounded border border-[#238636]/40 bg-[#238636]/10 px-2 py-1 text-[11px] text-[#7ee787]">
          {proc.bench_job_id && <span className="font-mono">{proc.bench_job_id}</span>}
          {proc.bench_run_id && <span className="font-mono"> · {proc.bench_run_id}</span>}
        </div>
      )}
      {proc.orphan_reason && (
        <div className="mt-1 rounded border border-[#f85149]/30 bg-[#f85149]/10 px-2 py-1 text-[11px] text-[#f85149]">
          {proc.orphan_reason}
          {proc.parent_command && (
            <span className="text-[#ffb4ad]"> · parent {proc.parent_ppid ? `ppid ${proc.parent_ppid}` : 'reported'}</span>
          )}
        </div>
      )}
      <div className="mt-1 break-all font-mono text-[11px] leading-4 text-[#6e7681]" title={proc.command}>
        {proc.command}
      </div>
      {proc.parent_command && (
        <div className="mt-1 break-all font-mono text-[10px] leading-4 text-[#484f58]" title={proc.parent_command}>
          parent: {proc.parent_command}
        </div>
      )}
      {proc.grandparent_command && (
        <div className="mt-1 break-all font-mono text-[10px] leading-4 text-[#484f58]" title={proc.grandparent_command}>
          ancestor: {proc.grandparent_command}
        </div>
      )}
    </div>
  );
}
