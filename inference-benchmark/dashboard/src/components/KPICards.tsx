import type { BenchmarkResult } from '../types';
import { PROFILE_META, AGENT_TYPE_COLORS } from '../profileMeta';

interface KPICardsProps {
  data: BenchmarkResult[];
  allData: BenchmarkResult[];
}

export function KPICards({ data, allData }: KPICardsProps) {
  const totalRuns = data.length;
  const hwConfigs = new Set(data.map((r) => r.hardware)).size;
  const allHwConfigs = new Set(allData.map((r) => r.hardware)).size;
  const models = new Set(data.map((r) => r.modelShort)).size;
  const allModels = new Set(allData.map((r) => r.modelShort)).size;

  const isFiltered = data.length !== allData.length;

  // Count distinct profiles by agent type in filtered data
  const profilesInData = new Set(data.map((r) => r.config.profile));
  const typeCounts: Record<string, number> = { 'chat': 0, 'coding': 0, 'terminal': 0, 'computer-use': 0, 'stress': 0 };
  for (const profile of profilesInData) {
    const meta = PROFILE_META[profile];
    if (meta) typeCounts[meta.agentType] = (typeCounts[meta.agentType] ?? 0) + 1;
  }

  const cards = [
    {
      label: 'Total Runs',
      value: totalRuns,
      suffix: isFiltered ? ` / ${allData.length}` : '',
      accent: '#00bcd4',
    },
    {
      label: 'Hardware Configs',
      value: hwConfigs,
      suffix: isFiltered ? ` / ${allHwConfigs}` : '',
      accent: '#ff9800',
    },
    {
      label: 'Models Tested',
      value: models,
      suffix: isFiltered ? ` / ${allModels}` : '',
      accent: '#a855f7',
    },
  ];

  const typeLabels: Array<{ key: string; short: string }> = [
    { key: 'chat', short: 'Chat' },
    { key: 'coding', short: 'Coding' },
    { key: 'terminal', short: 'Terminal' },
    { key: 'computer-use', short: 'Computer Use' },
    { key: 'stress', short: 'Stress' },
  ];

  return (
    <div className="mb-6 space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 sm:gap-4">
        {cards.map((card) => (
          <div
            key={card.label}
            className="rounded-lg border border-[#21262d] bg-[#161b22] p-4"
            style={{ borderTopColor: card.accent, borderTopWidth: '3px' }}
          >
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-widest text-[#8b949e]">{card.label}</div>
            <div className="font-mono text-3xl font-bold tracking-tight text-[#e6edf3]">
              {card.value}
              {card.suffix && (
                <span className="ml-1.5 text-sm font-normal text-[#8b949e]">{card.suffix}</span>
              )}
            </div>
          </div>
        ))}
      </div>
      {profilesInData.size > 0 && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-[#21262d] bg-[#161b22] px-4 py-2.5" style={{ borderLeftColor: '#00bcd4', borderLeftWidth: '3px' }}>
          <span className="text-xs font-medium text-[#8b949e]">Workload mix</span>
          <span className="text-[#30363d]">·</span>
          {typeLabels.map(({ key, short }) => {
            const count = typeCounts[key] ?? 0;
            if (count === 0) return null;
            const colors = AGENT_TYPE_COLORS[key] ?? { bg: 'rgba(139,148,158,0.12)', text: '#8b949e', border: 'rgba(139,148,158,0.3)' };
            return (
              <span
                key={key}
                className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium"
                style={{ backgroundColor: colors.bg, color: colors.text, borderColor: colors.border }}
              >
                <span className="font-mono font-bold">{count}</span>
                {short}
              </span>
            );
          })}
          <span className="ml-auto text-xs text-[#8b949e]">
            {profilesInData.size} profile{profilesInData.size !== 1 ? 's' : ''}
          </span>
        </div>
      )}
    </div>
  );
}
