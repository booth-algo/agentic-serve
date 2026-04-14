import type { TabId } from '../types';

interface TabsProps {
  active: TabId;
  onChange: (tab: TabId) => void;
}

const TABS: Array<{ id: TabId; label: string }> = [
  { id: 'latency', label: 'Latency' },
  { id: 'throughput', label: 'Throughput' },
  { id: 'comparison', label: 'Comparison' },
  { id: 'multi-turn', label: 'Multi-Turn' },
  { id: 'raw', label: 'Raw Data' },
];

export function Tabs({ active, onChange }: TabsProps) {
  return (
    <div className="mb-6 flex gap-1 border-b border-[#21262d]">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`px-4 py-2.5 text-sm font-medium transition-all ${
            active === tab.id
              ? 'border-b-2 border-[#00bcd4] text-[#e6edf3]'
              : 'border-b-2 border-transparent text-[#8b949e] hover:text-[#c9d1d9]'
          }`}
          style={{ marginBottom: '-1px' }}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
