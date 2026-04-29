import { useCallback, useMemo, useState } from 'react';
import type { FilterState, FilterOptions } from '../types';
import {
  PROFILE_META,
  AGENT_TYPE_COLORS,
  DATA_SOURCE_COLORS,
  FALLBACK_META_COLORS,
  profileDisplayName,
  isBenchmarkProfile,
} from '../profileMeta';

interface FiltersProps {
  filters: FilterState;
  options: FilterOptions;
  onToggle: (category: keyof FilterState, value: string) => void;
  onClear: () => void;
}

const CATEGORY_COLORS: Record<keyof FilterState, string> = {
  hardware: '#00bcd4',
  model: '#ff9800',
  backend: '#a855f7',
  agentType: '#3fb950',
  turnStyle: '#e78bfa',
  profile: '#79c0ff',
};

const PROFILE_GROUP_ORDER = [
  'Natural chat ST',
  'Natural chat MT',
  'Agentic coding',
  'Agentic terminal',
  'Computer-use',
  'Stress',
  'Legacy chat ST',
];

const PROFILE_ORDER = [
  'chat-singleturn',
  'chat-multiturn-short',
  'chat-multiturn-medium',
  'chat-multiturn-long',
  'coding-agent',
  'swebench-multiturn-short',
  'swebench-multiturn-medium',
  'swebench-multiturn-long',
  'terminalbench-multiturn-short',
  'terminalbench-multiturn-medium',
  'terminalbench-multiturn-long',
  'osworld-multiturn-short',
  'osworld-multiturn-medium',
  'osworld-multiturn-long',
  'prefill-heavy',
  'decode-heavy',
  'random-1k',
];

const GROUP_LABELS: Record<string, string> = {
  'Natural chat ST': 'Natural chat',
  'Natural chat MT': 'Natural chat, multi-turn',
  'Agentic coding': 'Agentic coding',
  'Agentic terminal': 'Agentic terminal',
  'Computer-use': 'Computer-use',
  Stress: 'Stress',
};

const GROUP_ACCENTS: Record<string, string> = {
  'Natural chat ST': '#3fb950',
  'Natural chat MT': '#3fb950',
  'Agentic coding': '#00bcd4',
  'Agentic terminal': '#f97583',
  'Computer-use': '#ec4899',
  Stress: '#ff9800',
};

interface MetaBadgeProps {
  label: string;
  colors: { bg: string; text: string; border: string };
}

function MetaBadge({ label, colors }: MetaBadgeProps) {
  return (
    <span
      className="inline-flex h-5 items-center rounded border px-1.5 text-[10px] font-medium"
      style={{ backgroundColor: colors.bg, color: colors.text, borderColor: colors.border }}
    >
      {label}
    </span>
  );
}

function PillRow({
  category,
  values,
  active,
  onToggle,
}: {
  category: keyof FilterState;
  values: string[];
  active: string[];
  onToggle: (cat: keyof FilterState, val: string) => void;
}) {
  const color = CATEGORY_COLORS[category];
  return (
    <div className="flex flex-wrap gap-1.5">
      {values.map((value) => {
        const isActive = active.includes(value);
        return (
          <button
            key={value}
            onClick={() => onToggle(category, value)}
            className="min-h-8 rounded-md border px-2.5 text-xs font-medium transition-colors"
            style={{
              borderColor: isActive ? color : '#30363d',
              backgroundColor: isActive ? `${color}18` : 'rgba(255,255,255,0.02)',
              color: isActive ? color : '#8b949e',
            }}
          >
            {value}
          </button>
        );
      })}
    </div>
  );
}

function SectionHeader({ label, accent }: { label: string; accent: string }) {
  return (
    <div className="mb-2 flex items-center gap-2">
      <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: accent }} />
      <span className="text-xs font-semibold uppercase tracking-wider" style={{ color: accent }}>
        {label}
      </span>
    </div>
  );
}

function FilterGroup({
  label,
  category,
  values,
  active,
  onToggle,
}: {
  label: string;
  category: keyof FilterState;
  values: string[];
  active: string[];
  onToggle: (cat: keyof FilterState, val: string) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs text-[#8b949e]">{label}</div>
      <PillRow category={category} values={values} active={active} onToggle={onToggle} />
    </div>
  );
}

function profileRank(profile: string): number {
  const index = PROFILE_ORDER.indexOf(profile);
  return index === -1 ? 10_000 : index;
}

function groupRank(group: string): number {
  const index = PROFILE_GROUP_ORDER.indexOf(group);
  return index === -1 ? 10_000 : index;
}

function turnBadgeLabel(turnStyle: string) {
  return turnStyle === 'multi-turn' ? 'MT' : 'ST';
}

export function Filters({ filters, options, onToggle, onClear }: FiltersProps) {
  const [expandedProfile, setExpandedProfile] = useState<string | null>(null);
  const [profileQuery, setProfileQuery] = useState('');
  const hasActiveFilters = Object.values(filters).some((arr) => arr.length > 0);
  const normalizedQuery = profileQuery.trim().toLowerCase();

  const allProfiles = useMemo(
    () => Object.keys(PROFILE_META)
      .filter(isBenchmarkProfile)
      .sort((a, b) => profileRank(a) - profileRank(b) || a.localeCompare(b)),
    [],
  );

  const profileMatchesFilters = useCallback((profileName: string): boolean => {
    const meta = PROFILE_META[profileName];
    if (!meta) return false;
    if (filters.agentType.length > 0 && !filters.agentType.includes(meta.agentType)) return false;
    if (filters.turnStyle.length > 0 && !filters.turnStyle.includes(meta.turnStyle)) return false;
    if (normalizedQuery) {
      const haystack = [
        profileName,
        meta.displayName,
        meta.workloadGroup,
        meta.agentType,
        meta.turnStyle,
        meta.dataSource,
      ].join(' ').toLowerCase();
      if (!haystack.includes(normalizedQuery)) return false;
    }
    return true;
  }, [filters.agentType, filters.turnStyle, normalizedQuery]);

  const profileGroups = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const profileName of allProfiles) {
      const meta = PROFILE_META[profileName];
      if (!meta) continue;
      const isSelected = filters.profile.includes(profileName);
      const matches = profileMatchesFilters(profileName);
      if (!matches && !isSelected) continue;
      const group = meta.workloadGroup;
      groups.set(group, [...(groups.get(group) ?? []), profileName]);
    }
    return Array.from(groups.entries())
      .sort(([a], [b]) => groupRank(a) - groupRank(b) || a.localeCompare(b))
      .map(([group, profiles]) => ({
        group,
        profiles: profiles.sort((a, b) => profileRank(a) - profileRank(b) || a.localeCompare(b)),
      }));
  }, [allProfiles, filters.profile, profileMatchesFilters]);

  const visibleCount = profileGroups.reduce((sum, group) => sum + group.profiles.length, 0);
  const selectedProfileCount = filters.profile.length;

  return (
    <div className="mb-6 rounded-lg border border-[#21262d] bg-[#161b22] p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-[#e6edf3]">Filters</div>
          <div className="mt-0.5 text-xs text-[#8b949e]">
            {selectedProfileCount > 0 ? `${selectedProfileCount} profile${selectedProfileCount === 1 ? '' : 's'} selected` : `${visibleCount} profiles`}
          </div>
        </div>
        {hasActiveFilters && (
          <button
            onClick={onClear}
            className="min-h-8 rounded-md border border-[#30363d] px-2.5 text-xs font-medium text-[#8b949e] transition-colors hover:border-[#6e7681] hover:text-[#e6edf3]"
          >
            Clear all
          </button>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(280px,2fr)]">
        <section className="border-t border-[#21262d] pt-3">
          <SectionHeader label="Infrastructure" accent="#00bcd4" />
          <div className="grid gap-3 md:grid-cols-3">
            <FilterGroup
              label="Hardware"
              category="hardware"
              values={options.hardware}
              active={filters.hardware}
              onToggle={onToggle}
            />
            <FilterGroup
              label="Model"
              category="model"
              values={options.model}
              active={filters.model}
              onToggle={onToggle}
            />
            <FilterGroup
              label="Backend"
              category="backend"
              values={options.backend}
              active={filters.backend}
              onToggle={onToggle}
            />
          </div>
        </section>

        <section className="border-t border-[#21262d] pt-3">
          <SectionHeader label="Workload Tags" accent="#3fb950" />
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
            <FilterGroup
              label="Agent Type"
              category="agentType"
              values={options.agentType}
              active={filters.agentType}
              onToggle={onToggle}
            />
            <FilterGroup
              label="Turn Style"
              category="turnStyle"
              values={options.turnStyle}
              active={filters.turnStyle}
              onToggle={onToggle}
            />
          </div>
        </section>
      </div>

      <section className="mt-4 border-t border-[#21262d] pt-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <SectionHeader label="Profiles" accent="#79c0ff" />
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#8b949e]">
              {visibleCount} of {allProfiles.length}
            </span>
            <input
              value={profileQuery}
              onChange={(e) => setProfileQuery(e.target.value)}
              className="h-8 w-44 rounded-md border border-[#30363d] bg-[#0d1117] px-2 text-xs text-[#e6edf3] outline-none transition-colors placeholder:text-[#6e7681] focus:border-[#79c0ff]"
              placeholder="Search profiles"
            />
          </div>
        </div>

        <div className="relative rounded-md border border-[#30363d] bg-[#0d1117]">
          <div className="pointer-events-none absolute left-0 right-3 top-0 z-10 h-5 rounded-t-md bg-gradient-to-b from-[#0d1117] to-transparent" />
          <div className="profile-scrollbar max-h-[420px] overflow-y-scroll p-2 pr-3">
            <div className="grid gap-3 xl:grid-cols-2">
              {profileGroups.map(({ group, profiles }) => {
                const accent = GROUP_ACCENTS[group] ?? '#8b949e';
                return (
                  <div key={group} className="rounded-md border border-[#21262d] bg-[#0d1117]/70">
                    <div className="flex items-center justify-between border-b border-[#21262d] px-3 py-2">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="h-2 w-2 shrink-0 rounded-sm" style={{ backgroundColor: accent }} />
                        <span className="truncate text-xs font-semibold text-[#e6edf3]">
                          {GROUP_LABELS[group] ?? group}
                        </span>
                      </div>
                      <span className="shrink-0 text-[10px] text-[#8b949e]">{profiles.length}</span>
                    </div>

                    <div className="divide-y divide-[#21262d]/70">
                      {profiles.map((profileName) => {
                        const meta = PROFILE_META[profileName];
                        const displayName = profileDisplayName(profileName);
                        const isSelected = filters.profile.includes(profileName);
                        const matches = profileMatchesFilters(profileName);
                        const agentColors = meta ? (AGENT_TYPE_COLORS[meta.agentType] || FALLBACK_META_COLORS) : FALLBACK_META_COLORS;
                        const dsColors = meta ? (DATA_SOURCE_COLORS[meta.dataSource] || FALLBACK_META_COLORS) : FALLBACK_META_COLORS;
                        const isExpanded = expandedProfile === profileName;

                        return (
                          <div key={profileName}>
                            <div
                              className="flex items-stretch transition-colors"
                              style={{
                                backgroundColor: isSelected ? 'rgba(121,192,255,0.10)' : 'transparent',
                                opacity: matches ? 1 : 0.42,
                              }}
                            >
                              <button
                                onClick={() => onToggle('profile', profileName)}
                                className="min-w-0 flex-1 px-3 py-2 text-left"
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="min-w-0">
                                    <div
                                      className="truncate text-xs font-medium"
                                      style={{ color: isSelected ? '#79c0ff' : '#e6edf3' }}
                                      title={displayName}
                                    >
                                      {displayName}
                                    </div>
                                    {meta && (
                                      <div className="mt-1 flex flex-wrap items-center gap-1">
                                        <MetaBadge label={turnBadgeLabel(meta.turnStyle)} colors={FALLBACK_META_COLORS} />
                                        <MetaBadge label={meta.agentType} colors={agentColors} />
                                        <MetaBadge label={meta.dataSource} colors={dsColors} />
                                      </div>
                                    )}
                                  </div>
                                  {meta && (
                                    <div className="shrink-0 text-right text-[10px] leading-4 text-[#8b949e]">
                                      <div>{meta.isl}</div>
                                      <div>{meta.osl}</div>
                                    </div>
                                  )}
                                </div>
                              </button>

                              {meta?.description && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setExpandedProfile(isExpanded ? null : profileName);
                                  }}
                                  className="flex w-9 shrink-0 items-center justify-center border-l border-[#21262d] text-[#8b949e] transition-colors hover:bg-[#21262d] hover:text-[#e6edf3]"
                                  title="Show workload description"
                                >
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="12" cy="12" r="10" />
                                    <line x1="12" y1="16" x2="12" y2="12" />
                                    <line x1="12" y1="8" x2="12.01" y2="8" />
                                  </svg>
                                </button>
                              )}
                            </div>

                            {isExpanded && meta?.description && (
                              <div className="border-t border-[#21262d] bg-[#161b22] px-3 py-2 text-[11px] leading-relaxed text-[#8b949e]">
                                {meta.description}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            {profileGroups.length === 0 && (
              <div className="rounded-md border border-[#21262d] bg-[#0d1117] px-3 py-6 text-center text-xs text-[#8b949e]">
                No profiles match the current filters.
              </div>
            )}
          </div>
          <div className="pointer-events-none absolute bottom-0 left-0 right-3 z-10 flex h-9 items-end justify-center rounded-b-md bg-gradient-to-t from-[#0d1117] via-[#0d1117]/85 to-transparent pb-1.5">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#79c0ff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.8">
              <path d="M6 9l6 6 6-6" />
            </svg>
          </div>
        </div>
      </section>
    </div>
  );
}
