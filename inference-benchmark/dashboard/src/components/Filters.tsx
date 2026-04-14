import { useState } from 'react';
import type { FilterState, FilterOptions } from '../types';
import { PROFILE_META, AGENT_TYPE_COLORS, DATA_SOURCE_COLORS } from '../profileMeta';

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
  servingStyle: '#f97583',
  profile: '#79c0ff',
};

interface MetaBadgeProps {
  label: string;
  colors: { bg: string; text: string; border: string };
}

function MetaBadge({ label, colors }: MetaBadgeProps) {
  return (
    <span
      className="inline-block rounded-full border px-1.5 py-0 text-[10px] font-medium leading-5"
      style={{ backgroundColor: colors.bg, color: colors.text, borderColor: colors.border }}
    >
      {label}
    </span>
  );
}

const FALLBACK_COLORS = { bg: 'rgba(139,148,158,0.12)', text: '#8b949e', border: 'rgba(139,148,158,0.3)' };

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
            className="rounded-md border px-2.5 py-1 text-xs font-medium transition-all"
            style={{
              borderColor: isActive ? color : '#21262d',
              backgroundColor: isActive ? `${color}18` : 'transparent',
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
    <div
      className="mb-3 border-l-2 pl-2.5 text-xs font-semibold uppercase tracking-wider"
      style={{ borderColor: accent, color: accent }}
    >
      {label}
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

export function Filters({ filters, options, onToggle, onClear }: FiltersProps) {
  const [expandedProfile, setExpandedProfile] = useState<string | null>(null);
  const hasActiveFilters = Object.values(filters).some((arr) => arr.length > 0);

  // All known profiles from PROFILE_META (complete list regardless of loaded data)
  const allProfiles = Object.keys(PROFILE_META);

  // Count how many pass the current workload tag filters
  const workloadFiltersActive =
    filters.agentType.length > 0 ||
    filters.turnStyle.length > 0 ||
    filters.servingStyle.length > 0;

  const profileMatchesFilters = (profileName: string): boolean => {
    const meta = PROFILE_META[profileName];
    if (!meta) return false;
    if (filters.agentType.length > 0 && !filters.agentType.includes(meta.agentType)) return false;
    if (filters.turnStyle.length > 0 && !filters.turnStyle.includes(meta.turnStyle)) return false;
    if (filters.servingStyle.length > 0 && !filters.servingStyle.includes(meta.servingStyle)) return false;
    return true;
  };

  const visibleCount = workloadFiltersActive
    ? allProfiles.filter(profileMatchesFilters).length
    : allProfiles.length;

  return (
    <div className="mb-6 rounded-lg border border-[#21262d] bg-[#161b22] p-4">
      {/* Header row */}
      <div className="mb-4 flex items-center justify-between">
        <span className="text-sm font-semibold text-[#e6edf3]">Filters</span>
        {hasActiveFilters && (
          <button
            onClick={onClear}
            className="rounded px-2.5 py-1 text-xs font-medium text-[#8b949e] transition-colors hover:bg-[#21262d] hover:text-[#e6edf3]"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Two-column layout */}
      <div className="grid grid-cols-[3fr_2fr] gap-4">
        {/* LEFT COLUMN — Infrastructure + Workload */}
        <div className="space-y-4">
          {/* Section A: Infrastructure */}
          <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3">
            <SectionHeader label="Infrastructure" accent="#00bcd4" />
            <div className="space-y-3">
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
          </div>

          {/* Section B: Workload */}
          <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3">
            <SectionHeader label="Workload" accent="#3fb950" />
            <div className="space-y-3">
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
              <FilterGroup
                label="Serving Style"
                category="servingStyle"
                values={options.servingStyle}
                active={filters.servingStyle}
                onToggle={onToggle}
              />
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN — Profile sidebar */}
        <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3">
          {/* Sidebar header */}
          <div className="mb-3 flex items-center justify-between border-b border-[#21262d] pb-2">
            <span
              className="border-l-2 pl-2.5 text-xs font-semibold uppercase tracking-wider"
              style={{ borderColor: '#79c0ff', color: '#79c0ff' }}
            >
              Profiles
            </span>
            <span className="text-xs text-[#8b949e]">
              {visibleCount} of {allProfiles.length}
            </span>
          </div>

          {/* Profile list — scrollable */}
          <div className="relative">
          <div className="scrollbar-thin max-h-[320px] space-y-1 overflow-y-auto pb-8 pr-1">
            {allProfiles.map((profileName) => {
              const meta = PROFILE_META[profileName];
              const isSelected = filters.profile.includes(profileName);
              const matches = workloadFiltersActive ? profileMatchesFilters(profileName) : true;
              const agentColors = meta ? (AGENT_TYPE_COLORS[meta.agentType] || FALLBACK_COLORS) : FALLBACK_COLORS;
              const dsColors = meta ? (DATA_SOURCE_COLORS[meta.dataSource] || FALLBACK_COLORS) : FALLBACK_COLORS;

              const isExpanded = expandedProfile === profileName;

              return (
                <div key={profileName}>
                  <div
                    className="flex w-full items-start rounded border transition-all"
                    style={{
                      borderColor: isSelected ? '#79c0ff' : matches ? '#21262d' : 'transparent',
                      backgroundColor: isSelected
                        ? 'rgba(121,192,255,0.10)'
                        : matches
                        ? 'rgba(255,255,255,0.02)'
                        : 'transparent',
                      opacity: matches ? 1 : 0.28,
                    }}
                  >
                    {/* Main clickable area — selects the profile filter */}
                    <button
                      onClick={() => onToggle('profile', profileName)}
                      className="flex-1 px-2.5 py-1.5 text-left"
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span
                          className="truncate text-xs font-medium"
                          style={{ color: isSelected ? '#79c0ff' : matches ? '#e6edf3' : '#8b949e' }}
                        >
                          {profileName}
                        </span>
                        {meta && (
                          <span className="shrink-0 text-[10px] text-[#8b949e]">
                            {meta.isl}/{meta.osl}
                          </span>
                        )}
                      </div>
                      {meta && (
                        <div className="mt-1 flex flex-wrap gap-1">
                          <MetaBadge label={meta.agentType} colors={agentColors} />
                          <MetaBadge label={meta.dataSource} colors={dsColors} />
                        </div>
                      )}
                    </button>

                    {/* Info button */}
                    {meta?.description && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setExpandedProfile(isExpanded ? null : profileName);
                        }}
                        className="shrink-0 px-2 py-2 text-[#8b949e] transition-colors hover:text-[#e6edf3]"
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

                  {/* Expanded description */}
                  {isExpanded && meta?.description && (
                    <div className="mx-1 mb-1 mt-0.5 rounded border border-[#30363d] bg-[#161b22] px-2.5 py-2 text-[11px] leading-relaxed text-[#8b949e]">
                      {meta.description}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          {/* Bottom fade to hint scrollable */}
          <div
            className="pointer-events-none absolute bottom-0 left-0 right-0 h-6 rounded-b"
            style={{ background: 'linear-gradient(transparent, #0d1117)' }}
          />
          </div>
        </div>
      </div>
    </div>
  );
}
