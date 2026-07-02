import type { ReactNode } from 'react';
import {
  DATA_SCOPE_META,
  DATA_SCOPE_OPTIONS,
  hasSyntheticRuntime,
  type DataScope,
} from '../profileMeta';

type PageId = 'benchmark' | 'coverage' | 'serving' | 'simulator' | 'simulator_v2' | 'gpu';
type NavPage = { id: PageId; label: string; icon: ReactNode };

interface LayoutProps {
  children: ReactNode;
  totalRuns: number;
  loading: boolean;
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  dataScope: DataScope;
  onDataScopeChange: (scope: DataScope) => void;
  scopePending?: boolean;
}

const BENCHMARK_NAV_PAGES: NavPage[] = [
  {
    id: 'benchmark',
    label: 'Home',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
  },
  {
    id: 'coverage',
    label: 'Coverage',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
      </svg>
    ),
  },
];

const RUNTIME_NAV_PAGES: NavPage[] = [
  {
    id: 'gpu',
    label: 'GPUs',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="16" height="16" rx="2" />
        <rect x="8" y="8" width="8" height="8" rx="1" />
        <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 15h3M1 9h3M1 15h3" />
      </svg>
    ),
  },
  {
    id: 'serving',
    label: 'Matrix',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="m7 15 4-4 3 3 5-7" />
      </svg>
    ),
  },
  {
    id: 'simulator',
    label: 'Simulator',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M7 15h3" />
        <path d="M12 11h3" />
        <path d="M17 7h3" />
      </svg>
    ),
  },
  {
    id: 'simulator_v2',
    label: 'Simulator v2',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M7 15h3" />
        <path d="M12 11h3" />
        <path d="M17 7h3" />
        <circle cx="20" cy="5" r="2" />
      </svg>
    ),
  },
];

export function Layout({
  children,
  totalRuns,
  loading,
  activePage,
  onPageChange,
  dataScope,
  onDataScopeChange,
  scopePending = false,
}: LayoutProps) {
  const scopeMeta = DATA_SCOPE_META[dataScope];
  const showRuntimeNav = hasSyntheticRuntime(dataScope);

  const renderNavButton = (page: NavPage) => (
    <button
      key={page.id}
      onClick={() => onPageChange(page.id)}
      className={`flex shrink-0 items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
        activePage === page.id
          ? 'bg-[#00bcd4]/12 text-[#00bcd4]'
          : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#c9d1d9]'
      }`}
    >
      {page.icon}
      {page.label}
    </button>
  );

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]" aria-busy={loading || scopePending}>
      {/* Sticky nav */}
      <nav className="sticky top-0 z-50 border-b border-[#21262d] bg-[#161b22]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3 sm:px-6">
          {/* Left: logo + page switcher */}
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <button
              type="button"
              onClick={() => onPageChange('benchmark')}
              aria-label="Home"
              className="flex shrink-0 items-center gap-3 rounded-md text-left transition-colors hover:text-[#00bcd4] focus:outline-none focus:ring-2 focus:ring-[#00bcd4]/50"
            >
              <div
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#00bcd4]/15 text-[#00bcd4]"
                style={{ boxShadow: '0 0 0 1px rgba(0,188,212,0.2), 0 0 8px rgba(0,188,212,0.12)' }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              <h1 className="hidden text-base font-semibold tracking-tight md:block lg:text-lg">
                Inference Benchmark
              </h1>
            </button>

            {/* Page nav pills */}
            <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
              {BENCHMARK_NAV_PAGES.map(renderNavButton)}
              {showRuntimeNav && (
                <>
                  <span className="mx-1 h-5 w-px shrink-0 bg-[#30363d]" aria-hidden="true" />
                  <span className="hidden shrink-0 px-1 text-[10px] font-semibold uppercase tracking-wide text-[#6e7681] sm:inline">
                    Runtime
                  </span>
                  {RUNTIME_NAV_PAGES.map(renderNavButton)}
                </>
              )}
            </div>
          </div>

          {/* Right: status */}
          <div className="hidden shrink-0 items-center gap-3 text-sm text-[#8b949e] lg:flex">
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#ff9800]" />
                Loading...
              </span>
            ) : scopePending ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#58a6ff]" />
                Updating view...
              </span>
            ) : showRuntimeNav && activePage === 'gpu' ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#58a6ff]" />
                GPU state loaded
              </span>
            ) : showRuntimeNav && activePage === 'serving' ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#58a6ff]" />
                Matrix loaded
              </span>
            ) : showRuntimeNav && (activePage === 'simulator' || activePage === 'simulator_v2') ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#58a6ff]" />
                Simulator target loaded
              </span>
            ) : activePage === 'coverage' ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#3fb950]" />
                Coverage loaded
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#3fb950]" />
                <span className="font-mono">{totalRuns}</span> runs loaded
              </span>
            )}
          </div>
        </div>

        <div className="border-t border-[#21262d] bg-[#0d1117]">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-2 text-xs sm:px-6">
            <ScopeSwitcher
              dataScope={dataScope}
              onDataScopeChange={onDataScopeChange}
              className="mr-1 max-w-full overflow-x-auto"
              compact
            />
            <span
              className="font-semibold uppercase tracking-wide"
              style={{ color: scopeMeta.accent }}
            >
              {scopeMeta.label}
            </span>
            <span className="text-[#6e7681]">{scopeMeta.eyebrow}</span>
            <span className="hidden text-[#8b949e] md:inline">{scopeMeta.description}</span>
            {!showRuntimeNav && (
              <span className="rounded border border-[#30363d] bg-[#161b22] px-2 py-0.5 text-[10px] font-medium text-[#8b949e]">
                Runtime tabs are synthetic-only
              </span>
            )}
            <span className="ml-auto hidden font-mono text-[#6e7681] sm:inline">
              {showRuntimeNav && activePage === 'gpu'
                ? 'live GPU state'
                : showRuntimeNav && activePage === 'serving'
                  ? 'matrix'
                  : showRuntimeNav && (activePage === 'simulator' || activePage === 'simulator_v2')
                    ? 'H100 / Llama-3.1-8B'
                    : activePage === 'coverage'
                      ? 'coverage view'
                    : `${totalRuns} ${scopeMeta.rowsLabel}`}
            </span>
          </div>
        </div>
      </nav>

      {/* Main content */}
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        {children}
      </main>
    </div>
  );
}

function ScopeSwitcher({
  dataScope,
  onDataScopeChange,
  className = '',
  compact = false,
}: {
  dataScope: DataScope;
  onDataScopeChange: (scope: DataScope) => void;
  className?: string;
  compact?: boolean;
}) {
  return (
    <div className={`inline-flex items-center rounded-md border border-[#30363d] bg-[#0d1117] p-0.5 ${className}`}>
      {DATA_SCOPE_OPTIONS.map((scope) => {
        const meta = DATA_SCOPE_META[scope];
        const selected = dataScope === scope;
        return (
          <button
            key={scope}
            onClick={() => onDataScopeChange(scope)}
            aria-pressed={selected}
            title={meta.description}
            className={`rounded px-2 py-1 text-left text-xs font-medium transition-colors ${
              selected
                ? 'text-white'
                : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#e6edf3]'
            }`}
            style={{
              backgroundColor: selected ? `${meta.accent}33` : undefined,
              boxShadow: selected ? `inset 0 0 0 1px ${meta.accent}99` : undefined,
              color: selected ? '#ffffff' : undefined,
            }}
          >
            <span className="whitespace-nowrap">{compact ? meta.shortLabel : meta.shortLabel}</span>
            {!compact && (
              <span className="ml-1 hidden text-[10px] font-normal text-[#8b949e] xl:inline">
                {meta.eyebrow}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
