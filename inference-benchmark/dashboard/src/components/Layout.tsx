import type { ReactNode } from 'react';
import type { DataScope } from '../profileMeta';

type PageId = 'benchmark' | 'coverage' | 'gemm' | 'serving';

interface LayoutProps {
  children: ReactNode;
  totalRuns: number;
  loading: boolean;
  activePage: PageId;
  onPageChange: (page: PageId) => void;
  dataScope: DataScope;
  onDataScopeChange: (scope: DataScope) => void;
}

const NAV_PAGES: Array<{ id: PageId; label: string; icon: ReactNode }> = [
  {
    id: 'benchmark',
    label: 'Benchmark',
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
  {
    id: 'gemm',
    label: 'GEMM',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M3 9h18M9 3v18" />
      </svg>
    ),
  },
  {
    id: 'serving',
    label: 'Serving',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="m7 15 4-4 3 3 5-7" />
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
}: LayoutProps) {
  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      {/* Sticky nav */}
      <nav className="sticky top-0 z-50 border-b border-[#21262d] bg-[#161b22]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3 sm:px-6">
          {/* Left: logo + page switcher */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div
                className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#00bcd4]/15 text-[#00bcd4]"
                style={{ boxShadow: '0 0 0 1px rgba(0,188,212,0.2), 0 0 8px rgba(0,188,212,0.12)' }}
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              <h1 className="text-base font-semibold tracking-tight sm:text-lg">
                Inference Benchmark
              </h1>
            </div>

            {/* Page nav pills */}
            <div className="hidden items-center gap-1 sm:flex">
              {NAV_PAGES.map((page) => (
                <button
                  key={page.id}
                  onClick={() => onPageChange(page.id)}
                  className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                    activePage === page.id
                      ? 'bg-[#00bcd4]/12 text-[#00bcd4]'
                      : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#c9d1d9]'
                  }`}
                >
                  {page.icon}
                  {page.label}
                </button>
              ))}
            </div>
          </div>

          {/* Right: status */}
          <div className="flex items-center gap-3 text-sm text-[#8b949e]">
            <div className="hidden items-center rounded-md border border-[#30363d] bg-[#0d1117] p-0.5 sm:flex">
              {(['current', 'archive'] as const).map((scope) => (
                <button
                  key={scope}
                  onClick={() => onDataScopeChange(scope)}
                  className={`rounded px-2.5 py-1 text-xs font-medium capitalize transition-colors ${
                    dataScope === scope
                      ? 'bg-[#1f6feb] text-white'
                      : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#e6edf3]'
                  }`}
                >
                  {scope}
                </button>
              ))}
            </div>
            {loading ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#ff9800]" />
                Loading...
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#3fb950]" />
                <span className="font-mono">{totalRuns}</span> runs loaded
              </span>
            )}
          </div>
        </div>

        {/* Mobile page nav */}
        <div className="flex items-center gap-1 border-t border-[#21262d] px-4 py-2 sm:hidden">
          {NAV_PAGES.map((page) => (
            <button
              key={page.id}
              onClick={() => onPageChange(page.id)}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                activePage === page.id
                  ? 'bg-[#00bcd4]/12 text-[#00bcd4]'
                  : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#c9d1d9]'
              }`}
            >
              {page.icon}
              {page.label}
            </button>
          ))}
          <div className="ml-auto flex items-center rounded-md border border-[#30363d] bg-[#0d1117] p-0.5">
            {(['current', 'archive'] as const).map((scope) => (
              <button
                key={scope}
                onClick={() => onDataScopeChange(scope)}
                className={`rounded px-2 py-1 text-xs font-medium capitalize transition-colors ${
                  dataScope === scope
                    ? 'bg-[#1f6feb] text-white'
                    : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#e6edf3]'
                }`}
              >
                {scope}
              </button>
            ))}
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
