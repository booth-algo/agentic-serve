import { useCallback, useEffect, useState, useTransition } from 'react';
import { useData } from './hooks/useData';
import { useSweepState } from './hooks/useSweepState';
import { Layout } from './components/Layout';
import { KPICards } from './components/KPICards';
import { Filters } from './components/Filters';
import { Tabs } from './components/Tabs';
import { LatencyChart } from './components/charts/LatencyChart';
import { ThroughputChart } from './components/charts/ThroughputChart';
import { ComparisonChart } from './components/charts/ComparisonChart';
import { PerTurnChart } from './components/charts/PerTurnChart';
import { DataTable } from './components/DataTable';
import { CoveragePage } from './components/CoveragePage';
import { GemmPage } from './components/GemmPage';
import { GpuStatePage } from './components/GpuStatePage';
import { ServingPredictionsPage } from './components/ServingPredictionsPage';
import type { TabId } from './types';
import { hasServingPredictions, normalizeDataScope, type DataScope } from './profileMeta';
import './index.css';

type PageId = 'benchmark' | 'coverage' | 'gemm' | 'serving' | 'gpu';
const PAGE_IDS: PageId[] = ['benchmark', 'coverage', 'gemm', 'serving', 'gpu'];
const DATA_SCOPE_STORAGE_KEY = 'inference-dashboard-data-scope';

function initialDataScope(): DataScope {
  const params = new URLSearchParams(window.location.search);
  const urlScope = params.get('scope');
  const normalizedUrlScope = normalizeDataScope(urlScope);
  if (normalizedUrlScope) return normalizedUrlScope;
  const storedScope = window.localStorage.getItem(DATA_SCOPE_STORAGE_KEY);
  return normalizeDataScope(storedScope) ?? 'trace_replay';
}

function initialPage(): PageId {
  const hashPage = window.location.hash.replace(/^#\/?/, '');
  return PAGE_IDS.includes(hashPage as PageId) ? (hashPage as PageId) : 'benchmark';
}

function pageUrl(page: PageId): string {
  const url = new URL(window.location.href);
  url.hash = page === 'benchmark' ? '' : page;
  return `${url.pathname}${url.search}${url.hash}`;
}

function pageAvailableInScope(page: PageId, scope: DataScope): boolean {
  return page !== 'serving' || hasServingPredictions(scope);
}

function App() {
  const [dataScope, setDataScopeState] = useState<DataScope>(initialDataScope);
  const [activePage, setActivePageState] = useState<PageId>(initialPage);
  const [activeTab, setActiveTab] = useState<TabId>('latency');
  const [scopePending, startScopeTransition] = useTransition();
  const visiblePage = pageAvailableInScope(activePage, dataScope) ? activePage : 'benchmark';
  const needsBenchmarkData = visiblePage === 'benchmark' || visiblePage === 'coverage';
  const {
    allData,
    data,
    seriesData,
    loading,
    error,
    filters,
    filterOptions,
    toggleFilter,
    clearFilters,
    clearWorkloadFilters,
  } = useData(dataScope, { deriveBenchmarkData: needsBenchmarkData, enabled: needsBenchmarkData });
  const { sweepState } = useSweepState();

  const setActivePage = useCallback((page: PageId) => {
    if (!pageAvailableInScope(page, dataScope)) return;
    setActivePageState(page);
    window.history.replaceState(null, '', pageUrl(page));
  }, [dataScope]);

  useEffect(() => {
    const onHashChange = () => setActivePageState(initialPage());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  useEffect(() => {
    if (!pageAvailableInScope(activePage, dataScope)) {
      setActivePageState('benchmark');
      window.history.replaceState(null, '', pageUrl('benchmark'));
    }
  }, [activePage, dataScope]);

  const setDataScope = useCallback((scope: DataScope) => {
    window.localStorage.setItem(DATA_SCOPE_STORAGE_KEY, scope);
    const url = new URL(window.location.href);
    if (scope !== 'trace_replay') {
      url.searchParams.set('scope', scope);
    } else {
      url.searchParams.delete('scope');
    }
    window.history.replaceState(null, '', `${url.pathname}${url.search}${url.hash}`);
    startScopeTransition(() => {
      setDataScopeState(scope);
      clearWorkloadFilters();
    });
  }, [clearWorkloadFilters]);

  if (error && visiblePage !== 'gpu') {
    return (
      <Layout
        totalRuns={0}
        loading={false}
        activePage={visiblePage}
        onPageChange={setActivePage}
        dataScope={dataScope}
        onDataScopeChange={setDataScope}
        scopePending={scopePending}
      >
        <div className="flex h-64 items-center justify-center rounded-lg border border-[#f97583]/30 bg-[#f97583]/10 text-[#f97583]">
          <div className="text-center">
            <div className="mb-2 text-lg font-semibold">Failed to load data</div>
            <div className="text-sm">{error}</div>
            <div className="mt-2 text-xs text-[#8b949e]">
              Run <code className="rounded bg-[#21262d] px-1">npx tsx scripts/build-data.ts</code> to generate data.json
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout
      totalRuns={allData.length}
      loading={visiblePage === 'gpu' ? false : loading}
      activePage={visiblePage}
      onPageChange={setActivePage}
      dataScope={dataScope}
      onDataScopeChange={setDataScope}
      scopePending={scopePending}
    >
      {visiblePage === 'gpu' ? (
        <GpuStatePage />
      ) : visiblePage === 'gemm' ? (
        <GemmPage />
      ) : visiblePage === 'serving' ? (
        <ServingPredictionsPage dataScope={dataScope} />
      ) : visiblePage === 'coverage' ? (
        <CoveragePage
          allData={allData}
          sweepState={sweepState}
          loading={loading}
          dataScope={dataScope}
        />
      ) : loading ? (
        <div className="flex h-64 items-center justify-center">
          <div className="text-[#8b949e]">Loading benchmark data...</div>
        </div>
      ) : (
        <>
          <KPICards data={data} allData={allData} />
          <Filters
            filters={filters}
            options={filterOptions}
            dataScope={dataScope}
            onToggle={toggleFilter}
            onClear={clearFilters}
          />
          <Tabs active={activeTab} onChange={setActiveTab} />

          {activeTab === 'latency' && <LatencyChart seriesData={seriesData} />}
          {activeTab === 'throughput' && <ThroughputChart seriesData={seriesData} />}
          {activeTab === 'comparison' && <ComparisonChart seriesData={seriesData} />}
          {activeTab === 'multi-turn' && <PerTurnChart data={data} />}
          {activeTab === 'raw' && <DataTable data={data} />}
        </>
      )}
    </Layout>
  );
}

export default App;
