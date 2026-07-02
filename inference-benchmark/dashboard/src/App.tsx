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
import { GpuStatePage } from './components/GpuStatePage';
import { ServingPredictionsPage } from './components/ServingPredictionsPage';
import { PredictionsMatrixPage } from './components/PredictionsMatrixPage';
import { simulatorPredictionsJsonUrl, simulatorV2SimPredictionsJsonUrl } from './dataUrls';
import type { TabId } from './types';
import { hasSyntheticRuntime, normalizeDataScope, type DataScope } from './profileMeta';
import './index.css';

type PageId = 'benchmark' | 'coverage' | 'serving' | 'simulator' | 'simulator_v2' | 'gpu';
const PAGE_IDS: PageId[] = ['benchmark', 'coverage', 'serving', 'simulator', 'simulator_v2', 'gpu'];
const SYNTHETIC_RUNTIME_PAGES = new Set<PageId>(['gpu', 'serving', 'simulator', 'simulator_v2']);
const DATA_SCOPE_STORAGE_KEY = 'inference-dashboard-data-scope';

function initialDataScope(): DataScope {
  const params = new URLSearchParams(window.location.search);
  const urlScope = params.get('scope');
  const normalizedUrlScope = normalizeDataScope(urlScope);
  if (normalizedUrlScope) return normalizedUrlScope;
  const storedScope = window.localStorage.getItem(DATA_SCOPE_STORAGE_KEY);
  return normalizeDataScope(storedScope) ?? 'trace_replay';
}

function hashPage(): PageId | null {
  const hashPage = window.location.hash.replace(/^#\/?/, '');
  return PAGE_IDS.includes(hashPage as PageId) ? (hashPage as PageId) : null;
}

function initialPage(): PageId {
  return hashPage() ?? 'benchmark';
}

function pageUrl(page: PageId): string {
  const url = new URL(window.location.href);
  url.hash = page === 'benchmark' ? '' : page;
  return `${url.pathname}${url.search}${url.hash}`;
}

function pageAvailableInScope(page: PageId, scope: DataScope): boolean {
  if (SYNTHETIC_RUNTIME_PAGES.has(page)) return hasSyntheticRuntime(scope);
  return true;
}

function App() {
  const [dataScope, setDataScopeState] = useState<DataScope>(initialDataScope);
  const [activePage, setActivePageState] = useState<PageId>(initialPage);
  const [activeTab, setActiveTab] = useState<TabId>('latency');
  const [scopePending, startScopeTransition] = useTransition();
  const visiblePage = pageAvailableInScope(activePage, dataScope) ? activePage : 'benchmark';
  const coverageUsesSummary = visiblePage === 'coverage' && dataScope === 'synthetic_distributional';
  const coverageNeedsSweepState = visiblePage === 'coverage' && dataScope === 'synthetic_distributional';
  const needsBenchmarkData = visiblePage === 'benchmark' || (visiblePage === 'coverage' && !coverageUsesSummary);
  const deriveBenchmarkData = visiblePage === 'benchmark';
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
  } = useData(dataScope, { deriveBenchmarkData, enabled: needsBenchmarkData });
  const { sweepState, loading: sweepLoading, error: sweepError } = useSweepState();
  const layoutLoading = visiblePage === 'coverage'
    ? (loading || (coverageNeedsSweepState && sweepLoading))
    : (needsBenchmarkData ? loading : false);
  const layoutTotalRuns = allData.length;

  const setActivePage = useCallback((page: PageId) => {
    if (!pageAvailableInScope(page, dataScope)) return;
    setActivePageState(page);
    window.history.replaceState(null, '', pageUrl(page));
  }, [dataScope]);

  useEffect(() => {
    const onHashChange = () => {
      const nextPage = hashPage();
      if (nextPage) {
        setActivePageState(nextPage);
        return;
      }
      setActivePageState('benchmark');
      window.history.replaceState(null, '', pageUrl('benchmark'));
    };
    window.addEventListener('hashchange', onHashChange);
    onHashChange();
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

  if (error && needsBenchmarkData) {
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

  if (sweepError && coverageNeedsSweepState) {
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
            <div className="mb-2 text-lg font-semibold">Failed to load sweep state</div>
            <div className="text-sm">{sweepError}</div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout
      totalRuns={layoutTotalRuns}
      loading={layoutLoading}
      activePage={visiblePage}
      onPageChange={setActivePage}
      dataScope={dataScope}
      onDataScopeChange={setDataScope}
      scopePending={scopePending}
    >
      {visiblePage === 'gpu' ? (
        <GpuStatePage />
      ) : visiblePage === 'serving' ? (
        <PredictionsMatrixPage dataScope={dataScope} predictionsUrl={simulatorPredictionsJsonUrl} />
      ) : visiblePage === 'simulator' ? (
        <ServingPredictionsPage
          dataScope={dataScope}
          predictionsUrl={simulatorPredictionsJsonUrl}
          pageKind="simulator"
        />
      ) : visiblePage === 'simulator_v2' ? (
        <ServingPredictionsPage
          dataScope={dataScope}
          predictionsUrl={simulatorV2SimPredictionsJsonUrl}
          pageKind="simulator"
        />
      ) : visiblePage === 'coverage' ? (
        <CoveragePage
          allData={allData}
          sweepState={sweepState}
          loading={loading || (coverageNeedsSweepState && sweepLoading)}
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
