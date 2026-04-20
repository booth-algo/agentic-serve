import { useState } from 'react';
import { useData } from './hooks/useData';
import { Layout } from './components/Layout';
import { KPICards } from './components/KPICards';
import { Filters } from './components/Filters';
import { Tabs } from './components/Tabs';
import { LatencyChart } from './components/charts/LatencyChart';
import { ThroughputChart } from './components/charts/ThroughputChart';
import { ComparisonChart } from './components/charts/ComparisonChart';
import { PerTurnChart } from './components/charts/PerTurnChart';
import { DataTable } from './components/DataTable';
import { RooflinePage } from './components/RooflinePage';
import { CoveragePage } from './components/CoveragePage';
import type { TabId } from './types';
import './index.css';

type PageId = 'benchmark' | 'roofline' | 'coverage';

function App() {
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
  } = useData();

  const [activePage, setActivePage] = useState<PageId>('benchmark');
  const [activeTab, setActiveTab] = useState<TabId>('latency');

  if (error) {
    return (
      <Layout totalRuns={0} loading={false} activePage={activePage} onPageChange={setActivePage}>
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
    <Layout totalRuns={allData.length} loading={loading} activePage={activePage} onPageChange={setActivePage}>
      {activePage === 'roofline' ? (
        <RooflinePage />
      ) : activePage === 'coverage' ? (
        <CoveragePage allData={allData} loading={loading} />
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
